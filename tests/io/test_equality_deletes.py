# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Acceptance tests for Iceberg v2 equality delete reads.

Every test builds a real table: real Parquet data and delete files plus real
Avro manifests and a manifest list, then scans it through the public
``Table.scan`` API.  Expected surviving rows are listed as explicit row
identities, so the tests do not simply recompute the result the same way the
implementation does.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pytest

from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import BooleanType, IntegerType, LongType, NestedField, StringType
from tests.io.equality_delete_helpers import build_table, write_parquet

FIELD_IDS = {"id": 1, "cat": 2, "flag": 3, "val": 4}
ARROW_TYPES = {"id": pa.int64(), "cat": pa.string(), "flag": pa.bool_(), "val": pa.int32()}


def simple_schema() -> Schema:
    return Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "cat", StringType(), required=False),
        NestedField(3, "flag", BooleanType(), required=False),
        NestedField(4, "val", IntegerType(), required=False),
    )


def _write(
    tmp: str,
    name: str,
    columns: Dict[str, List[Any]],
    types: Optional[Dict[str, pa.DataType]] = None,
    field_ids: Optional[Dict[str, int]] = None,
    row_group_size: Optional[int] = None,
) -> str:
    return write_parquet(
        os.path.join(tmp, name),
        columns,
        field_ids or FIELD_IDS,
        types or ARROW_TYPES,
        row_group_size,
    )


def _rows(table: Table, **scan_kwargs: Any) -> List[Tuple[Any, ...]]:
    result = table.scan(**scan_kwargs).to_arrow().to_pylist()
    return sorted(tuple(row[name] for name in result and list(row.keys()) or []) for row in result)


def _ids(table: Table, **scan_kwargs: Any) -> List[int]:
    return sorted(table.scan(**scan_kwargs).to_arrow()["id"].to_pylist())


@pytest.fixture
def tmp(tmp_path: Any) -> str:
    return str(tmp_path)


def _base_data(tmp: str, name: str = "data.parquet") -> str:
    """Rows keyed by id, with id==val for easy identity checks."""
    return _write(
        tmp,
        name,
        {
            "id": [0, 1, 2, 3, 4, 5],
            "cat": ["a", "b", "c", "a", "b", "c"],
            "flag": [True, False, True, None, False, None],
            "val": [10, 11, 12, 13, 14, 15],
        },
    )


def _table(tmp: str, data_files: List[Dict[str, Any]], delete_files: List[Dict[str, Any]], **kwargs: Any) -> Table:
    return build_table(tmp, simple_schema(), {0: UNPARTITIONED_PARTITION_SPEC}, data_files, delete_files, **kwargs)


# ---------------------------------------------------------------------------
# Sequence-number rules
# ---------------------------------------------------------------------------
def test_delete_applies_only_to_older_data(tmp: str) -> None:
    old = _base_data(tmp, "old.parquet")  # seq 1
    new = _base_data(tmp, "new.parquet")  # seq 20, same content
    delete = _write(tmp, "del.parquet", {"id": [1, 4]}, {"id": pa.int64()})

    table = _table(
        tmp,
        [
            {"path": old, "records": 6, "partition": [], "spec_id": 0, "seq": 1},
            {"path": new, "records": 6, "partition": [], "spec_id": 0, "seq": 20},
        ],
        [{"path": delete, "records": 2, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]}],
    )
    # Only the old file is older than the delete; the new file keeps all rows.
    assert _ids(table) == sorted([0, 2, 3, 5] + [0, 1, 2, 3, 4, 5])


def test_delete_with_equal_sequence_does_not_apply(tmp: str) -> None:
    data = _base_data(tmp)
    delete = _write(tmp, "del.parquet", {"id": [1]}, {"id": pa.int64()})
    table = _table(
        tmp,
        [{"path": data, "records": 6, "partition": [], "spec_id": 0, "seq": 10}],
        [{"path": delete, "records": 1, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]}],
    )
    assert _ids(table) == [0, 1, 2, 3, 4, 5]


def test_delete_with_lower_sequence_does_not_apply(tmp: str) -> None:
    data = _base_data(tmp)
    delete = _write(tmp, "del.parquet", {"id": [1]}, {"id": pa.int64()})
    table = _table(
        tmp,
        [{"path": data, "records": 6, "partition": [], "spec_id": 0, "seq": 10}],
        [{"path": delete, "records": 1, "partition": [], "spec_id": 0, "seq": 5, "equality_ids": [1]}],
    )
    assert _ids(table) == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Partition scope
# ---------------------------------------------------------------------------
def test_global_unpartitioned_delete_hits_all_partitions(tmp: str) -> None:
    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="cat"), spec_id=0)
    d_a = _write(tmp, "a.parquet", {"id": [0, 1], "cat": ["a", "a"], "flag": [True, False], "val": [0, 1]})
    d_c = _write(tmp, "c.parquet", {"id": [2, 3], "cat": ["c", "c"], "flag": [True, False], "val": [2, 3]})
    delete = _write(tmp, "del.parquet", {"id": [1, 2]}, {"id": pa.int64()})
    table = build_table(
        tmp,
        simple_schema(),
        {0: spec, 1: PartitionSpec(spec_id=1)},
        [
            {"path": d_a, "records": 2, "partition": ["a"], "spec_id": 0, "seq": 1},
            {"path": d_c, "records": 2, "partition": ["c"], "spec_id": 0, "seq": 1},
        ],
        [{"path": delete, "records": 2, "partition": [], "spec_id": 1, "seq": 10, "equality_ids": [1]}],
    )
    assert _ids(table) == [0, 3]


def test_partitioned_delete_only_hits_its_partition(tmp: str) -> None:
    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="cat"), spec_id=0)
    d_a = _write(tmp, "a.parquet", {"id": [0, 1], "cat": ["a", "a"], "flag": [True, False], "val": [0, 1]})
    d_c = _write(tmp, "c.parquet", {"id": [2, 3], "cat": ["c", "c"], "flag": [True, False], "val": [2, 3]})
    # The delete key id=2 lives in partition "a" (where no row has id=2): it must
    # not touch the id=2 row in partition "c".
    delete_a = _write(tmp, "del_a.parquet", {"id": [2]}, {"id": pa.int64()})
    table = build_table(
        tmp,
        simple_schema(),
        {0: spec},
        [
            {"path": d_a, "records": 2, "partition": ["a"], "spec_id": 0, "seq": 1},
            {"path": d_c, "records": 2, "partition": ["c"], "spec_id": 0, "seq": 1},
        ],
        [{"path": delete_a, "records": 1, "partition": ["a"], "spec_id": 0, "seq": 10, "equality_ids": [1]}],
    )
    assert _ids(table) == [0, 1, 2, 3]


def test_delete_spec_id_must_match_partition_values(tmp: str) -> None:
    """A delete in spec 0 only matches data in spec 0, even with equal values."""
    spec0 = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="cat"), spec_id=0)
    spec1 = PartitionSpec(PartitionField(source_id=2, field_id=1001, transform=IdentityTransform(), name="category"), spec_id=1)
    d = _write(tmp, "a.parquet", {"id": [0, 1], "cat": ["a", "a"], "flag": [True, False], "val": [0, 1]})
    delete = _write(tmp, "del.parquet", {"id": [0]}, {"id": pa.int64()})
    table = build_table(
        tmp,
        simple_schema(),
        {0: spec0, 1: spec1},
        [{"path": d, "records": 2, "partition": ["a"], "spec_id": 1, "seq": 1}],
        [{"path": delete, "records": 1, "partition": ["a"], "spec_id": 0, "seq": 10, "equality_ids": [1]}],
    )
    # Different spec id => not applicable.
    assert _ids(table) == [0, 1]


# ---------------------------------------------------------------------------
# Composite keys, nulls, duplicates
# ---------------------------------------------------------------------------
def test_composite_key_null_equals_null(tmp: str) -> None:
    data = _write(
        tmp,
        "data.parquet",
        {
            "id": [0, 1, 2, 3, 4],
            "cat": ["a", "a", None, None, "z"],
            "flag": [True, False, None, True, None],
            "val": [0, 1, 2, 3, 4],
        },
    )
    delete = _write(
        tmp,
        "del.parquet",
        {"id": [2, 0, 2], "cat": [None, "a", None], "flag": [None, True, None]},
        {"id": pa.int64(), "cat": pa.string(), "flag": pa.bool_()},
        field_ids={"id": 1, "cat": 2, "flag": 3},
    )
    table = _table(
        tmp,
        [{"path": data, "records": 5, "partition": [], "spec_id": 0, "seq": 1}],
        [{"path": delete, "records": 3, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1, 2, 3]}],
    )
    # (0,a,True) and (2,NULL,NULL) removed; (1,a,False), (3,NULL,True) and
    # (4,z,None) remain.  The null tuple matches only the all-null row.
    assert _ids(table) == [1, 3, 4]


def test_duplicate_delete_keys_do_not_over_delete(tmp: str) -> None:
    data = _base_data(tmp)
    delete = _write(tmp, "del.parquet", {"id": [1, 1, 1, 3, 3]}, {"id": pa.int64()})
    table = _table(
        tmp,
        [{"path": data, "records": 6, "partition": [], "spec_id": 0, "seq": 1}],
        [{"path": delete, "records": 5, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]}],
    )
    assert _ids(table) == [0, 2, 4, 5]


def test_duplicate_delete_files_do_not_over_delete(tmp: str) -> None:
    data = _base_data(tmp)
    delete = _write(tmp, "del.parquet", {"id": [1, 4]}, {"id": pa.int64()})
    table = _table(
        tmp,
        [{"path": data, "records": 6, "partition": [], "spec_id": 0, "seq": 1}],
        [
            {"path": delete, "records": 2, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]},
            {"path": delete, "records": 2, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]},
        ],
    )
    assert _ids(table) == [0, 2, 3, 5]


def test_identical_data_rows_are_independent(tmp: str) -> None:
    # Two rows with identical values are two physical rows; a delete matching
    # that value removes both only because both satisfy the predicate.
    data = _write(tmp, "data.parquet", {"id": [7, 7, 8], "cat": ["a", "a", "b"], "flag": [True, True, None], "val": [0, 0, 1]})
    delete = _write(tmp, "del.parquet", {"id": [7]}, {"id": pa.int64()})
    table = _table(
        tmp,
        [{"path": data, "records": 3, "partition": [], "spec_id": 0, "seq": 1}],
        [{"path": delete, "records": 1, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]}],
    )
    assert _ids(table) == [8]


# ---------------------------------------------------------------------------
# Schema evolution
# ---------------------------------------------------------------------------
def test_renamed_key_matches_by_id(tmp: str) -> None:
    current = Schema(
        NestedField(1, "identifier", LongType(), required=False),
        NestedField(2, "cat", StringType(), required=False),
        NestedField(5, "added", StringType(), required=False),
        schema_id=1,
    )
    old = simple_schema()
    data = _write(tmp, "data.parquet", {"id": [0, 1, 2], "cat": ["a", "b", "c"], "flag": [True, False, True], "val": [0, 1, 2]})
    delete = _write(tmp, "del.parquet", {"id": [1]}, {"id": pa.int64()})
    table = build_table(
        tmp,
        current,
        {0: UNPARTITIONED_PARTITION_SPEC},
        schemas=[old, current],
        data_files=[{"path": data, "records": 3, "partition": [], "spec_id": 0, "seq": 1}],
        delete_files=[{"path": delete, "records": 1, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]}],
    )
    assert sorted(table.scan().to_arrow()["identifier"].to_pylist()) == [0, 2]


def test_dropped_key_still_applied(tmp: str) -> None:
    # Key field 3 ("flag") was dropped from the current schema; the delete file
    # must still use it.
    current = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "cat", StringType(), required=False),
        schema_id=1,
    )
    old = simple_schema()
    data = _write(tmp, "data.parquet", {"id": [0, 1, 2], "cat": ["a", "b", "c"], "flag": [True, False, True], "val": [0, 1, 2]})
    delete = _write(tmp, "del.parquet", {"flag": [True]}, {"flag": pa.bool_()}, field_ids={"flag": 3})
    table = build_table(
        tmp,
        current,
        {0: UNPARTITIONED_PARTITION_SPEC},
        schemas=[old, current],
        data_files=[{"path": data, "records": 3, "partition": [], "spec_id": 0, "seq": 1}],
        delete_files=[{"path": delete, "records": 1, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [3]}],
    )
    # Both rows with flag=True (id 0 and 2) removed, though flag is not projected.
    assert _ids(table) == [1]
    assert table.scan(selected_fields=("id",)).to_arrow().schema.names == ["id"]


def test_old_file_missing_added_key_is_null(tmp: str) -> None:
    # Field 5 ("added") was added later; the old data file lacks it, so it reads
    # as null and a delete key of null should remove it.
    current = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(5, "added", StringType(), required=False),
        schema_id=1,
    )
    old = Schema(NestedField(1, "id", LongType(), required=False))
    data = _write(tmp, "data.parquet", {"id": [0, 1, 2]}, {"id": pa.int64()}, field_ids={"id": 1})
    delete = _write(tmp, "del.parquet", {"added": [None]}, {"added": pa.string()}, field_ids={"added": 5})
    table = build_table(
        tmp,
        current,
        {0: UNPARTITIONED_PARTITION_SPEC},
        schemas=[old, current],
        data_files=[{"path": data, "records": 3, "partition": [], "spec_id": 0, "seq": 1}],
        delete_files=[{"path": delete, "records": 1, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [5]}],
    )
    assert _ids(table) == []


def test_int_key_matches_promoted_long_column(tmp: str) -> None:
    current = Schema(NestedField(1, "id", LongType(), required=False), schema_id=1)
    old = Schema(NestedField(1, "id", IntegerType(), required=False))
    data = _write(tmp, "data.parquet", {"id": [1, 2, 3]}, {"id": pa.int32()}, field_ids={"id": 1})
    # Delete key declared as int32 (legal under the old schema).
    delete = _write(tmp, "del.parquet", {"id": [2]}, {"id": pa.int32()}, field_ids={"id": 1})
    table = build_table(
        tmp,
        current,
        {0: UNPARTITIONED_PARTITION_SPEC},
        schemas=[old, current],
        data_files=[{"path": data, "records": 3, "partition": [], "spec_id": 0, "seq": 1}],
        delete_files=[{"path": delete, "records": 1, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]}],
    )
    assert _ids(table) == [1, 3]


def test_equality_ids_order_and_subset(tmp: str) -> None:
    data = _write(tmp, "data.parquet", {"id": [0, 1, 2], "cat": ["a", "b", "c"], "flag": [True, False, True], "val": [0, 1, 2]})
    # equality_ids in a different order than the schema still means (flag, id).
    delete = _write(
        tmp, "del.parquet", {"flag": [True], "id": [0]}, {"id": pa.int64(), "flag": pa.bool_()}, field_ids={"id": 1, "flag": 3}
    )
    table = _table(
        tmp,
        [{"path": data, "records": 3, "partition": [], "spec_id": 0, "seq": 1}],
        [{"path": delete, "records": 1, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [3, 1]}],
    )
    assert _ids(table) == [1, 2]


# ---------------------------------------------------------------------------
# Empty / all-deleted / projection
# ---------------------------------------------------------------------------
def test_empty_delete_file(tmp: str) -> None:
    data = _base_data(tmp)
    delete = _write(tmp, "empty.parquet", {"id": []}, {"id": pa.int64()}, field_ids={"id": 1})
    table = _table(
        tmp,
        [{"path": data, "records": 6, "partition": [], "spec_id": 0, "seq": 1}],
        [{"path": delete, "records": 0, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]}],
    )
    assert _ids(table) == [0, 1, 2, 3, 4, 5]


def test_all_rows_deleted(tmp: str) -> None:
    data = _base_data(tmp)
    delete = _write(tmp, "del.parquet", {"id": [0, 1, 2, 3, 4, 5]}, {"id": pa.int64()}, field_ids={"id": 1})
    table = _table(
        tmp,
        [{"path": data, "records": 6, "partition": [], "spec_id": 0, "seq": 1}],
        [{"path": delete, "records": 6, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]}],
    )
    assert _ids(table) == []
    assert table.scan().count() == 0
    assert table.scan().to_arrow_batch_reader().read_all().num_rows == 0


def test_projection_ignores_delete_key_columns(tmp: str) -> None:
    data = _base_data(tmp)
    delete = _write(tmp, "del.parquet", {"id": [1, 4]}, {"id": pa.int64()}, field_ids={"id": 1})
    table = _table(
        tmp,
        [{"path": data, "records": 6, "partition": [], "spec_id": 0, "seq": 1}],
        [{"path": delete, "records": 2, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]}],
    )
    projected = table.scan(selected_fields=("cat",)).to_arrow()
    assert projected.schema.names == ["cat"]
    # surviving ids are 0, 2, 3, 5 -> cats a, c, a, c
    assert sorted(projected["cat"].to_pylist()) == ["a", "a", "c", "c"]


# ---------------------------------------------------------------------------
# Filters, limits, count, batch reader consistency
# ---------------------------------------------------------------------------
def _filtered_table(tmp: str) -> Table:
    data = _base_data(tmp)
    delete = _write(tmp, "del.parquet", {"id": [1, 4]}, {"id": pa.int64()}, field_ids={"id": 1})
    return _table(
        tmp,
        [{"path": data, "records": 6, "partition": [], "spec_id": 0, "seq": 1}],
        [{"path": delete, "records": 2, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]}],
    )


def test_user_filter_does_not_prune_delete(tmp: str) -> None:
    table = _filtered_table(tmp)
    assert _ids(table, row_filter="id > 2") == [3, 5]


def test_limit_applies_after_deletes_and_filter(tmp: str) -> None:
    table = _filtered_table(tmp)
    limited = sorted(table.scan(limit=2).to_arrow()["id"].to_pylist())
    assert limited == [0, 2]
    assert table.scan(row_filter="id >= 2", limit=2).to_arrow().num_rows == 2


def test_to_arrow_batch_reader_matches_to_arrow(tmp: str) -> None:
    table = _filtered_table(tmp)
    expected = _ids(table)
    reader = table.scan().to_arrow_batch_reader()
    assert isinstance(reader, pa.RecordBatchReader)
    assert sorted(reader.read_all()["id"].to_pylist()) == expected
    assert sorted(table.scan().to_arrow_batch_reader().read_all()["id"].to_pylist()) == expected


def test_count_matches_to_arrow_and_reader(tmp: str) -> None:
    table = _filtered_table(tmp)
    assert table.scan().count() == len(table.scan().to_arrow())
    assert table.scan(row_filter="id > 2").count() == len(table.scan(row_filter="id > 2").to_arrow())
    assert table.scan(row_filter="id > 2").count() == table.scan(row_filter="id > 2").to_arrow_batch_reader().read_all().num_rows


def test_batch_reader_limit(tmp: str) -> None:
    table = _filtered_table(tmp)
    assert sorted(table.scan(limit=2).to_arrow_batch_reader().read_all()["id"].to_pylist()) == [0, 2]
    assert table.scan(limit=0).to_arrow_batch_reader().read_all().num_rows == 0


# ---------------------------------------------------------------------------
# Mixed position + equality deletes
# ---------------------------------------------------------------------------
def test_mixed_position_and_equality_deletes_across_row_groups(tmp: str) -> None:
    data = _write(
        tmp,
        "data.parquet",
        {"id": list(range(10)), "cat": ["a"] * 10, "flag": [True] * 10, "val": list(range(100, 110))},
        row_group_size=3,
    )
    # Original row positions 1 and 7 (spanning row groups) plus id in {2, 8}.
    pos = write_parquet(
        os.path.join(tmp, "pos.parquet"),
        {"file_path": [data, data], "pos": [1, 7]},
        {"file_path": 2147483546, "pos": 2147483545},
        {"file_path": pa.string(), "pos": pa.int64()},
    )
    eq = _write(tmp, "eq.parquet", {"id": [2, 8]}, {"id": pa.int64()})
    table = _table(
        tmp,
        [{"path": data, "records": 10, "partition": [], "spec_id": 0, "seq": 1}],
        [
            {"path": pos, "records": 2, "partition": [], "spec_id": 0, "seq": 20, "pos": True},
            {"path": eq, "records": 2, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]},
        ],
    )
    assert _ids(table) == [0, 3, 4, 5, 6, 9]
    # With a filter, position deletes must still reference original row numbers.
    assert _ids(table, row_filter="val > 102") == [3, 4, 5, 6, 9]
    assert table.scan().count() == 6


def test_mixed_deletes_multiple_row_groups_filter_and_limit(tmp: str) -> None:
    data = _write(
        tmp,
        "data.parquet",
        {"id": list(range(20)), "cat": ["a"] * 20, "flag": [True] * 20, "val": list(range(20))},
        row_group_size=4,
    )
    pos = write_parquet(
        os.path.join(tmp, "pos.parquet"),
        {"file_path": [data] * 4, "pos": [0, 5, 10, 15]},
        {"file_path": 2147483546, "pos": 2147483545},
        {"file_path": pa.string(), "pos": pa.int64()},
    )
    eq = _write(tmp, "eq.parquet", {"id": [3, 12]}, {"id": pa.int64()})
    table = _table(
        tmp,
        [{"path": data, "records": 20, "partition": [], "spec_id": 0, "seq": 1}],
        [
            {"path": pos, "records": 4, "partition": [], "spec_id": 0, "seq": 20, "pos": True},
            {"path": eq, "records": 2, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [1]},
        ],
    )
    expected = sorted(set(range(20)) - {0, 5, 10, 15, 3, 12})
    assert _ids(table) == expected
    filtered = sorted(set(expected) & {i for i in range(20) if i > 8})
    assert _ids(table, row_filter="id > 8") == filtered


# ---------------------------------------------------------------------------
# Unsupported types
# ---------------------------------------------------------------------------
def test_unsupported_key_type_raises(tmp: str) -> None:
    from pyiceberg.types import DoubleType

    schema = Schema(NestedField(6, "dbl", DoubleType()))
    data = _write(tmp, "data.parquet", {"dbl": [1.5, 2.5]}, {"dbl": pa.float64()}, field_ids={"dbl": 6})
    delete = _write(tmp, "del.parquet", {"dbl": [1.5]}, {"dbl": pa.float64()}, field_ids={"dbl": 6})
    table = build_table(
        tmp,
        schema,
        {0: UNPARTITIONED_PARTITION_SPEC},
        [{"path": data, "records": 2, "partition": [], "spec_id": 0, "seq": 1}],
        [{"path": delete, "records": 1, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [6]}],
    )
    with pytest.raises(ValueError, match="only top-level boolean, int, long and string"):
        table.scan().to_arrow()


# ---------------------------------------------------------------------------
# Memory budget option
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("invalid", ["0", "-1", "abc", "", "65535", "65536.0", "1e6", " 65536", "+65536"])
def test_invalid_memory_budget_raises_before_reading(tmp: str, invalid: str) -> None:
    table = _filtered_table(tmp)
    # Remove the data file: if the option is validated first we get ValueError,
    # not a missing-file error.
    os.unlink(table.scan().plan_files()[0].file.file_path)
    with pytest.raises(ValueError, match="equality-delete-memory-budget"):
        table.scan(options={"equality-delete-memory-budget": invalid}).to_arrow()


def test_minimum_budget_accepted(tmp: str) -> None:
    table = _filtered_table(tmp)
    assert _ids(table, options={"equality-delete-memory-budget": "65536"}) == [0, 2, 3, 5]


def test_single_key_over_budget_errors(tmp: str) -> None:
    data = _write(tmp, "data.parquet", {"cat": ["a"]}, {"cat": pa.string()}, field_ids={"cat": 2})
    huge = "x" * 100000
    delete = _write(tmp, "del.parquet", {"cat": [huge]}, {"cat": pa.string()}, field_ids={"cat": 2})
    table = _table(
        tmp,
        [{"path": data, "records": 1, "partition": [], "spec_id": 0, "seq": 1}],
        [{"path": delete, "records": 1, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [2]}],
    )
    with pytest.raises(ValueError, match="exceeds the entire equality-delete-memory-budget"):
        table.scan(options={"equality-delete-memory-budget": "65536"}).to_arrow()
    # A budget large enough for the key succeeds (the key matches nothing).
    assert table.scan(options={"equality-delete-memory-budget": "1048576"}).to_arrow().num_rows == 1
