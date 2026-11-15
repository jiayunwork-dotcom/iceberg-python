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
# pylint: disable=redefined-outer-name,protected-access
"""End-to-end equality delete tests through the public scan entry points.

Every test constructs real Parquet files, Avro manifests/manifest lists and
JSON metadata on a local filesystem, then reads the table back through
``StaticTable.from_metadata(...).scan()``.
"""

from __future__ import annotations

import datetime
import decimal
import os
from typing import List, Tuple

import pyarrow as pa
import pytest

from pyiceberg.expressions import EqualTo, GreaterThanOrEqual
from pyiceberg.io.equality_deletes import EqualityDeleteError
from pyiceberg.manifest import (
    DataFile,
    DataFileContent,
    ManifestContent,
)
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import NestedField, Schema
from pyiceberg.table import StaticTable
from pyiceberg.transforms import IdentityTransform
from pyiceberg.typedef import Record
from pyiceberg.types import (
    BooleanType,
    DateType,
    DecimalType,
    IntegerType,
    ListType,
    LongType,
    StringType,
    StructType,
    TimestampType,
    TimestamptzType,
    TimeType,
)
from utils.equality_delete_table import (
    LocalV2TableBuilder,
    PlannedFile,
    PlannedSnapshot,
    parquet_table_with_field_ids,
    write_corrupt_parquet,
)

# ---------------------------------------------------------------------------
# Schemas / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_schema() -> Schema:
    return Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "name", StringType()),
        NestedField(3, "payload", StringType()),
        schema_id=0,
    )


@pytest.fixture
def unpartitioned_spec() -> PartitionSpec:
    return PartitionSpec(spec_id=0)


def _data_file(rows: pa.Table, spec_id: int, partition: Record, sequence_number: int) -> PlannedFile:
    return PlannedFile(rows, DataFileContent.DATA, spec_id, partition, sequence_number)


def _eq_delete_file(
    rows: pa.Table,
    spec_id: int,
    partition: Record,
    sequence_number: int,
    equality_ids: List[int],
) -> PlannedFile:
    return PlannedFile(
        rows,
        DataFileContent.EQUALITY_DELETES,
        spec_id,
        partition,
        sequence_number,
        equality_ids=equality_ids,
    )


def _build(builder: LocalV2TableBuilder, snapshots: List[PlannedSnapshot]) -> StaticTable:
    location = builder.build(snapshots)
    return builder.load(location)


def _rows(table_or_scan, **scan_kwargs) -> List[Tuple]:
    scan = table_or_scan.scan(**scan_kwargs) if hasattr(table_or_scan, "scan") else table_or_scan
    arrow_table = scan.to_arrow()
    return [tuple(row.values()) for row in arrow_table.to_pylist()]


def _batch_rows(table_or_scan, **scan_kwargs) -> List[Tuple]:
    scan = table_or_scan.scan(**scan_kwargs) if hasattr(table_or_scan, "scan") else table_or_scan
    reader = scan.to_arrow_batch_reader()
    rows: List[Tuple] = []
    for batch in reader:
        rows.extend(tuple(row.values()) for row in batch.to_pylist())
    return rows


# ---------------------------------------------------------------------------
# Basic matching semantics
# ---------------------------------------------------------------------------


def test_basic_equality_delete(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2, 3, 4], pa.int64()), 1),
            "name": (pa.array(["a", "b", "c", "d"]), 2),
            "payload": (pa.array(["p1", "p2", "p3", "p4"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([2, 4], pa.int64()), 1)})

    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(
                2,
                1,
                2,
                [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])],
                0,
            ),
        ],
    )

    expected = [(1, "a", "p1"), (3, "c", "p3")]
    assert _rows(table) == expected
    assert _batch_rows(table) == expected
    assert table.scan().count() == 2


def test_duplicate_data_rows_all_hit(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 1, 2, 1], pa.int64()), 1),
            "name": (pa.array(["a", "a", "b", "a"]), 2),
            "payload": (pa.array(["x", "y", "z", "w"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    assert _rows(table) == [(2, "b", "z")]
    assert table.scan().count() == 1


def test_duplicate_delete_keys_delete_once(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["x", "y"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([2, 2, 2], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    assert _rows(table) == [(1, "a", "x")]


def test_empty_equality_delete_file_deletes_nothing(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["x", "y"]), 3),
        }
    )
    empty_deletes = parquet_table_with_field_ids({"id": (pa.array([], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(
                2,
                1,
                2,
                [(0, ManifestContent.DELETES, [_eq_delete_file(empty_deletes, 0, Record(), 2, [1])])],
                0,
            ),
        ],
    )
    assert table.scan().count() == 2
    assert len(_rows(table)) == 2


def test_null_matches_null_and_composite_key(tmp_path, unpartitioned_spec) -> None:
    schema = Schema(
        NestedField(1, "k1", LongType()),
        NestedField(2, "k2", StringType()),
        NestedField(3, "payload", StringType()),
        schema_id=0,
    )
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "k1": (pa.array([1, None, 1, None], pa.int64()), 1),
            "k2": (pa.array(["a", "a", None, None]), 2),
            "payload": (pa.array(["r1", "r2", "r3", "r4"]), 3),
        }
    )
    # Composite key (k1, k2): delete (1, None) and (None, None), but not (1, "a") or (None, "a")
    deletes = parquet_table_with_field_ids(
        {
            "k1": (pa.array([1, None], pa.int64()), 1),
            "k2": (pa.array([None, None]), 2),
        }
    )
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1, 2])])], 0),
        ],
    )
    assert _rows(table) == [(1, "a", "r1"), (None, "a", "r2")]


def test_different_equality_id_sets_combine_and_extra_columns_ignored(tmp_path, unpartitioned_spec) -> None:
    schema = Schema(
        NestedField(1, "k1", LongType()),
        NestedField(2, "k2", StringType()),
        NestedField(3, "payload", StringType()),
        schema_id=0,
    )
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "k1": (pa.array([1, 2, 3], pa.int64()), 1),
            "k2": (pa.array(["a", "b", "c"]), 2),
            "payload": (pa.array(["x", "y", "z"]), 3),
        }
    )
    deletes_by_k1 = parquet_table_with_field_ids(
        {
            "k1": (pa.array([1], pa.int64()), 1),
            # Extra, non-key column must not participate in matching.
            "payload": (pa.array(["different"]), 3),
        }
    )
    deletes_by_k2 = parquet_table_with_field_ids({"k2": (pa.array(["c"]), 2)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(
                2,
                1,
                2,
                [
                    (
                        0,
                        ManifestContent.DELETES,
                        [
                            _eq_delete_file(deletes_by_k1, 0, Record(), 2, [1]),
                            _eq_delete_file(deletes_by_k2, 0, Record(), 2, [2]),
                        ],
                    )
                ],
                0,
            ),
        ],
    )
    assert _rows(table) == [(2, "b", "y")]


def test_all_rows_deleted_and_zero_hit(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["x", "y"]), 3),
        }
    )
    all_deletes = parquet_table_with_field_ids({"id": (pa.array([1, 2], pa.int64()), 1)})
    none_deletes = parquet_table_with_field_ids({"id": (pa.array([99], pa.int64()), 1)})

    all_deleted = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(all_deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    assert all_deleted.scan().count() == 0
    assert _rows(all_deleted) == []
    assert _batch_rows(all_deleted) == []

    builder2 = LocalV2TableBuilder(str(tmp_path), "t2", [simple_schema], 0, [unpartitioned_spec])
    none_deleted = _build(
        builder2,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(none_deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    assert none_deleted.scan().count() == 2


def test_empty_table_scan(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    table = _build(builder, [])
    assert table.scan().count() == 0
    assert table.scan().to_arrow().num_rows == 0
    assert _batch_rows(table) == []


# ---------------------------------------------------------------------------
# Sequence number semantics
# ---------------------------------------------------------------------------


def test_sequence_number_equal_keeps_new_data(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    old_data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["x", "y"]), 3),
        }
    )
    new_data = parquet_table_with_field_ids(
        {
            "id": (pa.array([2, 3], pa.int64()), 1),
            "name": (pa.array(["b", "c"]), 2),
            "payload": (pa.array(["y2", "z"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([2], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(old_data, 0, Record(), 1)])], 0),
            PlannedSnapshot(
                2,
                1,
                2,
                [
                    (0, ManifestContent.DATA, [_data_file(new_data, 0, Record(), 2)]),
                    (0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])]),
                ],
                0,
            ),
        ],
    )
    # seq1 row id=2 is removed, the seq2 row id=2 stays because equality
    # deletes never affect rows of the same sequence number.
    assert sorted(_rows(table)) == sorted([(1, "a", "x"), (2, "b", "y2"), (3, "c", "z")])


def test_sequence_number_greater_delete_does_not_apply_when_data_newer(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1], pa.int64()), 1),
            "name": (pa.array(["a"]), 2),
            "payload": (pa.array(["x"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    table = _build(
        builder,
        [
            # Delete committed first at seq 1 ...
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 1, [1])])], 0),
            # ... data arrives later at seq 2, so the older delete must not hit it.
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 2)])], 0),
        ],
    )
    assert _rows(table) == [(1, "a", "x")]


def test_sequence_number_inheritance_from_manifest(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["x", "y"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    # Entries with an explicit None sequence number inherit the manifest
    # list sequence number, so both must resolve to 1 (data) and 2 (delete).
    table = _build(
        builder,
        [
            PlannedSnapshot(
                1,
                None,
                1,
                [(0, ManifestContent.DATA, [PlannedFile(data, DataFileContent.DATA, 0, Record(), None)])],
                0,
            ),
            PlannedSnapshot(
                2,
                1,
                2,
                [
                    (
                        0,
                        ManifestContent.DELETES,
                        [PlannedFile(deletes, DataFileContent.EQUALITY_DELETES, 0, Record(), None, equality_ids=[1])],
                    )
                ],
                0,
            ),
        ],
    )
    assert _rows(table) == [(2, "b", "y")]


# ---------------------------------------------------------------------------
# Partition scoping and spec evolution
# ---------------------------------------------------------------------------


def test_partitioned_delete_requires_matching_partition(tmp_path) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "part", StringType()),
        NestedField(3, "payload", StringType()),
        schema_id=0,
    )
    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="part"), spec_id=0)
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [spec])
    data_a = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "part": (pa.array(["a", "a"]), 2),
            "payload": (pa.array(["x", "y"]), 3),
        }
    )
    data_b = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 3], pa.int64()), 1),
            "part": (pa.array(["b", "b"]), 2),
            "payload": (pa.array(["x", "z"]), 3),
        }
    )
    # Delete id=1 but only for partition "a"; the id=1 row in "b" survives.
    deletes_a = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(
                1,
                None,
                1,
                [
                    (0, ManifestContent.DATA, [_data_file(data_a, 0, Record(part="a"), 1)]),
                    (0, ManifestContent.DATA, [_data_file(data_b, 0, Record(part="b"), 1)]),
                ],
                0,
            ),
            PlannedSnapshot(
                2,
                1,
                2,
                [(0, ManifestContent.DELETES, [_eq_delete_file(deletes_a, 0, Record(part="a"), 2, [1])])],
                0,
            ),
        ],
    )
    assert sorted(_rows(table)) == sorted([(2, "a", "y"), (1, "b", "x"), (3, "b", "z")])


def test_unpartitioned_spec_delete_is_global_across_specs(tmp_path) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "part", StringType()),
        NestedField(3, "payload", StringType()),
        schema_id=0,
    )
    unpartitioned = PartitionSpec(spec_id=0)
    partitioned = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="part"), spec_id=1)
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [unpartitioned, partitioned], default_spec_id=1)
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "part": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["x", "y"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    table = _build(
        builder,
        [
            # Data written with the partitioned spec ...
            PlannedSnapshot(1, None, 1, [(1, ManifestContent.DATA, [_data_file(data, 1, Record(part="a"), 1)])], 0),
            # ... while the delete uses the legacy unpartitioned spec: global.
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    assert _rows(table) == [(2, "b", "y")]


def test_partition_evolution_multiple_specs_scoped_per_spec(tmp_path) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "part_a", StringType()),
        NestedField(3, "part_b", StringType()),
        NestedField(4, "payload", StringType()),
        schema_id=0,
    )
    spec_v0 = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="part_a"), spec_id=0)
    spec_v1 = PartitionSpec(PartitionField(source_id=3, field_id=1001, transform=IdentityTransform(), name="part_b"), spec_id=1)
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [spec_v0, spec_v1], default_spec_id=1)
    data_v0 = parquet_table_with_field_ids(
        {
            "id": (pa.array([1], pa.int64()), 1),
            "part_a": (pa.array(["x"]), 2),
            "part_b": (pa.array(["m"]), 3),
            "payload": (pa.array(["old"]), 4),
        }
    )
    data_v1_n = parquet_table_with_field_ids(
        {
            "id": (pa.array([1], pa.int64()), 1),
            "part_a": (pa.array(["x"]), 2),
            "part_b": (pa.array(["n"]), 3),
            "payload": (pa.array(["new1"]), 4),
        }
    )
    data_v1_o = parquet_table_with_field_ids(
        {
            "id": (pa.array([2], pa.int64()), 1),
            "part_a": (pa.array(["y"]), 2),
            "part_b": (pa.array(["o"]), 3),
            "payload": (pa.array(["new2"]), 4),
        }
    )
    # Delete written with the new spec id 1 partition part_b=n must not touch
    # the old spec-0 file even though id=1 exists there too.
    deletes = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data_v0, 0, Record(part_a="x"), 1)])], 0),
            PlannedSnapshot(
                2,
                1,
                2,
                [
                    (1, ManifestContent.DATA, [_data_file(data_v1_n, 1, Record(part_b="n"), 2)]),
                    (1, ManifestContent.DATA, [_data_file(data_v1_o, 1, Record(part_b="o"), 2)]),
                ],
                0,
            ),
            PlannedSnapshot(
                3,
                2,
                3,
                [(1, ManifestContent.DELETES, [_eq_delete_file(deletes, 1, Record(part_b="n"), 3, [1])])],
                0,
            ),
        ],
    )
    rows = _rows(table)
    # Old file row survives (different spec), spec-1 part_b=n id=1 removed,
    # part_b=o rows survive.
    assert (1, "x", "m", "old") in rows
    assert (2, "y", "o", "new2") in rows
    assert all(not (r[0] == 1 and r[2] == "n") for r in rows)


# ---------------------------------------------------------------------------
# Key types, nesting and field identity
# ---------------------------------------------------------------------------


def test_all_supported_primitive_key_types(tmp_path, unpartitioned_spec) -> None:
    schema = Schema(
        NestedField(1, "b", BooleanType()),
        NestedField(2, "i", IntegerType()),
        NestedField(3, "l", LongType()),
        NestedField(4, "s", StringType()),
        NestedField(5, "d", DecimalType(9, 2)),
        NestedField(6, "day", DateType()),
        NestedField(7, "tod", TimeType()),
        NestedField(8, "ts", TimestampType()),
        NestedField(9, "tstz", TimestamptzType()),
        NestedField(10, "payload", StringType()),
        schema_id=0,
    )
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [unpartitioned_spec])

    values = {
        "b": (pa.array([True, False, None], pa.bool_()), 1),
        "i": (pa.array([10, 20, 30], pa.int32()), 2),
        "l": (pa.array([100, 200, 300], pa.int64()), 3),
        "s": (pa.array(["a", "b", "c"]), 4),
        "d": (pa.array([decimal.Decimal("1.10"), decimal.Decimal("2.20"), decimal.Decimal("3.30")], pa.decimal128(9, 2)), 5),
        "day": (pa.array([datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)], pa.date32()), 6),
        "tod": (
            pa.array(
                [datetime.time(1, 2, 3), datetime.time(4, 5, 6), datetime.time(7, 8, 9)],
                pa.time64("us"),
            ),
            7,
        ),
        "ts": (
            pa.array(
                [datetime.datetime(2024, 1, 1, 1), datetime.datetime(2024, 1, 2, 2), datetime.datetime(2024, 1, 3, 3)],
                pa.timestamp("us"),
            ),
            8,
        ),
        "tstz": (
            pa.array(
                [
                    datetime.datetime(2024, 1, 1, 1, tzinfo=datetime.timezone.utc),
                    datetime.datetime(2024, 1, 2, 2, tzinfo=datetime.timezone.utc),
                    datetime.datetime(2024, 1, 3, 3, tzinfo=datetime.timezone.utc),
                ],
                pa.timestamp("us", tz="UTC"),
            ),
            9,
        ),
        "payload": (pa.array(["p1", "p2", "p3"]), 10),
    }
    data = parquet_table_with_field_ids(values)

    # Delete the middle row on every primitive type at once.
    delete_values = {name: (arr.slice(1, 1), field_id) for name, (arr, field_id) in values.items() if name != "payload"}
    deletes = parquet_table_with_field_ids(delete_values)
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(
                2,
                1,
                2,
                [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, list(range(1, 10)))])],
                0,
            ),
        ],
    )
    rows = _rows(table)
    payloads = [row[-1] for row in rows]
    assert payloads == ["p1", "p3"]


def test_optional_struct_key_leaves_and_null_parent(tmp_path, unpartitioned_spec) -> None:
    schema = Schema(
        NestedField(
            1,
            "location",
            StructType(
                NestedField(10, "lat", LongType()),
                NestedField(11, "lon", LongType()),
            ),
        ),
        NestedField(2, "payload", StringType()),
        schema_id=0,
    )
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "location": (
                pa.array(
                    [{"lat": 1, "lon": 2}, {"lat": 3, "lon": 4}, None, {"lat": 1, "lon": 9}],
                    pa.struct(
                        [
                            pa.field("lat", pa.int64(), metadata={b"PARQUET:field_id": "10"}),
                            pa.field("lon", pa.int64(), metadata={b"PARQUET:field_id": "11"}),
                        ]
                    ),
                ),
                1,
            ),
            "payload": (pa.array(["r1", "r2", "r3", "r4"]), 2),
        }
    )
    # Delete (lat=3, lon=4) and the null-parent row (both leaves null).
    deletes = parquet_table_with_field_ids(
        {
            "location": (
                pa.array(
                    [{"lat": 3, "lon": 4}, {"lat": None, "lon": None}],
                    pa.struct(
                        [
                            pa.field("lat", pa.int64(), metadata={b"PARQUET:field_id": "10"}),
                            pa.field("lon", pa.int64(), metadata={b"PARQUET:field_id": "11"}),
                        ]
                    ),
                ),
                1,
            ),
        }
    )
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [10, 11])])], 0),
        ],
    )
    assert [row[1] for row in _rows(table)] == ["r1", "r4"]


def test_field_identity_rename_reorder_and_same_name_different_id(tmp_path) -> None:
    old_schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "name", StringType()),
        schema_id=0,
    )
    # Column renamed and reordered; ids are unchanged.
    new_schema = Schema(
        NestedField(2, "label", StringType()),
        NestedField(1, "identifier", LongType()),
        schema_id=1,
    )
    spec = PartitionSpec(spec_id=0)
    builder = LocalV2TableBuilder(str(tmp_path), "t", [old_schema, new_schema], 1, [spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
        }
    )
    # Delete file written against the old names still matches by field id.
    deletes = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 1),
        ],
    )
    # Output follows the current (renamed, reordered) schema.
    arrow_table = table.scan().to_arrow()
    assert arrow_table.column_names == ["label", "identifier"]
    assert arrow_table.to_pylist() == [{"label": "b", "identifier": 2}]


def test_dropped_key_column_still_resolved_from_schema_history(tmp_path) -> None:
    old_schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "secret", StringType()),
        NestedField(3, "payload", StringType()),
        schema_id=0,
    )
    new_schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(3, "payload", StringType()),
        schema_id=1,
    )
    spec = PartitionSpec(spec_id=0)
    builder = LocalV2TableBuilder(str(tmp_path), "t", [old_schema, new_schema], 1, [spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "secret": (pa.array(["s1", "s2"]), 2),
            "payload": (pa.array(["p1", "p2"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"secret": (pa.array(["s1"]), 2)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [2])])], 1),
        ],
    )
    assert _rows(table) == [(2, "p2")]


def test_new_optional_column_missing_in_old_data_file_is_null(tmp_path) -> None:
    old_schema = Schema(
        NestedField(1, "id", LongType()),
        schema_id=0,
    )
    new_schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "tag", StringType()),
        schema_id=1,
    )
    spec = PartitionSpec(spec_id=0)
    builder = LocalV2TableBuilder(str(tmp_path), "t", [old_schema, new_schema], 1, [spec])
    # Old file only has id; tag is added later as an optional column.
    old_data = parquet_table_with_field_ids({"id": (pa.array([1, 2], pa.int64()), 1)})
    new_data = parquet_table_with_field_ids(
        {
            "id": (pa.array([3], pa.int64()), 1),
            "tag": (pa.array(["z"]), 2),
        }
    )
    # Deleting tag=null must remove the old-file rows (projected null), but
    # not the new row whose tag is "z".
    deletes = parquet_table_with_field_ids({"tag": (pa.array([None]), 2)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(old_data, 0, Record(), 1)])], 0),
            PlannedSnapshot(
                2,
                1,
                2,
                [
                    (0, ManifestContent.DATA, [_data_file(new_data, 0, Record(), 2)]),
                    (0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [2])]),
                ],
                1,
            ),
        ],
    )
    assert _rows(table) == [(3, "z")]


def test_int_long_and_decimal_precision_promotions(tmp_path, unpartitioned_spec) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "amount", DecimalType(12, 2)),
        schema_id=0,
    )
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [unpartitioned_spec])
    # Data file written with narrower physical types (int32, decimal(8,2)).
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int32()), 1),
            "amount": (
                pa.array([decimal.Decimal("10.00"), decimal.Decimal("20.00")], pa.decimal128(8, 2)),
                2,
            ),
        }
    )
    deletes = parquet_table_with_field_ids(
        {
            "id": (pa.array([1], pa.int64()), 1),
            "amount": (pa.array([decimal.Decimal("10.00")], pa.decimal128(12, 2)), 2),
        }
    )
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1, 2])])], 0),
        ],
    )
    assert _rows(table) == [(2, decimal.Decimal("20.00"))]


def test_list_key_type_is_rejected(tmp_path, unpartitioned_spec) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "tags", ListType(element_id=10, element_type=StringType())),
        schema_id=0,
    )
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1], pa.int64()), 1),
            "tags": (
                pa.array([["a"]], pa.list_(pa.field("element", pa.large_string(), metadata={b"PARQUET:field_id": "10"}))),
                2,
            ),
        }
    )
    deletes = parquet_table_with_field_ids(
        {
            "tags": (
                pa.array([["a"]], pa.list_(pa.field("element", pa.large_string(), metadata={b"PARQUET:field_id": "10"}))),
                2,
            ),
        }
    )
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [2])])], 0),
        ],
    )
    with pytest.raises(EqualityDeleteError, match="unsupported type"):
        table.scan().to_arrow()


# ---------------------------------------------------------------------------
# Filters, projections, limits and mixed position deletes
# ---------------------------------------------------------------------------


def test_hidden_key_projection_only_payload(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2, 3], pa.int64()), 1),
            "name": (pa.array(["a", "b", "c"]), 2),
            "payload": (pa.array(["p1", "p2", "p3"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([2], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    arrow_table = table.scan(selected_fields=("payload",)).to_arrow()
    assert arrow_table.column_names == ["payload"]
    assert arrow_table.to_pylist() == [{"payload": "p1"}, {"payload": "p3"}]
    assert table.scan(selected_fields=("payload",)).count() == 2


def test_row_filter_does_not_resurrect_deleted_rows(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2, 3, 4], pa.int64()), 1),
            "name": (pa.array(["a", "b", "c", "d"]), 2),
            "payload": (pa.array(["p1", "p2", "p3", "p4"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([2, 3], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    result = table.scan(row_filter=GreaterThanOrEqual("id", 2)).to_arrow()
    assert result.to_pylist() == [{"id": 4, "name": "d", "payload": "p4"}]
    assert table.scan(row_filter=EqualTo("name", "b")).count() == 0


def test_limit_fills_with_surviving_rows(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    # Two data files, each loses its first rows to the shared delete.
    data_one = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["p1", "p2"]), 3),
        }
    )
    data_two = parquet_table_with_field_ids(
        {
            "id": (pa.array([3, 4], pa.int64()), 1),
            "name": (pa.array(["c", "d"]), 2),
            "payload": (pa.array(["p3", "p4"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([1, 3], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(
                1,
                None,
                1,
                [
                    (0, ManifestContent.DATA, [_data_file(data_one, 0, Record(), 1)]),
                    (0, ManifestContent.DATA, [_data_file(data_two, 0, Record(), 1)]),
                ],
                0,
            ),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    limited = table.scan(limit=2).to_arrow()
    assert limited.num_rows == 2
    assert limited.to_pylist() == [
        {"id": 2, "name": "b", "payload": "p2"},
        {"id": 4, "name": "d", "payload": "p4"},
    ]


def test_equality_and_position_deletes_combined(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2, 3, 4, 5], pa.int64()), 1),
            "name": (pa.array(["a", "b", "c", "d", "e"]), 2),
            "payload": (pa.array(["p1", "p2", "p3", "p4", "p5"]), 3),
        }
    )
    data_file = PlannedFile(data, DataFileContent.DATA, 0, Record(), 1)
    # Positional delete points at the *physical* row 1 (id=2).  Equality
    # delete removes id=4.  The row filter id>=2 must not shift the positions.
    data_file.file_path = builder._write_parquet(data, "data")
    positional = parquet_table_with_field_ids(
        {
            "file_path": (pa.array([data_file.file_path]), 2147483546),
            "pos": (pa.array([1], pa.int32()), 2147483545),
        }
    )
    equality = parquet_table_with_field_ids({"id": (pa.array([4], pa.int64()), 1)})

    pos_planned = PlannedFile(positional, DataFileContent.POSITION_DELETES, 0, Record(), 2)
    eq_planned = _eq_delete_file(equality, 0, Record(), 2, [1])
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [data_file])], 0),
            PlannedSnapshot(
                2,
                1,
                2,
                [(0, ManifestContent.DELETES, [pos_planned, eq_planned])],
                0,
            ),
        ],
    )
    expected = [
        (1, "a", "p1"),
        (3, "c", "p3"),
        (5, "e", "p5"),
    ]
    assert _rows(table) == expected
    assert _rows(table, row_filter=GreaterThanOrEqual("id", 2)) == [(3, "c", "p3"), (5, "e", "p5")]
    assert table.scan().count() == 3
    assert _batch_rows(table) == expected


# ---------------------------------------------------------------------------
# Snapshot isolation
# ---------------------------------------------------------------------------


def test_historical_snapshot_not_affected_by_newer_deletes(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["p1", "p2"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    historical = table.scan(snapshot_id=1)
    assert sorted(_rows(historical)) == sorted([(1, "a", "p1"), (2, "b", "p2")])
    assert historical.count() == 2
    current = table.scan()
    assert _rows(current) == [(2, "b", "p2")]
    assert current.count() == 1


def test_interleaved_scans_do_not_pollute_each_other(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2, 3], pa.int64()), 1),
            "name": (pa.array(["a", "b", "c"]), 2),
            "payload": (pa.array(["p1", "p2", "p3"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([2], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    reader = table.scan().to_arrow_batch_reader()
    first = reader.read_next_batch()
    # A second, independent eager scan while the batch reader is open.
    eager = table.scan(selected_fields=("payload",)).to_arrow()
    rest = [batch.to_pylist() for batch in reader]
    assert eager.column_names == ["payload"]
    combined = first.to_pylist() + [row for batch in rest for row in batch]
    assert {row["id"] for row in combined} == {1, 3}


def test_failed_scan_does_not_poison_later_scan(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["p1", "p2"]), 3),
        }
    )
    good_deletes = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    bad_path = f"file://{builder.data_dir}/broken.parquet"
    write_corrupt_parquet(bad_path)
    good_table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(good_deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    # Rebuild including a manifest entry that points at the corrupt file.
    corrupt_planned = PlannedFile(good_deletes, DataFileContent.EQUALITY_DELETES, 0, Record(), 3, equality_ids=[1])
    corrupt_planned.file_path = bad_path
    location = builder.build(
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(good_deletes, 0, Record(), 2, [1])])], 0),
            PlannedSnapshot(3, 2, 3, [(0, ManifestContent.DELETES, [corrupt_planned])], 0),
        ]
    )
    broken_table = builder.load(location)
    with pytest.raises(EqualityDeleteError, match=bad_path):
        broken_table.scan().to_arrow()
    # The earlier, healthy table object still scans correctly afterwards.
    assert good_table.scan().count() == 1


# ---------------------------------------------------------------------------
# Malformed delete files
# ---------------------------------------------------------------------------


def _malformed_table(tmp_path, simple_schema, unpartitioned_spec, corrupt: PlannedFile) -> StaticTable:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["p1", "p2"]), 3),
        }
    )
    return _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [corrupt])], 0),
        ],
    )


def test_corrupt_delete_file_fails_with_path(tmp_path, simple_schema, unpartitioned_spec) -> None:
    bad_path = f"file://{tmp_path}/broken.parquet"
    write_corrupt_parquet(bad_path)
    planned = PlannedFile(
        parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)}),
        DataFileContent.EQUALITY_DELETES,
        0,
        Record(),
        2,
        equality_ids=[1],
    )
    planned.file_path = bad_path
    planned.record_count = 1
    table = _malformed_table(tmp_path, simple_schema, unpartitioned_spec, planned)
    with pytest.raises(EqualityDeleteError) as exc_info:
        table.scan().to_arrow()
    assert bad_path in str(exc_info.value)


def test_missing_key_column_in_delete_file_fails(tmp_path, simple_schema, unpartitioned_spec) -> None:
    # Delete declares id (field 1) but the Parquet file only carries name.
    planned = PlannedFile(
        parquet_table_with_field_ids({"name": (pa.array(["a"]), 2)}),
        DataFileContent.EQUALITY_DELETES,
        0,
        Record(),
        2,
        equality_ids=[1],
    )
    table = _malformed_table(tmp_path, simple_schema, unpartitioned_spec, planned)
    with pytest.raises(EqualityDeleteError, match="missing from its physical Parquet schema"):
        table.scan().to_arrow()


def test_unknown_field_id_fails(tmp_path, unpartitioned_spec) -> None:
    schema = Schema(NestedField(1, "id", LongType()), schema_id=0)
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    planned = PlannedFile(
        parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)}),
        DataFileContent.EQUALITY_DELETES,
        0,
        Record(),
        2,
        equality_ids=[999],
    )
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [planned])], 0),
        ],
    )
    with pytest.raises((ValueError, EqualityDeleteError), match="999"):
        table.scan().to_arrow()


def test_incompatible_key_type_fails(tmp_path, unpartitioned_spec) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "name", StringType()),
        schema_id=0,
    )
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1], pa.int64()), 1),
            "name": (pa.array(["a"]), 2),
        }
    )
    # Delete file physically stores strings under field id 1, while the table
    # schema says id is a long.
    planned = PlannedFile(
        parquet_table_with_field_ids({"id": (pa.array(["1"]), 1)}),
        DataFileContent.EQUALITY_DELETES,
        0,
        Record(),
        2,
        equality_ids=[1],
    )
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [planned])], 0),
        ],
    )
    with pytest.raises(EqualityDeleteError):
        table.scan().to_arrow()


def _raw_delete_data_file(file_path: str, equality_ids, record_count: int = 0) -> DataFile:
    return DataFile(
        content=DataFileContent.EQUALITY_DELETES,
        file_path=file_path,
        file_format="PARQUET",
        partition=Record(),
        record_count=record_count,
        file_size_in_bytes=os.path.getsize(file_path.replace("file://", "")),
        column_sizes=None,
        value_counts=None,
        null_value_counts=None,
        nan_value_counts=None,
        lower_bounds=None,
        upper_bounds=None,
        key_metadata=None,
        split_offsets=None,
        equality_ids=equality_ids,
        sort_order_id=None,
        spec_id=0,
    )


def _table_with_raw_delete(tmp_path, simple_schema, unpartitioned_spec, data_file_obj: DataFile) -> StaticTable:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["p1", "p2"]), 3),
        }
    )
    # The raw DataFile is reused for every builder invocation; planning must
    # reject it before its Parquet content is opened.
    data_file_obj.spec_id = 0
    return _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(
                2,
                1,
                2,
                [
                    (
                        0,
                        ManifestContent.DELETES,
                        [
                            PlannedFile(
                                None,
                                DataFileContent.EQUALITY_DELETES,
                                0,
                                Record(),
                                2,
                                raw_data_file=data_file_obj,
                            )
                        ],
                    )
                ],
                0,
            ),
        ],
    )


def test_manifest_missing_equality_ids_fails(tmp_path, simple_schema, unpartitioned_spec) -> None:
    delete_arrow = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    path = builder._write_parquet(delete_arrow, "del")
    raw = _raw_delete_data_file(path, equality_ids=None, record_count=1)
    table = _table_with_raw_delete(tmp_path, simple_schema, unpartitioned_spec, raw)
    with pytest.raises(ValueError, match="missing the equality_ids"):
        table.scan().plan_files()


def test_manifest_empty_equality_ids_fails(tmp_path, simple_schema, unpartitioned_spec) -> None:
    delete_arrow = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    path = builder._write_parquet(delete_arrow, "del")
    raw = _raw_delete_data_file(path, equality_ids=[], record_count=1)
    table = _table_with_raw_delete(tmp_path, simple_schema, unpartitioned_spec, raw)
    with pytest.raises(ValueError, match="empty equality_ids"):
        table.scan().plan_files()


def test_manifest_duplicate_equality_ids_fails(tmp_path, simple_schema, unpartitioned_spec) -> None:
    delete_arrow = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    path = builder._write_parquet(delete_arrow, "del")
    raw = _raw_delete_data_file(path, equality_ids=[1, 1], record_count=1)
    table = _table_with_raw_delete(tmp_path, simple_schema, unpartitioned_spec, raw)
    with pytest.raises(ValueError, match="duplicate equality_ids"):
        table.scan().plan_files()


def test_manifest_unknown_field_id_fails_at_planning(tmp_path, simple_schema, unpartitioned_spec) -> None:
    delete_arrow = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    path = builder._write_parquet(delete_arrow, "del")
    raw = _raw_delete_data_file(path, equality_ids=[4242], record_count=1)
    table = _table_with_raw_delete(tmp_path, simple_schema, unpartitioned_spec, raw)
    with pytest.raises(ValueError, match="4242"):
        table.scan().plan_files()


def test_struct_key_hidden_from_projection(tmp_path) -> None:
    schema = Schema(
        NestedField(
            1,
            "location",
            StructType(
                NestedField(10, "lat", LongType()),
                NestedField(11, "lon", LongType()),
            ),
        ),
        NestedField(2, "payload", StringType()),
        schema_id=0,
    )
    spec = PartitionSpec(spec_id=0)
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [spec])
    data = parquet_table_with_field_ids(
        {
            "location": (
                pa.array(
                    [{"lat": 1, "lon": 2}, {"lat": 3, "lon": 4}],
                    pa.struct(
                        [
                            pa.field("lat", pa.int64(), metadata={b"PARQUET:field_id": "10"}),
                            pa.field("lon", pa.int64(), metadata={b"PARQUET:field_id": "11"}),
                        ]
                    ),
                ),
                1,
            ),
            "payload": (pa.array(["p1", "p2"]), 2),
        }
    )
    deletes = parquet_table_with_field_ids(
        {
            "location": (
                pa.array(
                    [{"lat": 3, "lon": 4}],
                    pa.struct(
                        [
                            pa.field("lat", pa.int64(), metadata={b"PARQUET:field_id": "10"}),
                            pa.field("lon", pa.int64(), metadata={b"PARQUET:field_id": "11"}),
                        ]
                    ),
                ),
                1,
            ),
        }
    )
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [10, 11])])], 0),
        ],
    )
    # The key struct is not requested at all, yet matching still has to work.
    arrow_table = table.scan(selected_fields=("payload",)).to_arrow()
    assert arrow_table.column_names == ["payload"]
    assert arrow_table.to_pylist() == [{"payload": "p1"}]
    assert table.scan(selected_fields=("payload",)).count() == 1


def test_batch_reader_propagates_error_and_releases_resources(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["p1", "p2"]), 3),
        }
    )
    bad_path = f"file://{builder.data_dir}/broken.parquet"
    write_corrupt_parquet(bad_path)
    corrupt_planned = PlannedFile(
        parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)}),
        DataFileContent.EQUALITY_DELETES,
        0,
        Record(),
        2,
        equality_ids=[1],
    )
    corrupt_planned.file_path = bad_path
    corrupt_planned.record_count = 1
    location = builder.build(
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [corrupt_planned])], 0),
        ]
    )
    table = builder.load(location)
    reader = table.scan().to_arrow_batch_reader()
    with pytest.raises(EqualityDeleteError, match=bad_path):
        for _ in reader:
            pass
    # The failed reader must not poison later scans of the same table object.
    healthy_builder = LocalV2TableBuilder(str(tmp_path), "t2", [simple_schema], 0, [unpartitioned_spec])
    healthy = _build(
        healthy_builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
        ],
    )
    assert healthy.scan().count() == 2


def test_global_unpartitioned_delete_shared_across_data_specs(tmp_path) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "part", StringType()),
        NestedField(3, "payload", StringType()),
        schema_id=0,
    )
    unpartitioned = PartitionSpec(spec_id=0)
    partitioned = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="part"), spec_id=1)
    builder = LocalV2TableBuilder(str(tmp_path), "t", [schema], 0, [unpartitioned, partitioned], default_spec_id=1)
    data_a = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "part": (pa.array(["a", "a"]), 2),
            "payload": (pa.array(["x", "y"]), 3),
        }
    )
    data_b = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 3], pa.int64()), 1),
            "part": (pa.array(["b", "b"]), 2),
            "payload": (pa.array(["x", "z"]), 3),
        }
    )
    # One unpartitioned equality delete file shared by both partition specs.
    deletes = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(
                1,
                None,
                1,
                [
                    (1, ManifestContent.DATA, [_data_file(data_a, 1, Record(part="a"), 1)]),
                    (1, ManifestContent.DATA, [_data_file(data_b, 1, Record(part="b"), 1)]),
                ],
                0,
            ),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    assert sorted(_rows(table)) == sorted([(2, "a", "y"), (3, "b", "z")])
    # The single delete file is attached to both data tasks but read once.
    tasks = list(table.scan().plan_files())
    delete_paths = {
        delete_file.file_path
        for task in tasks
        for delete_file in task.delete_files
        if delete_file.content == DataFileContent.EQUALITY_DELETES
    }
    assert len(delete_paths) == 1


def test_file_sequence_number_is_not_used_for_applicability(tmp_path, simple_schema, unpartitioned_spec) -> None:
    """A reused delete file keeps its old file sequence number but applies
    through the inherited (data) sequence number of its manifest entry."""
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["p1", "p2"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    delete_planned = PlannedFile(
        deletes,
        DataFileContent.EQUALITY_DELETES,
        0,
        Record(),
        sequence_number=3,
        equality_ids=[1],
        # The physical file was first written in sequence 1, but it is being
        # referenced again by a sequence-3 snapshot.  Applicability must use
        # the entry's (inherited) sequence number 3, never 1.
        file_sequence_number=1,
    )
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 2)])], 0),
            PlannedSnapshot(2, 1, 3, [(0, ManifestContent.DELETES, [delete_planned])], 0),
        ],
    )
    # Data seq 2 < delete entry seq 3, so the delete applies even though the
    # delete file's own file sequence number is 1.
    assert _rows(table) == [(2, "b", "p2")]


def test_conservative_metrics_still_produce_correct_result(tmp_path, simple_schema, unpartitioned_spec) -> None:
    """When file metrics are missing entirely, no manifest entry is pruned."""
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2, 3], pa.int64()), 1),
            "name": (pa.array(["a", "b", "c"]), 2),
            "payload": (pa.array(["p1", "p2", "p3"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([2], pa.int64()), 1)})
    planned_data = _data_file(data, 0, Record(), 1)
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [planned_data])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    # Filter on a column with no bounds; correctness must not rely on metrics.
    result = table.scan(row_filter=EqualTo("name", "b")).to_arrow()
    assert result.num_rows == 0
    result = table.scan(row_filter=EqualTo("name", "c")).to_arrow()
    assert result.to_pylist() == [{"id": 3, "name": "c", "payload": "p3"}]


def test_count_with_limit_and_deletes(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2, 3, 4], pa.int64()), 1),
            "name": (pa.array(["a", "b", "c", "d"]), 2),
            "payload": (pa.array(["p1", "p2", "p3", "p4"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([2, 4], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    # count() ignores the limit parameter but must still apply deletes.
    assert table.scan().count() == 2
    assert table.scan(limit=1).count() == 2
    # And the limited scan itself returns one surviving row.
    assert table.scan(limit=1).to_arrow().num_rows == 1


def test_to_pandas_applies_equality_deletes(tmp_path, simple_schema, unpartitioned_spec) -> None:
    builder = LocalV2TableBuilder(str(tmp_path), "t", [simple_schema], 0, [unpartitioned_spec])
    data = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2], pa.int64()), 1),
            "name": (pa.array(["a", "b"]), 2),
            "payload": (pa.array(["p1", "p2"]), 3),
        }
    )
    deletes = parquet_table_with_field_ids({"id": (pa.array([1], pa.int64()), 1)})
    table = _build(
        builder,
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, [_data_file(data, 0, Record(), 1)])], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, [_eq_delete_file(deletes, 0, Record(), 2, [1])])], 0),
        ],
    )
    pandas_frame = table.scan().to_pandas()
    assert pandas_frame.to_dict("records") == [{"id": 2, "name": "b", "payload": "p2"}]
