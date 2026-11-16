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
"""Local, end-to-end tests for ``Table.update_rows`` / ``Transaction.update_rows``.

Every test uses a real local SQL catalog, real Parquet data files, real Avro
manifests and real JSON metadata.  The FileIO is a ``PyArrowFileIO`` subclass
that records every ``.parquet`` file it opens or creates, so the tests can
prove that pruned data files are never touched instead of only checking the
returned rows.
"""

from __future__ import annotations

import datetime
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pyiceberg.catalog import Catalog
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import CommitFailedException, CommitStateUnknownException
from pyiceberg.expressions import ColumnRef, EqualTo, GreaterThan, IsNull
from pyiceberg.io import PY_IO_IMPL
from pyiceberg.io.pyarrow import schema_to_pyarrow
from pyiceberg.manifest import DataFileContent, ManifestContent, ManifestEntryStatus
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.table.snapshots import Operation
from pyiceberg.transforms import BucketTransform, DayTransform, IdentityTransform, TruncateTransform
from pyiceberg.typedef import Record
from pyiceberg.types import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FixedType,
    FloatType,
    IntegerType,
    ListType,
    LongType,
    MapType,
    NestedField,
    StringType,
    StructType,
    TimestampType,
    TimestamptzType,
    TimeType,
    UUIDType,
)
from utils.equality_delete_table import (
    LocalV2TableBuilder,
    PlannedFile,
    PlannedSnapshot,
    parquet_table_with_field_ids,
)
from utils.recording_io import (
    RecordingFileIO,
    created_data_files,
    opened_data_files,
    reset,
)

SCHEMA = Schema(
    NestedField(1, "id", LongType()),
    NestedField(2, "name", StringType()),
    NestedField(3, "score", IntegerType()),
    NestedField(4, "metric", DoubleType()),
)

IDENTIFIER = "default.tbl"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    reset()
    catalog = SqlCatalog(
        "test_update_rows",
        uri=f"sqlite:///{tmp_path.as_posix()}/catalog.db",
        warehouse=f"file://{tmp_path.as_posix()}/warehouse",
        **{PY_IO_IMPL: "utils.recording_io.RecordingFileIO"},
    )
    catalog.create_namespace("default")
    yield catalog
    catalog.destroy_tables()


def _data(schema: Schema, rows: List[Dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema_to_pyarrow(schema, include_field_ids=False))


def _create(
    catalog: Catalog,
    schema: Schema = SCHEMA,
    partition_spec: Optional[PartitionSpec] = None,
    identifier: str = IDENTIFIER,
    properties: Optional[Dict[str, str]] = None,
) -> Table:
    return catalog.create_table(
        identifier,
        schema=schema,
        partition_spec=partition_spec or PartitionSpec(),
        properties={"format-version": "2", **(properties or {})},
    )


def _append(tbl: Table, schema: Schema, rows: List[Dict[str, Any]]) -> None:
    tbl.append(_data(schema, rows))


def _rows(tbl: Table, **scan_kwargs: Any) -> List[Dict[str, Any]]:
    """Return every row, including duplicates, in a deterministic order."""
    rows = tbl.scan(**scan_kwargs).to_arrow().to_pylist()
    return sorted(rows, key=lambda row: tuple(str(row[name]) for name in row))


def _basic_rows() -> List[Dict[str, Any]]:
    return [
        {"id": 1, "name": "a", "score": 10, "metric": 1.5},
        {"id": 2, "name": "b", "score": 20, "metric": 2.5},
        {"id": 3, "name": "c", "score": 30, "metric": 3.5},
        {"id": 4, "name": "d", "score": 40, "metric": 4.5},
    ]


def _snapshot_ids(tbl: Table) -> List[int]:
    return [snapshot.snapshot_id for snapshot in tbl.metadata.snapshots]


def _data_file_paths(tbl: Table, snapshot_id: int | None = None) -> List[str]:
    files = tbl.inspect.data_files(snapshot_id).to_pylist()
    return sorted(file["file_path"] for file in files)


def _file_bytes(tbl: Table, path: str) -> bytes:
    with tbl.io.new_input(path).open() as f:
        return f.read()


def _local_path(path: str) -> str:
    return path.replace("file://", "")


# ---------------------------------------------------------------------------
# Validation of ColumnRef
# ---------------------------------------------------------------------------


def test_column_ref_requires_non_empty_string_name() -> None:
    assert ColumnRef("id").name == "id"
    assert ColumnRef("id") == ColumnRef("id")
    assert ColumnRef("id") != ColumnRef("other")
    for bad in ["", None, 5, b"id"]:  # type: ignore[list-item]
        with pytest.raises(ValueError):
            ColumnRef(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Constants, duplicates, swaps
# ---------------------------------------------------------------------------


def test_constant_update_returns_changed_rows_and_updates_every_read_path(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())
    old_snapshot = tbl.current_snapshot().snapshot_id
    before = _rows(tbl)

    assert tbl.update_rows({"score": 99}, row_filter=EqualTo("id", 2)) == 1

    expected = [row for row in before if row["id"] != 2] + [{"id": 2, "name": "b", "score": 99, "metric": 2.5}]
    expected = sorted(expected, key=lambda row: tuple(str(row[name]) for name in row))

    assert _rows(tbl) == expected
    assert _rows(tbl, row_filter=EqualTo("score", 99)) == [{"id": 2, "name": "b", "score": 99, "metric": 2.5}]
    assert _rows(tbl, selected_fields=("id", "score")) == sorted(
        [{"id": row["id"], "score": row["score"]} for row in expected],
        key=lambda row: tuple(str(row[name]) for name in row),
    )
    assert tbl.scan().count() == 4
    batch_rows = [row for batch in tbl.scan().to_arrow_batch_reader() for row in batch.to_pylist()]
    assert sorted(batch_rows, key=lambda row: str(row["id"])) == sorted(expected, key=lambda row: str(row["id"]))

    # Historical snapshots keep the old values.
    assert _rows(tbl, snapshot_id=old_snapshot) == before

    # Exactly one new snapshot is published, and it is an overwrite.
    assert len(tbl.metadata.snapshots) == 2
    new_snapshot = tbl.current_snapshot()
    assert new_snapshot.snapshot_id != old_snapshot
    assert new_snapshot.summary.operation == Operation.OVERWRITE
    assert new_snapshot.parent_snapshot_id == old_snapshot


def test_duplicate_rows_each_count_once(catalog: Catalog) -> None:
    tbl = _create(catalog)
    rows = [
        {"id": 1, "name": "dup", "score": 1, "metric": 1.0},
        {"id": 1, "name": "dup", "score": 1, "metric": 1.0},
        {"id": 1, "name": "dup", "score": 1, "metric": 1.0},
        {"id": 2, "name": "other", "score": 2, "metric": 2.0},
    ]
    _append(tbl, SCHEMA, rows)

    assert tbl.update_rows({"name": "changed"}, row_filter=EqualTo("id", 1)) == 3
    assert _rows(tbl) == [
        {"id": 1, "name": "changed", "score": 1, "metric": 1.0},
        {"id": 1, "name": "changed", "score": 1, "metric": 1.0},
        {"id": 1, "name": "changed", "score": 1, "metric": 1.0},
        {"id": 2, "name": "other", "score": 2, "metric": 2.0},
    ]
    assert tbl.scan().count() == 4


def test_duplicates_of_untouched_rows_keep_their_multiplicity(catalog: Catalog) -> None:
    tbl = _create(catalog)
    rows = [
        {"id": 1, "name": "dup", "score": 1, "metric": 1.0},
        {"id": 1, "name": "dup", "score": 1, "metric": 1.0},
        {"id": 2, "name": "target", "score": 2, "metric": 2.0},
    ]
    _append(tbl, SCHEMA, rows)

    assert tbl.update_rows({"score": 22}, row_filter=EqualTo("id", 2)) == 1
    assert _rows(tbl) == [
        {"id": 1, "name": "dup", "score": 1, "metric": 1.0},
        {"id": 1, "name": "dup", "score": 1, "metric": 1.0},
        {"id": 2, "name": "target", "score": 22, "metric": 2.0},
    ]


def test_partial_match_in_one_file_preserves_other_rows(catalog: Catalog) -> None:
    tbl = _create(catalog)
    rows = [
        {"id": 1, "name": "keep", "score": 1, "metric": 1.0},
        {"id": 2, "name": "target", "score": 2, "metric": 2.0},
        {"id": 3, "name": "keep", "score": 3, "metric": 3.0},
        {"id": 2, "name": "target", "score": 2, "metric": 2.0},
        {"id": 4, "name": "keep", "score": 4, "metric": 4.0},
    ]
    _append(tbl, SCHEMA, rows)
    assert len(_data_file_paths(tbl)) == 1

    assert tbl.update_rows({"name": "changed"}, row_filter=EqualTo("id", 2)) == 2
    assert _rows(tbl) == sorted(
        [
            {"id": 1, "name": "keep", "score": 1, "metric": 1.0},
            {"id": 2, "name": "changed", "score": 2, "metric": 2.0},
            {"id": 2, "name": "changed", "score": 2, "metric": 2.0},
            {"id": 3, "name": "keep", "score": 3, "metric": 3.0},
            {"id": 4, "name": "keep", "score": 4, "metric": 4.0},
        ],
        key=lambda row: tuple(str(row[name]) for name in row),
    )
    assert tbl.scan().count() == 5
    assert tbl.scan(row_filter=EqualTo("name", "keep")).count() == 3


def test_column_ref_swap_reads_only_pre_update_values(catalog: Catalog) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "left", StringType()),
        NestedField(3, "right", StringType()),
    )
    tbl = _create(catalog, schema=schema)
    _append(
        tbl,
        schema,
        [
            {"id": 1, "left": "L1", "right": "R1"},
            {"id": 2, "left": "L2", "right": "R2"},
            {"id": 3, "left": "L3", "right": "R3"},
        ],
    )

    assert tbl.update_rows({"left": ColumnRef("right"), "right": ColumnRef("left")}, row_filter=EqualTo("id", 1)) == 1
    assert _rows(tbl) == [
        {"id": 1, "left": "R1", "right": "L1"},
        {"id": 2, "left": "L2", "right": "R2"},
        {"id": 3, "left": "L3", "right": "R3"},
    ]


def test_column_ref_whole_table_swap(catalog: Catalog) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "left", StringType()),
        NestedField(3, "right", StringType()),
    )
    tbl = _create(catalog, schema=schema)
    _append(tbl, schema, [{"id": 1, "left": "L", "right": "R"}, {"id": 2, "left": "X", "right": "Y"}])

    assert tbl.update_rows({"left": ColumnRef("right"), "right": ColumnRef("left")}) == 2
    assert _rows(tbl) == [{"id": 1, "left": "R", "right": "L"}, {"id": 2, "left": "Y", "right": "X"}]


def test_column_ref_three_way_rotation(catalog: Catalog) -> None:
    schema = Schema(
        NestedField(1, "a", LongType()),
        NestedField(2, "b", LongType()),
        NestedField(3, "c", LongType()),
    )
    tbl = _create(catalog, schema=schema)
    _append(tbl, schema, [{"a": 1, "b": 2, "c": 3}])

    assert tbl.update_rows({"a": ColumnRef("b"), "b": ColumnRef("c"), "c": ColumnRef("a")}) == 1
    assert tbl.scan().to_arrow().to_pylist() == [{"a": 2, "b": 3, "c": 1}]


def test_column_ref_allows_only_lossless_promotions(catalog: Catalog) -> None:
    schema = Schema(
        NestedField(1, "i", IntegerType()),
        NestedField(2, "l", LongType()),
        NestedField(3, "f", FloatType()),
        NestedField(4, "d", DoubleType()),
        NestedField(5, "s", StringType()),
        NestedField(6, "b", BinaryType()),
    )
    tbl = _create(catalog, schema=schema)
    _append(
        tbl,
        schema,
        [{"i": 7, "l": 8, "f": 1.5, "d": 2.5, "s": "text", "b": b"bytes"}],
    )

    # int -> long, float -> double are the Iceberg promotions.
    assert tbl.update_rows({"l": ColumnRef("i"), "d": ColumnRef("f")}) == 1
    assert tbl.scan().to_arrow().to_pylist() == [{"i": 7, "l": 7, "f": 1.5, "d": 1.5, "s": "text", "b": b"bytes"}]

    for assignments in [
        {"i": ColumnRef("l")},  # long -> int
        {"f": ColumnRef("d")},  # double -> float
        {"b": ColumnRef("s")},  # string -> binary is not a write promotion
        {"s": ColumnRef("b")},
    ]:
        with pytest.raises(ValueError, match="lossless promotion"):
            tbl.update_rows(assignments)


def test_string_constants_are_never_interpreted_as_column_names(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())

    assert tbl.update_rows({"name": "score"}, row_filter=EqualTo("id", 1)) == 1
    assert _rows(tbl)[0]["name"] == "score"

    with pytest.raises(ValueError, match="expected an int"):
        tbl.update_rows({"score": "name"})


def test_decimal_validation_is_exact() -> None:
    from decimal import Decimal

    from pyiceberg.table.update_rows import UpdateRowsError, _validate_assignments

    schema = Schema(
        NestedField(1, "dec", DecimalType(10, 2)),
        NestedField(2, "narrow", DecimalType(4, 2)),
    )

    assert _validate_assignments(schema, {"dec": Decimal("1.23")}, True)
    # Narrower precision / same scale is a lossless ColumnRef promotion.
    assert _validate_assignments(schema, {"dec": ColumnRef("narrow")}, True)

    assert _validate_assignments(schema, {"dec": Decimal("1.2")}, True)  # exact at scale 2 (1.20)

    for bad in [Decimal("1.234"), Decimal("1.2345"), 1.23, "1.23"]:
        with pytest.raises(UpdateRowsError):
            _validate_assignments(schema, {"dec": bad}, True)

    with pytest.raises(UpdateRowsError, match="lossless promotion"):
        _validate_assignments(schema, {"narrow": ColumnRef("dec")}, True)


def test_failed_table_level_update_does_not_abort_later_calls(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())

    with pytest.raises(ValueError):
        tbl.update_rows({"score": "not-an-int"})
    assert tbl.update_rows({"score": 7}, row_filter=EqualTo("id", 1)) == 1
    assert {row["id"]: row for row in tbl.scan().to_arrow().to_pylist()}[1]["score"] == 7
    assert len(tbl.metadata.snapshots) == 2


def test_transaction_noop_update_keeps_prior_staged_changes(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())

    with tbl.transaction() as tx:
        tx.append(_data(SCHEMA, [{"id": 5, "name": "e", "score": 50, "metric": 5.5}]))
        assert tx.update_rows({"name": "e"}, row_filter=EqualTo("id", 5)) == 0
        assert len(tx.table_metadata.snapshots) == 2

    assert len(tbl.metadata.snapshots) == 2
    assert tbl.scan().count() == 5


def test_renamed_column_is_updated_by_its_new_name(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())
    tbl.update_schema().rename_column("name", "label").commit()

    with pytest.raises(ValueError, match="Could not find field"):
        tbl.update_rows({"name": "x"})

    assert tbl.update_rows({"label": "renamed"}, row_filter=EqualTo("id", 1)) == 1
    assert {row["id"]: row for row in tbl.scan().to_arrow().to_pylist()}[1]["label"] == "renamed"


# ---------------------------------------------------------------------------
# Null / NaN / signed zero equality
# ---------------------------------------------------------------------------


def test_null_nan_and_signed_zero_do_not_count_as_changes(catalog: Catalog) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "metric", DoubleType()),
        NestedField(3, "name", StringType()),
    )
    tbl = _create(catalog, schema=schema)
    _append(
        tbl,
        schema,
        [
            {"id": 1, "metric": float("nan"), "name": None},
            {"id": 2, "metric": None, "name": "b"},
            {"id": 3, "metric": 0.0, "name": "c"},
            {"id": 4, "metric": -0.0, "name": "d"},
        ],
    )
    original_snapshot = tbl.current_snapshot().snapshot_id

    assert tbl.update_rows({"metric": float("nan")}, row_filter=EqualTo("id", 1)) == 0
    assert tbl.update_rows({"metric": None}, row_filter=EqualTo("id", 2)) == 0
    assert tbl.update_rows({"metric": 0.0}, row_filter=EqualTo("id", 3)) == 0
    assert tbl.update_rows({"metric": -0.0}, row_filter=EqualTo("id", 4)) == 0
    assert tbl.update_rows({"name": None}, row_filter=IsNull("name")) == 0

    assert tbl.current_snapshot().snapshot_id == original_snapshot
    assert len(tbl.metadata.snapshots) == 1


def test_nan_and_null_are_still_changed_when_the_value_differs(catalog: Catalog) -> None:
    schema = Schema(NestedField(1, "id", LongType()), NestedField(2, "metric", DoubleType()))
    tbl = _create(catalog, schema=schema)
    _append(tbl, schema, [{"id": 1, "metric": float("nan")}, {"id": 2, "metric": None}])

    assert tbl.update_rows({"metric": 1.0}) == 2
    result = tbl.scan().to_arrow().to_pylist()
    assert [row["metric"] for row in sorted(result, key=lambda row: row["id"])] == [1.0, 1.0]


# ---------------------------------------------------------------------------
# Zero-row cases
# ---------------------------------------------------------------------------


def test_empty_table_no_match_and_noop_create_no_snapshot(catalog: Catalog) -> None:
    tbl = _create(catalog)
    assert tbl.update_rows({"score": 1}) == 0
    assert len(tbl.metadata.snapshots) == 0

    _append(tbl, SCHEMA, _basic_rows())
    assert len(tbl.metadata.snapshots) == 1

    assert tbl.update_rows({"score": 1}, row_filter=EqualTo("id", 999)) == 0
    assert tbl.update_rows({"score": ColumnRef("score")}) == 0
    assert tbl.update_rows({"name": "a", "score": 10}, row_filter=EqualTo("id", 1)) == 0
    assert tbl.update_rows({"metric": 99.0}, row_filter="score > 1000") == 0

    assert len(tbl.metadata.snapshots) == 1
    assert _rows(tbl) == sorted(_basic_rows(), key=lambda row: tuple(str(row[name]) for name in row))


def test_validation_errors_are_reported_even_without_matching_rows(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())

    with pytest.raises(ValueError, match="expected an int"):
        tbl.update_rows({"score": "not-an-int"}, row_filter=EqualTo("id", 999))
    with pytest.raises(ValueError, match="Could not find field"):
        tbl.update_rows({"missing": 1}, row_filter=EqualTo("id", 999))
    with pytest.raises(ValueError, match="outside the range"):
        tbl.update_rows({"score": 2**40}, row_filter=EqualTo("id", 999))
    with pytest.raises(ValueError, match="assignments must not be empty"):
        tbl.update_rows({}, row_filter=EqualTo("id", 999))

    assert len(tbl.metadata.snapshots) == 1


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


@pytest.fixture
def nested_table(catalog: Catalog) -> Tuple[Table, Schema]:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "name", StringType()),
        NestedField(
            3,
            "info",
            StructType(NestedField(10, "x", LongType()), NestedField(11, "y", StringType())),
        ),
        NestedField(4, "tags", ListType(20, StringType(), element_required=True)),
        NestedField(5, "attrs", MapType(30, StringType(), 31, StringType(), value_required=True)),
    )
    tbl = _create(catalog, schema=schema)
    _append(
        tbl,
        schema,
        [{"id": 1, "name": "a", "info": {"x": 1, "y": "p"}, "tags": ["t"], "attrs": [("k", "v")]}],
    )
    return tbl, schema


@pytest.mark.parametrize("assignments", [{"info.x": 1}, {"info": 1}, {"tags": 1}, {"attrs": 1}])
def test_nested_and_collection_targets_are_rejected(nested_table: Tuple[Table, Schema], assignments: Dict[str, Any]) -> None:
    tbl, _ = nested_table
    with pytest.raises(ValueError):
        tbl.update_rows(assignments)
    assert len(tbl.metadata.snapshots) == 1


def test_top_level_update_on_table_with_nested_columns(nested_table: Tuple[Table, Schema]) -> None:
    tbl, _ = nested_table
    assert tbl.update_rows({"name": "z"}, row_filter=EqualTo("info.x", 1)) == 1
    assert tbl.scan().to_arrow().to_pylist() == [
        {"id": 1, "name": "z", "info": {"x": 1, "y": "p"}, "tags": ["t"], "attrs": [("k", "v")]}
    ]


def test_invalid_scalars_are_rejected_without_truncation(catalog: Catalog) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "score", IntegerType()),
        NestedField(3, "ratio", FloatType()),
        NestedField(4, "flag", BooleanType()),
        NestedField(5, "when", DateType()),
        NestedField(6, "uuid_col", UUIDType()),
        NestedField(7, "fixed_col", FixedType(4)),
        NestedField(8, "time_col", TimeType()),
        NestedField(9, "ts", TimestampType()),
        NestedField(10, "tstz", TimestamptzType()),
    )
    tbl = _create(catalog, schema=schema)

    bad_values: List[Tuple[str, Any]] = [
        ("score", 1.5),
        ("score", True),
        ("score", 2**40),
        ("ratio", 0.1),  # not exactly representable as float32
        ("ratio", 1e39),  # overflows float32
        ("flag", 1),
        ("when", "2020-01-01"),
        ("when", datetime.datetime(2020, 1, 1)),
        ("uuid_col", "not-a-uuid"),
        ("fixed_col", b"abc"),
        ("time_col", datetime.time(1, 2, 3, tzinfo=datetime.UTC)),
        ("ts", datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)),
        ("tstz", datetime.datetime(2020, 1, 1)),
        ("score", None) if False else ("ratio", object()),
    ]
    for field, value in bad_values:
        with pytest.raises(ValueError):
            tbl.update_rows({field: value})
    assert len(tbl.metadata.snapshots) == 0


def test_assigning_null_to_required_field_is_rejected(catalog: Catalog) -> None:
    schema = Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "name", StringType(), required=True),
        NestedField(3, "opt", StringType()),
    )
    tbl = _create(catalog, schema=schema)
    arrow_schema = schema_to_pyarrow(schema, include_field_ids=False)
    tbl.append(
        pa.Table.from_pylist(
            [{"id": 1, "name": "a", "opt": None}, {"id": 2, "name": "b", "opt": "x"}],
            schema=arrow_schema,
        )
    )

    with pytest.raises(ValueError, match="required"):
        tbl.update_rows({"name": None})
    # A ColumnRef whose source is null for a matched row is only detectable by
    # reading that row, and must still fail the whole operation.
    with pytest.raises(ValueError, match="required"):
        tbl.update_rows({"name": ColumnRef("opt")}, row_filter="id = 1")

    assert len(tbl.metadata.snapshots) == 1
    assert tbl.scan().to_arrow().to_pylist() == [
        {"id": 1, "name": "a", "opt": None},
        {"id": 2, "name": "b", "opt": "x"},
    ]


def test_column_ref_to_unknown_or_nested_source_is_rejected(nested_table: Tuple[Table, Schema]) -> None:
    tbl, _ = nested_table
    for assignments in [{"name": ColumnRef("missing")}, {"name": ColumnRef("info.x")}, {"name": ColumnRef("info")}]:
        with pytest.raises(ValueError):
            tbl.update_rows(assignments)


def test_assignments_resolving_to_the_same_field_are_rejected(catalog: Catalog) -> None:
    schema = Schema(NestedField(1, "Name", StringType()), NestedField(2, "score", IntegerType()))
    tbl = _create(catalog, schema=schema)

    with pytest.raises(ValueError, match="both resolve to field"):
        tbl.update_rows({"Name": "x", "name": "y"}, case_sensitive=False)
    # Exact duplicates with case sensitivity on resolve to different fields and fail as unknown.
    with pytest.raises(ValueError, match="Could not find field"):
        tbl.update_rows({"name": "y"})


def test_case_sensitive_and_ambiguous_field_resolution(catalog: Catalog) -> None:
    schema = Schema(NestedField(1, "Name", StringType()), NestedField(2, "score", IntegerType()))
    tbl = _create(catalog, schema=schema)
    _append(tbl, schema, [{"Name": "a", "score": 1}])

    with pytest.raises(ValueError, match="Could not find field"):
        tbl.update_rows({"NAME": "b"})

    assert tbl.update_rows({"NAME": "b"}, case_sensitive=False) == 1
    assert tbl.scan().to_arrow().to_pylist() == [{"Name": "b", "score": 1}]


def test_ambiguous_case_folded_schema_is_rejected(catalog: Catalog) -> None:
    schema = Schema(NestedField(1, "Value", StringType()), NestedField(2, "VALUE", StringType()))
    tbl = _create(catalog, schema=schema)
    _append(tbl, schema, [{"Value": "a", "VALUE": "b"}])

    with pytest.raises(ValueError, match="ambiguous"):
        tbl.update_rows({"value": "c"}, case_sensitive=False)


def test_format_version_one_is_rejected(catalog: Catalog) -> None:
    tbl = _create(catalog, properties={"format-version": "1"})
    with pytest.raises(ValueError, match="format-version=2"):
        tbl.update_rows({"score": 1})


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def test_identity_partition_source_change_moves_rows_to_new_partition(catalog: Catalog) -> None:
    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="name"))
    tbl = _create(catalog, partition_spec=spec)
    _append(tbl, SCHEMA, _basic_rows())

    old_files = _data_file_paths(tbl)
    assert len(old_files) == 4

    assert tbl.update_rows({"name": "moved"}, row_filter=EqualTo("id", 2)) == 1

    rows = tbl.scan().to_arrow().to_pylist()
    assert sorted(rows, key=lambda row: row["id"]) == [
        {"id": 1, "name": "a", "score": 10, "metric": 1.5},
        {"id": 2, "name": "moved", "score": 20, "metric": 2.5},
        {"id": 3, "name": "c", "score": 30, "metric": 3.5},
        {"id": 4, "name": "d", "score": 40, "metric": 4.5},
    ]
    assert tbl.scan(row_filter=EqualTo("name", "b")).count() == 0
    assert tbl.scan(row_filter=EqualTo("name", "moved")).count() == 1

    new_files = _data_file_paths(tbl)
    assert len(new_files) == 4
    assert any("/name=moved/" in path for path in new_files)
    assert not any(path.endswith(os.path.basename([p for p in old_files if "/name=b/" in p][0])) for path in new_files)
    # The other three files were not rewritten.
    assert len(set(old_files) & set(new_files)) == 3

    # Every new file's partition record matches its directory and the current spec.
    for data_file in tbl.inspect.data_files().to_pylist():
        assert data_file["spec_id"] == tbl.metadata.default_spec_id
        partition_name = data_file["file_path"].split("/name=")[1].split("/")[0]
        assert data_file["record_count"] == 1
        assert partition_name in {"a", "c", "d", "moved"}


@pytest.mark.parametrize(
    "transform,expected_field",
    [(BucketTransform(4), "id_bucket"), (TruncateTransform(2), "name_trunc")],
)
def test_writable_non_identity_partition_transforms(catalog: Catalog, transform: Any, expected_field: str) -> None:
    pytest.importorskip("pyiceberg_core")
    source_id = 1 if expected_field == "id_bucket" else 2
    spec = PartitionSpec(PartitionField(source_id=source_id, field_id=1000, transform=transform, name=expected_field))
    tbl = _create(catalog, partition_spec=spec)
    _append(tbl, SCHEMA, _basic_rows())

    assert tbl.update_rows({"name": "zzz"}, row_filter=EqualTo("id", 2)) == 1
    assert tbl.scan(row_filter=EqualTo("id", 2)).to_arrow().to_pylist()[0]["name"] == "zzz"
    assert tbl.scan().count() == 4
    assert len(tbl.metadata.snapshots) == 2
    for data_file in tbl.inspect.data_files().to_pylist():
        assert data_file["spec_id"] == tbl.metadata.default_spec_id
    total = sum(file["record_count"] for file in tbl.inspect.data_files().to_pylist())
    assert total == 4


def test_day_transform_partition(catalog: Catalog) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "name", StringType()),
        NestedField(3, "ts", TimestampType()),
    )
    spec = PartitionSpec(PartitionField(source_id=3, field_id=1000, transform=DayTransform(), name="ts_day"))
    tbl = _create(catalog, schema=schema, partition_spec=spec)
    _append(
        tbl,
        schema,
        [
            {"id": 1, "name": "a", "ts": datetime.datetime(2024, 1, 1)},
            {"id": 2, "name": "b", "ts": datetime.datetime(2024, 1, 2)},
        ],
    )

    assert tbl.update_rows({"name": "moved"}, row_filter=EqualTo("id", 2)) == 1
    assert tbl.scan().count() == 2
    assert any("/ts_day=2024-01-02/" in path for path in _data_file_paths(tbl))
    assert [row["name"] for row in tbl.scan(row_filter=EqualTo("id", 2)).to_arrow().to_pylist()] == ["moved"]


def test_old_spec_files_and_deletes_are_respected(catalog: Catalog, tmp_path: Path) -> None:
    schema = Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "name", StringType()),
        NestedField(3, "payload", StringType()),
        schema_id=0,
    )
    unpartitioned = PartitionSpec(spec_id=0)
    by_name = PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="name"),
        spec_id=1,
    )
    builder = LocalV2TableBuilder(str(tmp_path), "old_spec", [schema], 0, [unpartitioned, by_name], default_spec_id=1)

    first = parquet_table_with_field_ids(
        {
            "id": (pa.array([1, 2, 3], pa.int64()), 1),
            "name": (pa.array(["a", "b", "c"]), 2),
            "payload": (pa.array(["p1", "p2", "p3"]), 3),
        }
    )
    second = parquet_table_with_field_ids(
        {
            "id": (pa.array([4, 5], pa.int64()), 1),
            "name": (pa.array(["d", "e"]), 2),
            "payload": (pa.array(["p4", "p5"]), 3),
        }
    )
    first_planned = PlannedFile(first, DataFileContent.DATA, 0, Record(), 1)
    first_planned.file_path = builder._write_parquet(first, "data")
    second_planned = PlannedFile(second, DataFileContent.DATA, 0, Record(), 1)

    # Positional delete removes the physical row 1 of the first file (id=2).
    positional = parquet_table_with_field_ids(
        {
            "file_path": (pa.array([first_planned.file_path]), 2147483546),
            "pos": (pa.array([1], pa.int32()), 2147483545),
        }
    )
    # Equality delete removes id=4 from the second file.
    equality = parquet_table_with_field_ids({"id": (pa.array([4], pa.int64()), 1)})

    location = builder.build(
        [
            PlannedSnapshot(
                1,
                None,
                1,
                [
                    (0, ManifestContent.DATA, [first_planned, second_planned]),
                ],
                0,
            ),
            PlannedSnapshot(
                2,
                1,
                2,
                [
                    (0, ManifestContent.DELETES, [PlannedFile(positional, DataFileContent.POSITION_DELETES, 0, Record(), 2)]),
                    (
                        0,
                        ManifestContent.DELETES,
                        [PlannedFile(equality, DataFileContent.EQUALITY_DELETES, 0, Record(), 2, equality_ids=[1])],
                    ),
                ],
                0,
            ),
        ]
    )
    tbl = catalog.register_table(IDENTIFIER, location)

    assert tbl.spec().spec_id == 1
    assert sorted(row["id"] for row in tbl.scan().to_arrow().to_pylist()) == [1, 3, 5]

    unchanged_path = second_planned.file_path
    old_spec_file_path = first_planned.file_path

    assert tbl.update_rows({"payload": "Z"}, row_filter=GreaterThan("id", 0)) == 3

    rows = tbl.scan().to_arrow().to_pylist()
    assert sorted(row["id"] for row in rows) == [1, 3, 5]
    assert all(row["payload"] == "Z" for row in rows)

    # Deleted rows are not revived and the old snapshot still shows them.
    assert sorted(row["id"] for row in tbl.scan(snapshot_id=1).to_arrow().to_pylist()) == [1, 2, 3, 4, 5]
    assert sorted(row["id"] for row in tbl.scan(snapshot_id=2).to_arrow().to_pylist()) == [1, 3, 5]

    # New files use the current (spec 1) partition spec, and the files that were
    # rewritten no longer appear.
    files = tbl.inspect.data_files().to_pylist()
    assert files
    assert all(file["spec_id"] == 1 for file in files)
    assert {file["file_path"] for file in files}.isdisjoint({old_spec_file_path, unchanged_path})
    assert len(tbl.metadata.snapshots) == 3


# ---------------------------------------------------------------------------
# Physical file behaviour
# ---------------------------------------------------------------------------


def test_pruned_data_files_are_not_opened_or_rewritten(catalog: Catalog) -> None:
    tbl = _create(catalog)
    for chunk in [[1, 2], [3, 4], [5, 6]]:
        _append(
            tbl,
            SCHEMA,
            [{"id": value, "name": f"n{value}", "score": value * 10, "metric": float(value)} for value in chunk],
        )

    original_files = _data_file_paths(tbl)
    assert len(original_files) == 3
    target_files = [
        path
        for path in original_files
        if pq.read_metadata(_local_path(path)).num_rows == 2
        and any(row["id"] == 3 for row in pq.read_table(_local_path(path)).to_pylist())
    ]
    assert len(target_files) == 1
    target = target_files[0]
    untouched = [path for path in original_files if path != target]
    digests_before = {path: hashlib.sha256(_file_bytes(tbl, path)).hexdigest() for path in original_files}

    reset()
    assert tbl.update_rows({"name": "changed"}, row_filter=EqualTo("id", 3), snapshot_properties={"origin": "test"}) == 1
    opened = opened_data_files()
    created = created_data_files()

    assert [_local_path(target)] == opened
    assert len(created) == 1
    assert tbl.current_snapshot().summary.additional_properties["origin"] == "test"

    after = _data_file_paths(tbl)
    assert set(untouched).issubset(after)
    assert target not in after
    # The untouched files are byte-identical, and even the rewritten original is
    # still there (history is not cleaned up).
    for path in original_files:
        assert hashlib.sha256(_file_bytes(tbl, path)).hexdigest() == digests_before[path]
    assert tbl.scan().count() == 6


def test_new_data_files_match_their_partition_and_statistics(catalog: Catalog) -> None:
    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="name"))
    tbl = _create(catalog, partition_spec=spec)
    _append(tbl, SCHEMA, _basic_rows())

    original = {file["file_path"]: _file_bytes(tbl, file["file_path"]) for file in tbl.inspect.data_files().to_pylist()}
    assert tbl.update_rows({"score": 1234}, row_filter=EqualTo("id", 3)) == 1

    new_files = [file for file in tbl.inspect.data_files().to_pylist() if file["file_path"] not in original]
    assert len(new_files) == 1
    new_file = new_files[0]
    parquet_metadata = pq.read_metadata(_local_path(new_file["file_path"]))
    assert new_file["record_count"] == parquet_metadata.num_rows == 1
    assert new_file["file_size_in_bytes"] == os.path.getsize(_local_path(new_file["file_path"]))
    assert new_file["spec_id"] == tbl.metadata.default_spec_id == 0
    assert new_file["file_format"] == "PARQUET"
    assert "/name=c/" in new_file["file_path"]
    partitions = tbl.inspect.partitions().to_pylist()
    by_name = {partition["partition"]["name"]: partition for partition in partitions}
    assert (by_name["c"]["record_count"], by_name["c"]["file_count"], by_name["c"]["spec_id"]) == (1, 1, 0)
    assert sum(partition["record_count"] for partition in partitions) == 4
    assert new_file["readable_metrics"]["score"]["value_count"] == 1
    assert new_file["readable_metrics"]["score"]["lower_bound"] == new_file["readable_metrics"]["score"]["upper_bound"] == 1234
    assert new_file["readable_metrics"]["score"]["null_value_count"] == 0
    # Original files are untouched on disk.
    for path, content in original.items():
        assert _file_bytes(tbl, path) == content


def test_snapshot_contains_only_the_expected_manifest_entries(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())
    old_files = set(_data_file_paths(tbl))

    assert tbl.update_rows({"name": "x"}, row_filter="id <= 2") == 2

    snapshot = tbl.current_snapshot()
    assert snapshot.summary.operation == Operation.OVERWRITE
    added = []
    deleted = []
    for manifest in snapshot.manifests(tbl.io):
        for entry in manifest.fetch_manifest_entry(tbl.io, discard_deleted=False):
            if entry.status == ManifestEntryStatus.ADDED:
                added.append(entry.data_file.file_path)
            elif entry.status == ManifestEntryStatus.DELETED:
                deleted.append(entry.data_file.file_path)
    assert set(deleted) == old_files
    assert len(added) == len(set(added))
    assert set(added).isdisjoint(old_files)
    assert len(tbl.metadata.snapshots) == 2
    # The snapshot also carries exactly one manifest list and no intermediate
    # snapshot exists between the original append and this result.
    assert [snapshot.summary.operation for snapshot in tbl.metadata.snapshots] == [Operation.APPEND, Operation.OVERWRITE]


def test_old_data_files_are_not_modified_and_inputs_are_not_mutated(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())
    before = {path: hashlib.sha256(_file_bytes(tbl, path)).hexdigest() for path in _data_file_paths(tbl)}
    assignments = {"name": ColumnRef("name"), "score": 5}
    snapshot = assignments.copy()

    tbl.update_rows(assignments)
    # The caller's mapping is untouched.
    assert assignments == snapshot
    assert isinstance(assignments["name"], ColumnRef)

    # Every old file still exists with its original content.
    for path, digest in before.items():
        assert hashlib.sha256(_file_bytes(tbl, path)).hexdigest() == digest


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


def test_transaction_sees_prior_append_and_later_operations_use_the_update(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())
    base_snapshot = tbl.current_snapshot().snapshot_id

    with tbl.transaction() as tx:
        tx.append(_data(SCHEMA, [{"id": 5, "name": "e", "score": 50, "metric": 5.5}]))
        # The staged append is visible to this transaction.
        assert tx.update_rows({"name": "new"}, row_filter=EqualTo("id", 5)) == 1
        # ...and the update result is visible to later operations.
        tx.append(_data(SCHEMA, [{"id": 6, "name": "f", "score": 60, "metric": 6.5}]))
        assert tx.update_rows({"name": "later"}, row_filter=EqualTo("id", 5)) == 1
        assert tx._scan().to_arrow().to_pylist()

    rows = {row["id"]: row for row in tbl.scan().to_arrow().to_pylist()}
    assert rows[5]["name"] == "later"
    assert rows[6]["name"] == "f"
    # The initial append plus, inside the transaction: append + update + append + update.
    assert len(tbl.metadata.snapshots) == 5
    assert tbl.snapshot_by_id(base_snapshot).summary.operation == Operation.APPEND


def test_transaction_update_schema_then_update_rows(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())

    with tbl.transaction() as tx:
        tx.update_schema().add_column("extra", StringType()).commit()
        assert tx.update_rows({"extra": "hello"}, row_filter=EqualTo("id", 1)) == 1

    assert tbl.schema().find_field("extra").field_id is not None
    rows = {row["id"]: row for row in tbl.scan().to_arrow().to_pylist()}
    assert rows[1]["extra"] == "hello"
    assert rows[2]["extra"] is None


def test_transaction_update_spec_then_update_rows(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())

    with tbl.transaction() as tx:
        tx.update_spec().add_identity("name").commit()
        assert tx.update_rows({"name": "zz"}, row_filter=EqualTo("id", 2)) == 1

    assert tbl.spec().spec_id == 1
    files = tbl.inspect.data_files().to_pylist()
    assert all(file["spec_id"] == 1 for file in files)
    assert tbl.scan(row_filter=EqualTo("name", "zz")).count() == 1


def test_transaction_update_rows_then_schema_change(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())

    with tbl.transaction() as tx:
        assert tx.update_rows({"score": 999}, row_filter=EqualTo("id", 3)) == 1
        tx.update_schema().add_column("extra", StringType()).commit()
        assert tx.update_rows({"extra": "after"}, row_filter=EqualTo("id", 3)) == 1

    rows = {row["id"]: row for row in tbl.scan().to_arrow().to_pylist()}
    assert rows[3]["score"] == 999
    assert rows[3]["extra"] == "after"
    # The initial append plus two update snapshots; the schema change itself is
    # metadata-only.
    assert len(tbl.metadata.snapshots) == 3


def test_other_table_instance_sees_old_version_until_commit(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())
    other = catalog.load_table(IDENTIFIER)
    old_rows = _rows(other)
    old_snapshot = other.current_snapshot().snapshot_id

    with tbl.transaction() as tx:
        tx.update_rows({"name": "pending"}, row_filter=EqualTo("id", 1))
        assert _rows(other) == old_rows
        assert other.current_snapshot().snapshot_id == old_snapshot

    # A previously loaded instance keeps its snapshot until it is refreshed.
    assert other.current_snapshot().snapshot_id == old_snapshot
    assert _rows(other) == old_rows

    other.refresh()
    assert other.current_snapshot().snapshot_id != old_snapshot
    assert {row["id"]: row for row in _rows(other)}[1]["name"] == "pending"


def test_transaction_update_is_visible_after_commit(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())

    tx = tbl.transaction()
    assert tx.update_rows({"score": 1}, row_filter=EqualTo("id", 4)) == 1
    assert tx.commit_transaction() is tbl

    assert {row["id"]: row for row in tbl.scan().to_arrow().to_pylist()}[4]["score"] == 1
    assert catalog.load_table(IDENTIFIER).scan().count() == 4


def test_concurrent_updates_from_the_same_base_conflict(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())

    first = catalog.load_table(IDENTIFIER)
    second = catalog.load_table(IDENTIFIER)
    assert first.current_snapshot().snapshot_id == second.current_snapshot().snapshot_id

    assert first.update_rows({"name": "from-first"}, row_filter=EqualTo("id", 1)) == 1
    with pytest.raises(CommitFailedException):
        second.update_rows({"name": "from-second"}, row_filter=EqualTo("id", 1))

    fresh = catalog.load_table(IDENTIFIER)
    rows = {row["id"]: row for row in fresh.scan().to_arrow().to_pylist()}
    assert rows[1]["name"] == "from-first"
    assert len(fresh.metadata.snapshots) == 2


def test_failed_update_aborts_the_transaction(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())
    tx = tbl.transaction()
    tx.append(_data(SCHEMA, [{"id": 5, "name": "e", "score": 50, "metric": 5.5}]))

    with pytest.raises(ValueError):
        tx.update_rows({"score": 1.5})

    for operation in [
        lambda: tx.append(_data(SCHEMA, [{"id": 6, "name": "f", "score": 60, "metric": 6.5}])),
        lambda: tx.overwrite(_data(SCHEMA, _basic_rows())),
        lambda: tx.update_schema(),
        lambda: tx.update_spec(),
        lambda: tx.delete(EqualTo("id", 1)),
        lambda: tx.add_files([]),
        lambda: tx.commit_transaction(),
    ]:
        with pytest.raises(RuntimeError, match="aborted"):
            operation()

    # The staged append was never committed.
    assert len(catalog.load_table(IDENTIFIER).metadata.snapshots) == 1
    assert catalog.load_table(IDENTIFIER).scan().count() == 4


def test_other_transactions_are_unaffected_by_an_aborted_one(catalog: Catalog) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())

    failing = catalog.load_table(IDENTIFIER).transaction()
    with pytest.raises(ValueError):
        failing.update_rows({"score": "nope"})

    healthy = catalog.load_table(IDENTIFIER)
    assert healthy.update_rows({"score": 7}, row_filter=EqualTo("id", 2)) == 1
    assert {row["id"]: row for row in healthy.scan().to_arrow().to_pylist()}[2]["score"] == 7


def test_read_failure_publishes_nothing_and_keeps_history(catalog: Catalog, mocker: Any) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())
    original = {path: _file_bytes(tbl, path) for path in _data_file_paths(tbl)}
    reset()

    mocker.patch(
        "pyiceberg.table.update_rows._read_task_rows",
        side_effect=OSError("simulated read failure"),
    )
    with pytest.raises(OSError, match="simulated read failure"):
        tbl.update_rows({"name": "x"})

    fresh = catalog.load_table(IDENTIFIER)
    assert len(fresh.metadata.snapshots) == 1
    assert _rows(fresh) == sorted(_basic_rows(), key=lambda row: tuple(str(row[name]) for name in row))
    for path, content in original.items():
        assert _file_bytes(fresh, path) == content
    assert not created_data_files()


def test_write_failure_publishes_nothing_and_keeps_history(catalog: Catalog, mocker: Any) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())
    original = {path: _file_bytes(tbl, path) for path in _data_file_paths(tbl)}

    mocker.patch(
        "pyiceberg.io.pyarrow._dataframe_to_data_files",
        side_effect=OSError("simulated write failure"),
    )
    with pytest.raises(OSError, match="simulated write failure"):
        tbl.update_rows({"name": "x"})

    fresh = catalog.load_table(IDENTIFIER)
    assert len(fresh.metadata.snapshots) == 1
    for path, content in original.items():
        assert _file_bytes(fresh, path) == content


def test_commit_failure_exposes_no_partial_state(catalog: Catalog, mocker: Any) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())
    original = {path: _file_bytes(tbl, path) for path in _data_file_paths(tbl)}

    mocker.patch.object(catalog, "commit_table", side_effect=CommitFailedException("rejected by catalog"))
    with pytest.raises(CommitFailedException):
        tbl.update_rows({"name": "x"})

    fresh = catalog.load_table(IDENTIFIER)
    assert len(fresh.metadata.snapshots) == 1
    assert _rows(fresh) == sorted(_basic_rows(), key=lambda row: tuple(str(row[name]) for name in row))
    assert not fresh.inspect.delete_files().to_pylist()
    for path, content in original.items():
        assert _file_bytes(fresh, path) == content


def test_unknown_commit_result_does_not_delete_files(catalog: Catalog, mocker: Any) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())
    original = {path: _file_bytes(tbl, path) for path in _data_file_paths(tbl)}

    delete_spy = mocker.spy(RecordingFileIO, "delete")
    mocker.patch.object(catalog, "commit_table", side_effect=CommitStateUnknownException("timeout"))
    with pytest.raises(CommitStateUnknownException):
        tbl.update_rows({"name": "x"})

    assert delete_spy.call_count == 0
    fresh = catalog.load_table(IDENTIFIER)
    assert len(fresh.metadata.snapshots) == 1
    for path, content in original.items():
        assert _file_bytes(fresh, path) == content


def test_transaction_commit_failure_leaves_table_unchanged(catalog: Catalog, mocker: Any) -> None:
    tbl = _create(catalog)
    _append(tbl, SCHEMA, _basic_rows())
    tx = tbl.transaction()
    assert tx.update_rows({"name": "pending"}, row_filter=EqualTo("id", 1)) == 1

    mocker.patch.object(catalog, "commit_table", side_effect=CommitFailedException("rejected"))
    with pytest.raises(CommitFailedException):
        tx.commit_transaction()

    fresh = catalog.load_table(IDENTIFIER)
    assert len(fresh.metadata.snapshots) == 1
    assert _rows(fresh) == sorted(_basic_rows(), key=lambda row: tuple(str(row[name]) for name in row))
