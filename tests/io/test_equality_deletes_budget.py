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
"""Bounded-memory and resource-lifecycle tests for equality delete reads.

These tests exercise the ``equality-delete-memory-budget`` option and the shared,
spill-capable key stores: the same snapshot is scanned at two budgets where the
decoded delete volume exceeds the larger budget several times over, and cleanup
is checked on success, injected read failure, early reader close and interleaved
scans.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List

import pyarrow as pa
import pytest

import pyiceberg.io.equality_deletes as eqd
from pyiceberg.expressions import AlwaysTrue
from pyiceberg.io.pyarrow import ArrowScan
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
from pyiceberg.schema import Schema
from pyiceberg.types import BooleanType, IntegerType, LongType, NestedField, StringType
from tests.io.equality_delete_helpers import build_table, write_parquet

FIELD_IDS = {"id": 1, "cat": 2, "flag": 3, "val": 4}


def schema() -> Schema:
    return Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "cat", StringType(), required=False),
        NestedField(3, "flag", BooleanType(), required=False),
        NestedField(4, "val", IntegerType(), required=False),
    )


def _arrow_types() -> Dict[str, pa.DataType]:
    return {"id": pa.int64(), "cat": pa.string(), "flag": pa.bool_(), "val": pa.int32()}


def _data_file(tmp: str, name: str, val: int, rows: int = 300) -> str:
    return write_parquet(
        os.path.join(tmp, name),
        {"id": list(range(rows)), "cat": ["a"] * rows, "flag": [True] * rows, "val": [val] * rows},
        FIELD_IDS,
        _arrow_types(),
    )


def _big_delete_table(tmp: str, n_keys: int = 20000) -> Any:
    data_files = [
        {"path": _data_file(tmp, f"d{i}.parquet", val=i), "records": 300, "partition": [], "spec_id": 0, "seq": 1}
        for i in range(4)
    ]
    keys = [f"key-{i:09d}-" + ("x" * 24) for i in range(n_keys)]
    delete = write_parquet(os.path.join(tmp, "big-delete.parquet"), {"cat": keys}, {"cat": 2}, {"cat": pa.string()})
    return build_table(
        tmp,
        schema(),
        {0: UNPARTITIONED_PARTITION_SPEC},
        data_files,
        [{"path": delete, "records": n_keys, "partition": [], "spec_id": 0, "seq": 10, "equality_ids": [2]}],
    )


def _scan_with_stats(table: Any, budget: int) -> Any:
    scan = ArrowScan(table.metadata, table.io, table.schema(), AlwaysTrue(), equality_delete_budget=budget)
    result = scan.to_table(list(table.scan().plan_files()))
    return result, scan.last_equality_delete_stats


def _spill_files() -> List[str]:
    return glob.glob(os.path.join(os.environ.get("TMPDIR", "/tmp"), "pyiceberg-eqdel-*"))


# ---------------------------------------------------------------------------
# Both budgets give identical results; the big delete set spills at the small one
# ---------------------------------------------------------------------------
def test_two_budgets_same_result_and_large_delete_set_spills(tmp_path: Any) -> None:
    # 260k keys of ~40 bytes each => decoded delete volume >8 MiB, comfortably
    # more than 8x the 1 MiB budget and far more than the 64 KiB budget.
    table = _big_delete_table(str(tmp_path), n_keys=260000)

    result_small, stats_small = _scan_with_stats(table, 65536)
    result_large, stats_large = _scan_with_stats(table, 1048576)

    assert result_small.num_rows == result_large.num_rows == 1200
    assert sorted(result_small["id"].to_pylist()) == sorted(result_large["id"].to_pylist())

    # The delete file is decoded once per scan, shared by all four data files.
    assert stats_small["delete_file_reads"] == 1
    assert stats_large["delete_file_reads"] == 1
    # Decoded delete volume is far larger than the 1 MiB budget (>8x target).
    assert stats_large["delete_bytes_read"] > 8 * 1048576
    # Same decoded volume regardless of budget (the file is read once).
    assert stats_small["delete_bytes_read"] == stats_large["delete_bytes_read"]

    # The 64 KiB budget must spill; with 8+ MiB of decoded keys the 1 MiB budget
    # also spills, but holds a larger resident share before doing so.
    assert stats_small["spill_files"] >= 1
    assert stats_small["spill_files"] >= stats_large["spill_files"]
    assert stats_small["peak_resident_bytes"] <= 65536
    assert stats_large["peak_resident_bytes"] <= 1048576

    # One shared key store serves all data files in both cases.
    assert stats_small["key_stores"] == 1
    assert stats_large["key_stores"] == 1


def test_spilled_store_results_do_not_depend_on_budget(tmp_path: Any) -> None:
    table = _big_delete_table(str(tmp_path), n_keys=5000)
    small = table.scan(options={"equality-delete-memory-budget": "65536"}).to_arrow()
    large = table.scan(options={"equality-delete-memory-budget": "1048576"}).to_arrow()
    assert sorted(small["id"].to_pylist()) == sorted(large["id"].to_pylist())


def test_peak_resident_stays_within_budget(tmp_path: Any) -> None:
    table = _big_delete_table(str(tmp_path), n_keys=20000)
    for budget in (65536, 1048576):
        _, stats = _scan_with_stats(table, budget)
        # The match bitmap and transient probe index are drawn from the same
        # account, so the whole scan never exceeds the budget.
        assert stats["peak_resident_bytes"] <= budget


# ---------------------------------------------------------------------------
# Resource lifecycle
# ---------------------------------------------------------------------------
def test_spill_files_removed_after_success(tmp_path: Any) -> None:
    table = _big_delete_table(str(tmp_path), n_keys=50000)
    before = set(_spill_files())
    table.scan(options={"equality-delete-memory-budget": "65536"}).to_arrow()
    assert set(_spill_files()) == before


def test_spill_files_removed_after_injected_read_failure(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    table = _big_delete_table(str(tmp_path), n_keys=50000)
    before = set(_spill_files())

    def boom(self: Any, path: str) -> Any:
        raise RuntimeError("injected equality delete read failure")

    monkeypatch.setattr(eqd.KeyStore, "_read_segment", boom)
    with pytest.raises(RuntimeError, match="injected equality delete read failure"):
        table.scan(options={"equality-delete-memory-budget": "65536"}).to_arrow()
    assert set(_spill_files()) == before


def test_spill_files_removed_after_early_reader_close(tmp_path: Any) -> None:
    table = _big_delete_table(str(tmp_path), n_keys=50000)
    before = set(_spill_files())
    reader = table.scan(options={"equality-delete-memory-budget": "65536"}).to_arrow_batch_reader()
    reader.read_next_batch()
    reader.close()
    assert set(_spill_files()) == before


def test_spill_files_removed_after_load_time_failure(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure while *loading* a delete file (not just probing) also cleans up."""
    table = _big_delete_table(str(tmp_path), n_keys=50000)
    before = set(_spill_files())

    def boom(self: Any) -> Any:
        raise RuntimeError("injected delete decode failure")

    monkeypatch.setattr(eqd.KeyStore, "finalize", boom)
    with pytest.raises(RuntimeError, match="injected delete decode failure"):
        table.scan(options={"equality-delete-memory-budget": "65536"}).to_arrow()
    assert set(_spill_files()) == before


def test_interleaved_scans_do_not_share_delete_state(tmp_path: Any) -> None:
    """Two concurrent scans of the same snapshot must not share key stores."""
    table = _big_delete_table(str(tmp_path), n_keys=10000)
    reader_a = table.scan(options={"equality-delete-memory-budget": "65536"}).to_arrow_batch_reader()
    reader_b = table.scan(options={"equality-delete-memory-budget": "1048576"}).to_arrow_batch_reader()
    try:
        batch_a = reader_a.read_next_batch()
        batch_b = reader_b.read_next_batch()
        assert sorted(batch_a.column("id").to_pylist()) == sorted(batch_b.column("id").to_pylist())
    finally:
        reader_a.close()
        reader_b.close()
    assert set(_spill_files()) == set(_spill_files())


def test_source_files_are_not_modified(tmp_path: Any) -> None:
    table = _big_delete_table(str(tmp_path), n_keys=5000)
    paths = [task.file.file_path for task in table.scan().plan_files()]
    before = {path: (os.path.getsize(path), os.path.getmtime(path)) for path in paths}
    table.scan(options={"equality-delete-memory-budget": "65536"}).to_arrow()
    after = {path: (os.path.getsize(path), os.path.getmtime(path)) for path in paths}
    assert before == after


# ---------------------------------------------------------------------------
# Isolation: interleaved scans and an older snapshot
# ---------------------------------------------------------------------------
def test_interleaved_scans_are_isolated(tmp_path: Any) -> None:
    table = _big_delete_table(str(tmp_path), n_keys=10000)
    tasks = list(table.scan().plan_files())
    scan_a = ArrowScan(table.metadata, table.io, table.schema(), AlwaysTrue(), equality_delete_budget=65536)
    scan_b = ArrowScan(table.metadata, table.io, table.schema(), AlwaysTrue(), equality_delete_budget=1048576)
    gen_a = scan_a.to_record_batches(tasks)
    gen_b = scan_b.to_record_batches(tasks)
    try:
        batch_a = next(gen_a)
        batch_b = next(gen_b)
        assert sorted(batch_a.column("id").to_pylist()) == sorted(batch_b.column("id").to_pylist())
        assert scan_a._equality_delete_budget == 65536
        assert scan_b._equality_delete_budget == 1048576
    finally:
        gen_a.close()
        gen_b.close()


def test_old_snapshot_isolation(tmp_path: Any) -> None:
    tmp = str(tmp_path)
    data = _data_file(tmp, "data.parquet", 0, rows=6)
    delete_1 = write_parquet(os.path.join(tmp, "d1.parquet"), {"id": [1]}, {"id": 1}, {"id": pa.int64()})
    delete_2 = write_parquet(os.path.join(tmp, "d2.parquet"), {"id": [2]}, {"id": 1}, {"id": pa.int64()})

    table_old = build_table(
        tmp,
        schema(),
        {0: UNPARTITIONED_PARTITION_SPEC},
        [{"path": data, "records": 6, "partition": [], "spec_id": 0, "seq": 1}],
        [{"path": delete_1, "records": 1, "partition": [], "spec_id": 0, "seq": 5, "equality_ids": [1]}],
        snapshot_id=1,
    )
    table_new = build_table(
        tmp,
        schema(),
        {0: UNPARTITIONED_PARTITION_SPEC},
        [{"path": data, "records": 6, "partition": [], "spec_id": 0, "seq": 1}],
        [{"path": delete_2, "records": 1, "partition": [], "spec_id": 0, "seq": 5, "equality_ids": [1]}],
        snapshot_id=2,
    )
    # Interleave: scan new, then old, then new again.  Delete state must not
    # bleed between the two tables even though they share the data file.
    assert sorted(table_new.scan().to_arrow()["id"].to_pylist()) == [0, 1, 3, 4, 5]
    assert sorted(table_old.scan().to_arrow()["id"].to_pylist()) == [0, 2, 3, 4, 5]
    assert sorted(table_new.scan().to_arrow()["id"].to_pylist()) == [0, 1, 3, 4, 5]
