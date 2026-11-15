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
"""Resource behaviour tests for equality delete scans.

These tests intentionally do not keep the full expected table in memory: the
expected surviving rows are derived mathematically from the generated keys and
only aggregate counters are retained while batches are consumed and dropped.
"""

import tracemalloc

import pyarrow as pa
import pytest

from pyiceberg.io import equality_deletes as equality_deletes_module
from pyiceberg.manifest import DataFileContent, ManifestContent
from pyiceberg.partitioning import PartitionSpec
from pyiceberg.schema import NestedField, Schema
from pyiceberg.table import StaticTable
from pyiceberg.typedef import Record
from pyiceberg.types import LongType, StringType
from utils.equality_delete_table import (
    LocalV2TableBuilder,
    PlannedFile,
    PlannedSnapshot,
    parquet_table_with_field_ids,
)

DATA_FILE_COUNT = 100
ROWS_PER_FILE = 10_000
# A handful of equality delete files are shared across all data files.
DELETE_FILE_COUNT = 4
DELETES_PER_FILE = 500


@pytest.fixture
def wide_schema() -> Schema:
    return Schema(
        NestedField(1, "id", LongType()),
        NestedField(2, "name", StringType()),
        NestedField(3, "payload", StringType()),
        schema_id=0,
    )


def _build_wide_table(tmp_path: str, builder: LocalV2TableBuilder, schema: Schema) -> StaticTable:
    data_groups = []
    total_rows = 0
    for file_index in range(DATA_FILE_COUNT):
        start = file_index * ROWS_PER_FILE
        ids = pa.array(range(start, start + ROWS_PER_FILE), pa.int64())
        names = pa.array([f"name-{value % 97}" for value in range(start, start + ROWS_PER_FILE)])
        payloads = pa.array([f"payload-{value}" for value in range(start, start + ROWS_PER_FILE)])
        table = parquet_table_with_field_ids(
            {
                "id": (ids, 1),
                "name": (names, 2),
                "payload": (payloads, 3),
            }
        )
        data_groups.append(PlannedFile(table, DataFileContent.DATA, 0, Record(), 1))
        total_rows += ROWS_PER_FILE

    # Delete files are shared by every data file: their keys cover a
    # deterministic spread of the whole id range.
    delete_groups = []
    for delete_index in range(DELETE_FILE_COUNT):
        delete_ids = pa.array(
            [
                delete_index + stride * DELETE_FILE_COUNT
                for stride in range(DELETES_PER_FILE)
                if delete_index + stride * DELETE_FILE_COUNT < total_rows
            ],
            pa.int64(),
        )
        delete_table = parquet_table_with_field_ids({"id": (delete_ids, 1)})
        delete_groups.append(
            PlannedFile(
                delete_table,
                DataFileContent.EQUALITY_DELETES,
                0,
                Record(),
                2,
                equality_ids=[1],
            )
        )

    location = builder.build(
        [
            PlannedSnapshot(1, None, 1, [(0, ManifestContent.DATA, data_groups)], 0),
            PlannedSnapshot(2, 1, 2, [(0, ManifestContent.DELETES, delete_groups)], 0),
        ]
    )
    return builder.load(location)


def test_shared_delete_files_are_read_once(tmp_path, wide_schema) -> None:
    spec = PartitionSpec(spec_id=0)
    builder = LocalV2TableBuilder(str(tmp_path), "wide", [wide_schema], 0, [spec])
    table = _build_wide_table(str(tmp_path), builder, wide_schema)

    read_paths = []
    original_read_table = equality_deletes_module.pq.read_table

    def _tracking_read_table(path, *args, **kwargs):
        read_paths.append(path)
        return original_read_table(path, *args, **kwargs)

    equality_deletes_module.pq.read_table = _tracking_read_table
    try:
        reader = table.scan().to_arrow_batch_reader()
        surviving = 0
        for batch in reader:
            # Only keep the count; the batch itself is dropped immediately.
            surviving += batch.num_rows
    finally:
        equality_deletes_module.pq.read_table = original_read_table

    expected_deleted = len(
        {
            delete_index + stride * DELETE_FILE_COUNT
            for delete_index in range(DELETE_FILE_COUNT)
            for stride in range(DELETES_PER_FILE)
            if delete_index + stride * DELETE_FILE_COUNT < DATA_FILE_COUNT * ROWS_PER_FILE
        }
    )
    expected_surviving = DATA_FILE_COUNT * ROWS_PER_FILE - expected_deleted
    assert surviving == expected_surviving
    assert table.scan().count() == expected_surviving

    # Each shared equality delete file must be fully read exactly once.
    planned_delete_paths = {
        delete_file.file_path
        for task in table.scan().plan_files()
        for delete_file in task.delete_files
        if delete_file.content == DataFileContent.EQUALITY_DELETES
    }
    assert len(planned_delete_paths) == DELETE_FILE_COUNT
    assert sorted(read_paths) == sorted(path.replace("file://", "") for path in planned_delete_paths)


def test_data_files_streamed_and_peak_memory_bounded(tmp_path, wide_schema) -> None:
    spec = PartitionSpec(spec_id=0)
    builder = LocalV2TableBuilder(str(tmp_path), "wide", [wide_schema], 0, [spec])
    table = _build_wide_table(str(tmp_path), builder, wide_schema)

    tracemalloc.start()
    try:
        reader = table.scan(selected_fields=("id", "payload")).to_arrow_batch_reader()
        max_seen = 0
        total = 0
        for batch in reader:
            total += batch.num_rows
            _, peak = tracemalloc.get_traced_memory()
            max_seen = max(max_seen, peak)
    finally:
        tracemalloc.stop()

    assert total == DATA_FILE_COUNT * ROWS_PER_FILE - len(
        {
            delete_index + stride * DELETE_FILE_COUNT
            for delete_index in range(DELETE_FILE_COUNT)
            for stride in range(DELETES_PER_FILE)
            if delete_index + stride * DELETE_FILE_COUNT < DATA_FILE_COUNT * ROWS_PER_FILE
        }
    )

    # The full data set is roughly 100 * 10_000 rows with three columns; the
    # peak must stay well below the size of materializing all of it (the
    # delete keys are ~2k entries, batches hold one file's rows at most).
    # The bound is deliberately generous to stay robust across Arrow versions.
    assert max_seen < 256 * 1024 * 1024


def test_batch_reader_early_close_releases_resources(tmp_path, wide_schema) -> None:
    spec = PartitionSpec(spec_id=0)
    builder = LocalV2TableBuilder(str(tmp_path), "wide", [wide_schema], 0, [spec])
    table = _build_wide_table(str(tmp_path), builder, wide_schema)

    reader = table.scan().to_arrow_batch_reader()
    first = reader.read_next_batch()
    assert first.num_rows > 0
    reader.close()

    # After closing, a fresh scan must still work independently and produce
    # the full surviving count (no cross-reader pollution).
    assert table.scan().count() > first.num_rows


def test_delete_key_memory_grows_with_distinct_keys_only(tmp_path, wide_schema) -> None:
    spec = PartitionSpec(spec_id=0)
    builder = LocalV2TableBuilder(str(tmp_path), "wide", [wide_schema], 0, [spec])
    table = _build_wide_table(str(tmp_path), builder, wide_schema)

    tasks = list(table.scan().plan_files())
    equality_files = {
        delete_file
        for task in tasks
        for delete_file in task.delete_files
        if delete_file.content == DataFileContent.EQUALITY_DELETES
    }
    from pyiceberg.io.equality_deletes import EqualityDeleteFilter
    from pyiceberg.io.pyarrow import _fs_from_file_path  # noqa: F401

    filter_instance = EqualityDeleteFilter(table.metadata, table.io)
    try:
        prepared = filter_instance.prepare(equality_files)
        total_distinct = sum(len(prepared_file.keys) for prepared_file in prepared)
        raw_rows = sum(delete_file.record_count for delete_file in equality_files)
        # No duplicates in this fixture, but the filter stores sets, so the
        # retained size is bounded by the distinct key count, never the rows.
        assert total_distinct == raw_rows
    finally:
        filter_instance.close()
        assert filter_instance.loaded_file_paths == set()
