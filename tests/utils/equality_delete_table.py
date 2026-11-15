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
"""Helpers that build real on-disk Iceberg v2 tables for equality delete tests.

The helpers write actual Parquet data/delete files, Avro manifests and
manifest lists, and JSON table metadata.  Nothing about the result is mocked:
the tables produced here are read back through the public ``StaticTable`` /
``table.scan()`` entry points.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from pyiceberg.io import load_file_io
from pyiceberg.manifest import (
    DataFile,
    DataFileContent,
    FileFormat,
    ManifestContent,
    ManifestEntry,
    ManifestEntryStatus,
    ManifestFile,
    write_manifest,
    write_manifest_list,
)
from pyiceberg.partitioning import PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.serializers import ToOutputFile
from pyiceberg.table import StaticTable
from pyiceberg.table.metadata import TableMetadataV2
from pyiceberg.table.snapshots import Operation, Snapshot, Summary
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER
from pyiceberg.typedef import Record


@dataclass
class PlannedFile:
    """One data or delete file as it should appear in a manifest."""

    arrow_table: Optional[pa.Table]
    content: DataFileContent
    spec_id: int
    partition: Record
    sequence_number: Optional[int]
    equality_ids: Optional[List[int]] = None
    # When None the file sequence number equals the data sequence number.
    file_sequence_number: Optional[int] = None
    file_path: Optional[str] = None
    record_count: Optional[int] = None
    # When set, this raw DataFile is written to the manifest verbatim instead
    # of constructing one from arrow_table/file_path.  Useful for malformed
    # metadata tests (missing equality_ids, unknown ids, ...).
    raw_data_file: Optional[DataFile] = None


@dataclass
class PlannedSnapshot:
    snapshot_id: int
    parent_snapshot_id: Optional[int]
    sequence_number: int
    # manifest grouping: each entry is (manifest spec id, manifest content, files)
    groups: List[Tuple[int, ManifestContent, List[PlannedFile]]]
    schema_id: int


class LocalV2TableBuilder:
    """Builds an Iceberg format-version=2 table on a local filesystem."""

    def __init__(
        self,
        warehouse_dir: str,
        table_name: str,
        schemas: Sequence[Schema],
        current_schema_id: int,
        specs: Sequence[PartitionSpec],
        default_spec_id: int = 0,
    ) -> None:
        self.table_dir = os.path.join(warehouse_dir, table_name)
        self.data_dir = os.path.join(self.table_dir, "data")
        self.metadata_dir = os.path.join(self.table_dir, "metadata")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        self.schemas = list(schemas)
        self.current_schema_id = current_schema_id
        self.specs = list(specs)
        self.default_spec_id = default_spec_id
        self._counter = 0
        self.io = load_file_io()

    def _next_file_path(self, prefix: str) -> str:
        self._counter += 1
        return f"file://{self.data_dir}/{prefix}-{self._counter:05d}-{uuid.uuid4()}.parquet"

    def _write_parquet(self, arrow_table: pa.Table, prefix: str) -> str:
        path = self._next_file_path(prefix)
        pq.write_table(arrow_table, path.replace("file://", ""))
        return path

    def _data_file_from(self, planned: PlannedFile) -> DataFile:
        if planned.raw_data_file is not None:
            return planned.raw_data_file
        if planned.file_path is None:
            prefix = "data" if planned.content == DataFileContent.DATA else "del"
            assert planned.arrow_table is not None
            planned.file_path = self._write_parquet(planned.arrow_table, prefix)
        assert planned.arrow_table is not None
        record_count = planned.record_count if planned.record_count is not None else planned.arrow_table.num_rows
        return DataFile(
            content=planned.content,
            file_path=planned.file_path,
            file_format=FileFormat.PARQUET,
            partition=planned.partition,
            record_count=record_count,
            file_size_in_bytes=os.path.getsize(planned.file_path.replace("file://", "")),
            column_sizes=None,
            value_counts=None,
            null_value_counts=None,
            nan_value_counts=None,
            lower_bounds=None,
            upper_bounds=None,
            key_metadata=None,
            split_offsets=None,
            equality_ids=planned.equality_ids,
            sort_order_id=None,
            spec_id=planned.spec_id,
        )

    def _write_manifest(
        self,
        snapshot_id: int,
        spec: PartitionSpec,
        schema: Schema,
        planned_files: List[PlannedFile],
        manifest_index: int,
    ) -> ManifestFile:
        manifest_path = f"file://{self.metadata_dir}/snap-{snapshot_id}-{manifest_index}.avro"
        with write_manifest(
            format_version=2,
            spec=spec,
            schema=schema,
            output_file=self.io.new_output(manifest_path),
            snapshot_id=snapshot_id,
        ) as writer:
            for planned in planned_files:
                data_file = self._data_file_from(planned)
                entry = ManifestEntry(
                    status=ManifestEntryStatus.ADDED,
                    snapshot_id=snapshot_id,
                    # An explicit None exercises sequence number inheritance
                    # from the manifest list when the writer is allowed to
                    # omit it.  A concrete number is written as-is.
                    sequence_number=planned.sequence_number,
                    file_sequence_number=planned.file_sequence_number,
                    data_file=data_file,
                )
                writer.add_entry(entry)
            manifest_file = writer.to_manifest_file()
        return manifest_file

    def build(
        self,
        planned_snapshots: List[PlannedSnapshot],
        last_sequence_number: Optional[int] = None,
        properties: Optional[Dict[str, str]] = None,
    ) -> str:
        snapshots: List[Snapshot] = []
        # Manifest files that are still live; each new snapshot's manifest
        # list contains both its newly written manifests and the manifests
        # inherited from the previous snapshot, exactly like a real commit.
        carried_manifests: List[ManifestFile] = []
        for planned_snapshot in planned_snapshots:
            manifest_files: List[ManifestFile] = list(carried_manifests)
            for index, (spec_id, _content, planned_files) in enumerate(planned_snapshot.groups):
                spec = next(spec for spec in self.specs if spec.spec_id == spec_id)
                snapshot_schema = next(schema for schema in self.schemas if schema.schema_id == planned_snapshot.schema_id)
                manifest_file = self._write_manifest(planned_snapshot.snapshot_id, spec, snapshot_schema, planned_files, index)
                # The manifest writer always produces DATA manifests; delete
                # manifests only differ by their content flag in the manifest
                # list, which is what readers actually consume.
                if _content == ManifestContent.DELETES:
                    manifest_file.content = ManifestContent.DELETES
                manifest_files.append(manifest_file)

            manifest_list_path = f"file://{self.metadata_dir}/snap-{planned_snapshot.snapshot_id}.manifest-list.avro"
            prepared_manifests: List[ManifestFile] = []
            with write_manifest_list(
                format_version=2,
                output_file=self.io.new_output(manifest_list_path),
                snapshot_id=planned_snapshot.snapshot_id,
                parent_snapshot_id=planned_snapshot.parent_snapshot_id,
                sequence_number=planned_snapshot.sequence_number,
            ) as list_writer:
                # Mirror ManifestListWriterV2's sequence number assignment so
                # that manifests carried into the next snapshot already carry
                # their finalized numbers.
                for manifest_file in manifest_files:
                    prepared = list_writer.prepare_manifest(manifest_file)
                    prepared_manifests.append(prepared)
                list_writer.add_manifests(manifest_files)
            carried_manifests = prepared_manifests

            snapshots.append(
                Snapshot(
                    **{
                        "snapshot-id": planned_snapshot.snapshot_id,
                        "parent-snapshot-id": planned_snapshot.parent_snapshot_id,
                        "sequence-number": planned_snapshot.sequence_number,
                        "timestamp-ms": 1_700_000_000_000 + planned_snapshot.snapshot_id % 100_000,
                        "manifest-list": manifest_list_path,
                        "summary": Summary(Operation.APPEND),
                        "schema-id": planned_snapshot.schema_id,
                    }
                )
            )

        current_snapshot_id = planned_snapshots[-1].snapshot_id if planned_snapshots else None
        metadata = TableMetadataV2(
            **{
                "format-version": 2,
                "table-uuid": str(uuid.uuid4()),
                "location": f"file://{self.table_dir}",
                "last-column-id": max(schema.highest_field_id for schema in self.schemas),
                "last-sequence-number": last_sequence_number
                if last_sequence_number is not None
                else max((s.sequence_number for s in planned_snapshots), default=0),
                "last-updated-ms": 1_700_000_000_000,
                "last-partition-id": max((spec.last_assigned_field_id for spec in self.specs), default=999),
                "current-schema-id": self.current_schema_id,
                "schemas": self.schemas,
                "default-spec-id": self.default_spec_id,
                "partition-specs": self.specs,
                "default-sort-order-id": 0,
                "sort-orders": [UNSORTED_SORT_ORDER],
                "current-snapshot-id": current_snapshot_id,
                "snapshots": snapshots,
                "properties": properties or {},
                "snapshot-log": [
                    {"snapshot-id": snapshot.snapshot_id, "timestamp-ms": snapshot.timestamp_ms} for snapshot in snapshots
                ],
                "metadata-log": [],
                "refs": {},
            }
        )

        metadata_location = f"file://{self.metadata_dir}/v1.metadata.json"
        ToOutputFile.table_metadata(metadata, self.io.new_output(metadata_location), overwrite=True)
        return metadata_location

    def load(self, metadata_location: str) -> StaticTable:
        return StaticTable.from_metadata(metadata_location)


def parquet_table_with_field_ids(columns: Dict[str, Tuple[pa.Array, int]]) -> pa.Table:
    """Build a Parquet-ready Arrow table carrying Iceberg field ids.

    Args:
        columns: mapping of column name to (values, field id).
    """
    fields = []
    arrays = []
    for name, (array, field_id) in columns.items():
        fields.append(pa.field(name, array.type, nullable=True, metadata={b"PARQUET:field_id": str(field_id)}))
        arrays.append(array)
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def write_corrupt_parquet(path: str) -> None:
    """Write a file that is not a valid Parquet file."""
    with open(path.replace("file://", ""), "wb") as handle:
        handle.write(b"this is definitely not a parquet file")
