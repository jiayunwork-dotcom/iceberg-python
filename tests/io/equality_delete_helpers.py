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
"""Helpers that build real v2 tables (manifest list, manifests, Parquet files).

The helpers write genuine Parquet data/delete files and genuine Avro manifests so
that equality delete tests exercise real scan planning, not mocked results.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from pyiceberg.avro.file import AvroOutputFile
from pyiceberg.catalog.noop import NoopCatalog
from pyiceberg.io import load_file_io
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.manifest import (
    DataFile,
    DataFileContent,
    FileFormat,
    ManifestContent,
    ManifestEntry,
    ManifestEntryStatus,
    ManifestFile,
    construct_partition_summaries,
    data_file_with_partition,
    manifest_entry_schema_with_data_file,
    write_manifest_list,
)
from pyiceberg.partitioning import PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.table.metadata import TableMetadataV2
from pyiceberg.typedef import Record

PARQUET_FIELD_ID = b"PARQUET:field_id"


def write_parquet(
    path: str,
    columns: Dict[str, List[Any]],
    field_ids: Dict[str, int],
    types: Optional[Dict[str, pa.DataType]] = None,
    row_group_size: Optional[int] = None,
) -> str:
    """Write a Parquet file whose columns carry Iceberg field ids.

    ``field_ids`` and ``types`` are matched to ``columns`` by key; unused keys are
    ignored.  This lets a test pass only the field ids for its own columns.
    """
    fields = []
    for name, values in columns.items():
        arrow_type = types[name] if types and name in types else pa.array(values).type
        fields.append(pa.field(name, arrow_type, metadata={PARQUET_FIELD_ID: str(field_ids[name]).encode()}))
    schema = pa.schema(fields)
    arrays = [pa.array(values, type=schema.field(name).type) for name, values in columns.items()]
    table = pa.table(arrays, schema=schema)
    if row_group_size is not None:
        pq.write_table(table, path, row_group_size=row_group_size)
    else:
        pq.write_table(table, path)
    return path


def _partition_record(spec: PartitionSpec, values: Sequence[Any]) -> Record:
    return Record(*values) if spec.fields else Record()


def build_table(
    tmp: str,
    schema: Schema,
    specs: Dict[int, PartitionSpec],
    data_files: List[Dict[str, Any]],
    delete_files: List[Dict[str, Any]],
    sequence_number: int = 10,
    schemas: Optional[List[Schema]] = None,
    default_spec_id: int = 0,
    snapshot_id: int = 1,
    extra_snapshots: Optional[List[Dict[str, Any]]] = None,
    current_snapshot_id: Optional[int] = None,
    properties: Optional[Dict[str, str]] = None,
) -> Table:
    """Build a Table backed by real manifests and a real manifest list.

    Each ``data_files`` / ``delete_files`` entry has:
      path, records, partition (list), spec_id, seq, and optionally equality_ids
      or pos=True (position delete).
    """
    io = PyArrowFileIO()
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for kind, files in (("data", data_files), ("delete", delete_files)):
        for entry in files:
            groups.setdefault((kind, entry["spec_id"]), []).append(entry)

    manifest_files: List[ManifestFile] = []
    for (kind, spec_id), files in groups.items():
        spec = specs[spec_id]
        content = ManifestContent.DATA if kind == "data" else ManifestContent.DELETES
        path = os.path.join(tmp, f"manifest-{kind}-{spec_id}-{uuid.uuid4().hex[:8]}.avro")
        eschema = manifest_entry_schema_with_data_file(2, data_file_with_partition(spec.partition_type(schema), 2))
        writer = AvroOutputFile(
            output_file=io.new_output(path),
            file_schema=eschema,
            record_schema=eschema,
            schema_name="manifest_entry",
            metadata={
                "schema": schema.model_dump_json(),
                "partition-spec": "[]",
                "partition-spec-id": str(spec_id),
                "format-version": "2",
                "avro.codec": "null",
                "content": "data" if kind == "data" else "deletes",
            },
        )
        partitions = []
        with writer as w:
            for entry in files:
                if kind == "data":
                    file_content = DataFileContent.DATA
                elif entry.get("pos"):
                    file_content = DataFileContent.POSITION_DELETES
                else:
                    file_content = DataFileContent.EQUALITY_DELETES
                disk_path = entry["path"][len("file://") :] if entry["path"].startswith("file://") else entry["path"]
                data_file = DataFile.from_args(
                    content=file_content,
                    file_path=entry["path"],
                    file_format=FileFormat.PARQUET,
                    partition=_partition_record(spec, entry.get("partition", [])),
                    record_count=entry["records"],
                    file_size_in_bytes=os.path.getsize(disk_path),
                    equality_ids=entry.get("equality_ids"),
                    spec_id=spec_id,
                )
                w.write_block(
                    [
                        ManifestEntry.from_args(
                            status=ManifestEntryStatus.ADDED,
                            snapshot_id=snapshot_id,
                            sequence_number=entry["seq"],
                            file_sequence_number=entry["seq"],
                            data_file=data_file,
                        )
                    ]
                )
                partitions.append(_partition_record(spec, entry.get("partition", [])))

        manifest_files.append(
            ManifestFile.from_args(
                manifest_path=path,
                manifest_length=os.path.getsize(path),
                partition_spec_id=spec_id,
                content=content,
                sequence_number=max(entry["seq"] for entry in files),
                min_sequence_number=min(entry["seq"] for entry in files),
                added_snapshot_id=snapshot_id,
                added_files_count=len(files),
                existing_files_count=0,
                deleted_files_count=0,
                added_rows_count=sum(entry["records"] for entry in files),
                existing_rows_count=0,
                deleted_rows_count=0,
                partitions=construct_partition_summaries(spec, schema, partitions),
                key_metadata=None,
            )
        )

    list_path = os.path.join(tmp, f"manifest-list-{uuid.uuid4().hex[:8]}.avro")
    with write_manifest_list(
        2,
        io.new_output(list_path),
        snapshot_id=snapshot_id,
        parent_snapshot_id=None,
        sequence_number=sequence_number,
        avro_compression="null",
    ) as writer:
        writer.add_manifests(manifest_files)

    snapshots = [
        {
            "snapshot-id": snapshot_id,
            "timestamp-ms": 1,
            "sequence-number": sequence_number,
            "manifest-list": list_path,
            "summary": {"operation": "append"},
            "schema-id": schema.schema_id,
        }
    ]
    if extra_snapshots:
        snapshots = extra_snapshots + snapshots

    metadata = TableMetadataV2(
        location=tmp,
        last_column_id=100,
        format_version=2,
        current_schema_id=schema.schema_id,
        schemas=schemas or [schema],
        partition_specs=list(specs.values()),
        default_spec_id=default_spec_id,
        sort_orders=[],
        default_sort_order_id=0,
        last_partition_id=1000,
        last_sequence_number=sequence_number,
        snapshots=snapshots,
        current_snapshot_id=current_snapshot_id or snapshot_id,
        properties=properties or {},
    )
    return Table(
        identifier=("db", "t"),
        metadata=metadata,
        metadata_location=os.path.join(tmp, "metadata.json"),
        io=load_file_io(),
        catalog=NoopCatalog("noop"),
    )
