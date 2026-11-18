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
"""Shared, PyArrow-independent parts of equality delete support.

This module holds:

* validation of the ``equality-delete-memory-budget`` scan option (so an invalid
  value fails before any data page is opened),
* the immutable :class:`EqualityDeleteFile` value carried on a
  ``FileScanTask`` that records *which* delete file applies to *which* data
  file, and
* the Iceberg v2 scan-planning rules for equality deletes: strict data sequence
  number comparison and partition scope (including global, unpartitioned
  deletes).

It deliberately has no PyArrow dependency; the execution side lives in
``pyiceberg.io.equality_deletes``.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
from pyiceberg.schema import Schema
from pyiceberg.types import BooleanType, IcebergType, IntegerType, LongType, NestedField, StringType

EQUALITY_DELETE_MEMORY_BUDGET = "equality-delete-memory-budget"

# 64 MiB, in bytes.
DEFAULT_EQUALITY_DELETE_MEMORY_BUDGET = 67108864

# 64 KiB: below this, even one minimum-sized key plus its index cannot be held.
MIN_EQUALITY_DELETE_MEMORY_BUDGET = 65536

_DECIMAL_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*")

# Only these top-level delete-key types are supported.  Other types are rejected
# at planning time; they are never stringified or coerced.
SUPPORTED_EQUALITY_KEY_TYPES = (BooleanType, IntegerType, LongType, StringType)


def parse_equality_delete_memory_budget(options: Optional[Dict[str, Any]]) -> int:
    """Validate the ``equality-delete-memory-budget`` scan option.

    The value must be a decimal string of a positive integer, measured in bytes.
    Invalid values raise ``ValueError`` eagerly.

    Args:
        options: The options mapping passed to ``Table.scan``.

    Returns:
        The validated budget in bytes.
    """
    if not options or EQUALITY_DELETE_MEMORY_BUDGET not in options:
        return DEFAULT_EQUALITY_DELETE_MEMORY_BUDGET

    raw = options[EQUALITY_DELETE_MEMORY_BUDGET]
    if not isinstance(raw, str) or not _DECIMAL_POSITIVE_INTEGER.fullmatch(raw):
        raise ValueError(
            f"Invalid scan option {EQUALITY_DELETE_MEMORY_BUDGET}={raw!r}: "
            "expected a decimal positive integer string measured in bytes"
        )

    budget = int(raw)
    if budget < MIN_EQUALITY_DELETE_MEMORY_BUDGET:
        raise ValueError(
            f"Invalid scan option {EQUALITY_DELETE_MEMORY_BUDGET}={raw!r}: must be at least "
            f"{MIN_EQUALITY_DELETE_MEMORY_BUDGET} bytes"
        )
    return budget


@dataclass(frozen=True)
class EqualityDeleteFile:
    """An equality delete file that applies to one specific data file.

    Instances are produced by scan planning, the only place allowed to interpret
    data sequence numbers, spec ids and partition values.  The reader treats the
    object as authoritative and never re-derives applicability.

    Attributes:
        data_file: The delete file's manifest-level description (with
            ``equality_ids`` and ``spec_id``).
        data_sequence_number: The delete file's data sequence number, used for
            the strict ``data_seq < delete_seq`` rule.
        key_types: Canonical key types, in ``equality_ids`` order.
    """

    data_file: DataFile
    data_sequence_number: int
    key_types: Tuple[str, ...]

    @property
    def file_path(self) -> str:
        return self.data_file.file_path

    @property
    def equality_ids(self) -> Tuple[int, ...]:
        ids = self.data_file.equality_ids
        return tuple(ids) if ids else ()

    @property
    def spec_id(self) -> int:
        return self.data_file.spec_id


def validate_equality_delete_file(data_file: DataFile, schemas: Sequence[Schema], spec_ids: Sequence[int]) -> None:
    """Validate an equality delete file before any data page is opened.

    Raises:
        ValueError: When the file is not a Parquet equality delete file, declares
            no ``equality_ids``, references an unknown partition spec, uses a
            non-top-level delete key, or uses an unsupported key type.
    """
    if data_file.content != DataFileContent.EQUALITY_DELETES:
        raise ValueError(f"Not an equality delete file: {data_file.file_path}")
    if data_file.file_format != FileFormat.PARQUET:
        raise ValueError(
            f"Equality delete file {data_file.file_path} has unsupported format {data_file.file_format}; "
            "only Parquet equality delete files are supported"
        )
    if not data_file.equality_ids:
        raise ValueError(f"Equality delete file {data_file.file_path} does not declare equality_ids")
    if data_file.spec_id not in spec_ids:
        raise ValueError(f"Equality delete file {data_file.file_path} references unknown partition spec id {data_file.spec_id}")
    if len(set(data_file.equality_ids)) != len(data_file.equality_ids):
        raise ValueError(
            f"Equality delete file {data_file.file_path} declares duplicate equality_ids: {list(data_file.equality_ids)}"
        )
    for field_id in data_file.equality_ids:
        field = find_top_level_field(schemas, field_id, data_file.file_path)
        if not isinstance(field.field_type, SUPPORTED_EQUALITY_KEY_TYPES):
            raise ValueError(
                f"Equality delete key {field.name} (id {field_id}) in {data_file.file_path} has type "
                f"{field.field_type}; only top-level boolean, int, long and string keys are supported"
            )


def find_top_level_field(schemas: Sequence[Schema], field_id: int, file_path: str) -> NestedField:
    """Find a top-level field by id across all known schemas.

    Looking across historical schemas handles a delete key that was later
    dropped from the current schema: the delete file must still be applied.
    """
    for schema in schemas:
        for field in schema.fields:
            if field.field_id == field_id:
                return field
    raise ValueError(
        f"Equality delete file {file_path} uses delete key id {field_id}, which is not a top-level field in any "
        "schema known to the table"
    )


def canonical_key_type(field_type: IcebergType, field_id: int, file_path: str) -> str:
    """Map a supported Iceberg key type to its canonical encoding type."""
    if isinstance(field_type, BooleanType):
        return "boolean"
    if isinstance(field_type, (IntegerType, LongType)):
        # int and long share the canonical 64-bit representation, so int delete
        # keys match long data columns after legal promotion.
        return "long"
    if isinstance(field_type, StringType):
        return "string"
    raise ValueError(
        f"Equality delete key id {field_id} in {file_path} has type {field_type}; only boolean, int, long and "
        "string delete keys are supported"
    )


def equality_key_types(schemas: Sequence[Schema], equality_ids: Sequence[int], file_path: str) -> Tuple[str, ...]:
    """Return canonical key types for the delete file's key ids, by id."""
    types: List[str] = []
    for field_id in equality_ids:
        field = find_top_level_field(schemas, field_id, file_path)
        types.append(canonical_key_type(field.field_type, field_id, file_path))
    return tuple(types)


def _partition_values(record: object) -> List[Any]:
    """Extract ordered partition values from a ``Record`` or plain sequence."""
    if record is None:
        return []
    if isinstance(record, (list, tuple)):
        return list(record)
    if hasattr(record, "__len__") and hasattr(record, "__getitem__"):
        # ``Record`` (and similar structs) support len() and integer indexing.
        return [record[index] for index in range(len(record))]
    return [record]


def partition_values_equal(left: object, right: object) -> bool:
    """Compare two partition tuples per the Iceberg partition equality rules.

    Floating point values compare by bit layout with NaNs normalised, matching
    the spec's definition of partition equality.
    """
    left_values = _partition_values(left)
    right_values = _partition_values(right)
    if len(left_values) != len(right_values):
        return False
    return all(_single_partition_value_equal(a, b) for a, b in zip(left_values, right_values))


def _single_partition_value_equal(left: object, right: object) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return _float_bits_equal(left, right)
    return left == right


def _float_bits_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return left == right
    left_value = float(left)
    right_value = float(right)
    if math.isnan(left_value) or math.isnan(right_value):
        return math.isnan(left_value) and math.isnan(right_value)
    # Distinguish -0.0 from 0.0 by comparing bit patterns.
    return struct.pack(">d", left_value) == struct.pack(">d", right_value)


def matches_equality_delete(
    data_file: DataFile,
    data_sequence_number: int,
    delete_file: DataFile,
    delete_sequence_number: int,
    spec_is_unpartitioned: bool,
) -> bool:
    """Return whether an equality delete file applies to a data file.

    Iceberg v2 rules:
      * the data file's data sequence number is *strictly* less than the delete's
        data sequence number (so same-sequence data is never affected),
      * the partitions are equal (same spec id and same partition values), or the
        delete file's spec is unpartitioned, making it a global delete.
    """
    if not data_sequence_number < delete_sequence_number:
        return False
    if spec_is_unpartitioned:
        return True
    return delete_file.spec_id == data_file.spec_id and partition_values_equal(data_file.partition, delete_file.partition)


def equality_deletes_for_data_file(
    data_file: DataFile,
    data_sequence_number: int,
    delete_entries: Iterable[Tuple[DataFile, int, bool, Tuple[str, ...]]],
) -> Tuple[EqualityDeleteFile, ...]:
    """Select the equality delete files that apply to one data file.

    Args:
        data_file: The data file.
        data_sequence_number: The data file's data sequence number.
        delete_entries: ``(delete_file, delete_sequence_number, spec_is_unpartitioned, key_types)``.

    Returns:
        The applicable equality delete files, deduplicated by path.
    """
    applicable: List[EqualityDeleteFile] = []
    seen = set()
    for delete_file, delete_sequence_number, unpartitioned, key_types in delete_entries:
        if delete_file.file_path in seen:
            continue
        if matches_equality_delete(data_file, data_sequence_number, delete_file, delete_sequence_number, unpartitioned):
            seen.add(delete_file.file_path)
            applicable.append(
                EqualityDeleteFile(data_file=delete_file, data_sequence_number=delete_sequence_number, key_types=key_types)
            )
    return tuple(applicable)
