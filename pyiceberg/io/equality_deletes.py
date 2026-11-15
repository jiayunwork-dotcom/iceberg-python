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
"""Equality delete support for Iceberg v2 scans.

An equality delete file is a Parquet file that stores rows whose values on the
file's ``equality_ids`` identify data rows that must be removed.  The planning
side (:mod:`pyiceberg.table`) decides which equality delete files can apply to
which data files based on sequence numbers, partition specs and partition
values.  The read side (this module) materializes the delete keys into an
in-memory set and evaluates them against the Arrow batches streamed from the
data files.

Only the de-duplicated keys are held in memory; data files continue to be read
batch by batch.  A delete file that is shared between several data files of a
single scan is read only once.
"""

from __future__ import annotations

import datetime
import decimal
import threading
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pyiceberg.manifest import DataFile, DataFileContent
from pyiceberg.types import (
    BooleanType,
    DateType,
    DecimalType,
    IntegerType,
    ListType,
    LongType,
    MapType,
    NestedField,
    PrimitiveType,
    StringType,
    StructType,
    TimestampType,
    TimestamptzType,
    TimeType,
)

if TYPE_CHECKING:
    from pyiceberg.io import FileIO
    from pyiceberg.table.metadata import TableMetadata

# Key types that can be used in equality delete files.  The leaves have to be
# primitive, either at the top level or inside an optional struct.
_SUPPORTED_KEY_TYPES: Tuple[type, ...] = (
    BooleanType,
    IntegerType,
    LongType,
    StringType,
    DecimalType,
    DateType,
    TimeType,
    TimestampType,
    TimestamptzType,
)


class EqualityDeleteError(ValueError):
    """Raised when an equality delete file cannot be read or interpreted.

    The scan must fail rather than silently skipping the file, because doing so
    would return rows that should have been removed.
    """


@dataclass(frozen=True)
class _KeyField:
    """Resolved location of one equality key leaf field.

    Attributes:
        field_id: The Iceberg field id used in the delete file's equality_ids.
        iceberg_type: Canonical Iceberg type taken from the table's schema
            history.  Used to validate the physical type and to normalize
            values.
        arrow_path: Field names from the (physical) Arrow schema root to the
            column holding the leaf values.
    """

    field_id: int
    iceberg_type: PrimitiveType
    arrow_path: Tuple[str, ...]


@dataclass(frozen=True)
class _PreparedEqualityDelete:
    """The in-memory representation of one equality delete file."""

    data_file: DataFile
    equality_ids: Tuple[int, ...]
    key_fields: Tuple[_KeyField, ...]
    # The de-duplicated delete keys, each normalized to a tuple of Python
    # values ordered by equality_ids.  None values are kept so that SQL-style
    # null == null matching works.
    keys: Set[Tuple[Any, ...]]


def _is_supported_key_type(iceberg_type: PrimitiveType) -> bool:
    return isinstance(iceberg_type, _SUPPORTED_KEY_TYPES)


def _find_key_field_in_schemas(field_id: int, table_metadata: "TableMetadata") -> Optional[NestedField]:
    """Look up a field id across the current schema and the retained history.

    A column may have been used as an equality delete key and later dropped
    from the current schema.  Iceberg keeps old schemas in the table metadata,
    so the field can still be resolved there.
    """
    for schema in table_metadata.schemas:
        try:
            return schema.find_field(field_id)
        except ValueError:
            continue
    return None


def _index_arrow_leaves(arrow_schema: pa.Schema) -> Dict[int, Tuple[pa.Field, Tuple[str, ...]]]:
    """Map every Parquet field id to its Arrow field and name path.

    List and map leaves are intentionally indexed as well, so that a clear
    "unsupported key type" error can be raised instead of an unresolved field
    error when they are used as equality keys.
    """
    result: Dict[int, Tuple[pa.Field, Tuple[str, ...]]] = {}

    def _walk(arrow_type: pa.DataType, path: Tuple[str, ...]) -> None:
        if pa.types.is_struct(arrow_type):
            for field in arrow_type:
                field_id = field.metadata.get(b"PARQUET:field_id") if field.metadata else None
                next_path = path + (field.name,)
                if field_id is not None:
                    result[int(field_id)] = (field, next_path)
                _walk(field.type, next_path)
        elif pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
            value_field = arrow_type.value_field
            field_id = value_field.metadata.get(b"PARQUET:field_id") if value_field.metadata else None
            next_path = path + (value_field.name,)
            if field_id is not None:
                result[int(field_id)] = (value_field, next_path)
            _walk(value_field.type, next_path)
        elif pa.types.is_map(arrow_type):
            key_field = arrow_type.key_field
            item_field = arrow_type.item_field
            for sub_field in (key_field, item_field):
                field_id = sub_field.metadata.get(b"PARQUET:field_id") if sub_field.metadata else None
                next_path = path + (sub_field.name,)
                if field_id is not None:
                    result[int(field_id)] = (sub_field, next_path)
                _walk(sub_field.type, next_path)

    for field in arrow_schema:
        field_id = field.metadata.get(b"PARQUET:field_id") if field.metadata else None
        path = (field.name,)
        if field_id is not None:
            result[int(field_id)] = (field, path)
        _walk(field.type, path)

    return result


def _validate_equality_ids(data_file: DataFile) -> Tuple[int, ...]:
    """Validate the equality_ids manifest property of a delete file."""
    equality_ids = data_file.equality_ids
    if equality_ids is None:
        raise EqualityDeleteError(f"Equality delete file {data_file.file_path} is missing the equality_ids property")
    if len(equality_ids) == 0:
        raise EqualityDeleteError(f"Equality delete file {data_file.file_path} has an empty equality_ids list")
    if len(set(equality_ids)) != len(equality_ids):
        raise EqualityDeleteError(f"Equality delete file {data_file.file_path} has duplicate equality_ids: {list(equality_ids)}")
    return tuple(equality_ids)


def _assert_primitive_key(field: NestedField, data_file: DataFile) -> None:
    field_type = field.field_type
    if isinstance(field_type, (ListType, MapType, StructType)):
        raise EqualityDeleteError(
            f"Equality delete file {data_file.file_path} uses field id {field.field_id} "
            f"('{field.name}') of unsupported type {field_type}; only primitive keys at the top level "
            "or inside an optional struct are supported"
        )
    if not isinstance(field_type, PrimitiveType) or not _is_supported_key_type(field_type):
        raise EqualityDeleteError(
            f"Equality delete file {data_file.file_path} uses field id {field.field_id} "
            f"('{field.name}') of unsupported type {field_type}"
        )


def _normalize_primitive(value: Any, iceberg_type: PrimitiveType, file_path: str, field_id: int) -> Any:
    """Normalize a single key value into a hashable Python value.

    Integers stay Python ints (no precision loss), decimals stay
    ``decimal.Decimal`` (never a float), timestamps are converted following
    the Iceberg physical interpretation and nulls remain ``None``.
    """
    if value is None:
        return None

    if isinstance(iceberg_type, (IntegerType, LongType)):
        if isinstance(value, bool):
            # bool is a subclass of int; treat it as an incompatible value.
            raise EqualityDeleteError(f"Equality key field id {field_id} in {file_path} expects an integer but got a boolean")
        if not isinstance(value, int):
            raise EqualityDeleteError(
                f"Equality key field id {field_id} in {file_path} expects an integer but got {type(value).__name__}"
            )
        return value

    if isinstance(iceberg_type, DecimalType):
        if isinstance(value, decimal.Decimal):
            return value
        if isinstance(value, int):
            return decimal.Decimal(value)
        raise EqualityDeleteError(
            f"Equality key field id {field_id} in {file_path} expects a decimal but got {type(value).__name__}"
        )

    if isinstance(iceberg_type, BooleanType):
        if isinstance(value, bool):
            return value
        raise EqualityDeleteError(
            f"Equality key field id {field_id} in {file_path} expects a boolean but got {type(value).__name__}"
        )

    if isinstance(iceberg_type, StringType):
        if isinstance(value, str):
            return value
        raise EqualityDeleteError(
            f"Equality key field id {field_id} in {file_path} expects a string but got {type(value).__name__}"
        )

    if isinstance(iceberg_type, DateType):
        if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
            return value
        raise EqualityDeleteError(
            f"Equality key field id {field_id} in {file_path} expects a date but got {type(value).__name__}"
        )

    if isinstance(iceberg_type, TimeType):
        # Iceberg time is microseconds since midnight stored in a time64[us]
        # column, which Arrow converts to datetime.time.
        if isinstance(value, datetime.time):
            return value
        if isinstance(value, int):
            return value
        raise EqualityDeleteError(
            f"Equality key field id {field_id} in {file_path} expects a time but got {type(value).__name__}"
        )

    if isinstance(iceberg_type, (TimestampType, TimestamptzType)):
        if isinstance(value, datetime.datetime):
            if isinstance(iceberg_type, TimestamptzType):
                if value.tzinfo is None:
                    raise EqualityDeleteError(
                        f"Equality key field id {field_id} in {file_path} expects a timezone-aware timestamp"
                    )
                return value
            if value.tzinfo is not None:
                raise EqualityDeleteError(
                    f"Equality key field id {field_id} in {file_path} expects a timestamp without timezone "
                    "but got a timezone-aware value"
                )
            return value
        raise EqualityDeleteError(
            f"Equality key field id {field_id} in {file_path} expects a timestamp but got {type(value).__name__}"
        )

    raise EqualityDeleteError(f"Equality key field id {field_id} in {file_path} has unsupported key type {iceberg_type}")


def _get_arrow_column(batch_or_table: Union[pa.RecordBatch, pa.Table], path: Sequence[str]) -> pa.Array:
    """Read a (possibly nested) column by its field name path."""
    array = batch_or_table.column(path[0])
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    for name in path[1:]:
        array = array.field(name)
    return array


def _key_tuples_from_arrow(
    batch_or_table: Union[pa.RecordBatch, pa.Table],
    key_fields: Sequence[_KeyField],
    file_path: str,
) -> List[Tuple[Any, ...]]:
    """Extract normalized key tuples from an Arrow batch or table."""
    columns: List[List[Any]] = []
    for key_field in key_fields:
        array = _get_arrow_column(batch_or_table, key_field.arrow_path)
        # to_pylist gives us native Python values: ints stay ints, decimals
        # stay decimal.Decimal, dates/times/timestamps become datetime types
        # and nulls become None (including a null parent struct, whose leaves
        # are also null).
        values = array.to_pylist()
        columns.append([_normalize_primitive(value, key_field.iceberg_type, file_path, key_field.field_id) for value in values])

    row_count = batch_or_table.num_rows
    return [tuple(column[row] for column in columns) for row in range(row_count)]


def _types_are_compatible(data_type: PrimitiveType, delete_type: PrimitiveType) -> bool:
    """Check whether the data column type can be matched against a delete key.

    The legal Iceberg promotions are int -> long and a wider decimal precision
    or scale.  Everything else has to be the exact same primitive type.
    """
    if type(data_type) is type(delete_type):
        if isinstance(data_type, DecimalType):
            # Same primitive family; the numeric normalization below makes
            # values with different precision/scale compare correctly.
            return True
        return True
    if isinstance(data_type, LongType) and isinstance(delete_type, IntegerType):
        return True
    if isinstance(data_type, IntegerType) and isinstance(delete_type, LongType):
        # The reverse promotion does not occur in a valid table, but accepting
        # it on read keeps matching semantics consistent when both files are
        # physical int columns.
        return True
    if isinstance(data_type, DecimalType) and isinstance(delete_type, DecimalType):
        return True
    return False


class EqualityDeleteFilter:
    """Loads equality delete files once and filters Arrow batches per task.

    One filter instance is created per scan (per :class:`ArrowScan`).  Delete
    files are loaded lazily on first use, so a scan that finishes early (for
    example because of a limit) never reads delete files it did not need.
    Each delete file is read at most once for the whole scan, even when it is
    shared by many data files.
    """

    def __init__(self, table_metadata: "TableMetadata", io: "FileIO") -> None:
        self._table_metadata = table_metadata
        self._io = io
        self._cache: Dict[str, _PreparedEqualityDelete] = {}
        self._loaded_files: Set[str] = set()
        self._lock = threading.Lock()

    @property
    def loaded_file_paths(self) -> Set[str]:
        """Paths of the delete files that have actually been fully read."""
        with self._lock:
            return set(self._loaded_files)

    def _resolve_key_fields(
        self,
        data_file: DataFile,
        equality_ids: Tuple[int, ...],
        arrow_schema: pa.Schema,
    ) -> Tuple[_KeyField, ...]:
        leaf_index = _index_arrow_leaves(arrow_schema)
        key_fields: List[_KeyField] = []
        for field_id in equality_ids:
            iceberg_field = _find_key_field_in_schemas(field_id, self._table_metadata)
            if iceberg_field is None:
                raise EqualityDeleteError(
                    f"Equality delete file {data_file.file_path} references unknown field id {field_id} "
                    "that cannot be found in the table schema or its schema history"
                )
            _assert_primitive_key(iceberg_field, data_file)

            arrow_entry = leaf_index.get(field_id)
            if arrow_entry is None:
                raise EqualityDeleteError(
                    f"Equality delete file {data_file.file_path} requires key field id {field_id} "
                    f"('{iceberg_field.name}') but the column is missing from its physical Parquet schema"
                )
            arrow_field, arrow_path = arrow_entry

            if (
                pa.types.is_list(arrow_field.type)
                or pa.types.is_large_list(arrow_field.type)
                or pa.types.is_map(arrow_field.type)
            ):
                raise EqualityDeleteError(
                    f"Equality delete file {data_file.file_path} uses field id {field_id} inside a list or map, "
                    "which is not supported"
                )
            if pa.types.is_struct(arrow_field.type):
                raise EqualityDeleteError(
                    f"Equality delete file {data_file.file_path} uses struct field id {field_id} directly, "
                    "only primitive struct leaves are supported"
                )

            key_fields.append(
                _KeyField(
                    field_id=field_id,
                    iceberg_type=iceberg_field.field_type,  # type: ignore[arg-type]
                    arrow_path=arrow_path,
                )
            )
        return tuple(key_fields)

    def _load(self, data_file: DataFile) -> _PreparedEqualityDelete:
        with self._lock:
            cached = self._cache.get(data_file.file_path)
            if cached is not None:
                return cached

        # The Parquet read happens outside the lock so that distinct delete
        # files can still be read concurrently; the result is published under
        # the lock at the end.  If two threads race on the same file the
        # duplicate read is harmless because both produce the same keys.
        equality_ids = _validate_equality_ids(data_file)
        try:
            from pyiceberg.io.pyarrow import _fs_from_file_path

            _, _, path = _parse_location_for_io(data_file.file_path)
            fs = _fs_from_file_path(self._io, data_file.file_path)
            arrow_table = pq.read_table(path, filesystem=fs)
        except EqualityDeleteError:
            raise
        except Exception as e:
            raise EqualityDeleteError(f"Failed to read equality delete file {data_file.file_path}: {e}") from e

        key_fields = self._resolve_key_fields(data_file, equality_ids, arrow_table.schema)

        try:
            key_tuples = _key_tuples_from_arrow(arrow_table, key_fields, data_file.file_path)
        except EqualityDeleteError:
            raise
        except Exception as e:
            raise EqualityDeleteError(f"Failed to interpret equality delete file {data_file.file_path}: {e}") from e

        # De-duplicate: the same key appearing more than once in a delete file
        # must only produce a single deletion effect.
        prepared = _PreparedEqualityDelete(
            data_file=data_file,
            equality_ids=equality_ids,
            key_fields=key_fields,
            keys=set(key_tuples),
        )
        with self._lock:
            existing = self._cache.get(data_file.file_path)
            if existing is not None:
                return existing
            self._cache[data_file.file_path] = prepared
            self._loaded_files.add(data_file.file_path)
        return prepared

    def prepare(self, delete_files: Iterable[DataFile]) -> List[_PreparedEqualityDelete]:
        return [self._load(delete_file) for delete_file in delete_files]

    def mask_for_batch(
        self,
        batch: pa.RecordBatch,
        data_file_path: str,
        prepared_deletes: Sequence[_PreparedEqualityDelete],
    ) -> pa.Array:
        """Return a boolean mask of rows that survive all equality deletes.

        A row is removed as soon as all key components of one applicable delete
        file match.  Different key sets are combined with OR semantics.
        """
        if not prepared_deletes:
            return pa.array([True] * batch.num_rows, type=pa.bool_())

        # Resolve the key fields against the physical data file on every task,
        # because different data files may have different (older) schemas.
        physical_leaves = _index_arrow_leaves(batch.schema)
        deleted = [False] * batch.num_rows

        for prepared in prepared_deletes:
            data_key_fields = self._resolve_data_key_fields(prepared, physical_leaves, data_file_path)
            row_keys = _key_tuples_from_arrow_with_missing_nulls(batch, data_key_fields, data_file_path)
            delete_keys = prepared.keys
            for index, row_key in enumerate(row_keys):
                if not deleted[index] and row_key in delete_keys:
                    deleted[index] = True

        return pc.invert(pa.array(deleted, type=pa.bool_()))

    def _resolve_data_key_fields(
        self,
        prepared: _PreparedEqualityDelete,
        physical_leaves: Dict[int, Tuple[pa.Field, Tuple[str, ...]]],
        data_file_path: str,
    ) -> Tuple[_KeyField, ...]:
        resolved: List[_KeyField] = []
        for delete_key_field in prepared.key_fields:
            field_id = delete_key_field.field_id
            iceberg_field = _find_key_field_in_schemas(field_id, self._table_metadata)
            if iceberg_field is None:
                # The data file cannot contain a field the table never knew.
                raise EqualityDeleteError(
                    f"Cannot apply equality delete file {prepared.data_file.file_path} to data file "
                    f"{data_file_path}: field id {field_id} is unknown to the table"
                )
            _assert_primitive_key(iceberg_field, prepared.data_file)

            if not _types_are_compatible(iceberg_field.field_type, delete_key_field.iceberg_type):  # type: ignore[arg-type]
                raise EqualityDeleteError(
                    f"Cannot apply equality delete file {prepared.data_file.file_path} to data file "
                    f"{data_file_path}: key field id {field_id} has incompatible types "
                    f"{iceberg_field.field_type} (data) and {delete_key_field.iceberg_type} (delete)"
                )

            arrow_entry = physical_leaves.get(field_id)
            if arrow_entry is None:
                # A column added after this data file was written is projected
                # as null by the v2 column projection rules.  An all-null key
                # column can therefore only match null keys.
                resolved.append(
                    _KeyField(
                        field_id=field_id,
                        iceberg_type=delete_key_field.iceberg_type,
                        arrow_path=(),  # marks a missing, hence null, column
                    )
                )
                continue

            arrow_field, arrow_path = arrow_entry
            if (
                pa.types.is_list(arrow_field.type)
                or pa.types.is_large_list(arrow_field.type)
                or pa.types.is_map(arrow_field.type)
            ):
                raise EqualityDeleteError(
                    f"Data file {data_file_path} contains key field id {field_id} inside a list or map, " "which is not supported"
                )
            if pa.types.is_struct(arrow_field.type):
                raise EqualityDeleteError(
                    f"Data file {data_file_path} contains struct key field id {field_id} directly, "
                    "only primitive struct leaves are supported"
                )

            resolved.append(
                _KeyField(
                    field_id=field_id,
                    iceberg_type=delete_key_field.iceberg_type,
                    arrow_path=arrow_path,
                )
            )
        return tuple(resolved)

    def close(self) -> None:
        """Release the materialized delete keys held by this filter."""
        with self._lock:
            self._cache.clear()
            self._loaded_files.clear()


def _key_tuples_from_arrow_with_missing_nulls(
    batch: pa.RecordBatch,
    key_fields: Sequence[_KeyField],
    file_path: str,
) -> List[Tuple[Any, ...]]:
    """Variant that treats missing columns (empty path) as all-null."""
    columns: List[List[Any]] = []
    for key_field in key_fields:
        if not key_field.arrow_path:
            columns.append([None] * batch.num_rows)
            continue
        array = _get_arrow_column(batch, key_field.arrow_path)
        values = array.to_pylist()
        columns.append([_normalize_primitive(value, key_field.iceberg_type, file_path, key_field.field_id) for value in values])
    return [tuple(column[row] for column in columns) for row in range(batch.num_rows)]


def _parse_location_for_io(file_path: str) -> Tuple[Any, Any, str]:
    from pyiceberg.io.pyarrow import PyArrowFileIO

    return PyArrowFileIO.parse_location(file_path)


def split_delete_files(delete_files: Iterable[DataFile]) -> Tuple[List[DataFile], List[DataFile]]:
    """Partition the delete files of a scan task into positional and equality."""
    positional: List[DataFile] = []
    equality: List[DataFile] = []
    for delete_file in delete_files:
        if delete_file.content == DataFileContent.POSITION_DELETES:
            positional.append(delete_file)
        elif delete_file.content == DataFileContent.EQUALITY_DELETES:
            equality.append(delete_file)
        else:
            raise EqualityDeleteError(f"Unsupported delete file content {delete_file.content} for {delete_file.file_path}")
    return positional, equality
