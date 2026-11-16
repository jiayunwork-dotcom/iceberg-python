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
"""Conditional, copy-on-write updates of the rows that are currently alive in a table.

The entry point is :func:`update_rows`, which is wired up as
``Table.update_rows`` / ``Transaction.update_rows``.  The implementation
deliberately reuses the regular read path for the files that the scan
planning keeps, and the regular Arrow writer for the replacement files:

1. every assignment and the row filter are validated against the table
   schema *before* any data file is opened;
2. ``DataScan.plan_files`` prunes the data files that cannot contain matching
   rows using the manifest, partition and metrics metadata.  Those files are
   never opened nor rewritten;
3. every remaining data file is read through ``ArrowScan`` with its positional
   and equality deletes applied, so rows that were already deleted can never be
   revived;
4. rows are matched, all new values are computed from the pre-update rows and
   the surviving rows of a rewritten file are written back with the *current*
   partition spec.  A rewritten file is replaced by its new files, so partition
   source changes physically move rows to the new partition;
5. a single OVERWRITE snapshot is staged that deletes the old data files and
   adds the new ones.  If no value actually changes, nothing is staged at all.
"""

from __future__ import annotations

import datetime
import functools
import math
import struct
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple, cast

from pyiceberg.expressions import (
    AlwaysTrue,
    BooleanExpression,
    ColumnRef,
)
from pyiceberg.expressions.visitors import bind
from pyiceberg.manifest import DataFile, FileFormat
from pyiceberg.schema import Schema
from pyiceberg.types import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FixedType,
    FloatType,
    IcebergType,
    IntegerType,
    LongType,
    NestedField,
    PrimitiveType,
    StringType,
    TimestampType,
    TimestamptzType,
    TimeType,
    UUIDType,
)

if TYPE_CHECKING:
    from pyiceberg.table import Transaction


class UpdateRowsError(ValueError):
    """Raised when an update cannot be applied safely."""


ALWAYS_TRUE = AlwaysTrue()


@dataclass(frozen=True)
class _Assignment:
    """A validated assignment for one top-level field."""

    target: NestedField
    source: NestedField | None = None
    constant: Any = None
    is_constant: bool = True

    @property
    def name(self) -> str:
        return self.target.name


# ---------------------------------------------------------------------------
# Field / type / value validation
# ---------------------------------------------------------------------------


def _matching_top_level_fields(schema: Schema, name: str, case_sensitive: bool) -> List[NestedField]:
    if case_sensitive:
        return [field for field in schema.fields if field.name == name]
    return [field for field in schema.fields if field.name.lower() == name.lower()]


def _resolve_top_level_field(schema: Schema, name: str, case_sensitive: bool, role: str) -> NestedField:
    if not isinstance(name, str) or not name:
        raise UpdateRowsError(f"{role} must be a non-empty field name string, got: {name!r}")

    matches = _matching_top_level_fields(schema, name, case_sensitive)

    if not matches:
        # Distinguish "no such field" from "nested field of that name": both are
        # rejected, but the message should say which rule was violated.
        lowered = name.lower()
        nested = any(
            (key == name) if case_sensitive else (key.lower() == lowered)
            for key in schema._name_to_id  # pylint: disable=protected-access
        )
        if nested:
            raise UpdateRowsError(f"{role} {name!r} is not a top-level field; update_rows only supports top-level atomic fields")
        raise UpdateRowsError(f"Could not find field with name {name!r}, case_sensitive={case_sensitive}")

    if len(matches) > 1:
        names = ", ".join(repr(field.name) for field in matches)
        raise UpdateRowsError(f"{role} {name!r} is ambiguous with case_sensitive={case_sensitive}; it matches fields {names}")

    field = matches[0]
    if not isinstance(field.field_type, PrimitiveType):
        raise UpdateRowsError(
            f"{role} {name!r} has non-atomic type {field.field_type}; only top-level atomic fields are supported"
        )

    return field


def _can_assign(source: IcebergType, target: IcebergType) -> bool:
    """Return whether the source value can be written to the target without loss."""
    if source == target:
        return True
    if isinstance(source, IntegerType) and isinstance(target, LongType):
        return True
    if isinstance(source, FloatType) and isinstance(target, DoubleType):
        return True
    if isinstance(source, DecimalType) and isinstance(target, DecimalType):
        return target.precision >= source.precision and target.scale == source.scale
    return False


def _arrow_type(field_type: PrimitiveType) -> Any:
    from pyiceberg.io.pyarrow import schema_to_pyarrow

    return schema_to_pyarrow(field_type, include_field_ids=False)


def _invalid(field: NestedField, value: Any, reason: str) -> UpdateRowsError:
    return UpdateRowsError(f"Cannot assign {value!r} to field {field.name!r} of type {field.field_type}: {reason}")


def _float32_is_exact(value: float) -> bool:
    if math.isnan(value) or math.isinf(value):
        return True
    try:
        packed = struct.pack("<f", value)
    except OverflowError:
        return False
    return struct.unpack("<f", packed)[0] == value


def _decimal_is_exact(value: Decimal, field_type: DecimalType) -> bool:
    if not value.is_finite():
        return False
    unscaled = int(value.scaleb(field_type.scale))
    if Decimal(unscaled).scaleb(-field_type.scale) != value:
        return False
    return len(str(abs(unscaled))) <= field_type.precision


def _coerce_constant(field: NestedField, value: Any) -> Any:
    """Validate a Python scalar against the declared Iceberg type.

    The value must be accepted *exactly*: no rounding, truncation or range
    clamping is allowed.  The returned value is the Python value that will be
    handed to PyArrow.
    """
    field_type = field.field_type
    if value is None:
        if field.required:
            raise UpdateRowsError(f"Cannot assign null to required field {field.name!r}")
        return None

    if isinstance(field_type, BooleanType):
        if not isinstance(value, bool):
            raise _invalid(field, value, "expected a bool")
        return value

    if isinstance(field_type, (IntegerType, LongType)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise _invalid(field, value, "expected an int")
        if not (field_type.min <= value <= field_type.max):
            raise _invalid(field, value, f"outside the range [{field_type.min}, {field_type.max}]")
        return value

    if isinstance(field_type, (FloatType, DoubleType)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _invalid(field, value, "expected a float")
        as_float = float(value)
        if isinstance(value, int) and as_float != value:
            raise _invalid(field, value, "not exactly representable as a double")
        if isinstance(field_type, FloatType) and not _float32_is_exact(as_float):
            raise _invalid(field, value, "not exactly representable as a 32-bit float")
        return as_float

    if isinstance(field_type, StringType):
        if not isinstance(value, str):
            raise _invalid(field, value, "expected a str")
        return value

    if isinstance(field_type, FixedType):
        if not isinstance(value, (bytes, bytearray)):
            raise _invalid(field, value, "expected bytes")
        if len(value) != len(field_type):
            raise _invalid(field, value, f"expected {len(field_type)} bytes, got {len(value)}")
        return bytes(value)

    if isinstance(field_type, BinaryType):
        if not isinstance(value, (bytes, bytearray)):
            raise _invalid(field, value, "expected bytes")
        return bytes(value)

    if isinstance(field_type, UUIDType):
        if isinstance(value, uuid.UUID):
            return value.bytes
        if isinstance(value, (bytes, bytearray)) and len(value) == 16:
            return bytes(value)
        raise _invalid(field, value, "expected a uuid.UUID or 16 bytes")

    if isinstance(field_type, DecimalType):
        if not isinstance(value, Decimal):
            raise _invalid(field, value, "expected a decimal.Decimal")
        if not _decimal_is_exact(value, field_type):
            raise _invalid(field, value, f"not exactly representable as {field_type}")
        scale = field_type.scale
        unscaled = int(value.scaleb(scale))
        return Decimal(unscaled).scaleb(-scale)

    if isinstance(field_type, DateType):
        if isinstance(value, datetime.datetime) or not isinstance(value, datetime.date):
            raise _invalid(field, value, "expected a datetime.date")
        return value

    if isinstance(field_type, TimeType):
        if not isinstance(value, datetime.time) or value.tzinfo is not None:
            raise _invalid(field, value, "expected a naive datetime.time")
        return value

    if isinstance(field_type, TimestampType):
        if not isinstance(value, datetime.datetime) or value.tzinfo is not None:
            raise _invalid(field, value, "expected a naive datetime.datetime")
        return value

    if isinstance(field_type, TimestamptzType):
        if not isinstance(value, datetime.datetime) or value.tzinfo is None:
            raise _invalid(field, value, "expected a timezone-aware datetime.datetime")
        return value

    raise _invalid(field, value, f"unsupported type {field_type}")


def _make_scalar(field: NestedField, value: Any) -> Any:
    import pyarrow as pa

    try:
        return pa.scalar(value, type=_arrow_type(field.field_type))  # type: ignore[arg-type]
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise _invalid(field, value, str(exc)) from exc


def _validate_assignments(schema: Schema, assignments: Mapping[str, Any], case_sensitive: bool) -> List[_Assignment]:
    if not isinstance(assignments, Mapping):
        raise UpdateRowsError(f"assignments must be a mapping, got: {type(assignments).__name__}")
    if len(assignments) == 0:
        raise UpdateRowsError("assignments must not be empty")

    validated: List[_Assignment] = []
    seen_targets: Dict[int, str] = {}

    for raw_name, raw_value in assignments.items():
        if not isinstance(raw_name, str):
            raise UpdateRowsError(f"Assignment keys must be field name strings, got: {raw_name!r}")

        target = _resolve_top_level_field(schema, raw_name, case_sensitive, "Assignment target")
        if target.field_id in seen_targets:
            raise UpdateRowsError(
                f"Assignments {seen_targets[target.field_id]!r} and {raw_name!r} both resolve to field {target.name!r}"
            )
        seen_targets[target.field_id] = raw_name

        if isinstance(raw_value, ColumnRef):
            source = _resolve_top_level_field(schema, raw_value.name, case_sensitive, "ColumnRef source")
            if not _can_assign(source.field_type, target.field_type):
                raise UpdateRowsError(
                    f"Cannot assign field {source.name!r} of type {source.field_type} to field "
                    f"{target.name!r} of type {target.field_type}: not an identical type or a lossless promotion"
                )
            validated.append(_Assignment(target=target, source=source, is_constant=False))
        else:
            constant = _coerce_constant(target, raw_value)
            # Build the scalar eagerly so that any Arrow-level rejection is
            # reported even when no row matches the filter.
            _make_scalar(target, constant)
            validated.append(_Assignment(target=target, constant=constant, is_constant=True))

    return validated


def _parse_and_bind_filter(schema: Schema, row_filter: Any, case_sensitive: bool) -> Tuple[BooleanExpression, BooleanExpression]:
    """Return the unbound (for scan planning) and bound (for Arrow) filter."""
    from pyiceberg.table import _parse_row_filter  # pylint: disable=protected-access

    if not isinstance(row_filter, (str, BooleanExpression)):
        raise UpdateRowsError(f"row_filter must be a string or a BooleanExpression, got: {type(row_filter).__name__}")
    try:
        parsed = _parse_row_filter(row_filter)
        bound = bind(schema, parsed, case_sensitive)
    except (TypeError, ValueError) as exc:
        raise UpdateRowsError(f"Invalid row_filter: {exc}") from exc
    return parsed, bound


# ---------------------------------------------------------------------------
# Arrow helpers
# ---------------------------------------------------------------------------


def _field_id_paths(schema: Schema) -> Dict[int, Tuple[str, ...]]:
    """Map every struct field id to its path in the projected Arrow table."""
    from pyiceberg.types import ListType, MapType
    from pyiceberg.types import StructType as NestedStructType

    paths: Dict[int, Tuple[str, ...]] = {}

    def walk(struct: NestedStructType, prefix: Tuple[str, ...]) -> None:
        for field in struct.fields:
            path = prefix + (field.name,)
            paths[field.field_id] = path
            field_type = field.field_type
            if isinstance(field_type, NestedStructType):
                walk(field_type, path)
            elif isinstance(field_type, ListType):
                element = field_type.element_field
                paths[element.field_id] = path + (element.name,)
                if isinstance(element.field_type, NestedStructType):
                    walk(element.field_type, path + (element.name,))
            elif isinstance(field_type, MapType):
                key = field_type.key_field
                value = field_type.value_field
                paths[key.field_id] = path + (key.name,)
                paths[value.field_id] = path + (value.name,)
                if isinstance(value.field_type, NestedStructType):
                    walk(value.field_type, path + (value.name,))

    walk(schema.as_struct(), ())
    return paths


def _arrow_match_expression(bound_filter: BooleanExpression, schema: Schema) -> Any:
    from pyiceberg.io.pyarrow import expression_to_pyarrow

    return expression_to_pyarrow(bound_filter, field_paths=_field_id_paths(schema))


def _same_values(old: Any, new: Any, field_type: PrimitiveType) -> Any:
    """Return a boolean array that is true where old and new are the same value.

    ``None``/``None``, ``NaN``/``NaN`` and ``+0.0``/``-0.0`` are considered
    equal, matching the comparison rules documented for ``update_rows``.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    old_is_null = pc.is_null(old)
    new_is_null = pc.is_null(new)
    either_null = pc.or_(old_is_null, new_is_null)
    both_null = pc.and_(old_is_null, new_is_null)
    not_null = pc.invert(either_null)

    equal = pc.fill_null(pc.equal(old, new), False)
    same = pc.or_(both_null, pc.and_(not_null, equal))

    if isinstance(field_type, (FloatType, DoubleType)):
        both_nan = pc.fill_null(pc.and_(pc.is_nan(old), pc.is_nan(new)), False)
        same = pc.or_(same, pc.and_(not_null, both_nan))

    return pc.fill_null(pc.cast(same, pa.bool_()), False)


def _new_column(table: Any, assignment: _Assignment) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    if assignment.is_constant:
        scalar = _make_scalar(assignment.target, assignment.constant)
        return pa.repeat(scalar, table.num_rows)

    source = assignment.source
    assert source is not None
    column = table.column(source.name)
    target_type = _arrow_type(cast(PrimitiveType, assignment.target.field_type))
    try:
        if column.type == target_type:
            return column
        return pc.cast(column, target_type, safe=True)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise UpdateRowsError(f"Cannot read field {source.name!r} for assignment to {assignment.target.name!r}: {exc}") from exc


def _check_required_results(table: Any, assignments: Iterable[_Assignment], changed_mask: Any) -> None:
    """Reject nulls that a ColumnRef produced for a required target field."""
    import pyarrow.compute as pc

    changed_rows = table.filter(changed_mask)
    for assignment in assignments:
        if not assignment.target.required or assignment.is_constant:
            continue
        source = assignment.source
        assert source is not None
        column = changed_rows.column(assignment.name)
        if pc.any(pc.is_null(column)).as_py():
            raise UpdateRowsError(f"ColumnRef {source.name!r} produced null values for required field {assignment.target.name!r}")


# ---------------------------------------------------------------------------
# Core operation
# ---------------------------------------------------------------------------


def _read_task_rows(transaction: Transaction, task: Any) -> Any:
    from pyiceberg.io.pyarrow import ArrowScan

    metadata = transaction.table_metadata
    return ArrowScan(
        table_metadata=metadata,
        io=transaction._table.io,  # pylint: disable=protected-access
        projected_schema=metadata.schema(),
        row_filter=AlwaysTrue(),
        case_sensitive=True,
    ).to_table(tasks=[task])


def _rewrite_task(
    old_table: Any,
    match_expression: Any,
    assignments: List[_Assignment],
    index_column: str,
) -> Tuple[Any, int]:
    """Return the replacement table for one data file and the number of changed rows.

    If no row changes, the (unchanged) old table is returned with a count of 0
    and the caller keeps the original file untouched.  Otherwise every alive row
    of the file is kept in its original order and multiplicity; only the
    matching rows whose values actually change get the new values.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    indexed = old_table.append_column(index_column, pa.array(range(old_table.num_rows), type=pa.int64()))

    # ``Table.filter`` gives exactly the same Arrow semantics (nulls never
    # match) as the read path, so use it both to short-circuit and to obtain
    # the physical row numbers of the matching rows.
    matched_indices = indexed.filter(match_expression).column(index_column)
    if len(matched_indices) == 0:
        return old_table, 0

    matched_mask = pc.is_in(indexed.column(index_column), value_set=matched_indices)

    # All right-hand sides are computed from the pre-update rows; the columns
    # are only selected into the output where a row actually changed.
    new_columns = {}
    same_per_assignment = []
    for assignment in assignments:
        new_column = _new_column(indexed, assignment)
        new_columns[assignment.name] = new_column
        same_per_assignment.append(
            _same_values(indexed.column(assignment.name), new_column, cast(PrimitiveType, assignment.target.field_type))
        )

    same_mask = functools.reduce(pc.and_, same_per_assignment)
    changed_mask = pc.fill_null(pc.and_(matched_mask, pc.invert(same_mask)), False)
    changed_count = int(pc.sum(pc.cast(changed_mask, pa.int64())).as_py() or 0)
    if changed_count == 0:
        return old_table, 0

    result = indexed
    for name, column in new_columns.items():
        result = result.set_column(
            result.schema.get_field_index(name),
            name,
            pc.if_else(changed_mask, column, result.column(name)),
        )
    _check_required_results(result, assignments, changed_mask)
    result = result.drop_columns([index_column])

    return result, changed_count


def update_rows(
    transaction: Transaction,
    assignments: Mapping[str, Any],
    row_filter: Any = ALWAYS_TRUE,
    *,
    case_sensitive: bool = True,
    snapshot_properties: Dict[str, str] | None = None,
) -> int:
    """Apply ``assignments`` to the rows matching ``row_filter`` and return the changed row count."""
    import uuid as uuid_module
    from itertools import count as count_iter

    from pyiceberg.io.pyarrow import _dataframe_to_data_files

    metadata = transaction.table_metadata
    if metadata.format_version != 2:
        raise UpdateRowsError(f"update_rows requires a format-version=2 table, got: {metadata.format_version}")

    snapshot_properties = dict(snapshot_properties or {})
    schema = metadata.schema()

    # Validation happens before any data file is opened, so bad fields, types
    # and constants are reported even when the filter matches nothing.
    validated = _validate_assignments(schema, assignments, case_sensitive)
    parsed_filter, bound_filter = _parse_and_bind_filter(schema, row_filter, case_sensitive)
    match_expression = _arrow_match_expression(bound_filter, schema)

    if transaction._aborted:  # pylint: disable=protected-access
        raise transaction._aborted_error()  # pylint: disable=protected-access

    tasks = list(transaction._scan(row_filter=parsed_filter, case_sensitive=case_sensitive).plan_files())  # pylint: disable=protected-access
    if not tasks:
        return 0

    for task in tasks:
        if task.file.file_format != FileFormat.PARQUET:
            raise UpdateRowsError(f"update_rows only supports Parquet data files, got: {task.file.file_format}")

    index_column = "__pyiceberg_update_row_index__"
    while index_column in schema.column_names:
        index_column += "_"

    rewrites: List[Tuple[DataFile, Any]] = []
    total_changed = 0
    for task in tasks:
        old_table = _read_task_rows(transaction, task)
        replacement, changed = _rewrite_task(old_table, match_expression, validated, index_column)
        if changed == 0:
            continue
        total_changed += changed
        rewrites.append((task.file, replacement))

    if total_changed == 0:
        return 0

    write_uuid = uuid_module.uuid4()
    counter = count_iter(0)
    replacements: List[Tuple[DataFile, List[DataFile]]] = []
    for original, replacement_table in rewrites:
        new_files = list(
            _dataframe_to_data_files(
                table_metadata=metadata,
                df=replacement_table,
                io=transaction._table.io,  # pylint: disable=protected-access
                write_uuid=write_uuid,
                counter=counter,
            )
        )
        if not new_files:  # pragma: no cover - a non-empty replacement always writes a file
            raise UpdateRowsError(f"Failed to write replacement data for {original.file_path}")
        replacements.append((original, new_files))

    with transaction.update_snapshot(snapshot_properties=snapshot_properties).overwrite() as overwrite:
        overwrite.commit_uuid = write_uuid
        for original, new_files in replacements:
            overwrite.delete_data_file(original)
            for new_file in new_files:
                overwrite.append_data_file(new_file)

    return total_changed
