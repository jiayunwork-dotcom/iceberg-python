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
# pylint: disable=redefined-outer-name
"""Unit tests for the equality delete reader internals."""

import datetime
import decimal

import pyarrow as pa
import pytest

from pyiceberg.io.equality_deletes import (
    EqualityDeleteError,
    _index_arrow_leaves,
    _normalize_primitive,
    _types_are_compatible,
)
from pyiceberg.types import (
    BooleanType,
    DateType,
    DecimalType,
    IntegerType,
    ListType,
    LongType,
    StringType,
    TimestampType,
    TimestamptzType,
    TimeType,
)


def test_index_arrow_leaves_top_level_and_struct() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), metadata={b"PARQUET:field_id": "1"}),
            pa.field(
                "loc",
                pa.struct(
                    [
                        pa.field("lat", pa.float64(), metadata={b"PARQUET:field_id": "10"}),
                        pa.field(
                            "inner",
                            pa.struct([pa.field("z", pa.int32(), metadata={b"PARQUET:field_id": "11"})]),
                        ),
                    ]
                ),
                metadata={b"PARQUET:field_id": "9"},
            ),
        ]
    )
    leaves = _index_arrow_leaves(schema)
    assert set(leaves.keys()) == {1, 9, 10, 11}
    assert leaves[10][1] == ("loc", "lat")
    assert leaves[11][1] == ("loc", "inner", "z")


def test_index_arrow_leaves_keeps_list_and_map_for_diagnostics() -> None:
    schema = pa.schema(
        [
            pa.field(
                "tags",
                pa.list_(pa.field("element", pa.string(), metadata={b"PARQUET:field_id": "5"})),
                metadata={b"PARQUET:field_id": "4"},
            ),
        ]
    )
    leaves = _index_arrow_leaves(schema)
    assert 4 in leaves and 5 in leaves
    assert pa.types.is_list(leaves[4][0].type)


def test_normalize_primitive_keeps_integer_precision() -> None:
    big = 10_000_000_000
    assert _normalize_primitive(big, LongType(), "f", 1) == big
    assert _normalize_primitive(5, IntegerType(), "f", 1) == 5
    assert _normalize_primitive(None, LongType(), "f", 1) is None
    with pytest.raises(EqualityDeleteError):
        _normalize_primitive(True, LongType(), "f", 1)


def test_normalize_primitive_decimal_never_float() -> None:
    value = decimal.Decimal("123456789.12345")
    normalized = _normalize_primitive(value, DecimalType(14, 5), "f", 1)
    assert normalized == value and isinstance(normalized, decimal.Decimal)
    assert _normalize_primitive(7, DecimalType(5, 2), "f", 1) == decimal.Decimal("7")
    with pytest.raises(EqualityDeleteError):
        _normalize_primitive(1.5, DecimalType(5, 2), "f", 1)


def test_normalize_primitive_temporal_types() -> None:
    day = datetime.date(2024, 1, 2)
    assert _normalize_primitive(day, DateType(), "f", 1) == day
    moment = datetime.time(10, 11, 12, 345678)
    assert _normalize_primitive(moment, TimeType(), "f", 1) == moment
    ts = datetime.datetime(2024, 1, 2, 3, 4, 5, 678)
    assert _normalize_primitive(ts, TimestampType(), "f", 1) == ts
    aware = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    assert _normalize_primitive(aware, TimestamptzType(), "f", 1) == aware
    with pytest.raises(EqualityDeleteError):
        _normalize_primitive(aware, TimestampType(), "f", 1)
    with pytest.raises(EqualityDeleteError):
        _normalize_primitive(ts, TimestamptzType(), "f", 1)


def test_normalize_primitive_basic_types() -> None:
    assert _normalize_primitive(True, BooleanType(), "f", 1) is True
    assert _normalize_primitive("x", StringType(), "f", 1) == "x"
    with pytest.raises(EqualityDeleteError):
        _normalize_primitive(1, StringType(), "f", 1)
    with pytest.raises(EqualityDeleteError):
        _normalize_primitive("x", BooleanType(), "f", 1)


def test_types_are_compatible_promotions() -> None:
    assert _types_are_compatible(LongType(), IntegerType())
    assert _types_are_compatible(IntegerType(), LongType())
    assert _types_are_compatible(DecimalType(10, 2), DecimalType(12, 2))
    assert _types_are_compatible(StringType(), StringType())
    assert not _types_are_compatible(StringType(), LongType())
    assert not _types_are_compatible(
        LongType(),
        ListType(element_id=1, element_type=StringType()),  # type: ignore
    )
