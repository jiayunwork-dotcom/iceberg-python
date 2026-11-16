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
"""Decimal scale handling: 38-digit values, hostile decimal contexts, transforms.

All expected values are written out explicitly. The tests also verify that
the caller's decimal context (precision, rounding, flags and traps) is left
exactly as it was found.
"""

from decimal import (
    Context,
    Decimal,
    Inexact,
    InvalidOperation,
    Rounded,
    getcontext,
    localcontext,
)

import pyarrow as pa
import pytest

from pyiceberg.conversions import from_bytes, to_bytes
from pyiceberg.expressions import BoundIn, EqualTo, In
from pyiceberg.expressions.visitors import bind
from pyiceberg.schema import Schema
from pyiceberg.transforms import BucketTransform, TruncateTransform
from pyiceberg.types import DecimalType, NestedField
from pyiceberg.utils.decimal import (
    decimal_canonical,
    decimal_encode,
    decimal_from_float,
    decimal_rescale,
    decimal_to_bytes_at_scale,
    decimal_to_integral,
    decimal_to_unscaled,
    truncate_decimal,
    truncate_decimal_at_scale,
    unscaled_to_minimal_bytes,
)

# 38-digit decimal(38, 2) values and equivalent spellings.
BIG_38 = Decimal("123456789012345678901234567890123456.78")
BIG_38_TUPLE = (0, tuple(int(c) for c in "12345678901234567890123456789012345678"), -2)
BIG_38_SCIENTIFIC = Decimal("1.2345678901234567890123456789012345678E+35")
BIG_38_TRAILING = Decimal("123456789012345678901234567890123456.780")

T38 = DecimalType(38, 2)
SCHEMA_38 = Schema(NestedField(1, "amount", T38, required=False))

HOSTILE_CONTEXTS = [
    Context(prec=3),
    Context(prec=3, rounding="ROUND_DOWN"),
    Context(prec=6, rounding="ROUND_UP", traps=[Inexact, Rounded, InvalidOperation]),
    Context(prec=28, traps=[InvalidOperation, Inexact, Rounded]),
]


def _context_snapshot():
    ctx = getcontext()
    return ctx.prec, ctx.rounding, ctx.Emin, ctx.Emax, dict(ctx.flags), dict(ctx.traps)


@pytest.mark.parametrize("context", HOSTILE_CONTEXTS)
def test_binding_is_context_invariant(context):
    with localcontext(context):
        before = _context_snapshot()
        bound = bind(SCHEMA_38, EqualTo("amount", BIG_38_SCIENTIFIC), case_sensitive=True)
        assert bound.literal.value.as_tuple() == BIG_38_TUPLE
        assert decimal_canonical(BIG_38_TRAILING, 38, 2).as_tuple() == BIG_38_TUPLE
        assert decimal_encode(BIG_38_SCIENTIFIC, 38, 2) == int("12345678901234567890123456789012345678")
        assert decimal_to_bytes_at_scale(BIG_38_SCIENTIFIC, 2) == decimal_to_bytes_at_scale(BIG_38, 2)
        assert decimal_to_integral(Decimal("2.5")) == 2
        # 100.005 as a binary float is slightly below the tie, so it rounds down.
        assert decimal_from_float(100.005, 2) == Decimal("100.00")
        assert decimal_from_float(100.006, 2) == Decimal("100.01")
        assert truncate_decimal_at_scale(BIG_38, 10, 2) == Decimal("123456789012345678901234567890123456.70")
        # Nothing leaked into the caller's context.
        assert _context_snapshot() == before
    assert _context_snapshot()[0] == 28  # module default restored by localcontext


def test_38_digit_spellings_are_identical():
    tuples = {decimal_canonical(value, 38, 2).as_tuple() for value in (BIG_38, BIG_38_SCIENTIFIC, BIG_38_TRAILING)}
    assert tuples == {BIG_38_TUPLE}
    assert decimal_canonical(BIG_38, 38, 2) == BIG_38


def test_38_digit_precision_overflow_rejected():
    too_big = Decimal("1234567890123456789012345678901234567.89")  # 39 digits
    with pytest.raises(ValueError):
        decimal_canonical(too_big, 38, 2)
    with pytest.raises(ValueError):
        bind(SCHEMA_38, EqualTo("amount", too_big), case_sensitive=True)


def test_38_digit_roundtrip_bytes():
    encoded = to_bytes(T38, BIG_38)
    assert len(encoded) == 16
    assert from_bytes(T38, encoded) == BIG_38
    # Every spelling encodes to identical bytes.
    assert to_bytes(T38, BIG_38_SCIENTIFIC) == encoded
    assert to_bytes(T38, Decimal("123456789012345678901234567890123456.780")) == encoded


@pytest.mark.parametrize(
    "value, expected",
    [
        (Decimal("100.00"), Decimal("100.00")),
        (Decimal("1E+2"), Decimal("100.00")),
        (Decimal("100"), Decimal("100.00")),
        (Decimal("100.000"), Decimal("100.00")),
        (Decimal("-0"), Decimal("0.00")),
        (Decimal("-0.00"), Decimal("0.00")),
        (Decimal("-12.34"), Decimal("-12.40")),
        (Decimal("-0.05"), Decimal("-0.10")),
        (Decimal("0E+5"), Decimal("0.00")),
    ],
)
def test_truncate_at_declared_scale(value, expected):
    result = truncate_decimal_at_scale(value, 10, 2)
    assert result == expected
    assert result.as_tuple() == expected.as_tuple()


def test_bare_truncate_keeps_own_scale():
    # The bare helper still follows the Decimal's own scale convention.
    assert truncate_decimal(Decimal("12.345"), 10) == Decimal("12.340")
    assert truncate_decimal(Decimal("1E+2"), 10) == Decimal("0.00")


@pytest.mark.parametrize(
    "value, expected",
    [
        (Decimal("32767"), b"\x7f\xff"),
        (Decimal("32768"), b"\x00\x80\x00"),
        (Decimal("-128"), b"\x80"),
        (Decimal("-129"), b"\xff\x7f"),
        (Decimal("127"), b"\x7f"),
        (Decimal("128"), b"\x00\x80"),
        (Decimal("0"), b"\x00"),
    ],
)
def test_minimal_unscaled_bytes(value, expected):
    assert unscaled_to_minimal_bytes(decimal_to_unscaled(value)) == expected


def test_bucket_uses_declared_scale():
    transform = BucketTransform(100)
    decimal_type = DecimalType(9, 2)
    buckets = {transform.transform(decimal_type)(Decimal(spelling)) for spelling in ("100.00", "1E+2", "100", "100.000")}
    assert len(buckets) == 1
    # The bucket value must match hashing the minimal unscaled bytes at scale 2.
    import mmh3

    expected_hash = mmh3.hash(unscaled_to_minimal_bytes(10000))
    assert buckets == {(expected_hash & 0x7FFFFFFF) % 100}

    # Zero and negative zero are the same value.
    assert transform.transform(decimal_type)(Decimal("-0")) == transform.transform(decimal_type)(Decimal("0.00"))


def test_bucket_hashes_zero_instead_of_treating_it_as_null():
    transform = BucketTransform(10).transform(DecimalType(9, 2))
    assert transform(Decimal("0.00")) is not None
    assert transform(None) is None


@pytest.mark.parametrize("scale", [0, 2, 10])
def test_bucket_matches_pyarrow_transform_for_decimals(scale):
    values = [Decimal("0.00"), Decimal("-0.05"), Decimal("100.00"), Decimal("-128.00"), None]
    if scale == 0:
        values = [Decimal("0"), Decimal("-5"), Decimal("100"), Decimal("-128"), None]
    transform = BucketTransform(10)
    decimal_type = DecimalType(9, scale)
    arrow = pa.array(values, type=pa.decimal128(18, scale))
    from_arrow = transform.pyarrow_transform(decimal_type)(arrow).to_pylist()
    expected = [transform.transform(decimal_type)(value) for value in values]
    assert from_arrow == expected


def test_transforms_are_context_invariant():
    with localcontext(Context(prec=3, traps=[Inexact, Rounded, InvalidOperation])):
        before = _context_snapshot()
        bucket = BucketTransform(100).transform(DecimalType(9, 2))
        truncate = TruncateTransform(10).transform(DecimalType(9, 2))
        assert bucket(Decimal("1E+2")) == bucket(Decimal("100.00"))
        assert truncate(Decimal("1E+2")) == truncate(Decimal("100.00"))
        assert _context_snapshot() == before


def test_to_bytes_is_context_invariant():
    decimal_type = DecimalType(9, 2)
    with localcontext(Context(prec=2, traps=[Inexact, Rounded, InvalidOperation])):
        before = _context_snapshot()
        assert to_bytes(decimal_type, Decimal("1E+2")) == to_bytes(decimal_type, Decimal("100.00"))
        assert _context_snapshot() == before


def test_string_literal_binds_exactly():
    assert (
        bind(SCHEMA_38, EqualTo("amount", "123456789012345678901234567890123456.78"), True).literal.value.as_tuple()
        == BIG_38_TUPLE
    )
    assert (
        bind(SCHEMA_38, EqualTo("amount", "1.2345678901234567890123456789012345678E+35"), True).literal.value.as_tuple()
        == BIG_38_TUPLE
    )


def test_in_set_deduplicates_across_spellings():
    bound = bind(SCHEMA_38, In("amount", [BIG_38, BIG_38_SCIENTIFIC, BIG_38_TRAILING, Decimal("1.00")]), True)
    assert isinstance(bound, BoundIn)
    values = {lit.value for lit in bound.literals}
    # The three spellings collapse to one value; 1.00 is the only other.
    assert values == {BIG_38, Decimal("1.00")}
    assert {value.as_tuple() for value in values} == {BIG_38_TUPLE, (0, (1, 0, 0), -2)}


def test_literal_rescale_is_exact_or_rejected():
    assert decimal_rescale(Decimal("1E+2"), 2) == Decimal("100.00")
    with pytest.raises(ValueError):
        decimal_rescale(Decimal("100.001"), 2)
    with pytest.raises(ValueError):
        decimal_rescale(Decimal("NaN"), 2)
    with pytest.raises(ValueError):
        decimal_rescale(Decimal("Infinity"), 2)


def test_invalid_inputs_are_value_errors():
    for bad in (Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity"), Decimal("-Infinity"), "NaN", "Infinity", "abc"):
        with pytest.raises(ValueError):
            bind(SCHEMA_38, EqualTo("amount", bad), True)
