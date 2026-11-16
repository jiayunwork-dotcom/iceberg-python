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
"""Decimal literal binding: one numeric value, many spellings.

The expected values in these tests are written out by hand (as ``Decimal``
tuples) and are not produced by the code under test.
"""

from decimal import Decimal

import pytest

from pyiceberg.expressions import (
    And,
    BooleanExpression,
    EqualTo,
    GreaterThan,
    GreaterThanOrEqual,
    In,
    LessThan,
    LessThanOrEqual,
    Not,
    NotEqualTo,
    NotIn,
    Or,
)
from pyiceberg.expressions.visitors import bind, expression_evaluator
from pyiceberg.schema import Schema
from pyiceberg.typedef import Record
from pyiceberg.types import DecimalType, LongType, NestedField

AMOUNT = DecimalType(6, 2)
CANONICAL_100_00 = Decimal("100.00")
CANONICAL_100_00_TUPLE = (0, (1, 0, 0, 0, 0), -2)

SCHEMA = Schema(
    NestedField(1, "amount", AMOUNT, required=False),
    NestedField(2, "id", LongType(), required=False),
)

# Every spelling below has the same numeric value as 100.00.
SAME_VALUE_SPELLINGS = [
    Decimal("100.00"),
    Decimal("1E+2"),
    Decimal("100"),
    Decimal("100.000"),
    Decimal("1.0E+2"),
    Decimal("0.1E+3"),
    "100.00",
    "1E+2",
    "100",
    "100.000",
    100,
]

ZERO_SPELLINGS = [
    Decimal("0.00"),
    Decimal("0"),
    Decimal("-0"),
    Decimal("-0.00"),
    Decimal("0E+5"),
    "0.00",
    "-0",
    0,
]

PREDICATE_FACTORIES = [
    lambda value: EqualTo("amount", value),
    lambda value: NotEqualTo("amount", value),
    lambda value: LessThan("amount", value),
    lambda value: LessThanOrEqual("amount", value),
    lambda value: GreaterThan("amount", value),
    lambda value: GreaterThanOrEqual("amount", value),
    lambda value: In("amount", [value, Decimal("1.00")]),
    lambda value: NotIn("amount", [value, Decimal("1.00")]),
]


def _canonical_values(bound):
    if hasattr(bound, "literal"):
        return {bound.literal.value}
    return {literal.value for literal in bound.literals}


@pytest.mark.parametrize("value", SAME_VALUE_SPELLINGS)
@pytest.mark.parametrize("factory", PREDICATE_FACTORIES)
def test_bound_decimal_value_is_canonical(factory, value):
    bound = bind(SCHEMA, factory(value), case_sensitive=True)
    values = _canonical_values(bound)
    assert CANONICAL_100_00 in values
    assert all(candidate.as_tuple()[-1] == -2 for candidate in values)


@pytest.mark.parametrize("value", ZERO_SPELLINGS)
def test_zero_spellings_bind_to_same_value(value):
    bound = bind(SCHEMA, EqualTo("amount", value), case_sensitive=True)
    assert bound.literal.value == Decimal("0.00")
    assert bound.literal.value.as_tuple() == (0, (0,), -2)


def test_set_predicates_deduplicate_same_value_spellings():
    bound = bind(SCHEMA, In("amount", [Decimal("1E+2"), Decimal("100.00")]), case_sensitive=True)
    # Both spellings collapse to a single value, so In becomes an equality.
    assert isinstance(bound, type(bind(SCHEMA, EqualTo("amount", Decimal("100.00")), True)))
    assert bound.literal.value.as_tuple() == CANONICAL_100_00_TUPLE


ROWS = [
    Record(amount=Decimal("99.99"), id=1),
    Record(amount=Decimal("100.00"), id=2),
    Record(amount=Decimal("100.01"), id=3),
    Record(amount=None, id=4),
]

# (expression template with placeholders, expected ids). ``None`` placeholders
# are replaced with the spelling under test; the same numeric value as 100.00
# is also used literally so the expected row set can be written by hand.
EVALUATION_CASES = [
    (lambda v: EqualTo("amount", v), [2]),
    (lambda v: NotEqualTo("amount", v), [1, 3, 4]),
    (lambda v: LessThan("amount", v), [1]),
    (lambda v: LessThanOrEqual("amount", v), [1, 2]),
    (lambda v: GreaterThan("amount", v), [3]),
    (lambda v: GreaterThanOrEqual("amount", v), [2, 3]),
    (lambda v: In("amount", [v, Decimal("99.99")]), [1, 2]),
    (lambda v: NotIn("amount", [v]), [1, 3, 4]),
    (lambda v: And(GreaterThan("amount", Decimal("99.99")), LessThanOrEqual("amount", v)), [2]),
    (lambda v: Or(EqualTo("amount", v), EqualTo("amount", Decimal("99.99"))), [1, 2]),
    (lambda v: Not(EqualTo("amount", v)), [1, 3, 4]),
]


@pytest.mark.parametrize("spelling", SAME_VALUE_SPELLINGS)
def test_same_value_spellings_evaluate_identically(spelling):
    for factory, expected in EVALUATION_CASES:
        expr = factory(spelling)
        assert isinstance(expr, BooleanExpression)
        evaluator = expression_evaluator(SCHEMA, expr, case_sensitive=True)
        assert [row.id for row in ROWS if evaluator(row)] == expected


@pytest.mark.parametrize(
    "value",
    [
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        Decimal("100.001"),
        Decimal("100.0001"),
        Decimal("10000.00"),
        Decimal("1E+4"),
        "NaN",
        "Infinity",
        "abc",
        "100.001",
        "10000.00",
        10000,
        -10000,
    ],
)
def test_invalid_decimal_spellings_are_rejected(value):
    with pytest.raises(ValueError):
        bind(SCHEMA, EqualTo("amount", value), case_sensitive=True)


def test_non_decimal_input_keeps_original_behaviour():
    # A bool is not treated as a decimal; it keeps the previous TypeError.
    with pytest.raises(TypeError):
        bind(SCHEMA, EqualTo("amount", True), case_sensitive=True)
    # A float is still rounded half-up into the declared scale.
    assert bind(SCHEMA, EqualTo("amount", 100.006), case_sensitive=True).literal.value == Decimal("100.01")
