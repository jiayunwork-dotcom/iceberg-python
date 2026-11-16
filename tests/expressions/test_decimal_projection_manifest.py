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
"""Manifest and partition bound evaluation for decimal columns.

Partition bounds are encoded with the declared scale. Queries written with a
different spelling of the same number must produce identical keep/prune
decisions, and the inclusive projection must be conservative.
"""

from decimal import Decimal

import pytest

from pyiceberg.conversions import to_bytes
from pyiceberg.expressions import (
    And,
    EqualTo,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    NotEqualTo,
    Or,
)
from pyiceberg.expressions.visitors import inclusive_projection, manifest_evaluator, strict_projection
from pyiceberg.manifest import ManifestFile, PartitionFieldSummary
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import BucketTransform, IdentityTransform, TruncateTransform
from pyiceberg.types import DecimalType, IntegerType, NestedField
from pyiceberg.utils.decimal import decimal_to_bytes_at_scale

AMOUNT = DecimalType(6, 2)
SCHEMA = Schema(NestedField(1, "amount", AMOUNT, required=False))
SPELLINGS = ["100.00", "1E+2", "100", "100.000", "1.0E+2"]


def _spec(transform, field_id=1000):
    return PartitionSpec(PartitionField(1, field_id, transform, "amount_part"))


def _manifest(lower, upper, contains_null=False, partition_type=AMOUNT):
    def encode(value):
        if value is None:
            return None
        return decimal_to_bytes_at_scale(value, partition_type.scale)

    return _manifest_bounds(encode(lower), encode(upper), contains_null=contains_null)


def _manifest_bounds(lower_bound, upper_bound, contains_null=False):
    return ManifestFile(
        manifest_path="",
        manifest_length=0,
        partition_spec_id=0,
        partitions=(
            PartitionFieldSummary(
                contains_null=contains_null,
                contains_nan=None,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ),
        ),
    )


def _evaluate(spec, expression, manifest):
    partition_filter = inclusive_projection(SCHEMA, spec)(expression)
    return manifest_evaluator(spec, SCHEMA, partition_filter, case_sensitive=True)(manifest)


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_identity_and_truncate_manifest_keep_matching_partition(spelling):
    for transform in (IdentityTransform(), TruncateTransform(10)):
        spec = _spec(transform)
        manifest = _manifest(Decimal("100.00"), Decimal("100.00"))
        assert _evaluate(spec, EqualTo("amount", Decimal(spelling)), manifest) is True


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_bucket_manifest_keeps_matching_partition(spelling):
    transform = BucketTransform(8)
    spec = _spec(transform)
    bucket_value = transform.transform(AMOUNT)(Decimal("100.00"))
    # The bucket partition type is int, so its bounds are encoded as ints.
    bounds = to_bytes(IntegerType(), bucket_value)
    manifest = _manifest_bounds(bounds, bounds)
    assert _evaluate(spec, EqualTo("amount", Decimal(spelling)), manifest) is True


def test_manifest_prunes_far_partition():
    spec = _spec(TruncateTransform(10))
    manifest = _manifest(Decimal("0.00"), Decimal("10.00"))
    for spelling in SPELLINGS:
        assert _evaluate(spec, EqualTo("amount", Decimal(spelling)), manifest) is False
        assert _evaluate(spec, LessThan("amount", Decimal(spelling)), manifest) is True
        assert _evaluate(spec, GreaterThan("amount", Decimal(spelling)), manifest) is False


def test_inclusive_projection_is_spelling_invariant():
    spec = _spec(TruncateTransform(10))
    for expression_type in (EqualTo, NotEqualTo, LessThan, LessThanOrEqual, GreaterThan, GreaterThanOrEqual):
        projections = {
            repr(inclusive_projection(SCHEMA, spec)(expression_type("amount", Decimal(spelling)))) for spelling in SPELLINGS
        }
        assert len(projections) == 1, (expression_type, projections)


def test_strict_projection_is_spelling_invariant():
    spec = _spec(TruncateTransform(10))
    for expression_type in (EqualTo, NotEqualTo, LessThan, LessThanOrEqual, GreaterThan, GreaterThanOrEqual):
        projections = {
            repr(strict_projection(SCHEMA, spec)(expression_type("amount", Decimal(spelling)))) for spelling in SPELLINGS
        }
        assert len(projections) == 1, (expression_type, projections)


def test_strict_projection_never_over_claims():
    # 100.00 has truncate[10] = 100.00, and 100.00 is only *equal* to the
    # literal, not greater, so a strict `> 100.00` projection must exclude the
    # 100.00 partition (it lies in truncate 100.00, which contains 100.00 too).
    spec = _spec(TruncateTransform(10))
    strict = strict_projection(SCHEMA, spec)(GreaterThan("amount", Decimal("1E+2")))
    assert repr(strict) == repr(GreaterThan("amount_part", Decimal("100.00")))
    # The projection on the partition column is exclusive; evaluating it at
    # partition value 100.00 must be false (the partition may contain equals).
    from pyiceberg.expressions.visitors import expression_evaluator
    from pyiceberg.typedef import Record

    evaluator = expression_evaluator(
        Schema(NestedField(1000, "amount_part", AMOUNT, required=False)),
        strict,
        case_sensitive=True,
    )
    assert evaluator(Record(amount_part=Decimal("100.00"))) is False


def test_inclusive_projection_never_misses():
    spec = _spec(TruncateTransform(10))
    inclusive = inclusive_projection(SCHEMA, spec)(LessThanOrEqual("amount", Decimal("1E+2")))
    from pyiceberg.expressions.visitors import expression_evaluator
    from pyiceberg.typedef import Record

    evaluator = expression_evaluator(
        Schema(NestedField(1000, "amount_part", AMOUNT, required=False)),
        inclusive,
        case_sensitive=True,
    )
    # 100.00 itself lies in the truncate partition 100.00.
    assert evaluator(Record(amount_part=Decimal("100.00"))) is True
    # A partition beyond it is excluded.
    assert evaluator(Record(amount_part=Decimal("100.10"))) is False


def test_boolean_combinations_projection():
    spec = _spec(TruncateTransform(10))
    combined = And(
        GreaterThanOrEqual("amount", Decimal("99.95")),
        Or(EqualTo("amount", Decimal("1E+2")), EqualTo("amount", Decimal("100.10"))),
    )
    results = {
        repr(
            inclusive_projection(SCHEMA, spec)(
                And(
                    GreaterThanOrEqual("amount", Decimal(a)),
                    Or(EqualTo("amount", Decimal(b)), EqualTo("amount", Decimal("100.10"))),
                )
            )
        )
        for a, b in [("99.95", "1E+2"), ("99.950", "100.00"), ("99.95", "100")]
    }
    assert len(results) == 1
    assert isinstance(inclusive_projection(SCHEMA, spec)(combined), (Or, And))
