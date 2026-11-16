#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

"""Helper methods for working with Python Decimals."""

import math
from decimal import Decimal
from typing import Optional, Tuple, Union

# Iceberg supports at most 38 digits of precision.
MAX_DECIMAL_PRECISION = 38


def _decimal_tuple(value: Decimal) -> Tuple[int, Tuple[int, ...], int]:
    """Return the exact (sign, digits, exponent) of a finite Decimal.

    The tuple is read directly from the Decimal representation, so neither the
    caller's decimal context nor its traps are consulted. Non-finite values
    (NaN, sNaN, +/-Infinity) do not have an integer exponent and are rejected.
    """
    if not isinstance(value, Decimal):
        raise TypeError(f"Expected a Decimal, got: {type(value)}")
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError(f"Cannot use a non-finite decimal: {value}")
    return sign, digits, exponent


def _coefficient(digits: Tuple[int, ...]) -> int:
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    return coefficient


def _unscaled_at_scale(value: Decimal, scale: int, max_digits: Optional[int] = None) -> int:
    """Return the exact unscaled integer of ``value`` at ``scale``.

    Raises:
        ValueError: If the value cannot be represented exactly at the requested
            scale (it would need rounding), is not finite, or (when
            ``max_digits`` is given) would exceed that many digits. No rounding
            is ever performed.
    """
    if not isinstance(scale, int) or scale < 0:
        raise ValueError(f"Invalid decimal scale: {scale}")
    sign, digits, exponent = _decimal_tuple(value)
    coefficient = _coefficient(digits)
    # ``-0`` and ``0`` are the same numeric value; the canonical form is +0.
    if coefficient == 0:
        return 0
    shift = exponent + scale
    # Iceberg scales are at most 38, so legitimate exponents are small. Reject
    # absurd exponents early instead of materialising gigantic integers.
    if abs(shift) > 1000:
        raise ValueError(f"Decimal {value} cannot be represented with scale {scale}")
    if shift >= 0:
        digit_count = len(str(coefficient)) + shift
        if max_digits is not None and digit_count > max_digits:
            raise ValueError(f"Decimal {value} exceeds precision {max_digits}")
        unscaled = coefficient * 10**shift
    else:
        if -shift > len(str(coefficient)):
            # The value has more fractional digits than its coefficient; it
            # cannot be represented exactly at this scale.
            raise ValueError(f"Decimal {value} cannot be represented exactly with scale {scale}")
        divisor = 10**-shift
        quotient, remainder = divmod(coefficient, divisor)
        if remainder:
            raise ValueError(f"Decimal {value} cannot be represented exactly with scale {scale}")
        if max_digits is not None and len(str(quotient)) > max_digits:
            raise ValueError(f"Decimal {value} exceeds precision {max_digits}")
        unscaled = quotient
    return -unscaled if sign else unscaled


def _digits_count(unscaled: int) -> int:
    return len(str(abs(unscaled))) if unscaled else 1


def decimal_to_unscaled(value: Decimal) -> int:
    """Get an unscaled value given a Decimal value.

    This is the "bare" helper: it keeps the Decimal's own scale convention and
    returns the integer formed by its digits (the exponent is not applied).
    The result is computed with integer arithmetic only, so it does not depend
    on the caller's decimal context.

    Args:
        value (Decimal): A Decimal instance.

    Returns:
        int: The unscaled value.
    """
    sign, digits, _ = _decimal_tuple(value)
    unscaled = _coefficient(digits)
    return -unscaled if sign else unscaled


def decimal_unscaled_at_scale(value: Decimal, scale: int) -> int:
    """Return the exact unscaled integer of ``value`` at ``scale``.

    Raises:
        ValueError: If the value is not finite or cannot be represented exactly
            at ``scale`` without rounding.
    """
    return _unscaled_at_scale(value, scale)


def decimal_rescale(value: Decimal, scale: int) -> Decimal:
    """Return ``value`` encoded with exactly ``scale`` fractional digits.

    The numeric value is preserved exactly; values that would require rounding
    are rejected instead of being truncated or rounded. The returned Decimal
    has a deterministic tuple for a given (value, scale) pair, so different
    spellings of the same number compare and hash identically.

    Raises:
        ValueError: If the value is not finite or is not exactly representable
            at ``scale``.
    """
    return unscaled_to_decimal(_unscaled_at_scale(value, scale), scale)


def decimal_canonical(value: Decimal, precision: int, scale: int) -> Decimal:
    """Encode ``value`` at (precision, scale), preserving its numeric value.

    Raises:
        ValueError: If the value is not finite, needs rounding, or exceeds the
            declared precision.
    """
    if not isinstance(precision, int) or precision <= 0 or precision > MAX_DECIMAL_PRECISION:
        raise ValueError(f"Unsupported decimal precision: {precision}")
    return unscaled_to_decimal(_unscaled_at_scale(value, scale, max_digits=precision), scale)


def decimal_to_integral(value: Decimal) -> int:
    """Return ``value`` rounded to an integer using round-half-even.

    This mirrors ``Decimal.to_integral_value()`` under the default decimal
    context, but uses exact integer arithmetic so a caller's context precision,
    rounding mode, flags or traps cannot change the result.
    """
    sign, digits, exponent = _decimal_tuple(value)
    coefficient = _coefficient(digits)
    if exponent >= 0:
        integral = coefficient * 10**exponent
    else:
        divisor = 10**-exponent
        integral, remainder = divmod(coefficient, divisor)
        twice = 2 * remainder
        # Round half to even: a tie goes to the nearest even integer.
        if twice > divisor or (twice == divisor and integral % 2 == 1):
            integral += 1
    return -integral if sign else integral


def decimal_from_float(value: float, scale: int) -> Decimal:
    """Convert a float to a Decimal at ``scale`` using round-half-up.

    This mirrors the historical behaviour of ``Decimal(value).quantize(...,
    ROUND_HALF_UP)`` but performs the rounding with integer arithmetic, so it
    is independent of the caller's decimal context (precision, rounding mode
    and traps). The float's exact binary value is used, never ``str(value)``.
    """
    # ``Decimal(float)`` raises under a FloatOperation trap, and reading the
    # value must not depend on the caller's context.
    decimal_value = Decimal.from_float(value)
    sign, digits, exponent = _decimal_tuple(decimal_value)
    coefficient = _coefficient(digits)
    if coefficient == 0:
        return unscaled_to_decimal(0, scale)
    shift = exponent + scale
    if shift >= 0:
        unscaled = coefficient * 10**shift
    else:
        divisor = 10**-shift
        quotient, remainder = divmod(coefficient, divisor)
        if 2 * remainder >= divisor:
            quotient += 1
        unscaled = quotient
    return unscaled_to_decimal(-unscaled if sign else unscaled, scale)


def unscaled_to_decimal(unscaled: int, scale: int) -> Decimal:
    """Get a scaled Decimal value given an unscaled value and a scale.

    Args:
        unscaled (int): An unscaled value.
        scale (int): A scale to set for the returned Decimal instance.

    Returns:
        Decimal: A scaled Decimal instance.
    """
    sign, digits, _ = Decimal(unscaled).as_tuple()
    return Decimal((sign, digits, -scale))


def unscaled_to_minimal_bytes(unscaled: int) -> bytes:
    """Return the minimal big-endian two's complement encoding of ``unscaled``.

    This is the encoding Iceberg uses when hashing decimal values for the
    bucket transform. ``int.to_bytes`` with the formula used for serialization
    keeps a redundant sign byte for some negative values (for example -128);
    that would change the hash, so the leading byte is stripped here.
    """
    length = max(1, (unscaled.bit_length() + 8) // 8)
    encoded = unscaled.to_bytes(length, byteorder="big", signed=True)
    while len(encoded) > 1:
        if encoded[0] == 0x00 and encoded[1] < 0x80:
            encoded = encoded[1:]
        elif encoded[0] == 0xFF and encoded[1] >= 0x80:
            encoded = encoded[1:]
        else:
            break
    return encoded


def decimal_to_minimal_bytes_at_scale(value: Decimal, scale: int) -> bytes:
    """Return the minimal unscaled bytes of ``value`` at the declared ``scale``."""
    return unscaled_to_minimal_bytes(decimal_unscaled_at_scale(value, scale))


def bytes_required(value: Union[int, Decimal]) -> int:
    """Return the minimum number of bytes needed to serialize a decimal or unscaled value.

    Args:
        value (int | Decimal): a Decimal value or unscaled int value.

    Returns:
        int: the minimum number of bytes needed to serialize the value.
    """
    if isinstance(value, int):
        return (value.bit_length() + 8) // 8
    elif isinstance(value, Decimal):
        return (decimal_to_unscaled(value).bit_length() + 8) // 8

    raise ValueError(f"Unsupported value: {value}")


def decimal_to_bytes(value: Decimal, byte_length: Optional[int] = None) -> bytes:
    """Return a byte representation of a decimal.

    This is the "bare" helper: it follows the Decimal's own scale convention,
    like :func:`decimal_to_unscaled`. Use :func:`decimal_to_bytes_at_scale`
    when the target scale is known.

    Args:
        value (Decimal): a decimal value.
        byte_length (int): The number of bytes.
    Returns:
        bytes: the unscaled value of the Decimal as bytes.
    """
    unscaled_value = decimal_to_unscaled(value)
    if byte_length is None:
        byte_length = bytes_required(unscaled_value)
    return unscaled_value.to_bytes(byte_length, byteorder="big", signed=True)


def decimal_to_bytes_at_scale(value: Decimal, scale: int, byte_length: Optional[int] = None) -> bytes:
    """Encode ``value`` as unscaled bytes at the declared ``scale``.

    The numeric value is preserved exactly, so every spelling of the same
    number produces the same bytes. Values that would need rounding are
    rejected.

    Args:
        value (Decimal): a decimal value.
        scale (int): the scale declared by the column type.
        byte_length (int): The number of bytes.

    Returns:
        bytes: the unscaled value of the Decimal as bytes.
    """
    unscaled_value = _unscaled_at_scale(value, scale)
    if byte_length is None:
        byte_length = bytes_required(unscaled_value)
    return unscaled_value.to_bytes(byte_length, byteorder="big", signed=True)


def decimal_encode(value: Decimal, precision: int, scale: int) -> int:
    """Return the unscaled integer of ``value`` at (precision, scale).

    Raises:
        ValueError: If the value is not finite, needs rounding, or has more
            digits than ``precision``.
    """
    if not isinstance(precision, int) or precision <= 0 or precision > MAX_DECIMAL_PRECISION:
        raise ValueError(f"Unsupported decimal precision: {precision}")
    unscaled = _unscaled_at_scale(value, scale)
    if _digits_count(unscaled) > precision:
        raise ValueError(f"Decimal {value} exceeds precision {precision}")
    return unscaled


def bytes_to_decimal(value: bytes, scale: int) -> Decimal:
    """Return a decimal from the bytes.

    Args:
        value (bytes): the bytes to be converted into a decimal.
        scale (int): the scale of the decimal.

    Returns:
        Decimal: the scaled decimal.
    """
    unscaled_datum = int.from_bytes(value, byteorder="big", signed=True)
    return unscaled_to_decimal(unscaled_datum, scale)


def truncate_decimal(value: Decimal, width: int) -> Decimal:
    """Get a truncated Decimal value given a decimal value and a width.

    This is the "bare" helper and keeps the Decimal's own scale convention.
    The computation uses integer arithmetic plus :func:`unscaled_to_decimal`
    only, so it does not depend on the caller's decimal context.

    Args:
        value (Decimal): a decimal value.
        width (int): A width for the returned Decimal instance.
    Returns:
        Decimal: A truncated Decimal instance.
    """
    _, _, exponent = _decimal_tuple(value)
    unscaled_value = decimal_to_unscaled(value)
    applied_value = unscaled_value - (((unscaled_value % width) + width) % width)
    return unscaled_to_decimal(applied_value, abs(exponent))


def truncate_decimal_at_scale(value: Decimal, width: int, scale: int) -> Decimal:
    """Truncate ``value`` to ``width`` in unscaled units at the declared scale.

    The source value is first encoded exactly at ``scale`` (never rounded), and
    the truncated result is returned with exactly ``scale`` fractional digits.
    """
    unscaled_value = _unscaled_at_scale(value, scale)
    applied_value = unscaled_value - (((unscaled_value % width) + width) % width)
    return unscaled_to_decimal(applied_value, scale)


MAX_PRECISION = tuple(math.floor(math.log10(math.fabs(math.pow(2, 8 * pos - 1) - 1))) for pos in range(24))
REQUIRED_LENGTH = tuple(next(pos for pos in range(24) if p <= MAX_PRECISION[pos]) for p in range(40))


def decimal_required_bytes(precision: int) -> int:
    """Compute the number of bytes required to store a precision.

    Args:
        precision: The number of digits to store.

    Returns:
        The number of bytes required to store a decimal with a certain precision.
    """
    if precision <= 0 or precision >= 40:
        raise ValueError(f"Unsupported precision, outside of (0, 40]: {precision}")

    return REQUIRED_LENGTH[precision]
