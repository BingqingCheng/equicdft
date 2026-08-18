"""Strict checks for scalar Python API arguments."""

import math
from numbers import Integral, Real
from typing import Optional, Sequence, Tuple


def positive_integer(value: object, name: str) -> int:
    """Return a positive integer, rejecting Boolean and nonintegral values."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("{} must be a positive integer".format(name))
    value = int(value)
    if value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def nonnegative_integer(value: object, name: str) -> int:
    """Return a nonnegative integer, rejecting Boolean values."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("{} must be a nonnegative integer".format(name))
    value = int(value)
    if value < 0:
        raise ValueError("{} must be a nonnegative integer".format(name))
    return value


def optional_positive_integer(
    value: Optional[object],
    name: str,
) -> Optional[int]:
    """Return ``None`` or a positive integer."""

    if value is None:
        return None
    return positive_integer(value, name)


def finite_scalar(value: object, name: str) -> float:
    """Return a finite real scalar, rejecting Boolean values."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("{} must be a finite scalar".format(name))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be a finite scalar".format(name))
    return value


def positive_scalar(value: object, name: str) -> float:
    """Return a finite scalar strictly greater than zero."""

    value = finite_scalar(value, name)
    if value <= 0.0:
        raise ValueError("{} must be a finite positive scalar".format(name))
    return value


def nonnegative_scalar(value: object, name: str) -> float:
    """Return a finite scalar greater than or equal to zero."""

    value = finite_scalar(value, name)
    if value < 0.0:
        raise ValueError(
            "{} must be a finite nonnegative scalar".format(name)
        )
    return value


def boolean(value: object, name: str) -> bool:
    """Return a Boolean without coercing integers or strings."""

    if not isinstance(value, bool):
        raise TypeError("{} must be a boolean".format(name))
    return value


def optional_boolean(
    value: Optional[object],
    name: str,
) -> Optional[bool]:
    """Return ``None`` or a Boolean without coercion."""

    if value is None:
        return None
    return boolean(value, name)


def nonempty_string(value: object, name: str) -> str:
    """Return a nonempty string."""

    if not isinstance(value, str):
        raise TypeError("{} must be a nonempty string".format(name))
    if not value:
        raise ValueError("{} must be a nonempty string".format(name))
    return value


def unique_strings(
    values: Sequence[str],
    name: str,
) -> Tuple[str, ...]:
    """Return a nonempty tuple of unique, nonempty strings."""

    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError("{} must be a sequence of strings".format(name))
    values = tuple(values)
    if not values:
        raise ValueError("{} must not be empty".format(name))
    for value in values:
        nonempty_string(value, name)
    if len(set(values)) != len(values):
        raise ValueError("{} must contain unique values".format(name))
    return values
