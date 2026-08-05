"""Small private helpers shared by neural-network readouts."""

from numbers import Integral
from typing import Optional, Sequence, Tuple

import torch
from torch import nn


def positive_integer(value: int, name: str) -> int:
    """Validate and return a positive integer."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("{} must be a positive integer".format(name))
    value = int(value)
    if value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def optional_positive_integer(
    value: Optional[int],
    name: str,
) -> Optional[int]:
    """Validate an optional positive integer."""

    if value is None:
        return None
    return positive_integer(value, name)


def validate_hidden_sizes(values: Sequence[int]) -> Tuple[int, ...]:
    """Return hidden-layer widths as a validated tuple."""

    return tuple(
        positive_integer(value, "hidden_sizes entries") for value in values
    )


def positive_scalar_tensor(value: object, name: str) -> torch.Tensor:
    """Return a detached positive scalar using the default floating dtype."""

    scalar = torch.as_tensor(
        value,
        dtype=torch.get_default_dtype(),
    ).detach().clone().reshape(-1)
    if scalar.numel() != 1:
        raise ValueError("{} must be a positive scalar".format(name))
    scalar = scalar.reshape(())
    if not torch.isfinite(scalar).item() or scalar.item() <= 0.0:
        raise ValueError("{} must be a positive scalar".format(name))
    return scalar


def build_mlp(
    input_size: Optional[int],
    hidden_sizes: Sequence[int],
    output_size: int,
    zero_init: bool = False,
) -> nn.Sequential:
    """Build a SiLU MLP, optionally inferring its input width lazily."""

    input_size = optional_positive_integer(input_size, "input_size")
    hidden_sizes = validate_hidden_sizes(hidden_sizes)
    output_size = positive_integer(output_size, "output_size")
    if not isinstance(zero_init, bool):
        raise TypeError("zero_init must be a boolean")

    layers = []
    width = input_size
    for hidden_width in hidden_sizes:
        linear = (
            nn.LazyLinear(hidden_width)
            if width is None
            else nn.Linear(width, hidden_width)
        )
        layers.extend((linear, nn.SiLU()))
        width = hidden_width

    final_layer = (
        nn.LazyLinear(output_size)
        if width is None
        else nn.Linear(width, output_size)
    )
    if zero_init:
        if isinstance(final_layer, nn.LazyLinear):
            raise ValueError("zero_init requires a known input size")
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
    layers.append(final_layer)
    return nn.Sequential(*layers)
