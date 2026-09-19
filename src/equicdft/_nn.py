"""Small private helpers shared by neural-network readouts."""

from typing import Sequence, Tuple

import torch
from torch import nn

from ._validation import (
    boolean,
    positive_integer,
)


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
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int,
    zero_init: bool = False,
) -> nn.Sequential:
    """Build a SiLU MLP with a known input width.

    Every layer is a regular ``Linear`` module, so the parameters exist and
    have their final shapes as soon as the network is constructed.
    """

    input_size = positive_integer(input_size, "input_size")
    hidden_sizes = validate_hidden_sizes(hidden_sizes)
    output_size = positive_integer(output_size, "output_size")
    zero_init = boolean(zero_init, "zero_init")

    layers = []
    width = input_size
    for hidden_width in hidden_sizes:
        layers.extend((nn.Linear(width, hidden_width), nn.SiLU()))
        width = hidden_width

    final_layer = nn.Linear(width, output_size)
    if zero_init:
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
    layers.append(final_layer)
    return nn.Sequential(*layers)
