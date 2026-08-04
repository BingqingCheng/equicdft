"""Ordered target selection shared by losses and metrics."""

from typing import Dict, Sequence, Tuple, Union

import torch


TargetKeys = Union[str, Sequence[str]]


def normalize_target_keys(
    value: TargetKeys,
    field: str = "target_key",
) -> Tuple[str, ...]:
    """Return one or more unique target keys in priority order."""

    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        raise ValueError(
            "{} must be a nonempty string or sequence of strings".format(
                field
            )
        )
    if not values:
        raise ValueError("{} must not be empty".format(field))
    if any(not isinstance(key, str) or not key for key in values):
        raise ValueError(
            "{} must contain nonempty strings".format(field)
        )
    if len(set(values)) != len(values):
        raise ValueError("{} must contain unique values".format(field))
    return values


def resolve_target(
    prediction: torch.Tensor,
    prediction_key: str,
    target_keys: Sequence[str],
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Resolve ordered targets, filling nonfinite entries from later keys.

    The batch takes precedence over model outputs for each key. A target with
    no grid axis is expanded over the prediction's grid axis. Nonfinite values
    mark unavailable entries, allowing one mixed batch to use a known target
    for some fields and a model-derived fallback for the others.
    """

    target = torch.zeros_like(prediction)
    resolved = torch.zeros_like(prediction, dtype=torch.bool)
    found = []

    for key in target_keys:
        if key in batch:
            candidate = batch[key]
        elif key in outputs:
            candidate = outputs[key]
        else:
            continue
        found.append(key)
        if not isinstance(candidate, torch.Tensor):
            raise TypeError("target '{}' must be a tensor".format(key))

        if candidate.shape != prediction.shape:
            component_target_shape = (
                prediction.shape[:-2] + prediction.shape[-1:]
            )
            if candidate.shape == component_target_shape:
                candidate = candidate.unsqueeze(-2).expand_as(prediction)
            else:
                raise ValueError(
                    "prediction '{}' has shape {}, but target '{}' has shape "
                    "{}".format(
                        prediction_key,
                        tuple(prediction.shape),
                        key,
                        tuple(candidate.shape),
                    )
                )

        candidate = candidate.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        available = torch.isfinite(candidate)
        use_candidate = torch.logical_and(~resolved, available)
        target = torch.where(use_candidate, candidate, target)
        resolved = torch.logical_or(resolved, available)
        if torch.all(resolved).item():
            return target

    if not found:
        raise KeyError(
            "batch and model outputs are missing target candidates {}".format(
                tuple(target_keys)
            )
        )
    unresolved = int(torch.count_nonzero(~resolved).item())
    raise ValueError(
        "target candidates {} leave {} unresolved nonfinite values".format(
            tuple(target_keys),
            unresolved,
        )
    )
