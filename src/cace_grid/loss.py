"""Composable objectives for training grid density-functional models."""

import math
from typing import Dict, Optional, Sequence

import torch
from torch import nn


class TensorLoss(nn.Module):
    """Apply one weighted scalar loss to a prediction/target tensor pair.

    Parameters
    ----------
    name
        Unique name used in the dictionary returned by :class:`Loss`.
    prediction_key
        Key selecting the predicted tensor from the model outputs.
    target_key
        Key selecting the reference tensor from the training batch.
    loss_fn
        PyTorch loss module. It must reduce the selected tensors to one scalar.
        The default is mean-squared error.
    weight
        Nonnegative multiplier applied to this loss term.

    Notes
    -----
    Prediction and target shapes must match exactly. Grid and component axes
    are never reshaped implicitly because doing so could hide data-layout
    errors.
    """

    def __init__(
        self,
        name: str,
        prediction_key: str,
        target_key: str,
        loss_fn: Optional[nn.Module] = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.name = _validate_name(name)
        self.prediction_key = _validate_key(prediction_key, "prediction_key")
        self.target_key = _validate_key(target_key, "target_key")
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        if not isinstance(loss_fn, nn.Module):
            raise TypeError("loss_fn must be a torch.nn.Module")
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise ValueError("weight must be a finite nonnegative scalar")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be a finite nonnegative scalar")

        self.loss_fn = loss_fn
        self.weight = weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return this term's weighted scalar loss."""

        if self.prediction_key not in outputs:
            raise KeyError(
                "model outputs are missing prediction '{}'".format(
                    self.prediction_key
                )
            )
        if self.target_key not in batch:
            raise KeyError(
                "training batch is missing target '{}'".format(
                    self.target_key
                )
            )

        prediction = outputs[self.prediction_key]
        target = batch[self.target_key]
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction '{}' has shape {}, but target '{}' has shape {}".format(
                    self.prediction_key,
                    tuple(prediction.shape),
                    self.target_key,
                    tuple(target.shape),
                )
            )

        value = self.loss_fn(prediction, target)
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise ValueError("loss_fn must return one scalar tensor")
        return self.weight * value


class Loss(nn.Module):
    """Aggregate named scalar loss terms into one training objective.

    Each registered term must be an ``nn.Module`` with a unique string
    ``name`` attribute and must return one scalar tensor from
    ``term(outputs, batch)``. This permits specialized future terms to coexist
    with :class:`TensorLoss` without changing the trainer.
    """

    def __init__(self, terms: Sequence[nn.Module]) -> None:
        super().__init__()

        terms = list(terms)
        if not terms:
            raise ValueError("Loss requires at least one loss term")

        names = []
        for term in terms:
            if not isinstance(term, nn.Module):
                raise TypeError("every loss term must be a torch.nn.Module")
            name = _validate_name(getattr(term, "name", None))
            if name == "total":
                raise ValueError("'total' is reserved for the aggregate loss")
            if name in names:
                raise ValueError("loss term names must be unique")
            names.append(name)

        self.terms = nn.ModuleList(terms)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return weighted named terms and their scalar sum as ``total``."""

        values = {}
        total = None
        for term in self.terms:
            value = term(outputs, batch)
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError(
                    "loss term '{}' must return one scalar tensor".format(
                        term.name
                    )
                )
            values[term.name] = value
            total = value if total is None else total + value

        values["total"] = total
        return values


def _validate_name(name: Optional[str]) -> str:
    """Return a nonempty loss-term name."""

    if not isinstance(name, str) or not name:
        raise ValueError("loss term name must be a nonempty string")
    return name


def _validate_key(key: str, field: str) -> str:
    """Return a nonempty prediction or target key."""

    if not isinstance(key, str) or not key:
        raise ValueError("{} must be a nonempty string".format(field))
    return key
