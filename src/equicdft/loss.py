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
        Key selecting the reference tensor. The training batch is checked
        first, followed by the model outputs.
    weights_key
        Optional key selecting element weights from the model outputs or batch.
        With weights, the default is elementwise squared error followed by a
        weighted mean.
    loss_fn
        PyTorch loss module. It must reduce the selected tensors to one scalar.
        The default is mean-squared error.
    weight
        Nonnegative multiplier applied to this loss term.

    Notes
    -----
    A componentwise target lacking only the prediction's grid axis is expanded
    over that axis. All other shape mismatches are rejected.
    """

    def __init__(
        self,
        name: str,
        prediction_key: str,
        target_key: str,
        loss_fn: Optional[nn.Module] = None,
        weights_key: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.name = _validate_name(name)
        self.prediction_key = _validate_key(prediction_key, "prediction_key")
        self.target_key = _validate_key(target_key, "target_key")
        self.weights_key = (
            None
            if weights_key is None
            else _validate_key(weights_key, "weights_key")
        )
        if loss_fn is None:
            loss_fn = nn.MSELoss(
                reduction="none" if self.weights_key else "mean"
            )
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
        if self.target_key in batch:
            target = batch[self.target_key]
        elif self.target_key in outputs:
            target = outputs[self.target_key]
        else:
            raise KeyError(
                "batch and model outputs are missing target '{}'".format(
                    self.target_key
                )
            )

        prediction = outputs[self.prediction_key]
        if prediction.shape != target.shape:
            component_target_shape = (
                prediction.shape[:-2] + prediction.shape[-1:]
            )
            if target.shape == component_target_shape:
                target = target.unsqueeze(-2).expand_as(prediction)
            else:
                raise ValueError(
                    "prediction '{}' has shape {}, but target '{}' has shape "
                    "{}".format(
                        self.prediction_key,
                        tuple(prediction.shape),
                        self.target_key,
                        tuple(target.shape),
                    )
                )

        value = self.loss_fn(prediction, target)
        if self.weights_key is None:
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError("loss_fn must return one scalar tensor")
        else:
            if self.weights_key in outputs:
                weights = outputs[self.weights_key]
            elif self.weights_key in batch:
                weights = batch[self.weights_key]
            else:
                raise KeyError(
                    "batch and model outputs are missing weights '{}'".format(
                        self.weights_key
                    )
                )
            if weights.shape != prediction.shape:
                raise ValueError("weights and prediction must have same shape")
            if value.shape != prediction.shape:
                raise ValueError(
                    "weighted loss_fn must return one value per prediction"
                )
            weights = weights.detach().to(prediction)
            total_weight = weights.sum()
            if total_weight.item() <= 0.0:
                raise ValueError("weights must have a positive sum")
            value = (weights * value).sum() / total_weight
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
