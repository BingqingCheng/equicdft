"""Composable objectives for training grid density-functional models."""

from typing import Dict, Optional, Sequence

import torch
from torch import nn

from ._argument_checks import nonempty_string, nonnegative_scalar
from ._targets import TargetKeys, normalize_target_keys, resolve_target


class TensorLoss(nn.Module):
    """Apply one weighted scalar loss to a prediction/target tensor pair.

    ``target_key`` may contain one key or an ordered sequence. For every key,
    the batch is checked before the model outputs and the first finite value
    is used. A componentwise target without a grid axis is expanded over that
    axis. If ``weights_key`` is supplied, the loss function must return one
    value per prediction and those values are reduced by a weighted mean.
    """

    def __init__(
        self,
        name: str,
        prediction_key: str,
        target_key: TargetKeys,
        loss_fn: Optional[nn.Module] = None,
        weights_key: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.name = nonempty_string(name, "loss term name")
        self.prediction_key = nonempty_string(
            prediction_key,
            "prediction_key",
        )
        self.target_keys = normalize_target_keys(target_key)
        self.target_key = (
            self.target_keys[0]
            if len(self.target_keys) == 1
            else self.target_keys
        )
        self.weights_key = (
            None
            if weights_key is None
            else nonempty_string(weights_key, "weights_key")
        )
        if loss_fn is None:
            loss_fn = nn.MSELoss(
                reduction="none" if self.weights_key else "mean"
            )
        if not isinstance(loss_fn, nn.Module):
            raise TypeError("loss_fn must be a torch.nn.Module")

        self.loss_fn = loss_fn
        self.weight = nonnegative_scalar(weight, "weight")

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
        prediction = outputs[self.prediction_key]
        target = resolve_target(
            prediction=prediction,
            prediction_key=self.prediction_key,
            target_keys=self.target_keys,
            outputs=outputs,
            batch=batch,
        )
        value = self.loss_fn(prediction, target)
        if self.weights_key is None:
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError("loss_fn must return one scalar tensor")
        else:
            weights = _resolve_weights(self.weights_key, outputs, batch)
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
            value = torch.sum(weights * value) / total_weight
        return self.weight * value


class Loss(nn.Module):
    """Aggregate uniquely named scalar loss terms into one objective."""

    def __init__(self, terms: Sequence[nn.Module]) -> None:
        super().__init__()

        terms = list(terms)
        if not terms:
            raise ValueError("Loss requires at least one loss term")
        names = []
        for term in terms:
            if not isinstance(term, nn.Module):
                raise TypeError("every loss term must be a torch.nn.Module")
            name = nonempty_string(
                getattr(term, "name", None),
                "loss term name",
            )
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
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return weighted named terms and their scalar sum as ``total``."""

        values = {}
        total = None
        for term in self.terms:
            if getattr(term, "requires_model", False):
                value = term(outputs, batch, model=model)
            else:
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


def _resolve_weights(
    key: str,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return element weights, preferring model outputs over batch fields."""

    if key in outputs:
        return outputs[key]
    if key in batch:
        return batch[key]
    raise KeyError(
        "batch and model outputs are missing weights '{}'".format(key)
    )
