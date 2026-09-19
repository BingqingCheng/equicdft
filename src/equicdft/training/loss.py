"""Composable objectives for training grid density-functional models."""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from .._validation import nonempty_string
from ..objectives import LossTerm
from ._targets import TargetKeys, normalize_target_keys, resolve_target, center_fields

__all__ = [
    "LossTerm",
    "TensorLoss",
    "Loss",
]


class TensorLoss(LossTerm):
    """Apply one weighted scalar loss to a prediction/target tensor pair.

    ``target_key`` may contain one key or an ordered sequence. For every key,
    the batch is checked before the model outputs and the first finite value
    is used. A componentwise target without a grid axis is expanded over that
    axis. If ``weights_key`` is supplied, the loss function must return one
    value per prediction and those values are reduced by a weighted mean.

    ``center_axis`` removes each tensor's weighted mean along that axis,
    independently for all other indices (e.g. -2 for scalar grid/species
    fields). ``target_scale`` multiplies the target before projection;
    ``residual_scale`` multiplies both tensors before the criterion, hence
    enters squared in MSE. ``weight`` multiplies the reduced loss afterward.
    """

    def __init__(
        self,
        name: str,
        prediction_key: str,
        target_key: TargetKeys,
        *,
        loss_fn: Optional[nn.Module] = None,
        weights_key: Optional[str] = None,
        center_axis: Optional[int] = None,
        residual_scale=1.0,
        target_scale=1.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name, weight)

        self.prediction_key = nonempty_string(
            prediction_key,
            "prediction_key",
        )
        self.center_axis = center_axis
        if center_axis is not None and (isinstance(center_axis, bool) or not isinstance(center_axis, int)):
            raise TypeError("center_axis must be an integer axis or None")
        for name, value in (("residual_scale", residual_scale), ("target_scale", target_scale)):
            tensor = torch.as_tensor(value)
            if tensor.is_complex() or not torch.isfinite(tensor).all():
                raise ValueError(name + " must be finite and real")
            self.register_buffer(name, tensor, persistent=False)
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

    def evaluate(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return the weighted scalar loss; the model is not needed."""

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
        if (self.center_axis is None and torch.all(self.target_scale == 1)
                and torch.all(self.residual_scale == 1)):
            value = self.loss_fn(prediction, target)
        else:
            value = self._transformed_loss(prediction, target, outputs, batch)
        return self._reduce_loss(value, prediction, outputs, batch)

    def _transformed_loss(self, prediction, target, outputs, batch):
        target = target * self.target_scale.to(prediction)
        if self.center_axis is not None:
            weights = (None if self.weights_key is None else
                       _resolve_weights(self.weights_key, outputs, batch))
            prediction, target = center_fields(
                prediction, target, weights, self.center_axis,
            )
        scale = self.residual_scale.to(prediction)
        value = self.loss_fn(prediction * scale, target * scale)
        return value

    def _reduce_loss(self, value, prediction, outputs, batch):
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
        return {"loss": self.weight * value}


class Loss(nn.Module):
    """Aggregate uniquely named :class:`LossTerm` objects into one objective."""

    def __init__(self, terms: Sequence[LossTerm]) -> None:
        super().__init__()

        terms = list(terms)
        if not terms:
            raise ValueError("Loss requires at least one loss term")
        names = []
        for term in terms:
            if not isinstance(term, LossTerm):
                raise TypeError(
                    "every loss term must be an equicdft.LossTerm, got {}".format(
                        type(term).__name__
                    )
                )
            if term.name == "total":
                raise ValueError("'total' is reserved for the aggregate loss")
            if term.name in names:
                raise ValueError("loss term names must be unique")
            names.append(term.name)
        self.terms = nn.ModuleList(terms)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return weighted named terms and their scalar sum as ``total``."""

        values, _ = self.evaluate(outputs, batch, model=model)
        return values

    def evaluate(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, Any]],
    ]:
        """Return scalar terms and every term's reusable details.

        ``details[name]`` is the mapping returned by that term's
        :meth:`LossTerm.evaluate`, so metric collections can reuse expensive
        diagnostics without repeating the evaluation.
        """

        values = {}
        details = {}
        total = None
        for term in self.terms:
            term_details = term.evaluate(outputs, batch, model)
            if not isinstance(term_details, Mapping) or "loss" not in term_details:
                raise ValueError(
                    "loss term '{}' evaluate must return a mapping with "
                    "'loss'".format(term.name)
                )
            value = term_details["loss"]
            details[term.name] = dict(term_details)
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError(
                    "loss term '{}' must return one scalar tensor".format(
                        term.name
                    )
                )
            values[term.name] = value
            total = value if total is None else total + value
        values["total"] = total
        return values, details


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
