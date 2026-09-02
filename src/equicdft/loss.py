"""Composable objectives for training grid density-functional models."""

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from ._argument_checks import (
    nonempty_string,
    nonnegative_scalar,
)
from ._fourier import (
    average_fourier_phases,
    normalized_directions,
)
from .response import FourierResponse
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


class FourierResponseLoss(nn.Module):
    r"""Fit projected homogeneous Fourier curvatures to response data.

    Each batch item is one homogeneous, periodic, unmasked state. Integer mode
    triplets are read from ``modes_key``. Symmetric fixed-number perturbations
    evaluate ``beta * (F_id + F_exc)`` along each component-space direction;
    valid cosine and sine estimates are averaged before comparison with the
    target. A one-component target is ``1/S(k)``. Mixture targets must project
    the full inverse response matrix using the same direction convention.
    """

    requires_model = True
    provides_details = True

    def __init__(
        self,
        directions: Sequence[Sequence[float]],
        modes_key: str = "fourier_modes",
        target_key: str = "fourier_curvature",
        scale_key: Optional[str] = None,
        weights_key: Optional[str] = None,
        relative_amplitude: float = 0.01,
        perturbations_per_forward: Optional[int] = None,
        loss_fn: Optional[nn.Module] = None,
        weight: float = 1.0,
        name: str = "fourier_response",
    ) -> None:
        super().__init__()

        if loss_fn is None:
            loss_fn = nn.SmoothL1Loss(reduction="none")
        if not isinstance(loss_fn, nn.Module):
            raise TypeError("loss_fn must be a torch.nn.Module")

        self.name = nonempty_string(name, "name")
        self.modes_key = nonempty_string(modes_key, "modes_key")
        self.target_key = nonempty_string(target_key, "target_key")
        self.scale_key = _optional_key(scale_key, "scale_key")
        self.weights_key = _optional_key(weights_key, "weights_key")
        self.response = FourierResponse(
            relative_amplitude=relative_amplitude,
            perturbations_per_forward=perturbations_per_forward,
            require_uniform=True,
        )
        self.loss_fn = loss_fn
        self.weight = nonnegative_scalar(weight, "weight")
        self.register_buffer("directions", normalized_directions(directions))

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Return the weighted projected-response loss."""

        return self.evaluate(outputs, batch, model=model)["loss"]

    def evaluate(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return the loss and raw response tensors from one evaluation.

        ``prediction`` and ``target`` retain physical curvature units, while
        ``element_weight`` includes validity and optional user weights.  This
        lets training and validation code report response metrics without
        repeating the expensive perturbed model evaluations.
        """

        if model is None:
            raise ValueError("FourierResponseLoss requires the model")
        modes = _required_batch_value(batch, self.modes_key)
        curvature, valid = self.response(
            model=model,
            batch=batch,
            modes=modes,
            directions=self.directions,
            outputs=outputs,
        )
        prediction, valid = average_fourier_phases(curvature, valid)
        target = _response_tensor(batch, self.target_key, prediction)
        raw_prediction = prediction
        raw_target = target
        scale = torch.ones_like(prediction)
        if self.scale_key is not None:
            scale = _response_tensor(
                batch,
                self.scale_key,
                prediction,
                positive=True,
            )
            prediction = prediction / scale
            target = target / scale

        element_loss = self.loss_fn(prediction, target)
        if (
            not isinstance(element_loss, torch.Tensor)
            or element_loss.shape != target.shape
        ):
            raise ValueError("loss_fn must return one value per response target")
        if not torch.all(torch.isfinite(element_loss)).item():
            raise ValueError("loss_fn returned nonfinite values")

        element_weight = valid.to(element_loss)
        if self.weights_key is not None:
            element_weight = element_weight * _response_tensor(
                batch,
                self.weights_key,
                prediction,
                nonnegative=True,
            )
        total_weight = element_weight.sum()
        if total_weight.item() <= 0.0:
            raise ValueError("response weights must have a positive valid sum")
        value = (
            self.weight
            * torch.sum(element_loss * element_weight)
            / total_weight
        )
        return {
            "loss": value,
            "element_loss": self.weight * element_loss,
            "prediction": raw_prediction,
            "target": raw_target,
            "scale": scale,
            "element_weight": element_weight,
        }


def _response_tensor(
    batch: Dict[str, torch.Tensor],
    key: str,
    reference: torch.Tensor,
    positive: bool = False,
    nonnegative: bool = False,
) -> torch.Tensor:
    """Return one finite response tensor with the prediction shape."""

    value = torch.as_tensor(_required_batch_value(batch, key)).detach().to(
        reference
    )
    if value.shape != reference.shape:
        raise ValueError(
            "{} must have shape [n_fields, n_modes, n_directions]".format(key)
        )
    if not torch.all(torch.isfinite(value)).item():
        raise ValueError("{} must be finite".format(key))
    if positive and torch.any(value <= 0.0).item():
        raise ValueError("{} values must be positive".format(key))
    if nonnegative and torch.any(value < 0.0).item():
        raise ValueError("{} values must be nonnegative".format(key))
    return value


def _optional_key(value: Optional[str], name: str) -> Optional[str]:
    """Validate an optional batch key."""

    return None if value is None else nonempty_string(value, name)


def _required_batch_value(
    batch: Dict[str, torch.Tensor],
    key: str,
) -> torch.Tensor:
    """Return one required batch value."""

    if key not in batch:
        raise KeyError("batch is missing '{}'".format(key))
    return batch[key]


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

        values, _ = self.evaluate(outputs, batch, model=model)
        return values

    def evaluate(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        """Return scalar terms and optional reusable term details.

        Terms defining an ``evaluate`` method may return a mapping containing
        their scalar ``loss`` and diagnostic tensors. Ordinary scalar terms
        retain the exact existing calling convention.
        """

        values = {}
        details = {}
        total = None
        for term in self.terms:
            if getattr(term, "provides_details", False):
                if getattr(term, "requires_model", False):
                    term_details = term.evaluate(outputs, batch, model=model)
                else:
                    term_details = term.evaluate(outputs, batch)
                if not isinstance(term_details, dict) or "loss" not in term_details:
                    raise ValueError(
                        "loss term '{}' evaluate must return a mapping with "
                        "'loss'".format(term.name)
                    )
                value = term_details["loss"]
                details[term.name] = term_details
            elif getattr(term, "requires_model", False):
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
