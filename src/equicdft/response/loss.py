"""Fourier-response loss term comparing projected curvatures with reference data."""

from typing import Dict, Optional, Sequence
import torch
from torch import nn
from .._validation import nonempty_string
from .modes import average_fourier_phases, normalized_directions
from .response import FourierResponse
from ..training.loss import LossTerm

__all__ = [
    "FourierResponseLoss",
]


class FourierResponseLoss(LossTerm):
    r"""Fit projected homogeneous Fourier curvatures to response data.

    Each batch item is one homogeneous, periodic, unmasked state. Integer
    reciprocal-lattice triplets are read from ``wavevector_indices_key``.
    Symmetric fixed-number perturbations evaluate
    ``beta * (F_id + F_exc)`` along each component-space direction; valid
    cosine and sine estimates are averaged before comparison with the target.
    A one-component target is ``1/S(k)``. Mixture targets must project the full
    inverse response matrix using the same direction convention.
    """

    def __init__(
        self,
        directions: Sequence[Sequence[float]],
        *,
        wavevector_indices_key: str = "fourier_wavevector_indices",
        target_key: str = "fourier_curvature",
        scale_key: Optional[str] = None,
        weights_key: Optional[str] = None,
        relative_amplitude: float = 0.01,
        perturbations_per_forward: Optional[int] = None,
        loss_fn: Optional[nn.Module] = None,
        weight: float = 1.0,
        name: str = "fourier_response",
    ) -> None:
        super().__init__(name, weight)

        if loss_fn is None:
            loss_fn = nn.SmoothL1Loss(reduction="none")
        if not isinstance(loss_fn, nn.Module):
            raise TypeError("loss_fn must be a torch.nn.Module")

        self.wavevector_indices_key = nonempty_string(
            wavevector_indices_key,
            "wavevector_indices_key",
        )
        self.target_key = nonempty_string(target_key, "target_key")
        self.scale_key = _optional_key(scale_key, "scale_key")
        self.weights_key = _optional_key(weights_key, "weights_key")
        self.response = FourierResponse(
            relative_amplitude=relative_amplitude,
            perturbations_per_forward=perturbations_per_forward,
            require_uniform=True,
        )
        self.loss_fn = loss_fn
        self.register_buffer("directions", normalized_directions(directions))

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
        wavevector_indices = _required_batch_value(
            batch,
            self.wavevector_indices_key,
        )
        curvature, valid = self.response(
            model=model,
            batch=batch,
            wavevector_indices=wavevector_indices,
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
            "{} must have shape "
            "[n_fields, n_wavevectors, n_directions]".format(key)
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
