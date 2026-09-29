"""Composable objectives for training grid density-functional models."""

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from .._validation import boolean, nonempty_string, nonnegative_scalar
from ..geometry.grid import common_grid_size, grid_spacing_tensor
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
    Weights may match the prediction or omit its final component axis;
    voxel weights then apply equally to every vector component. Model-output
    weights take precedence over batch weights.

    ``center_axis`` removes each tensor's weighted mean along that axis,
    independently for all other indices (e.g. -2 for scalar grid/species
    fields). ``target_scale`` multiplies the target before projection;
    ``residual_scale`` multiplies both tensors before the criterion, hence
    enters squared in MSE. ``weight`` multiplies the reduced loss afterward.

    ``residual_smoothing`` convolves both scaled tensors with a periodic
    Gaussian of that standard deviation (in the length units of the batch
    ``grid_spacing``) on the regular grid before the criterion, so the
    criterion compares smoothed fields; for the default mean-squared error
    that is the squared smoothed residual, ``exp(-sigma^2 k^2 / 2)`` times the
    residual in reciprocal space. It requires ``grid_size`` and
    ``grid_spacing`` in the batch and fields shaped ``[n_fields, n_grid, ...]``
    in canonical grid order. With ``weights_key``, smoothing requires hard
    zero/one weights. Unselected residual sources are zeroed before the
    convolution and filtered values outside the selection are omitted from
    the mean. Centering precedes smoothing. ``None`` or ``0`` leaves the
    criterion unsmoothed. The smoothing belongs to the objective only: no
    metric, model output or functional term is affected.

    Smoothing reads the batch geometry, which on a GPU batch is a host
    synchronization per evaluation. ``fixed_grid=True`` reads and validates
    ``grid_size`` and ``grid_spacing`` once, on the first evaluation, and
    reuses them for every later batch; only the number of grid points is
    re-checked (a batch with a different grid spacing on the same grid is
    not detected). Use it for a training stream whose loader serves one
    grid, such as the fixed-grid water campaigns. The hard-weight checks and
    the weight total share one host transfer per evaluation.
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
        residual_smoothing: Optional[float] = None,
        fixed_grid: bool = False,
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
        if residual_smoothing is not None:
            residual_smoothing = nonnegative_scalar(
                residual_smoothing, "residual_smoothing",
            )
            if residual_smoothing == 0.0:
                residual_smoothing = None
        self.residual_smoothing = residual_smoothing
        self.fixed_grid = boolean(fixed_grid, "fixed_grid")
        # Derived runtime data only: transfers and the once-validated geometry
        # are absent from state_dict.
        self._smoothing_transfer_cache = {}
        self._smoothing_geometry = None
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
        weight_checks = None
        if (self.center_axis is None and self.residual_smoothing is None
                and torch.all(self.target_scale == 1)
                and torch.all(self.residual_scale == 1)):
            value = self.loss_fn(prediction, target)
        else:
            value, weight_checks = self._transformed_loss(
                prediction, target, outputs, batch,
            )
        return self._reduce_loss(
            value, prediction, outputs, batch, weight_checks,
        )

    def _transformed_loss(self, prediction, target, outputs, batch):
        target = target * self.target_scale.to(prediction)
        weights = None
        if self.weights_key is not None and (
                self.center_axis is not None or self.residual_smoothing is not None):
            weights = _resolve_weights(self.weights_key, outputs, batch, prediction)
        if self.center_axis is not None:
            prediction, target = center_fields(
                prediction, target, weights, self.center_axis,
            )
        scale = self.residual_scale.to(prediction)
        prediction = prediction * scale
        target = target * scale
        weight_checks = None
        if self.residual_smoothing is not None:
            prediction, target, weight_checks = self._smoothed_fields(
                prediction, target, weights, batch,
            )
        return self.loss_fn(prediction, target), weight_checks

    def _smoothed_fields(self, prediction, target, weights, batch):
        """Zero unselected entries, then smooth both fields on the grid."""

        grid_size, grid_spacing = self._smoothing_geometry_for(batch, prediction)
        weight_checks = None
        if weights is not None:
            if weights.shape != prediction.shape:
                raise ValueError("weights and prediction must have same shape")
            selected, weight_checks = _hard_selection(weights, prediction)
            zeros = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
            prediction = torch.where(selected, prediction, zeros)
            target = torch.where(selected, target, zeros)
        transfer = self._smoothing_transfer(
            grid_size, grid_spacing, prediction,
        )
        return (
            _smooth_on_grid(prediction, grid_size, transfer),
            _smooth_on_grid(target, grid_size, transfer),
            weight_checks,
        )

    def _smoothing_geometry_for(self, batch, field):
        """Return the batch geometry, validated once when ``fixed_grid`` is set."""

        if not self.fixed_grid:
            return _shared_grid_geometry(batch, field)
        cached = self._smoothing_geometry
        if cached is None:
            cached = _shared_grid_geometry(batch, field)
            self._smoothing_geometry = cached
        elif field.ndim < 2 or math.prod(cached[0]) != field.shape[1]:
            raise ValueError(
                "fixed_grid smoothing received a field whose number of grid "
                "points differs from the validated grid"
            )
        return cached

    def _smoothing_transfer(self, grid_size, grid_spacing, reference):
        """Return one lazily cached transfer for this geometry and tensor type."""

        key = (
            grid_size,
            tuple(float(value) for value in grid_spacing),
            self.residual_smoothing,
            reference.device.type,
            reference.device.index,
            reference.dtype,
        )
        transfer = self._smoothing_transfer_cache.get(key)
        if transfer is None:
            # One entry covers the ordinary fixed-grid training stream and
            # avoids retaining device tensors for every geometry ever seen.
            self._smoothing_transfer_cache.clear()
            transfer = _gaussian_transfer(
                grid_size, grid_spacing, self.residual_smoothing,
                device=reference.device, dtype=reference.dtype,
            )
            self._smoothing_transfer_cache[key] = transfer
        return transfer

    def _apply(self, fn, recurse=True):
        """Move registered state and discard device-specific derived tensors."""

        result = super()._apply(fn, recurse=recurse)
        self._smoothing_transfer_cache.clear()
        return result

    def _reduce_loss(
        self, value, prediction, outputs, batch, weight_checks=None,
    ):
        if self.weights_key is None:
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError("loss_fn must return one scalar tensor")
        else:
            weights = _resolve_weights(self.weights_key, outputs, batch, prediction)
            if weights.shape != prediction.shape:
                raise ValueError("weights and prediction must have same shape")
            if value.shape != prediction.shape:
                raise ValueError(
                    "weighted loss_fn must return one value per prediction"
                )
            weights = weights.detach().to(prediction)
            total_weight = weights.sum()
            if weight_checks is None:
                positive = total_weight.item() > 0.0
            else:
                # One device-to-host transfer serves the smoothing weight
                # checks and the weight total together.
                finite, hard, positive = torch.stack(
                    (weight_checks[0], weight_checks[1], total_weight > 0.0),
                ).tolist()
                if not finite:
                    raise ValueError("residual smoothing weights must be finite")
                if not hard:
                    raise ValueError("residual smoothing requires hard zero/one weights")
            if not positive:
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
    prediction: torch.Tensor,
) -> torch.Tensor:
    """Resolve weights and expand a shared trailing-component selection."""

    if key in outputs:
        weights = outputs[key]
    elif key in batch:
        weights = batch[key]
    else:
        raise KeyError(
            "batch and model outputs are missing weights '{}'".format(key)
        )
    if weights.shape == prediction.shape[:-1]:
        weights = weights.unsqueeze(-1).expand_as(prediction)
    return weights


def _hard_selection(
    weights: torch.Tensor,
    reference: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a Boolean support mask and its device-side validity checks.

    The checks, ``[all finite, all zero or one]``, stay on the device so the
    caller can read them together with its other host transfer; an
    ambiguous soft weight is rejected there rather than silently selected.
    """

    weights = weights.detach().to(reference)
    checks = torch.stack((
        torch.isfinite(weights).all(),
        torch.all((weights == 0) | (weights == 1)),
    ))
    return weights != 0, checks


def _shared_grid_geometry(
    batch: Dict[str, torch.Tensor],
    field: torch.Tensor,
) -> Tuple[Tuple[int, int, int], torch.Tensor]:
    """Return the one grid size and spacing shared by every field in a batch.

    The field must be shaped ``[n_fields, n_grid, ...]`` with ``n_grid`` the
    product of ``grid_size``; the flattened grid axis is assumed to follow
    the canonical C order that the data loaders produce.
    """

    for key in ("grid_size", "grid_spacing"):
        if key not in batch:
            raise KeyError(
                "batch is missing '{}', required for residual smoothing".format(key)
            )
    if field.ndim < 2:
        raise ValueError(
            "residual smoothing requires fields shaped [n_fields, n_grid, ...]"
        )
    grid_size = common_grid_size(batch["grid_size"], field.shape[:1])
    if math.prod(grid_size) != field.shape[1]:
        raise ValueError("grid_size product must match the number of grid points")
    spacing = torch.as_tensor(batch["grid_spacing"]).detach().cpu()
    if spacing.dtype == torch.bool or spacing.is_complex():
        raise TypeError("grid_spacing must contain real values")
    spacing = spacing.to(torch.float64).reshape(-1, 3)
    if spacing.shape[0] not in (1, field.shape[0]):
        raise ValueError("grid_spacing leading shape must match the fields")
    if not torch.all(spacing == spacing[0]).item():
        raise ValueError("all fields in one batch must share grid_spacing")
    return grid_size, grid_spacing_tensor(spacing[0], dtype=torch.float64)


def _gaussian_transfer(
    grid_size: Tuple[int, int, int],
    grid_spacing: torch.Tensor,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``exp(-sigma^2 k^2 / 2)`` on the real-FFT wavevector grid."""

    axes = [
        2.0 * math.pi * torch.fft.fftfreq(
            n, d=float(grid_spacing[axis]), dtype=torch.float64,
        )
        for axis, n in enumerate(grid_size)
    ]
    axes[-1] = axes[-1][: grid_size[-1] // 2 + 1]
    kx, ky, kz = torch.meshgrid(*axes, indexing="ij")
    transfer = torch.exp(-0.5 * sigma ** 2 * (kx.square() + ky.square() + kz.square()))
    return transfer.to(device=device, dtype=dtype)


def _smooth_on_grid(
    field: torch.Tensor,
    grid_size: Tuple[int, int, int],
    transfer: torch.Tensor,
) -> torch.Tensor:
    """Convolve ``[n_fields, n_grid, ...]`` with the periodic transfer function."""

    shape = field.shape
    grid = field.reshape(shape[0], *grid_size, *shape[2:])
    spectrum = torch.fft.rfftn(grid, dim=(1, 2, 3))
    spectrum = spectrum * transfer.reshape(1, *transfer.shape, *([1] * (field.ndim - 2)))
    smoothed = torch.fft.irfftn(spectrum, s=grid_size, dim=(1, 2, 3))
    return smoothed.reshape(shape)
