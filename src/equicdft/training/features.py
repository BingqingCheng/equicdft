"""Population statistics of local readout inputs and the scales they define.

The invariants one CACE pipeline feeds its :class:`~equicdft.LocalReadout`
span orders of magnitude: smeared densities and the polarization centre
channel are order one, gradient-type moments of the same fields orders of
magnitude smaller. An unscaled network fits along the largest inputs first.
:func:`local_feature_scale` measures the spread of every input over the
training voxels and :func:`standardize_local_features` installs it as the
readout's ``feature_scale`` before a fit, so every invariant enters with unit
spread. The scale is data-derived like a polarization normalization and lives
in the model configuration.
"""

from typing import Any, Dict, Iterable, Mapping, Optional, Union

import torch
from torch import nn

from .._device import module_device, move_batch
from .._validation import boolean, finite_scalar
from ..functional.readout import LocalReadout

__all__ = [
    "local_feature_statistics",
    "local_feature_scale",
    "local_feature_offset",
    "standardize_local_features",
]


class _Welford:
    """Streaming mean and variance of a ``[n_samples, n_features]`` stream."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=torch.float64, device="cpu")
        if values.ndim != 2:
            raise ValueError("feature samples must be [n_samples, n_features]")
        count = int(values.shape[0])
        if count == 0:
            return
        mean = values.mean(dim=0)
        m2 = (values - mean).square().sum(dim=0)
        if self.mean is None:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        total = self.count + count
        delta = mean - self.mean
        self.mean = self.mean + delta * (count / total)
        self.m2 = self.m2 + m2 + delta.square() * (self.count * count / total)
        self.count = total

    def result(self) -> Dict[str, Any]:
        if self.mean is None:
            raise ValueError("no selected voxels reached this readout")
        variance = self.m2 / self.count   # population variance, as declared
        return {
            "count": self.count,
            "mean": self.mean.clone(),
            "std": variance.clamp_min(0.0).sqrt(),
        }


def _local_readouts(model: nn.Module) -> Dict[str, LocalReadout]:
    readouts = {
        name: module for name, module in model.named_modules()
        if isinstance(module, LocalReadout)
    }
    if not readouts:
        raise ValueError("the model has no LocalReadout")
    return readouts


def _selection(
    outputs: Mapping[str, Any],
    features: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Return the ``[..., n_grid]`` Boolean selection of energy centres."""

    mask = outputs.get("local_energy_mask")
    if mask is None:
        return None
    if mask.dtype != torch.bool or mask.shape[:-1] != features.shape[:-1]:
        raise ValueError(
            "local_energy_mask does not match the local features' field and "
            "grid shape"
        )
    return mask.any(dim=-1)


@torch.no_grad()
def local_feature_statistics(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    *,
    device: Optional[Union[str, torch.device]] = None,
    use_voxel_mask: bool = True,
    model_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return per-feature statistics of every local readout's raw inputs.

    The model is evaluated on each batch (moved to ``device``, by default the
    model's device; with every derivative output switched off unless
    ``model_kwargs`` says otherwise) and the input each
    :class:`~equicdft.LocalReadout` receives, before its own
    ``feature_scale``, is accumulated over the selected energy centres: the model's ``local_energy_mask`` when
    ``use_voxel_mask`` is true (a voxel counts when any component is
    selected), every voxel otherwise. The result maps the readout's module
    name, as in ``model.named_modules()``, to ``{"count", "mean", "std"}``
    with ``mean`` and ``std`` float64 tensors of the readout's input width
    (``std`` the population value, divided by ``n``). Statistics depend on the
    feature modules' current parameters, so compute them on the model that
    will be fitted, before the fit.
    """

    use_voxel_mask = boolean(use_voxel_mask, "use_voxel_mask")
    readouts = _local_readouts(model)
    device = module_device(model) if device is None else torch.device(device)
    accumulators = {name: _Welford() for name in readouts}
    captured: Dict[str, torch.Tensor] = {}
    handles = [
        readout.register_forward_hook(
            lambda module, inputs, output, name=name: captured.__setitem__(
                name, inputs[0]
            )
        )
        for name, readout in readouts.items()
    ]
    kwargs = dict(model_kwargs or {})
    # Only the readout inputs are needed: skip the derivative outputs the
    # model would otherwise differentiate for.
    for key in ("compute_c1", "compute_c2", "compute_polarization_derivative", "compute_local_mu"):
        kwargs.setdefault(key, False)
    was_training = model.training
    model.eval()
    try:
        for batch in batches:
            captured.clear()
            moved = move_batch(batch, device)
            outputs = model(moved, use_voxel_mask=use_voxel_mask, **kwargs)
            for name in readouts:
                features = captured.get(name)
                if features is None:
                    continue
                selected = _selection(outputs, features)
                flat = features.reshape(-1, features.shape[-1])
                if selected is not None:
                    flat = flat[selected.reshape(-1)]
                accumulators[name].update(flat)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return {name: accumulator.result() for name, accumulator in accumulators.items()}


def local_feature_scale(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    *,
    relative_floor: float = 1.0e-4,
    device: Optional[Union[str, torch.device]] = None,
    use_voxel_mask: bool = True,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    statistics: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, torch.Tensor]:
    """Return one ``feature_scale`` per local readout from the training data.

    Each feature's scale is its population standard deviation over the
    selected voxels (:func:`local_feature_statistics`). Features whose spread
    is below ``relative_floor`` times the largest spread of the same readout
    keep a scale of one: a constant input (the temperature feature at the
    reference temperature, an invariant that vanishes on the data) must not
    be amplified into noise. The default ``1e-4`` standardizes every
    invariant that carries information on the fits measured so far (the
    weakest polarization-gradient invariants sit near ``1e-3`` of the centre
    channel) and leaves degenerate ones alone. Pass ``statistics`` to reuse a
    previous accumulation instead of evaluating the model again.
    """

    relative_floor = finite_scalar(relative_floor, "relative_floor")
    if relative_floor <= 0 or relative_floor >= 1:
        raise ValueError("relative_floor must lie strictly between 0 and 1")
    if statistics is None:
        statistics = local_feature_statistics(
            model, batches, device=device, use_voxel_mask=use_voxel_mask,
            model_kwargs=model_kwargs,
        )
    scales = {}
    for name, entry in statistics.items():
        std = torch.as_tensor(entry["std"], dtype=torch.float64).reshape(-1)
        selected = _selected_features(std, relative_floor, name)
        scales[name] = torch.where(
            selected, std, torch.ones_like(std)
        )
    return scales


def _selected_features(
    std: torch.Tensor,
    relative_floor: float,
    name: str,
) -> torch.Tensor:
    """Return features whose spread is large enough to standardize."""

    if std.numel() == 0 or not torch.isfinite(std).all():
        raise ValueError("feature statistics for {} are not finite".format(name))
    largest = std.max()
    if largest <= 0:
        raise ValueError(
            "every input of {} is constant over the data; nothing to "
            "standardize".format(name)
        )
    return std > relative_floor * largest


def local_feature_offset(
    statistics: Mapping[str, Mapping[str, Any]],
    *,
    relative_floor: float = 1.0e-4,
) -> Dict[str, torch.Tensor]:
    """Return one ``feature_offset`` per local readout: the population mean.

    Features excluded by the same ``relative_floor`` rule as
    :func:`local_feature_scale` keep a zero offset, so the transform leaves
    them exactly as they are. Selection is recomputed from the spread rather
    than inferred from a numerical scale of one: a selected feature may
    legitimately have unit standard deviation.
    """

    relative_floor = finite_scalar(relative_floor, "relative_floor")
    if relative_floor <= 0 or relative_floor >= 1:
        raise ValueError("relative_floor must lie strictly between 0 and 1")
    offsets = {}
    for name, entry in statistics.items():
        std = torch.as_tensor(entry["std"], dtype=torch.float64).reshape(-1)
        mean = torch.as_tensor(entry["mean"], dtype=torch.float64).reshape(-1)
        if mean.shape != std.shape or not torch.isfinite(mean).all():
            raise ValueError("feature means for {} do not match the spread".format(name))
        selected = _selected_features(std, relative_floor, name)
        offsets[name] = torch.where(selected, mean, torch.zeros_like(mean))
    return offsets


def standardize_local_features(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    *,
    relative_floor: float = 1.0e-4,
    center: bool = False,
    keep_function: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    use_voxel_mask: bool = True,
    model_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Install a data-derived ``feature_scale`` (and offset) on every local readout.

    Computes :func:`local_feature_scale` and, with ``center=True``,
    :func:`local_feature_offset`, then calls
    :meth:`LocalReadout.set_feature_scale` on each readout. Before a fit,
    leave ``keep_function=False`` so the freshly initialized network starts
    from unit-spread (and zero-mean) inputs; on a fitted model,
    ``keep_function=True`` keeps its function and only reparametrizes it.
    Returns ``{name: {"scale": ..., "offset": ...}}`` by readout name
    (``offset`` is ``None`` without centring); scale and offset are now part
    of the model's state dict and of ``model.to_config()``.
    """

    keep_function = boolean(keep_function, "keep_function")
    center = boolean(center, "center")
    readouts = _local_readouts(model)
    statistics = local_feature_statistics(
        model, batches, device=device, use_voxel_mask=use_voxel_mask,
        model_kwargs=model_kwargs,
    )
    scales = local_feature_scale(
        model, batches, relative_floor=relative_floor, statistics=statistics,
    )
    offsets = (
        local_feature_offset(statistics, relative_floor=relative_floor)
        if center else {}
    )
    result = {}
    for name, scale in scales.items():
        offset = offsets.get(name)
        readouts[name].set_feature_scale(scale, offset, keep_function=keep_function)
        result[name] = {"scale": scale, "offset": offset}
    return result
