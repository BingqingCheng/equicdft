"""Explicit, nonmutating resolution transfer for scalar Cartesian models.

This is an inference/model-construction operation, not density interpolation.
Reference normalization is an exploratory signed-kernel extension; strict
center exclusion remains the default. Geometry is structural configuration,
while every radial exponent remains a live fitted parameter.
"""

import copy
import math
from typing import Optional, Sequence, Union

import torch

from .functional.ideal import IdealGas
from .functional.readout import BulkReadout, LocalReadout, LongRangeReadout
from .functional.semilocal import LDAReadout
from .geometry.grid import grid_spacing_tensor
from .representation.cartesian import CartesianAFeatures
from .representation.message import BChiMessage
from .serialization import default_dtype

__all__ = ["regrid_model"]


def _isotropic_spacing(value, name):
    raw = torch.as_tensor(value)
    if raw.dtype == torch.bool or raw.is_complex():
        raise TypeError(name + " must contain real positive lengths")
    spacing = grid_spacing_tensor(value, dtype=torch.float64, device="cpu")
    if not torch.equal(spacing, spacing[:1].expand_as(spacing)):
        raise ValueError(name + " must be isotropic for Cartesian regridding")
    return float(spacing[0])


def _validate_kernel(owner, label):
    if owner.radial_basis not in ("none", "gaussian", "shared"):
        raise ValueError(label + " requires an untransformed Gaussian or equal-weight basis")
    if getattr(owner, "radial_transform", None) is not None:
        raise ValueError(label + " radial transforms are not supported by regrid")
    if getattr(owner, "include_polarization", False):
        raise ValueError(label + " polarization is not supported by regrid")
    if owner.radial_basis == "shared":
        return
    if getattr(owner, "trainable_radial_centers", False):
        raise ValueError(label + " trainable radial centers are not supported by regrid")
    if torch.any(owner.radial_centers.detach() != 0).item():
        raise ValueError(label + " requires zero-centered Gaussian primitives")


def _validate_source_geometry(features):
    """Reject unrecorded geometry edits; allow historical float32 rounding."""
    anchor = features.mean_density
    with default_dtype(anchor.dtype):
        configured = CartesianAFeatures.from_config(features.to_config())
    configured = configured.to(device=anchor.device, dtype=anchor.dtype)
    for name in ("local_density_positions", "powers", "squared_distances", "neighbor_mask"):
        if not torch.equal(getattr(features, name), getattr(configured, name)):
            raise ValueError("a_features " + name + " differs from its configuration; "
                             "unrecorded geometry edits cannot be regridded")
    # A model cast to double can retain moments originally built in float32.
    # Allow eight rounding units per degree, but no absolute error at zeros.
    epsilon = max(torch.finfo(anchor.dtype).eps, torch.finfo(torch.float32).eps)
    tolerance = 8 * max(1, features.max_power) * epsilon
    if (features.monomial_values.shape != configured.monomial_values.shape
            or not torch.allclose(features.monomial_values, configured.monomial_values,
                                  rtol=tolerance, atol=0)):
        raise ValueError("a_features monomial_values differ from configuration beyond "
                         "conversion precision; record their units with monomial_scale")


def _validate_model(model):
    features = model.a_features
    if type(features) is not CartesianAFeatures:
        raise ValueError("regrid requires top-level CartesianAFeatures")
    if not features.separate_center:
        raise ValueError("regrid currently requires separate_center=True")
    if "stencil_basis" in features.__dict__:
        raise ValueError("regrid does not accept a runtime-replaced stencil_basis")
    _validate_kernel(features, "a_features")
    if model.requires_dipole_density:
        raise ValueError("regrid currently supports scalar density models only")
    if model.ideal is not None and type(model.ideal) is not IdealGas:
        raise ValueError("regrid currently supports only the scalar IdealGas reference")
    supported = (LDAReadout, LocalReadout, BulkReadout, LongRangeReadout)
    for readout in model.readout:
        if type(readout) not in supported:
            raise ValueError("regrid does not support readout " + type(readout).__name__)
    for index, message in enumerate(model.message_layers):
        if type(message) is not BChiMessage:
            raise ValueError("regrid requires BChiMessage layers")
        if "stencil_basis" in message.__dict__:
            raise ValueError("regrid does not accept a runtime-replaced message kernel")
        _validate_kernel(message, "message_layers[{}]".format(index))
    # Reconstruction validates reference units/radius and checks the stored
    # geometry, including its spherical center-excluded mask, in one place.
    _validate_source_geometry(features)


def _reference_geometry(features, current_spacing):
    existing = getattr(features, "normalization_reference", None)
    if existing is not None:
        reference = copy.deepcopy(existing)
        ratio = float(reference["spacing_ratio"])
        expected_spacing = float(reference["grid_spacing"]) * ratio
        if not math.isclose(expected_spacing, current_spacing, rel_tol=1e-12, abs_tol=0):
            raise ValueError("normalization reference does not match the model grid spacing")
        return reference
    return dict(
        grid_spacing=current_spacing,
        offsets=features.local_density_positions.detach().cpu().tolist(),
        neighbor_mask=features.neighbor_mask.detach().cpu().tolist(),
        cutoff_grid=features.cutoff_grid,
        stencil_radius=float(getattr(features, "stencil_radius", features.cutoff_grid)),
        monomial_scale=float(getattr(features, "monomial_scale", 1.0)),
        coordinate_scaling=features.coordinate_scaling,
        spacing_ratio=1.0,
    )


def _rebuild_features(features, config, spacing_ratio):
    """Reconstruct geometry while retaining exact fitted state and flags."""
    anchor = features.mean_density
    with default_dtype(anchor.dtype):
        rebuilt = CartesianAFeatures.from_config(config)
    rebuilt = rebuilt.to(device=anchor.device, dtype=anchor.dtype)
    state = features.state_dict()
    geometry = ("local_density_positions", "squared_distances", "powers", "monomial_values")
    if not all(torch.equal(getattr(features, name), getattr(rebuilt, name))
               for name in (*geometry, "neighbor_mask")):
        for name in geometry:
            state[name] = getattr(rebuilt, name)
    # Only the local exponent's grid-unit representation changes. Assign new
    # tensors: state_dict values can share storage with the source model.
    if spacing_ratio != 1.0:
        if "log_radial_exponents" in state:
            state["log_radial_exponents"] = state["log_radial_exponents"] + 2.0 * math.log(spacing_ratio)
        else:
            state["fixed_radial_exponents"] = state["fixed_radial_exponents"] * spacing_ratio ** 2
    # Strict loading already checks all fitted-state keys and tensor shapes.
    rebuilt.load_state_dict(state, strict=True)
    old_parameters = dict(features.named_parameters())
    for name, parameter in rebuilt.named_parameters():
        parameter.requires_grad_(old_parameters[name].requires_grad)
    old_modules = dict(features.named_modules())
    for name, module in rebuilt.named_modules():
        module.training = old_modules[name].training
    return rebuilt


def regrid_model(
    model,
    grid_spacing: Union[float, Sequence[float], torch.Tensor],
    *,
    center_normalization: str = "strict",
    convolution_backend: Optional[str] = None,
):
    """Return a scalar Cartesian model evaluated at another isotropic spacing.

    Physical cutoff, Gaussian widths, monomial coordinate units and all
    nonlocal/pointwise parameters are preserved. ``"strict"`` retains the
    original unit-sum center-excluding descriptor. ``"reference"`` adds the
    signed central compensation relative to the original reference geometry.
    This changes the off-grid model, not its learned parameters or input data.

    The returned model and reference metadata are independent of ``model``.
    Reference geometry is retained through save/load and repeated regridding;
    it is never reset to an intermediate grid. Unsupported grid-dependent
    readouts or radial families raise instead of silently changing physics.
    ``convolution_backend`` may be ``"gather"`` or ``"fft"``; omission retains
    each module's current backend. Input fields must be supplied on the new
    grid; no field interpolation, resampling, solve or retraining is performed.
    """
    if not isinstance(center_normalization, str):
        raise TypeError("center_normalization must be 'strict' or 'reference'")
    center_normalization = center_normalization.lower()
    if center_normalization not in ("strict", "reference"):
        raise ValueError("center_normalization must be 'strict' or 'reference'")
    if convolution_backend is not None and convolution_backend not in ("gather", "fft"):
        raise ValueError("convolution_backend must be 'gather', 'fft', or None")
    _validate_model(model)
    old_spacing = _isotropic_spacing(model.grid_spacing, "model grid_spacing")
    _isotropic_spacing(grid_spacing, "grid_spacing")
    # Geometry must use the spacing the returned model can actually represent.
    target_spacing = grid_spacing_tensor(
        grid_spacing, dtype=model.grid_spacing.dtype, device=model.grid_spacing.device)
    new_spacing = _isotropic_spacing(target_spacing, "grid_spacing")
    features = model.a_features
    reference = _reference_geometry(features, old_spacing)
    radius = float(reference["stencil_radius"]) * float(reference["grid_spacing"]) / new_spacing
    if radius < 1.0 - 1e-12:
        raise ValueError("grid_spacing leaves no noncentral stencil neighbors inside the physical cutoff")
    rounded_radius = round(radius)
    if math.isclose(radius, rounded_radius, rel_tol=1e-12, abs_tol=1e-12):
        radius = float(rounded_radius)
    # The allocation cutoff can be larger than the active radius. Preserve
    # that original geometry at native spacing instead of shrinking it to
    # ceil(radius), including when returning through an intermediate grid.
    reference_spacing = float(reference["grid_spacing"])
    if new_spacing == old_spacing:
        cutoff = features.cutoff_grid
        radius = float(getattr(features, "stencil_radius", cutoff))
    elif new_spacing == reference_spacing:
        cutoff = int(reference["cutoff_grid"])
        radius = float(reference["stencil_radius"])
    else:
        cutoff = max(1, math.ceil(radius))
    reference["spacing_ratio"] = new_spacing / float(reference["grid_spacing"])
    monomial_scale = float(reference["monomial_scale"]) * reference["spacing_ratio"]
    if features.coordinate_scaling == "cutoff":
        monomial_scale *= cutoff / int(reference["cutoff_grid"])
    if new_spacing == old_spacing:
        monomial_scale = float(getattr(features, "monomial_scale", 1.0))
    config = copy.deepcopy(features.to_config())
    config.update(cutoff_grid=cutoff, stencil_radius=radius, monomial_scale=monomial_scale,
                  center_normalization=center_normalization,
                  normalization_reference=reference)
    if convolution_backend is not None:
        config["convolution_backend"] = convolution_backend
    ratio = new_spacing / old_spacing
    result = copy.deepcopy(model)
    result.a_features = _rebuild_features(features, config, ratio)
    result.grid_spacing = target_spacing
    with torch.no_grad():
        for message in result.message_layers:
            if message.radial_basis == "gaussian" and ratio != 1.0:
                if message.trainable_radial_exponents:
                    message.log_radial_exponents.add_(2.0 * math.log(ratio))
                else:
                    message.fixed_radial_exponents.mul_(ratio ** 2)
            if convolution_backend is not None:
                message.convolution_backend = convolution_backend
            message.bind_geometry(result.a_features)
    return result
