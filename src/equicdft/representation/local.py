"""Shared A/B/message pipeline for model-level and independent CACE branches."""

import torch

from .message import BChiMessage


_A_CONTRACT = (
    "n_types", "cutoff_grid", "mean_density", "n_radial_channels",
    "n_output_channels", "separate_center", "convolution_backend",
    "include_polarization", "requires_dipole_density", "transform_density",
)
_MESSAGE_CONTRACT = (
    "stencil_basis", "local_density_positions", "polarization_fields",
    "bessel_stencil_basis", "max_power", "powers", "coordinate_scaling",
)


def validate_features(a_features, b_features, message_layers):
    """Validate a feature pair and bind its messages; return a plain list."""
    if (a_features is None) != (b_features is None):
        raise ValueError("a_features and b_features must be supplied together")
    if isinstance(message_layers, BChiMessage):
        raise TypeError("message_layers must be a sequence of BChiMessage modules")
    messages = [] if message_layers is None else list(message_layers)
    if not all(isinstance(module, BChiMessage) for module in messages):
        raise TypeError("every message layer must be a BChiMessage module")
    if a_features is None:
        if messages:
            raise ValueError("a_features and b_features are required by message_layers")
        return messages
    for module, contract, label in (
        (a_features, _A_CONTRACT + (_MESSAGE_CONTRACT if messages else ()), "a_features"),
        (b_features, ("n_features", "validate_a_features"), "b_features"),
    ):
        missing = [name for name in contract if not hasattr(module, name)]
        if missing:
            raise TypeError("{} ({}) does not provide {}".format(
                label, type(module).__name__, ", ".join(missing),
            ))
    b_features.validate_a_features(a_features)
    for module in messages:
        if module.include_polarization != a_features.include_polarization:
            raise ValueError("message layers and A/B include_polarization must match")
        expected = (a_features.n_radial_channels, b_features.n_features,
                    a_features.n_output_channels)
        actual = (module.n_radial_channels, module.n_invariant_features, module.n_channels)
        if actual != expected:
            raise ValueError("message layer dimensions {} do not match local feature dimensions {}".format(
                actual, expected,
            ))
        module.bind_geometry(a_features)
    return messages


def local_feature_width(a_features, b_features, n_messages):
    """Width of [center_density?, B0, ..., Bn, T/T_ref]."""
    invariant_width = (a_features.n_radial_channels * b_features.n_features
                       * a_features.n_output_channels)
    center_width = (a_features.n_output_channels
                    if a_features.separate_center and not a_features.include_polarization else 0)
    return center_width + (1 + n_messages) * invariant_width + 1


def local_invariants(a_features, b_features, messages, data, polarization_scale):
    """Evaluate one independent pipeline without changing physical inputs."""
    grid_rank = data["rho"].ndim - 1
    A = (a_features(data, polarization_scale=polarization_scale)
         if a_features.include_polarization else a_features(data))
    B = b_features(A)
    levels = [B.flatten(start_dim=grid_rank)]
    if messages:
        shared_basis = a_features.stencil_basis()
        message_fields = {}
        if a_features.include_polarization:
            fields, scales = a_features.polarization_fields(
                data, polarization_scale=polarization_scale,
            )
            message_fields = {
                "normalized_fields": fields / scales,
                "separate_center": a_features.separate_center,
            }
        for message in messages:
            A = message(
                B, data.get("local_density_index"),
                message.stencil_basis(a_features, shared_basis),
                grid_positions=data.get("grid_positions"),
                grid_size=data.get("grid_size"),
                stencil_positions=a_features.local_density_positions,
                **message_fields,
            )
            B = b_features(A)
            levels.append(B.flatten(start_dim=grid_rank))
    return torch.cat(levels, dim=-1)


def needs_neighbor_indices(a_features, messages):
    """Whether this pipeline contains an explicit-gather convolution."""
    return a_features.convolution_backend == "gather" or any(
        message.convolution_backend == "gather" for message in messages
    )
