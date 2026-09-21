"""Independent local CACE excess-energy branches."""

from dataclasses import replace

from torch import nn

from .._config import Configurable, build, constructor_arguments, make_config, register
from ..representation.local import (
    local_feature_width, local_invariants, needs_neighbor_indices, validate_features,
)
from .base import EnergyReadout
from .readout import LocalReadout

__all__ = ["CACEBranch"]


@register
class CACEBranch(EnergyReadout, Configurable):
    """One A → B → optional messages → LocalReadout excess-energy pipeline.

    Put independent branches in ``GridCACEModel(readout=[...])``. They share
    the parent's physical fields and normalization, not invariant features or
    trainable modules. Ideal and global readouts belong to the parent model.
    The local readout is fully bound at construction; no warm-up is needed.
    """

    # A branch is built on the model's stencil, so every branch and any
    # top-level A features must agree on one cutoff_grid.
    shares_grid_stencil = True

    def __init__(self, a_features, b_features, readout, *, message_layers=None):
        super().__init__()
        if a_features is None or b_features is None:
            raise ValueError("CACEBranch requires a_features and b_features")
        if not isinstance(readout, LocalReadout):
            raise TypeError("CACEBranch readout must be a LocalReadout")
        messages = validate_features(a_features, b_features, message_layers)
        if readout.n_types != a_features.n_types:
            raise ValueError("branch features and readout must use the same n_types")
        freshly_built = not readout.is_bound
        readout.bind_input_size(
            local_feature_width(a_features, b_features, len(messages))
        )
        if freshly_built:
            readout.to(a_features.mean_density)
        self.a_features = a_features
        self.b_features = b_features
        self.message_layers = nn.ModuleList(messages)
        self.readout = readout

    @property
    def n_types(self):
        return self.a_features.n_types

    @property
    def cutoff_grid(self):
        return self.a_features.cutoff_grid

    @property
    def requires_local_density_index(self):
        return needs_neighbor_indices(self.a_features, self.message_layers)

    @property
    def requires_dipole_density(self):
        return self.a_features.requires_dipole_density

    @property
    def requires_polarization_scale(self):
        return getattr(
            self.a_features,
            "requires_polarization_scale",
            self.a_features.include_polarization,
        )

    @property
    def requires_alignment_scale(self):
        return getattr(self.a_features, "requires_alignment_scale", False)

    def wrapped_readouts(self):
        return (self.readout,)

    def energy(self, context):
        """Integrate this branch only, with a private local-feature context."""
        data = {
            "rho": context.rho,
            "dipole_density": context.dipole_density,
            "grid_size": context.grid_size,
            "grid_positions": context.grid_positions,
            "local_density_index": context.local_density_index,
        }
        invariants = local_invariants(
            self.a_features, self.b_features, self.message_layers, data,
            context.polarization_scale,
            context.alignment_scale,
            context.dipole_magnitude,
            context.alignment_density_floor,
        )
        center = None
        if self.a_features.separate_center and not self.a_features.include_polarization:
            center = (
                self.a_features.transform_density(context.rho)
                / context.require("mean_density")
            )
        # replace constructs a fresh cache; branches never overwrite each
        # other's (or the parent's) invariant/center/temperature inputs.
        return self.readout.energy(
            replace(context, invariants=invariants, center_density=center)
        )

    def to_config(self):
        return make_config(
            self, a_features=self.a_features.to_config(),
            b_features=self.b_features.to_config(), readout=self.readout.to_config(),
            message_layers=[item.to_config() for item in self.message_layers],
        )

    @classmethod
    def from_config(cls, config):
        arguments = constructor_arguments(cls, config)
        for name in ("a_features", "b_features", "readout"):
            arguments[name] = build(arguments[name])
        arguments["message_layers"] = [
            build(item) for item in arguments.get("message_layers", [])
        ]
        return cls(**arguments)
