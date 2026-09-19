"""Common interface between the model and its additive free-energy readouts.

:class:`GridCACEModel` evaluates every configured :class:`EnergyReadout` on
one :class:`ReadoutContext`, sums the returned scalars and differentiates the
total. The context carries the primitives of one batch of density fields and
composes the derived inputs (normalized densities, state vectors, local
feature vectors) lazily, so readouts that share an input pay for it once and
no readout needs to know how another one assembles its inputs.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .geometry.grid import (
    grid_spacing_tensor,
    voxel_volume as compute_voxel_volume,
)

__all__ = [
    "FREE_ENERGY_MODES",
    "ReadoutContext",
    "EnergyReadout",
]


FREE_ENERGY_MODES = ("beta", "physical")


@dataclass(eq=False)
class ReadoutContext:
    r"""Everything a readout may read about one batch of density fields.

    Primitives are set once per forward pass by the model. Optional
    primitives that the model did not supply are ``None``; a readout asks for
    them with :meth:`require`, which raises a ``KeyError`` naming the missing
    quantity. Composed inputs are cached on first use.

    Parameters
    ----------
    rho
        Physical density with shape ``[..., n_grid, n_types]``. Functional
        derivatives are taken with respect to this tensor.
    normalized_temperature
        ``T / T_ref`` with the leading field shape ``[...]``.
    voxel_volume
        Quadrature volume ``Delta V`` of one grid point, scalar or ``[...]``.
    mean_density
        Positive density scale used by every normalized density input.
    grid_spacing
        Cartesian spacing ``[3]`` of the regular grid.
    grid_size
        Integer grid dimensions ``[3]`` (or one row per field).
    beta
        ``1 / (k_B T)`` with the leading field shape; the model's unit
        contract, not a batch field.
    reference_energy
        ``k_B T_ref`` used by readouts that evaluate physical energies.
    free_energy_mode
        ``"beta"`` or ``"physical"``, the convention of the summed energy.
    invariants
        Flattened invariant environment features ``[..., n_grid, n_B]``,
        present when a readout declares ``requires_local_features``.
    center_density
        Normalized central density channels ``[..., n_grid, n_channels]``
        that the local feature vector starts with when the feature module
        separates the central grid point from its neighbor moments.
    dipole_density
        Independent polar vector field ``[..., n_grid, n_types, 3]``.
    local_density_index, grid_positions, grid_center, excluded_mask
        Grid geometry passed through from the batch when available.
    """

    rho: torch.Tensor
    normalized_temperature: torch.Tensor
    voxel_volume: Union[float, torch.Tensor]
    mean_density: Optional[Union[float, torch.Tensor]] = None
    grid_spacing: Optional[torch.Tensor] = None
    grid_size: Optional[torch.Tensor] = None
    beta: Optional[torch.Tensor] = None
    reference_energy: Optional[torch.Tensor] = None
    free_energy_mode: str = "beta"
    invariants: Optional[torch.Tensor] = None
    center_density: Optional[torch.Tensor] = None
    dipole_density: Optional[torch.Tensor] = None
    local_density_index: Optional[torch.Tensor] = None
    grid_positions: Optional[torch.Tensor] = None
    grid_center: Optional[torch.Tensor] = None
    excluded_mask: Optional[torch.Tensor] = None
    _cache: Dict[str, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        rho = self.rho
        if not torch.is_tensor(rho) or rho.ndim < 2:
            raise ValueError("rho must have shape [..., n_grid, n_types]")
        if not torch.is_tensor(self.normalized_temperature):
            raise TypeError("normalized_temperature must be a torch.Tensor")
        if self.normalized_temperature.shape != rho.shape[:-2]:
            raise ValueError(
                "normalized_temperature must match rho's leading field shape"
            )
        self.voxel_volume = torch.as_tensor(
            self.voxel_volume, dtype=rho.dtype, device=rho.device,
        )
        if self.mean_density is not None:
            scale = torch.as_tensor(
                self.mean_density, dtype=rho.dtype, device=rho.device,
            )
            if scale.numel() != 1:
                raise ValueError("mean_density must be a positive scalar")
            self.mean_density = scale.reshape(())
        if self.free_energy_mode not in FREE_ENERGY_MODES:
            raise ValueError("free_energy_mode must be 'beta' or 'physical'")

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Any],
        *,
        grid_spacing: Union[float, Sequence[float], torch.Tensor],
        mean_temperature: Union[float, torch.Tensor] = 1.0,
        boltzmann_constant: Union[float, torch.Tensor] = 1.0,
        mean_density: Optional[Union[float, torch.Tensor]] = None,
        free_energy_mode: str = "beta",
    ) -> "ReadoutContext":
        """Build the context of one GridData-like batch under a unit contract.

        ``data`` supplies ``rho`` and ``temperature`` (scalar for one field or
        with rho's leading batch shape) and, optionally, the geometry fields
        ``grid_size``, ``dipole_density``, ``local_density_index``,
        ``grid_positions``, ``grid_center`` and ``excluded_mask``. The fixed
        model geometry and thermodynamic scales are passed explicitly; this
        is what :class:`equicdft.model.GridCACEModel` does on every forward
        pass and what a standalone readout evaluation needs.
        """

        rho = data["rho"]
        temperature = data["temperature"].to(device=rho.device, dtype=rho.dtype)
        if temperature.shape != rho.shape[:-2]:
            raise ValueError(
                "temperature must be scalar for one field or have the "
                "same leading batch shape as rho"
            )
        spacing = torch.as_tensor(grid_spacing)
        if not spacing.is_floating_point():
            spacing = spacing.to(dtype=rho.dtype)
        spacing = grid_spacing_tensor(spacing, dtype=spacing.dtype)
        reference_temperature = torch.as_tensor(mean_temperature)
        boltzmann = torch.as_tensor(boltzmann_constant)
        return cls(
            rho=rho,
            normalized_temperature=temperature / reference_temperature.to(rho),
            voxel_volume=compute_voxel_volume(spacing).to(rho),
            mean_density=mean_density,
            grid_spacing=spacing.to(rho),
            # Physical analytical readouts use this unit contract, not a
            # possibly stale beta field supplied by a caller.
            beta=1.0 / (boltzmann.to(rho) * temperature),
            reference_energy=(boltzmann * reference_temperature).to(rho),
            free_energy_mode=free_energy_mode,
            grid_size=data.get("grid_size"),
            dipole_density=data.get("dipole_density"),
            local_density_index=data.get("local_density_index"),
            grid_positions=data.get("grid_positions"),
            grid_center=data.get("grid_center"),
            excluded_mask=data.get("excluded_mask"),
        )

    def require(self, name: str) -> torch.Tensor:
        """Return one optional primitive, or raise when the model lacks it."""

        value = getattr(self, name)
        if value is None:
            raise KeyError(
                "this readout requires '{}', which the model did not supply; "
                "check the data fields and model configuration".format(name)
            )
        return value

    def _cached(
        self,
        name: str,
        compose: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        value = self._cache.get(name)
        if value is None:
            value = compose()
            self._cache[name] = value
        return value

    def normalized_rho(self) -> torch.Tensor:
        """Return ``rho / mean_density`` with rho's shape."""

        return self._cached(
            "normalized_rho",
            lambda: self.rho / self.require("mean_density"),
        )

    def temperature_feature(self) -> torch.Tensor:
        """Return the normalized temperature broadcast to ``[..., n_grid, 1]``."""

        return self._cached(
            "temperature_feature",
            lambda: self.normalized_temperature[..., None, None].expand(
                *self.rho.shape[:-1], 1,
            ),
        )

    def local_features(self) -> torch.Tensor:
        """Return ``[center_density?, invariants, T/T_ref]`` per grid point."""

        def compose() -> torch.Tensor:
            blocks = []
            if self.center_density is not None:
                blocks.append(self.center_density)
            blocks.append(self.require("invariants"))
            blocks.append(self.temperature_feature())
            return torch.cat(blocks, dim=-1)

        return self._cached("local_features", compose)

    def pointwise_state(self) -> torch.Tensor:
        """Return ``[rho / mean_density, T/T_ref]`` per grid point."""

        return self._cached(
            "pointwise_state",
            lambda: torch.cat(
                (self.normalized_rho(), self.temperature_feature()), dim=-1,
            ),
        )

    def state_features(self) -> torch.Tensor:
        """Return the global state ``[T/T_ref, <rho_a>/mean_density]`` per field.

        The mean density keeps its connection to ``rho``, so readouts of this
        state remain part of the functional derivative.
        """

        return self._cached(
            "state_features",
            lambda: torch.cat(
                (
                    self.normalized_temperature[..., None],
                    self.rho.mean(dim=-2) / self.require("mean_density"),
                ),
                dim=-1,
            ),
        )


class EnergyReadout(nn.Module):
    """Readout supplying a scalar functional and optional observable outputs.

    Subclasses declare what they need through the class attributes below and
    implement :meth:`energy`. Readouts that also return observables override
    :meth:`energy_and_outputs`, which the model calls.

    ``requires_local_features``
        The readout reads ``context.local_features()``; the model then
        evaluates the invariant environment features.
    ``requires_local_density_index``
        The readout gathers explicit periodic neighborhoods and exposes a
        ``cutoff_grid`` that must agree with the feature module.
    ``requires_dipole_density``
        The readout depends on the independent polarization field.
    ``n_types``
        Number of physical density components the readout is built for, or
        ``None`` for readouts that accept any component count.

    Readouts whose first layer consumes the model's local feature vector
    learn its width from :meth:`bind_input_size`, which the model calls at
    construction; until then :attr:`is_bound` is false. Every other readout
    is complete when constructed and inherits the no-op hook below.
    """

    requires_local_features = False
    requires_local_density_index = False
    requires_dipole_density = False
    n_types: Optional[int] = None

    @property
    def is_bound(self) -> bool:
        """Whether every parameter of this readout exists with its final shape."""

        return True

    def bind_input_size(self, n_features: int) -> None:
        """Fix the local feature width; a no-op for readouts without one."""

    def wrapped_readouts(self) -> Tuple["EnergyReadout", ...]:
        """Return readouts whose energy this readout already includes.

        The model refuses to list such a readout beside its wrapper, which
        would count its energy twice.
        """

        return ()

    def energy(self, context: ReadoutContext) -> torch.Tensor:
        """Return one scalar energy per complete density field."""

        raise NotImplementedError

    def energy_and_outputs(
        self,
        context: ReadoutContext,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return energy and observables from one evaluation.

        Scalar-only readouts need not override this method. Output names must
        be unique across readouts and must not shadow model outputs.
        """

        return self.energy(context), {}
