"""Gaussian metal electrodes coupled to a liquid Coulomb readout."""

import math
from typing import Any, Dict, Mapping, NamedTuple

import torch

from ._argument_checks import positive_scalar
from ._fourier_sites import FourierSites
from ._grid import common_grid_size, grid_spacing_tensor
from ._metal_data import normalize_metal_site_mapping
from .energy import EnergyReadout
from .readout import LongRangeReadout
from .reciprocal import _coulomb_values, _squared_wavevectors


def _field_vector(value, name):
    """Fixed Cartesian vector, shared or per field; retain input precision."""
    raw = torch.as_tensor(value)
    if (
        raw.dtype == torch.bool
        or raw.is_complex()
        or not torch.all(torch.isfinite(raw)).item()
    ):
        raise ValueError(name + " must contain real finite values")
    if raw.ndim < 1 or raw.shape[-1] != 3:
        raise ValueError(name + " must have shape [3] or [..., 3]")
    if raw.requires_grad:
        raise ValueError(name + " must be fixed, not a differentiable input")
    return raw.to(dtype=torch.float64).detach().clone()


def _periodic_field_coordinates(positions, cell):
    """Place one periodic electrode on its most compact Cartesian branch."""

    wrapped = torch.remainder(positions, cell)
    coordinates = torch.empty_like(wrapped)
    for axis in range(3):
        values = torch.sort(wrapped[:, axis]).values
        gaps = torch.cat((
            values[1:] - values[:-1],
            values[:1] + cell[axis] - values[-1:],
        ))
        gap = torch.argmax(gaps)
        cut = values[gap] + 0.5 * gaps[gap]
        center = torch.remainder(cut + 0.5 * cell[axis], cell[axis])
        displacement = wrapped[:, axis] - center
        coordinates[:, axis] = (
            displacement - cell[axis] * torch.round(displacement / cell[axis])
        )
    return coordinates


class _MetalSystem(NamedTuple):
    """Cached geometry and factorization for an electrode configuration."""

    interaction: torch.Tensor
    constraints: torch.Tensor
    factor: torch.Tensor
    pivots: torch.Tensor
    field_coordinates: torch.Tensor


class _MetalField(NamedTuple):
    """Per-field values consumed by the induced-charge solve."""

    liquid_total_charge: torch.Tensor
    liquid_potential: torch.Tensor
    amplitude: torch.Tensor
    energy_scale: torch.Tensor
    external_field: torch.Tensor


class MetalWall(EnergyReadout):
    r"""Extend a liquid Coulomb readout with Gaussian metal electrodes.

    MetalWall passes the electrode positions to the wrapped Coulomb readout,
    which returns the potential generated there by the liquid. MetalWall owns
    only the Gaussian metal-metal interaction, charge constraints, applied
    field, and induced-charge solve.
    """

    requires_state_features = True

    def __init__(
        self,
        liquid_coulomb: LongRangeReadout,
        metal_sites: Mapping[str, Any],
        metal_sigma: float,
        tolerance: float = 1.0e-6,
        *,
        external_field=(0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        if not isinstance(liquid_coulomb, LongRangeReadout):
            raise TypeError("liquid_coulomb must be a LongRangeReadout")
        if not liquid_coulomb.provides_coulomb_potential:
            raise ValueError(
                "MetalWall requires one charge-factorized Coulomb kernel"
            )
        self.liquid_coulomb = liquid_coulomb
        self.n_types = liquid_coulomb.n_types

        sites = normalize_metal_site_mapping(metal_sites)
        for name in (
            "metal_positions", "metal_site_groups", "metal_group_ids",
            "metal_total_charge",
        ):
            self.register_buffer(name, sites[name].detach().clone())
        self.metal_sigma = positive_scalar(metal_sigma, "metal_sigma")
        self.tolerance = positive_scalar(tolerance, "tolerance")
        field = _field_vector(external_field, "external_field")
        if torch.any(field != 0.0).item() and self.metal_group_ids.numel() != 1:
            raise ValueError(
                "a nonzero external_field currently requires exactly one "
                "electrode group"
            )
        self.register_buffer("external_field", field)
        self._cache_key = None
        self._cache = None

    def _apply(self, fn):
        self._cache_key = self._cache = None
        return super()._apply(fn)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache_key"] = state["_cache"] = None
        return state

    @staticmethod
    def _factor_system(A, C):
        n_groups = C.shape[1]
        matrix = torch.cat((
            torch.cat((A, C), dim=1),
            torch.cat((C.T, A.new_zeros(n_groups, n_groups)), dim=1),
        ), dim=0)
        try:
            factor, pivots = torch.linalg.lu_factor(matrix)
        except RuntimeError as exc:
            raise ValueError(
                "singular metal charge system; check grid resolution, Gaussian "
                "width and electrode geometry"
            ) from exc
        return factor, pivots

    def _geometry(self, shape, spacing, reference, positions, groups, ids):
        key = (
            shape, tuple(spacing.detach().cpu().tolist()),
            tuple(map(tuple, positions.detach().cpu().tolist())),
            tuple(groups.detach().cpu().tolist()), tuple(ids.detach().cpu().tolist()),
            self.metal_sigma, reference.dtype, reference.device,
        )
        if key == self._cache_key:
            return self._cache
        with torch.inference_mode(False), torch.no_grad():
            k2 = _squared_wavevectors(
                shape, spacing, reference.device, reference.dtype,
            )
            full_kernel = _coulomb_values(k2, k2.new_zeros(1))[0]
            metal_kernel = full_kernel * torch.exp(
                -(self.metal_sigma ** 2) * k2
            )
            sites = FourierSites(
                positions, shape, spacing, reference.dtype, reference.device,
            )
            A = sites.matrix(metal_kernel)
            C = (groups[:, None] == ids[None, :]).to(A)
            cell = spacing * spacing.new_tensor(shape)
            field_coordinates = _periodic_field_coordinates(positions, cell)
            factor, pivots = self._factor_system(A, C)
            system = _MetalSystem(
                interaction=A,
                constraints=C,
                factor=factor,
                pivots=pivots,
                field_coordinates=field_coordinates,
            )
        self._cache_key = key
        self._cache = system
        return self._cache

    @staticmethod
    def _energy_scale(context, reference):
        mode = context.get("free_energy_mode", "beta")
        if mode == "physical":
            scale = 1.0 / torch.as_tensor(context["reference_energy"]).to(reference)
        elif mode == "beta":
            scale = torch.as_tensor(context["beta"]).to(reference)
        else:
            raise ValueError("free_energy_mode must be 'beta' or 'physical'")
        if not torch.all(torch.isfinite(scale) & (scale > 0)).item():
            raise ValueError("energy scale must be finite and positive")
        return scale

    def _prepare_fields(self, context, liquid_coulomb, energy_scale):
        rho = context["rho"]
        if not torch.all(torch.isfinite(rho)).item() or torch.any(rho < 0).item():
            raise ValueError("rho must be finite and nonnegative")
        leading, n_grid = rho.shape[:-2], rho.shape[-2]
        shape = common_grid_size(context["grid_size"], leading)
        if math.prod(shape) != n_grid:
            raise ValueError("grid_size product does not match rho")
        spacing = grid_spacing_tensor(
            context["grid_spacing"], device=rho.device, dtype=rho.dtype,
        )
        excluded = context.get("excluded_mask")
        if excluded is not None:
            excluded = torch.as_tensor(excluded, device=rho.device)
            valid_shapes = ((n_grid,), (*leading, n_grid))
            if excluded.dtype != torch.bool or excluded.shape not in valid_shapes:
                raise ValueError(
                    "excluded_mask must be Boolean with shape [n_grid] or "
                    "[..., n_grid]"
                )
            if torch.any(rho[excluded.expand(*leading, n_grid)] != 0.0).item():
                raise ValueError("rho must be zero at excluded grid points")

        n_fields = math.prod(leading) if leading else 1
        if self.external_field.shape not in ((3,), (*leading, 3)):
            raise ValueError(
                "external_field must be [3] or match rho batch shape"
            )
        external = self.external_field.to(rho).expand(
            *leading, 3,
        ).reshape(n_fields, 3)
        scale = (
            energy_scale.expand(leading).reshape(n_fields)
            if leading else energy_scale.reshape(1)
        )
        amplitudes = liquid_coulomb.amplitude.reshape(n_fields)
        liquid_total_charge = liquid_coulomb.total_charge.reshape(n_fields)
        potentials = liquid_coulomb.potential.reshape(
            n_fields, self.metal_positions.shape[0],
        )
        fields = [
            _MetalField(
                liquid_total_charge=liquid_total_charge[i],
                liquid_potential=potentials[i],
                amplitude=amplitudes[i],
                energy_scale=scale[i],
                external_field=external[i],
            )
            for i in range(n_fields)
        ]
        return leading, shape, spacing, fields

    def _solve_charges(
        self, liquid_total_charge, liquid_potential, external_potential,
        group_totals, amplitude, energy_scale, system,
    ):
        roundoff = 64 * torch.finfo(liquid_total_charge.dtype).eps
        charge_scale = 1.0 + liquid_total_charge.abs() + group_totals.abs().sum()
        if (liquid_total_charge + group_totals.sum()).abs().item() > (
            self.tolerance + roundoff * charge_scale.item()
        ):
            raise ValueError(
                "periodic metal cells require combined liquid/electrode "
                "charge neutrality; no background charge is added"
            )
        if not torch.isfinite(amplitude).item() or amplitude.item() <= 0.0:
            raise ValueError("MetalWall requires a positive Coulomb amplitude")

        A, C = system.interaction, system.constraints
        liquid_potential_unit = liquid_potential / amplitude
        rhs = torch.cat((
            -(liquid_potential_unit + energy_scale * external_potential / amplitude),
            group_totals,
        ))
        lu_solve = getattr(torch.linalg, "lu_solve", None)
        solution = (
            torch.lu_solve(rhs[:, None], system.factor, system.pivots)
            if lu_solve is None else
            lu_solve(system.factor, system.pivots, rhs[:, None])
        ).squeeze(-1)
        n_sites = liquid_potential_unit.numel()
        q = solution[:n_sites]
        metal_potential = -amplitude * solution[n_sites:] / energy_scale
        metal_charge = C.T @ q
        self_potential = amplitude * (A @ q) / energy_scale
        liquid_potential = liquid_potential / energy_scale
        charge_residual = (metal_charge - group_totals).abs().max()
        stationarity = (
            self_potential + liquid_potential + external_potential
            - C @ metal_potential
        )
        potential_residual = stationarity.abs().max()
        charge_limit = (self.tolerance + roundoff) * (
            1.0 + group_totals.abs().max()
        )
        potential_limit = (self.tolerance + roundoff) * (
            1.0 + liquid_potential.abs().max()
            + external_potential.abs().max() + self_potential.abs().max()
        )
        if (
            not torch.all(torch.isfinite(solution)).item()
            or charge_residual.item() > charge_limit.item()
            or potential_residual.item() > potential_limit.item()
        ):
            raise RuntimeError("metal charge solve failed residual tolerances")
        return {
            "metal_site_q": q,
            "metal_charge": metal_charge,
            "metal_potential": metal_potential,
            "charge_residual": charge_residual,
            "potential_residual": potential_residual,
            "coulomb_cross_energy": torch.sum(q * liquid_potential),
            "coulomb_metal_energy": 0.5 * torch.sum(q * self_potential),
            "metal_external_energy": torch.sum(q * external_potential),
        }

    def _evaluate_field(self, shape, spacing, field):
        reference = field.liquid_potential
        group_ids = self.metal_group_ids.to(device=reference.device)
        group_totals = self.metal_total_charge.to(reference)
        positions = self.metal_positions.to(device=reference.device)
        groups = self.metal_site_groups.to(device=reference.device)
        system = self._geometry(
            shape, spacing, reference, positions, groups, group_ids,
        )
        external_potential = -(
            system.field_coordinates * field.external_field
        ).sum(-1)
        solved = self._solve_charges(
            field.liquid_total_charge, field.liquid_potential,
            external_potential,
            group_totals, field.amplitude, field.energy_scale, system,
        )
        solved["electrode_coulomb_energy"] = (
            solved["coulomb_cross_energy"] + solved["coulomb_metal_energy"]
        )
        return solved

    def _metal_state(self, context, liquid_coulomb, energy_scale):
        leading, shape, spacing, fields = self._prepare_fields(
            context, liquid_coulomb, energy_scale,
        )
        outputs = [
            self._evaluate_field(shape, spacing, field) for field in fields
        ]
        if not leading:
            return outputs[0]
        return {
            key: torch.stack([field[key] for field in outputs]).reshape(
                *leading, *outputs[0][key].shape,
            )
            for key in outputs[0]
        }

    def energy_and_outputs(self, context: Dict[str, Any]):
        """Return the liquid LR plus electrode energy and electrode state."""

        rho = context["rho"]
        if rho.dtype not in (torch.float32, torch.float64):
            raise TypeError("MetalWall requires float32 or float64 rho")
        liquid_coulomb = self.liquid_coulomb.coulomb_at(
            context,
            self.metal_positions.to(rho),
            target_sigma=self.metal_sigma,
        )
        scale = self._energy_scale(context, rho)
        metal = self._metal_state(context, liquid_coulomb, scale)
        electrode_energy = (
            metal["electrode_coulomb_energy"]
            + metal["metal_external_energy"]
        )
        return liquid_coulomb.energy + scale * electrode_energy, metal

    def energy(self, context: Dict[str, Any]):
        return self.energy_and_outputs(context)[0]

    def forward(self, context: Dict[str, Any]):
        """Return electrode outputs for direct diagnostic evaluation."""
        return self.energy_and_outputs(context)[1]
