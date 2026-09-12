"""Electrode-only electrostatics for Gaussian metals coupled to density grids."""

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch import nn

from ._argument_checks import nonnegative_scalar, positive_scalar
from ._grid import (
    common_grid_size, grid_spacing_tensor, _flat_grid_positions, _grid_center_origin,
)
from ._metal_data import METAL_DATA_KEYS, normalize_metal_sites
from ._metal_sites import ExplicitSiteGrid
from .energy import EnergyReadout
from .reciprocal import _coulomb_values, _squared_wavevectors


def _field_vector(value, name):
    """Fixed Cartesian vector, shared or per field; retain input precision."""
    raw = torch.as_tensor(value)
    if raw.dtype == torch.bool or raw.is_complex() or not torch.all(torch.isfinite(raw)).item():
        raise ValueError(name + " must contain real finite values")
    if raw.ndim < 1 or raw.shape[-1] != 3:
        raise ValueError(name + " must have shape [3] or [..., 3]")
    if raw.requires_grad:
        raise ValueError(name + " must be fixed, not a differentiable input")
    return torch.as_tensor(value, dtype=torch.float64).detach().clone()


class MetalWall(nn.Module):
    r"""Electrode-only Gaussian electrostatics for a complete liquid functional.

    `liquid_charges` are fixed charges per density component, in e, not
    electrode charges. `metal_sigma` and `liquid_sigma` are Gaussian standard
    deviations in coordinate units. With liquid_sigma=0, integrated voxel
    charges are unsmeared center sources on the finite Fourier grid; voxel
    volume normalizes charge but does not provide spatial smearing.
    `coulomb_amplitude` is the physical e²/(4*pi*epsilon), in energy*length,
    not a fitted or beta-energy coefficient. Boundary must be "periodic".

    Configure `external_field` on this module, in energy/(e*length), as [3]
    or [...,3] matching a density batch. `field_origin` is its physical
    periodic wrapping center, default [0,0,0]. The liquid field belongs in
    V_ext; this module adds only metal field work. Do not put field settings
    in data. No independent fixed-potential electrode ensemble is implemented.

    ``metal_sites`` defines one immutable electrode geometry and its charge
    constraints. Runtime inputs contain only the liquid density and grid:
    rho[...,G,S], grid_size, grid_spacing, optional physical
    grid_center[...,G,3], and optional excluded_mask. Electrode positions and
    grid centers use the same physical coordinate frame. Without grid_center,
    grid zero is physical zero.

    Outputs include liquid_charge_density (e/length³), q_liquid and
    metal_site_q (integrated e), physical metal_positions, group charges and
    potentials, charge/equipotential residuals, separate cross/metal Coulomb
    energies, electrode_coulomb_energy (their sum), and metal_external_energy.
    No liquid-liquid energy is evaluated.

    The neutral periodic cell uses k=0 omitted and retains Gaussian metal
    self interaction. This finite-grid mean-density model is not a complete
    point-particle Ewald or microscopic image-correlation functional.
    Charge relaxation remains differentiable with respect to liquid density;
    cached matrices depend only on fixed geometry and kernel parameters.
    See METAL_WALL.md for usage and METAL_WALL_NUMERICS.md for the equations.
    """

    def __init__(
        self,
        metal_sites: Mapping[str, Any],
        liquid_charges: Sequence[float],
        metal_sigma: float,
        liquid_sigma: float,
        coulomb_amplitude: float,
        boundary: str,
        tolerance: float = 1.0e-6,
        *,
        external_field=(0.0, 0.0, 0.0),
        field_origin=(0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        if boundary != "periodic":
            raise ValueError("MetalWall boundary must be explicitly 'periodic'")
        raw_charges = torch.as_tensor(liquid_charges)
        if raw_charges.dtype == torch.bool or raw_charges.is_complex():
            raise ValueError("liquid_charges must contain real finite values")
        valencies = torch.as_tensor(liquid_charges, dtype=torch.float64)
        if valencies.ndim != 1 or valencies.numel() == 0:
            raise ValueError("liquid_charges must contain one valency per liquid species")
        if not torch.all(torch.isfinite(valencies)).item():
            raise ValueError("liquid_charges must be finite")
        self.register_buffer("liquid_charges", valencies.detach().clone())
        if not isinstance(metal_sites, Mapping):
            raise TypeError("metal_sites must be a mapping")
        required = {
            "metal_positions", "metal_site_groups", "metal_group_ids",
            "metal_total_charge", "metal_charge_units",
        }
        missing = required - set(metal_sites)
        unknown = set(metal_sites) - required
        if missing:
            raise ValueError("metal_sites is missing: " + ", ".join(sorted(missing)))
        if unknown:
            raise ValueError("unknown metal_sites entries: " + ", ".join(sorted(unknown)))
        sites = normalize_metal_sites(
            metal_sites["metal_group_ids"], metal_sites["metal_total_charge"],
            metal_sites["metal_charge_units"],
            metal_positions=metal_sites["metal_positions"],
            metal_site_groups=metal_sites["metal_site_groups"],
        )
        for name in (
            "metal_positions", "metal_site_groups", "metal_group_ids",
            "metal_total_charge",
        ):
            self.register_buffer(name, sites[name].detach().clone())
        self.metal_sigma = positive_scalar(metal_sigma, "metal_sigma")
        self.liquid_sigma = nonnegative_scalar(liquid_sigma, "liquid_sigma")
        self.coulomb_amplitude = positive_scalar(
            coulomb_amplitude, "coulomb_amplitude",
        )
        self.tolerance = positive_scalar(tolerance, "tolerance")
        self.boundary = boundary
        self.register_buffer("external_field", _field_vector(external_field, "external_field"))
        self.register_buffer("field_origin", _field_vector(field_origin, "field_origin"))
        self._cache_key = None
        self._cache = None

    def _apply(self, fn):
        self._cache_key = self._cache = None
        return super()._apply(fn)

    def __getstate__(self):
        # Older supported PyTorch/Python versions lack nn.Module.__getstate__.
        state = self.__dict__.copy()
        state["_cache_key"] = state["_cache"] = None
        return state

    def _kernels(self, shape, spacing, rho):
        """Liquid-metal and metal-metal Coulomb kernels."""
        k2 = _squared_wavevectors(shape, spacing, rho.device, rho.dtype)
        exponents = rho.new_tensor([
            (self.liquid_sigma ** 2 + self.metal_sigma ** 2) / 2.0,
            self.metal_sigma ** 2,
        ])
        return self.coulomb_amplitude * _coulomb_values(k2, exponents)

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
        return A, C, factor, pivots

    def _geometry(self, shape, spacing, positions, groups, ids, rho):
        key = (
            shape, tuple(spacing.cpu().tolist()),
            tuple(map(tuple, positions.cpu().tolist())), tuple(groups.cpu().tolist()),
            tuple(ids.cpu().tolist()), self.metal_sigma, self.liquid_sigma,
            self.coulomb_amplitude, self.boundary, rho.dtype, rho.device,
        )
        if key == self._cache_key:
            return self._cache
        with torch.inference_mode(False), torch.no_grad():
            kernels = self._kernels(shape, spacing, rho)
            sampling = ExplicitSiteGrid(
                positions, shape, spacing, rho.dtype, rho.device,
            )
            A = sampling.matrix(kernels[1])
            C = (groups[:, None] == ids[None, :]).to(rho)
            system = self._factor_system(A, C)
        self._cache_key = key
        self._cache = (kernels, sampling, system)
        return self._cache

    def _prepare_fields(self, data):
        """Validate liquid inputs and expand the fixed electrode over a batch."""

        rho = data["rho"]
        if not torch.is_tensor(rho) or rho.ndim < 2:
            raise ValueError("rho must have shape [..., n_grid, n_types]")
        if rho.dtype not in (torch.float32, torch.float64):
            raise TypeError("MetalWall requires float32 or float64 rho")
        if rho.shape[-1] != self.liquid_charges.numel():
            raise ValueError("rho n_types must match liquid_charges")
        if not torch.all(torch.isfinite(rho)).item() or torch.any(rho < 0).item():
            raise ValueError("rho must be finite and nonnegative")
        leading, n_grid = rho.shape[:-2], rho.shape[-2]
        if not all(leading):
            raise ValueError("MetalWall requires a nonempty batch")
        shape = common_grid_size(data["grid_size"], leading)
        if math.prod(shape) != n_grid:
            raise ValueError("grid_size product does not match rho n_grid")
        if "grid_positions" in data:
            _, canonical = _flat_grid_positions(
                data["grid_positions"], shape, leading, rho.device,
            )
            if not canonical:
                raise ValueError("MetalWall requires canonical C-order grid rows")
        excluded = data.get("excluded_mask")
        if excluded is not None:
            excluded = torch.as_tensor(excluded, device=rho.device)
            if excluded.dtype != torch.bool or excluded.shape not in ((n_grid,), (*leading, n_grid)):
                raise ValueError("excluded_mask must be Boolean with shape [n_grid] or [..., n_grid]")
            if torch.any(rho[excluded.expand(*leading, n_grid)] != 0.0).item():
                raise ValueError("rho must be zero at excluded grid points")
        n_fields = math.prod(leading) if leading else 1
        raw_spacing = torch.as_tensor(
            data["grid_spacing"], dtype=rho.dtype, device=rho.device,
        )
        if raw_spacing.ndim <= 1:
            spacings = grid_spacing_tensor(
                raw_spacing, dtype=rho.dtype, device=rho.device,
            ).expand(n_fields, 3)
        elif raw_spacing.shape == (*leading, 3):
            spacings = raw_spacing.detach().to(rho).reshape(n_fields, 3)
            if not torch.all(torch.isfinite(spacings) & (spacings > 0)).item():
                raise ValueError("grid_spacing must be finite and positive")
        else:
            raise ValueError("grid_spacing must be [3] or match rho batch shape")

        grid_origins = torch.zeros((n_fields, 3), dtype=torch.float64, device=rho.device)
        if data.get("grid_center") is not None:
            centers = torch.as_tensor(data["grid_center"], device=rho.device)
            if centers.shape not in ((n_grid, 3), (*leading, n_grid, 3)):
                raise ValueError("grid_center must have shape [n_grid, 3] or match rho batch shape")
            indices = torch.cartesian_prod(*(torch.arange(n, device=rho.device) for n in shape))
            grid_origins = _grid_center_origin(
                data["grid_center"], indices, spacings.reshape(*leading, 3),
            ).reshape(n_fields, 3)

        for name in ("external_field", "field_origin"):
            if getattr(self, name).shape not in ((3,), (*leading, 3)):
                raise ValueError(name + " must be [3] or match rho batch shape")
        external = self.external_field.to(rho).expand(*leading, 3)
        origins = self.field_origin.to(rho).expand(*leading, 3)
        fields = {
            "density": rho.reshape(n_fields, n_grid, -1),
            "spacing": spacings,
            "external_field": external.reshape(n_fields, 3),
            "field_origin": origins.reshape(n_fields, 3),
            "grid_origin": grid_origins,
        }
        return shape, leading, (
            {name: values[index] for name, values in fields.items()}
            for index in range(n_fields)
        )

    def _solve_charges(self, q_liquid, potential, group_totals, system):
        """Solve fixed electrode totals and check neutrality and KKT residuals."""

        roundoff = 64 * torch.finfo(q_liquid.dtype).eps
        charge_scale = 1.0 + q_liquid.abs().sum() + group_totals.abs().sum()
        if (q_liquid.sum() + group_totals.sum()).abs().item() > (
            self.tolerance + roundoff * charge_scale.item()
        ):
            raise ValueError(
                "periodic metal cells require combined liquid/electrode "
                "charge neutrality; no background charge is added"
            )
        A, C, factor, pivots = system
        rhs = torch.cat((-potential, group_totals))
        # torch 1.12 has linalg.lu_factor but not linalg.lu_solve.
        lu_solve = getattr(torch.linalg, "lu_solve", None)
        solution = (
            torch.lu_solve(rhs[:, None], factor, pivots)
            if lu_solve is None else
            lu_solve(factor, pivots, rhs[:, None])
        ).squeeze(-1)
        n_sites = potential.numel()
        q = solution[:n_sites]
        # A q + (liquid + external potential) + C lambda = 0.
        metal_potential = -solution[n_sites:]
        metal_charge = C.T @ q
        self_potential = A @ q
        charge_residual = (metal_charge - group_totals).abs().max()
        stationarity = self_potential + potential - C @ metal_potential
        potential_residual = stationarity.abs().max()
        charge_limit = (self.tolerance + roundoff) * (
            1.0 + group_totals.abs().max()
        )
        potential_limit = (self.tolerance + roundoff) * (
            1.0 + potential.abs().max() + self_potential.abs().max()
        )
        if (
            not torch.all(torch.isfinite(solution)).item()
            or charge_residual.item() > charge_limit.item()
            or potential_residual.item() > potential_limit.item()
        ):
            raise RuntimeError("metal charge solve failed residual tolerances")
        metal_energy = 0.5 * torch.sum(q * self_potential)
        return (q, metal_charge, metal_potential, charge_residual,
                potential_residual, metal_energy)

    def _evaluate_field(
        self, shape, density, spacing, external_field, field_origin, grid_origin,
    ):
        """Form voxel charges, solve the electrodes, and evaluate their energy."""
        group_ids = self.metal_group_ids.to(device=density.device)
        group_totals = self.metal_total_charge.to(density)
        site_positions = self.metal_positions.to(device=density.device)
        site_groups = self.metal_site_groups.to(device=density.device)
        kernels, sampling, system = self._geometry(
            shape, spacing, site_positions - grid_origin, site_groups, group_ids, density,
        )
        liquid_charge_density = (density * self.liquid_charges.to(density)).sum(-1)
        q_liquid = spacing.prod() * liquid_charge_density
        phi_liquid = sampling.potential(q_liquid, kernels[0])
        coordinates = site_positions - field_origin
        cell = spacing * spacing.new_tensor(shape)
        wrapped = coordinates - cell * torch.round(coordinates / cell)
        external_potential = -(wrapped * external_field).sum(-1).to(density)
        (q, metal_charge, metal_potential, charge_residual,
         potential_residual, metal_energy) = self._solve_charges(
            q_liquid, phi_liquid + external_potential, group_totals, system,
        )
        cross_energy = torch.sum(q * phi_liquid)
        metal_external_energy = torch.sum(q * external_potential)
        return {
            "liquid_charge_density": liquid_charge_density,
            "q_liquid": q_liquid,
            "metal_site_q": q,
            "metal_positions": site_positions,
            "metal_site_groups": site_groups,
            "metal_group_ids": group_ids,
            "metal_total_charge": group_totals,
            "metal_charge": metal_charge, "metal_potential": metal_potential,
            "charge_residual": charge_residual,
            "potential_residual": potential_residual,
            "coulomb_cross_energy": cross_energy,
            "coulomb_metal_energy": metal_energy,
            "electrode_coulomb_energy": cross_energy + metal_energy,
            "metal_external_energy": metal_external_energy,
        }

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if METAL_DATA_KEYS & data.keys():
            raise ValueError("Set electrode geometry, constraints, and field on MetalWall, not in data")
        shape, leading, fields = self._prepare_fields(data)
        outputs = [self._evaluate_field(shape, **field) for field in fields]
        if not leading:
            return outputs[0]
        return {
            key: torch.stack([field[key] for field in outputs]).reshape(
                *leading, *outputs[0][key].shape,
            )
            for key in outputs[0]
        }


class MetalElectrodeReadout(EnergyReadout):
    """Thin adapter from physical metal electrostatics to model energy units.

    Adds liquid-metal, metal-metal and metal-only imposed-field energies to
    an existing complete liquid functional. Liquid-liquid interactions belong
    to that functional; supply the liquid external potential in V_ext.
    Independent electrode-voltage work is not implemented. Charges, liquid
    charge density and physical electrode-energy diagnostics are returned
    through the model, using the same charge solve.
    """

    def __init__(self, metalwall: MetalWall):
        super().__init__()
        if not isinstance(metalwall, MetalWall):
            raise TypeError("metalwall must be a MetalWall module")
        self.metalwall = metalwall
        self.n_types = metalwall.liquid_charges.numel()

    def energy(self, context: Dict[str, Any]) -> torch.Tensor:
        return self.energy_and_outputs(context)[0]

    def energy_and_outputs(
        self, context: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state = self.metalwall(context)
        energy = state["electrode_coulomb_energy"] + state["metal_external_energy"]
        mode = context.get("free_energy_mode", "beta")
        if mode == "physical":
            scale = 1.0 / torch.as_tensor(context["reference_energy"]).to(energy)
        elif mode == "beta":
            scale = torch.as_tensor(context["beta"]).to(energy)
        else:
            raise ValueError("free_energy_mode must be 'beta' or 'physical'")
        if scale.shape not in (torch.Size([]), energy.shape):
            raise ValueError("energy scale must be scalar or match field batch shape")
        if not torch.all(torch.isfinite(scale) & (scale > 0)).item():
            raise ValueError("energy scale must be finite and positive")
        return scale * energy, state
