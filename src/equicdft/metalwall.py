"""Charge-constrained Gaussian metal electrodes on periodic density grids."""

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from ._argument_checks import nonnegative_scalar, positive_scalar
from ._grid import common_grid_size, grid_spacing_tensor, _flat_grid_positions
from ._metal_data import normalize_metal_field, normalize_metal_metadata
from .energy import EnergyReadout
from .reciprocal import _coulomb_values, _squared_wavevectors


def _potential(charge, kernel, spacing, shape):
    """Potential conjugate to integrated grid charges, not charge density."""

    values = charge.reshape(*charge.shape[:-1], *shape)
    transformed = torch.fft.fftn(values, dim=(-3, -2, -1))
    result = torch.fft.ifftn(
        transformed * kernel, dim=(-3, -2, -1),
    ).real / spacing.prod()
    return result.reshape_as(charge)


def _grid_indices(index, shape):
    """Integer coordinates of selected canonical C-order grid rows."""
    return torch.stack((
        index // (shape[1] * shape[2]),
        (index // shape[2]) % shape[1], index % shape[2],
    ), dim=-1)


class MetalWall(nn.Module):
    r"""Solve Gaussian electrode charges, then evaluate combined Coulomb energy.

    Compute the liquid potential, solve metal charges at fixed group totals,
    and evaluate the energy. All physical parameters are explicit inputs:

    * ``charges``: fixed liquid species valencies (elementary-charge units).
    * ``sigma``: metal Gaussian standard deviation, in coordinate units (>0).
    * ``liquid_sigma``: liquid charge-basis standard deviation (>=0); zero is
      the unsmeared grid source. Widths are not automatically ionic diameters.
    * ``coulomb_amplitude``: e²/(4 pi epsilon), in energy * coordinate units.
    * ``boundary``: must explicitly be ``"periodic"`` (orthorhombic 3D PBC).

    The Gaussian convention is g_sigma(k)=exp(-sigma² k²/2). The three
    interaction blocks use K(k)*g_source(k)*g_target(k), K=4 pi/k². k=0 is
    omitted, fixing the potential gauge; a metal-containing cell must be
    neutral. Metal self interaction is retained. This is a finite Fourier-grid
    discretization, not a complete point-particle Ewald/slab implementation.

    ``forward(data)`` accepts canonical C-order ``rho[..., G, S]``, grid_size,
    grid_spacing, integer metal_mask[..., G], metal_group_ids[..., K],
    metal_total_charge[..., K] and metal_charge_units="e". Unbatched geometry
    may be shared. Every present group must have an explicit total charge.

    Optional ``metal_external_field[..., 3]`` is a Cartesian field in
    energy/(e * coordinate_unit), applied to metal only. ``metal_field_origin``
    [.., 3] is the wrapping center relative to grid index zero, in coordinate
    units (default zero). For r = grid_index * grid_spacing, use
    r_wrap = r - origin - L * round((r - origin)/L), componentwise. Choose
    the branch cut away from metal sites. The liquid field belongs in V_ext.

    Returns a new dictionary. ``q_liquid``, ``metal_q`` and their sum ``q_mw``
    are integrated charge coefficients [.., G] in e, not fluid densities.
    ``liquid_charge_density`` is sum_i(z_i*rho_i), in e / coordinate_unit³.
    ``metal_charge`` and ``metal_potential`` follow the supplied group order.
    Potentials include the imposed metal field and are energy/e. All Coulomb
    outputs are physical energies and exclude the imposed field work, returned
    separately as ``metal_external_energy`` = -sum_m q_m E dot r_wrap,m.
    Charge and equipotential residuals are absolute maxima per field.

    Only fixed parameters/geometry are cached. The charge solve stays in the
    density autograd graph. This describes response to mean liquid density;
    it does not supply microscopic image-charge correlation free energies.
    """

    def __init__(
        self,
        charges: Sequence[float],
        sigma: float,
        liquid_sigma: float,
        coulomb_amplitude: float,
        boundary: str,
        tolerance: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if boundary != "periodic":
            raise ValueError("MetalWall boundary must be explicitly 'periodic'")
        valencies = torch.as_tensor(list(charges), dtype=torch.float64)
        if valencies.ndim != 1 or valencies.numel() == 0:
            raise ValueError("charges must contain one valency per liquid species")
        if not torch.all(torch.isfinite(valencies)).item():
            raise ValueError("charges must be finite")
        self.register_buffer("charges", valencies.detach().clone())
        self.sigma = positive_scalar(sigma, "sigma")
        self.liquid_sigma = nonnegative_scalar(liquid_sigma, "liquid_sigma")
        self.coulomb_amplitude = positive_scalar(
            coulomb_amplitude, "coulomb_amplitude",
        )
        self.tolerance = positive_scalar(tolerance, "tolerance")
        self.boundary = boundary
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

    def _geometry(self, shape, spacing, mask, ids, rho):
        # Match exact cell, masks, parameters and ordering, not just positions.
        key = (
            shape, tuple(spacing.cpu().tolist()), tuple(mask.cpu().tolist()),
            tuple(ids.cpu().tolist()), self.sigma, self.liquid_sigma,
            self.coulomb_amplitude, self.boundary, rho.dtype, rho.device,
        )
        if key == self._cache_key:
            return self._cache
        # A cache first built by inference must remain usable by later c1/c2
        # calls. Inference tensors cannot be saved for autograd backward.
        with torch.inference_mode(False), torch.no_grad():
            k2 = _squared_wavevectors(shape, spacing, rho.device, rho.dtype)
            exponents = rho.new_tensor([
                self.liquid_sigma ** 2,
                (self.liquid_sigma ** 2 + self.sigma ** 2) / 2.0,
                self.sigma ** 2,
            ])
            kernels = self.coulomb_amplitude * _coulomb_values(k2, exponents)
            index = torch.nonzero(mask >= 0, as_tuple=False).flatten()
            system = None
            if index.numel():
                system = self._compute_S_matrix(
                    index, mask, ids, kernels[2], spacing, shape,
                )
        self._cache_key = key
        self._cache = (kernels, index, system)
        return self._cache

    def _compute_S_matrix(self, index, mask, ids, kernel, spacing, shape):
        """Sample the periodic impulse response; factor the constrained system."""

        n_groups = ids.numel()
        # Translation invariance gives A_ij = G[(r_i-r_j) mod grid_size].
        # This is the same discrete convolution as full-grid unit probes,
        # including self interaction and 1/voxel-volume normalization.
        response = torch.fft.ifftn(kernel, dim=(-3, -2, -1)).real / spacing.prod()
        coordinates = _grid_indices(index, shape)
        offsets = tuple(
            (axis[:, None] - axis[None, :]) % n
            for axis, n in zip(coordinates.T, shape)
        )
        A = response[offsets]
        C = (mask[index, None] == ids[None, :]).to(A)
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

    def _prepare_fields(self, data):
        """Validate inputs and normalize shared or per-field batch metadata."""

        rho = data["rho"]
        if not torch.is_tensor(rho) or rho.ndim < 2:
            raise ValueError("rho must have shape [..., n_grid, n_types]")
        if rho.dtype not in (torch.float32, torch.float64):
            raise TypeError("MetalWall requires float32 or float64 rho")
        if rho.shape[-1] != self.charges.numel():
            raise ValueError("rho n_types must match charges")
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
        metadata = normalize_metal_metadata(
            data.get("metal_mask"), data.get("metal_group_ids"),
            data.get("metal_total_charge"), data.get("metal_charge_units"),
            n_grid=n_grid, dtype=rho.dtype, device=rho.device,
            batch_shape=leading,
        )
        mask = metadata.get(
            "metal_mask", torch.full(
                (*leading, n_grid), -1, dtype=torch.long, device=rho.device,
            ),
        )
        if torch.any(rho[mask >= 0] != 0.0).item():
            raise ValueError("rho must be zero at metal grid points")
        ids = metadata.get(
            "metal_group_ids", torch.empty(
                (*leading, 0), dtype=torch.long, device=rho.device,
            ),
        )
        totals = metadata.get("metal_total_charge", rho.new_empty(*leading, 0))
        n_fields, n_groups = math.prod(leading) if leading else 1, ids.shape[-1]
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

        field_data = normalize_metal_field(
            data.get("metal_external_field"), data.get("metal_field_origin"),
            dtype=rho.dtype, device=rho.device, batch_shape=leading,
        )
        external = field_data.get("metal_external_field", rho.new_zeros(*leading, 3))
        origins = field_data.get("metal_field_origin", rho.new_zeros(*leading, 3))
        fields = zip(
            rho.reshape(n_fields, n_grid, -1), mask.reshape(n_fields, n_grid),
            ids.reshape(n_fields, n_groups), totals.reshape(n_fields, n_groups),
            spacings, external.reshape(n_fields, 3), origins.reshape(n_fields, 3),
        )
        return shape, leading, fields

    def _solve_charges(self, q_liquid, potential, group_totals, index, system):
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
        q = solution[:index.numel()]
        # A q + (liquid + external potential) + C lambda = 0.
        metal_potential = -solution[index.numel():]
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
        self, shape, density, field_mask, group_ids, group_totals, spacing,
        external_field, field_origin,
    ):
        """Form voxel charges, solve the electrodes, and evaluate their energy."""

        kernels, index, system = self._geometry(
            shape, spacing, field_mask, group_ids, density,
        )
        liquid_charge_density = (density * self.charges.to(density)).sum(-1)
        q_liquid = spacing.prod() * liquid_charge_density
        metal_q = q_liquid * 0.0
        zero = density.square().sum() * 0.0
        phi_liquid = _potential(q_liquid, kernels[1], spacing, shape)
        metal_charge = group_totals + zero
        metal_potential = group_totals * 0.0 + zero
        charge_residual = potential_residual = zero
        metal_energy = metal_external_energy = zero
        if index.numel():
            # Canonical C-order indices; only the field uses this wrapped
            # coordinate branch. Coulomb positions and the cached A stay fixed.
            coordinates = _grid_indices(index, shape).to(density) * spacing - field_origin
            cell = spacing * spacing.new_tensor(shape)
            wrapped = coordinates - cell * torch.round(coordinates / cell)
            external_potential = -(wrapped * external_field).sum(-1)
            driving_potential = phi_liquid[index] + external_potential
            (q, metal_charge, metal_potential, charge_residual,
             potential_residual, metal_energy) = self._solve_charges(
                q_liquid, driving_potential, group_totals, index, system,
            )
            metal_q = metal_q.index_copy(0, index, q)
            metal_external_energy = torch.sum(q * external_potential)
        liquid_energy = 0.5 * torch.sum(
            q_liquid * _potential(q_liquid, kernels[0], spacing, shape),
        )
        cross_energy = torch.sum(metal_q * phi_liquid)
        return {
            "liquid_charge_density": liquid_charge_density,
            "q_liquid": q_liquid, "metal_q": metal_q,
            "q_mw": q_liquid + metal_q,
            "metal_charge": metal_charge, "metal_potential": metal_potential,
            "charge_residual": charge_residual,
            "potential_residual": potential_residual,
            "coulomb_liquid_energy": liquid_energy,
            "coulomb_cross_energy": cross_energy,
            "coulomb_metal_energy": metal_energy,
            "coulomb_energy": liquid_energy + cross_energy + metal_energy,
            "metal_external_energy": metal_external_energy,
        }

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        shape, leading, fields = self._prepare_fields(data)
        outputs = [self._evaluate_field(shape, *field) for field in fields]
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

    ``contribution='correction'`` (default) adds only liquid-metal plus
    metal-metal energy to an existing complete liquid functional. ``'total'``
    includes liquid-liquid Coulomb too and requires an appropriately defined
    residual liquid functional: never append it to an existing liquid LR term.
    Both modes add the metal-only imposed-field energy when supplied. Neither
    adds the liquid field (supply it in V_ext) or independent electrode-voltage
    work. Charges, liquid charge density and physical Coulomb diagnostics
    are also returned through the model, using the same charge solve.
    """

    def __init__(self, metalwall: MetalWall, contribution: str = "correction"):
        super().__init__()
        if not isinstance(metalwall, MetalWall):
            raise TypeError("metalwall must be a MetalWall module")
        if contribution not in ("correction", "total"):
            raise ValueError("contribution must be 'correction' or 'total'")
        self.metalwall = metalwall
        self.n_types = metalwall.charges.numel()
        self.contribution = contribution

    def energy(self, context: Dict[str, Any]) -> torch.Tensor:
        return self.energy_and_outputs(context)[0]

    def energy_and_outputs(
        self, context: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state = self.metalwall(context)
        energy = (
            state["coulomb_energy"] if self.contribution == "total" else
            state["coulomb_cross_energy"] + state["coulomb_metal_energy"]
        ) + state["metal_external_energy"]
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
