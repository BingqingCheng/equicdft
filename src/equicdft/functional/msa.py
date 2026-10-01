"""Fixed mean-spherical-approximation reference for the restricted primitive model."""

import math
from dataclasses import replace
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .._config import Configurable, make_config, register
from .._validation import boolean, finite_scalar, nonnegative_scalar, positive_integer, positive_scalar
from ..geometry.grid import _flat_grid_positions, common_grid_size, grid_spacing_tensor
from .base import EnergyReadout, ReadoutContext
from .readout import LongRangeReadout
from .reciprocal import ReciprocalFeatures

__all__ = [
    "MSAReferenceReadout",
    "msa_screening_parameter",
    "scaled_particle_free_energy",
    "percus_yevick_coefficients",
    "percus_yevick_direct_correlation",
    "msa_electrostatic_direct_correlation",
]


def msa_screening_parameter(debye_wavenumber, diameter: float):
    r"""Return the Waisman-Lebowitz ``B`` for ``x = kappa_D * sigma``.

    ``B = (x + 1 - sqrt(1 + 2x)) / x = 1 - 2 Gamma / kappa_D`` with the MSA
    screening parameter ``2 Gamma (1 + Gamma sigma) = kappa_D``. It vanishes
    in the Debye-Hückel limit ``x -> 0`` and tends to one at strong coupling.
    The cancellation-free form ``x / (x + 1 + sqrt(1 + 2x))`` is evaluated,
    which is exact to working precision for every ``x >= 0``. Accepts a
    float or a tensor.
    """

    x = debye_wavenumber * diameter
    if torch.is_tensor(x):
        return x / (x + 1.0 + torch.sqrt(1.0 + 2.0 * x))
    if x <= 0.0:
        return 0.0
    return x / (x + 1.0 + math.sqrt(1.0 + 2.0 * x))


def percus_yevick_coefficients(packing_fraction):
    r"""Return ``(a, b, d)`` with ``-c_HS(r) = a - b (r/sigma) + d (r/sigma)^3`` inside the core.

    Wertheim-Thiele: ``a = (1+2eta)^2/(1-eta)^4``, ``b = 6 eta (1+eta/2)^2/(1-eta)^4``,
    ``d = eta (1+2eta)^2 / (2 (1-eta)^4)``. Accepts a float or a tensor.
    """

    eta = packing_fraction
    denominator = (1.0 - eta) ** 4
    a = (1.0 + 2.0 * eta) ** 2 / denominator
    b = 6.0 * eta * (1.0 + 0.5 * eta) ** 2 / denominator
    d = 0.5 * eta * (1.0 + 2.0 * eta) ** 2 / denominator
    return a, b, d


def percus_yevick_direct_correlation(
    r: torch.Tensor,
    diameter: float,
    packing_fraction: float,
) -> torch.Tensor:
    r"""Percus-Yevick hard-sphere ``c(r)`` (Wertheim-Thiele), zero outside the core."""

    a, b, d = percus_yevick_coefficients(packing_fraction)
    s = r / diameter
    inside = -(a - b * s + d * s ** 3)
    return torch.where(s < 1.0, inside, torch.zeros_like(inside))


def msa_electrostatic_direct_correlation(
    r: torch.Tensor,
    diameter: float,
    coulomb_amplitude: float,
    screening_parameter: float,
) -> torch.Tensor:
    r"""MSA charge direct correlation per unit ``z_i z_j`` (Waisman-Lebowitz).

    ``-(A / sigma) [2B - B^2 r / sigma]`` inside the core and the bare
    ``-A / r`` outside, with ``A`` the Coulomb amplitude (the Bjerrum length
    in coordinate units for a beta free energy).
    """

    s = r / diameter
    B = screening_parameter
    inside = -(coulomb_amplitude / diameter) * (2.0 * B - B * B * s)
    safe_r = torch.where(r > 0.0, r, torch.ones_like(r))
    outside = -coulomb_amplitude / safe_r
    return torch.where(s < 1.0, inside, outside)


def _radial_fourier_transform(
    radii: torch.Tensor,
    values: torch.Tensor,
    wavenumbers: torch.Tensor,
) -> torch.Tensor:
    r"""``4 pi int r^2 f(r) sin(kr)/(kr) dr`` by the trapezoidal rule.

    ``values`` may carry leading axes; the result has them followed by ``k``.
    """

    kr = wavenumbers[:, None] * radii[None, :]
    small = kr.abs() <= 1.0e-12
    kernel = torch.where(small, torch.ones_like(kr), torch.sin(kr) / torch.where(small, torch.ones_like(kr), kr))
    integrand = 4.0 * math.pi * radii.square() * values[..., None, :] * kernel
    return torch.trapezoid(integrand, radii, dim=-1)


def _walk_readouts(readouts: Sequence[EnergyReadout]):
    """Yield every readout reachable from ``readouts`` through ``wrapped_readouts``."""

    for item in readouts:
        yield item
        yield from _walk_readouts(item.wrapped_readouts())


def scaled_particle_free_energy(density, diameter: float):
    r"""Hard-sphere excess free energy density ``beta f(rho)`` of scaled-particle theory.

    ``rho [-ln(1-eta) + 3 eta/(1-eta) + 3 eta^2 / (2 (1-eta)^2)]`` with
    ``eta = pi sigma^3 rho / 6``; its second density derivative is minus the
    ``k = 0`` Percus-Yevick direct correlation function (the compressibility
    route), which is what makes it the bulk partner of the Wertheim-Thiele
    kernel. Accepts a float or a tensor.
    """

    eta = math.pi * diameter ** 3 * density / 6.0
    log = torch.log(1.0 - eta) if torch.is_tensor(eta) else math.log(1.0 - eta)
    return density * (-log + 3.0 * eta / (1.0 - eta) + 1.5 * eta ** 2 / (1.0 - eta) ** 2)


@register
class MSAReferenceReadout(EnergyReadout, Configurable):
    r"""Fixed quadratic reference of the restricted primitive model.

    The readout adds the Ramakrishnan-Yussouff quadratic functional built
    from analytic direct correlation functions,

    ``beta F_ref = 1/2 sum_ij int int rho_i(r) K_ij(|r - r'|) rho_j(r') dr dr'``,

    with ``K_ij = -c_ij`` and

    ``c_ij(r) = c_HS^PY(r) + z_i z_j c_el^MSA(r)``,

    the Percus-Yevick hard-sphere function (Wertheim-Thiele) for the number
    channel and the Waisman-Lebowitz mean-spherical solution for the charge
    channel: ``-A/r`` outside the core and ``-(A/sigma)(2B - B^2 r/sigma)``
    inside it, ``B = (x + 1 - sqrt(1 + 2x))/x``, ``x = kappa_D sigma``. Both
    functions depend on the bulk state only through the Debye wavenumber
    ``kappa_D^2 = 4 pi A sum_i z_i^2 rho_i`` and the packing fraction
    ``eta = pi sigma^3 sum_i rho_i / 6``.

    **Reference state.** By default that state is each field's own: the mean
    density of every component over the accessible voxels (``excluded_mask``
    false, or all voxels), a differentiable functional of the density. This
    bulk reference must be electroneutral; local charge separation is allowed.
    A finite liquid carrying net charge compensated by electrodes instead
    needs an explicit neutral ``reference_density``. The readout is then the
    Ramakrishnan-Yussouff expansion about that state,

    ``beta F_ref = V_acc beta f_bulk(rho_bar)
    + 1/2 sum_ij int int drho_i K_ij(|r - r'|; rho_bar) drho_j``,

    with ``drho = rho - rho_bar`` on the accessible voxels (zero elsewhere)
    and ``beta f_bulk`` the bulk excess free energy density whose Hessian is
    ``-c_ij(k=0)``: the scaled-particle (Percus-Yevick compressibility)
    hard-sphere free energy plus ``1/2 K_Z(0) rho_Z^2`` for the charge
    channel. The homogeneous mode is therefore carried by the integrated
    bulk free energy and the state-dependent kernel acts on mean-zero
    fluctuations only, so the functional is one variational ``F[rho]``: at a
    homogeneous state its uniform (``k = 0``) curvature is exactly the MSA/PY
    ``-c(0)``, continuous with the ``k -> 0`` limit of the kernel, and
    energies of different densities and compositions are mutually
    consistent. A fixed ``reference_density`` replaces the state for every
    field and makes the readout the plain quadratic form
    ``1/2 rho K(rho_ref) rho`` including its homogeneous mode. The reference
    does not describe an excluded region itself: with a mask it expands about
    the flat accessible profile, and wall structure beyond quadratic order is
    left to other terms.

    **Coulomb partner.** The bare Coulomb tail ``-A z_i z_j / r`` is carried
    by a charge-factorized :class:`LongRangeReadout` with the Gaussian-damped
    kernel ``4 pi exp(-alpha k^2)/k^2``, and this readout subtracts the
    corresponding ``A erf(r / 2 sqrt(alpha)) / r`` from its charge kernel so
    that nothing is counted twice; what remains, ``A erfc(r / 2 sqrt(alpha))
    / r`` outside the core and the polynomial correction inside it, is short
    ranged. With ``include_long_range=True`` (default) the readout owns that
    damped Coulomb branch, built from the same charges, amplitude and
    exponent and reported through :meth:`wrapped_readouts`; the model must
    then contain no other charged Coulomb ``LongRangeReadout``, at top level
    or wrapped (by a :class:`MetalWall`, say). With
    ``include_long_range=False`` the model must contain exactly one, at top
    level or wrapped, with the same charges, a fixed equal
    ``coulomb_amplitude``, one Coulomb kernel with exponent
    ``long_range_alpha`` and no polarization source; this is the composition
    for electrode calculations, where the wall wraps the liquid Coulomb
    readout. :meth:`validate_readout_graph` enforces both at model
    construction. ``long_range_alpha = 0`` selects the exact kernel, leaving
    only the inside-core correction.

    **Evaluation.** The kernels are applied in reciprocal space on the
    canonical grid order (rows are reordered through ``grid_positions`` when
    they are not canonical). Inside the core both direct correlation
    functions are polynomials in ``r/sigma``, so their transforms are
    state-dependent coefficients times the density-independent transforms of
    ``1``, ``r/sigma`` and ``(r/sigma)^3`` on the core (plus the fixed
    erf/erfc part), integrated once per distinct grid wavenumber and cached
    per grid geometry; contact discontinuities enter as band-limited
    transforms rather than as shell samples.

    **Units.** ``A`` is a fixed Bjerrum length, so the readout is defined for
    ``free_energy_mode="beta"`` at the reference temperature only; other
    modes or temperatures are rejected at evaluation.

    Together with the ideal gas, a model built only from this readout has
    the MSA response ``1/S_ZZ(k) = 1 - rho c_el(k)`` and the Percus-Yevick
    ``1/S_NN(k)``, both positive at every wavevector; a learned excess then
    only has to represent correlations beyond the mean-spherical closure.
    This is a state-conditioned quadratic reference reproducing the bulk
    second derivative, not a nonlinear hard-sphere functional: packing and
    confinement structure beyond quadratic order are left to other terms.

    Parameters
    ----------
    n_types
        Number of density components; ``charges`` (and ``reference_density``
        when given) must have this length.
    charges
        Fixed valency of each component.
    diameter
        Common hard-sphere diameter ``sigma`` in coordinate units.
    coulomb_amplitude
        ``A`` multiplying ``z_i z_j / r``: the Bjerrum length in coordinate
        units of the beta free energy.
    long_range_alpha
        Gaussian exponent ``alpha`` (squared length) of the damped Coulomb
        kernel; ``0`` for the exact kernel.
    include_long_range
        Own the damped Coulomb branch (default) instead of relying on a
        separate, validated ``LongRangeReadout``.
    reference_density
        Optional fixed bulk density of each component defining ``kappa_D``
        and ``eta`` for every field. It must be electroneutral. ``None``
        (default) uses each field's accessible-region mean densities and
        requires those means to be electroneutral.
        A caller may instead supply ``msa_reference_region_mask`` in the
        input batch to select a soft-wall reference region without excluding
        any voxels from the functional or solver. Where regional composition
        is not neutral, ``msa_reference_composition`` can supply a fixed
        normalized, electroneutral component vector. The regional total
        density is then partitioned according to that vector; the physical
        density field and canonical particle numbers are unchanged.
    include_hard_sphere
        Add the Percus-Yevick number-channel kernel and the scaled-particle
        bulk free energy (default). Disable to keep only the electrostatic
        correction, e.g. under a separate hard-sphere functional.
    radial_resolution
        Radial quadrature points per diameter for the kernel transforms.
    """

    def __init__(
        self,
        n_types: int,
        *,
        charges: Sequence[float],
        diameter: float,
        coulomb_amplitude: float,
        long_range_alpha: float,
        include_long_range: bool = True,
        reference_density: Optional[Sequence[float]] = None,
        include_hard_sphere: bool = True,
        radial_resolution: int = 2000,
    ) -> None:
        super().__init__()

        self.n_types = positive_integer(n_types, "n_types")
        charges = tuple(finite_scalar(value, "charges") for value in charges)
        if len(charges) != self.n_types:
            raise ValueError("charges must have n_types entries")
        self.charges = charges
        self.diameter = positive_scalar(diameter, "diameter")
        self.coulomb_amplitude = nonnegative_scalar(coulomb_amplitude, "coulomb_amplitude")
        self.long_range_alpha = nonnegative_scalar(long_range_alpha, "long_range_alpha")
        self.include_long_range = boolean(include_long_range, "include_long_range")
        if reference_density is not None:
            densities = tuple(nonnegative_scalar(value, "reference_density") for value in reference_density)
            if len(densities) != self.n_types:
                raise ValueError("reference_density must have n_types entries")
            if sum(densities) <= 0.0:
                raise ValueError("reference_density must have a positive total")
            self.bulk_state(torch.tensor(densities, dtype=torch.float64))
            reference_density = densities
        self.reference_density = reference_density
        self.include_hard_sphere = boolean(include_hard_sphere, "include_hard_sphere")
        self.radial_resolution = positive_integer(radial_resolution, "radial_resolution")
        if self.radial_resolution < 100:
            raise ValueError("radial_resolution must be at least 100 points per diameter")
        self.long_range = None
        if self.include_long_range:
            self.long_range = LongRangeReadout(
                n_kernels=1,
                n_types=self.n_types,
                charges=self.charges,
                coulomb_amplitude=self.coulomb_amplitude,
                features=ReciprocalFeatures(
                    kernel="coulomb",
                    n_types=self.n_types,
                    radial_exponents=(self.long_range_alpha,),
                ),
            )
        self._basis: Dict[Tuple, torch.Tensor] = {}
        self._basis_zero: Optional[torch.Tensor] = None

    def to_config(self) -> Dict[str, Any]:
        """Return the constructor arguments describing this readout."""

        return make_config(
            self,
            n_types=self.n_types,
            charges=self.charges,
            diameter=self.diameter,
            coulomb_amplitude=self.coulomb_amplitude,
            long_range_alpha=self.long_range_alpha,
            include_long_range=self.include_long_range,
            reference_density=self.reference_density,
            include_hard_sphere=self.include_hard_sphere,
            radial_resolution=self.radial_resolution,
        )

    # --------------------------------------------------------------- partners
    def wrapped_readouts(self) -> Tuple[EnergyReadout, ...]:
        return () if self.long_range is None else (self.long_range,)

    def validate_readout_graph(self, readouts: Sequence[EnergyReadout]) -> None:
        """Require exactly the damped Coulomb branch the MSA assumes, once.

        Charged Coulomb ``LongRangeReadout`` instances are collected from the
        top level and from every wrapped readout (recursively), so a branch
        owned by a :class:`MetalWall` counts.
        """

        others = [item for item in readouts if item is not self]
        if any(isinstance(item, MSAReferenceReadout) for item in others):
            raise ValueError("a model may contain one MSAReferenceReadout")
        coulomb = [
            item for item in _walk_readouts(others)
            if self.is_charged_coulomb_long_range(item)
        ]
        if self.long_range is not None:
            if coulomb:
                raise ValueError(
                    "MSAReferenceReadout(include_long_range=True) owns the damped Coulomb "
                    "branch; remove the other charged Coulomb LongRangeReadout (top level "
                    "or wrapped) or construct the reference with include_long_range=False"
                )
            return
        matching = [item for item in coulomb if self.is_matching_long_range(item)]
        if len(coulomb) != 1 or len(matching) != 1:
            raise ValueError(
                "MSAReferenceReadout(include_long_range=False) requires exactly one "
                "charged Coulomb LongRangeReadout, at top level or wrapped, with the "
                "same charges, a fixed equal coulomb_amplitude, one Coulomb kernel with "
                "exponent long_range_alpha and no polarization source"
            )

    def is_matching_long_range(self, readout: EnergyReadout) -> bool:
        """Whether ``readout`` is the damped Coulomb partner this reference assumes."""

        if not isinstance(readout, LongRangeReadout) or readout.features is None:
            return False
        features = readout.features
        if (
            features.kernel != "coulomb"
            or features.requires_dipole_density
            or readout.include_polarization
            or readout.charges is None
            or readout.coulomb_amplitude is None
        ):
            return False

        # Buffers may hold float32 copies of the declared values.
        def same(actual, expected):
            return math.isclose(actual, expected, rel_tol=1.0e-6, abs_tol=1.0e-9)

        exponents = features.radial_exponents.detach().cpu().double().reshape(-1)
        if exponents.numel() != 1 or not same(exponents.item(), self.long_range_alpha):
            return False
        charges = readout.charges.detach().cpu().double().reshape(-1).tolist()
        if len(charges) != self.n_types or not all(
            same(actual, expected) for actual, expected in zip(charges, self.charges)
        ):
            return False
        return same(readout.coulomb_amplitude.detach().cpu().double().item(), self.coulomb_amplitude)

    @staticmethod
    def is_charged_coulomb_long_range(readout: EnergyReadout) -> bool:
        """Whether ``readout`` is any charged Coulomb ``LongRangeReadout``."""

        return (
            isinstance(readout, LongRangeReadout)
            and readout.features is not None
            and readout.features.kernel == "coulomb"
            and readout.charges is not None
        )

    # -------------------------------------------------------------- bulk state
    def _packing_fraction(self, densities: torch.Tensor) -> torch.Tensor:
        return math.pi * self.diameter ** 3 * densities.sum(dim=-1) / 6.0

    def _debye_wavenumber(self, densities: torch.Tensor) -> torch.Tensor:
        charges = torch.as_tensor(self.charges, dtype=densities.dtype, device=densities.device)
        squared = (
            4.0
            * math.pi
            * self.coulomb_amplitude
            * (charges.square() * densities).sum(dim=-1)
        )
        active = squared > 0.0
        # sqrt has an infinite derivative at zero. Evaluating it on a benign
        # inactive value before selecting the exact zero limit keeps vacuum,
        # zero-charge and zero-amplitude first derivatives finite without
        # introducing an artificial screening floor.
        safe_squared = torch.where(active, squared, torch.ones_like(squared))
        wavenumber = torch.sqrt(safe_squared)
        return torch.where(active, wavenumber, torch.zeros_like(wavenumber))

    def bulk_state(self, densities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(eta, B)`` for per-component densities ``[..., n_types]``."""

        if (
            not torch.is_tensor(densities)
            or not densities.is_floating_point()
            or densities.ndim < 1
            or densities.shape[-1] != self.n_types
            or not torch.all(torch.isfinite(densities)).item()
            or torch.any(densities < 0.0).item()
        ):
            raise ValueError(
                "reference densities must be finite nonnegative values with "
                "n_types entries"
            )
        charges = torch.as_tensor(
            self.charges, dtype=densities.dtype, device=densities.device,
        )
        net_charge = (charges * densities).sum(dim=-1)
        charge_scale = (charges.abs() * densities).sum(dim=-1)
        neutral = net_charge.abs() <= 1.0e-12 + 1.0e-6 * charge_scale
        if not torch.all(neutral).item():
            raise ValueError(
                "the MSA reference state must be electroneutral; for a finite "
                "charged liquid compensated by electrodes, supply an explicit "
                "neutral reference_density"
            )
        eta = self._packing_fraction(densities)
        if torch.any(eta >= 1.0).item():
            raise ValueError("the reference packing fraction must be below one in every field")
        B = msa_screening_parameter(self._debye_wavenumber(densities), self.diameter)
        return eta, B

    def field_densities(
        self,
        rho: torch.Tensor,
        excluded_mask: Optional[torch.Tensor],
        reference_region_mask: Optional[torch.Tensor] = None,
        reference_composition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the reference densities ``[n_fields, n_types]`` of a batch.

        ``rho`` is ``[n_fields, n_grid, n_types]``. The per-field accessible
        mean keeps its dependence on ``rho``; a fixed ``reference_density``
        is returned as a constant.
        """

        n_fields = rho.shape[0]
        if self.reference_density is not None:
            if reference_region_mask is not None or reference_composition is not None:
                raise ValueError("MSA reference-region inputs require dynamic reference_density=None")
            fixed = torch.as_tensor(self.reference_density, dtype=rho.dtype, device=rho.device)
            return fixed.expand(n_fields, self.n_types).clone()
        region = self._reference_region(rho, excluded_mask, reference_region_mask)
        counts = region.sum(dim=-1)
        if torch.any(counts <= 0).item():
            raise ValueError("every field needs reference-region voxels for its reference density")
        densities = (rho * region[..., None]).sum(dim=-2) / counts[:, None]
        if reference_composition is not None:
            composition = torch.as_tensor(reference_composition, dtype=rho.dtype, device=rho.device)
            charges = torch.as_tensor(self.charges, dtype=rho.dtype, device=rho.device)
            if (
                composition.shape != (self.n_types,)
                or not torch.isfinite(composition).all().item()
                or torch.any(composition < 0).item()
                or not torch.isclose(composition.sum(), composition.new_tensor(1.0), atol=1e-6, rtol=0).item()
                or not torch.isclose((composition * charges).sum(), composition.new_tensor(0.0), atol=1e-6, rtol=0).item()
            ):
                raise ValueError("msa_reference_composition must be nonnegative, normalized, and electroneutral")
            densities = densities.sum(dim=-1, keepdim=True) * composition
        return densities

    def _reference_region(
        self,
        rho: torch.Tensor,
        excluded_mask: Optional[torch.Tensor],
        reference_region_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if reference_region_mask is None:
            return self._accessible(rho, excluded_mask)
        if (
            not torch.is_tensor(reference_region_mask)
            or reference_region_mask.dtype != torch.bool
            or reference_region_mask.shape != rho.shape[:2]
        ):
            raise ValueError("msa_reference_region_mask must be Boolean with shape [n_fields, n_grid]")
        if excluded_mask is not None and torch.any(reference_region_mask & excluded_mask).item():
            raise ValueError("msa_reference_region_mask cannot include hard-excluded voxels")
        return reference_region_mask.detach().to(rho)

    @staticmethod
    def _accessible(rho: torch.Tensor, excluded_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Return the ``[n_fields, n_grid]`` accessible-voxel indicator."""

        if excluded_mask is None:
            return torch.ones(rho.shape[:2], dtype=rho.dtype, device=rho.device)
        if (
            not torch.is_tensor(excluded_mask)
            or excluded_mask.dtype != torch.bool
            or excluded_mask.shape != rho.shape[:2]
        ):
            raise ValueError(
                "excluded_mask must be Boolean with shape [n_fields, n_grid]"
            )
        return (~excluded_mask.detach().reshape(rho.shape[0], -1)).to(rho)

    def zero_wavevector_kernels(self, densities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``K_N(0)``, ``K_Z(0)`` for per-field states ``[n_fields, n_types]``."""

        if self._basis_zero is None:
            self._basis_zero = self.basis_transforms(torch.zeros(1, dtype=torch.float64))[:, 0]
        eta, B = self.bulk_state(densities)
        number, charge = self._assemble(self._basis_zero.to(eta)[:, None], eta, B)
        return number[:, 0], charge[:, 0]

    def bulk_free_energy_density(self, densities: torch.Tensor) -> torch.Tensor:
        r"""Return ``beta f_bulk`` for per-field states ``[n_fields, n_types]``.

        The scaled-particle hard-sphere free energy of the total density
        (Hessian ``K_N(0)`` in the number channel) plus ``1/2 K_Z(0) rho_Z^2``
        (Hessian ``z_i z_j K_Z(0)`` in the charge channel at an electroneutral
        state); the latter vanishes for neutral fields.
        """

        charges = torch.as_tensor(self.charges, dtype=densities.dtype, device=densities.device)
        total = densities.sum(dim=-1)
        charge_density = (charges * densities).sum(dim=-1)
        _, charge_zero = self.zero_wavevector_kernels(densities)
        bulk = 0.5 * charge_zero * charge_density.square()
        if self.include_hard_sphere:
            bulk = bulk + scaled_particle_free_energy(total, self.diameter)
        return bulk

    # ------------------------------------------------------------------ kernels
    def real_space_kernels(
        self,
        r: torch.Tensor,
        densities: Sequence[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return ``K_N(r)``, ``K_Z(r)`` at one bulk state, for inspection and tests.

        ``K_N = -c_HS`` (or zero) and ``K_Z = -c_el - A erf(r / 2 sqrt(alpha)) / r``,
        so ``K_ij = K_N + z_i z_j K_Z``.
        """

        r = r.to(torch.float64)
        eta, B = self.bulk_state(torch.as_tensor(densities, dtype=torch.float64))
        if self.include_hard_sphere:
            number = -percus_yevick_direct_correlation(r, self.diameter, eta.item())
        else:
            number = torch.zeros_like(r)
        charge = -msa_electrostatic_direct_correlation(r, self.diameter, self.coulomb_amplitude, B.item())
        return number, charge - self.coulomb_amplitude * self._erf_over_r(r)

    def _erf_over_r(self, r: torch.Tensor) -> torch.Tensor:
        safe_r = torch.where(r > 0.0, r, torch.ones_like(r))
        if self.long_range_alpha > 0.0:
            width = 2.0 * math.sqrt(self.long_range_alpha)
            return torch.where(
                r > 0.0, torch.erf(r / width) / safe_r,
                torch.full_like(r, 2.0 / (math.sqrt(math.pi) * width)),
            )
        return torch.where(r > 0.0, 1.0 / safe_r, torch.zeros_like(r))

    def basis_transforms(self, wavenumbers: torch.Tensor) -> torch.Tensor:
        r"""Return ``[4, n_k]``: transforms of ``1``, ``r/sigma``, ``(r/sigma)^3`` on the core and of the erf part.

        The last row is ``-A`` times the transform of ``erf(r/2 sqrt(alpha))/r``
        inside the core plus ``A`` times that of ``erfc(r/2 sqrt(alpha))/r``
        outside it, i.e. the state-independent piece of ``K_Z``.
        """

        k = wavenumbers.detach().to(torch.float64).reshape(-1)
        inside = torch.linspace(0.0, self.diameter, self.radial_resolution + 1, dtype=torch.float64)
        # Approach the core boundary from inside: the last node carries the
        # inside limit of the discontinuous kernels.
        inside = inside * (1.0 - 1.0e-12)
        s = inside / self.diameter
        monomials = torch.stack((torch.ones_like(s), s, s ** 3))
        core = _radial_fourier_transform(inside, monomials, k)
        erf_part = _radial_fourier_transform(inside, -self.coulomb_amplitude * self._erf_over_r(inside), k)
        # erfc(6) ~ 2e-17: the tail beyond diameter + 12 sqrt(alpha) is nothing.
        outer = self.diameter + 12.0 * math.sqrt(self.long_range_alpha)
        if outer > self.diameter:
            n_outside = int(math.ceil(self.radial_resolution * (outer - self.diameter) / self.diameter))
            outside = torch.linspace(self.diameter, outer, n_outside + 1, dtype=torch.float64) * (1.0 + 1.0e-12)
            tail = self.coulomb_amplitude * (1.0 / outside - self._erf_over_r(outside))
            erf_part = erf_part + _radial_fourier_transform(outside, tail, k)
        return torch.cat((core, erf_part[None, :]), dim=0)

    def reciprocal_kernels(
        self,
        wavenumbers: torch.Tensor,
        densities: Sequence[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return the continuum transforms ``K_N(k)``, ``K_Z(k)`` at one bulk state."""

        basis = self.basis_transforms(wavenumbers)
        eta, B = self.bulk_state(torch.as_tensor(densities, dtype=torch.float64))
        number, charge = self._assemble(basis, eta.reshape(1), B.reshape(1))
        return number[0], charge[0]

    def _assemble(
        self,
        basis: torch.Tensor,
        eta: torch.Tensor,
        B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine ``[4, ...]`` basis transforms with per-field ``[n_fields]`` states."""

        one, linear, cubic, erf_part = basis.to(eta)
        shape = (-1,) + (1,) * (basis.ndim - 1)
        if self.include_hard_sphere:
            a, b, d = percus_yevick_coefficients(eta)
            number = a.reshape(shape) * one + (-b).reshape(shape) * linear + d.reshape(shape) * cubic
        else:
            number = torch.zeros((eta.shape[0],) + basis.shape[1:], dtype=eta.dtype, device=eta.device)
        amplitude = self.coulomb_amplitude / self.diameter
        charge = amplitude * (
            (2.0 * B).reshape(shape) * one - (B * B).reshape(shape) * linear
        ) + erf_part
        return number, charge

    def _grid_basis(
        self,
        grid_size: Tuple[int, int, int],
        grid_spacing: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the cached ``[4, nx, ny, nz//2 + 1]`` basis transforms."""

        spacing = tuple(float(value) for value in grid_spacing.detach().cpu().reshape(-1).tolist())
        key = (tuple(grid_size), spacing, str(device), str(dtype))
        cached = self._basis.get(key)
        if cached is None:
            axes = [
                2.0 * math.pi * torch.fft.fftfreq(n, d=h, dtype=torch.float64)
                for n, h in zip(grid_size, spacing)
            ]
            axes[-1] = axes[-1][: grid_size[-1] // 2 + 1]
            kx, ky, kz = torch.meshgrid(*axes, indexing="ij")
            magnitude = torch.sqrt(kx.square() + ky.square() + kz.square())
            unique, inverse = torch.unique(
                torch.round(magnitude * 1.0e9) / 1.0e9, return_inverse=True,
            )
            basis = self.basis_transforms(unique)[:, inverse].reshape(4, *magnitude.shape)
            cached = basis.to(device=device, dtype=dtype)
            self._basis[key] = cached
        return cached

    # ------------------------------------------------------------------- energy
    def _check_unit_contract(self, context: ReadoutContext) -> None:
        if context.free_energy_mode != "beta":
            raise ValueError(
                "MSAReferenceReadout defines a beta free energy with a fixed Bjerrum "
                "length; use free_energy_mode='beta'"
            )
        temperature = context.normalized_temperature
        if not torch.allclose(temperature, torch.ones_like(temperature), rtol=1.0e-6, atol=1.0e-6):
            raise ValueError(
                "MSAReferenceReadout is defined at the reference temperature only; "
                "its Coulomb amplitude is a fixed Bjerrum length"
            )

    def energy(self, context: ReadoutContext) -> torch.Tensor:
        """Return the reference free energy of every field."""

        self._check_unit_contract(context)
        rho = context.rho
        if (
            not rho.is_floating_point()
            or rho.ndim < 2
            or rho.shape[-1] != self.n_types
            or not torch.all(torch.isfinite(rho)).item()
            or torch.any(rho < 0.0).item()
        ):
            raise ValueError(
                "rho must be finite nonnegative [..., n_grid, n_types] with "
                "the configured n_types"
            )
        leading = rho.shape[:-2]
        n_fields = math.prod(leading)
        n_grid = rho.shape[-2]
        grid_size = common_grid_size(context.require("grid_size"), leading)
        if math.prod(grid_size) != n_grid:
            raise ValueError("grid_size product does not match rho n_grid")
        spacing = grid_spacing_tensor(context.require("grid_spacing"), dtype=torch.float64)
        flat_positions, rows_are_canonical = _flat_grid_positions(
            context.require("grid_positions"), grid_size, leading, rho.device,
        )
        basis = self._grid_basis(grid_size, spacing, rho.device, rho.dtype)

        rho_flat = rho.reshape(n_fields, n_grid, self.n_types)
        excluded = context.excluded_mask
        if excluded is not None:
            if (
                not torch.is_tensor(excluded)
                or excluded.dtype != torch.bool
                or excluded.shape != rho.shape[:-1]
            ):
                raise ValueError(
                    "excluded_mask must be Boolean with shape rho.shape[:-1]"
                )
            excluded = excluded.reshape(n_fields, n_grid)
        reference_region = context.msa_reference_region_mask
        if reference_region is not None:
            if not torch.is_tensor(reference_region) or reference_region.shape != rho.shape[:-1]:
                raise ValueError("msa_reference_region_mask must have shape rho.shape[:-1]")
            reference_region = reference_region.reshape(n_fields, n_grid)
        densities = self.field_densities(
            rho_flat, excluded, reference_region, context.msa_reference_composition,
        )
        eta, B = self.bulk_state(densities)
        number_hat, charge_hat = self._assemble(basis, eta, B)  # [n_fields, kx, ky, kz2]

        if rows_are_canonical:
            canonical = rho_flat
        else:
            canonical = torch.zeros_like(rho_flat).scatter(
                1, flat_positions[..., None].expand_as(rho_flat), rho_flat,
            )
        volume_element = context.voxel_volume.to(rho)
        if self.reference_density is None:
            # The reference region sets the bulk state and background, but
            # does not hard-exclude density or potential outside that region.
            region = self._reference_region(rho_flat, excluded, reference_region)
            if not rows_are_canonical:
                region = torch.zeros_like(region).scatter(1, flat_positions, region)
            source = canonical - densities[:, None, :] * region[..., None]
            reference_volume = region.sum(dim=-1) * volume_element
            bulk = reference_volume * self.bulk_free_energy_density(densities)
        else:
            source = canonical
            bulk = torch.zeros(n_fields, dtype=rho.dtype, device=rho.device)
        source_grid = source.reshape(n_fields, *grid_size, self.n_types).movedim(-1, 1)
        source_hat = torch.fft.rfftn(source_grid, dim=(-3, -2, -1))
        charges = torch.as_tensor(self.charges, dtype=rho.dtype, device=rho.device).reshape(1, -1, 1, 1, 1)
        total_hat = source_hat.sum(dim=1, keepdim=True)
        charge_density_hat = (charges * source_hat).sum(dim=1, keepdim=True)
        potential_hat = number_hat[:, None] * total_hat + charges * (charge_hat[:, None] * charge_density_hat)
        potential = torch.fft.irfftn(potential_hat, s=grid_size, dim=(-3, -2, -1))
        energy = bulk + 0.5 * volume_element * torch.sum(source_grid * potential, dim=(1, 2, 3, 4))
        if self.long_range is not None:
            if not rows_are_canonical:
                # The reciprocal features read the grid in canonical order.
                context = replace(
                    context, rho=canonical.reshape(rho.shape), grid_positions=None,
                )
            energy = energy + self.long_range.energy(context).reshape(n_fields)
        return energy.reshape(leading)
