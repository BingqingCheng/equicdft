"""Differentiable reciprocal-space features of periodic density fields."""

import math
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ._argument_checks import boolean, positive_integer
from ._component_pairs import symmetric_component_pairs
from ._config import Configurable, make_config, register
from ._fourier_sites import FourierSites
from ._grid import (
    _grid_center_origin,
    common_grid_size,
    grid_spacing_tensor,
    voxel_volume,
)


def _wavevector_axes(grid_size, grid_spacing, device, dtype, *, derivative=False):
    """FFT wavevectors; real derivatives omit each even-axis Nyquist mode."""
    axes = [
        2.0 * math.pi * torch.fft.fftfreq(
            n, d=float(grid_spacing[axis].detach().cpu().item()),
            device=device, dtype=dtype,
        )
        for axis, n in enumerate(grid_size)
    ]
    if derivative:
        for n, axis in zip(grid_size, axes):
            if n % 2 == 0:
                axis[n // 2] = 0.0
    return axes


def _squared_wavevectors(grid_size, grid_spacing, device, dtype):
    """Full FFT-grid wave numbers, in inverse coordinate units squared."""
    axes = _wavevector_axes(grid_size, grid_spacing, device, dtype)
    kx, ky, kz = torch.meshgrid(*axes, indexing="ij")
    return kx.square() + ky.square() + kz.square()


def _coulomb_values(squared_wavevector, exponents):
    """4 pi exp(-alpha k²)/k²; zero mode is a declared potential gauge."""

    gaussian = torch.exp(
        -exponents[:, None, None, None] * squared_wavevector[None, ...]
    )
    safe = torch.where(
        squared_wavevector > 0.0, squared_wavevector,
        torch.ones_like(squared_wavevector),
    )
    values = 4.0 * math.pi * gaussian / safe[None, ...]
    return torch.where(
        squared_wavevector[None, ...] > 0.0, values, torch.zeros_like(values),
    )


class _CoulombEvaluation(NamedTuple):
    """Internal unit-amplitude Coulomb evaluation for one liquid charge field."""

    long_range_potential_spectrum: torch.Tensor
    full_potential_spectrum: torch.Tensor
    base_energy: torch.Tensor
    total_charge: torch.Tensor
    grid_size: Tuple[int, int, int]
    grid_spacing: torch.Tensor

    def potential_at(
        self,
        positions: torch.Tensor,
        *,
        grid_center: Optional[torch.Tensor] = None,
        target_sigma: float = 0.0,
    ) -> torch.Tensor:
        """Return the unit-amplitude full Coulomb potential at fixed sites."""

        sigma = float(target_sigma)
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError("target_sigma must be finite and nonnegative")

        raw_positions = torch.as_tensor(positions)
        if (
            raw_positions.dtype == torch.bool
            or raw_positions.is_complex()
            or raw_positions.requires_grad
            or raw_positions.ndim != 2
            or raw_positions.shape[1] != 3
            or not torch.all(torch.isfinite(raw_positions)).item()
        ):
            raise ValueError(
                "positions must be a fixed finite real [n_sites, 3] array"
            )
        positions = raw_positions.to(self.grid_spacing)

        spectrum = self.full_potential_spectrum
        if sigma:
            k2 = _squared_wavevectors(
                self.grid_size,
                self.grid_spacing,
                spectrum.device,
                self.grid_spacing.dtype,
            )
            spectrum = spectrum * torch.exp(-0.5 * sigma ** 2 * k2)

        leading = spectrum.shape[:-3]
        n_fields = math.prod(leading) if leading else 1
        origins = self.grid_spacing.new_zeros((n_fields, 3))
        if grid_center is not None:
            n_grid = math.prod(self.grid_size)
            centers = torch.as_tensor(grid_center, device=spectrum.device)
            if centers.shape not in ((n_grid, 3), (*leading, n_grid, 3)):
                raise ValueError(
                    "grid_center must have shape [n_grid, 3] or match rho batch shape"
                )
            indices = torch.cartesian_prod(*(
                torch.arange(n, device=spectrum.device) for n in self.grid_size
            ))
            spacing = (
                self.grid_spacing.expand(*leading, 3)
                if leading else self.grid_spacing
            )
            origins = _grid_center_origin(
                centers, indices, spacing,
            ).reshape(n_fields, 3).to(self.grid_spacing)

        spectra = spectrum.reshape(n_fields, *self.grid_size)
        potential = [
            FourierSites(
                positions - origin,
                self.grid_size,
                self.grid_spacing,
                self.grid_spacing.dtype,
                spectrum.device,
            ).sample_spectrum(field_spectrum)
            for field_spectrum, origin in zip(spectra, origins)
        ]
        potential = torch.stack(potential)
        if leading:
            return potential.reshape(*leading, positions.shape[0])
        return potential[0]


@register
class ReciprocalFeatures(nn.Module, Configurable):
    r"""Contract scalar or vector Fourier fields against fixed radial kernels.

    For density component ``i``, the continuum-normalized discrete Fourier
    mode is

    ``delta_rho_hat_i(k) = Delta V * sum_g delta_rho_i(r_g) exp(-i k.r_g)``.

    The zero mode is removed by subtracting the spatial mean density. For each
    fixed kernel ``phi_n(k)``, the returned feature for component pair ``ij``
    is

    ``P_nij = (1 / 2V) * sum_{k != 0} phi_n(k)``
    ``         * Re[delta_rho_hat_i(k)^* delta_rho_hat_j(k)]``.

    Off-diagonal component pairs include the factor of two obtained when a
    symmetric quadratic form is reduced to unique pairs ``i <= j``. The
    resulting features are translation invariant, extensive, and exactly zero
    for a homogeneous density field.

    With ``variable="dipole_density"``, Gaussian kernels instead contract
    ``Re[P_hat_i(k)^* dot P_hat_j(k)]``, summing all three vector components
    with the same weight. The full polarization FFT and the zero mode are
    retained, so both uniform and transverse polarization contribute. This
    is a direct orientational-response term, not a Coulomb interaction.

    ``include_divergence=True`` appends a second block of Gaussian features
    formed from ``-i k_D.P_hat``. Independent coefficients for the two blocks
    give the tensor kernel ``A(k) I + B(k) k_D k_D``: away from Nyquist,
    ``G_T=A`` and ``G_L=A+k^2 B``. The residual splitting vanishes at k=0,
    preserving the long-wavelength dipolar interaction in a separate Coulomb
    branch.

    Parameters
    ----------
    radial_exponents
        Positive ``alpha_n`` values in squared-length units for the Gaussian
        factor ``exp(-alpha_n * |k|^2)``. For ``kernel="coulomb"``, zero is
        also allowed and gives the full nonzero-mode Coulomb kernel on the
        finite grid, without an LR/SR split.
    screening
        Nonnegative ``kappa_n`` values in inverse-length units. One scalar is
        broadcast over kernels. These values are used only by the
        ``screened_inverse_laplacian`` kernel.
    kernel
        ``"gaussian"`` gives ``exp(-alpha_n k^2)``. The
        ``"screened_inverse_laplacian"`` option gives
        ``exp(-alpha_n k^2) / (k^2 + kappa_n^2)``. Setting ``kappa_n=0``
        recovers the nonzero-mode inverse-Laplacian form. ``"coulomb"``
        gives ``4 pi exp(-alpha_n k^2) / k^2`` for nonzero modes, matching
        the Gaussian-damped Coulomb convention used in Ewald-type methods.
    n_types
        Number of density components.
    variable
        ``"rho"`` (default) keeps the density-fluctuation / Coulomb-source
        behavior. ``"dipole_density"`` selects direct polarization pair
        features and requires ``kernel="gaussian"``. Inputs are physical
        fields, without an extra charge, dipole-magnitude or reference-scale
        factor. For P in charge/length^2, these features have units
        charge^2/length; beta-free-energy coefficients have length/charge^2.
    include_divergence
        Append divergence-pair features after all vector-dot-product kernels.
        Only supported for direct Gaussian polarization. The output has twice
        as many kernel channels as radial exponents. Divergence features have
        units charge^2/length^3, so their beta-free-energy coefficients have
        length^3/charge^2. No Gaussian-width or other scale factor is applied.
        The same real spectral derivative as Coulomb is used, including its
        zero even-axis Nyquist components. Uniform and pure Nyquist modes
        therefore receive only the vector-dot-product contribution.
    """

    _KERNELS = (
        "gaussian",
        "screened_inverse_laplacian",
        "coulomb",
    )

    def __init__(
        self,
        radial_exponents: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
        screening: Optional[Union[float, Sequence[float]]] = None,
        kernel: str = "gaussian",
        n_types: int = 1,
        *,
        variable: str = "rho",
        include_divergence: bool = False,
    ) -> None:
        super().__init__()

        exponents = torch.as_tensor(
            list(radial_exponents),
            dtype=torch.get_default_dtype(),
        ).detach().clone().reshape(-1)
        if (
            exponents.numel() == 0
            or not torch.all(torch.isfinite(exponents)).item()
            or torch.any(exponents < 0.0).item()
            or (kernel != "coulomb" and torch.any(exponents == 0.0).item())
        ):
            raise ValueError("radial_exponents must be positive (or zero for Coulomb)")

        n_types = positive_integer(n_types, "n_types")

        if kernel not in self._KERNELS:
            raise ValueError("kernel must be one of {}".format(self._KERNELS))
        if variable not in ("rho", "dipole_density"):
            raise ValueError("variable must be 'rho' or 'dipole_density'")
        if variable == "dipole_density" and kernel != "gaussian":
            raise ValueError(
                "direct dipole_density features require a Gaussian kernel"
            )
        include_divergence = boolean(include_divergence, "include_divergence")
        if include_divergence and variable != "dipole_density":
            raise ValueError("include_divergence requires direct polarization features")

        if screening is None:
            screening_values = torch.zeros_like(exponents)
        else:
            screening_values = torch.as_tensor(
                screening,
                dtype=torch.get_default_dtype(),
            ).detach().clone().reshape(-1)
            if screening_values.numel() == 1:
                screening_values = screening_values.repeat(exponents.numel())
            if screening_values.shape != exponents.shape:
                raise ValueError(
                    "screening must contain one value or one per kernel"
                )
            if (
                not torch.all(torch.isfinite(screening_values)).item()
                or torch.any(screening_values < 0.0).item()
            ):
                raise ValueError("screening values must be nonnegative")

        self.kernel = kernel
        self.variable = variable
        self.include_divergence = include_divergence
        self.n_types = n_types
        self.n_kernels = int(exponents.numel()) * (2 if include_divergence else 1)
        self.n_type_pairs = len(symmetric_component_pairs(n_types))
        self.register_buffer("radial_exponents", exponents)
        self.register_buffer("screening", screening_values)

    @property
    def requires_dipole_density(self) -> bool:
        return self.variable == "dipole_density"

    def to_config(self) -> Dict[str, Any]:
        """Return the constructor arguments describing this module."""

        return make_config(
            self,
            radial_exponents=self.radial_exponents,
            screening=self.screening,
            kernel=self.kernel,
            n_types=self.n_types,
            variable=self.variable,
            include_divergence=self.include_divergence,
        )

    def forward(
        self,
        rho: torch.Tensor,
        grid_size: torch.Tensor,
        grid_spacing: torch.Tensor,
        dipole_density: Optional[torch.Tensor] = None,
        charges: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return features with shape ``[..., n_kernels, n_type_pairs]``.

        In direct ``variable="dipole_density"`` mode, only the polarization
        pair features are returned; ``rho`` supplies grid/species shape and
        charges are not allowed. Use a separate density readout if needed.
        With ``include_divergence=True``, the kernel axis contains all vector
        kernels followed by all divergence kernels, in radial-exponent order.

        Otherwise, optional ``charges`` weights each density component before
        the pair contraction. With ``dipole_density`` (``rho.shape + (3,)``),
        the Coulomb source for component a is instead
        ``q_a * rho_hat_a - i k.P_hat_a``. Charges are then required, including
        explicit zeros for neutral polar molecules. P already includes the
        electric dipole magnitude; it is not multiplied by q or by a moment.
        This uses the same kernels and pair contraction as density-only mode.
        No individual-particle self energy is subtracted.

        The real spectral divergence zeros each even-axis Nyquist derivative
        component, preserving Hermitian symmetry. The radial kernel retains
        the ordinary FFT wavevectors, including Nyquist frequencies.
        """

        if rho.ndim < 2 or rho.shape[-1] != self.n_types:
            raise ValueError(
                "rho must have shape [..., n_grid, n_types] with the "
                "configured n_types"
            )
        direct_polarization = self.requires_dipole_density
        self._validate_polarization(rho, dipole_density, charges)
        if charges is not None:
            charges = torch.as_tensor(
                charges, dtype=rho.dtype, device=rho.device,
            )
            if (charges.shape != (self.n_types,)
                    or not torch.isfinite(charges).all().item()):
                raise ValueError("charges must contain one finite value per type")
        nx, ny, nz = common_grid_size(grid_size, rho.shape[:-2])
        if nx * ny * nz != rho.shape[-2]:
            raise ValueError("grid_size product does not match rho n_grid")
        spacing = self._spacing(grid_spacing, rho)

        leading_shape = rho.shape[:-2]
        volume_element = voxel_volume(spacing)
        volume = volume_element * float(nx * ny * nz)
        polarization_hat = None
        if dipole_density is not None:
            p_grid = dipole_density.reshape(
                *leading_shape, nx, ny, nz, self.n_types, 3,
            )
            polarization_hat = self._field_fft(
                p_grid, (-5, -4, -3), volume_element,
                remove_mean=not direct_polarization,
            )

        if direct_polarization:
            fourier_source = polarization_hat
        else:
            rho_grid = rho.reshape(*leading_shape, nx, ny, nz, self.n_types)
            fourier_source = self._field_fft(
                rho_grid, (-4, -3, -2), volume_element,
            )
            if charges is not None:
                fourier_source = fourier_source * charges
            if polarization_hat is not None:
                fourier_source = fourier_source + self._bound_charge_hat(
                    polarization_hat, (nx, ny, nz), spacing,
                )
            fourier_source = fourier_source.unsqueeze(-1)

        kernels = self._kernel_values(
            grid_size=(nx, ny, nz),
            grid_spacing=spacing,
            device=rho.device,
            dtype=rho.dtype,
        ).flatten(start_dim=1)
        features = self._pair_features(fourier_source, kernels, volume)
        # Previously serialized reciprocal modules have no divergence flag.
        if self.include_divergence:
            divergence_hat = self._bound_charge_hat(
                polarization_hat, (nx, ny, nz), spacing,
            ).unsqueeze(-1)
            divergence_features = self._pair_features(
                divergence_hat, kernels, volume,
            )
            features = torch.cat((features, divergence_features), dim=-2)
        return features

    def _pair_features(self, fourier_source, kernels, volume):
        """Contract ``[..., nx, ny, nz, n_types, n_vector]`` pair spectra."""

        fourier_source = fourier_source.flatten(start_dim=-5, end_dim=-3)
        pair_features = []
        for first, second in symmetric_component_pairs(self.n_types):
            cross_power = torch.real(
                torch.conj(fourier_source[..., first, :])
                * fourier_source[..., second, :]
            ).sum(dim=-1)
            multiplicity = 1.0 if first == second else 2.0
            contracted = torch.einsum(
                "nk,...k->...n",
                kernels,
                cross_power,
            )
            pair_features.append(
                multiplicity * contracted / (2.0 * volume)
            )
        return torch.stack(pair_features, dim=-1)

    def _coulomb_evaluation(
        self,
        rho: torch.Tensor,
        grid_size: torch.Tensor,
        grid_spacing: torch.Tensor,
        charges: torch.Tensor,
        dipole_density: Optional[torch.Tensor] = None,
    ) -> _CoulombEvaluation:
        """Use one charge-plus-bound-charge source for energy and site potential."""

        if self.kernel != "coulomb" or self.n_kernels != 1:
            raise ValueError(
                "a reusable Coulomb field requires one Coulomb kernel"
            )
        if rho.ndim < 2 or rho.shape[-1] != self.n_types:
            raise ValueError(
                "rho must have shape [..., n_grid, n_types] with the "
                "configured n_types"
            )
        charges = torch.as_tensor(charges, device=rho.device, dtype=rho.dtype)
        if charges.shape != (self.n_types,) or not torch.isfinite(charges).all().item():
            raise ValueError("charges must contain one finite value per type")
        self._validate_polarization(rho, dipole_density, charges)

        shape = common_grid_size(grid_size, rho.shape[:-2])
        if math.prod(shape) != rho.shape[-2]:
            raise ValueError("grid_size product does not match rho n_grid")
        spacing = self._spacing(grid_spacing, rho)
        volume_element = voxel_volume(spacing)
        volume = volume_element * float(math.prod(shape))

        charge_density = torch.sum(rho * charges, dim=-1)
        q_liquid = volume_element * charge_density
        charge_spectrum = torch.fft.fftn(
            q_liquid.reshape(*rho.shape[:-2], *shape),
            dim=(-3, -2, -1),
        )
        if dipole_density is not None:
            p_grid = dipole_density.reshape(
                *rho.shape[:-2], *shape, self.n_types, 3,
            )
            polarization_hat = self._field_fft(
                p_grid, (-5, -4, -3), volume_element,
            )
            charge_spectrum = charge_spectrum + self._bound_charge_hat(
                polarization_hat, shape, spacing,
            ).sum(dim=-1)
        kernel = self._kernel_values(
            grid_size=shape,
            grid_spacing=spacing,
            device=rho.device,
            dtype=rho.dtype,
        )[0]
        k2 = _squared_wavevectors(shape, spacing, rho.device, rho.dtype)
        full_kernel = _coulomb_values(k2, k2.new_zeros(1))[0]
        long_potential = charge_spectrum * kernel
        full_potential = charge_spectrum * full_kernel
        base_energy = 0.5 * torch.sum(
            torch.real(torch.conj(charge_spectrum) * long_potential),
            dim=(-3, -2, -1),
        ) / volume
        return _CoulombEvaluation(
            long_range_potential_spectrum=long_potential,
            full_potential_spectrum=full_potential,
            base_energy=base_energy,
            # Periodic divergence has zero integral; only free charge counts.
            total_charge=q_liquid.sum(dim=-1),
            grid_size=shape,
            grid_spacing=spacing,
        )

    def _validate_polarization(self, rho, polarization, charges):
        if self.requires_dipole_density:
            if polarization is None:
                raise ValueError("dipole_density is required for direct polarization features")
            if charges is not None:
                raise ValueError("direct polarization features do not use charges")
        elif polarization is not None and (self.kernel != "coulomb" or charges is None):
            raise ValueError("dipole_density requires a Coulomb kernel and charges")
        if polarization is None:
            return
        if polarization.shape != rho.shape + (3,):
            raise ValueError("dipole_density must have shape rho.shape + (3,)")
        if polarization.dtype != rho.dtype or polarization.device != rho.device:
            raise ValueError("dipole_density must match rho dtype and device")
        if not torch.isfinite(polarization).all().item():
            raise ValueError("dipole_density must be finite")

    @staticmethod
    def _field_fft(field, spatial_dims, volume_element, *, remove_mean=True):
        """Continuum-normalized FFT with an explicit homogeneous-mode policy."""

        if remove_mean:
            field = field - field.mean(dim=spatial_dims, keepdim=True)
        return volume_element * torch.fft.fftn(field, dim=spatial_dims)

    def _bound_charge_hat(self, polarization_hat, grid_size, spacing):
        """Contract a polarization FFT with -i k_D, retaining species indices."""

        axes = _wavevector_axes(
            grid_size, spacing, polarization_hat.device, polarization_hat.real.dtype,
            derivative=True,
        )
        wavevector = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return -1j * (polarization_hat * wavevector[..., None, :]).sum(dim=-1)

    def _kernel_values(
        self,
        grid_size: Tuple[int, int, int],
        grid_spacing: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return one ``[nx, ny, nz]`` kernel per radial exponent."""

        squared_wavevector = _squared_wavevectors(
            grid_size, grid_spacing, device, dtype,
        )
        exponents = self.radial_exponents.to(device=device, dtype=dtype)
        if self.kernel == "coulomb":
            return _coulomb_values(squared_wavevector, exponents)
        gaussian = torch.exp(
            -exponents[:, None, None, None]
            * squared_wavevector[None, ...]
        )

        if self.kernel == "gaussian":
            if self.requires_dipole_density:
                return gaussian
            values = gaussian
        elif self.kernel == "screened_inverse_laplacian":
            screening = self.screening.to(device=device, dtype=dtype)
            denominator = (
                squared_wavevector[None, ...]
                + screening[:, None, None, None].square()
            )
            safe_denominator = torch.where(
                squared_wavevector[None, ...] > 0.0,
                denominator,
                torch.ones_like(denominator),
            )
            values = gaussian / safe_denominator
        # The homogeneous mode belongs to the bulk/local functional. Removing
        # it also makes the inverse-Laplacian kernel finite when kappa is zero.
        return torch.where(
            squared_wavevector[None, ...] > 0.0,
            values,
            torch.zeros_like(values),
        )

    @staticmethod
    def _spacing(
        grid_spacing: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Return one validated three-component spacing on rho's device."""

        return grid_spacing_tensor(
            grid_spacing,
            device=rho.device,
            dtype=rho.dtype,
        )
