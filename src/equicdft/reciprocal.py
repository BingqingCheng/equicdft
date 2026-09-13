"""Differentiable reciprocal-space features of periodic density fields."""

import math
from typing import NamedTuple, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ._argument_checks import positive_integer
from ._component_pairs import symmetric_component_pairs
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


class ReciprocalFeatures(nn.Module):
    r"""Contract Fourier density fluctuations against fixed radial kernels.

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
        self.n_types = n_types
        self.n_kernels = int(exponents.numel())
        self.n_type_pairs = len(symmetric_component_pairs(n_types))
        self.register_buffer("radial_exponents", exponents)
        self.register_buffer("screening", screening_values)

    def forward(
        self,
        rho: torch.Tensor,
        grid_size: torch.Tensor,
        grid_spacing: torch.Tensor,
        dipole_density: Optional[torch.Tensor] = None,
        charges: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return features with shape ``[..., n_kernels, n_type_pairs]``.

        Optional ``charges`` weights each density component before the pair
        contraction. With ``dipole_density`` (shape ``rho.shape + (3,)``),
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
        rho_grid = rho.reshape(
            *leading_shape,
            nx,
            ny,
            nz,
            self.n_types,
        )
        spatial_dims = (-4, -3, -2)
        volume_element = voxel_volume(spacing)
        volume = volume_element * float(nx * ny * nz)
        fourier_source = self._fluctuation_fft(
            rho_grid, spatial_dims, volume_element,
        )
        if charges is not None:
            fourier_source = fourier_source * charges
        if dipole_density is not None:
            fourier_source = fourier_source + self._bound_charge_hat(
                dipole_density, (nx, ny, nz), spacing, volume_element,
            )

        kernels = self._kernel_values(
            grid_size=(nx, ny, nz),
            grid_spacing=spacing,
            device=rho.device,
            dtype=rho.dtype,
        ).reshape(self.n_kernels, -1)
        fourier_source = fourier_source.reshape(
            *leading_shape,
            nx * ny * nz,
            self.n_types,
        )

        pair_features = []
        for first, second in symmetric_component_pairs(self.n_types):
            cross_power = torch.real(
                torch.conj(fourier_source[..., first])
                * fourier_source[..., second]
            )
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
            charge_spectrum = charge_spectrum + self._bound_charge_hat(
                dipole_density, shape, spacing, volume_element,
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
        if polarization is None:
            return
        if self.kernel != "coulomb" or charges is None:
            raise ValueError("dipole_density requires a Coulomb kernel and charges")
        if polarization.shape != rho.shape + (3,):
            raise ValueError("dipole_density must have shape rho.shape + (3,)")
        if polarization.dtype != rho.dtype or polarization.device != rho.device:
            raise ValueError("dipole_density must match rho dtype and device")
        if not torch.isfinite(polarization).all().item():
            raise ValueError("dipole_density must be finite")

    @staticmethod
    def _fluctuation_fft(field, spatial_dims, volume_element):
        """Continuum-normalized FFT with the homogeneous mode removed."""

        fluctuation = field - field.mean(dim=spatial_dims, keepdim=True)
        return volume_element * torch.fft.fftn(fluctuation, dim=spatial_dims)

    def _bound_charge_hat(self, polarization, grid_size, spacing, volume_element):
        """Fourier bound charge -i k_D.P_hat, retaining component indices."""

        p_grid = polarization.reshape(
            *polarization.shape[:-3], *grid_size, self.n_types, 3,
        )
        fourier_p = self._fluctuation_fft(
            p_grid, (-5, -4, -3), volume_element,
        )
        axes = _wavevector_axes(
            grid_size, spacing, polarization.device, polarization.dtype,
            derivative=True,
        )
        wavevector = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return -1j * (fourier_p * wavevector[..., None, :]).sum(dim=-1)

    def _kernel_values(
        self,
        grid_size: Tuple[int, int, int],
        grid_spacing: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``[n_kernels, nx, ny, nz]`` radial kernel values."""

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
