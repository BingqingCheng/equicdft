"""Differentiable reciprocal-space features of periodic density fields."""

import math
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ._argument_checks import positive_integer
from ._component_pairs import symmetric_component_pairs
from ._grid import common_grid_size, grid_spacing_tensor, voxel_volume


class ReciprocalFeatures(nn.Module):
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

    Parameters
    ----------
    radial_exponents
        Positive ``alpha_n`` values in squared-length units for the Gaussian
        factor ``exp(-alpha_n * |k|^2)``.
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
    ) -> None:
        super().__init__()

        exponents = torch.as_tensor(
            list(radial_exponents),
            dtype=torch.get_default_dtype(),
        ).detach().clone().reshape(-1)
        if (
            exponents.numel() == 0
            or not torch.all(torch.isfinite(exponents)).item()
            or torch.any(exponents <= 0.0).item()
        ):
            raise ValueError("radial_exponents must contain positive values")

        n_types = positive_integer(n_types, "n_types")

        if kernel not in self._KERNELS:
            raise ValueError("kernel must be one of {}".format(self._KERNELS))
        if variable not in ("rho", "dipole_density"):
            raise ValueError("variable must be 'rho' or 'dipole_density'")
        if variable == "dipole_density" and kernel != "gaussian":
            raise ValueError(
                "direct dipole_density features require a Gaussian kernel"
            )

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
        self.n_types = n_types
        self.n_kernels = int(exponents.numel())
        self.n_type_pairs = len(symmetric_component_pairs(n_types))
        self.register_buffer("radial_exponents", exponents)
        self.register_buffer("screening", screening_values)

    @property
    def requires_dipole_density(self) -> bool:
        # Density/Coulomb checkpoints predating the direct-vector mode have
        # no variable attribute.
        return getattr(self, "variable", "rho") == "dipole_density"

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
        if direct_polarization:
            if dipole_density is None:
                raise ValueError(
                    "dipole_density is required for direct polarization features"
                )
            if charges is not None:
                raise ValueError("direct polarization features do not use charges")
        elif dipole_density is not None:
            if self.kernel != "coulomb" or charges is None:
                raise ValueError(
                    "dipole_density requires a Coulomb kernel and charges"
                )

        if dipole_density is not None:
            if dipole_density.shape != rho.shape + (3,):
                raise ValueError("dipole_density must have shape rho.shape + (3,)")
            if (dipole_density.dtype != rho.dtype
                    or dipole_density.device != rho.device):
                raise ValueError("dipole_density must match rho dtype and device")
            if not torch.isfinite(dipole_density).all().item():
                raise ValueError("dipole_density must be finite")
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
        ).reshape(self.n_kernels, -1)
        fourier_source = fourier_source.reshape(
            *leading_shape,
            nx * ny * nz,
            self.n_types,
            3 if direct_polarization else 1,
        )

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

    @staticmethod
    def _field_fft(field, spatial_dims, volume_element, *, remove_mean=True):
        """Continuum-normalized FFT with an explicit homogeneous-mode policy."""

        if remove_mean:
            field = field - field.mean(dim=spatial_dims, keepdim=True)
        return volume_element * torch.fft.fftn(field, dim=spatial_dims)

    def _bound_charge_hat(self, polarization_hat, grid_size, spacing):
        """Contract a polarization FFT with -i k_D, retaining species indices."""

        axes = self._wavevector_axes(
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
        """Return ``[n_kernels, nx, ny, nz]`` radial kernel values."""

        k_axes = self._wavevector_axes(grid_size, grid_spacing, device, dtype)
        kx, ky, kz = torch.meshgrid(*k_axes, indexing="ij")
        squared_wavevector = kx.square() + ky.square() + kz.square()
        exponents = self.radial_exponents.to(device=device, dtype=dtype)
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
        else:
            safe_squared_wavevector = torch.where(
                squared_wavevector > 0.0,
                squared_wavevector,
                torch.ones_like(squared_wavevector),
            )
            values = (
                4.0
                * math.pi
                * gaussian
                / safe_squared_wavevector[None, ...]
            )

        # The homogeneous mode belongs to the bulk/local functional. Removing
        # it also makes the inverse-Laplacian kernel finite when kappa is zero.
        return torch.where(
            squared_wavevector[None, ...] > 0.0,
            values,
            torch.zeros_like(values),
        )

    @staticmethod
    def _wavevector_axes(
        grid_size, grid_spacing, device, dtype, *, derivative=False,
    ):
        """FFT wavevectors, optionally zeroing even-axis Nyquist derivatives."""

        axes = [
            2.0
            * math.pi
            * torch.fft.fftfreq(
                n,
                d=float(grid_spacing[axis].detach().cpu().item()),
                device=device,
                dtype=dtype,
            )
            for axis, n in enumerate(grid_size)
        ]
        if derivative:
            for n, axis in zip(grid_size, axes):
                if n % 2 == 0:
                    axis[n // 2] = 0.0
        return axes

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
