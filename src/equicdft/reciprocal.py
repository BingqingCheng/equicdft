"""Differentiable reciprocal-space features of periodic density fields."""

import math
from numbers import Integral
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn


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
        factor ``exp(-alpha_n * |k|^2)``.
    screening
        Nonnegative ``kappa_n`` values in inverse-length units. One scalar is
        broadcast over kernels. These values are used only by the
        ``screened_inverse_laplacian`` kernel.
    kernel
        ``"gaussian"`` gives ``exp(-alpha_n k^2)``. The
        ``"screened_inverse_laplacian"`` option gives
        ``exp(-alpha_n k^2) / (k^2 + kappa_n^2)``. Setting ``kappa_n=0``
        recovers the nonzero-mode inverse-Laplacian form.
    n_types
        Number of density components.
    """

    _KERNELS = ("gaussian", "screened_inverse_laplacian")

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
            or torch.any(exponents <= 0.0).item()
        ):
            raise ValueError("radial_exponents must contain positive values")

        if isinstance(n_types, bool) or not isinstance(n_types, Integral):
            raise TypeError("n_types must be a positive integer")
        n_types = int(n_types)
        if n_types < 1:
            raise ValueError("n_types must be a positive integer")

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
        self.n_type_pairs = n_types * (n_types + 1) // 2
        self.register_buffer("radial_exponents", exponents)
        self.register_buffer("screening", screening_values)

    def forward(
        self,
        rho: torch.Tensor,
        grid_size: torch.Tensor,
        grid_spacing: torch.Tensor,
    ) -> torch.Tensor:
        """Return features with shape ``[..., n_kernels, n_type_pairs]``."""

        if rho.ndim < 2 or rho.shape[-1] != self.n_types:
            raise ValueError(
                "rho must have shape [..., n_grid, n_types] with the "
                "configured n_types"
            )
        nx, ny, nz = self._common_grid_size(grid_size, rho.shape[:-2])
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
        density_fluctuation = rho_grid - rho_grid.mean(
            dim=spatial_dims,
            keepdim=True,
        )

        cell_volume = torch.prod(spacing)
        volume = cell_volume * float(nx * ny * nz)
        fourier_density = cell_volume * torch.fft.fftn(
            density_fluctuation,
            dim=spatial_dims,
        )

        kernels = self._kernel_values(
            grid_size=(nx, ny, nz),
            grid_spacing=spacing,
            device=rho.device,
            dtype=rho.dtype,
        ).reshape(self.n_kernels, -1)
        fourier_density = fourier_density.reshape(
            *leading_shape,
            nx * ny * nz,
            self.n_types,
        )

        pair_features = []
        for first in range(self.n_types):
            for second in range(first, self.n_types):
                cross_power = torch.real(
                    torch.conj(fourier_density[..., first])
                    * fourier_density[..., second]
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

    def _kernel_values(
        self,
        grid_size: Tuple[int, int, int],
        grid_spacing: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``[n_kernels, nx, ny, nz]`` radial kernel values."""

        nx, ny, nz = grid_size
        k_axes = [
            2.0
            * math.pi
            * torch.fft.fftfreq(
                n,
                d=float(grid_spacing[axis].detach().cpu().item()),
                device=device,
                dtype=dtype,
            )
            for axis, n in enumerate((nx, ny, nz))
        ]
        kx, ky, kz = torch.meshgrid(*k_axes, indexing="ij")
        squared_wavevector = kx.square() + ky.square() + kz.square()
        exponents = self.radial_exponents.to(device=device, dtype=dtype)
        gaussian = torch.exp(
            -exponents[:, None, None, None]
            * squared_wavevector[None, ...]
        )

        if self.kernel == "gaussian":
            values = gaussian
        else:
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
    def _common_grid_size(
        grid_size: torch.Tensor,
        leading_shape: torch.Size,
    ) -> Tuple[int, int, int]:
        """Validate that a batch uses one common regular-grid shape."""

        sizes = torch.as_tensor(grid_size).detach().reshape(-1, 3)
        if sizes.shape[0] not in (1, math.prod(leading_shape)):
            raise ValueError("grid_size leading shape must match rho")
        rounded = torch.round(sizes).to(dtype=torch.long)
        if not torch.allclose(sizes.to(dtype=torch.float64), rounded.to(torch.float64)):
            raise ValueError("grid_size values must be integers")
        if torch.any(rounded <= 0).item():
            raise ValueError("grid_size values must be positive")
        if not torch.all(rounded == rounded[0]).item():
            raise ValueError("all fields in one batch must share grid_size")
        return tuple(int(value) for value in rounded[0].cpu().tolist())

    @staticmethod
    def _spacing(
        grid_spacing: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Return one validated three-component spacing on rho's device."""

        spacing = torch.as_tensor(
            grid_spacing,
            device=rho.device,
            dtype=rho.dtype,
        ).reshape(-1)
        if spacing.numel() == 1:
            spacing = spacing.repeat(3)
        if spacing.shape != (3,):
            raise ValueError("grid_spacing must contain one or three values")
        if (
            not torch.all(torch.isfinite(spacing)).item()
            or torch.any(spacing <= 0.0).item()
        ):
            raise ValueError("grid_spacing values must be finite and positive")
        return spacing
