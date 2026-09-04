"""Small tensor helpers for regular three-dimensional grids."""

import math
from typing import Optional, Tuple

import torch
from torch.nn import functional as F


def common_grid_size(
    grid_size: torch.Tensor,
    leading_shape: torch.Size,
) -> Tuple[int, int, int]:
    """Validate and return one regular-grid shape shared by a batch."""

    sizes = torch.as_tensor(grid_size).detach().reshape(-1, 3)
    if sizes.shape[0] not in (1, math.prod(leading_shape)):
        raise ValueError("grid_size leading shape must match rho")
    rounded = torch.round(sizes).to(dtype=torch.long)
    if not torch.allclose(
        sizes.to(dtype=torch.float64),
        rounded.to(dtype=torch.float64),
    ):
        raise ValueError("grid_size values must be integers")
    if torch.any(rounded <= 0).item():
        raise ValueError("grid_size values must be positive")
    if not torch.all(rounded == rounded[0]).item():
        raise ValueError("all fields in one batch must share grid_size")
    return tuple(int(value) for value in rounded[0].cpu().tolist())


def gather_neighbors(
    values: torch.Tensor,
    neighbor_index: torch.Tensor,
) -> torch.Tensor:
    """Gather periodic neighborhoods of a grid-aligned tensor.

    ``values`` has shape ``[..., n_grid, *feature_shape]`` and the integer
    neighbor table has shape ``[..., n_grid, n_neighbors]``. The returned
    tensor has shape ``[..., n_grid, n_neighbors, *feature_shape]``.
    """

    if neighbor_index.ndim < 2:
        raise ValueError(
            "neighbor_index must have shape [..., n_grid, n_neighbors]"
        )
    if neighbor_index.dtype != torch.long:
        raise TypeError("neighbor_index must have dtype torch.long")

    n_leading = neighbor_index.ndim - 2
    if values.ndim <= n_leading:
        raise ValueError(
            "values must have shape [..., n_grid, *feature_shape]"
        )
    leading_shape = neighbor_index.shape[:-2]
    if values.shape[:n_leading] != leading_shape:
        raise ValueError("values and neighbor_index leading shapes must match")
    n_grid = neighbor_index.shape[-2]
    if values.shape[n_leading] != n_grid:
        raise ValueError("values and neighbor_index grid sizes must match")

    feature_shape = values.shape[n_leading + 1 :]
    n_features = math.prod(feature_shape) if feature_shape else 1
    n_neighbors = neighbor_index.shape[-1]

    # Flatten leading field dimensions and trailing feature dimensions around
    # the grid axis. torch.gather then selects the grid axis independently for
    # every field and feature.
    values_flat = values.reshape(-1, n_grid, n_features)
    index_flat = neighbor_index.reshape(-1, n_grid * n_neighbors)
    gather_index = index_flat.unsqueeze(-1).expand(-1, -1, n_features)
    local_values = torch.gather(values_flat, dim=1, index=gather_index)
    return local_values.reshape(
        *leading_shape,
        n_grid,
        n_neighbors,
        *feature_shape,
    )


def _integer_tensor(
    values: torch.Tensor,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    """Return integer-valued geometry as a long tensor."""

    tensor = torch.as_tensor(values, device=device)
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise TypeError(f"{name} must be integer-valued")
    if not tensor.is_floating_point():
        return tensor.to(dtype=torch.long)
    rounded = torch.round(tensor).to(dtype=torch.long)
    if not torch.allclose(
        tensor.to(dtype=torch.float64),
        rounded.to(dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError(f"{name} must be integer-valued")
    return rounded


def _ravel_grid_indices(
    positions: torch.Tensor,
    grid_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Return C-order flat indices for three-dimensional coordinates."""

    return (
        (positions[..., 0] * grid_shape[1] + positions[..., 1])
        * grid_shape[2]
        + positions[..., 2]
    )


def _flat_grid_positions(
    grid_positions: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    leading_shape: torch.Size,
    device: torch.device,
) -> Tuple[torch.Tensor, bool]:
    """Return validated C-order indices and whether rows are canonical."""

    n_grid = math.prod(grid_shape)
    positions = _integer_tensor(grid_positions, "grid_positions", device)
    if positions.shape == (n_grid, 3):
        positions = positions.reshape(1, n_grid, 3).expand(
            math.prod(leading_shape) if leading_shape else 1,
            -1,
            -1,
        )
    elif positions.shape == (*leading_shape, n_grid, 3):
        positions = positions.reshape(-1, n_grid, 3)
    else:
        raise ValueError(
            "grid_positions must have shape [n_grid, 3] or "
            "[..., n_grid, 3] matching values"
        )
    sizes = positions.new_tensor(grid_shape)
    if torch.any(positions < 0).item() or torch.any(positions >= sizes).item():
        raise ValueError("grid_positions are outside grid_size")
    flat = _ravel_grid_indices(positions, grid_shape)
    canonical = torch.arange(n_grid, device=device).expand_as(flat)
    rows_are_canonical = torch.equal(flat, canonical)
    if (
        not rows_are_canonical
        and not torch.equal(torch.sort(flat, dim=-1).values, canonical)
    ):
        raise ValueError(
            "grid_positions must contain one complete regular grid"
        )
    return flat, rows_are_canonical


def _dense_stencil_weight(
    stencil_basis: torch.Tensor,
    offsets: torch.Tensor,
    n_channels: int,
) -> Tuple[
    torch.Tensor,
    Tuple[int, int, int],
    Tuple[int, int, int],
]:
    """Scatter a sparse stencil into grouped-convolution weights."""

    n_neighbors, n_radial, n_monomials = stencil_basis.shape
    minimum_offsets = tuple(
        int(value)
        for value in offsets.min(dim=0).values.detach().cpu().tolist()
    )
    maximum_offsets = tuple(
        int(value)
        for value in offsets.max(dim=0).values.detach().cpu().tolist()
    )
    kernel_shape = tuple(
        maximum - minimum + 1
        for minimum, maximum in zip(minimum_offsets, maximum_offsets)
    )
    kernel_coordinates = offsets - offsets.new_tensor(minimum_offsets)
    kernel_flat = _ravel_grid_indices(kernel_coordinates, kernel_shape)
    dense_basis = stencil_basis.new_zeros(
        n_radial,
        n_monomials,
        math.prod(kernel_shape),
    ).scatter_add(
        2,
        kernel_flat.reshape(1, 1, n_neighbors).expand(
            n_radial,
            n_monomials,
            -1,
        ),
        stencil_basis.permute(1, 2, 0),
    )
    weight = dense_basis.reshape(
        n_radial,
        n_monomials,
        *kernel_shape,
    )[:, None].expand(
        n_radial,
        n_channels,
        n_monomials,
        *kernel_shape,
    ).reshape(
        n_radial * n_channels * n_monomials,
        1,
        *kernel_shape,
    )
    return weight, minimum_offsets, maximum_offsets


def _periodic_extend_grid(
    values: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    minimum_offsets: Tuple[int, int, int],
    maximum_offsets: Tuple[int, int, int],
) -> torch.Tensor:
    """Extend three grid axes by modular indexing for periodic convolution."""

    for axis, (size, minimum, maximum) in enumerate(
        zip(grid_shape, minimum_offsets, maximum_offsets),
        start=2,
    ):
        periodic_index = torch.arange(
            minimum,
            size + maximum,
            device=values.device,
        ).remainder(size)
        values = values.index_select(axis, periodic_index)
    return values


def _periodic_stencil_fft(
    grid_values: torch.Tensor,
    stencil_basis: torch.Tensor,
    offsets: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    *,
    shared_values: bool,
) -> torch.Tensor:
    """Return a periodic stencil cross-correlation on canonical grids.

    ``grid_values`` is ``[F, N, C, X, Y, Z]`` for radial-channel-paired
    message inputs. With ``shared_values=True`` it is instead
    ``[F, C, X, Y, Z]`` and the same density spectrum is shared by every
    radial channel. Both paths return ``[F, N, C, K, X, Y, Z]``.
    """

    n_neighbors, n_radial, n_monomials = stencil_basis.shape
    n_grid = math.prod(grid_shape)
    periodic_offsets = offsets.remainder(offsets.new_tensor(grid_shape))
    kernel_flat = _ravel_grid_indices(periodic_offsets, grid_shape)
    periodic_basis = stencil_basis.new_zeros(
        n_radial,
        n_monomials,
        n_grid,
    ).scatter_add(
        2,
        kernel_flat.reshape(1, 1, n_neighbors).expand(
            n_radial,
            n_monomials,
            -1,
        ),
        stencil_basis.permute(1, 2, 0),
    ).reshape(n_radial, n_monomials, *grid_shape)
    value_spectrum = torch.fft.rfftn(
        grid_values,
        dim=(-3, -2, -1),
    )
    basis_spectrum = torch.fft.rfftn(
        periodic_basis,
        dim=(-3, -2, -1),
    )
    if shared_values:
        product = (
            value_spectrum[:, None, :, None]
            * basis_spectrum[None, :, None].conj()
        )
    else:
        product = (
            value_spectrum[:, :, :, None]
            * basis_spectrum[:, None].conj()
        )
    return torch.fft.irfftn(
        product,
        s=grid_shape,
        dim=(-3, -2, -1),
    )


def periodic_density_convolution(
    values: torch.Tensor,
    stencil_basis: torch.Tensor,
    grid_positions: torch.Tensor,
    grid_size: torch.Tensor,
    stencil_positions: torch.Tensor,
) -> torch.Tensor:
    r"""Contract one density spectrum with every periodic stencil channel.

    ``values`` has shape ``[..., G, C]`` and ``stencil_basis`` has shape
    ``[J, N, K]``. The returned circular cross-correlation

    ``output[i,n,k,c] = sum_j basis[j,n,k] values[i+s_j,c]``

    has shape ``[..., G, N, K, C]``. Each input-channel Fourier transform is
    computed once and shared over all radial channels and monomials.
    """

    if values.ndim < 2:
        raise ValueError("values must have shape [..., G, C]")
    if stencil_basis.ndim != 3:
        raise ValueError("stencil_basis must have shape [J, N, K]")
    leading_shape = values.shape[:-2]
    n_grid, n_channels = values.shape[-2:]
    n_neighbors, n_radial, n_monomials = stencil_basis.shape
    offsets = _integer_tensor(
        stencil_positions,
        "stencil_positions",
        values.device,
    )
    if offsets.ndim != 2 or offsets.shape[-1] != 3:
        raise ValueError("stencil_positions must have shape [J, 3]")
    if offsets.shape[0] != n_neighbors:
        raise ValueError(
            "stencil_positions and stencil_basis counts must match"
        )
    grid_shape = common_grid_size(grid_size, leading_shape)
    if math.prod(grid_shape) != n_grid:
        raise ValueError("grid_size product must match the values grid axis")

    flat_positions, rows_are_canonical = _flat_grid_positions(
        grid_positions,
        grid_shape,
        leading_shape,
        values.device,
    )
    n_fields = math.prod(leading_shape) if leading_shape else 1
    values_flat = values.reshape(n_fields, n_grid, n_channels)
    if rows_are_canonical:
        canonical_values = values_flat
    else:
        scatter_index = flat_positions[..., None].expand_as(values_flat)
        canonical_values = torch.zeros_like(values_flat).scatter(
            1,
            scatter_index,
            values_flat,
        )
    grid_values = canonical_values.reshape(
        n_fields,
        *grid_shape,
        n_channels,
    ).permute(0, 4, 1, 2, 3)
    convolved = _periodic_stencil_fft(
        grid_values,
        stencil_basis,
        offsets,
        grid_shape,
        shared_values=True,
    )
    canonical_output = convolved.permute(0, 4, 5, 6, 1, 3, 2).reshape(
        n_fields,
        n_grid,
        n_radial,
        n_monomials,
        n_channels,
    )
    if rows_are_canonical:
        output = canonical_output
    else:
        gather_index = flat_positions[..., None, None, None].expand_as(
            canonical_output
        )
        output = torch.gather(canonical_output, 1, gather_index)
    return output.reshape(
        *leading_shape,
        n_grid,
        n_radial,
        n_monomials,
        n_channels,
    )


def periodic_stencil_convolution(
    values: torch.Tensor,
    stencil_basis: torch.Tensor,
    grid_positions: torch.Tensor,
    grid_size: torch.Tensor,
    stencil_positions: torch.Tensor,
    *,
    backend: str = "conv3d",
) -> torch.Tensor:
    r"""Contract a periodic stencil without materializing neighborhoods.

    ``values`` has shape ``[..., G, N, C]``, ``stencil_basis`` has shape
    ``[J, N, K]``, and ``stencil_positions`` gives the corresponding integer
    offsets as ``[J, 3]``. The result is

    ``output[i,n,k,c] = sum_j basis[j,n,k] values[i+s_j,n,c]``

    with shape ``[..., G, N, K, C]``. ``backend="conv3d"`` uses a grouped
    three-dimensional cross-correlation, while ``backend="fft"`` evaluates
    the same circular cross-correlation in reciprocal space. Both avoid the
    much larger explicit ``[..., G, J, N, C]`` neighbor tensor. Arbitrary
    complete input row order is preserved through ``grid_positions``.
    """

    if not isinstance(backend, str):
        raise TypeError("backend must be 'conv3d' or 'fft'")
    backend = backend.lower()
    if backend not in ("conv3d", "fft"):
        raise ValueError("backend must be 'conv3d' or 'fft'")
    if values.ndim < 3:
        raise ValueError("values must have shape [..., G, N, C]")
    if stencil_basis.ndim != 3:
        raise ValueError("stencil_basis must have shape [J, N, K]")
    leading_shape = values.shape[:-3]
    n_grid, n_radial, n_channels = values.shape[-3:]
    n_neighbors, basis_radial, n_monomials = stencil_basis.shape
    if basis_radial != n_radial:
        raise ValueError(
            "stencil_basis radial channels do not match values"
        )
    offsets = _integer_tensor(
        stencil_positions,
        "stencil_positions",
        values.device,
    )
    if offsets.ndim != 2 or offsets.shape[-1] != 3:
        raise ValueError("stencil_positions must have shape [J, 3]")
    if offsets.shape[0] != n_neighbors:
        raise ValueError(
            "stencil_positions and stencil_basis counts must match"
        )
    grid_shape = common_grid_size(grid_size, leading_shape)
    if math.prod(grid_shape) != n_grid:
        raise ValueError("grid_size product must match the values grid axis")

    flat_positions, rows_are_canonical = _flat_grid_positions(
        grid_positions,
        grid_shape,
        leading_shape,
        values.device,
    )
    n_fields = math.prod(leading_shape) if leading_shape else 1
    values_flat = values.reshape(n_fields, n_grid, n_radial, n_channels)
    if rows_are_canonical:
        canonical_values = values_flat
    else:
        scatter_index = flat_positions[..., None, None].expand_as(values_flat)
        canonical_values = torch.zeros_like(values_flat).scatter(
            1,
            scatter_index,
            values_flat,
        )

    grid_values = canonical_values.reshape(
        n_fields,
        *grid_shape,
        n_radial,
        n_channels,
    ).permute(0, 4, 5, 1, 2, 3)
    if backend == "conv3d":
        # Each (radial, channel) pair is an independent convolution group.
        # conv3d is cross-correlation, so offsets are not reversed.
        weight, minimum_offsets, maximum_offsets = _dense_stencil_weight(
            stencil_basis,
            offsets,
            n_channels,
        )
        extended_values = _periodic_extend_grid(
            grid_values.reshape(
                n_fields,
                n_radial * n_channels,
                *grid_shape,
            ),
            grid_shape,
            minimum_offsets,
            maximum_offsets,
        )
        convolved = F.conv3d(
            extended_values,
            weight,
            groups=n_radial * n_channels,
        ).reshape(
            n_fields,
            n_radial,
            n_channels,
            n_monomials,
            *grid_shape,
        )
    else:
        # y[i] = sum_j basis[j] * values[i + offset[j]]. Placing each basis
        # row at its periodic positive offset therefore requires the complex
        # conjugate of its Fourier transform. scatter_add also preserves exact
        # duplicate offsets and aliases on grids smaller than the stencil.
        convolved = _periodic_stencil_fft(
            grid_values,
            stencil_basis,
            offsets,
            grid_shape,
            shared_values=False,
        )

    canonical_output = convolved.permute(0, 4, 5, 6, 1, 3, 2).reshape(
        n_fields,
        n_grid,
        n_radial,
        n_monomials,
        n_channels,
    )
    if rows_are_canonical:
        output = canonical_output
    else:
        gather_index = flat_positions[..., None, None, None].expand_as(
            canonical_output
        )
        output = torch.gather(canonical_output, 1, gather_index)
    return output.reshape(
        *leading_shape,
        n_grid,
        n_radial,
        n_monomials,
        n_channels,
    )


def grid_spacing_tensor(
    grid_spacing: object,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return finite positive Cartesian spacing with shape ``[3]``.

    A scalar spacing is expanded to the three Cartesian directions. The
    returned tensor is detached and cloned because grid spacing is fixed
    geometry rather than a differentiable model input.
    """

    if dtype is None:
        dtype = torch.get_default_dtype()
    spacing = torch.as_tensor(
        grid_spacing,
        dtype=dtype,
        device=device,
    ).detach().clone().reshape(-1)
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


def voxel_volume(grid_spacing: torch.Tensor) -> torch.Tensor:
    r"""Return the regular-grid quadrature weight ``Delta V``.

    ``grid_spacing`` must have shape ``[..., 3]``. The result has shape
    ``[...]`` and is ``h_x h_y h_z``; for cubic voxels this is ``h**3``.
    """

    if not torch.is_tensor(grid_spacing):
        raise TypeError("grid_spacing must be a torch.Tensor")
    if grid_spacing.ndim < 1 or grid_spacing.shape[-1] != 3:
        raise ValueError("grid_spacing must end with three Cartesian values")
    if (
        not torch.all(torch.isfinite(grid_spacing)).item()
        or torch.any(grid_spacing <= 0.0).item()
    ):
        raise ValueError("grid_spacing values must be finite and positive")
    return torch.prod(grid_spacing, dim=-1)


def require_matching_grid_spacing(
    grid_spacing: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    """Reject input spacing inconsistent with fixed model geometry."""

    if not torch.is_tensor(grid_spacing):
        raise TypeError("input grid_spacing must be a torch.Tensor")
    if grid_spacing.ndim < 1 or grid_spacing.shape[-1] != 3:
        raise ValueError(
            "input grid_spacing must end with three Cartesian values"
        )
    comparison = grid_spacing.to(dtype=reference.dtype)
    expected = reference.to(device=grid_spacing.device).expand_as(comparison)
    if not torch.allclose(comparison, expected):
        raise ValueError(
            "input grid_spacing does not match the trained model"
        )
