"""Bulk density-polarization response from the exact Hessian symbol.

For a homogeneous, unpolarized state of a translation-invariant grid
functional the second variation is diagonal in Fourier space. Each
wavevector carries a Hermitian matrix over the density and polarization
channels of every species, the Hessian symbol. Its inverse is the static
structure-factor matrix of the corresponding fluid, so the symbol can be
fitted to equilibrium spectra from simulation: number ``S(k)``, longitudinal
and transverse dipole spectra and the density-dipole cross spectrum.

The symbol is obtained exactly, not by finite differences: the gradient of
the excess free energy is differentiated once more at one anchor voxel per
channel (one Hessian row per channel), and the rows are Fourier summed. With
``create_graph`` the result remains differentiable with respect to the model
parameters and can be used as a training loss.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .._validation import finite_scalar, nonempty_string, nonnegative_scalar
from ..functional.polarization_ideal import FixedDipoleIdeal
from ..geometry.grid import voxel_volume
from ..objectives import LossTerm
from .curvature import _energy_only_model

__all__ = [
    "bulk_response_symbol",
    "reduce_bulk_symbol",
    "spectra_to_curvature",
    "curvature_to_spectra",
    "ngp_window",
    "BulkResponseLoss",
]

_WINDOWS = ("ngp",)


def ngp_window(
    wavevector_indices: torch.Tensor,
    grid_size: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Nearest-grid-point assignment window ``W(k)`` of lattice vectors.

    Binning point particles to the voxels of a grid with spacing ``h``
    multiplies every Fourier component by ``W(k) = prod_i sinc(k_i h_i / 2)``
    (``sinc(x) = sin(x) / x``), so the linear response of a binned field to a
    smooth perturbation is ``W(k)`` times the point-particle response. For a
    lattice vector ``k_i = 2 pi n_i / (N_i h_i)`` the spacing cancels and
    ``W = prod_i sinc(pi n_i / N_i)``. ``wavevector_indices`` is an integer
    ``[..., 3]`` tensor and ``grid_size`` (finite positive integers, three per
    grid) broadcasts against it; the result is real with the leading shape.
    ``W(0) = 1``, ``W = 2 / pi`` on an axis at the Nyquist index ``N_i / 2``,
    and ``W > 0`` inside the first Brillouin zone; an index beyond it is an
    alias of a smaller vector and raises, because the frame of
    :func:`reduce_bulk_symbol` also takes the indices as given. ``mask``
    (boolean, the leading shape) marks real members: padding entries are not
    checked and return ``W = 0``.
    """

    if (
        not torch.is_tensor(wavevector_indices)
        or wavevector_indices.dtype.is_floating_point
        or wavevector_indices.dtype == torch.bool
        or wavevector_indices.ndim < 1
        or wavevector_indices.shape[-1] != 3
    ):
        raise ValueError("wavevector_indices must be an integer [..., 3] tensor")
    device = wavevector_indices.device
    size = torch.as_tensor(grid_size, device=device)
    if size.dtype == torch.bool or size.is_complex() or size.ndim < 1 or size.shape[-1] != 3:
        raise ValueError("grid_size must hold three extents in its last axis")
    size = size.to(torch.get_default_dtype())
    if (
        not torch.all(torch.isfinite(size)).item()
        or torch.any(size < 1).item()
        or torch.any(size != size.round()).item()
    ):
        raise ValueError("grid_size must be finite positive integers")
    try:
        size = size.expand(wavevector_indices.shape)
    except RuntimeError as error:
        raise ValueError(
            "grid_size of shape {} does not broadcast against wavevector_indices of shape {}".format(
                tuple(torch.as_tensor(grid_size).shape), tuple(wavevector_indices.shape),
            )
        ) from error
    if mask is None:
        mask = torch.ones(wavevector_indices.shape[:-1], dtype=torch.bool, device=device)
    else:
        mask = torch.as_tensor(mask, device=device)
        if mask.dtype != torch.bool or mask.shape != wavevector_indices.shape[:-1]:
            raise ValueError("mask must be a boolean tensor with the leading shape of wavevector_indices")
    fraction = wavevector_indices.to(size) / size                               # n_i / N_i = k_i h_i / (2 pi)
    if torch.any(mask & (fraction.abs() > 0.5).any(dim=-1)).item():
        raise ValueError(
            "wavevector indices must lie inside the first Brillouin zone (|n_i| <= N_i / 2); "
            "reduce aliases to the centred representative"
        )
    window = torch.sinc(fraction).prod(dim=-1)                                   # torch.sinc(x) = sin(pi x)/(pi x)
    return torch.where(mask, window, torch.zeros_like(window))


def _moment(model, n_types, reference):
    ideal = getattr(model, "ideal", None)
    if not isinstance(ideal, FixedDipoleIdeal):
        raise TypeError("bulk response requires model.ideal=FixedDipoleIdeal(...)")
    moment = ideal.dipole_magnitude.to(reference).reshape(-1)
    if moment.numel() == 1:
        moment = moment.expand(n_types)
    elif moment.numel() != n_types:
        raise ValueError("dipole_magnitude must contain one value per density type")
    return moment


def _validated_bulk_state(batch):
    for key in ("rho", "dipole_density", "grid_positions", "grid_size", "grid_spacing"):
        if key not in batch:
            raise KeyError("batch is missing '{}'".format(key))
    rho, polarization = batch["rho"], batch["dipole_density"]
    if rho.ndim != 3 or polarization.shape != rho.shape + (3,):
        raise ValueError("rho must be [field, grid, type] and dipole_density [field, grid, type, 3]")
    if not torch.all(torch.isfinite(rho)).item() or torch.any(rho <= 0.0).item():
        raise ValueError("bulk response requires finite, positive density")
    spread = torch.amax(rho, dim=1) - torch.amin(rho, dim=1)
    if torch.any(spread > 1.0e-7 * torch.amax(rho, dim=1)).item():
        raise ValueError("bulk response requires a spatially uniform density")
    if torch.any(polarization != 0.0).item():
        raise ValueError("bulk response requires an unpolarized state (dipole_density = 0)")
    if "excluded_mask" in batch and torch.any(batch["excluded_mask"]).item():
        raise ValueError("bulk response batches must not exclude grid points")
    return rho, polarization


def bulk_response_symbol(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    wavevector_indices: torch.Tensor,
    *,
    create_graph: Optional[bool] = None,
    use_fast_path: bool = True,
) -> torch.Tensor:
    r"""Return the ideal-normalized Hessian symbol ``K(k)`` of a bulk state.

    ``batch`` holds homogeneous, unpolarized fields (``rho`` uniform per
    species, ``dipole_density`` zero) on periodic grids. ``wavevector_indices``
    are integer reciprocal-lattice triplets ``[field, n_vectors, 3]`` (any
    integers; they are reduced modulo the grid).

    The result is complex, ``[field, n_vectors, C, C]`` with ``C = 4 n_types``
    and channel order (density of every species, then ``P_x, P_y, P_z`` of
    species 0, 1, ...). Coordinates are normalized by the ideal gas,
    ``delta rho_a / sqrt(rho_a / dV)`` and ``sqrt(3) delta P_a / (m_a sqrt(rho_a /
    dV))`` per voxel, so the ideal gas has ``K = I`` and the equilibrium
    covariance of the Fourier modes ``exp(-i k.r)`` (normalized by particle
    number) is ``K^{-1}``: ``K^{-1}[rho, rho] = S(k)``, the transverse and
    longitudinal polarization blocks are the dipole spectra ``G_T, G_L``.

    The excess part is exact: the gradient of ``beta_F_exc`` is differentiated
    at voxel 0 for every channel and the rows are summed with
    ``exp(+i k.(r - r_0))``. The ideal part (translational plus orientational
    entropy of fixed dipoles) is the identity in these coordinates at zero
    polarization. ``create_graph`` (default: whether gradients are enabled)
    keeps the result differentiable with respect to the model parameters.
    ``use_fast_path=False`` disables exact unpolarized second-variation
    shortcuts for reference comparisons. Supported models skip nonlinear
    reciprocal transformations whose energy starts at fourth order in P;
    ordinary finite-P evaluations and finite-difference probes are unaffected.
    """

    rho, polarization = _validated_bulk_state(batch)
    if not isinstance(use_fast_path, bool):
        raise TypeError("use_fast_path must be Boolean")
    n_fields, n_grid, n_types = rho.shape
    if (
        not torch.is_tensor(wavevector_indices)
        or wavevector_indices.ndim != 3
        or wavevector_indices.shape[0] != n_fields
        or wavevector_indices.shape[2] != 3
        or wavevector_indices.dtype.is_floating_point
        or wavevector_indices.dtype == torch.bool
    ):
        raise ValueError("wavevector_indices must be an integer [field, n_vectors, 3] tensor")
    differentiable = torch.is_grad_enabled() if create_graph is None else bool(create_graph)
    moment = _moment(model, n_types, rho)
    volume = voxel_volume(batch["grid_spacing"].to(rho))
    positions = batch["grid_positions"].to(rho)
    grid_size = batch["grid_size"].to(rho)
    n_channels = 4 * n_types
    symbols = []
    for field in range(n_fields):
        single = {
            key: (value[field:field + 1] if torch.is_tensor(value) and value.ndim > 0
                  and value.shape[0] == n_fields else value)
            for key, value in batch.items()
        }
        with torch.enable_grad():
            density = rho[field:field + 1].detach().clone().requires_grad_(True)
            dipoles = polarization[field:field + 1].detach().clone().requires_grad_(True)
            single["rho"], single["dipole_density"] = density, dipoles
            energy = _energy_only_model(
                model, single, bulk_second_variation=use_fast_path,
            )["beta_F_exc"].sum()
            # An excess that does not depend on rho or P (ideal-only or
            # density-only models) contributes exact zeros, not an error.
            first_rho, first_pol = (
                torch.autograd.grad(
                    energy, (density, dipoles), create_graph=True, allow_unused=True,
                ) if energy.requires_grad else (None, None)
            )
            first_rho = torch.zeros_like(density) if first_rho is None else first_rho
            first_pol = torch.zeros_like(dipoles) if first_pol is None else first_pol
            first = torch.cat((first_rho[0, 0], first_pol[0, 0].reshape(-1)))   # anchor voxel 0
            rows = []
            for channel in range(n_channels):
                if first[channel].requires_grad:
                    row_rho, row_pol = torch.autograd.grad(
                        first[channel], (density, dipoles),
                        create_graph=differentiable, retain_graph=True, allow_unused=True,
                    )
                else:
                    row_rho = row_pol = None
                row_rho = torch.zeros_like(density) if row_rho is None else row_rho
                row_pol = torch.zeros_like(dipoles) if row_pol is None else row_pol
                rows.append(torch.cat((row_rho[0], row_pol[0].reshape(n_grid, -1)), dim=-1))
        hessian = torch.stack(rows, dim=0)                                    # [C, grid, C]
        if not differentiable:
            hessian = hessian.detach()
        relative = torch.remainder(positions[field] - positions[field, 0], grid_size[field])
        indices = wavevector_indices[field].to(rho)
        phase = 2.0 * math.pi * (indices[:, None, :] * relative[None, :, :] / grid_size[field]).sum(-1)
        waves = torch.polar(torch.ones_like(phase), phase)                    # exp(+i k.(r - r0))
        complex_dtype = waves.dtype
        symbol = torch.einsum("vg,cgd->vcd", waves, hessian.to(complex_dtype))
        density_scale = torch.sqrt(rho[field, 0] / volume[field])             # [type]
        scale = torch.cat((density_scale, (density_scale * moment / math.sqrt(3.0)).repeat_interleave(3)))
        excess = symbol * (scale[:, None] * scale[None, :]).to(complex_dtype)
        identity = torch.eye(n_channels, dtype=complex_dtype, device=rho.device)
        symbols.append(identity + excess)
    return torch.stack(symbols, dim=0)


def _k_frame(indices, lengths):
    """Return orthonormal (L, T1, T2) frames ``[..., 3, 3]`` for wavevectors."""

    wavevector = indices.to(lengths) / lengths[..., None, :]
    norm = torch.linalg.vector_norm(wavevector, dim=-1, keepdim=True)
    longitudinal = wavevector / torch.where(norm > 0, norm, torch.ones_like(norm))
    axis = torch.zeros_like(longitudinal)
    smallest = torch.argmin(longitudinal.abs(), dim=-1, keepdim=True)
    axis.scatter_(-1, smallest, 1.0)
    first = torch.linalg.cross(longitudinal, axis)
    first = first / torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    second = torch.linalg.cross(longitudinal, first)
    return torch.stack((longitudinal, first, second), dim=-2)


def reduce_bulk_symbol(
    symbol: torch.Tensor,
    wavevector_indices: torch.Tensor,
    box_lengths: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    r"""Project symbols onto the wavevector frame of an isotropic fluid.

    Returns real ``[field, n_vectors, n, n]`` curvature blocks ``density``
    (``K_{rho rho}``), ``longitudinal`` (``K_{LL}``), ``transverse`` (the mean of
    the two transverse diagonal blocks) and ``cross`` (``Im K_{rho L}``), with
    ``L = k_hat`` and ``k = 2 pi n / L_box``. For an isotropic,
    inversion-symmetric fluid these are the only nonzero blocks; the largest
    remaining coupling (density or longitudinal with transverse, between the
    two transverse directions, and the parts that symmetry makes zero) is
    returned as ``anisotropy`` relative to the largest entry of each symbol.
    """

    n_types = symbol.shape[-1] // 4
    frame = _k_frame(wavevector_indices, box_lengths).to(symbol.dtype)           # [f, v, 3, 3]
    density = symbol[..., :n_types, :n_types]
    rho_pol = symbol[..., :n_types, n_types:].reshape(*symbol.shape[:-2], n_types, n_types, 3)
    pol = symbol[..., n_types:, n_types:].reshape(*symbol.shape[:-2], n_types, 3, n_types, 3)
    rho_frame = torch.einsum("fvabd,fvsd->fvabs", rho_pol, frame)                # [f,v,n,n,3(L,T1,T2)]
    pol_frame = torch.einsum("fvsd,fvadbe,fvte->fvabst", frame, pol, frame)       # [f,v,n,n,3,3]
    longitudinal = pol_frame[..., 0, 0]
    transverse = 0.5 * (pol_frame[..., 1, 1] + pol_frame[..., 2, 2])
    cross = rho_frame[..., 0]
    scale = symbol.abs().amax(dim=(-2, -1)).clamp_min(torch.finfo(symbol.real.dtype).tiny)
    leftover = torch.stack((
        rho_frame[..., 1:].abs().amax(dim=(-3, -2, -1)),
        pol_frame[..., 0, 1:].abs().amax(dim=(-3, -2, -1)),
        pol_frame[..., 1:, 0].abs().amax(dim=(-3, -2, -1)),
        pol_frame[..., 1, 2].abs().amax(dim=(-2, -1)),
        density.imag.abs().amax(dim=(-2, -1)),
        longitudinal.imag.abs().amax(dim=(-2, -1)),
        transverse.imag.abs().amax(dim=(-2, -1)),
        cross.real.abs().amax(dim=(-2, -1)),
    ), dim=-1).amax(dim=-1) / scale
    return {
        "density": density.real,
        "longitudinal": longitudinal.real,
        "transverse": transverse.real,
        "cross": cross.imag,
        "anisotropy": leftover,
    }


def _symmetric(matrix):
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def spectra_to_curvature(spectra: torch.Tensor) -> Dict[str, torch.Tensor]:
    r"""Convert equilibrium spectra to the reduced curvature blocks.

    ``spectra`` is ``[..., 4, n, n]`` with channels ``S`` (number,
    ``<n_a n_b*> / sqrt(N_a N_b)``), ``G_L`` and ``G_T`` (``3 <p_a p_b*> /
    (sqrt(N_a N_b) m_a m_b)`` for the longitudinal and per-direction transverse
    dipole components) and ``C`` (``Im <n_a p_{L,b}*> / (sqrt(N_a N_b) m_b)``),
    for Fourier modes ``exp(-i k.r)``. The longitudinal covariance
    ``[[S, i sqrt(3) C], [-i sqrt(3) C^T, G_L]]`` is inverted to the density,
    longitudinal and cross curvature blocks and ``G_T`` to the transverse
    block, matching :func:`reduce_bulk_symbol`.
    """

    number, longitudinal, transverse, cross = (spectra[..., i, :, :] for i in range(4))
    n_types = number.shape[-1]
    imaginary = torch.complex(torch.zeros_like(cross), math.sqrt(3.0) * cross)
    top = torch.cat((_symmetric(number).to(imaginary.dtype), imaginary), dim=-1)
    bottom = torch.cat((imaginary.conj().transpose(-1, -2), _symmetric(longitudinal).to(imaginary.dtype)), dim=-1)
    inverse = torch.linalg.inv(torch.cat((top, bottom), dim=-2))
    return {
        "density": inverse[..., :n_types, :n_types].real,
        "longitudinal": inverse[..., n_types:, n_types:].real,
        "transverse": torch.linalg.inv(_symmetric(transverse)),
        "cross": inverse[..., :n_types, n_types:].imag,
    }


def curvature_to_spectra(blocks: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Inverse of :func:`spectra_to_curvature`: ``[..., 4, n, n]`` spectra."""

    density, longitudinal = blocks["density"], blocks["longitudinal"]
    n_types = density.shape[-1]
    cross = torch.complex(torch.zeros_like(blocks["cross"]), blocks["cross"])
    top = torch.cat((density.to(cross.dtype), cross), dim=-1)
    bottom = torch.cat((cross.conj().transpose(-1, -2), longitudinal.to(cross.dtype)), dim=-1)
    covariance = torch.linalg.inv(torch.cat((top, bottom), dim=-2))
    return torch.stack((
        covariance[..., :n_types, :n_types].real,
        covariance[..., n_types:, n_types:].real,
        torch.linalg.inv(blocks["transverse"]),
        covariance[..., :n_types, n_types:].imag / math.sqrt(3.0),
    ), dim=-3)


_BLOCKS = ("density", "longitudinal", "transverse", "cross")


def _elements(blocks, n_types):
    """Flatten curvature blocks to compared elements and their reference sizes."""

    upper = torch.triu_indices(n_types, n_types)
    values, references = [], []
    diagonal = {}
    for name in ("density", "longitudinal", "transverse"):
        block = blocks[name]
        diagonal[name] = torch.diagonal(block, dim1=-2, dim2=-1).abs()
        values.append(block[..., upper[0], upper[1]])
        references.append(torch.sqrt(diagonal[name][..., upper[0]] * diagonal[name][..., upper[1]]))
    values.append(blocks["cross"].reshape(*blocks["cross"].shape[:-2], -1))
    references.append(torch.sqrt(
        diagonal["density"][..., :, None] * diagonal["longitudinal"][..., None, :]
    ).reshape(*blocks["cross"].shape[:-2], -1))
    return torch.cat(values, dim=-1), torch.cat(references, dim=-1)


def _longitudinal_block(blocks):
    """Hermitian ``[[K_rr, i Im K_rL], [-i Im K_rL^T, K_LL]]`` of reduced blocks."""

    cross = torch.complex(torch.zeros_like(blocks["cross"]), blocks["cross"])
    top = torch.cat((blocks["density"].to(cross.dtype), cross), dim=-1)
    bottom = torch.cat((cross.conj().transpose(-1, -2), blocks["longitudinal"].to(cross.dtype)), dim=-1)
    return torch.cat((top, bottom), dim=-2)


def _bin_average(reduced, mask, n_types, covariance=True):
    """Average reduced member blocks over each bin.

    ``covariance=True`` averages the members' covariances (the inverse
    blocks) and inverts the mean, the same statistic as a reference spectrum
    averaged over the same vectors, so a model that reproduces every member's
    spectrum has zero loss whatever the spread of ``k`` within the bin. A bin
    in which any member block is not positive definite has no covariance;
    it uses the arithmetic mean of the curvature blocks (finite, and pointing
    back towards stability) and is reported as unstable. ``covariance=False``
    always averages curvature. The two agree when the members are related by
    a symmetry of the functional (e.g. a cubic orbit on a cubic grid).
    Returns the averaged blocks ``[field, bin, n, n]`` and the unstable mask.
    """

    n_fields, n_bins, n_members = mask.shape
    weight = mask.to(reduced["density"].dtype)
    counts = weight.sum(-1)
    safe = torch.where(counts > 0, counts, torch.ones_like(counts))

    def mean(block):
        block = block.reshape(n_fields, n_bins, n_members, *block.shape[-2:])
        kept = torch.where(mask[..., None, None], block, torch.zeros_like(block))
        return kept.sum(2) / safe.to(block.dtype)[..., None, None]

    arithmetic = {name: mean(reduced[name]) for name in _BLOCKS}
    unstable = torch.zeros((n_fields, n_bins), dtype=torch.bool, device=mask.device)
    if not covariance:
        return arithmetic, unstable
    longitudinal = _longitudinal_block(reduced)
    transverse = reduced["transverse"]
    definite = (
        (torch.linalg.eigvalsh(longitudinal).amin(-1) > 0)
        & (torch.linalg.eigvalsh(transverse).amin(-1) > 0)
    ).reshape(n_fields, n_bins, n_members)
    unstable = (mask & ~definite).any(-1)
    eye_l = torch.eye(2 * n_types, dtype=longitudinal.dtype, device=longitudinal.device)
    eye_t = torch.eye(n_types, dtype=transverse.dtype, device=transverse.device)
    # Padding members and unstable bins are inverted at the identity; the
    # former are dropped by the mask, the latter replaced below.
    replace = (~mask | unstable[..., None]).reshape(n_fields, -1)
    longitudinal = torch.where(replace[..., None, None], eye_l, longitudinal)
    transverse = torch.where(replace[..., None, None], eye_t, transverse)
    has_members = counts > 0
    mean_l = mean(torch.linalg.inv(longitudinal))
    mean_t = mean(torch.linalg.inv(transverse))
    covariance_l = torch.linalg.inv(torch.where(has_members[..., None, None], mean_l, eye_l))
    covariance_t = torch.linalg.inv(torch.where(has_members[..., None, None], mean_t, eye_t))
    consistent = {
        "density": covariance_l[..., :n_types, :n_types].real,
        "longitudinal": covariance_l[..., n_types:, n_types:].real,
        "cross": covariance_l[..., :n_types, n_types:].imag,
        "transverse": covariance_t,
    }
    averaged = {
        name: torch.where((unstable | ~has_members)[..., None, None], arithmetic[name], consistent[name])
        for name in _BLOCKS
    }
    return averaged, unstable


def _fold_symmetric(jacobian):
    """Derivative with respect to each independent entry of a symmetric matrix.

    The last two axes index the matrix. Perturbing the off-diagonal pair
    ``(a, b) = (b, a)`` changes both entries, so their derivatives are added
    on the upper triangle and the lower triangle is zeroed.
    """

    folded = jacobian + jacobian.transpose(-1, -2)
    folded = folded - torch.diag_embed(torch.diagonal(jacobian, dim1=-2, dim2=-1))
    return torch.triu(folded)


def _propagated_scale(spectra, sem, relative_floor, n_types):
    """Linear propagation of independent spectrum SEMs to curvature elements."""

    flat = spectra.reshape(-1, 4 * n_types * n_types)

    def elements(vector):
        return _elements(spectra_to_curvature(vector.reshape(4, n_types, n_types)), n_types)[0]

    target, reference = _elements(spectra_to_curvature(spectra), n_types)
    if sem is None:
        propagated = torch.zeros_like(target)
    else:
        jacobian = torch.func.vmap(torch.func.jacrev(elements))(flat)          # [B, E, 4 n n]
        jacobian = jacobian.reshape(*jacobian.shape[:2], 4, n_types, n_types)
        # S, G_L and G_T are symmetric: one independent quantity per pair.
        symmetric = _fold_symmetric(jacobian[:, :, :3])
        jacobian = torch.cat((symmetric, jacobian[:, :, 3:]), dim=2)
        sigma = sem.reshape(-1, 4, n_types, n_types)
        variance = (jacobian.square() * sigma.square()[:, None]).sum((-3, -2, -1))
        propagated = variance.sqrt().reshape(target.shape)
    scale = torch.sqrt(propagated.square() + (relative_floor * reference).square())
    return target, scale


class BulkResponseLoss(LossTerm):
    r"""Fit the exact bulk Hessian symbol to equilibrium spectra.

    Each batch item is one homogeneous, unpolarized, unmasked state. For every
    wavevector bin the model's symbol (:func:`bulk_response_symbol`) is
    reduced to the curvature blocks of an isotropic fluid
    (:func:`reduce_bulk_symbol`) and averaged over the bin's member vectors.
    With ``bin_average="covariance"`` (default) the members' covariances are
    averaged and the mean inverted, the same statistic as a reference
    spectrum averaged over the same vectors: a model that reproduces every
    member's spectrum has zero loss whatever the spread of ``k`` within a
    bin. Bins with a member that is not positive definite (no covariance)
    use the arithmetic mean of curvature instead and are reported in
    ``unstable_bins``. ``"curvature"`` always averages curvature; it differs
    from the reference statistic unless the members are related by a
    symmetry of the functional, so bins should then be cubic orbits. The reference spectra are converted to the same blocks with
    :func:`spectra_to_curvature`; the comparison is in curvature (``K = S^{-1}``),
    which is defined even for an unstable model and is the quantity the solver
    sees. For one species the compared elements are ``K_{rho rho}``,
    ``K_{LL}``, ``K_{TT}`` and ``Im K_{rho L}``; for ``n`` species the upper
    triangles of the three symmetric blocks and the full cross block.

    ``window="ngp"`` compares point-particle spectra with a model of binned
    fields. The response of a field binned to the voxels (nearest grid point)
    to a smooth perturbation is the point response times the assignment
    window ``W(k)`` (:func:`ngp_window`), the same factor on every channel, so
    each member's symbol is multiplied by ``W(k_m)`` (its spectrum divided by
    ``W(k_m)``) before the bin average, and the reference keeps its
    point-particle values: do not also scale the reference. The factor is
    exact per member, so bins may mix lattice vectors with different ``W``;
    the zero wavevector has ``W = 1``. Without the option a model fitted to
    point spectra is softer than its own training fields by ``1 / W - 1``.
    ``window=None`` (default) compares the model's spectra as they are. The
    result reports the applied ``window`` per member (one without the option,
    zero on padding).

    Like :class:`FourierResponseLoss`, ``loss_fn`` (default smooth L1) is applied
    to prediction and target divided by a per-element scale. The scale is the
    reference SEM propagated linearly to each curvature element, combined in
    quadrature with ``relative_floor`` times the element's reference size
    (its own magnitude for diagonal entries, the geometric mean of the
    matching diagonals otherwise); without SEM it is the relative floor alone.

    The zero wavevector is handled separately: a canonical (NVT) reference has
    no number fluctuation at ``k = 0``, so only the polarization block enters.
    Its isotropic average ``(1/3) sum_d K[P_a d, P_b d]`` is compared with the
    inverse of the total-dipole fluctuation matrix ``g_K``
    (``<M_a . M_b> / (sqrt(N_a N_b) m_a m_b)``).

    Batch keys (shapes per field): ``wavevector_indices_key``
    ``[bin, member, 3]`` integer lattice vectors, ``member_mask_key``
    ``[bin, member]`` (padding false), ``spectra_key`` ``[bin, 4, n, n]`` with
    channels ``S, G_L, G_T, C`` (see :func:`spectra_to_curvature`), optional
    ``spectra_sem_key`` (same shape), ``weights_key`` ``[bin]``, and
    ``zero_wavevector_key`` / ``zero_wavevector_sem_key`` ``[n, n]``.
    """

    def __init__(
        self,
        *,
        wavevector_indices_key: str = "response_wavevector_indices",
        member_mask_key: str = "response_member_mask",
        spectra_key: str = "response_spectra",
        spectra_sem_key: Optional[str] = "response_spectra_sem",
        weights_key: Optional[str] = None,
        zero_wavevector_key: Optional[str] = None,
        zero_wavevector_sem_key: Optional[str] = None,
        zero_wavevector_weight: float = 1.0,
        relative_floor: float = 0.02,
        bin_average: str = "covariance",
        window: Optional[str] = None,
        loss_fn: Optional[nn.Module] = None,
        weight: float = 1.0,
        name: str = "bulk_response",
    ) -> None:
        super().__init__(name, weight)
        if loss_fn is None:
            loss_fn = nn.SmoothL1Loss(reduction="none")
        if not isinstance(loss_fn, nn.Module):
            raise TypeError("loss_fn must be a torch.nn.Module")
        self.loss_fn = loss_fn
        self.wavevector_indices_key = nonempty_string(wavevector_indices_key, "wavevector_indices_key")
        self.member_mask_key = nonempty_string(member_mask_key, "member_mask_key")
        self.spectra_key = nonempty_string(spectra_key, "spectra_key")
        self.spectra_sem_key = None if spectra_sem_key is None else nonempty_string(spectra_sem_key, "spectra_sem_key")
        self.weights_key = None if weights_key is None else nonempty_string(weights_key, "weights_key")
        self.zero_wavevector_key = None if zero_wavevector_key is None else nonempty_string(zero_wavevector_key, "zero_wavevector_key")
        self.zero_wavevector_sem_key = (
            None if zero_wavevector_sem_key is None
            else nonempty_string(zero_wavevector_sem_key, "zero_wavevector_sem_key")
        )
        if self.zero_wavevector_sem_key is not None and self.zero_wavevector_key is None:
            raise ValueError("zero_wavevector_sem_key requires zero_wavevector_key")
        self.zero_wavevector_weight = nonnegative_scalar(zero_wavevector_weight, "zero_wavevector_weight")
        relative_floor = finite_scalar(relative_floor, "relative_floor")
        if relative_floor <= 0.0:
            raise ValueError("relative_floor must be positive")
        self.relative_floor = relative_floor
        bin_average = nonempty_string(bin_average, "bin_average")
        if bin_average not in ("covariance", "curvature"):
            raise ValueError("bin_average must be 'covariance' or 'curvature'")
        self.bin_average = bin_average
        if window is not None:
            window = nonempty_string(window, "window")
            if window not in _WINDOWS:
                raise ValueError("window must be None or one of {}".format(", ".join(repr(w) for w in _WINDOWS)))
        self.window = window

    def evaluate(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return the loss with reduced predictions, targets and scales."""

        if model is None:
            raise ValueError("BulkResponseLoss requires the model")
        rho = batch["rho"]
        n_fields, _, n_types = rho.shape
        indices = _required(batch, self.wavevector_indices_key)
        mask = _required(batch, self.member_mask_key).to(torch.bool)
        if indices.ndim != 4 or indices.shape[0] != n_fields or indices.shape[-1] != 3:
            raise ValueError("{} must have shape [field, bin, member, 3]".format(self.wavevector_indices_key))
        if mask.shape != indices.shape[:-1]:
            raise ValueError("{} must have shape [field, bin, member]".format(self.member_mask_key))
        n_bins, n_members = indices.shape[1:3]
        spectra = _required(batch, self.spectra_key).detach().to(rho)
        if spectra.shape != (n_fields, n_bins, 4, n_types, n_types):
            raise ValueError("{} must have shape [field, bin, 4, n_types, n_types]".format(self.spectra_key))
        sem = None
        if self.spectra_sem_key is not None and self.spectra_sem_key in batch:
            sem = batch[self.spectra_sem_key].detach().to(rho)
            if sem.shape != spectra.shape or torch.any(sem < 0).item() or not torch.all(torch.isfinite(sem)).item():
                raise ValueError("{} must be finite, nonnegative and shaped like the spectra".format(self.spectra_sem_key))
        if not torch.all(torch.isfinite(spectra)).item():
            raise ValueError("{} must be finite".format(self.spectra_key))

        # Padding members must never reach the frame construction: their
        # index (often (0, 0, 0)) has no wavevector frame. Replace them by a
        # harmless lattice vector and drop them with a where, not a product.
        if torch.any(mask & (indices == 0).all(dim=-1)).item():
            raise ValueError("bin members must be nonzero wavevectors; k = 0 is the zero_wavevector term")
        placeholder = torch.zeros_like(indices)
        placeholder[..., 0] = 1
        safe_indices = torch.where(mask[..., None], indices, placeholder)
        flat_indices = safe_indices.reshape(n_fields, n_bins * n_members, 3)
        include_zero = self.zero_wavevector_key is not None
        if include_zero:
            flat_indices = torch.cat((flat_indices, torch.zeros_like(flat_indices[:, :1])), dim=1)
        symbol = bulk_response_symbol(model, batch, flat_indices)
        lengths = batch["grid_size"].to(rho) * batch["grid_spacing"].to(rho)
        vector_symbol = symbol[:, : n_bins * n_members]
        member_indices = flat_indices[:, : n_bins * n_members]
        window = torch.where(mask, torch.ones_like(mask, dtype=rho.dtype), torch.zeros_like(mask, dtype=rho.dtype))
        if self.window == "ngp":
            # The binned model responds with W(k) times the point response:
            # scale each member's curvature by W (its spectrum by 1 / W)
            # so the point reference is compared without change. W is a
            # positive scalar per member, so definiteness and the relative
            # anisotropy are unaffected. Only real members are checked
            # against the zone and scaled; padding keeps its placeholder
            # symbol and is dropped by the mask as before.
            flat_mask = mask.reshape(n_fields, -1)
            window = ngp_window(member_indices, batch["grid_size"][:, None, :], flat_mask).to(rho)
            factor = torch.where(flat_mask, window, torch.ones_like(window))
            vector_symbol = vector_symbol * factor[..., None, None].to(vector_symbol.dtype)
            window = window.reshape(mask.shape)
        reduced = reduce_bulk_symbol(vector_symbol, member_indices, lengths)
        counts = mask.sum(dim=-1).to(rho)                                    # [field, bin]
        valid_bin = counts > 0
        averaged, unstable = _bin_average(
            reduced, mask, n_types, covariance=self.bin_average == "covariance",
        )
        prediction, _ = _elements(averaged, n_types)
        target, scale = _propagated_scale(spectra, sem, self.relative_floor, n_types)
        element_weight = valid_bin.to(rho)[..., None].expand_as(target).clone()
        if self.weights_key is not None:
            weights = _required(batch, self.weights_key).detach().to(rho)
            if weights.shape != (n_fields, n_bins) or torch.any(weights < 0).item():
                raise ValueError("{} must be nonnegative with shape [field, bin]".format(self.weights_key))
            element_weight = element_weight * weights[..., None]
        element_loss = self.loss_fn(prediction / scale, target / scale)
        total = (element_loss * element_weight).sum()
        total_weight = element_weight.sum()
        result = {
            "prediction": prediction,
            "target": target,
            "scale": scale,
            "element_weight": element_weight,
            "anisotropy": torch.where(
                mask.reshape(n_fields, -1), reduced["anisotropy"], torch.zeros_like(reduced["anisotropy"]),
            ).amax(dim=-1),
            "unstable_bins": unstable & valid_bin,
            "window": window,
        }
        if include_zero:
            zero_symbol = symbol[:, -1, n_types:, n_types:].real.reshape(n_fields, n_types, 3, n_types, 3)
            zero_prediction_block = torch.diagonal(zero_symbol, dim1=2, dim2=4).mean(-1)   # [f, n, n]
            fluctuation = _symmetric(_required(batch, self.zero_wavevector_key).detach().to(rho))
            if fluctuation.shape != (n_fields, n_types, n_types):
                raise ValueError("{} must have shape [field, n_types, n_types]".format(self.zero_wavevector_key))
            zero_target_block = torch.linalg.inv(fluctuation)
            upper = torch.triu_indices(n_types, n_types)
            zero_prediction = zero_prediction_block[..., upper[0], upper[1]]
            zero_target = zero_target_block[..., upper[0], upper[1]]
            diagonal = torch.diagonal(zero_target_block, dim1=-2, dim2=-1).abs()
            zero_reference = torch.sqrt(diagonal[..., upper[0]] * diagonal[..., upper[1]])
            zero_propagated = torch.zeros_like(zero_target)
            if self.zero_wavevector_sem_key is not None and self.zero_wavevector_sem_key in batch:
                zero_sem = batch[self.zero_wavevector_sem_key].detach().to(rho)
                if zero_sem.shape != fluctuation.shape or torch.any(zero_sem < 0).item():
                    raise ValueError("{} must be nonnegative and shaped like the fluctuation".format(self.zero_wavevector_sem_key))

                def inverse_elements(matrix):
                    return torch.linalg.inv(_symmetric(matrix))[upper[0], upper[1]]

                jacobian = _fold_symmetric(
                    torch.func.vmap(torch.func.jacrev(inverse_elements))(fluctuation)
                )
                variance = (jacobian.square() * zero_sem.square()[:, None]).sum((-2, -1))
                zero_propagated = variance.sqrt()
            zero_scale = torch.sqrt(zero_propagated.square() + (self.relative_floor * zero_reference).square())
            zero_loss = self.loss_fn(zero_prediction / zero_scale, zero_target / zero_scale)
            zero_weight = torch.full_like(zero_loss, self.zero_wavevector_weight)
            total = total + (zero_loss * zero_weight).sum()
            total_weight = total_weight + zero_weight.sum()
            result.update(
                zero_prediction=zero_prediction, zero_target=zero_target,
                zero_scale=zero_scale,
            )
        if total_weight.item() <= 0.0:
            raise ValueError("bulk response weights must have a positive sum")
        result["loss"] = self.weight * total / total_weight
        return result


def _required(batch, key):
    if key not in batch:
        raise KeyError("batch is missing '{}'".format(key))
    return batch[key]
