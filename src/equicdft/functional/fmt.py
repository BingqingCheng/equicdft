"""Fundamental measure theory for hard-sphere mixtures on the periodic grid."""

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .._config import Configurable, make_config, register
from .._validation import nonempty_string, positive_integer, positive_scalar
from ..geometry.grid import _flat_grid_positions, common_grid_size, grid_spacing_tensor
from .base import EnergyReadout, ReadoutContext
from .reciprocal import _wavevector_axes

__all__ = [
    "FMTReadout",
    "FMT_VERSIONS",
    "FMT_THIRD_TERMS",
    "scaled_particle_free_energy_density",
    "carnahan_starling_free_energy_density",
    "fmt_free_energy_density",
]


FMT_VERSIONS = ("rosenfeld", "white_bear_i")
FMT_THIRD_TERMS = ("q3", "tensor", "original")


# ----------------------------------------------------------------- bulk limits
def scaled_particle_free_energy_density(density, packing_fraction):
    r"""``beta f_ex`` of scaled-particle theory (the Rosenfeld bulk limit).

    ``rho [-ln(1-eta) + 3 eta/(1-eta) + 3 eta^2 / (2 (1-eta)^2)]``, the
    Percus-Yevick compressibility equation of state.
    """

    eta = packing_fraction
    log = torch.log1p(-eta) if torch.is_tensor(eta) else math.log1p(-eta)
    return density * (-log + 3.0 * eta / (1.0 - eta) + 1.5 * eta ** 2 / (1.0 - eta) ** 2)


def carnahan_starling_free_energy_density(density, packing_fraction):
    r"""``beta f_ex`` of Carnahan-Starling (the White Bear bulk limit).

    ``rho eta (4 - 3 eta) / (1 - eta)^2``.
    """

    eta = packing_fraction
    return density * eta * (4.0 - 3.0 * eta) / (1.0 - eta) ** 2


def _white_bear_bracket(n3: torch.Tensor) -> torch.Tensor:
    r"""``[n3 + (1 - n3)^2 ln(1 - n3)] / n3^2`` with its small-``n3`` series.

    The bracket cancels to ``O(n3^2)`` at small packing; below ``1e-2`` the
    series ``3/2 - n3/3 - n3^2/12 - n3^3/30 - n3^4/60`` is used instead.
    """

    small = n3.abs() < 1.0e-2
    safe = torch.where(small, torch.full_like(n3, 0.5), n3)
    exact = (safe + (1.0 - safe) ** 2 * torch.log1p(-safe)) / safe.square()
    series = 1.5 - n3 / 3.0 - n3.square() / 12.0 - n3 ** 3 / 30.0 - n3 ** 4 / 60.0
    return torch.where(small, series, exact)


def fmt_free_energy_density(
    n0: torch.Tensor,
    n1: torch.Tensor,
    n2: torch.Tensor,
    n3: torch.Tensor,
    n1v: torch.Tensor,
    n2v: torch.Tensor,
    version: str = "white_bear_i",
    third_term: str = "q3",
    n2t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Return ``Phi`` per grid point from the weighted densities.

    ``n1v`` and ``n2v`` carry the Cartesian component last, ``n2t`` the
    symmetric tensor ``[..., 3, 3]`` (needed for ``third_term="tensor"``).

    ``Phi = -n0 ln(1-n3) + (n1 n2 - n1v.n2v)/(1-n3) + P phi_3(n3)``,

    where ``phi_3`` selects the version, ``1 / (24 pi (1-n3)^2)`` for
    Rosenfeld and ``[n3 + (1-n3)^2 ln(1-n3)] / (36 pi n3^2 (1-n3)^2)`` for
    White Bear mark I (Carnahan-Starling bulk equation of state, Rosenfeld's
    coefficient at low packing), and ``P`` the third-term polynomial, equal
    to ``n2^3`` in a homogeneous fluid:

    - ``"q3"``: ``n2^3 (1 - xi^2)^3`` with ``xi = |n2v| / n2``, the
      antisymmetrized dimensional-crossover form of Rosenfeld, Schmidt,
      Löwen and Tarazona (1997); nonnegative wherever ``n2 >= 0`` and exact
      for a single zero-dimensional cavity.
    - ``"tensor"``: ``(9/2) [n2v.T.n2v - n2 n2v.n2v - Tr T^3 + n2 Tr T^2]``
      with ``T = n2t`` (Tarazona 2000); exact in the zero-dimensional limit
      including overlapping cavities and the usual choice for freezing, not
      globally bounded below.
    - ``"original"``: ``n2^3 - 3 n2 n2v.n2v``, Rosenfeld's scalar-vector
      form (1989); unbounded below for strongly localized density.
    """

    free = 1.0 - n3
    log = torch.log1p(-n3)
    n2v_squared = (n2v * n2v).sum(dim=-1)
    n1v_n2v = (n1v * n2v).sum(dim=-1)
    first = -n0 * log
    second = (n1 * n2 - n1v_n2v) / free
    if third_term == "original":
        polynomial = n2 ** 3 - 3.0 * n2 * n2v_squared
    elif third_term == "q3":
        tiny = torch.finfo(n2.dtype).tiny
        xi_squared = n2v_squared / (n2.square() + tiny)
        polynomial = n2 ** 3 * (1.0 - xi_squared).clamp_min(0.0) ** 3
    elif third_term == "tensor":
        if n2t is None:
            raise ValueError("third_term='tensor' requires the tensor weighted density n2t")
        vector_tensor_vector = torch.einsum("...i,...ij,...j->...", n2v, n2t, n2v)
        squared = n2t @ n2t
        trace_squared = torch.diagonal(squared, dim1=-2, dim2=-1).sum(dim=-1)
        trace_cubed = torch.diagonal(squared @ n2t, dim1=-2, dim2=-1).sum(dim=-1)
        polynomial = 4.5 * (
            vector_tensor_vector - n2 * n2v_squared - trace_cubed + n2 * trace_squared
        )
    else:
        raise ValueError("third_term must be one of " + ", ".join(FMT_THIRD_TERMS))
    if version == "rosenfeld":
        third = polynomial / (24.0 * math.pi * free.square())
    elif version in ("white_bear_i", "white_bear"):
        third = polynomial * _white_bear_bracket(n3) / (36.0 * math.pi * free.square())
    else:
        raise ValueError("version must be one of " + ", ".join(FMT_VERSIONS))
    return first + second + third


def _sphere_transforms(kR: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Return ``j1(x)/x``, ``j0(x)`` and ``j2(x)`` with their ``x -> 0`` limits.

    ``j1(x)/x = (sin x - x cos x)/x^3`` is the volume weight per ``4 pi R^3``,
    ``j0 = sin x / x`` the shell weight per ``4 pi R^2``, and ``j2`` enters
    the tensor weight; the small-argument series avoid the cancellations.
    """

    small = kR.abs() < 1.0e-3
    safe = torch.where(small, torch.ones_like(kR), kR)
    sin, cos = torch.sin(safe), torch.cos(safe)
    volume = torch.where(small, 1.0 / 3.0 - kR.square() / 30.0, (sin - safe * cos) / safe ** 3)
    surface = torch.where(small, 1.0 - kR.square() / 6.0, sin / safe)
    j2 = torch.where(small, kR.square() / 15.0, (3.0 / safe.square() - 1.0) * sin / safe - 3.0 * cos / safe.square())
    return volume, surface, j2


@register
class FMTReadout(EnergyReadout, Configurable):
    r"""Fundamental measure theory for hard-sphere mixtures.

    The excess free energy is a local function of weighted densities,

    ``beta F_HS = int Phi({n_alpha(r)}) dr``,
    ``n_alpha(r) = sum_i int rho_i(r') omega_alpha^(i)(r - r') dr'``,

    with the weights of a sphere of radius ``R_i``: ``omega_3 = Theta(R - r)``,
    ``omega_2 = delta(R - r)``, ``omega_1 = omega_2 / (4 pi R)``,
    ``omega_0 = omega_2 / (4 pi R^2)``, the vectors
    ``omega_2v = (r/r) delta(R - r) = -grad omega_3``,
    ``omega_1v = omega_2v / (4 pi R)``, and, for the tensor third term,
    ``omega_2t = (r r / r^2) delta(R - r)``. The weights are applied through
    their analytic transforms on the FFT grid,

    ``omega_3(k) = 4 pi (sin kR - kR cos kR) / k^3``,
    ``omega_2(k) = 4 pi R sin(kR) / k``,
    ``omega_2v(k) = -i k omega_3(k)``,
    ``omega_2t(k) = 4 pi R^2 [delta_ij j1(kR)/(kR) - k_i k_j j2(kR) / k^2]``,

    so they are exactly isotropic and normalized (the ``k -> 0`` values are
    the fundamental measures) and carry no stencil; the vector weights use
    the real spectral derivative that zeroes each even-axis Nyquist
    component, as the Coulomb source does. Grid rows are reordered through
    ``grid_positions`` when they are not canonical.

    ``version`` selects the free-volume factor ``phi_3`` of the third term:
    ``"rosenfeld"`` (1989) has the scaled-particle (Percus-Yevick
    compressibility) bulk free energy and the Percus-Yevick bulk direct
    correlation function; ``"white_bear_i"`` (default) is the factor of the
    White Bear functional of Roth, Evans, Lang and Kahl (2002; Yu and Wu
    2002), the mark I form rather than mark II, with the Carnahan-Starling
    bulk equation of state. Both are exact at low density (second virial
    coefficient) and satisfy the hard-wall contact theorem with their own
    pressure. ``version`` and ``third_term`` are independent choices, and
    only some of their combinations are published functionals:
    ``("rosenfeld", "original")`` is Rosenfeld's 1989 functional,
    ``("rosenfeld", "q3")`` the 1997 antisymmetrized one,
    ``("rosenfeld", "tensor")`` Tarazona's 2000 one, ``("white_bear_i",
    "original")`` White Bear mark I as published, and ``("white_bear_i",
    "tensor")`` the White Bear tensor functional used for freezing. The
    default ``("white_bear_i", "q3")`` is the White Bear I bulk factor with
    the q3 closure, chosen here for its bounded third term and its bulk
    equation of state; it is not itself a named functional.

    ``third_term`` selects the polynomial in the shell densities that
    multiplies that factor and equals ``n2^3`` in a homogeneous fluid, so
    that all three choices share the bulk free energy and the bulk second
    derivative (the Percus-Yevick direct correlation function for
    ``version="rosenfeld"``) and differ only away from homogeneity. The
    original scalar-vector form ``n2^3 - 3 n2 n2v.n2v`` (Rosenfeld 1989)
    has no lower bound: one particle confined to a fraction of its radius
    drives it to minus infinity, and it does not reproduce the
    zero-dimensional cavity limit. The default ``"q3"`` is the
    "antisymmetrized" third term ``n2^3 (1 - |n2v|^2 / n2^2)^3`` of
    Rosenfeld, Schmidt, Löwen and Tarazona (1997): the exponent 3 keeps
    the original term through order ``|n2v|^2`` (hence the same
    Percus-Yevick pair correlations), the factor vanishes where
    ``|n2v| = n2`` so that a single zero-dimensional cavity is exact, and
    the term is nonnegative wherever ``n2 >= 0``. It is an interpolation
    the authors presented as provisional, with crystal density peaks that
    come out too narrow; ``"tensor"`` is Tarazona's (2000) form built on
    the tensor weighted density, whose dimensional crossover extends to
    overlapping cavities and lost dimensions and which is the usual choice
    for freezing, at the cost of six more inverse transforms per
    evaluation and without a global lower bound (Lutsko 2020). Neither is
    a statement about the grid: what the discretization adds is described
    under :meth:`guarded_measures`.

    Resolution. The weights are sharp, so the weighted densities are
    exactly translation covariant for a density the grid resolves and the
    energy of a peak at least one spacing wide is the same to a percent
    whether it sits on a grid point or between grid points, on the ``0.25``
    and the ``0.125`` grids alike and for every third term. A peak narrower
    than the spacing is not represented (its samples change with its
    position, and so does the ideal-gas term by ``N ln 8`` between the two
    alignments); its FMT energy then depends on the alignment as well, and
    the sharp weights ring around it. The continuum bounds
    ``n3_i >= 0``, ``n2_i >= 0``, ``|n2v_i| <= n2_i`` are enforced on the
    per-species weighted densities exactly where that ringing violates
    them, which keeps ``Phi >= 0`` pointwise for ``"q3"`` on any grid, so an
    unresolved peak costs energy at either alignment and cannot open a
    direction of descent for a minimizer. Reference densities should be
    smoothed to at least the grid spacing, as the Gaussian-binned data
    are.

    The functional is entropic: in ``free_energy_mode="beta"`` the readout
    returns ``beta F_HS``; in ``"physical"`` mode it returns
    ``F_HS / (k_B T_ref) = beta F_HS T / T_ref`` so that the model's
    conversion recovers the same ``beta F_HS`` at every temperature.

    The readout is charge-blind: it owns packing only. Next to an
    :class:`MSAReferenceReadout` the latter must be built with
    ``include_hard_sphere=False``, which :meth:`validate_readout_graph`
    enforces. Excluded (wall) voxels need no special treatment: the density
    is zero there, ``Phi`` is integrated over the whole grid, and the
    weighted densities reaching into the wall from the fluid are how the
    functional feels it.

    Parameters
    ----------
    n_types
        Number of density components.
    radii
        Hard-sphere radius of every component in coordinate units (``sigma /
        2`` for the restricted primitive model).
    version
        ``"white_bear_i"`` (default; ``"white_bear"`` is accepted as the same)
        or ``"rosenfeld"``.
    third_term
        ``"q3"`` (default), ``"tensor"`` or ``"original"``.
    maximum_packing
        Local packing fraction ``n3`` above which the free-volume terms are
        regularized: with ``packing_guard="clamp"`` (default) ``n3`` is
        replaced by the smooth ``n3 - s softplus((n3 - n_max) / s)`` with
        ``s = 0.01``, which equals ``n3`` exactly (in single precision too)
        more than a few ``s`` below ``n_max`` and saturates at ``n_max``
        above it, so that a noisy
        reference density or an early solver iterate cannot reach the
        singularity; ``"raise"`` rejects such a field instead. The lower
        bounds of the weighted densities (see :meth:`guarded_measures`)
        are enforced in both modes.
    """

    def __init__(
        self,
        n_types: int,
        *,
        radii: Sequence[float],
        version: str = "white_bear_i",
        third_term: str = "q3",
        maximum_packing: float = 0.95,
        packing_guard: str = "clamp",
    ) -> None:
        super().__init__()

        self.n_types = positive_integer(n_types, "n_types")
        radii = tuple(positive_scalar(value, "radii") for value in radii)
        if len(radii) != self.n_types:
            raise ValueError("radii must have n_types entries")
        self.radii = radii
        version = nonempty_string(version, "version").lower()
        if version == "white_bear":
            version = "white_bear_i"
        if version not in FMT_VERSIONS:
            raise ValueError("version must be one of " + ", ".join(FMT_VERSIONS))
        self.version = version
        third_term = nonempty_string(third_term, "third_term").lower()
        if third_term not in FMT_THIRD_TERMS:
            raise ValueError("third_term must be one of " + ", ".join(FMT_THIRD_TERMS))
        self.third_term = third_term
        self.maximum_packing = positive_scalar(maximum_packing, "maximum_packing")
        if self.maximum_packing >= 1.0:
            raise ValueError("maximum_packing must be below one")
        packing_guard = nonempty_string(packing_guard, "packing_guard").lower()
        if packing_guard not in ("clamp", "raise"):
            raise ValueError("packing_guard must be 'clamp' or 'raise'")
        self.packing_guard = packing_guard
        self._spectra: Dict[Tuple, Tuple[torch.Tensor, ...]] = {}

    def to_config(self) -> Dict[str, Any]:
        """Return the constructor arguments describing this readout."""

        return make_config(
            self,
            n_types=self.n_types,
            radii=self.radii,
            version=self.version,
            third_term=self.third_term,
            maximum_packing=self.maximum_packing,
            packing_guard=self.packing_guard,
        )

    def validate_readout_graph(self, readouts: Sequence[EnergyReadout]) -> None:
        """Reject a second hard-sphere reference in the same model."""

        from .msa import MSAReferenceReadout

        for item in readouts:
            if item is self:
                continue
            if isinstance(item, FMTReadout):
                raise ValueError("a model may contain one FMTReadout")
            if isinstance(item, MSAReferenceReadout) and item.include_hard_sphere:
                raise ValueError(
                    "FMTReadout beside MSAReferenceReadout requires "
                    "include_hard_sphere=False, or the Percus-Yevick hard-sphere "
                    "kernel is counted twice"
                )

    # ------------------------------------------------------------------ spectra
    def weight_spectra(
        self,
        grid_size: Tuple[int, int, int],
        grid_spacing: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, ...]:
        r"""Return cached weight spectra on the real-FFT grid.

        ``(omega_3, omega_2, k_D, omega_2t_iso, omega_2t_dir, k_hat)``: the
        scalar spectra ``[n_types, nx, ny, nz//2 + 1]``, the derivative
        wavevector ``[3, ...]`` with each even-axis Nyquist component
        omitted, the two radial factors of the tensor weight
        ``4 pi R^2 j1(kR)/(kR)`` and ``4 pi R^2 j2(kR)`` per species, and the
        unit wavevector ``[3, ...]`` (zero at ``k = 0``).
        """

        spacing = tuple(float(value) for value in grid_spacing.detach().cpu().reshape(-1).tolist())
        key = (tuple(grid_size), spacing, str(device), str(dtype))
        cached = self._spectra.get(key)
        if cached is None:
            spacing_tensor = torch.as_tensor(spacing, dtype=torch.float64)
            axes = _wavevector_axes(grid_size, spacing_tensor, "cpu", torch.float64)
            axes[-1] = axes[-1][: grid_size[-1] // 2 + 1]
            wavevector = torch.stack(torch.meshgrid(*axes, indexing="ij"))
            magnitude = torch.sqrt(wavevector.square().sum(dim=0))
            unit = wavevector / torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
            derivative_axes = _wavevector_axes(
                grid_size, spacing_tensor, "cpu", torch.float64, derivative=True,
            )
            derivative_axes[-1] = derivative_axes[-1][: grid_size[-1] // 2 + 1]
            derivative = torch.stack(torch.meshgrid(*derivative_axes, indexing="ij"))
            volume, surface, tensor_iso, tensor_dir = [], [], [], []
            for radius in self.radii:
                v, s, j2 = _sphere_transforms(magnitude * radius)
                volume.append(4.0 * math.pi * radius ** 3 * v)
                surface.append(4.0 * math.pi * radius ** 2 * s)
                tensor_iso.append(4.0 * math.pi * radius ** 2 * v)
                tensor_dir.append(4.0 * math.pi * radius ** 2 * j2)
            cached = tuple(
                item.to(device=device, dtype=dtype) for item in (
                    torch.stack(volume), torch.stack(surface), derivative,
                    torch.stack(tensor_iso), torch.stack(tensor_dir), unit,
                )
            )
            self._spectra[key] = cached
        return cached

    # ------------------------------------------------------- weighted densities
    def weighted_densities(
        self,
        rho_grid: torch.Tensor,
        grid_size: Tuple[int, int, int],
        grid_spacing: torch.Tensor,
        tensor: Optional[bool] = None,
        guard: bool = True,
    ) -> Dict[str, torch.Tensor]:
        r"""Return ``n0, n1, n2, n3`` ``[n_fields, nx, ny, nz]``, ``n1v, n2v`` ``[..., 3]`` and, on request, ``n2t`` ``[..., 3, 3]``.

        ``rho_grid`` is the canonical-order density ``[n_fields, n_types, nx, ny, nz]``.
        ``tensor`` defaults to whether the configured third term needs it.
        With ``guard`` the per-species weighted densities are held to the
        bounds every resolved nonnegative density satisfies (see
        :meth:`guarded_measures`) before they are combined.
        """

        if tensor is None:
            tensor = self.third_term == "tensor"
        volume_hat, surface_hat, derivative, tensor_iso, tensor_dir, unit = self.weight_spectra(
            grid_size, grid_spacing, rho_grid.device, rho_grid.dtype,
        )
        n_types = rho_grid.shape[1]
        rho_hat = torch.fft.rfftn(rho_grid, dim=(-3, -2, -1))
        volume_terms = volume_hat[None] * rho_hat  # per-species n3
        surface_terms = surface_hat[None] * rho_hat  # per-species n2
        vector_terms = -1j * derivative[None, None] * volume_terms[:, :, None]  # per-species n2v
        spectra = [volume_terms, surface_terms, *vector_terms.unbind(dim=2)]
        if tensor:
            # omega_2t(k) = iso(k) delta_ij - dir(k) k_i k_j / k^2 per species;
            # the six upper components, whose trace is omega_2(k) exactly.
            iso_hat = tensor_iso[None] * rho_hat
            dir_hat = tensor_dir[None] * rho_hat
            for i in range(3):
                for j in range(i, 3):
                    component = -dir_hat * (unit[i] * unit[j])[None, None]
                    if i == j:
                        component = component + iso_hat
                    spectra.append(component)
        fields = torch.fft.irfftn(torch.cat(spectra, dim=1), s=grid_size, dim=(-3, -2, -1))
        n3_species = fields[:, :n_types]
        n2_species = fields[:, n_types : 2 * n_types]
        n2v_species = fields[:, 2 * n_types : 5 * n_types].reshape(
            fields.shape[0], 3, n_types, *grid_size,
        ).movedim(1, -1)
        n2t_species = None
        if tensor:
            xx, xy, xz, yy, yz, zz = fields[:, 5 * n_types : 11 * n_types].reshape(
                fields.shape[0], 6, n_types, *grid_size,
            ).unbind(dim=1)
            n2t_species = torch.stack((
                torch.stack((xx, xy, xz), dim=-1),
                torch.stack((xy, yy, yz), dim=-1),
                torch.stack((xz, yz, zz), dim=-1),
            ), dim=-2)
        if guard:
            n3_species, n2_species, n2v_species, n2t_species = self.guarded_measures(
                n3_species, n2_species, n2v_species, n2t_species,
            )
        radii = torch.as_tensor(self.radii, dtype=rho_grid.dtype, device=rho_grid.device)
        inverse_perimeter = (1.0 / (4.0 * math.pi * radii)).reshape(1, -1, 1, 1, 1)
        inverse_area = inverse_perimeter / radii.reshape(1, -1, 1, 1, 1)
        result = {
            "n0": (inverse_area * n2_species).sum(dim=1),
            "n1": (inverse_perimeter * n2_species).sum(dim=1),
            "n2": n2_species.sum(dim=1),
            "n3": n3_species.sum(dim=1),
            "n1v": (inverse_perimeter[..., None] * n2v_species).sum(dim=1),
            "n2v": n2v_species.sum(dim=1),
        }
        if tensor:
            result["n2t"] = n2t_species.sum(dim=1)
        return result

    def guarded_measures(
        self,
        n3_species: torch.Tensor,
        n2_species: torch.Tensor,
        n2v_species: torch.Tensor,
        n2t_species: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        r"""Hold the per-species weighted densities to their continuum bounds.

        For a nonnegative density resolved by the grid, ``n3_i >= 0``,
        ``n2_i >= 0`` and ``|n2v_i| <= n2_i`` hold pointwise, because the
        volume and shell weights are nonnegative and ``|omega_2v| = omega_2``.
        A density with structure below the grid spacing violates them
        through the ringing of the sharp weights (a one-voxel peak gives a
        negative shell density at its own site), and it is those violations
        that let ``Phi`` change sign. The bounds are enforced exactly where
        they are violated and nothing is changed elsewhere: ``n3_i`` is
        clamped at zero; where the shell density ``n2_i`` is negative the
        whole shell family of that species (``n2_i``, ``n2v_i`` and the
        tensor ``n2t_i``) is set to zero, as every moment of a shell of no
        mass is, which keeps ``Tr n2t_i = n2_i`` intact; elsewhere
        ``n2v_i`` is shortened to length ``n2_i`` and the tensor is left
        alone. Together with the packing bound this keeps ``Phi >= 0``
        pointwise for the ``"q3"`` third term on any grid. A hard wall also
        rings at the few-percent level (Gibbs overshoot of the sharp weights
        at a density step), so the bounds are enforced independently of
        ``packing_guard``: rejecting such fields would reject every slab.
        """

        n3_species = n3_species.clamp_min(0.0)
        no_shell = n2_species < 0.0
        n2_species = n2_species.clamp_min(0.0)
        # A uniform fluid has n2v = 0. Its norm is not twice differentiable
        # there, even when the overlong branch is inactive; c2 differentiates
        # through this guard twice. Test squared lengths first and take the
        # square root only after substituting a positive inactive value.
        norm_squared = n2v_species.square().sum(dim=-1)
        overlong = norm_squared > n2_species.square()
        safe_norm = torch.sqrt(torch.where(
            overlong, norm_squared, torch.ones_like(norm_squared),
        ))
        scale = torch.where(
            overlong, n2_species / safe_norm, torch.ones_like(safe_norm),
        )
        n2v_species = n2v_species * scale[..., None]
        if n2t_species is not None:
            n2t_species = torch.where(no_shell[..., None, None], torch.zeros_like(n2t_species), n2t_species)
        return n3_species, n2_species, n2v_species, n2t_species

    def _guarded_packing(self, n3: torch.Tensor) -> torch.Tensor:
        if self.packing_guard == "raise":
            if torch.any(n3 >= self.maximum_packing).item():
                raise ValueError(
                    "a local packing fraction reached maximum_packing; the field "
                    "is outside the hard-sphere functional's domain"
                )
            return n3
        # n3 - s softplus((n3 - n_max) / s) equals n_max - s softplus((n_max - n3) / s)
        # but returns n3 itself, without cancellation, wherever n3 is well below n_max.
        scale = 0.01
        return n3 - scale * torch.nn.functional.softplus((n3 - self.maximum_packing) / scale)

    # ------------------------------------------------------------------- energy
    def energy(self, context: ReadoutContext) -> torch.Tensor:
        """Return the hard-sphere excess free energy of every field."""

        rho = context.rho
        if (
            not rho.is_floating_point()
            or rho.ndim < 2
            or rho.shape[-1] != self.n_types
            or not torch.all(torch.isfinite(rho)).item()
            or torch.any(rho < 0.0).item()
        ):
            raise ValueError(
                "rho must be finite and nonnegative [..., n_grid, n_types] with the "
                "configured n_types"
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
        rho_flat = rho.reshape(n_fields, n_grid, self.n_types)
        if rows_are_canonical:
            canonical = rho_flat
        else:
            canonical = torch.zeros_like(rho_flat).scatter(
                1, flat_positions[..., None].expand_as(rho_flat), rho_flat,
            )
        rho_grid = canonical.reshape(n_fields, *grid_size, self.n_types).movedim(-1, 1)
        weighted = self.weighted_densities(rho_grid, grid_size, spacing)
        phi = fmt_free_energy_density(
            weighted["n0"], weighted["n1"], weighted["n2"],
            self._guarded_packing(weighted["n3"]),
            weighted["n1v"], weighted["n2v"], self.version, self.third_term,
            weighted.get("n2t"),
        )
        volume_element = context.voxel_volume.to(rho)
        energy = volume_element * phi.sum(dim=(-3, -2, -1))
        if context.free_energy_mode == "physical":
            # Entropic term: F / (k_B T_ref) = beta F * T / T_ref.
            energy = energy * context.normalized_temperature.reshape(-1)
        return energy.reshape(leading)
