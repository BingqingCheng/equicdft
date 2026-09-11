"""Scalar contractions of particle-density and electric-dipole moments."""

from itertools import product
from typing import Dict, Mapping, Sequence, Union

import torch
from torch import nn

from ._argument_checks import boolean, nonnegative_integer, positive_integer
from ._grid import gather_neighbors
from ._nn import positive_scalar_tensor
from .features import CartesianAFeatures


# Moment notation used in the contraction list:
#   s, v, Q = density moments of spatial power 0, 1, 2;
#   p, D, H = dipole moments of spatial power 0, 1, 2;
#   u_k = H_iik and w_i = H_ijj are the two distinct traces of H.
# s and p also include the unsmeared center density and center dipole.
# Each row gives: name, einsum, operands, minimum power, dipole-factor count.
# The retained channels of each operand are independently combined. This is
# an explicit, finite invariant set, not a complete or minimal invariant basis.
# Cartesian indices i,j,k are contracted; a,b,c label radial/species channels.
_CONTRACTIONS = (
    ("density", "...a->...a", ("s",), 0, 0),
    ("trace_Q", "...aii->...a", ("Q",), 2, 0),
    ("trace_D", "...aii->...a", ("D",), 1, 1),
    ("v_dot_v", "...ai,...bi->...ab", ("v", "v"), 1, 0),
    ("p_dot_p", "...ai,...bi->...ab", ("p", "p"), 0, 2),
    ("v_dot_p", "...ai,...bi->...ab", ("v", "p"), 1, 1),
    ("Q_colon_Q", "...aij,...bij->...ab", ("Q", "Q"), 2, 0),
    ("Q_colon_D", "...aij,...bij->...ab", ("Q", "D"), 2, 1),
    ("D_colon_D", "...aij,...bij->...ab", ("D", "D"), 1, 2),
    ("trace_D_D", "...aij,...bji->...ab", ("D", "D"), 1, 2),
    ("H_colon_H", "...aijk,...bijk->...ab", ("H", "H"), 2, 2),
    ("u_dot_u", "...ai,...bi->...ab", ("u", "u"), 2, 2),
    ("w_dot_w", "...ai,...bi->...ab", ("w", "w"), 2, 2),
    ("u_dot_w", "...ai,...bi->...ab", ("u", "w"), 2, 2),
    ("v_dot_u", "...ai,...bi->...ab", ("v", "u"), 2, 1),
    ("v_dot_w", "...ai,...bi->...ab", ("v", "w"), 2, 1),
    ("p_dot_u", "...ai,...bi->...ab", ("p", "u"), 2, 2),
    ("p_dot_w", "...ai,...bi->...ab", ("p", "w"), 2, 2),
    ("v_Q_v", "...ai,...bij,...cj->...abc", ("v", "Q", "v"), 2, 0),
    ("p_Q_p", "...ai,...bij,...cj->...abc", ("p", "Q", "p"), 2, 2),
    ("v_Q_p", "...ai,...bij,...cj->...abc", ("v", "Q", "p"), 2, 1),
    ("v_D_v", "...ai,...bij,...cj->...abc", ("v", "D", "v"), 1, 1),
    ("p_D_p", "...ai,...bij,...cj->...abc", ("p", "D", "p"), 1, 3),
    ("v_D_p", "...ai,...bij,...cj->...abc", ("v", "D", "p"), 1, 2),
    ("p_D_v", "...ai,...bij,...cj->...abc", ("p", "D", "v"), 1, 2),
    ("Q_H_p", "...aij,...bijk,...ck->...abc", ("Q", "H", "p"), 2, 2),
    ("Q_H_v", "...aij,...bijk,...ck->...abc", ("Q", "H", "v"), 2, 1),
    ("D_H_p", "...aij,...bijk,...ck->...abc", ("D", "H", "p"), 2, 3),
    ("D_H_v", "...aij,...bijk,...ck->...abc", ("D", "H", "v"), 2, 2),
)


class PolarizationFeatures(nn.Module):
    """Invariant local features of scalar density and a polar vector field.

    The calculation has three steps: gather neighboring fields, form their
    Cartesian moments, and contract all Cartesian indices to obtain scalars.

    Inputs are ``rho[..., G, t]``, ``dipole_density[..., G, t, 3]`` and the
    canonical ``local_density_index[..., G, J]``. Dipole density means dipole
    moment per volume, not mean orientation per particle. Each entire vector
    is divided by the same positive ``dipole_density_scale``; its components
    are never independently normalized or treated as scalar species.

    At integer offset q, the shared radial weight is
    ``w_n(q) = exp(-alpha_n |q|^2) / sum_q exp(-alpha_n |q|^2)``.
    Moments sum ``w_n * q**tensor_power * rho/mean_density`` or
    ``w_n * q**tensor_power * P/dipole_density_scale`` over the inclusive
    stencil. The latter retains an independent, final dipole-vector index.
    No voxel-volume factor enters these normalized local averages.

    Spatial powers 0--2 and moment product orders 1--3 are supported. All
    channels (radial index first, species second) can interact. Unsmeared
    center rho and P are appended to the scalar and vector channels, so
    contractions retain the center's orientation relative to its neighbors.
    ``feature_names`` records every flattened contraction in output order.

    Delta contractions are O(3)-invariant algebraically. On a fixed cubic
    voxel lattice, exact field covariance is limited to lattice-preserving
    signed axis permutations; arbitrary rotations require resampling.
    ``dipole_reversal_symmetry=True`` additionally removes odd-P terms.
    This is a separate physical assumption, not implied by spatial inversion.
    """

    def __init__(
        self,
        mean_density: Union[float, torch.Tensor],
        dipole_density_scale: Union[float, torch.Tensor],
        cutoff_grid: int = 3,
        max_power: int = 2,
        max_product_order: int = 3,
        radial_exponents: Sequence[float] = (0.125,),
        trainable_radial_exponents: bool = True,
        n_types: int = 1,
        dipole_reversal_symmetry: bool = False,
    ) -> None:
        super().__init__()
        self.max_power = nonnegative_integer(max_power, "max_power")
        self.max_product_order = positive_integer(
            max_product_order, "max_product_order"
        )
        if self.max_power > 2 or self.max_product_order > 3:
            raise ValueError("supported max_power <= 2 and max_product_order <= 3")
        self.dipole_reversal_symmetry = boolean(
            dipole_reversal_symmetry, "dipole_reversal_symmetry"
        )
        # Reuse the scalar implementation's stencil, Gaussian parameters and
        # monomials. Only moment accumulation is shared, not symmetrization.
        self.scalar_features = CartesianAFeatures(
            mean_density=mean_density,
            cutoff_grid=cutoff_grid,
            max_power=self.max_power,
            radial_basis="gaussian",
            radial_exponents=radial_exponents,
            trainable_radial_exponents=trainable_radial_exponents,
            separate_center=False,
            n_types=n_types,
        )
        self.register_buffer(
            "dipole_density_scale",
            positive_scalar_tensor(dipole_density_scale, "dipole_density_scale"),
        )

        # Expanded tensor indices restore repeated off-diagonal entries of
        # q tensor q, while leaving the independent dipole index untouched.
        powers = self.scalar_features.powers.tolist()
        if self.max_power >= 1:
            self.register_buffer("vector_indices", torch.tensor([1, 2, 3]))
        if self.max_power >= 2:
            indices = []
            for i, j in product(range(3), repeat=2):
                power = [0, 0, 0]
                power[i] += 1
                power[j] += 1
                indices.append(powers.index(power))
            self.register_buffer("matrix_indices", torch.tensor(indices))

        channels = self.scalar_features.radial_exponents.numel() * self.n_types
        sizes = {key: channels for key in ("v", "Q", "D", "H", "u", "w")}
        sizes.update(s=channels + self.n_types, p=channels + self.n_types)
        contractions, feature_names = [], []
        for rule in _CONTRACTIONS:
            name, equation, operands, minimum_power, dipole_factors = rule
            if minimum_power > self.max_power:
                continue
            if len(operands) > self.max_product_order:
                continue
            if self.dipole_reversal_symmetry and dipole_factors % 2:
                continue
            contractions.append(rule)
            channel_ranges = [range(sizes[key]) for key in operands]
            for indices in product(*channel_ranges):
                feature_names.append(
                    "{}[{}]".format(name, ",".join(map(str, indices)))
                )
        self.contractions = tuple(contractions)
        self.feature_names = tuple(feature_names)
        self.n_features = len(self.feature_names)

    @property
    def mean_density(self) -> torch.Tensor:
        return self.scalar_features.mean_density

    @property
    def cutoff_grid(self) -> int:
        return self.scalar_features.cutoff_grid

    @property
    def n_types(self) -> int:
        return self.scalar_features.n_types

    @property
    def radial_exponents(self) -> torch.Tensor:
        """The shared, optionally trainable positive Gaussian exponents."""
        return self.scalar_features.radial_exponents

    def _moments(self, data: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Form scalar and vector moments using the same Gaussian basis."""

        rho = data["rho"]
        dipole = data["dipole_density"]
        if rho.ndim < 2 or rho.shape[-1] != self.n_types:
            raise ValueError("rho must have shape [..., n_grid, n_types]")
        if dipole.shape != (*rho.shape, 3):
            raise ValueError("dipole_density must have shape [..., n_grid, n_types, 3]")
        if dipole.dtype != rho.dtype or dipole.device != rho.device:
            raise ValueError("rho and dipole_density must share dtype and device")
        if not torch.isfinite(dipole).all():
            raise ValueError("dipole_density must be finite")
        # 1. Normalize and gather both fields without detaching their graphs.
        rho = rho / self.mean_density
        dipole = dipole / self.dipole_density_scale
        neighbors = data["local_density_index"]
        local_rho = gather_neighbors(rho, neighbors)
        local_dipole = gather_neighbors(dipole, neighbors)

        # 2. Sum against one shared radial/monomial basis [neighbor, radial,
        # monomial]. Put radial and species next to each other, so flattening
        # those two axes gives a single channel index C in a fixed order.
        basis = self.scalar_features.stencil_basis().to(rho)
        density_moments = torch.einsum("...gjt,jnk->...gntk", local_rho, basis)
        dipole_moments = torch.einsum("...gjti,jnk->...gntki", local_dipole, basis)
        density_moments = density_moments.flatten(-3, -2)  # [..., G, C, K]
        dipole_moments = dipole_moments.flatten(-4, -3)  # [..., G, C, K, 3]

        # Expand the monomials into full Cartesian tensors. In a dipole
        # moment only the spatial indices are symmetric, not the last index.
        moments = {
            "s": torch.cat((density_moments[..., 0], rho), dim=-1),
            "p": torch.cat((dipole_moments[..., 0, :], dipole), dim=-2),
        }
        if self.max_power >= 1:
            moments["v"] = density_moments.index_select(-1, self.vector_indices)
            moments["D"] = dipole_moments.index_select(-2, self.vector_indices)
        if self.max_power >= 2:
            moments["Q"] = density_moments.index_select(
                -1, self.matrix_indices
            ).unflatten(-1, (3, 3))
            moments["H"] = dipole_moments.index_select(
                -2, self.matrix_indices
            ).unflatten(-2, (3, 3))
            # H_ijk has symmetric spatial indices i,j, but its dipole index
            # k is independent. Its two inequivalent traces are both kept.
            moments["u"] = torch.einsum("...aiik->...ak", moments["H"])
            moments["w"] = torch.einsum("...aijj->...ai", moments["H"])
        return moments

    def _contract(self, moments: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """3. Contract each retained tensor product and flatten its channels."""

        leading_shape = moments["s"].shape[:-1]  # includes central-grid axis
        values = []
        for _, equation, operands, _, _ in self.contractions:
            tensors = [moments[key] for key in operands]
            invariant = torch.einsum(equation, *tensors)
            values.append(invariant.reshape(*leading_shape, -1))
        return torch.cat(values, dim=-1)

    def forward(self, data: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return invariant features with shape ``[..., n_grid, n_features]``."""
        moments = self._moments(data)
        return self._contract(moments)
