"""Symmetrized Cartesian features for the simple cubic grid.

The exact point symmetry used here consists of all axis permutations and
independent axis reflections. These 48 signed permutations preserve the
integer spherical stencil constructed by :func:`cace_grid.stencil.make_stencil`.
"""

from itertools import combinations_with_replacement, permutations, product
from numbers import Integral
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from .features import _make_powers


def _make_group_actions(
    powers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the signed-permutation action on Cartesian monomials.

    For each of the 48 group elements and each monomial component ``k``, the
    returned tensors specify

    ``transformed_A[..., k, :] = sign[g, k] * A[..., index[g, k], :]``.
    """

    powers_list = [tuple(int(value) for value in row) for row in powers.tolist()]
    index_by_power = {power: index for index, power in enumerate(powers_list)}

    component_indices: List[List[int]] = []
    component_signs: List[List[int]] = []

    # output_axis i receives sign[i] * input_axis permutation[i].
    for permutation in permutations(range(3)):
        for reflections in product((1, -1), repeat=3):
            indices_now = []
            signs_now = []
            for power in powers_list:
                mapped_power = [0, 0, 0]
                sign = 1
                for output_axis, input_axis in enumerate(permutation):
                    mapped_power[input_axis] = power[output_axis]
                    sign *= reflections[output_axis] ** power[output_axis]
                indices_now.append(index_by_power[tuple(mapped_power)])
                signs_now.append(sign)
            component_indices.append(indices_now)
            component_signs.append(signs_now)

    return (
        torch.tensor(component_indices, dtype=torch.long),
        torch.tensor(component_signs, dtype=torch.get_default_dtype()),
    )


def _signed_orbit(
    indices: Tuple[int, ...],
    component_indices: Sequence[Sequence[int]],
    component_signs: Sequence[Sequence[int]],
) -> Tuple[Dict[Tuple[int, ...], int], bool]:
    """Generate one signed product orbit and report whether it vanishes."""

    terms: Dict[Tuple[int, ...], int] = {}
    vanishes = False
    for index_map, sign_map in zip(component_indices, component_signs):
        mapped_indices = tuple(sorted(index_map[index] for index in indices))
        mapped_sign = 1
        for index in indices:
            mapped_sign *= sign_map[index]

        previous_sign = terms.get(mapped_indices)
        if previous_sign is not None and previous_sign != mapped_sign:
            # A negative stabilizer pairs every term with its negative, so the
            # Reynolds average over this complete orbit is identically zero.
            vanishes = True
        terms[mapped_indices] = mapped_sign

    return terms, vanishes


def _make_product_recipes(
    n_components: int,
    max_nu: int,
    component_indices: torch.Tensor,
    component_signs: torch.Tensor,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Build sparse signed orbit sums for correlation orders up to three."""

    index_maps = component_indices.tolist()
    sign_maps = component_signs.to(dtype=torch.long).tolist()
    recipes: Dict[int, Dict[str, torch.Tensor]] = {}

    for nu in range(1, max_nu + 1):
        unseen = set(combinations_with_replacement(range(n_components), nu))
        product_indices: List[Tuple[int, ...]] = []
        coefficients: List[float] = []
        output_indices: List[int] = []
        representatives: List[Tuple[int, ...]] = []

        while unseen:
            representative = min(unseen)
            orbit, vanishes = _signed_orbit(
                representative,
                index_maps,
                sign_maps,
            )
            unseen.difference_update(orbit)
            if vanishes:
                continue

            output_index = len(representatives)
            representatives.append(representative)
            normalization = float(len(orbit))
            for term, sign in sorted(orbit.items()):
                product_indices.append(term)
                coefficients.append(sign / normalization)
                output_indices.append(output_index)

        recipes[nu] = {
            "product_indices": torch.tensor(product_indices, dtype=torch.long),
            "coefficients": torch.tensor(
                coefficients,
                dtype=torch.get_default_dtype(),
            ),
            "output_indices": torch.tensor(output_indices, dtype=torch.long),
            "representatives": torch.tensor(representatives, dtype=torch.long),
        }

    return recipes


class CartesianBFeatures(nn.Module):
    """Symmetrize Cartesian ``A`` features under cubic-grid point symmetry.

    Parameters
    ----------
    max_power
        Maximum total Cartesian power used by the corresponding
        :class:`cace_grid.features.CartesianAFeatures` module.
    max_nu
        Maximum correlation order, meaning the largest number of ``A``
        factors in one invariant product. Only values one through three are
        currently supported.

    Notes
    -----
    Symmetrization acts only on the Cartesian-monomial dimension. Gaussian
    radial channels and density/type channels remain separate, so input and
    output shapes are

    ``A: [..., n_grid, n_alphas, n_monomials, n_types]``

    ``B: [..., n_grid, n_alphas, n_B, n_types]``.
    """

    def __init__(self, max_power: int, max_nu: int = 3) -> None:
        super().__init__()

        if isinstance(max_nu, bool) or not isinstance(max_nu, Integral):
            raise TypeError("max_nu must be an integer between one and three")
        max_nu = int(max_nu)
        if max_nu < 1 or max_nu > 3:
            raise ValueError("max_nu must be between one and three")

        powers = _make_powers(max_power)
        component_indices, component_signs = _make_group_actions(powers)
        recipes = _make_product_recipes(
            n_components=powers.shape[0],
            max_nu=max_nu,
            component_indices=component_indices,
            component_signs=component_signs,
        )

        self.max_power = int(max_power)
        self.max_nu = max_nu
        self.n_components = powers.shape[0]
        self.register_buffer("powers", powers)
        self.register_buffer("component_indices", component_indices)
        self.register_buffer("component_signs", component_signs)

        n_features_by_order = []
        correlation_orders = []
        for nu in range(1, max_nu + 1):
            recipe = recipes[nu]
            self.register_buffer(
                "product_indices_{}".format(nu),
                recipe["product_indices"],
            )
            self.register_buffer(
                "coefficients_{}".format(nu),
                recipe["coefficients"],
            )
            self.register_buffer(
                "output_indices_{}".format(nu),
                recipe["output_indices"],
            )
            self.register_buffer(
                "representatives_{}".format(nu),
                recipe["representatives"],
            )

            n_features = recipe["representatives"].shape[0]
            n_features_by_order.append(n_features)
            correlation_orders.extend([nu] * n_features)

        self.n_features_by_order = tuple(n_features_by_order)
        self.n_features = sum(n_features_by_order)
        self.register_buffer(
            "correlation_orders",
            torch.tensor(correlation_orders, dtype=torch.long),
        )

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """Return signed-orbit averages of ``A`` through ``max_nu``."""

        if A.ndim < 4:
            raise ValueError(
                "A must have shape "
                "[..., n_grid, n_alphas, n_monomials, n_types]"
            )
        if A.shape[-2] != self.n_components:
            raise ValueError(
                "A monomial count does not match max_power={}".format(
                    self.max_power
                )
            )

        features_by_order = []
        for nu, n_features in enumerate(self.n_features_by_order, start=1):
            product_indices = getattr(self, "product_indices_{}".format(nu))
            coefficients = getattr(self, "coefficients_{}".format(nu))
            output_indices = getattr(self, "output_indices_{}".format(nu))

            # Explicit multiplication keeps higher derivatives reliable on
            # MPS. The term axis replaces the monomial axis.
            products = A.index_select(-2, product_indices[:, 0])
            for factor in range(1, nu):
                products = products * A.index_select(
                    -2,
                    product_indices[:, factor],
                )

            coefficient_shape = [1] * products.ndim
            coefficient_shape[-2] = coefficients.shape[0]
            weighted_products = products * coefficients.to(dtype=A.dtype).view(
                coefficient_shape
            )

            output_shape = list(A.shape)
            output_shape[-2] = n_features
            features_now = torch.zeros(
                output_shape,
                dtype=A.dtype,
                device=A.device,
            ).index_add(-2, output_indices, weighted_products)
            features_by_order.append(features_now)

        return torch.cat(features_by_order, dim=-2)
