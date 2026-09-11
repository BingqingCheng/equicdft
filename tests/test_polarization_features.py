import itertools
import unittest
from unittest import mock

import numpy as np
import torch

from equicdft import PolarizationFeatures
from equicdft.stencil import get_neighbor_indices


def periodic_field(size=5, cutoff=2, n_types=1):
    positions = np.indices((size, size, size)).reshape(3, -1).T
    neighbors, _ = get_neighbor_indices(positions, cutoff_grid=cutoff)
    generator = torch.Generator().manual_seed(38)
    return {
        "rho": 0.2 + torch.rand(len(positions), n_types, generator=generator, dtype=torch.float64),
        "dipole_density": torch.randn(len(positions), n_types, 3, generator=generator, dtype=torch.float64),
        "local_density_index": torch.tensor(neighbors),
    }, torch.tensor(positions)


class TestPolarizationFeatures(unittest.TestCase):
    def features(self, **kwargs):
        arguments = dict(
            mean_density=0.7, dipole_density_scale=2.0,
            cutoff_grid=2, radial_exponents=(0.125, 0.5),
        )
        arguments.update(kwargs)
        return PolarizationFeatures(**arguments).double()

    def test_all_48_signed_axis_permutations(self):
        data, positions = periodic_field(n_types=2)
        features = self.features(n_types=2, radial_exponents=(0.25,))
        reference = features(data)
        self.assertEqual(reference.shape, (125, features.n_features))
        self.assertEqual(len(set(features.feature_names)), features.n_features)
        for permutation in itertools.permutations(range(3)):
            for signs in itertools.product((-1, 1), repeat=3):
                rotation = torch.eye(3, dtype=torch.float64)[list(permutation)]
                rotation = rotation * torch.tensor(signs)[:, None]
                transformed_positions = (positions @ rotation.long().T) % 5
                index = (transformed_positions[:, 0] * 5 + transformed_positions[:, 1]) * 5 + transformed_positions[:, 2]
                rho = torch.empty_like(data["rho"])
                dipole = torch.empty_like(data["dipole_density"])
                rho[index] = data["rho"]
                dipole[index] = data["dipole_density"] @ rotation.T
                transformed = dict(data, rho=rho, dipole_density=dipole)
                torch.testing.assert_close(features(transformed)[index], reference, atol=2e-12, rtol=2e-12)

    def test_contractions_under_arbitrary_orthogonal_rotations(self):
        # This checks tensor algebra only, not a nonexistent arbitrary
        # rotation/permutation of a fixed lattice without interpolation.
        data, _ = periodic_field()
        features = self.features()
        moments = features._moments(data)
        rotation, _ = torch.linalg.qr(torch.tensor(
            [[0.4, 0.7, -0.1], [0.9, -0.2, 0.3], [0.5, 0.1, 0.8]],
            dtype=torch.float64,
        ))
        for parity in (-1, 1):
            orthogonal = rotation.clone()
            orthogonal[:, 0] *= parity
            rotated = {"s": moments["s"]}
            for key in ("v", "p", "u", "w"):
                rotated[key] = torch.einsum("...ai,ji->...aj", moments[key], orthogonal)
            for key in ("Q", "D"):
                rotated[key] = torch.einsum("...aij,ki,lj->...akl", moments[key], orthogonal, orthogonal)
            rotated["H"] = torch.einsum(
                "...aijk,li,mj,nk->...almn", moments["H"], orthogonal, orthogonal, orthogonal
            )
            torch.testing.assert_close(features._contract(rotated), features(data), atol=2e-12, rtol=2e-12)

    def test_translation_and_batching(self):
        data, positions = periodic_field()
        features = self.features()
        shifted_positions = (positions + torch.tensor([1, -1, 2])) % 5
        index = (shifted_positions[:, 0] * 5 + shifted_positions[:, 1]) * 5 + shifted_positions[:, 2]
        shifted = dict(data)
        for key in ("rho", "dipole_density"):
            shifted[key] = torch.empty_like(data[key])
            shifted[key][index] = data[key]
        torch.testing.assert_close(features(shifted)[index], features(data))
        batch = {key: torch.stack((data[key], shifted[key])) for key in data}
        actual = features(batch)
        torch.testing.assert_close(actual[0], features(data))
        torch.testing.assert_close(actual[1], features(shifted))

    def test_spatial_and_dipole_indices_are_not_symmetrized_together(self):
        data, _ = periodic_field()
        features = self.features(radial_exponents=(0.2,))
        moments = features._moments(data)
        self.assertFalse(torch.allclose(moments["D"][..., 0, 1], moments["D"][..., 1, 0]))
        torch.testing.assert_close(moments["Q"], moments["Q"].transpose(-1, -2))
        torch.testing.assert_close(moments["H"], moments["H"].transpose(-2, -3))
        self.assertFalse(torch.allclose(moments["H"], moments["H"].transpose(-1, -2)))

    def test_moments_match_direct_neighbor_sums_with_one_shared_basis(self):
        data, _ = periodic_field(n_types=2)
        features = self.features(n_types=2)
        with mock.patch.object(
            features.scalar_features, "stencil_basis",
            wraps=features.scalar_features.stencil_basis,
        ) as basis:
            moments = features._moments(data)
        basis.assert_called_once()

        # Independent reference using full coordinate tensors, without the
        # implementation's monomial indices or axis expansion. This checks
        # every radial/species channel and the independent dipole index.
        q = features.scalar_features.local_density_positions.double()
        weights = torch.exp(-q.square().sum(-1, keepdim=True) * features.radial_exponents)
        weights = weights / weights.sum(0, keepdim=True)
        neighbors = data["local_density_index"]
        rho = data["rho"][neighbors] / features.mean_density
        dipole = data["dipole_density"][neighbors] / features.dipole_density_scale
        Q = torch.einsum("jn,gjt,ja,jb->gntab", weights, rho, q, q)
        D = torch.einsum("jn,gjti,ja->gntai", weights, dipole, q)
        H = torch.einsum("jn,gjti,ja,jb->gntabi", weights, dipole, q, q)
        torch.testing.assert_close(moments["Q"], Q.flatten(1, 2))
        torch.testing.assert_close(moments["D"], D.flatten(1, 2))
        torch.testing.assert_close(moments["H"], H.flatten(1, 2))

    def test_center_vector_cross_contractions_and_optional_reversal(self):
        data, _ = periodic_field()
        features = self.features(radial_exponents=(0.2,))
        moments = features._moments(data)
        torch.testing.assert_close(moments["p"][..., -1, :], data["dipole_density"][..., 0, :] / 2)
        column = features.feature_names.index("p_dot_p[0,1]")
        expected = (moments["p"][..., 0, :] * moments["p"][..., 1, :]).sum(-1)
        torch.testing.assert_close(features(data)[..., column], expected)
        reversed_data = dict(data, dipole_density=-data["dipole_density"])
        self.assertFalse(torch.allclose(features(data), features(reversed_data)))
        even = self.features(dipole_reversal_symmetry=True)
        torch.testing.assert_close(even(data), even(reversed_data))

    def test_gradients_include_radial_parameters_and_are_finite_at_zero_dipole(self):
        data, _ = periodic_field()
        data["rho"].requires_grad_()
        data["dipole_density"].requires_grad_()
        features = self.features()
        objective = features(data)[7].square().sum()
        targets = (data["rho"], data["dipole_density"], features.scalar_features.log_radial_exponents)
        gradients = torch.autograd.grad(objective, targets, create_graph=True)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().max().item(), 0)
        zero = torch.zeros_like(data["dipole_density"], requires_grad=True)
        second = features(dict(data, dipole_density=zero))[7].sum()
        derivative = torch.autograd.grad(second, zero, create_graph=True)[0]
        self.assertTrue(torch.isfinite(derivative).all())

    def test_configuration_and_shapes(self):
        data, _ = periodic_field()
        for power in (0, 1, 2):
            for order in (1, 2, 3):
                features = self.features(max_power=power, max_product_order=order)
                self.assertEqual(features(data).shape[-1], features.n_features)
        for arguments in (
            {"max_power": 3}, {"max_product_order": 4},
            {"dipole_density_scale": 0}, {"dipole_density_scale": float("nan")},
        ):
            with self.assertRaises(ValueError):
                self.features(**arguments)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.features()(dict(data, dipole_density=data["dipole_density"].squeeze(-2)))
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.features()(dict(data, dipole_density=data["dipole_density"].float()))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.features()(dict(data, dipole_density=data["dipole_density"] * float("nan")))


if __name__ == "__main__":
    unittest.main()
