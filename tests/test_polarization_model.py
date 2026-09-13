"""Variational and compatibility checks for scalar/vector free energies."""

import io
import unittest

import numpy as np
import torch

from equicdft.features import CartesianAFeatures
from equicdft.model import GridCACEModel
from equicdft.readout import LocalReadout
from equicdft.semilocal import LDAReadout
from equicdft.solver import GridSolver
from equicdft.stencil import get_neighbor_indices
from equicdft.symmetrize import CartesianBFeatures


class TestPolarizationModel(unittest.TestCase):
    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(17)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)

    @staticmethod
    def data(spacing=0.5, batched=False):
        positions = np.indices((3, 3, 3)).reshape(3, -1).T
        indices, _ = get_neighbor_indices(positions, cutoff_grid=1)
        shape = (2, 27, 1) if batched else (27, 1)
        return {
            "rho": 0.4 + 0.2 * torch.rand(shape),
            "dipole_density": 0.05 * torch.randn(*shape, 3),
            "temperature": torch.tensor([1.2, 1.7] if batched else 1.2),
            "grid_spacing": torch.full((3,), spacing),
            "grid_size": torch.tensor([3, 3, 3]),
            "local_density_index": (
                torch.tensor(indices).expand(2, -1, -1).clone()
                if batched else torch.tensor(indices)
            ),
        }

    @staticmethod
    def model(spacing=0.5, mode="beta", with_scalar=False, **kwargs):
        a_features = CartesianAFeatures(
            mean_density=0.5, dipole_density_scale=0.3, include_polarization=True,
            cutoff_grid=1, max_power=2, radial_basis="gaussian",
            radial_exponents=(0.125, 0.5), trainable_radial_exponents=True,
        )
        b_features = CartesianBFeatures(2, 3, include_polarization=True)
        readouts = [LocalReadout(
            n_features=2 * b_features.n_features + 1, hidden_sizes=(5,),
        )]
        if with_scalar:
            readouts.append(LDAReadout(mean_density=0.5, hidden_sizes=(4,)))
        return GridCACEModel(
            a_features=a_features,
            b_features=b_features,
            readout=readouts,
            grid_spacing=spacing,
            mean_temperature=1.5,
            free_energy_mode=mode,
            compute_polarization_derivative=True,
            **kwargs
        )

    @staticmethod
    def cloned(data):
        return {key: value.detach().clone() for key, value in data.items()}

    def test_metadata_without_scalar_representation(self):
        model = self.model()
        self.assertEqual(model.n_types, 1)
        self.assertEqual(model.cutoff_grid, 1)
        self.assertTrue(model.requires_dipole_density)
        self.assertTrue(model.requires_local_density_index)
        self.assertEqual(model.grid_info["cutoff_grid"], 1)
        self.assertTrue(torch.allclose(model.mean_density, torch.tensor([0.5])))
        self.assertTrue(model.readout[0].requires_local_features)

    def test_finite_difference_both_fields_modes_and_voxel_volumes(self):
        # Changing one independent field holds the other fixed. The explicit
        # Delta V factor distinguishes discrete from functional derivatives.
        epsilon = 1.0e-5
        for spacing in (0.5, 0.8):
            for mode in ("beta", "physical"):
                model = self.model(spacing=spacing, mode=mode).eval()
                data = self.data(spacing=spacing)
                output = model(data)
                for name, index, response, sign in (
                    ("rho", (7, 0), "c1", -1),
                    ("dipole_density", (9, 0, 1), "polarization_derivative", 1),
                ):
                    plus, minus = self.cloned(data), self.cloned(data)
                    plus[name][index] += epsilon
                    minus[name][index] -= epsilon
                    positive = model(
                        plus, compute_c1=False,
                        compute_polarization_derivative=False,
                    )["beta_F_exc"]
                    negative = model(
                        minus, compute_c1=False,
                        compute_polarization_derivative=False,
                    )["beta_F_exc"]
                    finite_difference = (positive - negative) / (2 * epsilon)
                    expected = sign * spacing ** 3 * output[response][index]
                    torch.testing.assert_close(
                        finite_difference, expected, atol=2e-8, rtol=2e-6
                    )

    def test_batched_response_shapes_and_independent_fields(self):
        model = self.model().eval()
        data = self.data(batched=True)
        output = model(data)
        self.assertEqual(output["beta_F_exc"].shape, (2,))
        self.assertEqual(output["c1"].shape, (2, 27, 1))
        self.assertEqual(output["polarization_derivative"].shape, (2, 27, 1, 3))
        for field in range(2):
            single = self.cloned(data)
            for name in (
                "rho", "dipole_density", "temperature", "local_density_index"
            ):
                single[name] = single[name][field]
            expected = model(single)
            for name in output:
                torch.testing.assert_close(output[name][field], expected[name])

    def test_mixed_hessian_reciprocity(self):
        model = self.model().train()
        data = self.data()
        output = model(data)
        # c1 has a minus sign whereas the polarization response does not:
        # d(c1_i)/d(P_j) = -d(polarization_derivative_j)/d(rho_i).
        rho_index, polar_index = (5, 0), (6, 0, 2)
        mixed_from_c1 = torch.autograd.grad(
            output["c1"][rho_index], data["dipole_density"], retain_graph=True
        )[0][polar_index]
        mixed_from_polarization = torch.autograd.grad(
            output["polarization_derivative"][polar_index], data["rho"]
        )[0][rho_index]
        torch.testing.assert_close(
            mixed_from_c1, -mixed_from_polarization, atol=1e-12, rtol=1e-10
        )

    def test_c2_remains_density_density_response_at_fixed_polarization(self):
        model = self.model().eval()
        data = self.data()
        output = model(data, compute_c2=True, c2_reference=(4, 0))
        epsilon = 1e-5
        plus, minus = self.cloned(data), self.cloned(data)
        plus["rho"][7, 0] += epsilon
        minus["rho"][7, 0] -= epsilon
        positive = model(plus)["c1"][4, 0]
        negative = model(minus)["c1"][4, 0]
        finite_difference = (positive - negative) / (2 * epsilon * 0.5 ** 3)
        torch.testing.assert_close(
            finite_difference, output["c2"][7, 0], atol=1e-8, rtol=2e-6
        )
        self.assertFalse(output["c1"].requires_grad)
        self.assertFalse(output["polarization_derivative"].requires_grad)

    def test_optional_derivatives_under_no_grad(self):
        model = self.model().eval()
        with torch.no_grad():
            energy = model(
                self.data(), compute_c1=False,
                compute_polarization_derivative=False,
            )
            polar_only = model(self.data(), compute_c1=False)
            density_only = model(
                self.data(), compute_polarization_derivative=False
            )
        self.assertEqual(set(energy), {"beta_F_exc"})
        self.assertFalse(energy["beta_F_exc"].requires_grad)
        self.assertEqual(set(polar_only), {"beta_F_exc", "polarization_derivative"})
        self.assertEqual(set(density_only), {"beta_F_exc", "c1"})

    def test_response_loss_trains_radials_and_readout(self):
        model = self.model().train()
        output = model(self.data())
        loss = output["c1"].square().mean()
        loss = loss + output["polarization_derivative"].square().mean()
        loss.backward()
        feature_gradients = [
            parameter.grad for parameter in model.a_features.parameters()
        ]
        self.assertTrue(feature_gradients)
        self.assertTrue(all(value is not None for value in feature_gradients))
        self.assertTrue(all(torch.isfinite(value).all() for value in feature_gradients))
        self.assertTrue(any(value.abs().max() > 0 for value in feature_gradients))
        self.assertTrue(torch.isfinite(model.readout[0].mlp[0].weight.grad).all())

    def test_energy_invariance_and_response_covariance(self):
        model = self.model().eval()
        data = self.data()
        output = model(data)
        positions = np.indices((3, 3, 3)).reshape(3, -1).T
        # An axis swap and a reflection both test the electric (polar) vector
        # convention, including transformations with negative determinant.
        for rotation in (
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            np.diag([-1, 1, 1]),
        ):
            transformed_positions = (positions @ rotation.T) % 3
            destination = np.ravel_multi_index(
                transformed_positions.T, (3, 3, 3)
            )
            destination = torch.tensor(destination)
            rotation = torch.tensor(rotation, dtype=data["rho"].dtype)
            transformed = self.cloned(data)
            transformed["rho"][destination] = data["rho"].detach()
            transformed["dipole_density"][destination] = (
                data["dipole_density"].detach() @ rotation.T
            )
            actual = model(transformed)
            torch.testing.assert_close(
                actual["beta_F_exc"], output["beta_F_exc"],
                atol=1e-11, rtol=1e-10,
            )
            torch.testing.assert_close(
                actual["c1"][destination], output["c1"],
                atol=1e-11, rtol=1e-10,
            )
            torch.testing.assert_close(
                actual["polarization_derivative"][destination],
                output["polarization_derivative"] @ rotation.T,
                atol=1e-11, rtol=1e-10,
            )

    def test_additive_scalar_readouts_and_full_model_save_load(self):
        model = self.model(with_scalar=True).eval()
        data = self.data()
        output = model(data)
        stream = io.BytesIO()
        torch.save(model, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        actual = restored(self.cloned(data))
        for key in output:
            torch.testing.assert_close(actual[key], output[key], atol=0, rtol=0)

    def test_rejects_incompatible_cutoff_and_missing_polar_field(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, "cubic voxels"):
            GridCACEModel(
                model.a_features, model.b_features, list(model.readout),
                grid_spacing=(0.5, 0.5, 0.8),
            )
        data = self.data()
        del data["dipole_density"]
        with self.assertRaisesRegex(ValueError, "dipole_density is required"):
            model(data)

    def test_rejects_scalar_chemical_potential_and_unsupported_response(self):
        with self.assertRaisesRegex(ValueError, "orientational ideal"):
            self.model(compute_local_mu=True)
        scalar = GridCACEModel(
            None, None, [LDAReadout(mean_density=0.5)], grid_spacing=0.5
        )
        self.assertFalse(scalar.requires_dipole_density)
        with self.assertRaisesRegex(ValueError, "requires a dipole-density readout"):
            scalar(self.data(), compute_polarization_derivative=True)
        with self.assertRaisesRegex(ValueError, "orientational ideal"):
            GridSolver(self.model())

    def test_zero_polarization_response_for_scalar_only_invariant_selection(self):
        a = CartesianAFeatures(
            mean_density=0.5, dipole_density_scale=0.3, cutoff_grid=1,
            max_power=0, include_polarization=True,
        )
        b = CartesianBFeatures(0, 1, include_polarization=True)
        model = GridCACEModel(
            a, b, [LocalReadout(n_features=b.n_features + 1, hidden_sizes=(4,))],
            grid_spacing=0.5, compute_polarization_derivative=True,
        )
        for training in (False, True):
            model.train(training)
            data = self.data()
            output = model(data)
            response = output["polarization_derivative"]
            self.assertEqual(response.shape, (27, 1, 3))
            self.assertTrue(torch.equal(response, torch.zeros_like(response)))
            self.assertEqual(response.requires_grad, training)
            if training:
                second = torch.autograd.grad(
                    response.sum(), data["dipole_density"], retain_graph=True
                )[0]
                self.assertTrue(torch.equal(second, torch.zeros_like(second)))
                response.square().mean().backward()


if __name__ == "__main__":
    unittest.main()
