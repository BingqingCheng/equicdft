"""Pointwise rho/P excess energy, smooth response, and scalar compatibility."""

import io
import unittest

import torch

from equicdft import GridCACEModel, LDAReadout, PolarizationFeatures, PolarizationReadout, PolarizationSolver
from equicdft.stencil import get_neighbor_indices
import numpy as np


class TestPolarizationLDA(unittest.TestCase):
    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(29)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)

    @staticmethod
    def readout(n_types=1, **kwargs):
        return LDAReadout(mean_density=.5, dipole_density_scale=.3,
                          n_types=n_types, hidden_sizes=kwargs.pop("hidden_sizes", (5,)), **kwargs)

    @staticmethod
    def data(n_types=1, batched=False):
        shape = (2, 8, n_types) if batched else (8, n_types)
        return {"rho": .4 + .2 * torch.rand(shape),
                "dipole_density": .03 * torch.randn(*shape, 3),
                "temperature": torch.tensor([1.2, 1.7] if batched else 1.2)}

    @staticmethod
    def model(readouts, spacing=.5, mode="beta"):
        return GridCACEModel(None, None, readouts, grid_spacing=spacing,
                             mean_temperature=1.5, free_energy_mode=mode,
                             compute_polarization_derivative=True)

    @staticmethod
    def clone(data):
        return {k: v.detach().clone() for k, v in data.items()}

    def test_exact_multi_species_input_order_and_energy(self):
        readout = self.readout(3, hidden_sizes=())
        data = self.data(3, batched=True)
        context = dict(data, normalized_temperature=data["temperature"] / 1.5, voxel_volume=.125)
        with torch.no_grad():
            readout.mlp[0].weight.copy_(torch.tensor([[1., 2., 3., 4., 5., 6., 7.]]))
            readout.mlp[0].bias.fill_(.2)
        rho, p = data["rho"], data["dipole_density"]
        expected_a = ((rho / .5) * torch.tensor([1., 2., 3.])).sum(-1)
        expected_a += ((p / .3).square().sum(-1) * torch.tensor([4., 5., 6.])).sum(-1)
        expected_a += 7 * context["normalized_temperature"][:, None] + .2
        expected = .125 * (rho.sum(-1) * expected_a).sum(-1)
        torch.testing.assert_close(readout.energy(context), expected, atol=1e-12, rtol=1e-12)

    def test_standalone_metadata_and_no_neighbor_requirement(self):
        model = self.model([self.readout()])
        self.assertTrue(model.requires_dipole_density)
        self.assertFalse(model.requires_local_density_index)
        self.assertEqual(model.cutoff_grid, 0)
        self.assertEqual(model.grid_info["cutoff_grid"], 0)
        self.assertTrue(torch.isfinite(model(self.data())["beta_F_exc"]))

    def test_analytic_zero_p_response_hessian_and_empty_voxel(self):
        readout = self.readout(hidden_sizes=())
        with torch.no_grad():
            readout.mlp[0].weight.copy_(torch.tensor([[.7, 1.2, .4]]))
            readout.mlp[0].bias.fill_(.1)
        model = self.model([readout]).train()
        data = self.data()
        data["rho"][0] = 0.
        data["dipole_density"].zero_()
        output = model(data)
        response = output["polarization_derivative"]
        self.assertTrue(torch.equal(response, torch.zeros_like(response)))
        hessian_row = torch.autograd.grad(response[2, 0, 1], data["dipole_density"], retain_graph=True)[0]
        expected = torch.zeros_like(response)
        expected[2, 0, 1] = 2 * 1.2 * data["rho"][2, 0] / .3**2
        torch.testing.assert_close(hessian_row, expected)
        for value in output.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_finite_difference_both_fields_modes_and_volumes(self):
        for mode in ("beta", "physical"):
            for spacing in (.5, .8):
                model = self.model([self.readout()], spacing, mode).eval()
                data = self.data()
                output = model(data)
                for field, index, key, sign in (("rho", (3, 0), "c1", -1),
                                                ("dipole_density", (3, 0, 2), "polarization_derivative", 1)):
                    plus, minus = self.clone(data), self.clone(data)
                    plus[field][index] += 1e-5
                    minus[field][index] -= 1e-5
                    def energy(values):
                        return model(values, compute_c1=False, compute_polarization_derivative=False)["beta_F_exc"]
                    finite_difference = (energy(plus) - energy(minus)) / 2e-5
                    torch.testing.assert_close(finite_difference, sign * spacing**3 * output[key][index],
                                               atol=2e-9, rtol=2e-6)

    def test_locality_and_mixed_hessian_reciprocity(self):
        model = self.model([self.readout()]).train()
        data = self.data()
        output = model(data)
        left = torch.autograd.grad(output["c1"][2, 0], data["dipole_density"], retain_graph=True)[0]
        right = torch.autograd.grad(output["polarization_derivative"][2, 0, 1], data["rho"], retain_graph=True)[0]
        torch.testing.assert_close(left[2, 0, 1], -right[2, 0], atol=1e-12, rtol=1e-10)
        self.assertGreater(abs(float(left[2, 0, 1])), 1e-8)
        left[2] = 0
        right[2] = 0
        self.assertTrue(torch.equal(left, torch.zeros_like(left)))
        self.assertTrue(torch.equal(right, torch.zeros_like(right)))

    def test_rotation_reflection_reversal_and_response(self):
        model = self.model([self.readout()]).eval()
        data = self.data()
        output = model(data)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        for transform in (rotation, torch.diag(torch.tensor([-1., 1., 1.])), -torch.eye(3)):
            changed = self.clone(data)
            changed["dipole_density"] = changed["dipole_density"] @ transform.T
            actual = model(changed)
            torch.testing.assert_close(actual["beta_F_exc"], output["beta_F_exc"])
            torch.testing.assert_close(actual["c1"], output["c1"])
            torch.testing.assert_close(actual["polarization_derivative"], output["polarization_derivative"] @ transform.T)

    def test_batch_shapes_and_response_loss_gradients(self):
        model = self.model([self.readout(3)]).train()
        data = self.data(3, batched=True)
        output = model(data)
        self.assertEqual(output["c1"].shape, (2, 8, 3))
        self.assertEqual(output["polarization_derivative"].shape, (2, 8, 3, 3))
        for index in range(2):
            single = model({k: v[index].detach().clone() for k, v in data.items()})
            for key in output:
                torch.testing.assert_close(single[key], output[key][index])
        (output["c1"].square().mean() + output["polarization_derivative"].square().mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.readout[0].mlp[0].weight.grad.abs().max()), 0.)

    def test_additive_neighbor_branch_in_either_order(self):
        lda = self.readout()
        polar = PolarizationReadout(PolarizationFeatures(.5, .3, cutoff_grid=1, max_power=1), hidden_sizes=(4,))
        data = self.data()
        indices, _ = get_neighbor_indices(np.indices((2, 2, 2)).reshape(3, -1).T, cutoff_grid=1)
        data["local_density_index"] = torch.tensor(indices)
        separate = [self.model([readout]).eval()(self.clone(data)) for readout in (lda, polar)]
        for readouts in ([lda, polar], [polar, lda]):
            model = self.model(readouts).eval()
            self.assertEqual(model.cutoff_grid, 1)
            self.assertTrue(model.requires_local_density_index)
            actual = model(self.clone(data))
            for key in actual:
                torch.testing.assert_close(actual[key], separate[0][key] + separate[1][key], atol=1e-12, rtol=1e-10)

    def test_zero_init_keeps_other_branch_and_no_ideal_entropy(self):
        model = self.model([self.readout(zero_init=True)]).train()
        data = self.data()
        actual = model(data)
        for value in actual.values():
            self.assertTrue(torch.equal(value, torch.zeros_like(value)))
        actual["polarization_derivative"].square().mean().backward()

    def test_zero_lda_preserves_ideal_density_and_orientation_equilibrium(self):
        model = self.model([self.readout(zero_init=True)]).eval()
        data = {"V_ext": .1 * torch.randn(8, 1),
                "E_ext": .1 * torch.randn(8, 1, 3),
                "temperature": torch.tensor(1.2),
                "beta": 1. / (model.boltzmann_constant * 1.2),
                "grid_spacing": torch.full((3,), .5)}
        ideal = PolarizationSolver(.7).solve(data, [.5], tolerance_residual=1e-10)
        with_lda = PolarizationSolver(.7, model).solve(data, [.5], tolerance_residual=1e-10)
        self.assertEqual(ideal["status"], "converged")
        self.assertEqual(with_lda["status"], "converged")
        for key in ("rho", "dipole_density", "beta_A"):
            torch.testing.assert_close(with_lda[key], ideal[key], atol=1e-12, rtol=1e-12)

    def test_serialization_and_density_only_state_layout(self):
        scalar = LDAReadout(mean_density=.5, hidden_sizes=(5,))
        self.assertFalse(scalar.requires_dipole_density)
        self.assertEqual(scalar.n_state_features, 2)
        self.assertNotIn("dipole_density_scale", scalar.state_dict())
        for readout in (scalar, self.readout()):
            model = GridCACEModel(None, None, [readout], grid_spacing=.5,
                                  compute_polarization_derivative=readout.requires_dipole_density).eval()
            data = self.data()
            output = model(data)
            stream = io.BytesIO()
            torch.save(model, stream)
            stream.seek(0)
            restored = torch.load(stream, weights_only=False)
            for key, value in restored(self.clone(data)).items():
                torch.testing.assert_close(value, output[key], atol=0, rtol=0)

    def test_rejects_invalid_scale_shapes_and_nonfinite_p(self):
        for scale in (0., -1., float("nan"), float("inf"), [.1, .2]):
            with self.assertRaisesRegex(ValueError, "dipole_density_scale"):
                LDAReadout(.5, dipole_density_scale=scale)
        model = self.model([self.readout()])
        data = self.data()
        del data["dipole_density"]
        with self.assertRaisesRegex(ValueError, "dipole_density is required"):
            model(data)
        for value in (torch.ones(8, 3), torch.full((8, 1, 3), float("nan")), torch.ones(8, 1, 3, dtype=torch.float32)):
            data = self.data()
            data["dipole_density"] = value
            with self.assertRaises(ValueError):
                model(data)


if __name__ == "__main__":
    unittest.main()
