"""Joint-invariant B-chi messages: symmetry, gradients and compatibility."""
import copy
import itertools
import unittest

import torch

from equicdft import BChiMessage, GridCACEModel, PolarizationFeatures, PolarizationReadout
from test_polarization_features import periodic_field


class TestPolarizationMessage(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(17)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)

    def model(self, n_types=1, **kwargs):
        features = PolarizationFeatures(.7, 2., cutoff_grid=1, max_power=1,
                                        max_product_order=2, n_types=n_types)
        message = BChiMessage(features.n_features, 1, 1, hidden_sizes=(5,),
                              radial_exponents=(.2,), trainable_radial_exponents=True)
        readout = PolarizationReadout(features, (6,), message=message)
        return GridCACEModel(None, None, [readout], grid_spacing=.8,
                             compute_polarization_derivative=True, **kwargs)

    @staticmethod
    def data(n_types=1, size=5):
        data, positions = periodic_field(size=size, cutoff=1, n_types=n_types)
        data["temperature"] = torch.tensor(1.3)
        return data, positions

    def test_all_48_lattice_symmetries_energy_and_responses(self):
        model = self.model().eval()
        data, positions = self.data()
        reference = model(data)
        for permutation in itertools.permutations(range(3)):
            for signs in itertools.product((-1, 1), repeat=3):
                rotation = torch.eye(3)[list(permutation)] * torch.tensor(signs)[:, None]
                transformed_positions = (positions @ rotation.long().T) % 5
                index = (transformed_positions[:, 0]*5+transformed_positions[:, 1])*5+transformed_positions[:, 2]
                rho, dipole = torch.empty_like(data["rho"]), torch.empty_like(data["dipole_density"])
                rho[index] = data["rho"].detach()
                dipole[index] = data["dipole_density"].detach() @ rotation.T
                result = model(dict(data, rho=rho, dipole_density=dipole))
                torch.testing.assert_close(result["beta_F_exc"], reference["beta_F_exc"], atol=1e-11, rtol=1e-11)
                torch.testing.assert_close(result["c1"][index], reference["c1"], atol=1e-11, rtol=1e-11)
                torch.testing.assert_close(result["polarization_derivative"][index],
                                           reference["polarization_derivative"] @ rotation.T, atol=1e-11, rtol=1e-11)

    def test_finite_differences_and_mixed_reciprocity(self):
        for mode in ("beta", "physical"):
            model = self.model(free_energy_mode=mode).train()
            data, _ = self.data()
            result = model(data)
            for field, index, key, sign in (("rho", (42, 0), "c1", -1),
                                            ("dipole_density", (42, 0, 1), "polarization_derivative", 1)):
                plus = {k: v.detach().clone() for k, v in data.items()}
                minus = {k: v.detach().clone() for k, v in data.items()}
                plus[field][index] += 1e-5
                minus[field][index] -= 1e-5
                a, b = [model(x, compute_c1=False, compute_polarization_derivative=False)["beta_F_exc"] for x in (plus, minus)]
                torch.testing.assert_close((a-b)/2e-5, sign*.8**3*result[key][index], atol=1e-9, rtol=1e-6)
            left = torch.autograd.grad(result["c1"][42, 0], data["dipole_density"], retain_graph=True)[0][43, 0, 1]
            right = torch.autograd.grad(result["polarization_derivative"][43, 0, 1], data["rho"], retain_graph=True)[0][42, 0]
            torch.testing.assert_close(left, -right, atol=1e-12, rtol=1e-10)

    def test_batch_multispecies_empty_voxel_and_parameter_gradients(self):
        model = self.model(2).train()
        data, _ = self.data(2)
        data["rho"][0] = 0
        data["dipole_density"][0] = 0
        batch = {k: torch.stack((v, v)) for k, v in data.items()}
        result = model(batch)
        single = model(data)
        for key, value in result.items():
            torch.testing.assert_close(value[0], single[key])
        (result["c1"].square().mean()+result["polarization_derivative"].square().mean()).backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(model.readout[0].message.mlp[0].weight.grad.abs().max()), 0.)
        self.assertGreater(float(model.readout[0].message.log_radial_exponents.grad.abs().max()), 0.)
        clone = self.model(2).eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        for key, value in clone(data).items():
            torch.testing.assert_close(value, single[key])

    def test_message_has_two_stencil_reach_in_polarization(self):
        model = self.model()
        data, _ = self.data(size=7)
        data["dipole_density"].requires_grad_()
        readout = model.readout[0]
        center = (3*7+3)*7+3
        output = readout(dict(data, normalized_temperature=torch.tensor(1.)))
        gradient = torch.autograd.grad(output[center, 0], data["dipole_density"])[0]
        self.assertGreater(float(gradient[(5*7+3)*7+3].abs().max()), 1e-12)
        self.assertEqual(float(gradient[(6*7+3)*7+3].abs().max()), 0.)

    def test_zero_message_recovers_no_message_energy_and_responses(self):
        model = self.model().eval()
        old = PolarizationReadout(copy.deepcopy(model.readout[0].features), (6,))
        baseline = GridCACEModel(None, None, [old], grid_spacing=.8, compute_polarization_derivative=True).eval()
        with torch.no_grad():
            model.readout[0].message.mlp[-1].weight.zero_()
            old.mlp[0].weight.copy_(model.readout[0].mlp[0].weight[:, :old.features.n_features+1])
            # Temperature is the last input, after appended message invariants.
            old.mlp[0].weight[:, -1].copy_(model.readout[0].mlp[0].weight[:, -1])
            old.mlp[0].bias.copy_(model.readout[0].mlp[0].bias)
            old.mlp[-1].load_state_dict(model.readout[0].mlp[-1].state_dict())
        data, _ = self.data()
        for key, value in model(data).items():
            torch.testing.assert_close(value, baseline(data)[key], atol=1e-12, rtol=1e-12)

    def test_rejects_incompatible_message(self):
        features = PolarizationFeatures(.7, 2., max_power=1)
        with self.assertRaises(TypeError):
            PolarizationReadout(features, message="one")
        with self.assertRaises(ValueError):
            PolarizationReadout(features, message=BChiMessage(3, 1, 1))
        with self.assertRaises(ValueError):
            PolarizationReadout(features, message=BChiMessage(features.n_features, 1, 1, convolution_backend="fft"))


if __name__ == "__main__":
    unittest.main()
