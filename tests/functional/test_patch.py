"""Raw-patch ordering, scalar derivatives, and model integration."""

import copy
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from equicdft import (
    DensityPatchReadout, GridCACEModel, GridData, TensorLoss, load_model, save_model,
)
from equicdft.functional.base import ReadoutContext
from equicdft.geometry.stencil import make_stencil
from tests.model_fixtures import grid_data


def field(n_types=1):
    data = grid_data(shape=(3, 3, 3), cutoff_grid=1, n_types=n_types, grid_spacing=0.5)
    return {
        key: value.double() if value.is_floating_point() else value
        for key, value in data.items()
    }


def context(data, **overrides):
    arguments = dict(
        rho=data["rho"],
        normalized_temperature=data["temperature"] / 1.25,
        voxel_volume=torch.tensor(0.125, dtype=torch.float64),
        mean_density=torch.tensor(0.5, dtype=torch.float64),
        local_density_index=data["local_density_index"],
        grid_size=data["grid_size"],
        grid_positions=data["grid_positions"],
    )
    arguments.update(overrides)
    return ReadoutContext(**arguments)


def patch_model(n_types=1, **overrides):
    arguments = dict(
        readout=[DensityPatchReadout(n_types, cutoff_grid=1, hidden_sizes=(4,))],
        grid_spacing=0.5,
        mean_density=0.5,
        mean_temperature=1.25,
        boltzmann_constant=1.0,
        free_energy_mode="beta",
    )
    arguments.update(overrides)
    return GridCACEModel(**arguments).double()


class TestDensityPatchReadout(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_default_patch_width_and_center(self):
        readout = DensityPatchReadout()
        offsets = make_stencil(readout.cutoff_grid)
        self.assertEqual(readout.n_neighbors, 257)
        self.assertEqual(readout.n_features, 258)
        self.assertEqual(readout.mlp[0].in_features, 258)
        self.assertTrue(np.array_equal(offsets[0], [0, 0, 0]))
        self.assertEqual(np.sum(np.all(offsets == 0, axis=-1)), 1)
        self.assertTrue(np.all(np.sum(offsets ** 2, axis=-1) <= 16))

    def test_one_domain_flag_with_the_same_model_and_data_api(self):
        data = field()
        for domain, neighbors in (("sphere", 7), ("cube", 27)):
            with self.subTest(domain=domain):
                readout = DensityPatchReadout(cutoff_grid=1, domain=domain)
                model = patch_model(readout=[readout]).eval()
                outputs = model(copy.deepcopy(data), compute_c2=True)
                self.assertEqual(readout.n_neighbors, neighbors)
                self.assertEqual(readout.to_config()["domain"], domain)
                for key in ("beta_F_exc", "c1", "c2"):
                    self.assertTrue(torch.isfinite(outputs[key]).all())

    def test_cube_patch_width_order_and_periodic_values(self):
        wide = DensityPatchReadout(domain="cube")
        self.assertEqual(wide.n_neighbors, 729)
        self.assertEqual(wide.n_features, 730)
        self.assertEqual(sum(p.numel() for p in wide.parameters()), 23937)
        self.assertFalse(wide.requires_local_density_index)
        data = field(n_types=2)
        readout = DensityPatchReadout(2, cutoff_grid=1, domain="cube", hidden_sizes=()).double()
        offsets = np.asarray(list(itertools.product(range(-1, 2), repeat=3)))
        self.assertEqual(np.sum(np.all(offsets == 0, axis=-1)), 1)
        coordinates = data["grid_positions"].numpy()
        index = np.ravel_multi_index(((coordinates[:, None]+offsets) % 3).reshape(-1, 3).T, (3, 3, 3)).reshape(27, 27)
        captured = []
        hook = readout.mlp.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0]))
        readout.energy(context(data, local_density_index=None))
        hook.remove()
        expected = torch.cat(((data["rho"][torch.as_tensor(index)]/.5).flatten(-2),
                              torch.full((27, 1), 1.2, dtype=torch.float64)), dim=-1)
        torch.testing.assert_close(captured[0], expected)
        self.assertNotIn("_cube_index", readout.state_dict())

    def test_cube_batched_and_shuffled_rows(self):
        data = field(n_types=2)
        readout = DensityPatchReadout(2, cutoff_grid=1, domain="cube").double()
        expected = readout.energy(context(data))
        permutation = torch.randperm(27)
        shuffled = context(data, rho=data["rho"][permutation],
                           grid_positions=data["grid_positions"][permutation])
        torch.testing.assert_close(readout.energy(shuffled), expected)
        rho = torch.stack((data["rho"], data["rho"][permutation]))
        positions = torch.stack((data["grid_positions"], data["grid_positions"][permutation]))
        batch = context(data, rho=rho, grid_positions=positions,
                        normalized_temperature=torch.full((2,), 1.2, dtype=torch.float64))
        torch.testing.assert_close(readout.energy(batch), expected.expand(2))

    def test_cube_quadratic_c1_and_c2_include_all_overlaps(self):
        data = field()
        readout = DensityPatchReadout(cutoff_grid=1, domain="cube", hidden_sizes=()).double()
        weights = torch.arange(1, 28, dtype=torch.float64)/27
        with torch.no_grad():
            readout.mlp[0].weight.copy_(torch.cat((weights, weights.new_tensor([.3])))[None])
            readout.mlp[0].bias.fill_(.2)
        offsets = np.asarray(list(itertools.product(range(-1, 2), repeat=3)))
        positions = data["grid_positions"].numpy()
        neighbors = (positions[:, None]+offsets) % 3
        index = torch.from_numpy(np.ravel_multi_index(neighbors.reshape(-1, 3).T, (3, 3, 3)).reshape(27, 27))
        matrix = torch.zeros(27, 27, dtype=torch.float64).scatter_add_(1, index, weights.expand(27, -1))
        rho = data["rho"][:, 0]
        result = patch_model(readout=[readout]).eval()(data, compute_c2=True, c2_reference=(5, 0))
        torch.testing.assert_close(result["beta_F_exc"], .125*(rho*(matrix@rho/.5+.56)).sum())
        torch.testing.assert_close(result["c1"][:, 0], -(matrix+matrix.T)@rho/.5-.56)
        torch.testing.assert_close(result["c2"][:, 0], -(matrix+matrix.T)[5]/(.5*.125))

    def test_cube_nonlinear_derivatives_and_loss_gradients(self):
        model = patch_model(readout=[DensityPatchReadout(cutoff_grid=1, domain="cube", hidden_sizes=(4,))],
                            compute_local_mu=True).eval()
        data = field()
        result = model(copy.deepcopy(data), compute_c2=True, c2_reference=(4, 0))
        shifted = []
        for sign in (-1, 1):
            sample = copy.deepcopy(data)
            sample["rho"][13, 0] += sign*1e-5
            shifted.append(model(sample))
        torch.testing.assert_close(result["c1"][13, 0],
            -(shifted[1]["beta_F_exc"]-shifted[0]["beta_F_exc"])/(2e-5*.125), rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(result["c2"][13, 0],
            (shifted[1]["c1"][4, 0]-shifted[0]["c1"][4, 0])/(2e-5*.125), rtol=1e-7, atol=1e-8)
        term = TensorLoss(name="mu", prediction_key="beta_mu_local", target_key="beta_mu_average", weights_key="voxel_weights")
        model.train()
        result = model(copy.deepcopy(data))
        term(result, data).backward()
        self.assertTrue(torch.isfinite(model.readout[0].mlp[0].weight.grad).all())
        self.assertGreater(model.readout[0].mlp[0].weight.grad.abs().sum().item(), 0)

    def test_cube_serialization_and_old_sphere_configuration(self):
        model = patch_model(readout=[DensityPatchReadout(cutoff_grid=1, domain="cube")]).eval()
        data = field()
        expected = model(copy.deepcopy(data))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"cube.pt"
            save_model(model, path)
            restored = load_model(path)
        self.assertEqual(restored.readout[0].domain, "cube")
        self.assertEqual(restored.cutoff_grid, 1)
        self.assertEqual(restored.readout[0]._cube_index.numel(), 0)
        actual = restored(data)
        for key in ("beta_F_exc", "c1"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        config = DensityPatchReadout(cutoff_grid=1).to_config()
        config.pop("domain")
        self.assertEqual(DensityPatchReadout.from_config(config).domain, "sphere")

    def test_raw_input_order_normalization_and_integral(self):
        data = field(n_types=2)
        readout = DensityPatchReadout(2, cutoff_grid=1, hidden_sizes=()).double()
        captured = []
        handle = readout.mlp.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0])
        )
        energy = readout.energy(context(data))
        handle.remove()
        expected = torch.cat((
            (data["rho"][data["local_density_index"]] / 0.5).flatten(-2),
            torch.full((27, 1), 1.2, dtype=torch.float64),
        ), dim=-1)
        torch.testing.assert_close(captured[0], expected)
        per_particle = readout(expected)
        torch.testing.assert_close(energy, 0.125 * (data["rho"] * per_particle).sum())

    def test_batched_fields_match_independent_fields(self):
        data = field(n_types=2)
        readout = DensityPatchReadout(2, cutoff_grid=1, hidden_sizes=(4,)).double()
        rho = torch.stack((data["rho"], 2 * data["rho"]))
        temperatures = torch.tensor([1.2, 0.8], dtype=torch.float64)
        expected = torch.stack([
            readout.energy(context(data, rho=rho[i], normalized_temperature=temperatures[i]))
            for i in range(2)
        ])
        for index in (
            data["local_density_index"],
            data["local_density_index"].expand(2, -1, -1),
        ):
            actual = readout.energy(context(
                data, rho=rho, normalized_temperature=temperatures,
                local_density_index=index,
            ))
            self.assertEqual(actual.shape, (2,))
            torch.testing.assert_close(actual, expected)

    def test_exact_quadratic_energy_c1_and_c2(self):
        data = field()
        readout = DensityPatchReadout(cutoff_grid=1, hidden_sizes=()).double()
        # An explicit quadratic scalar checks overlap contributions and both
        # 1/DeltaV functional-derivative factors independently of autograd.
        weights = torch.arange(1, 8, dtype=torch.float64) / 7
        with torch.no_grad():
            readout.mlp[0].weight.copy_(torch.cat((weights, weights.new_tensor([0.3])))[None])
            readout.mlp[0].bias.fill_(0.2)
        model = patch_model(readout=[readout], compute_c2=True).eval()
        matrix = torch.zeros(27, 27, dtype=torch.float64)
        matrix.scatter_add_(1, data["local_density_index"], weights.expand(27, -1))
        rho = data["rho"].squeeze(-1)
        constant = 0.3 * 1.2 + 0.2
        expected_energy = 0.125 * (rho * (matrix @ rho / 0.5 + constant)).sum()
        expected_c1 = -(matrix + matrix.T) @ rho / 0.5 - constant
        reference = 5
        expected_c2 = -(matrix + matrix.T)[reference] / (0.5 * 0.125)
        outputs = model(data, c2_reference=(reference, 0))
        torch.testing.assert_close(outputs["beta_F_exc"], expected_energy)
        torch.testing.assert_close(outputs["c1"][:, 0], expected_c1)
        torch.testing.assert_close(outputs["c2"][:, 0], expected_c2)

    def test_nonlinear_c1_and_c2_match_finite_differences(self):
        model = patch_model().eval()
        data = field()
        reference, perturbed = 4, 13
        outputs = model(copy.deepcopy(data), compute_c2=True, c2_reference=(reference, 0))
        energies, first_derivatives = [], []
        step = 1e-5
        for sign in (-1, 1):
            shifted = copy.deepcopy(data)
            shifted["rho"][perturbed, 0] += sign * step
            result = model(shifted)
            energies.append(result["beta_F_exc"].detach())
            first_derivatives.append(result["c1"][reference, 0])
        c1_fd = -(energies[1] - energies[0]) / (2 * step * 0.125)
        c2_fd = (first_derivatives[1] - first_derivatives[0]) / (2 * step * 0.125)
        torch.testing.assert_close(outputs["c1"][perturbed, 0], c1_fd, rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(outputs["c2"][perturbed, 0], c2_fd, rtol=1e-7, atol=1e-8)

    def test_physical_convention_applies_model_beta_factor(self):
        beta_model = patch_model().eval()
        physical_model = copy.deepcopy(beta_model)
        physical_model.free_energy_mode = "physical"
        data = field()
        beta_outputs = beta_model(copy.deepcopy(data), compute_c2=True)
        physical_outputs = physical_model(copy.deepcopy(data), compute_c2=True)
        factor = physical_model.mean_temperature / data["temperature"]
        for key in ("beta_F_exc", "c1", "c2"):
            torch.testing.assert_close(physical_outputs[key], factor * beta_outputs[key])

    def test_local_mu_loss_backpropagates_to_patch_network(self):
        model = patch_model(compute_local_mu=True)
        data = field()
        outputs = model(data)
        term = TensorLoss(
            name="local_mu", prediction_key="beta_mu_local",
            target_key="beta_mu_average", weights_key="voxel_weights",
        )
        loss = term(outputs, data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient = model.readout[0].mlp[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_batched_model_energy_c1_and_c2_match_single_fields(self):
        model = patch_model(n_types=2).eval()
        first = field(n_types=2)
        second = field(n_types=2)
        second["rho"] *= 0.8
        second["temperature"] *= 1.1
        second["beta"] = 1.0 / second["temperature"]
        batch = {key: torch.stack((first[key], second[key])) for key in first}
        actual = model(batch, compute_c2=True, c2_reference=(4, 1))
        independent = [
            model(data, compute_c2=True, c2_reference=(4, 1))
            for data in (first, second)
        ]
        for key in ("beta_F_exc", "c1", "c2"):
            expected = torch.stack([outputs[key] for outputs in independent])
            torch.testing.assert_close(actual[key], expected)

    def test_translation_symmetry_but_not_axis_swap_symmetry(self):
        data = field()
        readout = DensityPatchReadout(cutoff_grid=1, hidden_sizes=(1,)).double()
        # Deliberately select one direction to show that no lattice-symmetry
        # averaging has slipped into the raw-patch baseline.
        offsets = make_stencil(1)
        x_neighbor = int(np.flatnonzero(np.all(offsets == [1, 0, 0], axis=1))[0])
        with torch.no_grad():
            readout.mlp[0].weight.zero_()
            readout.mlp[0].weight[0, x_neighbor] = 1.0
            readout.mlp[0].bias.zero_()
            readout.mlp[-1].weight.fill_(1.0)
            readout.mlp[-1].bias.zero_()
        model = patch_model(readout=[readout]).eval()
        original = model(copy.deepcopy(data))
        grid_rho = data["rho"].reshape(3, 3, 3, 1)
        shifted = copy.deepcopy(data)
        shifted["rho"] = grid_rho.roll(1, dims=0).reshape(27, 1)
        translated = model(shifted)
        torch.testing.assert_close(original["beta_F_exc"], translated["beta_F_exc"])
        torch.testing.assert_close(
            original["c1"].reshape(3, 3, 3, 1).roll(1, dims=0).reshape(27, 1),
            translated["c1"],
        )
        rotated = copy.deepcopy(data)
        rotated["rho"] = grid_rho.transpose(0, 1).reshape(27, 1)
        rotation_difference = original["beta_F_exc"] - model(rotated)["beta_F_exc"]
        self.assertGreater(rotation_difference.abs().item(), 1e-5)

    def test_model_grid_info_serialization_and_no_features(self):
        model = patch_model().eval()
        self.assertIsNone(model.a_features)
        self.assertIsNone(model.b_features)
        self.assertEqual(len(model.readout), 1)
        self.assertEqual(model.cutoff_grid, 1)
        self.assertTrue(model.requires_local_density_index)
        data = GridData.from_dict(
            {"grid_size": [3, 3, 3], "temperature": 1.5}, grid_info=model.grid_info,
        )
        self.assertEqual(data["local_density_index"].shape, (27, 7))
        expected = model(field())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch.pt"
            save_model(model, path)
            restored = load_model(path)
        self.assertEqual(restored.to_config(), model.to_config())
        actual = restored(field())
        for key in ("beta_F_exc", "c1"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_vacuum_energy_and_center_only_patch(self):
        data = field()
        data["rho"].zero_()
        model = patch_model().eval()
        outputs = model(data)
        self.assertEqual(outputs["beta_F_exc"].item(), 0)
        self.assertTrue(torch.isfinite(outputs["c1"]).all())
        readout = DensityPatchReadout(cutoff_grid=0)
        self.assertEqual(readout.n_neighbors, 1)
        self.assertEqual(readout.n_features, 2)

    def test_invalid_inputs_fail_clearly(self):
        for arguments in ({"cutoff_grid": -1}, {"n_types": 0}, {"hidden_sizes": (0,)}, {"domain": "ball"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                DensityPatchReadout(**arguments)
        data = field()
        readout = DensityPatchReadout(cutoff_grid=1).double()
        with self.assertRaisesRegex(KeyError, "local_density_index"):
            readout.energy(context(data, local_density_index=None))
        with self.assertRaisesRegex(KeyError, "mean_density"):
            readout.energy(context(data, mean_density=None))
        with self.assertRaisesRegex(ValueError, "cutoff_grid"):
            readout.energy(context(data, local_density_index=data["local_density_index"][:, :1]))
        with self.assertRaisesRegex(ValueError, "n_types"):
            readout.energy(context(field(2)))
        with self.assertRaisesRegex(ValueError, "n_features"):
            readout(torch.zeros(27, 2, dtype=torch.float64))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_model_and_derivatives(self):
        for domain in ("sphere", "cube"):
            model = patch_model(readout=[DensityPatchReadout(cutoff_grid=1, domain=domain)]).cuda().eval()
            data = {key: value.cuda() for key, value in field().items()}
            outputs = model(data, compute_c2=True)
            for key in ("beta_F_exc", "c1", "c2"):
                self.assertEqual(outputs[key].device.type, "cuda")
                self.assertTrue(torch.isfinite(outputs[key]).all())


if __name__ == "__main__":
    unittest.main()
