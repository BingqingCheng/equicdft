import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from equicdft import (
    GridCACEModel, Loss, PairwiseReadout, TensorLoss, Trainer,
    load_model, save_model,
)
from equicdft._config import build
from equicdft.serialization import default_dtype
from tests.functional.test_pairwise import _context


def _density_context(rho, shape=(5, 5, 5), **kwargs):
    return replace(_context(rho, shape, **kwargs), mean_density=0.6)


def _endpoint_energy(readout, context):
    """Explicit physical-species pair sum, independent of the FFT path."""
    rho = context.rho
    shape = tuple(context.grid_size.tolist())
    rho_grid = rho.reshape(*shape, readout.n_types)
    inputs = (
        context.pointwise_state() if readout.density_temperature
        else context.normalized_rho()
    )
    coefficients = readout.density_mlp(inputs).reshape(
        *shape, readout.n_types, readout.density_channels,
    )
    shell_values = readout.shell_kernel_values(context.normalized_temperature)
    energy = rho.new_zeros(())
    for offset, shell in zip(readout.offsets.tolist(), readout.offset_shell_index):
        shifts = tuple(-value for value in offset)
        neighbor_rho = torch.roll(rho_grid, shifts, (0, 1, 2))
        neighbor_coefficients = torch.roll(coefficients, shifts, (0, 1, 2))
        for pair, (first, second) in enumerate(readout.kernel_pairs):
            endpoint = (
                coefficients[..., :, first, None]
                * neighbor_coefficients[..., None, :, second]
            )
            if first != second:
                endpoint = endpoint + (
                    coefficients[..., :, second, None]
                    * neighbor_coefficients[..., None, :, first]
                )
            energy = energy + shell_values[shell, pair] * torch.sum(
                rho_grid[..., :, None] * neighbor_rho[..., None, :] * endpoint,
            )
    return 0.5 * context.voxel_volume.square() * energy


class TestDensityDependentPairwise(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def readout(self, n_types=2, channels=3, zero_init=False, temperature=False):
        return PairwiseReadout(
            cutoff_grid=2, n_types=n_types, hidden_sizes=(5,),
            zero_init=zero_init, density_channels=channels,
            density_hidden_sizes=(4,), density_temperature=temperature,
        ).double()

    def test_options_are_validated(self):
        for value, error in (
            (0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError),
        ):
            with self.subTest(value=value), self.assertRaises(error):
                PairwiseReadout(2, 2, density_channels=value)
        with self.assertRaises(ValueError):
            PairwiseReadout(2, 2, density_channels=3, density_hidden_sizes=(0,))
        with self.assertRaisesRegex(ValueError, "density_temperature requires"):
            PairwiseReadout(2, 2, density_temperature=True)
        with self.assertRaises(TypeError):
            PairwiseReadout(2, 2, density_channels=3, density_temperature=1)

    def test_default_keeps_legacy_configuration_state_and_sources(self):
        legacy = {
            "type": "PairwiseReadout", "cutoff_grid": 2, "n_types": 2,
            "hidden_sizes": [5], "zero_init": False,
        }
        torch.manual_seed(24)
        original = build(legacy)
        torch.manual_seed(24)
        explicit = PairwiseReadout(2, 2, hidden_sizes=(5,), zero_init=False, density_channels=None)
        self.assertEqual(original.to_config(), legacy)
        self.assertEqual(explicit.to_config(), legacy)
        self.assertEqual(set(original.state_dict()), {
            "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias",
        })
        explicit.load_state_dict(original.state_dict(), strict=True)
        rho = torch.rand(125, 2)
        context = _context(rho, (5, 5, 5))  # no density normalization needed
        self.assertIs(original.density_fields(context), rho)
        self.assertTrue(torch.equal(original.energy(context), explicit.energy(context)))

    def test_endpoint_features_are_pointwise_use_all_species_and_keep_vacuum(self):
        readout = PairwiseReadout(2, 3, density_channels=2, density_hidden_sizes=()).double()
        with torch.no_grad():
            readout.density_mlp[0].weight.fill_(0.2)
            readout.density_mlp[0].bias.fill_(0.3)
        rho = torch.rand(125, 3, dtype=torch.float64)
        rho[0] = 0.0
        fields = readout.density_fields(_density_context(rho))
        self.assertEqual(fields.shape, (125, 2))
        self.assertTrue(torch.equal(fields[0], torch.zeros(2, dtype=rho.dtype)))
        changed = rho.clone()
        changed[1, 2] += 0.1
        actual = readout.density_fields(_density_context(changed))
        self.assertTrue(torch.equal(fields[2:], actual[2:]))
        self.assertFalse(torch.equal(fields[1], actual[1]))
        with self.assertRaisesRegex(KeyError, "mean_density"):
            readout.energy(_context(rho, (5, 5, 5)))
        with self.assertRaisesRegex(ValueError, "n_types"):
            readout.energy(_density_context(rho[:, :2]))

    def test_density_temperature_couples_features_keeps_vacuum_and_configuration(self):
        plain, coupled = self.readout(), self.readout(temperature=True)
        self.assertEqual(plain.density_mlp[0].in_features, 2)
        self.assertEqual(coupled.density_mlp[0].in_features, 3)
        self.assertNotIn("density_temperature", plain.to_config())
        config = coupled.to_config()
        self.assertIs(config["density_temperature"], True)
        rebuilt = build(config)
        rebuilt.load_state_dict(coupled.state_dict(), strict=True)
        self.assertEqual(rebuilt.to_config(), config)
        self.assertTrue(rebuilt.density_temperature)

        rho = torch.rand(125, 2, dtype=torch.float64)
        rho[0] = 0.0
        cold, hot = (_density_context(rho, temperature=t) for t in (0.8, 1.3))
        self.assertTrue(torch.equal(plain.density_fields(cold), plain.density_fields(hot)))
        fields_cold, fields_hot = coupled.density_fields(cold), coupled.density_fields(hot)
        self.assertFalse(torch.equal(fields_cold[1:], fields_hot[1:]))
        self.assertTrue(torch.equal(fields_cold[0], torch.zeros(3, dtype=rho.dtype)))
        self.assertTrue(torch.equal(fields_hot[0], torch.zeros(3, dtype=rho.dtype)))
        with self.assertRaisesRegex(KeyError, "mean_density"):
            coupled.energy(_context(rho, (5, 5, 5)))

        # Batched temperatures feed each field its own temperature.
        batch = torch.stack((rho, torch.rand_like(rho)))
        temperatures = torch.tensor([0.8, 1.3], dtype=rho.dtype)
        energies = coupled.energy(_density_context(batch, temperature=temperatures))
        for index in range(2):
            single = coupled.energy(_density_context(
                batch[index], temperature=temperatures[index],
            ))
            torch.testing.assert_close(energies[index], single, rtol=1e-12, atol=1e-12)
        # Temperature matters through the density network alone when the
        # distance network ignores it.
        with torch.no_grad():
            coupled.mlp[0].weight[:, 1].zero_()
        self.assertNotEqual(
            coupled.energy(cold).item(), coupled.energy(hot).item(),
        )

    def test_fft_matches_physical_endpoint_sum_energy_gradient_and_hvp(self):
        for n_types, channels, temperature in ((1, 2, False), (3, 4, False), (2, 2, True)):
            with self.subTest(n_types=n_types, channels=channels, temperature=temperature):
                readout = self.readout(n_types, channels, temperature=temperature)
                rho = torch.rand(125, n_types, dtype=torch.float64, requires_grad=True)
                context = _density_context(rho)
                fft = readout.energy(context)
                direct = _endpoint_energy(readout, _density_context(rho))
                torch.testing.assert_close(fft, direct, rtol=1e-12, atol=1e-12)
                first_fft = torch.autograd.grad(fft, rho, create_graph=True)[0]
                first_direct = torch.autograd.grad(direct, rho, create_graph=True)[0]
                torch.testing.assert_close(first_fft, first_direct, rtol=1e-11, atol=1e-12)
                direction = torch.randn_like(rho)
                second_fft = torch.autograd.grad((first_fft * direction).sum(), rho)[0]
                second_direct = torch.autograd.grad((first_direct * direction).sum(), rho)[0]
                torch.testing.assert_close(second_fft, second_direct, rtol=1e-11, atol=1e-12)

    def test_endpoint_exchange_symmetry(self):
        readout = self.readout(n_types=3, channels=4)
        local = torch.rand(2, 3, dtype=torch.float64)
        coefficients = readout.density_mlp(local).reshape(2, 3, 4)
        values = readout.shell_kernel_values(torch.tensor(1.0, dtype=local.dtype))[0]
        matrix = local.new_zeros(4, 4)
        for value, (first, second) in zip(values, readout.kernel_pairs):
            matrix[first, second] = value
            matrix[second, first] = value
        forward = coefficients[0] @ matrix @ coefficients[1].T
        reverse = coefficients[1] @ matrix @ coefficients[0].T
        torch.testing.assert_close(forward, reverse.T, rtol=1e-12, atol=1e-12)

    def test_zero_initialization_preserves_energy_and_density_derivatives(self):
        readout = self.readout(zero_init=True)
        rho = torch.rand(125, 2, dtype=torch.float64, requires_grad=True)
        context = _density_context(rho)
        energy = readout.energy(context)
        first = torch.autograd.grad(energy, rho, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), rho)[0]
        self.assertEqual(energy.item(), 0.0)
        self.assertTrue(torch.equal(first, torch.zeros_like(first)))
        self.assertTrue(torch.equal(second, torch.zeros_like(second)))
        # The distance output can leave zero initialization on the first step.
        output_grad = torch.autograd.grad(readout.energy(context), readout.mlp[-1].bias)[0]
        self.assertGreater(output_grad.abs().sum().item(), 0.0)

    def test_no_self_pair_retained_zero_mode_and_translation_symmetry(self):
        readout = self.readout()
        rho = torch.full((5, 5, 5, 2), 0.2, dtype=torch.float64)
        uniform = readout.energy(_density_context(rho.reshape(-1, 2)))
        self.assertNotEqual(uniform.item(), 0.0)
        kernel = readout.real_space_kernel(
            torch.tensor(1.0, dtype=rho.dtype), torch.tensor([5, 5, 5]),
        )
        self.assertTrue(torch.equal(
            kernel[:, 0, 0, 0], torch.zeros(readout.n_kernel_pairs, dtype=rho.dtype),
        ))
        rho = torch.rand_like(rho)
        reference = readout.energy(_density_context(rho.reshape(-1, 2)))
        for transformed in (
            torch.roll(rho, (1, -1, 2), (0, 1, 2)),
            rho.flip(0), rho.transpose(0, 2),
        ):
            actual = readout.energy(_density_context(transformed.reshape(-1, 2)))
            torch.testing.assert_close(reference, actual, rtol=1e-12, atol=1e-12)

    def test_batched_fields_and_voxel_volumes_match_separate_evaluation(self):
        readout = self.readout()
        rho = torch.rand(2, 2, 125, 2, dtype=torch.float64)
        temperatures = torch.tensor([[0.8, 1.2], [0.9, 1.1]], dtype=rho.dtype)
        volumes = torch.tensor([[0.4, 0.5], [0.6, 0.7]], dtype=rho.dtype)
        context = _density_context(rho, temperature=temperatures, voxel_volume=volumes)
        batch = readout.energy(context)
        for first in range(2):
            for second in range(2):
                single = readout.energy(_density_context(
                    rho[first, second], temperature=temperatures[first, second],
                    voxel_volume=volumes[first, second],
                ))
                torch.testing.assert_close(batch[first, second], single, rtol=1e-12, atol=1e-12)

    def test_finite_difference_derivatives_at_vacuum_and_positive_density(self):
        readout = self.readout(n_types=1, channels=2)
        rho = torch.rand(4**3, 1, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            rho[0] = 0.0
        energy = lambda value: readout.energy(_density_context(value, (4, 4, 4)))
        self.assertTrue(torch.autograd.gradcheck(energy, (rho,), fast_mode=True))
        self.assertTrue(torch.autograd.gradgradcheck(energy, (rho,), fast_mode=True))

    def test_curvature_loss_backpropagates_through_both_networks(self):
        readout = self.readout()
        rho = torch.rand(125, 2, dtype=torch.float64, requires_grad=True)
        energy = readout.energy(_density_context(rho))
        first = torch.autograd.grad(energy, rho, create_graph=True)[0]
        direction = torch.randn_like(rho)
        hvp = torch.autograd.grad((first * direction).sum(), rho, create_graph=True)[0]
        hvp.square().mean().backward()
        for module in (readout.mlp, readout.density_mlp):
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0.0)

    def test_analytic_density_dependent_c2_includes_endpoint_derivatives(self):
        readout = PairwiseReadout(
            2, 1, hidden_sizes=(), zero_init=False,
            density_channels=1, density_hidden_sizes=(),
        ).double()
        with torch.no_grad():
            readout.density_mlp[0].weight.fill_(1.0)
            readout.density_mlp[0].bias.zero_()
            readout.mlp[0].weight.zero_()
            readout.mlp[0].bias.fill_(1.0)
        # A=rho/scale, f=rho^2/scale. Off-site c2=-4 rho^2 W/scale^2.
        model = GridCACEModel(
            readout=[readout], grid_spacing=0.5, mean_density=0.6,
            boltzmann_constant=1.0, compute_c2=True,
        ).double()
        kernel = readout.real_space_kernel(
            torch.tensor(1.0, dtype=torch.float64), torch.tensor([5, 5, 5]),
        )
        previous = None
        for density in (0.2, 0.4):
            output = model({
                "rho": torch.full((125, 1), density, dtype=torch.float64),
                "temperature": torch.tensor(1.0, dtype=torch.float64),
                "grid_size": torch.tensor([5, 5, 5]),
            }, c2_reference=(0, 0))
            # model.mean_density was initialized in float32 before .double().
            scale = model.mean_density
            expected = -4.0 * density**2 * kernel[0, 1, 0, 0] / scale.square()
            torch.testing.assert_close(output["c2"][25, 0], expected, rtol=1e-12, atol=1e-12)
            self.assertNotEqual(output["c2"][0, 0].item(), 0.0)  # not a self energy
            if previous is not None:
                torch.testing.assert_close(
                    output["c2"][25, 0], 4.0 * previous, rtol=1e-12, atol=1e-12,
                )
            previous = output["c2"][25, 0].detach()

    def quadratic_model(self):
        # Built under a float64 default, so a model file reloads bitwise.
        with default_dtype(torch.float64):
            quadratic = PairwiseReadout(2, 2, hidden_sizes=(5,), zero_init=False)
            model = GridCACEModel(
                readout=[quadratic], grid_spacing=0.5, mean_density=0.6,
                boltzmann_constant=1.0, compute_c2=True,
            ).eval()
        data = {
            "rho": torch.rand(125, 2, dtype=torch.float64) + 0.05,
            "temperature": torch.tensor(1.2, dtype=torch.float64),
            "grid_size": torch.tensor([5, 5, 5]),
        }
        return quadratic, model, data

    def test_warm_start_reproduces_quadratic_readout_and_keeps_channels_trainable(self):
        quadratic, model, data = self.quadratic_model()
        expected = model(copy.deepcopy(data), c2_reference=(0, 0))
        for channels, temperature in ((2, False), (2, True), (4, False), (5, True)):
            with self.subTest(channels=channels, temperature=temperature):
                readout = self.readout(2, channels, zero_init=True, temperature=temperature)
                self.assertIs(readout.warm_start_from(quadratic), readout)
                if channels == 2:
                    for key, value in quadratic.mlp.state_dict().items():
                        self.assertTrue(torch.equal(readout.mlp.state_dict()[key], value), key)
                # Sources equal physical rho at any density and temperature.
                for value in (0.7, 1.4):
                    context = _density_context(data["rho"], temperature=value)
                    self.assertTrue(torch.equal(
                        readout.density_fields(context)[:, :2], data["rho"],
                    ))
                # Physical pairs keep their kernels; pairs with an extra
                # feature are zero.
                kernel, reference = (
                    module.real_space_kernel(
                        torch.tensor(1.2, dtype=torch.float64), data["grid_size"],
                    )
                    for module in (readout, quadratic)
                )
                index = {pair: i for i, pair in enumerate(readout.kernel_pairs)}
                for position, pair in enumerate(quadratic.type_pairs):
                    # A wider output layer can change matrix-product roundoff.
                    torch.testing.assert_close(
                        kernel[index[pair]], reference[position],
                        rtol=1e-13, atol=1e-13,
                    )
                extra = [i for pair, i in index.items() if pair not in quadratic.type_pairs]
                self.assertEqual(len(extra), readout.n_kernel_pairs - 3)
                self.assertTrue(torch.equal(kernel[extra], torch.zeros_like(kernel[extra])))
                with default_dtype(torch.float64):
                    warm = GridCACEModel(
                        readout=[readout], grid_spacing=0.5, mean_density=0.6,
                        boltzmann_constant=1.0, compute_c2=True,
                    ).eval()
                actual = warm(copy.deepcopy(data), c2_reference=(0, 0))
                for key in ("beta_F_exc", "c1", "c2"):
                    torch.testing.assert_close(
                        actual[key], expected[key], rtol=1e-13, atol=1e-13,
                    )
                # Neither the extra kernels nor the identity features are
                # stuck: the trained kernel moves the physical features and
                # the random extra features move their kernels.
                energy = readout.energy(_density_context(data["rho"], temperature=1.2))
                final = readout.mlp[-1]
                kernel_grad, bias_grad = torch.autograd.grad(
                    energy, (final.weight, readout.density_mlp[-1].bias),
                )
                self.assertTrue((kernel_grad != 0).all())
                self.assertTrue((bias_grad.view(2, channels)[:, :2] != 0).all())
                # One plain gradient step then reaches every parameter.
                optimizer = torch.optim.SGD(readout.parameters(), lr=1e-3)
                readout.energy(_density_context(data["rho"], temperature=1.2)).backward()
                optimizer.step()
                optimizer.zero_grad()
                readout.energy(_density_context(data["rho"], temperature=1.2)).backward()
                for name, parameter in readout.named_parameters():
                    self.assertTrue((parameter.grad != 0).any(), name)

    def test_from_quadratic_replaces_a_model_readout_and_round_trips(self):
        quadratic, model, data = self.quadratic_model()
        expected = model(copy.deepcopy(data), c2_reference=(0, 0))
        config = model.to_config()
        readout = PairwiseReadout.from_quadratic(
            quadratic, 3, density_hidden_sizes=(4,), density_temperature=True,
        )
        self.assertEqual(readout.to_config(), dict(
            quadratic.to_config(), density_channels=3,
            density_hidden_sizes=[4], density_temperature=True,
        ))
        self.assertEqual(readout.mlp[-1].weight.dtype, torch.float64)
        model.readout[0] = readout
        self.assertEqual(model.to_config()["readout"][0], readout.to_config())
        self.assertNotEqual(model.to_config(), config)
        actual = model(copy.deepcopy(data), c2_reference=(0, 0))
        for key in ("beta_F_exc", "c1", "c2"):
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-13, atol=1e-13)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            loaded = load_model(path)
        self.assertEqual(loaded.to_config(), model.to_config())
        reloaded = loaded(copy.deepcopy(data), c2_reference=(0, 0))
        for key in ("beta_F_exc", "c1", "c2"):
            self.assertTrue(torch.equal(reloaded[key], actual[key]), key)
        # The new readout follows the source's floating dtype.
        single = PairwiseReadout.from_quadratic(quadratic.float(), 2)
        self.assertEqual(single.density_mlp[-1].bias.dtype, torch.float32)

    def test_warm_start_rejects_mismatched_readouts(self):
        quadratic = PairwiseReadout(2, 2, hidden_sizes=(5,))
        with self.assertRaisesRegex(ValueError, "at least n_types"):
            PairwiseReadout(2, 2, hidden_sizes=(5,), density_channels=1).warm_start_from(quadratic)
        with self.assertRaisesRegex(ValueError, "requires density_channels"):
            PairwiseReadout(2, 2, hidden_sizes=(5,)).warm_start_from(quadratic)
        with self.assertRaisesRegex(ValueError, "source has density_channels"):
            self.readout().warm_start_from(self.readout())
        with self.assertRaises(TypeError):
            self.readout().warm_start_from(quadratic.mlp)
        for kwargs, name in (
            ({"cutoff_grid": 3}, "cutoff_grid"),
            ({"n_types": 3, "density_channels": 3}, "n_types"),
            ({"hidden_sizes": (6,)}, "hidden_sizes"),
        ):
            arguments = dict(
                cutoff_grid=2, n_types=2, hidden_sizes=(5,), density_channels=2,
            )
            arguments.update(kwargs)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                PairwiseReadout(**arguments).warm_start_from(quadratic)
        with self.assertRaises(TypeError):
            PairwiseReadout.from_quadratic(quadratic.mlp, 2)

    def test_model_file_and_strict_state_round_trip(self):
        readout = PairwiseReadout(2, 2, density_channels=3, zero_init=False)
        model = GridCACEModel(
            readout=[readout], grid_spacing=0.5, mean_density=0.6,
            boltzmann_constant=1.0, compute_c2=True,
        ).eval()
        data = {
            "rho": torch.rand(125, 2) + 0.1,
            "temperature": torch.tensor(1.0), "grid_size": torch.tensor([5, 5, 5]),
        }
        config = json.loads(json.dumps(model.to_config()))
        rebuilt = build(config)
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        rebuilt.eval()  # load_model returns evaluation mode too.
        self.assertEqual(rebuilt.to_config(), config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            loaded = load_model(path)
        expected = model(copy.deepcopy(data), c2_reference=(0, 0))
        for other in (rebuilt, loaded):
            actual = other(copy.deepcopy(data), c2_reference=(0, 0))
            for key in ("beta_F_exc", "c1", "c2"):
                self.assertTrue(torch.equal(actual[key], expected[key]), key)
        self.assertEqual(loaded.readout[0].density_channels, 3)

    def test_trainer_checkpoint_restores_both_networks_and_adam(self):
        model = GridCACEModel(
            readout=[PairwiseReadout(2, 2, density_channels=3, zero_init=False)],
            grid_spacing=0.5, mean_density=0.6,
            boltzmann_constant=1.0, compute_local_mu=False,
        )
        loader = DataLoader([{
            "rho": torch.rand(64, 2) + 0.1,
            "temperature": torch.tensor(1.0), "grid_size": torch.tensor([4, 4, 4]),
            "target_c1": torch.zeros(64, 2),
        }], batch_size=1)

        def trainer_for(module, directory=None):
            return Trainer(
                module, Loss([TensorLoss("field", "c1", "target_c1")]),
                optimizer_args={"lr": 1e-4}, checkpoint_dir=directory,
            )

        with tempfile.TemporaryDirectory() as directory:
            original = trainer_for(model, directory)
            original.fit(loader, loader, epochs=1, verbose=False)
            resumed = trainer_for(build(model.to_config()))
            epoch = resumed.load_checkpoint(Path(directory) / "last.pt", loader)
            self.assertEqual(epoch, 1)
            self.assertEqual(len(resumed.optimizer.state), len(tuple(model.parameters())))
            original.fit(loader, loader, epochs=1, verbose=False)
            resumed.fit(loader, loader, epochs=1, verbose=False)
        for key, expected in original.model.state_dict().items():
            self.assertTrue(torch.equal(resumed.model.state_dict()[key], expected), key)


if __name__ == "__main__":
    unittest.main()
