"""Local density-dependent Gaussian kernels, with k=0 excluded."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from equicdft import (
    GridCACEModel, LongRangeReadout, ReciprocalFeatures, load_model, save_model,
)
from equicdft._config import build
from equicdft.functional.base import ReadoutContext
from equicdft.serialization import default_dtype


def _context(rho, shape=(3, 3, 3), spacing=(0.5, 0.6, 0.7), temperature=1.1):
    spacing = rho.new_tensor(spacing)
    return ReadoutContext(
        rho=rho, mean_density=0.6,
        normalized_temperature=torch.as_tensor(
            temperature, dtype=rho.dtype, device=rho.device,
        ).expand(rho.shape[:-2]),
        grid_size=torch.tensor(shape, device=rho.device),
        grid_spacing=spacing, voxel_volume=spacing.prod(),
    )


def _direct_energy(readout, context):
    """Explicit DFT and symmetric matrices, independent of FFT contraction."""
    rho = context.rho
    shape = context.grid_size.tolist()
    axes = [torch.arange(n, dtype=rho.dtype) for n in shape]
    positions = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    positions = positions * context.grid_spacing
    axes = [
        2.0 * torch.pi * torch.fft.fftfreq(n, d=float(h), dtype=rho.dtype)
        for n, h in zip(shape, context.grid_spacing)
    ]
    wavevectors = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    phase = wavevectors @ positions.T
    transform = torch.complex(torch.cos(phase), -torch.sin(phase))
    inputs = context.pointwise_state() if readout.density_temperature else context.normalized_rho()
    mapping = readout.density_mlp(inputs).reshape(*rho.shape, readout.density_channels)
    fields = (rho[..., None] * mapping).sum(-2)
    source = context.voxel_volume * transform @ fields.to(transform.dtype)
    coefficients = readout.coefficients(context.normalized_temperature[None])
    pairs = [(i, j) for i in range(readout.n_kernel_channels) for j in range(i, readout.n_kernel_channels)]
    indices = {pair: index for index, pair in enumerate(pairs)}
    matrices = torch.stack([
        torch.stack([
            coefficients[:, indices[min(i, j), max(i, j)]]
            for j in range(readout.n_kernel_channels)
        ], -1)
        for i in range(readout.n_kernel_channels)
    ], -2)
    k2 = wavevectors.square().sum(-1)
    kernels = torch.exp(-readout.features.radial_exponents[:, None] * k2)
    kernels = torch.where(k2 > 0, kernels, torch.zeros_like(kernels))
    weights = torch.einsum("nk,nab->kab", kernels, matrices).to(source.dtype)
    volume = context.voxel_volume * rho.shape[-2]
    return torch.einsum("ka,kab,kb->", source.conj(), weights, source).real / (2 * volume)


class TestDensityDependentGaussian(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)

    def readout(self, n_types=2, channels=3, **kwargs):
        return LongRangeReadout(
            features=ReciprocalFeatures((0.08, 0.2), n_types=n_types),
            hidden_sizes=(5,), zero_init=False, density_channels=channels,
            density_hidden_sizes=(4,), **kwargs,
        ).double()

    def test_options_reject_non_density_gaussian_modes(self):
        for options in (
            {}, {"features": ReciprocalFeatures(kernel="coulomb")},
            {"features": ReciprocalFeatures(variable="dipole_density")},
            {"features": ReciprocalFeatures((0.2,)), "charges": [1.]},
        ):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "Gaussian features"):
                LongRangeReadout(n_kernels=1, density_channels=2, **options)
        for value, error in ((0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)):
            with self.subTest(value=value), self.assertRaises(error):
                self.readout(channels=value)
        with self.assertRaisesRegex(ValueError, "density_temperature requires"):
            LongRangeReadout(features=ReciprocalFeatures(), density_temperature=True)
        with self.assertRaises(TypeError):
            self.readout(density_temperature=1)
        with self.assertRaises(ValueError):
            LongRangeReadout(features=ReciprocalFeatures(), density_channels=2, density_hidden_sizes=(0,))

    def test_legacy_config_state_rng_and_contraction_are_unchanged(self):
        options = dict(features=ReciprocalFeatures((0.2,), n_types=2), hidden_sizes=(4,), zero_init=False)
        torch.manual_seed(42)
        original = LongRangeReadout(**options)
        state_after_original = torch.get_rng_state()
        torch.manual_seed(42)
        explicit = LongRangeReadout(**options, density_channels=None)
        self.assertTrue(torch.equal(state_after_original, torch.get_rng_state()))
        self.assertNotIn("density_channels", original.to_config())
        self.assertEqual(original.to_config(), explicit.to_config())
        self.assertFalse(any(key.startswith("density_mlp") for key in original.state_dict()))
        rebuilt = build(original.to_config())
        rebuilt.load_state_dict(original.state_dict(), strict=True)
        rho = torch.rand(27, 2)
        context = _context(rho)
        self.assertIs(original.density_fields(context), rho)
        expected = original(original.features(rho, context.grid_size, context.grid_spacing), context.state_features())
        for module in (original, explicit, rebuilt):
            self.assertTrue(torch.equal(module.energy(context), expected))

    def test_explicit_dft_energy_density_temperature_gradients_and_hvp(self):
        readout = self.readout(density_temperature=True)
        rho = (torch.rand(27, 2, dtype=torch.float64) + 0.1).requires_grad_()
        temperature = torch.tensor(1.2, dtype=rho.dtype, requires_grad=True)
        context = _context(rho, temperature=temperature)
        actual, expected = readout.energy(context), _direct_energy(readout, context)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        gradients = [torch.autograd.grad(energy, (rho, temperature), create_graph=True) for energy in (actual, expected)]
        direction = torch.randn_like(rho)
        for first, second in zip(*gradients):
            torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-12)
        hvps = [
            torch.autograd.grad(
                (grad[0] * direction).sum() + 0.3 * grad[1],
                (rho, temperature), retain_graph=True,
            ) for grad in gradients
        ]
        for first, second in zip(*hvps):
            torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-12)

    def test_local_sources_use_every_species_and_vanish_in_vacuum(self):
        readout = self.readout(n_types=3, channels=2)
        rho = torch.rand(27, 3, dtype=torch.float64)
        rho[0] = 0.0
        original = readout.density_fields(_context(rho))
        changed = rho.clone()
        changed[1, 2] += 0.2
        actual = readout.density_fields(_context(changed))
        self.assertEqual(original.shape, (27, 2))
        self.assertTrue(torch.equal(original[0], torch.zeros_like(original[0])))
        self.assertFalse(torch.equal(original[1], actual[1]))
        self.assertTrue(torch.equal(original[2:], actual[2:]))
        with self.assertRaisesRegex(ValueError, "n_types"):
            readout.energy(_context(rho[:, :2]))
        context = _context(rho)
        context.mean_density = None
        with self.assertRaisesRegex(KeyError, "mean_density"):
            readout.energy(context)

    def test_physical_species_and_latent_channel_counts_are_independent(self):
        for n_types, channels in ((1, 1), (3, 2), (4, 5)):
            with self.subTest(n_types=n_types, channels=channels):
                readout = self.readout(n_types=n_types, channels=channels)
                context = _context(torch.rand(27, n_types, dtype=torch.float64))
                torch.testing.assert_close(
                    readout.energy(context), _direct_energy(readout, context),
                    rtol=1e-12, atol=1e-12,
                )

    def test_uniform_energy_c1_and_uniform_direction_curvature_are_zero(self):
        readout = self.readout(density_temperature=True)
        for densities in ((0., 0.), (0.2, 0.4), (0.9, 0.1)):
            rho = torch.tensor(densities, dtype=torch.float64).expand(27, 2).clone().requires_grad_()
            context = _context(rho)
            energy = readout.energy(context)
            gradient = torch.autograd.grad(energy, rho, create_graph=True)[0]
            uniform_hvp = torch.autograd.grad(gradient.sum(), rho)[0]
            torch.testing.assert_close(energy, torch.zeros_like(energy), rtol=0, atol=1e-28)
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=1e-25)
            torch.testing.assert_close(uniform_hvp, torch.zeros_like(uniform_hvp), rtol=0, atol=1e-14)
            kernels = readout.features._kernel_values((3, 3, 3), context.grid_spacing, rho.device, rho.dtype)
            self.assertTrue(torch.equal(kernels[:, 0, 0, 0], torch.zeros(2, dtype=rho.dtype)))

    def test_homogeneous_finite_k_curvature_matches_local_jacobian(self):
        readout = self.readout()
        shape = (5, 3, 3)
        grid = torch.arange(5, dtype=torch.float64)[:, None, None].expand(shape).reshape(-1)
        mode = torch.cos(2 * torch.pi * grid / 5)
        direction = torch.tensor([0.7, -0.3], dtype=torch.float64)
        for densities in ((0.2, 0.3), (0.5, 0.1)):
            base = torch.tensor(densities, dtype=torch.float64)
            amplitude = torch.zeros((), dtype=torch.float64, requires_grad=True)
            rho = base + amplitude * mode[:, None] * direction
            context = _context(rho, shape=shape)
            energy = readout.energy(context)
            first = torch.autograd.grad(energy, amplitude, create_graph=True)[0]
            actual = torch.autograd.grad(first, amplitude, create_graph=True)[0]
            jacobian = torch.autograd.functional.jacobian(
                lambda value: readout.density_fields(_context(value[None]))[0], base,
            )
            response = jacobian @ direction
            coefficients = readout.coefficients(context.normalized_temperature[None])
            pairs = [(i, j) for i in range(3) for j in range(i, 3)]
            pair_response = torch.stack([
                response[i] * response[j] * (1 if i == j else 2) for i, j in pairs
            ])
            k2 = (2 * torch.pi / (5 * context.grid_spacing[0]))**2
            kernels = torch.exp(-readout.features.radial_exponents * k2)
            expected = context.voxel_volume * mode.square().sum() * (kernels[:, None] * coefficients * pair_response).sum()
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            gradients = torch.autograd.grad(actual.square(), tuple(readout.parameters()))
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients[-4:]))

    def test_zero_init_is_zero_but_coefficient_head_can_learn(self):
        readout = LongRangeReadout(features=ReciprocalFeatures(n_types=2), density_channels=3).double()
        rho = torch.rand(27, 2, dtype=torch.float64, requires_grad=True)
        energy = readout.energy(_context(rho))
        gradient = torch.autograd.grad(energy, rho, create_graph=True)[0]
        hvp = torch.autograd.grad(gradient.sum(), rho)[0]
        self.assertEqual(energy.item(), 0.)
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))
        self.assertTrue(torch.equal(hvp, torch.zeros_like(hvp)))
        readout.energy(_context(rho)).backward()
        self.assertGreater(readout.mlp[-1].bias.grad.abs().sum().item(), 0.)

    def test_temperature_only_coefficients_and_optional_local_temperature(self):
        for coupled in (False, True):
            readout = self.readout(density_temperature=coupled)
            rho = torch.rand(27, 2, dtype=torch.float64)
            cold, hot = (_context(rho, temperature=t) for t in (0.8, 1.3))
            self.assertEqual(readout.mlp[0].in_features, 1)
            self.assertEqual(readout.density_mlp[0].in_features, 2 + int(coupled))
            other = _context(rho * 2, temperature=0.8)
            self.assertTrue(torch.equal(readout._state_input(cold), readout._state_input(other)))
            self.assertEqual(torch.equal(readout.density_fields(cold), readout.density_fields(hot)), not coupled)
            with self.assertRaisesRegex(ValueError, "state_features"):
                readout.coefficients(torch.ones(3, dtype=rho.dtype))

    def test_batch_translation_and_volume_normalization(self):
        readout = self.readout(density_temperature=True)
        batch = torch.rand(2, 27, 2, dtype=torch.float64)
        temperatures = torch.tensor([0.8, 1.3], dtype=batch.dtype)
        actual = readout.energy(_context(batch, temperature=temperatures))
        for index in range(2):
            context = _context(batch[index], temperature=temperatures[index])
            torch.testing.assert_close(actual[index], _direct_energy(readout, context), rtol=1e-12, atol=1e-12)
            shifted = torch.roll(batch[index].reshape(3, 3, 3, 2), 1, 0).reshape(27, 2)
            torch.testing.assert_close(actual[index], readout.energy(_context(shifted, temperature=temperatures[index])), rtol=1e-12, atol=1e-12)
            rotated = batch[index].reshape(3, 3, 3, 2).transpose(0, 1).reshape(27, 2)
            torch.testing.assert_close(
                actual[index], readout.energy(_context(
                    rotated, spacing=(0.6, 0.5, 0.7), temperature=temperatures[index],
                )), rtol=1e-12, atol=1e-12,
            )
        rho = batch[0].reshape(3, 3, 3, 2).repeat(2, 1, 1, 1).reshape(54, 2)
        doubled = readout.energy(_context(rho, shape=(6, 3, 3), temperature=temperatures[0]))
        torch.testing.assert_close(doubled, 2 * actual[0], rtol=1e-12, atol=1e-12)

    def test_scaled_initial_coefficients_use_latent_pair_count(self):
        readout = self.readout(coefficient_scale=[2., 3.], initial_coefficients=[0.4, 0.7])
        state = torch.tensor([1.1], dtype=torch.float64)
        expected = readout.mlp(state).reshape(2, 6) * state.new_tensor([2., 3.])[:, None] + state.new_tensor([0.4, 0.7])[:, None]
        torch.testing.assert_close(readout.coefficients(state), expected)
        self.assertEqual(readout.n_type_pairs, 3)
        self.assertEqual(readout.n_kernel_pairs, 6)
        with self.assertRaisesRegex(ValueError, "source pair"):
            self.readout(initial_coefficients=torch.ones(2, 3))

    def test_model_checkpoint_c1_c2_and_optimizer_round_trip(self):
        with default_dtype(torch.float64):
            readout = self.readout(density_temperature=True)
            model = GridCACEModel(
                readout=[readout], grid_spacing=0.5, mean_density=0.6,
                boltzmann_constant=1., compute_c2=True,
            )
        data = {"rho": torch.rand(27, 2, dtype=torch.float64) + 0.1, "grid_size": torch.tensor([3, 3, 3]), "temperature": torch.tensor(1.2, dtype=torch.float64)}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model(copy.deepcopy(data), c2_reference=(0, 0))["c2"].square().sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        expected = model(copy.deepcopy(data), c2_reference=(0, 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            loaded = load_model(path)
        self.assertEqual(loaded.to_config(), model.to_config())
        with default_dtype(torch.float64):
            rebuilt = build(model.to_config())
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        for candidate in (loaded, rebuilt):
            candidate.eval()
            actual = candidate(copy.deepcopy(data), c2_reference=(0, 0))
            for key in ("beta_F_exc", "c1", "c2"):
                self.assertTrue(torch.equal(actual[key], expected[key]), key)
        continued = torch.optim.Adam(loaded.parameters(), lr=1e-3)
        continued.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for candidate, opt in ((model, optimizer), (loaded, continued)):
            candidate.train()
            candidate(copy.deepcopy(data), c2_reference=(0, 0))["c2"].square().sum().backward()
            opt.step()
        for first, second in zip(model.parameters(), loaded.parameters()):
            self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
