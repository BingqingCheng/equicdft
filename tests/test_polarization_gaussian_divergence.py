"""Gaussian A|P|^2 + B|div P|^2 features and their variational responses."""

import io
import math
import unittest

import torch

from equicdft import GridCACEModel, LDAReadout, LongRangeReadout, ReciprocalFeatures


class TestPolarizationGaussianDivergence(unittest.TestCase):
    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        self.rng_state = torch.random.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(91)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)
        torch.random.set_rng_state(self.rng_state)

    @staticmethod
    def data(shape=(5, 3, 3), n_types=1, spacing=.8):
        return {
            "rho": .4 + .05 * torch.rand(math.prod(shape), n_types),
            "dipole_density": .02 * torch.randn(math.prod(shape), n_types, 3),
            "grid_size": torch.tensor(shape),
            "grid_spacing": torch.full((3,), spacing),
            "temperature": torch.tensor(1.),
        }

    @staticmethod
    def clone(data):
        return {key: value.detach().clone() for key, value in data.items()}

    @staticmethod
    def features(exponents=(.2, .7), n_types=1, include_divergence=True):
        return ReciprocalFeatures(
            exponents, n_types=n_types, variable="dipole_density",
            include_divergence=include_divergence,
        )

    @staticmethod
    def evaluate(features, data):
        return features(
            data["rho"], data["grid_size"], data["grid_spacing"],
            dipole_density=data["dipole_density"],
        )

    def readout(self, coefficients=(1.2, -.3, .4, -.1), **kwargs):
        features = self.features(**kwargs)
        readout = LongRangeReadout(
            hidden_sizes=(), features=features,
        )
        with torch.no_grad():
            readout.mlp[-1].weight.zero_()
            readout.mlp[-1].bias.copy_(torch.tensor(coefficients))
        return readout

    @staticmethod
    def model(readout, spacing=.8, **kwargs):
        lda = LDAReadout(mean_density=.4, n_types=readout.n_types, zero_init=True)
        return GridCACEModel(
            None, None, [lda, readout], grid_spacing=spacing,
            compute_polarization_derivative=True, **kwargs,
        )

    def test_uniform_has_only_a_energy_and_derivative(self):
        for spacing in (.5, 2.):
            data = self.data(spacing=spacing)
            data["dipole_density"][:] = torch.tensor([.03, -.02, .01])
            features = self.features()
            actual = self.evaluate(features, data)
            volume = data["rho"].shape[0] * spacing**3
            norm = data["dipole_density"][0].square().sum()
            expected = torch.tensor([1., 1., 0., 0.])[:, None] * volume * norm / 2
            self.assertEqual(features.n_kernels, 4)
            torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
            output = self.model(self.readout(), spacing).eval()(self.clone(data))
            torch.testing.assert_close(output["beta_F_exc"], .9 * volume * norm / 2, atol=1e-13, rtol=1e-12)
            torch.testing.assert_close(output["polarization_derivative"], .9 * data["dipole_density"], atol=1e-13, rtol=1e-12)

    def test_mixed_wavevector_longitudinal_and_two_transverse_modes(self):
        shape, amplitude = (7, 5, 3), .03
        indices = torch.stack(torch.meshgrid(
            *(torch.arange(n) for n in shape), indexing="ij",
        ), -1).reshape(-1, 3)
        mode = torch.tensor([1., 1., 0.])
        phase = 2 * math.pi * (indices * mode / torch.tensor(shape)).sum(-1)
        for spacing in (.5, 2.):
            k = 2 * math.pi * mode / (torch.tensor(shape) * spacing)
            k2 = k.square().sum()
            longitudinal = k / k.norm()
            directions = (
                longitudinal,
                torch.tensor([-longitudinal[1], longitudinal[0], 0.]),
                torch.tensor([0., 0., 1.]),
            )
            damping = torch.exp(-torch.tensor([.2, .7]) * k2)
            a = torch.dot(torch.tensor([1.2, -.3]), damping)
            b = torch.dot(torch.tensor([.4, -.1]), damping)
            for direction in directions:
                for wave in (phase.cos(), phase.sin()):
                    data = self.data(shape, spacing=spacing)
                    data["dipole_density"] = amplitude * wave[:, None, None] * direction
                    features = self.evaluate(self.features(), data)
                    prefactor = math.prod(shape) * spacing**3 * amplitude**2 / 4
                    expected = prefactor * torch.cat((damping, damping * torch.dot(k, direction).square()))[:, None]
                    torch.testing.assert_close(features, expected, atol=1e-13, rtol=1e-12)
                    output = self.model(self.readout(), spacing).eval()(self.clone(data))
                    response = a * direction + b * k * torch.dot(k, direction)
                    expected_derivative = amplitude * wave[:, None, None] * response
                    torch.testing.assert_close(output["polarization_derivative"], expected_derivative, atol=1e-13, rtol=1e-12)
                    expected_energy = prefactor * (a + b * torch.dot(k, direction).square())
                    torch.testing.assert_close(output["beta_F_exc"], expected_energy, atol=1e-13, rtol=1e-12)

    def test_long_wavelength_difference_scales_as_k_squared(self):
        # Gaussian factors cancel in B_n / A_n, with no hidden alpha factor.
        shape = (9, 1, 1)
        wave = (2 * math.pi * torch.arange(9) / 9).cos()
        for spacing in (.7, 1.4):
            data = self.data(shape, spacing=spacing)
            data["dipole_density"].zero_()
            data["dipole_density"][:, 0, 0] = .03 * wave
            values = self.evaluate(self.features(), data)[:, 0]
            expected = torch.full((2,), (2 * math.pi / (9 * spacing))**2)
            torch.testing.assert_close(values[2:] / values[:2], expected, atol=1e-13, rtol=1e-12)

    def test_nyquist_derivative_zero_and_mixed_mode_convention(self):
        shape, spacing, amplitude = (4, 5, 1), .8, .03
        checker = (-1.) ** torch.arange(shape[0])
        cosine = (2 * math.pi * torch.arange(shape[1]) / shape[1]).cos()
        data = self.data(shape, spacing=spacing)
        for wave, direction, surviving_k2 in (
            (checker[:, None].expand(4, 5), (1., 0., 0.), 0.),
            (checker[:, None] * cosine, (1., 0., 0.), 0.),
            (checker[:, None] * cosine, (0., 1., 0.), (2 * math.pi / (5 * spacing))**2),
        ):
            data["dipole_density"] = amplitude * wave.reshape(-1, 1, 1) * torch.tensor(direction)
            values = self.evaluate(self.features(), data)[:, 0]
            self.assertTrue(torch.all(values[:2] > 0))
            torch.testing.assert_close(values[2:], surviving_k2 * values[:2], atol=1e-13, rtol=1e-12)

    def test_explicit_dft_batched_anisotropic_multispecies(self):
        # This reference constructs the full Fourier matrix independently.
        shape = (4, 3, 2)
        spacing = torch.tensor([.6, .8, 1.1])
        rho = .4 + .05 * torch.rand(2, 1, math.prod(shape), 2)
        p = (.02 * torch.randn(*rho.shape, 3)).requires_grad_()
        actual = self.features(n_types=2)(rho, torch.tensor(shape), spacing, dipole_density=p)
        positions = torch.stack(torch.meshgrid(
            *(torch.arange(n) * spacing[i] for i, n in enumerate(shape)), indexing="ij",
        ), -1).reshape(-1, 3)
        k_axes, derivative_axes = [], []
        for i, n in enumerate(shape):
            indices = torch.arange(n)
            signed = torch.where(indices <= (n - 1) // 2, indices, indices - n)
            axis = signed * (2 * math.pi / (n * spacing[i]))
            derivative = axis.clone()
            if n % 2 == 0:
                derivative[n // 2] = 0.
            k_axes.append(axis)
            derivative_axes.append(derivative)
        k = torch.stack(torch.meshgrid(*k_axes, indexing="ij"), -1).reshape(-1, 3)
        kd = torch.stack(torch.meshgrid(*derivative_axes, indexing="ij"), -1).reshape(-1, 3)
        phase = torch.exp(-1j * (positions @ k.T))
        p_hat = spacing.prod() * torch.einsum("...gac,gk->...kac", p.to(phase.dtype), phase)
        divergence_hat = -1j * (p_hat * kd[:, None, :]).sum(-1)
        kernels = torch.exp(-torch.tensor([.2, .7])[:, None] * k.square().sum(-1))
        volume = math.prod(shape) * spacing.prod()
        expected = []
        for a, b in ((0, 0), (0, 1), (1, 1)):
            vector_power = (p_hat[..., a, :].conj() * p_hat[..., b, :]).real.sum(-1)
            divergence_power = (divergence_hat[..., a].conj() * divergence_hat[..., b]).real
            contracted = torch.cat([
                torch.einsum("nk,...k->...n", kernels, power)
                for power in (vector_power, divergence_power)
            ], dim=-1)
            expected.append((1. if a == b else 2.) * contracted / (2 * volume))
        expected = torch.stack(expected, -1)
        self.assertEqual(actual.shape, (2, 1, 4, 3))
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
        actual_gradient = torch.autograd.grad(actual.sum(), p, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected.sum(), p)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-13, rtol=1e-12)

    def test_b_zero_recovers_model_energy_gradient_and_hessian(self):
        baseline = self.readout(coefficients=(1.2, -.3), include_divergence=False)
        extended = self.readout(coefficients=(1.2, -.3, 0., 0.))
        with torch.no_grad():
            baseline.mlp[-1].weight.copy_(torch.tensor([[.2, .6], [-.1, .4]]))
            extended.mlp[-1].weight[:2].copy_(baseline.mlp[-1].weight)
        data = self.data()
        inputs = [self.clone(data), self.clone(data)]
        outputs = [self.model(readout).train()(item) for readout, item in zip((baseline, extended), inputs)]
        for key in ("beta_F_exc", "c1", "polarization_derivative"):
            torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=1e-13, rtol=1e-12)
        rho_probe, p_probe = torch.randn_like(data["rho"]), torch.randn_like(data["dipole_density"])
        second_derivatives = []
        for output, item in zip(outputs, inputs):
            probe = (output["c1"] * rho_probe).sum() + (output["polarization_derivative"] * p_probe).sum()
            second_derivatives.append(torch.autograd.grad(probe, (item["rho"], item["dipole_density"])))
        for left, right in zip(*second_derivatives):
            torch.testing.assert_close(left, right, atol=1e-13, rtol=1e-12)

    def test_zero_and_near_zero_p_gradcheck_and_gradgradcheck(self):
        data = self.data((3, 1, 1), n_types=2)
        features = self.features(n_types=2)
        coefficients = torch.tensor([[1., -.2, .4], [-.3, .7, .5], [.6, -.1, .8], [.9, .2, -.4]])

        def energy(p):
            return (self.evaluate(features, dict(data, dipole_density=p)) * coefficients).sum()

        for amplitude in (0., 1e-8):
            p = (amplitude * data["dipole_density"]).requires_grad_()
            self.assertTrue(torch.autograd.gradcheck(energy, (p,), atol=2e-8, rtol=2e-6))
            self.assertTrue(torch.autograd.gradgradcheck(energy, (p,), atol=2e-8, rtol=2e-6))

    def test_joint_field_loss_trains_a_b_and_state_dependence(self):
        readout = self.readout(coefficients=(0., 0., 0., 0.))
        data = self.data()
        model = self.model(readout).train()
        output = model(data)
        target = data["dipole_density"].detach()
        loss = output["c1"].square().mean() + (output["polarization_derivative"] - target).square().mean()
        loss.backward()
        for parameter in readout.parameters():
            self.assertTrue(parameter.grad is not None and torch.isfinite(parameter.grad).all())
            self.assertTrue(torch.all(parameter.grad.abs() > 1e-10))
        with torch.no_grad():
            readout.mlp[-1].weight.copy_(torch.tensor([[.2, .6], [-.1, .4], [.3, -.2], [.1, .5]]))
        data = self.clone(data)
        output = model(data)
        plus, minus = self.clone(data), self.clone(data)
        plus["rho"][4, 0] += 1e-6
        minus["rho"][4, 0] -= 1e-6
        finite_difference = (model(plus)["beta_F_exc"] - model(minus)["beta_F_exc"]) / (2e-6 * .8**3)
        torch.testing.assert_close(finite_difference, -output["c1"][4, 0], atol=2e-9, rtol=2e-7)
        left = torch.autograd.grad(output["c1"][4, 0], data["dipole_density"], retain_graph=True)[0][5, 0, 2]
        right = torch.autograd.grad(output["polarization_derivative"][5, 0, 2], data["rho"])[0][4, 0]
        self.assertGreater(float(left.abs()), 1e-8)
        torch.testing.assert_close(left, -right, atol=1e-12, rtol=1e-12)

    def test_default_and_legacy_features_unchanged(self):
        data = self.data()
        args = (data["rho"], data["grid_size"], data["grid_spacing"])
        for variable in ("rho", "dipole_density"):
            default = ReciprocalFeatures((.2, .7), variable=variable)
            explicit = ReciprocalFeatures((.2, .7), variable=variable, include_divergence=False)
            kwargs = {"dipole_density": data["dipole_density"]} if variable == "dipole_density" else {}
            expected = explicit(*args, **kwargs)
            self.assertEqual(default.n_kernels, 2)
            torch.testing.assert_close(default(*args, **kwargs), expected, atol=0., rtol=0.)
            del default.include_divergence
            buffer = io.BytesIO()
            torch.save(default, buffer)
            buffer.seek(0)
            legacy = torch.load(buffer, weights_only=False)
            torch.testing.assert_close(legacy(*args, **kwargs), expected, atol=0., rtol=0.)

    def test_invalid_divergence_contracts(self):
        invalid_modes = (
            {},
            {"kernel": "coulomb"},
            {"variable": "dipole_density", "kernel": "coulomb"},
            {"variable": "dipole_density", "kernel": "screened_inverse_laplacian"},
        )
        for kwargs in invalid_modes:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ReciprocalFeatures(include_divergence=True, **kwargs)
        for flag in (None, 1, "yes"):
            with self.subTest(flag=flag), self.assertRaises(TypeError):
                self.features(include_divergence=flag)
        features = self.features()
        with self.assertRaises(ValueError):
            LongRangeReadout(2, features=features)
        invalid_readouts = (
            {"charges": (0.,)},
            {"coulomb_amplitude": 1.},
            {"include_polarization": True},
        )
        for kwargs in invalid_readouts:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LongRangeReadout(features.n_kernels, features=features, **kwargs)
        data = self.data()
        invalid_fields = (
            None,
            data["dipole_density"][..., 0],
            data["dipole_density"].float(),
            data["dipole_density"] * float("nan"),
        )
        for index, p in enumerate(invalid_fields):
            with self.subTest(field=index), self.assertRaises(ValueError):
                self.evaluate(features, dict(data, dipole_density=p))

    def test_divergence_checkpoint_roundtrip(self):
        data = self.data(n_types=2)
        features = self.features(n_types=2)
        buffer = io.BytesIO()
        torch.save(features, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        self.assertTrue(restored.include_divergence)
        self.assertEqual(restored.n_kernels, 4)
        torch.testing.assert_close(
            self.evaluate(restored, data), self.evaluate(features, data),
            atol=0., rtol=0.,
        )


if __name__ == "__main__":
    unittest.main()
