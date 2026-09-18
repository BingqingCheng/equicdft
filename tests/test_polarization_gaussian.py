"""Gaussian vector correlations: uniform modes, normalization and responses."""

import io
import math
import unittest

import torch

from equicdft import GridCACEModel, LDAReadout, LongRangeReadout, ReciprocalFeatures
from equicdft.loss import TensorLoss


class TestPolarizationGaussian(unittest.TestCase):
    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        self.rng_state = torch.random.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(83)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)
        torch.random.set_rng_state(self.rng_state)

    @staticmethod
    def data(shape=(5, 3, 3), n_types=1, spacing=.8):
        size = math.prod(shape)
        return {
            "rho": .4 + .05 * torch.rand(size, n_types),
            "dipole_density": .02 * torch.randn(size, n_types, 3),
            "grid_size": torch.tensor(shape),
            "grid_spacing": torch.full((3,), spacing),
            "temperature": torch.tensor(1.),
        }

    @staticmethod
    def clone(data):
        return {key: value.detach().clone() for key, value in data.items()}

    @staticmethod
    def readout(exponents=(.3,), coefficients=(1.7,), n_types=1):
        features = ReciprocalFeatures(
            exponents, kernel="gaussian", n_types=n_types,
            variable="dipole_density",
        )
        readout = LongRangeReadout(
            len(exponents), n_types=n_types, features=features,
            hidden_sizes=(),
        )
        with torch.no_grad():
            readout.mlp[-1].weight.zero_()
            readout.mlp[-1].bias.copy_(torch.tensor(coefficients))
        return readout

    @staticmethod
    def context(data):
        return dict(data, state_features=torch.cat((
            data["temperature"][..., None], data["rho"].mean(-2) / .4,
        ), dim=-1))

    @staticmethod
    def model(readout, spacing=.8, **kwargs):
        # The zero LDA supplies the existing model-level density scale only.
        lda = LDAReadout(mean_density=.4, n_types=readout.n_types, zero_init=True)
        return GridCACEModel(
            None, None, [lda, readout], grid_spacing=spacing,
            compute_polarization_derivative=True, **kwargs,
        )

    def test_uniform_p_energy_and_restoring_derivative(self):
        readout = self.readout(exponents=(.2, 1.3), coefficients=(1.2, -.3))
        self.assertTrue(readout.requires_dipole_density)
        self.assertTrue(readout.requires_state_features)
        self.assertFalse(readout.requires_local_density_index)
        for spacing in (.5, 2.):
            data = self.data(spacing=spacing)
            data["dipole_density"][:] = torch.tensor([.03, -.02, .01])
            output = self.model(readout, spacing).eval()(self.clone(data))
            volume = data["rho"].shape[0] * spacing**3
            expected = .5 * volume * .9 * data["dipole_density"][0].square().sum()
            torch.testing.assert_close(output["beta_F_exc"], expected, atol=1e-13, rtol=1e-12)
            torch.testing.assert_close(
                output["polarization_derivative"], .9 * data["dipole_density"],
                atol=1e-13, rtol=1e-12,
            )
            torch.testing.assert_close(output["c1"], torch.zeros_like(data["rho"]), atol=0., rtol=0.)

    def test_longitudinal_and_both_transverse_sinusoidal_modes(self):
        shape, amplitude, alpha, coefficient = (8, 3, 3), .03, .3, 1.7
        for spacing in (.5, 2.):
            phase = torch.arange(shape[0]) * (2 * math.pi / shape[0])
            k = 2 * math.pi / (shape[0] * spacing)
            factor = coefficient * math.exp(-alpha * k*k)
            expected_energy = factor * math.prod(shape) * spacing**3 * amplitude**2 / 4
            for direction in range(3):
                for wave in (phase.cos(), phase.sin()):
                    with self.subTest(spacing=spacing, direction=direction):
                        data = self.data(shape, spacing=spacing)
                        data["dipole_density"].zero_()
                        data["dipole_density"][:, 0, direction] = amplitude * wave[:, None, None].expand(shape).reshape(-1)
                        result = self.model(self.readout(), spacing).eval()(self.clone(data))
                        torch.testing.assert_close(result["beta_F_exc"], torch.tensor(expected_energy), atol=1e-13, rtol=1e-12)
                        torch.testing.assert_close(
                            result["polarization_derivative"], factor * data["dipole_density"],
                            atol=1e-13, rtol=1e-12,
                        )

    def test_gaussian_retains_vector_nyquist_modes(self):
        shape, alpha = (4, 4, 4), .3
        data = self.data(shape, spacing=1.)
        checker = (-1.) ** torch.arange(shape[0])
        data["dipole_density"].zero_()
        data["dipole_density"][:, 0, 0] = .03 * checker[:, None, None].expand(shape).reshape(-1)
        result = self.model(self.readout(), spacing=1.).eval()(self.clone(data))
        factor = 1.7 * math.exp(-alpha * math.pi**2)
        expected = .5 * math.prod(shape) * .03**2 * factor
        torch.testing.assert_close(result["beta_F_exc"], torch.tensor(expected), atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(result["polarization_derivative"], factor * data["dipole_density"], atol=1e-13, rtol=1e-12)

    def test_independent_gaussian_and_coulomb_uniform_l_t_response(self):
        shape, spacing, alpha = (8, 3, 3), .8, .3
        gaussian = self.readout()
        coulomb = LongRangeReadout(
            1, charges=(0.,), coulomb_amplitude=1.3, include_polarization=True,
            features=ReciprocalFeatures((alpha,), kernel="coulomb"),
        )
        lda = LDAReadout(mean_density=.4, zero_init=True)
        model = GridCACEModel(
            None, None, [lda, gaussian, coulomb], grid_spacing=spacing,
            compute_polarization_derivative=True,
        ).eval()
        k = 2 * math.pi / (shape[0] * spacing)
        damping = math.exp(-alpha * k*k)
        cosine = (2 * math.pi * torch.arange(shape[0]) / shape[0]).cos()
        for sector, direction in (("uniform", 0), ("longitudinal", 0), ("transverse", 1)):
            with self.subTest(sector=sector):
                data = self.data(shape)
                data["dipole_density"].zero_()
                wave = torch.ones(shape[0]) if sector == "uniform" else cosine
                data["dipole_density"][:, 0, direction] = .03 * wave[:, None, None].expand(shape).reshape(-1)
                gaussian_factor = 1.7 if sector == "uniform" else 1.7 * damping
                coulomb_factor = 1.3 * 4 * math.pi * damping if sector == "longitudinal" else 0.
                factor = gaussian_factor + coulomb_factor
                output = model(self.clone(data))
                expected_energy = .5 * spacing**3 * factor * data["dipole_density"].square().sum()
                torch.testing.assert_close(output["beta_F_exc"], expected_energy, atol=1e-13, rtol=1e-12)
                torch.testing.assert_close(output["polarization_derivative"], factor * data["dipole_density"], atol=1e-13, rtol=1e-12)
                torch.testing.assert_close(
                    output["beta_F_exc"],
                    gaussian.energy(self.context(data)) + coulomb.energy(data),
                    atol=1e-13, rtol=1e-12,
                )

    def test_multispecies_pair_dot_products_and_factor_two(self):
        data = self.data(n_types=2)
        vectors = torch.tensor([[.03, -.02, .01], [-.01, .02, .04]])
        data["dipole_density"][:] = vectors
        readout = self.readout(coefficients=(1.2, -.4, .7), n_types=2)
        features = readout.features(
            data["rho"], data["grid_size"], data["grid_spacing"],
            dipole_density=data["dipole_density"],
        )
        volume = data["rho"].shape[0] * .8**3
        expected = volume * torch.stack((
            vectors[0].square().sum() / 2,
            torch.dot(vectors[0], vectors[1]),
            vectors[1].square().sum() / 2,
        ))
        torch.testing.assert_close(features[0], expected, atol=1e-13, rtol=1e-12)
        result = self.model(readout).eval()(self.clone(data))
        matrix = torch.tensor([[1.2, -.4], [-.4, .7]])
        torch.testing.assert_close(result["beta_F_exc"], (expected * torch.tensor([1.2, -.4, .7])).sum(), atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(result["polarization_derivative"], (matrix @ vectors).expand_as(data["dipole_density"]), atol=1e-13, rtol=1e-12)

    def test_explicit_fourier_sum_batched_anisotropic_grid(self):
        # Independent DFT reference, including k=0, without reusing FFT helpers.
        shape = (3, 5, 1)
        spacing = torch.tensor([.6, .8, 1.1])
        rho = .4 + .05 * torch.rand(2, 1, math.prod(shape), 2)
        p = (.02 * torch.randn(*rho.shape, 3)).requires_grad_()
        features = ReciprocalFeatures(
            (.23, .7), n_types=2, variable="dipole_density",
        )
        actual = features(rho, torch.tensor(shape), spacing, dipole_density=p)
        r_axes = [torch.arange(n) * spacing[i] for i, n in enumerate(shape)]
        k_axes = [torch.arange(-(n // 2), n // 2 + 1) * (2 * math.pi / (n * spacing[i])) for i, n in enumerate(shape)]
        positions = torch.stack(torch.meshgrid(*r_axes, indexing="ij"), -1).reshape(-1, 3)
        k = torch.stack(torch.meshgrid(*k_axes, indexing="ij"), -1).reshape(-1, 3)
        phase = torch.exp(-1j * (positions @ k.T))
        volume_element = spacing.prod()
        p_hat = volume_element * torch.einsum("...gac,gk->...kac", p.to(phase.dtype), phase)
        kernels = torch.exp(-torch.tensor([.23, .7])[:, None] * k.square().sum(-1))
        volume = math.prod(shape) * volume_element
        expected = []
        for a, b in ((0, 0), (0, 1), (1, 1)):
            power = (p_hat[..., a, :].conj() * p_hat[..., b, :]).real.sum(-1)
            factor = 1. if a == b else 2.
            expected.append(factor * torch.einsum("nk,...k->...n", kernels, power) / (2 * volume))
        expected = torch.stack(expected, -1)
        self.assertEqual(actual.shape, (2, 1, 2, 3))
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
        actual_grad = torch.autograd.grad(actual.sum(), p, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), p)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-13, rtol=1e-12)

    def test_translation_extensivity_and_density_independence(self):
        shape = (4, 3, 3)
        data = self.data(shape, n_types=2)
        readout = self.readout(coefficients=(1.2, -.4, .7), n_types=2)
        reference = readout.energy(self.context(data))
        shifted, tiled = self.clone(data), self.clone(data)
        for key in ("rho", "dipole_density"):
            grid = data[key].reshape(*shape, *data[key].shape[1:])
            shifted[key] = grid.roll((1, -1, 1), (0, 1, 2)).reshape_as(data[key])
            repeats = (2, 1, 1) + (1,) * (grid.ndim - 3)
            tiled[key] = grid.repeat(*repeats).reshape(-1, *data[key].shape[1:])
        tiled["grid_size"] = torch.tensor([8, 3, 3])
        torch.testing.assert_close(readout.energy(self.context(shifted)), reference, atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(readout.energy(self.context(tiled)), 2 * reference, atol=1e-13, rtol=1e-12)
        # With fixed coefficients, physical P enters directly, not P/rho.
        changed = self.clone(data)
        changed["rho"] *= 3
        torch.testing.assert_close(readout.energy(self.context(changed)), reference, atol=0., rtol=0.)

    def test_zero_p_higher_derivatives(self):
        shape = (3, 1, 1)
        rho = .4 + .05 * torch.rand(3, 2)
        p = torch.zeros(*rho.shape, 3, requires_grad=True)
        module = ReciprocalFeatures((.3, .8), n_types=2, variable="dipole_density")

        def energy(polarization):
            return module(rho, torch.tensor(shape), torch.ones(3), dipole_density=polarization).sum()

        self.assertEqual(float(energy(p)), 0.)
        self.assertTrue(torch.autograd.gradcheck(energy, (p,), atol=2e-8, rtol=2e-6))
        self.assertTrue(torch.autograd.gradgradcheck(energy, (p,), atol=2e-8, rtol=2e-6))

    def test_p_field_loss_trains_zero_initialized_kernel(self):
        readout = self.readout(coefficients=(0.,))
        data = self.data()
        data["dipole_density"] += torch.tensor([.02, -.01, .03])
        output = self.model(readout).train()(data)
        loss = TensorLoss("polarization", "polarization_derivative", "target")
        loss(output, {"target": data["dipole_density"].detach()}).backward()
        self.assertGreater(float(readout.mlp[-1].bias.grad.abs().max()), 1e-8)
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in readout.parameters()))

    def test_state_dependence_and_physical_mode_derivatives(self):
        for mode in ("beta", "physical"):
            readout = self.readout()
            with torch.no_grad():
                readout.mlp[-1].weight[:] = torch.tensor([[.2, .6]])
            data = self.data()
            data["temperature"] = torch.tensor(2.)
            model = self.model(readout, free_energy_mode=mode, mean_temperature=1.4, boltzmann_constant=.8).train()
            output = model(data)
            for field, index, key, sign in (
                ("rho", (4, 0), "c1", -1),
                ("dipole_density", (4, 0, 2), "polarization_derivative", 1),
            ):
                plus, minus = self.clone(data), self.clone(data)
                plus[field][index] += 1e-6
                minus[field][index] -= 1e-6
                fd = (model(plus)["beta_F_exc"] - model(minus)["beta_F_exc"]) / (2e-6 * .8**3)
                torch.testing.assert_close(fd, sign * output[key][index], atol=2e-9, rtol=2e-7)
            self.assertGreater(float(output["c1"].abs().max()), 1e-8)
            left = torch.autograd.grad(output["c1"][4, 0], data["dipole_density"], retain_graph=True)[0][5, 0, 2]
            right = torch.autograd.grad(output["polarization_derivative"][5, 0, 2], data["rho"], retain_graph=True)[0][4, 0]
            torch.testing.assert_close(left, -right, atol=1e-12, rtol=1e-12)

    def test_model_batch_and_additivity_with_lda_and_coulomb(self):
        gaussian = self.readout()
        lda = LDAReadout(mean_density=.4, dipole_density_scale=.2, hidden_sizes=(4,))
        coulomb = LongRangeReadout(
            1, charges=(0.,), coulomb_amplitude=1.3, include_polarization=True,
            features=ReciprocalFeatures((.3,), kernel="coulomb"),
        )
        data = self.data()
        individual = [self.model(item).eval()(self.clone(data)) for item in (gaussian, lda, coulomb)]
        model = GridCACEModel(
            None, None, [lda, gaussian, coulomb], grid_spacing=.8,
            compute_polarization_derivative=True,
        ).eval()
        result = model(self.clone(data))
        for key, value in result.items():
            torch.testing.assert_close(value, sum(output[key] for output in individual), atol=2e-12, rtol=2e-12)
        other = self.data()
        batch = {key: torch.stack((data[key], other[key])) for key in ("rho", "dipole_density", "temperature")}
        batch.update(grid_size=data["grid_size"], grid_spacing=data["grid_spacing"])
        batched = model(batch)
        self.assertEqual(batched["polarization_derivative"].shape, (2, 45, 1, 3))
        for index, single in enumerate((data, other)):
            for key, value in model(self.clone(single)).items():
                torch.testing.assert_close(value, batched[key][index], atol=2e-12, rtol=2e-12)

    def test_scalar_default_and_legacy_checkpoint_unchanged(self):
        data = self.data()
        default = ReciprocalFeatures((.3, .8))
        explicit = ReciprocalFeatures((.3, .8), variable="rho")
        args = (data["rho"], data["grid_size"], data["grid_spacing"])
        torch.testing.assert_close(default(*args), explicit(*args), atol=0., rtol=0.)
        del default.variable
        buffer = io.BytesIO()
        torch.save(default, buffer)
        buffer.seek(0)
        from equicdft.legacy import upgrade_legacy_model
        legacy = upgrade_legacy_model(torch.load(buffer, weights_only=False))
        torch.testing.assert_close(legacy(*args), explicit(*args), atol=0., rtol=0.)
        homogeneous = torch.full_like(data["rho"], .4)
        torch.testing.assert_close(explicit(homogeneous, *args[1:]), torch.zeros(2, 1), atol=0., rtol=0.)

    def test_invalid_contracts(self):
        for kernel in ("coulomb", "screened_inverse_laplacian"):
            with self.assertRaises(ValueError):
                ReciprocalFeatures(kernel=kernel, variable="dipole_density")
        with self.assertRaises(ValueError):
            ReciprocalFeatures(variable="unknown")
        module = ReciprocalFeatures((.3,), variable="dipole_density")
        for kwargs in ({"charges": (0.,)}, {"coulomb_amplitude": 1.}, {"include_polarization": True}):
            with self.assertRaises(ValueError):
                LongRangeReadout(1, features=module, **kwargs)
        data = self.data()
        args = (data["rho"], data["grid_size"], data["grid_spacing"])
        for p in (None, data["dipole_density"][..., 0], data["dipole_density"].float(), data["dipole_density"] * float("nan")):
            with self.assertRaises(ValueError):
                module(*args, dipole_density=p)
        with self.assertRaises(ValueError):
            module(*args, dipole_density=data["dipole_density"], charges=torch.tensor([0.]))
        model = self.model(self.readout())
        self.assertFalse(model.requires_local_density_index)
        self.assertEqual(model.cutoff_grid, 0)
        del data["dipole_density"]
        with self.assertRaisesRegex(ValueError, "dipole_density is required"):
            model(data)


if __name__ == "__main__":
    unittest.main()
