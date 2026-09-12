"""Shared charge/dipole Coulomb source, independent sums and variational tests."""

import io
import math
import unittest

import torch

from equicdft import (
    FixedDipoleIdeal, GridCACEModel, LDAReadout, LongRangeReadout,
    PolarizationSolver, ReciprocalFeatures,
)


class TestPolarizationCoulomb(unittest.TestCase):
    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(71)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)

    @staticmethod
    def readout(charges=(0.,), amplitude=1.7, alpha=.3, polar=True):
        features = ReciprocalFeatures((alpha,), kernel="coulomb", n_types=len(charges))
        return LongRangeReadout(1, n_types=len(charges), charges=charges,
                                coulomb_amplitude=amplitude, features=features,
                                include_polarization=polar)

    @staticmethod
    def data(shape=(5, 3, 3), n_types=1, spacing=.8):
        n = math.prod(shape)
        return dict(rho=.4 + .1 * torch.rand(n, n_types),
                    dipole_density=.02 * torch.randn(n, n_types, 3),
                    grid_size=torch.tensor(shape), grid_spacing=torch.full((3,), spacing),
                    temperature=torch.tensor(1.))

    @staticmethod
    def clone(data):
        return {k: v.detach().clone() for k, v in data.items()}

    @staticmethod
    def model(readout, spacing=.8, **kwargs):
        return GridCACEModel(None, None, [readout], grid_spacing=spacing,
                             compute_polarization_derivative=True, **kwargs)

    def test_charge_only_limit_and_legacy_serialization(self):
        charges = (1.3, -.7, 0.)
        old, new = self.readout(charges, polar=False), self.readout(charges)
        new.load_state_dict(old.state_dict(), strict=True)
        data = self.data(n_types=3)
        data["dipole_density"].zero_()
        torch.testing.assert_close(old.energy(data), new.energy(data), atol=1e-13, rtol=1e-13)
        del old.include_polarization  # simulate a pre-extension whole-module checkpoint
        buffer = io.BytesIO()
        torch.save(old, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=False)
        self.assertFalse(loaded.requires_dipole_density)
        self.assertFalse(loaded.requires_state_features)
        torch.testing.assert_close(loaded.energy(data), new.energy(data))
        scalar = GridCACEModel(None, None, [loaded], grid_spacing=.8).eval()
        polar = self.model(new).eval()
        for key, value in scalar(self.clone(data)).items():
            torch.testing.assert_close(value, polar(self.clone(data))[key])

    def test_independent_les_style_positive_phase_sum_and_derivatives(self):
        # Odd grid: no ambiguous Nyquist modes. Explicit all-mode particle-style
        # sum uses +ikr, opposite to torch FFT; voxel q=rho*q*dV, mu=P*dV.
        shape, spacing, alpha, amplitude = (3, 5, 3), .7, .23, 2.1
        charges = torch.tensor([1.4, -.8, 0.])
        data = self.data(shape, 3, spacing)
        rho = data["rho"].requires_grad_()
        p = data["dipole_density"].requires_grad_()
        axes = [torch.arange(n) * spacing for n in shape]
        r = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        modes = [torch.arange(-(n // 2), n // 2 + 1) * (2 * math.pi / (n * spacing)) for n in shape]
        k = torch.stack(torch.meshgrid(*modes, indexing="ij"), -1).reshape(-1, 3)
        k = k[k.square().sum(-1) > 0]
        phase = torch.exp(1j * (r @ k.T))
        source = spacing**3 * (
            ((rho * charges).sum(-1)[:, None] + 1j * (p.sum(-2) @ k.T)) * phase
        ).sum(0)
        k2 = k.square().sum(-1)
        reference = amplitude * 2 * math.pi / (math.prod(shape) * spacing**3) * (
            torch.exp(-alpha * k2) / k2 * source.abs().square()
        ).sum()
        actual = self.readout(charges, amplitude, alpha).energy(data)
        torch.testing.assert_close(actual, reference, atol=2e-13, rtol=2e-13)
        for a, b in zip(torch.autograd.grad(actual, (rho, p), retain_graph=True),
                        torch.autograd.grad(reference, (rho, p))):
            torch.testing.assert_close(a, b, atol=2e-13, rtol=2e-12)

    def test_longitudinal_transverse_energy_and_restoring_sign(self):
        for spacing in (.5, 2.):
            shape = (8, 3, 3)
            data = self.data(shape, spacing=spacing)
            x = torch.arange(shape[0]) * 2 * math.pi / shape[0]
            cosine = torch.cos(x)[:, None, None].expand(shape).reshape(-1)
            data["rho"].fill_(.4)
            data["dipole_density"].zero_()
            data["dipole_density"][:, 0, 0] = .03 * cosine
            result = self.model(self.readout(), spacing).eval()(self.clone(data))
            k = 2 * math.pi / (shape[0] * spacing)
            factor = 1.7 * 4 * math.pi * math.exp(-.3 * k*k)
            expected = factor * math.prod(shape) * spacing**3 * .03**2 / 4
            torch.testing.assert_close(result["beta_F_exc"], torch.tensor(expected), atol=1e-13, rtol=1e-12)
            torch.testing.assert_close(result["polarization_derivative"], factor * data["dipole_density"], atol=1e-13, rtol=1e-12)
            torch.testing.assert_close(result["c1"], torch.zeros_like(data["rho"]), atol=0., rtol=0.)
            data["dipole_density"] = data["dipole_density"].roll(1, -1)
            transverse = self.readout().energy(data)
            self.assertLess(abs(float(transverse)), 1e-28)

    def test_mixed_cross_term_sign_and_cancellation(self):
        shape = (7, 3, 3)
        data = self.data(shape, spacing=1.)
        x = torch.arange(7) * (2 * math.pi / 7)
        k, q, a, b = 2 * math.pi / 7, 1.4, .04, .03
        cosine = x.cos()[:, None, None].expand(shape).reshape(-1)
        sine = x.sin()[:, None, None].expand(shape).reshape(-1)
        data["rho"][:, 0] = .4 + a * cosine
        data["dipole_density"].zero_()
        data["dipole_density"][:, 0, 0] = b * sine
        readout = self.readout((q,))
        prefactor = 1.7 * math.prod(shape) * math.pi * math.exp(-.3*k*k) / (k*k)
        torch.testing.assert_close(readout.energy(data), torch.tensor(prefactor * (q*a-k*b)**2), atol=1e-13, rtol=1e-12)
        scalar, dipole = self.clone(data), self.clone(data)
        scalar["dipole_density"].zero_()
        dipole["rho"].fill_(.4)
        cross = readout.energy(data) - readout.energy(scalar) - readout.energy(dipole)
        torch.testing.assert_close(cross, torch.tensor(-2*prefactor*q*a*k*b), atol=1e-13, rtol=1e-12)
        data["dipole_density"][:, 0, 0] = q*a/k * sine
        self.assertLess(abs(float(readout.energy(data))), 1e-27)

    def test_zero_modes_and_nyquist_real_divergence(self):
        data = self.data((4, 4, 4))
        data["rho"].fill_(.4)
        data["dipole_density"][:] = torch.tensor([.01, -.02, .03])
        readout = self.readout((1.,))
        self.assertLess(abs(float(readout.energy(data))), 1e-28)
        checker = (-1.) ** torch.arange(4)
        data["dipole_density"][:, 0, 0] = checker[:, None, None].expand(4, 4, 4).reshape(-1)
        self.assertLess(abs(float(readout.energy(data))), 1e-28)
        # Charge Nyquist energy is retained; only the derivative is zeroed.
        data["rho"][:, 0] += .02 * data["dipole_density"][:, 0, 0]
        self.assertGreater(float(readout.energy(data)), 0.)
        p = data["dipole_density"].clone()
        data["dipole_density"].zero_()
        torch.testing.assert_close(readout.energy(dict(data, dipole_density=p)), readout.energy(data))

    def test_batch_and_neutral_species_not_erased(self):
        data = self.data(n_types=3)
        other = self.data(n_types=3)
        readout = self.readout((1., -2., 0.))
        batch = {k: torch.stack((data[k], other[k])) for k in ("rho", "dipole_density", "temperature")}
        batch.update(grid_size=data["grid_size"], grid_spacing=data["grid_spacing"])
        torch.testing.assert_close(readout.energy(batch), torch.stack((readout.energy(data), readout.energy(other))))
        model = self.model(readout).eval()
        output = model(batch)
        self.assertEqual(output["polarization_derivative"].shape, (2, 45, 3, 3))
        for i, single in enumerate((data, other)):
            for key, value in model(self.clone(single)).items():
                torch.testing.assert_close(value, output[key][i])
        data["rho"].fill_(.4)
        data["dipole_density"][:, :2].zero_()
        self.assertGreater(float(readout.energy(data)), 0.)

    def test_model_finite_differences_and_mixed_hessian(self):
        for mode in ("beta", "physical"):
            for spacing in (.5, 2.):
                data = self.data((3, 3, 3), 2, spacing)
                data["temperature"] = torch.tensor(2.)
                model = self.model(self.readout((1.3, -.7)), spacing, free_energy_mode=mode,
                                   mean_temperature=1.4, boltzmann_constant=.8).train()
                output = model(data)
                for field, index, key, sign in (("rho", (4, 1), "c1", -1),
                                               ("dipole_density", (4, 1, 2), "polarization_derivative", 1)):
                    plus, minus = self.clone(data), self.clone(data)
                    plus[field][index] += 1e-6
                    minus[field][index] -= 1e-6
                    fd = (model(plus)["beta_F_exc"] - model(minus)["beta_F_exc"]) / 2e-6 / spacing**3
                    torch.testing.assert_close(fd, sign * output[key][index], atol=2e-9, rtol=2e-7)
                left = torch.autograd.grad(output["c1"][4, 1], data["dipole_density"], retain_graph=True)[0][5, 0, 2]
                right = torch.autograd.grad(output["polarization_derivative"][5, 0, 2], data["rho"], retain_graph=True)[0][4, 1]
                self.assertGreater(abs(float(left)), 1e-8)
                torch.testing.assert_close(left, -right, atol=1e-12, rtol=1e-12)

    def test_physical_temperature_conversion(self):
        data = self.data()
        c, kb, tref = 14.39964547842567, 8.617333262145177e-5, 298.15
        for temperature in (250., 350.):
            data["temperature"] = torch.tensor(temperature)
            beta_model = self.model(self.readout(amplitude=c/(kb*temperature)), boltzmann_constant=kb).eval()
            physical = self.model(self.readout(amplitude=c/(kb*tref)), mean_temperature=tref,
                                  boltzmann_constant=kb, free_energy_mode="physical").eval()
            for key, value in beta_model(self.clone(data)).items():
                torch.testing.assert_close(value, physical(self.clone(data))[key], atol=1e-12, rtol=1e-12)

    def test_zero_p_higher_derivatives_and_anisotropic_features(self):
        shape = (3, 3, 1)
        rho = (.4 + .02 * torch.randn(2, 1, 9, 2)).requires_grad_()
        p = torch.zeros(*rho.shape, 3, requires_grad=True)
        spacing = torch.tensor([.6, .8, 1.1])
        features = self.readout((1.2, -.7)).features
        def energy(r, polarization):
            return features(r, torch.tensor(shape), spacing,
                            dipole_density=polarization, charges=torch.tensor([1.2, -.7])).sum()
        self.assertTrue(torch.autograd.gradcheck(energy, (rho, p), atol=2e-8, rtol=2e-6))
        self.assertTrue(torch.autograd.gradgradcheck(energy, (rho, p), atol=2e-8, rtol=2e-6))

    def test_additivity_with_squared_p_lda(self):
        lda = LDAReadout(mean_density=.4, dipole_density_scale=.2, hidden_sizes=(4,))
        coulomb = self.readout()
        data = self.data()
        separate = [self.model(item).eval()(self.clone(data)) for item in (lda, coulomb)]
        for readouts in ([lda, coulomb], [coulomb, lda]):
            model = GridCACEModel(None, None, readouts, grid_spacing=.8,
                                 compute_polarization_derivative=True).eval()
            result = model(self.clone(data))
            for key in result:
                torch.testing.assert_close(result[key], separate[0][key] + separate[1][key], atol=2e-12, rtol=2e-12)

    def test_lattice_covariance_translation_and_extensivity(self):
        shape = (4, 4, 4)  # includes nontrivial Nyquist planes
        data = self.data(shape, 2)
        readout = self.readout((1., -.6))
        model = self.model(readout).eval()
        reference = model(self.clone(data))
        def transform(value, vector=False):
            grid = value.reshape(*shape, *value.shape[1:]).transpose(0, 1).flip(2)
            if vector:
                grid = grid[..., [1, 0, 2]] * torch.tensor([1., 1., -1.])
            return grid.reshape_as(value)
        changed = self.clone(data)
        changed["rho"] = transform(data["rho"])
        changed["dipole_density"] = transform(data["dipole_density"], True)
        result = model(changed)
        torch.testing.assert_close(result["beta_F_exc"], reference["beta_F_exc"], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(result["polarization_derivative"], transform(reference["polarization_derivative"], True), atol=1e-12, rtol=1e-12)
        shifted, tiled = self.clone(data), self.clone(data)
        for key in ("rho", "dipole_density"):
            grid = data[key].reshape(*shape, *data[key].shape[1:])
            shifted[key] = grid.roll((1, -2, 1), (0, 1, 2)).reshape_as(data[key])
            tiled[key] = grid.repeat(2, 1, 1, *([1] * (grid.ndim-3))).reshape(-1, *data[key].shape[1:])
        tiled["grid_size"] = torch.tensor([8, 4, 4])
        torch.testing.assert_close(readout.energy(shifted), reference["beta_F_exc"], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(readout.energy(tiled), 2*reference["beta_F_exc"], atol=1e-12, rtol=1e-12)

    def test_learned_amplitude_has_response_loss_gradients(self):
        readout = self.readout((0., 1.), amplitude=None)
        lda = LDAReadout(mean_density=.4, n_types=2, zero_init=True)
        model = GridCACEModel(None, None, [lda, readout], grid_spacing=.8,
                             compute_polarization_derivative=True).train()
        # A spatially varying target tests response-loss training even from
        # zero amplitude; a constant target is orthogonal to these modes.
        output = model(self.data(n_types=2))
        (output["polarization_derivative"] - torch.randn_like(output["polarization_derivative"])).square().mean().backward()
        self.assertGreater(float(readout.mlp[-1].bias.grad.abs().max()), 1e-8)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in readout.parameters()))

    def test_manufactured_coupled_equilibrium_two_starts(self):
        data = self.data((5, 3, 3), spacing=1.)
        data["rho"][:, 0] = .4 + .02 * torch.arange(45).sin()
        data["dipole_density"] *= .25
        model = self.model(self.readout((.4,), amplitude=.2), spacing=1.).eval()
        response = model(self.clone(data))
        ideal = FixedDipoleIdeal(.7)(data["rho"], data["dipole_density"], 1.)
        external = {k: data[k] for k in ("grid_size", "grid_spacing", "temperature")}
        external.update(beta=torch.tensor(1.), V_ext=(response["c1"]-ideal["density_derivative"]).detach(),
                        E_ext=(response["polarization_derivative"]+ideal["polarization_derivative"]).detach())
        numbers = data["rho"].sum(0)
        solver = PolarizationSolver(.7, model)
        perturbed = data["rho"] * (1 + .03 * torch.randn_like(data["rho"]))
        perturbed *= numbers / perturbed.sum(0)
        for start in ({}, dict(initial_rho=perturbed, initial_polarization=.002*torch.randn_like(data["dipole_density"]))):
            result = solver.solve(external, numbers, tolerance_residual=1e-9, max_iter=300, **start)
            self.assertEqual(result["status"], "converged")
            torch.testing.assert_close(result["rho"], data["rho"], atol=2e-9, rtol=2e-9)
            torch.testing.assert_close(result["dipole_density"], data["dipole_density"], atol=2e-9, rtol=2e-9)
            torch.testing.assert_close(result["rho"].sum(0), numbers, atol=1e-12, rtol=1e-12)

    def test_invalid_contracts(self):
        with self.assertRaisesRegex(ValueError, "requires charges and Coulomb"):
            LongRangeReadout(1, include_polarization=True)
        with self.assertRaisesRegex(ValueError, "requires charges and Coulomb"):
            LongRangeReadout(1, charges=(0.,), features=ReciprocalFeatures((.3,)), include_polarization=True)
        data = self.data()
        module = self.readout().features
        args = (data["rho"], data["grid_size"], data["grid_spacing"])
        for p, q in ((data["dipole_density"], None), (data["dipole_density"][..., 0], [0.]),
                     (data["dipole_density"], [0., 1.]), (data["dipole_density"]*float("nan"), [0.]),
                     (data["dipole_density"].float(), [0.])):
            with self.assertRaises(ValueError):
                module(*args, dipole_density=p, charges=q)
        model = self.model(self.readout())
        self.assertFalse(model.requires_local_density_index)
        self.assertEqual(model.cutoff_grid, 0)
        del data["dipole_density"]
        with self.assertRaisesRegex(ValueError, "dipole_density is required"):
            model(data)


if __name__ == "__main__":
    unittest.main()
