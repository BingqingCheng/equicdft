"""Full-ideal-aware fixed-density probes, with legacy normalization intact."""
import math
import unittest
import torch
from equicdft import FixedDipoleIdeal, FourierStabilityLoss, Loss
from equicdft.response.curvature import (
    coupled_fourier_curvature_matrix, polarization_fourier_curvature,
)
from equicdft.functional.polarization_ideal import inverse_langevin, _langevin_prime
from tests.training.test_loss import _PolarizationQuadraticExcessModel, TestFourierStabilityLoss


class TestPolarizationIdealReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls.old_dtype)

    def data(self, u=0.0, moment=.7):
        data = TestFourierStabilityLoss._add_polarization(TestFourierStabilityLoss._batch())
        data["dipole_density"][..., 0] = u * moment * data["rho"]
        return data

    def model(self, coefficient=0.0, moment=.7):
        return _PolarizationQuadraticExcessModel(coefficient * torch.eye(3), moment=moment)

    def curvature(self, model, data, direction=(1., 0., 0.), **kwargs):
        return polarization_fourier_curvature(
            model, model(data), data, data["rho"],
            torch.tensor([[[0, 0, 0], [1, 0, 0]]]), .02,
            torch.tensor([direction]), **kwargs,
        )

    def test_closed_form_parallel_perpendicular_and_arbitrary_direction(self):
        model = self.model()
        for u in (0., .005, .5, .78, .97):
            data = self.data(u)
            if u:
                x = inverse_langevin(torch.tensor(u))
                parallel, perpendicular = 1 / (3 * _langevin_prime(x)), x / (3 * u)
            else:
                parallel = perpendicular = torch.tensor(1.)
            for direction in ((1., 0., 0.), (0., 1., 0.), (.6, .8, 0.)):
                value, valid = self.curvature(model, data, direction, alignment_reference="exact")
                expected = parallel * direction[0]**2 + perpendicular * (1 - direction[0]**2)
                torch.testing.assert_close(value[valid], expected.expand_as(value[valid]), rtol=1e-11, atol=1e-11)

    def test_full_reference_removes_a_false_stability_penalty(self):
        data, model = self.data(.78), self.model(-12., moment=.7)
        # The quadratic proxy is .02, while full ideal curvature is >6 here.
        options = dict(wavevector_indices=((1, 0, 0),), variable="dipole_density",
                       minimum_curvature=.27, include_zero_wavevector=True)
        torch.manual_seed(12)
        quadratic = FourierStabilityLoss(**options)(model(data), data, model=model)
        torch.manual_seed(12)
        exact = FourierStabilityLoss(**options, alignment_reference="exact")(model(data), data, model=model)
        self.assertGreater(quadratic.item(), 0.)
        self.assertEqual(exact.item(), 0.)

    def test_default_is_bitwise_quadratic_and_random_draws_are_identical(self):
        data, model = self.data(.5), self.model(-12.)
        options = dict(variable="dipole_density", random_wavevectors_per_field=2,
                       relative_amplitude=(.01, .03), polarization_direction="transverse")
        values, states, displacements = [], [], []
        for reference in (None, "quadratic", "exact"):
            torch.manual_seed(44)
            term = FourierStabilityLoss(**options, alignment_reference=reference)
            values.append(term(model(data), data, model=model))
            states.append(torch.get_rng_state())
            displacements.append(model.last_polarization.clone())
        self.assertTrue(torch.equal(values[0], values[1]))
        for index in (1, 2):
            self.assertTrue(torch.equal(states[0], states[index]))
            self.assertTrue(torch.equal(displacements[0], displacements[index]))
        legacy = FourierStabilityLoss(wavevector_indices=((1, 0, 0),), variable="dipole_density")
        legacy.alignment_reference = None
        self.assertTrue(torch.isfinite(legacy(model(data), data, model=model)))

    def test_matches_coupled_P_block_for_weighted_inhomogeneous_fields(self):
        model = self.model(-.2)
        data = self.data()
        index = data["grid_positions"][0, :, 0]
        data["rho"][0, :, 0] = .5 + .2 * torch.cos(2 * math.pi * index / 8)
        u = .4 + .1 * torch.sin(2 * math.pi * index / 8)
        data["dipole_density"][0, :, 0, 0] = u * .7 * data["rho"][0, :, 0]
        waves = torch.tensor([[[1, 0, 0], [2, 0, 0]]])
        direction = torch.tensor([[.6, .8, 0.]])
        for power in (0., 2.):
            for phases in (None, torch.tensor([[.2, 1.4]])):
                common = dict(wavevector_phases=phases, density_power=power, alignment_reference="exact")
                p, valid = polarization_fourier_curvature(model, model(data), data, data["rho"], waves, .02, direction, **common)
                matrix, active = coupled_fourier_curvature_matrix(model, model(data), data, data["rho"], waves, .02, direction, **common)
                self.assertTrue(bool(valid.all()) and bool(active.all()))
                torch.testing.assert_close(p, matrix[..., 1, 1], rtol=1e-10, atol=1e-10)

    def test_species_moments_and_zero_density(self):
        data = self.data()
        data["rho"] = data["rho"].expand(-1, -1, 2).clone()
        data["rho"][0, 2, 1] = 0.
        moments = torch.tensor([.5, .9])
        alignments = torch.tensor([.3, .78])
        data["dipole_density"] = torch.zeros(data["rho"].shape + (3,))
        data["dipole_density"][..., 0] = data["rho"] * moments * alignments
        model = self.model()
        model.ideal = FixedDipoleIdeal(moments)
        value, valid = polarization_fourier_curvature(
            model, model(data), data, data["rho"], torch.tensor([[[0, 0, 0]]]),
            .02, torch.tensor([[1., 0., 0.]]), alignment_reference="exact")
        x = inverse_langevin(alignments)
        curvatures = 1 / (3 * _langevin_prime(x))
        expected = (data["rho"] * curvatures).sum() / data["rho"].sum()
        torch.testing.assert_close(value[valid], expected.expand_as(value[valid]), rtol=1e-11, atol=1e-11)

    def test_boundary_crossing_capped_values_and_parameter_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for u in (0., .97, 1., 1.05):
                data = {key:val.to(dtype) if val.is_floating_point() else val for key,val in self.data(u).items()}
                model = self.model(-100.).to(dtype)
                term = FourierStabilityLoss(wavevector_indices=((1, 0, 0),), variable="dipole_density",
                    alignment_reference="exact", relative_amplitude=.05, minimum_curvature=.27)
                value = term(model(data), data, model=model)
                value.backward()
                self.assertTrue(bool(torch.isfinite(value)))
                self.assertTrue(bool(torch.isfinite(model.matrix.grad).all()))
                self.assertTrue(bool(torch.isfinite(model.last_polarization).all()))

    def test_parameter_gradient_matches_scalar_finite_difference(self):
        data, model = self.data(.5), self.model(-20.)
        def objective():
            value, valid = self.curvature(model, data, alignment_reference="exact")
            return torch.relu(.27 - value[valid]).square().mean()
        value = objective()
        value.backward()
        gradient = model.matrix.grad[0, 0].item()
        original = model.matrix[0, 0].item()
        values = []
        with torch.no_grad():
            for shift in (1e-3, -1e-3):
                model.matrix[0, 0] = original + shift
                values.append(objective().item())
            model.matrix[0, 0] = original
        self.assertAlmostEqual(gradient, (values[0] - values[1]) / .002, places=7)

    def test_metadata_reload_and_validation_zero(self):
        data, model = self.data(.5), self.model(-20.)
        options = dict(wavevector_indices=((1, 0, 0),), variable="dipole_density", alignment_reference="exact")
        term = FourierStabilityLoss(**options)
        loss = Loss([term])
        replica = Loss([FourierStabilityLoss(**options)])
        replica.load_state_dict(loss.state_dict(), strict=True)
        self.assertEqual(term.ideal_reference, "langevin_alignment")
        self.assertIn("langevin_alignment", repr(term))
        _, details = loss.evaluate(model(data), data, model=model)
        self.assertEqual(details[term.name]["ideal_reference"], "langevin_alignment")
        term.eval()
        self.assertEqual(term(model(data), data, model=model).item(), 0.)

    def test_invalid_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "alignment_reference"):
            self.curvature(self.model(), self.data(), alignment_reference="bad")
        for variable in ("rho", "dipole_density", "coupled"):
            with self.assertRaisesRegex(ValueError, "alignment_reference"):
                FourierStabilityLoss(wavevector_indices=((1, 0, 0),), variable=variable, alignment_reference="bad")


if __name__ == "__main__":
    unittest.main()
