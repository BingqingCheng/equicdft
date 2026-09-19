"""Model-owned moments and explicitly approximate Fourier stability references."""

import unittest
import torch
from equicdft import FixedDipoleIdeal, FourierStabilityLoss, GridCACEModel, LDAReadout, Loss
from equicdft.response.curvature import polarization_fourier_curvature
from equicdft.training.trainer import _load_loss_state
from tests.training import test_loss as fixtures


class TestStabilityReference(unittest.TestCase):
    def batch(self):
        fixture = fixtures.TestFourierStabilityLoss
        return fixture._add_polarization(fixture._batch())

    def model(self, coefficient=0., moment=1.):
        return fixtures._PolarizationQuadraticExcessModel(
            coefficient * torch.eye(3), moment=moment,
        )

    def term(self, **kwargs):
        return FourierStabilityLoss(
            wavevector_indices=((1, 0, 0),), variable="dipole_density", **kwargs,
        )

    def test_only_model_owns_moment_and_changes_are_used_immediately(self):
        batch, model = self.batch(), self.model(-12.)
        term = self.term(relative_amplitude=2.**-10)
        self.assertFalse(hasattr(term, "dipole_magnitude"))
        self.assertNotIn("dipole_magnitude", term.state_dict())
        for moment, expected in ((.5, 0.), (1., 1.), (2., 49.)):
            model.ideal.dipole_magnitude.fill_(moment)
            value = term(model(batch), batch, model=model)
            torch.testing.assert_close(value, value.new_tensor(expected), rtol=1e-10, atol=1e-10)
        with self.assertRaisesRegex(TypeError, "dipole_magnitude"):
            self.term(dipole_magnitude=1.)

    def test_reference_is_exposed_in_repr_and_diagnostics(self):
        batch, model, term = self.batch(), self.model(), self.term()
        self.assertEqual(term.ideal_reference, "quadratic_alignment")
        self.assertIn("quadratic_alignment", repr(term))
        values, details = Loss([term]).evaluate(model(batch), batch, model=model)
        self.assertEqual(details[term.name]["ideal_reference"], "quadratic_alignment")
        self.assertEqual(values["total"].item(), 0.)
        scalar = FourierStabilityLoss(wavevector_indices=((1, 0, 0),))
        self.assertEqual(scalar.ideal_reference, "translational_ideal_gas")

    def test_missing_or_wrong_ideal_is_rejected(self):
        from equicdft import IdealGas
        batch, model = self.batch(), self.model()
        for ideal in (None, IdealGas()):
            model.ideal = ideal
            with self.assertRaisesRegex(TypeError, "FixedDipoleIdeal"):
                self.term()(model(batch), batch, model=model)

    def test_exact_ideal_boundary_is_not_applied_to_stability_probes(self):
        batch = self.batch()
        batch["dipole_density"][..., 0] = 1.05 * batch["rho"]
        model = GridCACEModel(
            None, None, [LDAReadout(dipole_density_scale=.5)],
            ideal=FixedDipoleIdeal(1.), grid_spacing=1., mean_density=.5,
            compute_c1=False,
        ).double()
        outputs = model(batch, compute_ideal=False)
        with self.assertRaises(ValueError):
            model(batch)
        value = self.term()(outputs, batch, model=model)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_quadratic_reference_does_not_depend_on_base_alignment(self):
        batch, model = self.batch(), self.model(moment=.7)
        batch["dipole_density"][..., 0] = .8 * .7 * batch["rho"]
        curvature, valid = polarization_fourier_curvature(
            model, model(batch), batch, batch["rho"], torch.tensor([[[1, 0, 0]]]),
            relative_amplitude=.1, polarization_directions=torch.tensor([[1., 0., 0.]]),
        )
        torch.testing.assert_close(curvature[valid], torch.ones_like(curvature[valid]))

    def test_legacy_loss_state_is_checked_then_moment_is_discarded(self):
        model = self.model(moment=.7)
        for loss in (Loss([self.term()]), torch.nn.ModuleDict({"P": Loss([self.term()])})):
            state = loss.state_dict()
            prefix = "P.terms.0." if isinstance(loss, torch.nn.ModuleDict) else "terms.0."
            state[prefix + "dipole_magnitude"] = torch.tensor([.7])
            _load_loss_state(loss, state, model)
            self.assertNotIn(prefix + "dipole_magnitude", loss.state_dict())
            self.assertIn(prefix + "dipole_magnitude", state)
            state[prefix + "dipole_magnitude"] = torch.tensor([1.])
            with self.assertRaisesRegex(ValueError, "conflicts"):
                _load_loss_state(loss, state, model)


if __name__ == "__main__":
    unittest.main()
