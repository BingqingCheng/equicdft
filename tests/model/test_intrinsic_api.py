"""Explicit ideal/excess ownership and numerical-versus-autograd boundaries."""

import unittest

import torch

from equicdft import FixedDipoleIdeal, GridCACEModel, GridSolver, IdealGas, LDAReadout
from equicdft.training.noise import add_field_noise


class TestIntrinsicAPI(unittest.TestCase):
    def setUp(self):
        self.previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(self.previous_dtype)

    def data(self, polarized):
        rho = torch.tensor([[.2], [.3], [.4], [.5]], dtype=torch.float64)
        data = dict(rho=rho, V_ext=rho * .2, beta=rho.new_tensor(1.),
                    temperature=rho.new_tensor(1.), grid_spacing=rho.new_full((3,), .7),
                    grid_size=torch.tensor([4, 1, 1]))
        if polarized:
            data.update(dipole_density=rho[..., None] * rho.new_tensor([.1, .02, -.03]),
                        E_ext=rho.new_zeros(4, 1, 3))
        return data

    def model(self, polarized):
        readout = LDAReadout(n_types=1, dipole_density_scale=.3 if polarized else None)
        model = GridCACEModel(
            None, None, [readout], ideal=FixedDipoleIdeal(.7) if polarized else IdealGas(),
            n_types=1, grid_spacing=.7, mean_density=.3, boltzmann_constant=1.,
            compute_polarization_derivative=polarized,
        ).double()
        # Nonzero learned contribution ensures total/excess confusion is visible.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(.1)
        return model

    def test_shared_local_and_integrated_ideal_contract(self):
        for polarized in (False, True):
            data = self.data(polarized)
            ideal = FixedDipoleIdeal(.7) if polarized else IdealGas()
            local = ideal.local(data["rho"], data.get("dipole_density"))
            volume = data["grid_spacing"].prod()
            integrated = ideal.evaluate(data, volume)
            torch.testing.assert_close(integrated["beta_F_id"],
                                       (data["rho"] * local["energy_per_particle"]).sum() * volume,
                                       rtol=0, atol=0)
            for key in local.keys() - {"energy_per_particle"}:
                torch.testing.assert_close(local[key], integrated[key], rtol=0, atol=0)
            self.assertNotIn("energy_per_particle", integrated)
            self.assertNotIn("density_derivative", integrated)
            self.assertNotIn("polarization_derivative", integrated)

    def test_total_excess_names_and_continuum_normalization(self):
        for polarized in (False, True):
            data, model = self.data(polarized), self.model(polarized)
            out = model(data)
            volume = data["grid_spacing"].prod()
            fields = [data["rho"]] + ([data["dipole_density"]] if polarized else [])
            for energy, suffix in (("beta_F", ""), ("beta_F_exc", "_excess")):
                gradients = torch.autograd.grad(out[energy], fields, retain_graph=True)
                for name, gradient in zip(("density", "polarization"), gradients):
                    torch.testing.assert_close(out[name + "_derivative" + suffix], gradient / volume)
            torch.testing.assert_close(out["density_derivative_excess"], -out["c1"], rtol=0, atol=0)
            for name in ("density", "polarization") if polarized else ("density",):
                torch.testing.assert_close(out[name + "_derivative"],
                                           out["ideal_" + name + "_derivative"] + out[name + "_derivative_excess"])
            self.assertNotIn("total_polarization_derivative", out)
            if polarized:
                self.assertGreater(out["polarization_derivative_excess"].abs().max().item(), 0.)
            excess = model(data, compute_ideal=False)
            self.assertNotIn("density_derivative", excess)
            self.assertNotIn("polarization_derivative", excess)
            self.assertIn("density_derivative_excess", excess)
            if polarized:
                self.assertIn("polarization_derivative_excess", excess)

    def test_model_retains_training_and_mixed_response_graphs(self):
        data, model = self.data(True), self.model(True)
        out = model(data)
        rho_p = torch.autograd.grad(out["density_derivative"][0, 0],
                                    data["dipole_density"], retain_graph=True)[0][0, 0, 1]
        p_rho = torch.autograd.grad(out["polarization_derivative"][0, 0, 1],
                                    data["rho"], retain_graph=True)[0][0, 0]
        torch.testing.assert_close(rho_p, p_rho, rtol=1e-12, atol=1e-12)
        (out["density_derivative"].square().sum()
         + out["polarization_derivative"].square().sum()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters()))

    def test_solver_detaches_outputs_without_modifying_caller_or_model(self):
        for polarized in (False, True):
            for training in (False, True):
                for derivatives in (False, True):
                    for requires_grad in (False, True):
                        with self.subTest(polarized=polarized, training=training,
                                          derivatives=derivatives, requires_grad=requires_grad):
                            data, model = self.data(polarized), self.model(polarized)
                            model.train(training)
                            fields = [data["rho"]] + ([data["dipole_density"]] if polarized else [])
                            for field in fields:
                                field.requires_grad_(requires_grad)
                            values = [field.detach().clone() for field in fields]
                            with torch.no_grad():
                                result = GridSolver(model).evaluate(data, compute_derivatives=derivatives)
                            for value in result.values():
                                if torch.is_tensor(value):
                                    self.assertFalse(value.requires_grad)
                                    self.assertIsNone(value.grad_fn)
                            for field, expected in zip(fields, values):
                                self.assertEqual(field.requires_grad, requires_grad)
                                torch.testing.assert_close(field, expected, rtol=0, atol=0)
                            self.assertNotEqual(result["rho"].data_ptr(), data["rho"].data_ptr())
                            self.assertEqual(model.training, training)
                            self.assertTrue(all(p.grad is None for p in model.parameters()))
                            self.assertEqual("density_derivative" in result, derivatives)
                            self.assertEqual("polarization_derivative" in result, polarized and derivatives)

    def test_sem_preserves_all_supplied_targets(self):
        data = self.data(True)
        ideal = FixedDipoleIdeal(.7)
        data["dipole_density_std"] = torch.full_like(data["dipole_density"], .002)
        total_target = torch.full_like(data["dipole_density"], .4)
        data["polarization_derivative"] = total_target
        before = ideal.local(data["rho"], data["dipole_density"])
        data["polarization_derivative_excess"] = total_target - before["ideal_polarization_derivative"]
        result = add_field_noise(data, polarization_noise=True, ideal=ideal)
        torch.testing.assert_close(result["polarization_derivative"], total_target, rtol=0, atol=0)
        torch.testing.assert_close(result["polarization_derivative_excess"],
                                   data["polarization_derivative_excess"], rtol=0, atol=0)
