"""One intrinsic-functional contract for scalar and vector densities."""

import tempfile
import unittest
from pathlib import Path

import torch

from equicdft import (GridCACEModel, GridSolver, IdealGas, FixedDipoleIdeal,
                      LDAReadout, load_model, save_model)


class TestModelIdealReference(unittest.TestCase):
    def setUp(self):
        self.previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(self.previous)

    def data(self, polarized=False):
        rho = torch.full((8, 2), 0.3)
        data = dict(rho=rho, temperature=torch.tensor(1.), beta=torch.tensor(1.),
                    grid_spacing=torch.ones(3), grid_size=torch.tensor([2, 2, 2]),
                    V_ext=torch.zeros_like(rho))
        if polarized:
            data.update(dipole_density=torch.zeros(8, 2, 3), E_ext=torch.zeros(8, 2, 3))
        return data

    def model(self, polarized=False, excess=False):
        return GridCACEModel(
            None, None, [LDAReadout(n_types=2)] if excess else [],
            ideal=FixedDipoleIdeal([0.7, 1.9]) if polarized else IdealGas(),
            n_types=2, grid_spacing=1., boltzmann_constant=1.,
            mean_density=0.3, compute_polarization_derivative=polarized,
        )

    def test_zero_polarization_reduces_to_scalar(self):
        scalar = self.model()(self.data())
        polar = self.model(True)(self.data(True))
        for key in ("beta_F_id", "beta_F", "density_derivative"):
            torch.testing.assert_close(scalar[key], polar[key], rtol=0, atol=0)
        self.assertEqual(polar["polarization_derivative"].abs().max(), 0)

    def test_total_derivatives_match_autograd(self):
        for polarized in (False, True):
            data = self.data(polarized)
            if polarized:
                data["dipole_density"] += 0.01
            outputs = self.model(polarized, True)(data)
            torch.testing.assert_close(outputs["beta_F"], outputs["beta_F_id"] + outputs["beta_F_exc"])
            fields = [data["rho"]] + ([data["dipole_density"]] if polarized else [])
            gradients = torch.autograd.grad(outputs["beta_F"], fields)
            torch.testing.assert_close(gradients[0], outputs["density_derivative"])
            if polarized:
                torch.testing.assert_close(gradients[1], outputs["polarization_derivative"])

    def test_masked_vacuum_and_batches(self):
        data = self.data(True)
        data["rho"][0] = 0
        data["excluded_mask"] = torch.arange(8) == 0
        model = self.model(True)
        result = model(data)
        self.assertFalse(result["ideal_derivative_mask"][0].any())
        self.assertTrue(torch.isfinite(result["beta_F"]))
        batch = dict(data)
        for key in ("rho", "dipole_density", "temperature", "beta", "excluded_mask"):
            batch[key] = torch.stack([data[key], data[key]])
        torch.testing.assert_close(model(batch)["beta_F"], result["beta_F"].expand(2))

    def test_batched_wavelengths_and_density_curvature(self):
        data = self.data(True)
        batch = dict(data)
        for key in ("rho", "dipole_density", "temperature", "beta"):
            batch[key] = torch.stack([data[key], data[key]])
        batch["thermal_wavelength"] = torch.tensor([[.8, 1.3], [1.1, .9]])
        model = self.model(True)
        outputs = model(batch, compute_c2=True)
        for index in range(2):
            frame = dict(data, thermal_wavelength=batch["thermal_wavelength"][index])
            torch.testing.assert_close(outputs["beta_F_id"][index], model(frame)["beta_F_id"])
        self.assertEqual(outputs["c2"].abs().max(), 0)

    def test_old_polar_model_requires_explicit_physical_metadata(self):
        from equicdft.serialization import model_payload, build_model
        model = GridCACEModel(None, None, [LDAReadout(
            n_types=2, dipole_density_scale=.3,
        )], ideal=FixedDipoleIdeal([.7, 1.9]), grid_spacing=1.,
            mean_density=.3, boltzmann_constant=1.)
        payload = model_payload(model)
        payload["format_version"] = 2
        payload["config"].pop("ideal")
        payload["config"]["thermal_wavelength"] = 1.
        payload["state_dict"] = {k:v for k,v in payload["state_dict"].items()
                                 if not k.startswith("ideal.")}
        payload["state_dict"]["thermal_wavelength"] = torch.ones(2)
        with self.assertRaisesRegex(ValueError, "molecular dipole magnitude"):
            build_model(payload)
        converted = build_model(payload, ideal=FixedDipoleIdeal([.7, 1.9]))
        torch.testing.assert_close(model(self.data(True))["beta_F"],
                                   converted(self.data(True))["beta_F"], rtol=0, atol=0)

    def test_excess_only_skips_ideal_domain(self):
        data = self.data(True)
        data["dipole_density"][:] = 10
        model = self.model(True)
        with self.assertRaisesRegex(ValueError, "fixed dipoles"):
            model(data)
        result = model(data, compute_ideal=False)
        self.assertNotIn("beta_F_id", result)
        self.assertEqual(result["beta_F_exc"], 0)

    def test_model_roundtrip_contains_ideal(self):
        for polarized in (False, True):
            model, data = self.model(polarized), self.data(polarized)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.pt"
                save_model(model, path)
                restored = load_model(path)
            torch.testing.assert_close(model(data)["beta_F"], restored(data)["beta_F"])
            self.assertEqual(restored.ideal.requires_dipole_density, polarized)

    def test_solver_consumes_model_ideal(self):
        for polarized in (False, True):
            model, data = self.model(polarized), self.data(polarized)
            solver = GridSolver(model)
            result = solver.solve(data, particle_numbers=[2.4, 2.4], max_iter=4)
            torch.testing.assert_close(result["rho"], data["rho"])
            self.assertTrue(result["converged"])


if __name__ == "__main__":
    unittest.main()
