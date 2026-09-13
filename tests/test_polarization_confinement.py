"""Hard exclusion and full-Coulomb ideal-dipole equilibrium checks."""

import unittest

import torch

from equicdft import GridCACEModel, MetalWall, PolarizationSolver, ReciprocalFeatures
from test_metalwall_polarization import _case, _liquid, _oracle
import test_polarization_ideal as ideal_reference


class TestPolarizationConfinement(unittest.TestCase):
    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(53)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)

    def test_confined_ideal_matches_angular_integral_both_starts(self):
        data = ideal_reference.TestPolarizationSolver.case()
        mask = torch.arange(24) % 3 == 0
        data["excluded_mask"] = mask
        active_data = dict(data, V_ext=data["V_ext"][~mask], E_ext=data["E_ext"][~mask])
        numbers, moments = torch.tensor([3., 4.]), torch.tensor([.7, 1.9])
        expected_rho, expected_p = ideal_reference.exact_solution(active_data, numbers, moments)
        solver = PolarizationSolver(moments)
        for perturbed in (False, True):
            initial = {}
            if perturbed:
                rho = torch.zeros_like(data["V_ext"])
                rho[~mask] = torch.exp(.3 * torch.randn(16, 2))
                rho *= numbers / (data["grid_spacing"].prod() * rho.sum(0))
                p = .02 * (rho * moments)[..., None] * torch.randn(24, 2, 3)
                initial = dict(initial_rho=rho, initial_polarization=p)
            result = solver.solve(data, numbers, tolerance_residual=1e-9, **initial)
            self.assertTrue(result["converged"])
            self.assertLess(result["maximum_residual"].item(), 1e-9)
            torch.testing.assert_close(result["rho"][~mask], expected_rho, atol=1e-9, rtol=0)
            torch.testing.assert_close(result["dipole_density"][~mask], expected_p, atol=1e-9, rtol=0)
            for key in ("rho", "dipole_density", "density_residual", "polarization_residual"):
                self.assertEqual(torch.count_nonzero(result[key][mask]).item(), 0)
            torch.testing.assert_close(result["particle_numbers"], numbers, atol=1e-13, rtol=0)

        # Excluded finite fields cannot change the accessible solution or gauge.
        changed = {**data, "V_ext": data["V_ext"].clone(), "E_ext": data["E_ext"].clone()}
        changed["V_ext"][mask] = 1000.
        changed["E_ext"][mask] = -3000.
        changed["V_ext"][~mask] += torch.tensor([2., -3.])
        shifted = solver.solve(changed, numbers, tolerance_residual=1e-9)
        torch.testing.assert_close(shifted["rho"], result["rho"], atol=1e-9, rtol=0)
        torch.testing.assert_close(shifted["dipole_density"], result["dipole_density"], atol=1e-9, rtol=0)
        torch.testing.assert_close(shifted["beta_mu"] - result["beta_mu"], data["beta"] * torch.tensor([2., -3.]), atol=1e-9, rtol=0)

    def test_invalid_masks_and_excluded_initial_fields(self):
        data = ideal_reference.TestPolarizationSolver.case()
        solver = PolarizationSolver([.7, 1.9])
        for mask in (torch.ones(24, dtype=torch.bool), torch.zeros(24), torch.zeros(24, 2, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                solver.solve(dict(data, excluded_mask=mask), [3., 4.])
        data["excluded_mask"] = torch.arange(24) == 0
        result = solver.solve(data, [3., 4.])
        for name in ("rho", "dipole_density"):
            bad = result[name].clone()
            bad[0] = .01
            with self.assertRaisesRegex(ValueError, "zero in excluded"):
                solver.evaluate({**data, "rho": result["rho"], "dipole_density": result["dipole_density"], name: bad})

    def test_full_coulomb_matches_independent_metal_oracle(self):
        for shape in ((3, 5, 3), (4, 3, 2)):
            data, sites, charges = _case(shape, neutral_species=True)
            liquid = _liquid(charges, alpha=0.)
            energy, outputs = MetalWall(liquid, sites, metal_sigma=.35).energy_and_outputs(data)
            expected = _oracle(data, sites, charges, alpha=0.)
            torch.testing.assert_close(energy, expected["total_energy"], atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(outputs["metal_site_q"], expected["metal_site_q"], atol=1e-12, rtol=1e-12)
            self.assertGreaterEqual(energy.item(), -1e-12)
        for kernel in ("gaussian", "screened_inverse_laplacian"):
            with self.assertRaises(ValueError):
                ReciprocalFeatures((0.,), kernel=kernel)
        with self.assertRaises(ValueError):
            ReciprocalFeatures((-.1,), kernel="coulomb")

    def test_manufactured_confined_metal_equilibrium(self):
        shape = (3, 3, 8)
        data, sites, charges = _case(shape, neutral_species=True, spacing=(.8, .8, .8))
        mask = (torch.arange(72) % 8 == 0) | (torch.arange(72) % 8 == 7)
        data["excluded_mask"] = mask
        data["rho"][mask] = 0
        data["dipole_density"] *= .1
        data["dipole_density"][mask] = 0
        # Two off-grid planes, one neutral conducting group, finite field.
        sites["metal_positions"] = torch.tensor([[.2, .2, .1], [1.3, 1.4, .1], [.2, .2, 5.9], [1.3, 1.4, 5.9]])
        sites["metal_site_groups"] = torch.zeros(4, dtype=torch.long)
        sites["metal_group_ids"] = torch.tensor([0])
        sites["metal_total_charge"] = torch.zeros(1)
        wall = MetalWall(_liquid(charges, alpha=0., amplitude=.3), sites, .35, external_field=(0., 0., -.03))
        model = GridCACEModel(None, None, [wall], grid_spacing=.8).eval()
        solver = PolarizationSolver(.7, model)
        data["V_ext"] = torch.zeros_like(data["rho"])
        data["E_ext"] = torch.zeros_like(data["dipole_density"])
        target = solver.evaluate(data)
        data["V_ext"] = -target["density_residual"] / data["beta"]
        data["E_ext"] = target["polarization_residual"] / data["beta"]
        result = solver.solve(data, target["particle_numbers"],
                              initial_rho=torch.where(mask[:, None], 0., target["particle_numbers"] / ((~mask).sum() * .8**3)),
                              initial_polarization=torch.zeros_like(data["dipole_density"]), tolerance_residual=1e-8)
        self.assertTrue(result["converged"], result["status"])
        torch.testing.assert_close(result["rho"], data["rho"], atol=1e-8, rtol=0)
        torch.testing.assert_close(result["dipole_density"], data["dipole_density"], atol=1e-8, rtol=0)
        endpoint = model(dict(data, rho=result["rho"], dipole_density=result["dipole_density"]), compute_c1=False)
        self.assertLess(endpoint["charge_residual"].item(), 1e-10)
        self.assertLess(endpoint["potential_residual"].item(), 1e-10)


if __name__ == "__main__":
    unittest.main()
