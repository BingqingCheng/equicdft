"""Quadrupole-trace coupling of a molecular species to the applied potential."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from equicdft import (
    FixedDipoleIdeal, GridCACEModel, GridSolver, IdealGas, LDAReadout,
    load_model, save_model,
)
from equicdft.geometry.spectral import spectral_laplacian
from equicdft.model import EXTERNAL_NUMBER_KEY, LAPLACIAN_KEY
from equicdft.response.bulk import bulk_response_symbol
from equicdft.solver import Constraints, ConvergenceOptions

TRACE = 0.8476   # e A^2, SPC/E about the oxygen


class TestSecondMoment(unittest.TestCase):
    def setUp(self):
        self.previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)

    def tearDown(self):
        torch.set_default_dtype(self.previous)

    @staticmethod
    def data(n=4, n_types=1, polarized=False, laplacian=True):
        n_grid = n ** 3
        rho = 0.03 * (1 + 0.1 * torch.rand(n_grid, n_types))
        data = dict(
            rho=rho, temperature=torch.tensor(1.0), beta=torch.tensor(1.0),
            grid_spacing=torch.full((3,), 2.0), grid_size=torch.tensor([n, n, n]),
            V_ext=0.2 * torch.randn(n_grid, n_types),
        )
        if laplacian:
            data[LAPLACIAN_KEY] = 0.05 * torch.randn(n_grid, 1)
        if polarized:
            data.update(dipole_density=torch.zeros(n_grid, n_types, 3),
                        E_ext=torch.zeros(n_grid, n_types, 3))
        return data

    @staticmethod
    def model(second_moment=TRACE, n_types=1, excess=True, polarized=False):
        return GridCACEModel(
            None, None,
            [LDAReadout(n_types=n_types, hidden_sizes=(4,))] if excess else [],
            ideal=FixedDipoleIdeal([0.5] * n_types) if polarized else IdealGas(),
            n_types=n_types, grid_spacing=2.0, boltzmann_constant=1.0,
            mean_density=0.03, polarization_scale=0.1 if polarized else None,
            compute_c1=True, compute_local_mu=True, second_moment=second_moment,
            compute_polarization_derivative=polarized,
        )

    def test_off_by_default_and_ignores_the_field(self):
        model = self.model(second_moment=None)
        self.assertIsNone(model.second_moment)
        self.assertIsNone(model.to_config()["second_moment"])
        data = self.data()
        torch.testing.assert_close(model.external_potential(data), data["V_ext"], rtol=0, atol=0)
        self.assertIs(model.resolve_external_potential(data), data)
        without = dict(data)
        without.pop(LAPLACIAN_KEY)
        torch.testing.assert_close(model(data)["beta_mu_local"], model(without)["beta_mu_local"], rtol=0, atol=0)

    def test_chemical_potential_shifts_by_the_trace_term(self):
        data = self.data()
        reference = self.model(second_moment=None)
        model = self.model()
        model.load_state_dict(reference.state_dict())
        base = reference(data)
        with_term = model(data)
        shift = data["beta"] * TRACE / 6.0 * data[LAPLACIAN_KEY]
        torch.testing.assert_close(with_term["beta_mu_local"], base["beta_mu_local"] + shift)
        # The intrinsic functional is untouched: the coupling is external.
        for key in ("beta_F", "beta_F_exc", "density_derivative", "c1"):
            torch.testing.assert_close(with_term[key], base[key], rtol=0, atol=0)
        # Zero Laplacian (a uniform field) changes nothing.
        data[LAPLACIAN_KEY].zero_()
        torch.testing.assert_close(model(data)["beta_mu_local"], base["beta_mu_local"], rtol=0, atol=0)

    def test_per_species_trace_and_shapes(self):
        n_types = 2
        data = self.data(n_types=n_types)
        model = self.model(second_moment=[TRACE, 0.0], n_types=n_types)
        effective = model.external_potential(data)
        torch.testing.assert_close(effective[:, 0], data["V_ext"][:, 0] + TRACE / 6 * data[LAPLACIAN_KEY][:, 0])
        torch.testing.assert_close(effective[:, 1], data["V_ext"][:, 1], rtol=0, atol=0)
        # A per-species Laplacian is accepted too.
        data[LAPLACIAN_KEY] = data[LAPLACIAN_KEY].expand(-1, n_types).clone()
        torch.testing.assert_close(model.external_potential(data), effective)
        self.assertEqual(model.to_config()["second_moment"], [TRACE, 0.0])
        with self.assertRaisesRegex(ValueError, "one value per type"):
            self.model(second_moment=[1.0, 2.0, 3.0], n_types=n_types)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model(second_moment=float("nan"))
        # Complex traces are rejected, not truncated to their real part.
        for value in (torch.tensor(TRACE + 0.1j), torch.tensor([TRACE + 0j]), True):
            with self.assertRaisesRegex(ValueError, "real"):
                self.model(second_moment=value)

    def test_missing_or_malformed_laplacian(self):
        model = self.model()
        data = self.data(laplacian=False)
        with self.assertRaisesRegex(ValueError, LAPLACIAN_KEY):
            model(data)
        with self.assertRaisesRegex(ValueError, LAPLACIAN_KEY):
            model.external_potential(data)
        bad = self.data()
        bad[LAPLACIAN_KEY] = bad[LAPLACIAN_KEY][:-1]
        with self.assertRaisesRegex(ValueError, "shape"):
            model.external_potential(bad)
        bad = self.data()
        bad[LAPLACIAN_KEY][0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            model.external_potential(bad)

    def test_resolution_is_idempotent(self):
        model = self.model()
        data = self.data()
        resolved = model.resolve_external_potential(data)
        self.assertIsNot(resolved, data)
        self.assertTrue(EXTERNAL_NUMBER_KEY.startswith("_"))
        torch.testing.assert_close(resolved[EXTERNAL_NUMBER_KEY], data["V_ext"], rtol=0, atol=0)
        torch.testing.assert_close(resolved["V_ext"], model.external_potential(data))
        # Already resolved batches are left alone, also by the diagnostic.
        self.assertIs(model.resolve_external_potential(resolved), resolved)
        torch.testing.assert_close(model.external_potential(resolved), resolved["V_ext"], rtol=0, atol=0)
        torch.testing.assert_close(model(resolved)["beta_mu_local"], model(data)["beta_mu_local"])

    def test_public_keys_never_switch_the_coupling_off(self):
        # A loader may carry the products' raw column names; none of them may
        # be mistaken for the private resolution marker.
        model = self.model()
        data = self.data()
        expected = model.external_potential(data)
        for key in ("V_ext_number", "phi_ext", "E_ext", "V_ext_total"):
            carried = dict(data)
            carried[key] = data["V_ext"].clone()
            torch.testing.assert_close(model.external_potential(carried), expected, rtol=0, atol=0)
            resolved = model.resolve_external_potential(carried)
            self.assertIsNot(resolved, carried)
            torch.testing.assert_close(resolved["V_ext"], expected, rtol=0, atol=0)
            torch.testing.assert_close(model(carried)["beta_mu_local"], model(data)["beta_mu_local"], rtol=0, atol=0)

    def test_config_and_file_round_trip(self):
        model = self.model()
        rebuilt = GridCACEModel.from_config(model.to_config())
        torch.testing.assert_close(rebuilt.second_moment, model.second_moment)
        config = model.to_config()
        config.pop("second_moment")             # a configuration written before the option
        self.assertIsNone(GridCACEModel.from_config(config).second_moment)
        self.assertNotIn("second_moment", model.state_dict())
        data = self.data()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_model(model, path)
            restored = load_model(path)
        torch.testing.assert_close(restored.second_moment, model.second_moment)
        torch.testing.assert_close(restored(data)["beta_mu_local"], model(data)["beta_mu_local"])

    def test_solver_objective_and_ideal_gas_profile(self):
        # An ideal gas in a plane-wave potential: the solved density follows
        # exp(-beta V_eff) with V_eff = V_ext + Tr M / 6 * laplacian(phi).
        n = 6
        n_grid = n ** 3
        spacing = 2.0
        k = 2 * math.pi / (n * spacing)
        x = (torch.arange(n, dtype=torch.float64) + 0.5) * spacing
        phase = torch.cos(k * x)[:, None, None].expand(n, n, n).reshape(-1, 1)
        phi = 0.3 * phase                              # applied potential
        laplacian = -k ** 2 * phi                      # its exact Laplacian
        data = dict(
            rho=torch.full((n_grid, 1), 0.03), temperature=torch.tensor(1.0),
            beta=torch.tensor(1.0), grid_spacing=torch.full((3,), spacing),
            grid_size=torch.tensor([n, n, n]), V_ext=0.1 * phase,
        )
        data[LAPLACIAN_KEY] = laplacian
        torch.testing.assert_close(
            spectral_laplacian(phi[None], (n, n, n), spacing)[0], laplacian, rtol=1e-10, atol=1e-12,
        )
        for trace in (None, TRACE):
            model = self.model(second_moment=trace, excess=False)
            solver = GridSolver(model)
            evaluation = solver.evaluate(data)
            effective = data["V_ext"] + (0.0 if trace is None else trace / 6 * laplacian)
            expected = (data["rho"] * effective).sum() * spacing ** 3
            torch.testing.assert_close(evaluation["beta_V_ext"], expected)
            total = float(data["rho"].sum() * spacing ** 3)
            result = solver.solve(
                data, convergence=ConvergenceOptions(max_iter=200),
                constraints=Constraints(particle_numbers=[total]),
            )
            self.assertTrue(result["converged"])
            boltzmann = torch.exp(-effective)
            profile = total * boltzmann / (boltzmann.sum() * spacing ** 3)
            torch.testing.assert_close(result["rho"], profile, rtol=1e-3, atol=1e-8)
            torch.testing.assert_close(result["V_ext"], effective)

    def test_bulk_symbol_is_unchanged(self):
        n = 4
        positions = torch.tensor(np.indices((n,) * 3).reshape(3, -1).T)
        batch = dict(
            rho=torch.full((1, n ** 3, 1), 0.03), dipole_density=torch.zeros(1, n ** 3, 1, 3),
            temperature=torch.tensor([1.0]), beta=torch.tensor([1.0]),
            grid_spacing=torch.full((1, 3), 2.0), grid_size=torch.tensor([[n, n, n]]),
            grid_positions=positions[None],
        )
        indices = torch.tensor([[[1, 0, 0], [1, 1, 0]]])
        reference = self.model(second_moment=None, polarized=True)
        model = self.model(polarized=True)
        model.load_state_dict(reference.state_dict())
        torch.testing.assert_close(
            bulk_response_symbol(model, batch, indices),
            bulk_response_symbol(reference, batch, indices), rtol=0, atol=0,
        )


if __name__ == "__main__":
    unittest.main()
