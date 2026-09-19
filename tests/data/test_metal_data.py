"""Liquid-grid exclusion behavior after electrode metadata was separated."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from torch import nn

from equicdft import GridData, GridSolver


def _grid_atoms():
    shape = (4, 2, 2)
    positions = np.indices(shape).reshape(3, -1).T
    excluded = np.zeros(len(positions), dtype=bool)
    excluded[[0, 1, 2]] = True
    rho = np.full((len(positions), 2), .2)
    rho[excluded] = 0.
    order = np.random.default_rng(11).permutation(len(positions))
    atoms = Atoms("X" * len(positions), positions=positions[order],
                  cell=np.diag(shape), pbc=True)
    atoms.arrays.update(
        density=rho[order], V_ext=np.zeros_like(rho),
        excluded_mask=excluded[order],
    )
    atoms.info.update(grid_size=np.array(shape), grid_spacing=1., T=1.)
    return atoms, excluded


class _IdealGas(nn.Module):
    def forward(self, data, compute_c1=True):
        rho = data["rho"]
        result = {"beta_F_exc": rho.sum(dim=(-2, -1)) * 0.}
        if compute_c1:
            result["c1"] = torch.zeros_like(rho)
        return result


class TestGridExclusion(unittest.TestCase):
    def _read(self, atoms):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.extxyz"
            write(path, atoms, format="extxyz")
            return GridData.from_xyz(
                path, cutoff_grid=0, boltzmann_constant=1.,
            )[0]

    def test_extxyz_preserves_exclusion_and_zero_density(self):
        atoms, excluded = _grid_atoms()
        frame = self._read(atoms)
        self.assertTrue(torch.equal(frame["excluded_mask"], torch.tensor(excluded)))
        self.assertTrue(torch.all(frame["rho"][excluded] == 0.))
        self.assertTrue(torch.all(frame["c1_plus_beta_mu"][excluded] == 0.))

    def test_nonzero_density_on_exclusion_is_rejected(self):
        atoms, _ = _grid_atoms()
        atoms.arrays["density"][atoms.arrays["excluded_mask"]] = .01
        with self.assertRaisesRegex(ValueError, "rho must be zero"):
            self._read(atoms)

    def test_exclusions_must_leave_accessible_grid_points(self):
        with self.assertRaisesRegex(ValueError, "accessible"):
            GridData.from_dict(
                dict(grid_size=[2, 1, 1], grid_spacing=1., temperature=1.,
                     n_types=1, excluded_mask=[True, True]),
                cutoff_grid=0,
            )

    def test_solver_preserves_exclusions(self):
        data = self._read(_grid_atoms()[0])
        result = GridSolver(_IdealGas()).solve(
            data, method="minimize", particle_numbers=[1.3, 2.6],
            max_iter=20, tolerance_residual=1.e-6,
        )
        self.assertTrue(result["converged"])
        self.assertTrue(torch.all(result["rho"][data["excluded_mask"]] == 0.))
        torch.testing.assert_close(result["rho"].sum(0), torch.tensor([1.3, 2.6]))


if __name__ == "__main__":
    unittest.main()
