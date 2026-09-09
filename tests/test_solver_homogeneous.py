"""Constrained stationarity, normalization and metal-response regression tests."""
import unittest
import itertools

import torch
from torch import nn

from equicdft import GridData, GridSolver, GridCACEModel, LDAReadout, MetalWall, MetalElectrodeReadout
from equicdft._solver_symmetry import _HomogeneousDensityProjection


class IdealGas(nn.Module):
    def forward(self, data, compute_c1=True):
        out = {"beta_F_exc": data["rho"].sum()*0}
        if compute_c1:
            out["c1"] = torch.zeros_like(data["rho"])
        return out


class TestHomogeneousSolver(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)

    def data(self, types=3):
        d = GridData.from_dict({"grid_size": [4, 4, 6], "grid_spacing": .5,
                               "temperature": 1., "n_types": types},
                              cutoff_grid=0, boltzmann_constant=1.)
        x, y, z = d["grid_positions"].T
        d["V_ext"] = (.08*z[:, None] + .2*(1+.1*z[:, None])*torch.cos(torch.pi*x[:, None]/2)) * torch.arange(1, types+1)[None, :]
        d["excluded_mask"] = (z == 0) | (z == 5)
        return d

    def test_projection_values_and_gradients_match_explicit_group_means(self):
        generator = torch.Generator().manual_seed(42)
        for shape in ((4, 4, 6), (1, 3, 4)):
            positions = torch.cartesian_prod(*(torch.arange(n) for n in shape))
            for shuffled in (False, True):
                if shuffled:
                    positions = positions[torch.randperm(len(positions), generator=generator)]
                data = {"grid_size": shape, "grid_positions": positions}
                accessible = torch.ones(len(positions), dtype=torch.bool)
                for count in range(4):
                    for axes in itertools.combinations(range(3), count):
                        remaining = [a for a in range(3) if a not in axes]
                        same_group = (positions[:, None, remaining] == positions[None, :, remaining]).all(-1)
                        projection = _HomogeneousDensityProjection(data, axes, accessible)
                        for dtype in (torch.float32, torch.float64):
                            for channels in (1, 3):
                                with self.subTest(shape=shape, shuffled=shuffled, axes=axes,
                                                  dtype=dtype, channels=channels):
                                    values = torch.randn(len(positions), channels, generator=generator,
                                                         dtype=dtype, requires_grad=True)
                                    weights = same_group.to(dtype)
                                    weights = weights / weights.sum(1, keepdim=True)
                                    expected = weights @ values
                                    actual = projection.average(values)
                                    probe = torch.randn(values.shape, generator=generator, dtype=dtype)
                                    actual_grad, = torch.autograd.grad((actual * probe).sum(), values)
                                    expected_grad, = torch.autograd.grad((expected * probe).sum(), values)
                                    tolerance = 1e-6 if dtype == torch.float32 else 1e-14
                                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0)
                                    torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=0)

    def test_canonical_three_species_projects_gradient_not_boltzmann_density(self):
        d = self.data()
        n = torch.tensor([1., 2., 3.])
        p = _HomogeneousDensityProjection(d, (0, 1), ~d["excluded_mask"])
        # Crucially use exp(-mean(V)), not mean(exp(-V)).
        expected = torch.exp(-p.average(d["V_ext"]))
        expected[d["excluded_mask"]] = 0
        expected *= n/(expected.sum(0)*.125)
        result = GridSolver(IdealGas()).solve(d, particle_numbers=n, method="minimize",
                    homogeneous_axes=(0, 1), tolerance_residual=1e-9, max_iter=200)
        torch.testing.assert_close(result["rho"], expected, atol=1e-9, rtol=0)
        self.assertTrue(result["converged"])
        self.assertFalse(result["full_converged"])
        self.assertGreater(result["full_max_euler_lagrange_residual"], .5)
        self.assertLess(result["max_euler_lagrange_residual"], 1e-9)
        grid = result["rho"].reshape(4, 4, 6, 3)
        self.assertTrue(torch.equal(grid, grid[:1, :1].expand_as(grid)))
        self.assertTrue(all(a >= b for a, b in zip(result["objective_history"], result["objective_history"][1:])))

    def test_grand_canonical_and_permuted_rows(self):
        d = self.data(types=1)
        d["mu"] = torch.tensor([.3])
        perm = torch.randperm(len(d["V_ext"]))
        for key in ("V_ext", "excluded_mask", "grid_positions"):
            d[key] = d[key][perm]
        p = _HomogeneousDensityProjection(d, (0, 1), ~d["excluded_mask"])
        target = torch.exp(.3-p.average(d["V_ext"]))
        target[d["excluded_mask"]] = 0
        r = GridSolver(IdealGas()).solve(d, homogeneous_axes=(0, 1), method="minimize", tolerance_residual=1e-9)
        torch.testing.assert_close(r["rho"], target, atol=1e-9, rtol=0)
        self.assertTrue(r["converged"])

    def test_cap_and_canonical_gauge(self):
        d = self.data(types=1)
        d["V_ext"] *= 10
        s = GridSolver(IdealGas())
        kw = dict(particle_numbers=[2.], homogeneous_axes=(0, 1), method="minimize",
                  maximum_density=.3, tolerance_residual=1e-8, max_iter=300)
        a = s.solve(d, **kw)
        b = s.solve(dict(d, V_ext=d["V_ext"]+7.), **kw)
        self.assertTrue(a["converged"])
        torch.testing.assert_close(a["rho"], b["rho"], atol=1e-8, rtol=0)
        self.assertLessEqual(float(a["rho"].max()), .3+1e-12)
        self.assertAlmostEqual(float(a["rho"].sum()*.125), 2.)

    def test_empty_axes_preserve_default_exactly(self):
        d = self.data(types=1)
        s = GridSolver(IdealGas())
        a = s.solve(d, particle_numbers=[2.], method="minimize")
        b = s.solve(d, particle_numbers=[2.], method="minimize", homogeneous_axes=())
        self.assertTrue(torch.equal(a["rho"], b["rho"]))
        self.assertEqual(a["objective_history"], b["objective_history"])

    def test_named_restrictions_match_integer_axes_and_ideal_gas(self):
        d = self.data()
        d["excluded_mask"].zero_()
        d["V_ext"] += .07 * d["grid_positions"][:, 1:2]
        numbers = torch.tensor([1., 2., 3.])
        solver = GridSolver(IdealGas())
        for count in (1, 2, 3):
            for axes in itertools.combinations(range(3), count):
                names = ["xyz"[a] for a in axes]
                # Exercise both string and sequence forms for singleton axes.
                for restrict in ([names[0], names] if count == 1 else [names]):
                    with self.subTest(restrict=restrict):
                        options = dict(method="minimize", particle_numbers=numbers,
                                       tolerance_residual=1e-9, max_iter=200)
                        actual = solver.solve(d, homogeneous_axes=restrict, **options)
                        old = solver.solve(d, homogeneous_axes=axes, **options)
                        self.assertTrue(torch.equal(actual["rho"], old["rho"]))
                        self.assertEqual(actual["objective_history"], old["objective_history"])
                        self.assertEqual(actual["solver_homogeneous_axes"], list(axes))
                        potential = d["V_ext"].reshape(4, 4, 6, 3)
                        average = potential.mean(dim=axes, keepdim=True).expand_as(potential)
                        expected = torch.exp(-average).reshape(-1, 3)
                        expected *= numbers / (.125 * expected.sum(0))
                        torch.testing.assert_close(actual["rho"], expected, atol=1e-9, rtol=0)
                        self.assertTrue(actual["converged"])

    def test_named_empty_restrictions_preserve_defaults(self):
        d = self.data(types=1)
        solver = GridSolver(IdealGas())
        for method in ("minimize", "euler"):
            options = dict(method=method, particle_numbers=[2.], max_iter=5)
            expected = solver.solve(d, **options)
            for restrict in (None, [], ()):
                with self.subTest(method=method, restrict=restrict):
                    actual = solver.solve(d, homogeneous_axes=restrict, **options)
                    self.assertTrue(torch.equal(actual["rho"], expected["rho"]))
                    self.assertEqual(actual["objective_history"], expected["objective_history"])
                    self.assertEqual(actual.keys(), expected.keys())

    def test_invalid_named_axes_and_equivalent_index_duplicates(self):
        solver = GridSolver(IdealGas())
        d = self.data(types=1)
        options = dict(method="minimize", particle_numbers=[2.])
        for axes in ("", "xy", "X", "q", ["x", "x"], ["x", 0],
                     0, True, [None], ["x", 3]):
            with self.subTest(axes=axes), self.assertRaisesRegex(ValueError, "homogeneous_axes"):
                solver.solve(d, homogeneous_axes=axes, **options)
        with self.assertRaisesRegex(ValueError, "only with"):
            solver.solve(d, particle_numbers=[2.], homogeneous_axes="x")
        with self.assertRaisesRegex(ValueError, "accessibility"):
            solver.solve(d, homogeneous_axes="z", **options)
        with self.assertRaisesRegex(TypeError, "restrict"):
            solver.solve(d, restrict="x", **options)

    def test_named_restriction_with_permuted_rows_and_grand_canonical(self):
        d = self.data(types=1)
        d["mu"] = torch.tensor([.3])
        permutation = torch.arange(len(d["V_ext"])-1, -1, -1)
        for key in ("grid_positions", "V_ext", "excluded_mask"):
            d[key] = d[key][permutation]
        solver = GridSolver(IdealGas())
        options = dict(method="minimize", tolerance_residual=1e-9)
        actual = solver.solve(d, homogeneous_axes=("y", "x"), **options)
        old = solver.solve(d, homogeneous_axes=(1, 0), **options)
        self.assertTrue(torch.equal(actual["rho"], old["rho"]))
        self.assertTrue(actual["converged"])
        mixed = solver.solve(d, homogeneous_axes=(1, "x"), **options)
        self.assertTrue(torch.equal(mixed["rho"], old["rho"]))

    def test_other_and_all_axes(self):
        d = self.data(types=1)
        d["excluded_mask"].zero_()
        for axes in ((0, 2), (0, 1, 2)):
            p = _HomogeneousDensityProjection(d, axes, ~d["excluded_mask"])
            a = p.average(torch.rand_like(d["V_ext"]))
            torch.testing.assert_close(a, p.average(a), atol=1e-15, rtol=0)

    def test_reject_invalid_axes_grid_or_mask(self):
        for axes in ((0, 0), (-1,), (3,), (True,), (.5,)):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                GridSolver(IdealGas()).solve(self.data(), particle_numbers=[1., 2., 3.],
                                             method="minimize", homogeneous_axes=axes)
        with self.assertRaisesRegex(ValueError, "only with"):
            GridSolver(IdealGas()).solve(self.data(), particle_numbers=[1., 2., 3.], homogeneous_axes=(0, 1))
        d = self.data()
        d["excluded_mask"][1] = True
        with self.assertRaisesRegex(ValueError, "accessibility"):
            _HomogeneousDensityProjection(d, (0, 1), ~d["excluded_mask"])
        d = self.data()
        d["grid_positions"][1] = d["grid_positions"][0]
        with self.assertRaisesRegex(ValueError, "unique"):
            _HomogeneousDensityProjection(d, (0, 1), ~d["excluded_mask"])

    def test_metal_response_and_projected_directional_derivative(self):
        d = self.data(types=2)
        x, y, z = d["grid_positions"].T
        d["V_ext"] = .1*z[:, None]*torch.tensor([1., -1.])
        sites = d["excluded_mask"] & (x.remainder(2) == 0) & (y.remainder(2) == 0)
        d["metal_mask"] = torch.where(sites, 0, -1)
        d["metal_group_ids"] = torch.tensor([0])
        d["metal_total_charge"] = torch.tensor([0.])
        d["metal_external_field"] = torch.tensor([0., 0., -.1])
        d["metal_field_origin"] = torch.tensor([0., 0., 0.])
        d["metal_charge_units"] = "e"
        wall = MetalWall(charges=[1., -1.], sigma=.4, liquid_sigma=0.,
                         coulomb_amplitude=.02, boundary="periodic").double()
        model = GridCACEModel(a_features=None, b_features=None,
            readout=[LDAReadout(mean_density=1., n_types=2, hidden_sizes=(), zero_init=True),
                     MetalElectrodeReadout(wall, contribution="correction")],
            grid_spacing=.5, mean_temperature=1., boltzmann_constant=1., free_energy_mode="beta").double()
        solver = GridSolver(model)
        r = solver.solve(d, particle_numbers=[2., 2.], method="minimize", homogeneous_axes=(0, 1),
                         tolerance_residual=1e-7, max_iter=200)
        named = solver.solve(d, particle_numbers=[2., 2.], method="minimize",
                             homogeneous_axes=["x", "y"], tolerance_residual=1e-7, max_iter=200)
        self.assertTrue(torch.equal(named["rho"], r["rho"]))
        self.assertTrue(torch.equal(named["metal_q"], r["metal_q"]))
        self.assertTrue(r["converged"])
        self.assertLess(float(r["charge_residual"]), 1e-10)
        self.assertGreater(float(r["metal_q"].abs().max()), 1e-5)
        p = _HomogeneousDensityProjection(d, (0, 1), ~d["excluded_mask"])
        torch.testing.assert_close(r["rho"], p.average(r["rho"]), atol=1e-14, rtol=0)
        rho = r["rho"].detach()
        direction = torch.zeros_like(rho)
        direction[z == 2, 0] = 1.
        direction[z == 3, 0] = -1.
        # Test away from stationarity, where the derivative is nonzero.
        rho = rho + .01*direction
        ev = solver.evaluate(dict(d, rho=rho))
        projected_gradient = p.average(torch.log(rho.clamp_min(1e-300))+d["V_ext"]-ev["c1"])
        energy = []
        for sign in (1., -1.):
            v = solver.evaluate(dict(d, rho=rho+sign*1e-5*direction))
            energy.append(v["beta_F"]+v["beta_V_ext"])
        torch.testing.assert_close((energy[0]-energy[1])/2e-5,
                                   .125*(projected_gradient*direction).sum(), atol=1e-8, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
