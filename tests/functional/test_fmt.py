"""Fundamental measure theory with analytic weight spectra."""

import math
import os
import tempfile
import unittest

import numpy as np
import torch

from equicdft import (
    FMTReadout,
    FourierResponse,
    GridCACEModel,
    GridData,
    GridSolver,
    MSAReferenceReadout,
    load_model,
    save_model,
)
from equicdft._config import build, registered_types
from equicdft.functional.fmt import (
    FMT_THIRD_TERMS,
    carnahan_starling_free_energy_density,
    fmt_free_energy_density,
    scaled_particle_free_energy_density,
)
from equicdft.solver.options import ConvergenceOptions, EulerOptions


RADIUS = 0.5
SPACING = 0.25
GRID = 32

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy < 2.0 has only trapz


def _homogeneous_batch(densities, grid=GRID, spacing=SPACING, n_types=2):
    """One homogeneous field per entry of ``densities`` (per-species value)."""

    n_fields = len(densities)
    positions = torch.tensor(np.indices((grid, grid, grid)).reshape(3, -1).T)
    return {
        "rho": torch.stack([torch.full((grid ** 3, n_types), value, dtype=torch.float64) for value in densities]),
        "temperature": torch.ones(n_fields, dtype=torch.float64),
        "grid_spacing": torch.full((n_fields, 3), spacing, dtype=torch.float64),
        "grid_size": torch.tensor([[grid, grid, grid]] * n_fields),
        "grid_positions": torch.stack([positions] * n_fields),
    }


def _fmt(**overrides):
    arguments = dict(radii=(RADIUS, RADIUS), version="rosenfeld", third_term="q3")
    arguments.update(overrides)
    n_types = len(arguments["radii"])
    return FMTReadout(n_types, **arguments)


def _model(readouts, spacing=SPACING, mean_density=0.4, **kwargs):
    arguments = dict(
        a_features=None, b_features=None, readout=readouts, grid_spacing=spacing,
        mean_temperature=1.0, boltzmann_constant=1.0, compute_c1=False,
        mean_density=mean_density,
    )
    arguments.update(kwargs)
    return GridCACEModel(**arguments).to(dtype=torch.float64)


def _bulk(version, rho):
    eta = math.pi * rho * (2 * RADIUS) ** 3 / 6
    if version == "rosenfeld":
        return scaled_particle_free_energy_density(rho, eta)
    return carnahan_starling_free_energy_density(rho, eta)


def _percus_yevick_inverse_structure(k, rho):
    eta = math.pi * rho / 6
    r = np.linspace(1e-9, 1.0, 100001)
    c_hs = -((1 + 2 * eta) ** 2 - 6 * eta * (1 + eta / 2) ** 2 * r
             + 0.5 * eta * (1 + 2 * eta) ** 2 * r ** 3) / (1 - eta) ** 4
    return 1 - rho * _trapezoid(4 * np.pi * r ** 2 * np.sinc(k * r / np.pi) * c_hs, r)


class TestBulkLimits(unittest.TestCase):
    def test_homogeneous_free_energy_is_the_bulk_equation_of_state(self):
        volume = (GRID * SPACING) ** 3
        for version in ("rosenfeld", "white_bear_i"):
            for third_term in FMT_THIRD_TERMS:
                with self.subTest(version=version, third_term=third_term):
                    outputs = _model([_fmt(version=version, third_term=third_term)])(_homogeneous_batch([0.2, 0.3]), compute_c1=True)
                    for field, per_species in enumerate((0.2, 0.3)):
                        rho = 2 * per_species
                        self.assertAlmostEqual(outputs["beta_F_exc"][field].item() / (volume * _bulk(version, rho)), 1.0, places=12)
                    # c1 is uniform and equals minus the bulk excess chemical potential.
                        density = torch.tensor(rho, dtype=torch.float64, requires_grad=True)
                        mu_excess, = torch.autograd.grad(_bulk(version, density), density)
                        torch.testing.assert_close(outputs["c1"][field], torch.full_like(outputs["c1"][field], -mu_excess.item()))
        # "white_bear" is accepted as the mark I form.
        self.assertEqual(_fmt(version="white_bear").version, "white_bear_i")

    def test_weighted_densities_of_a_uniform_field_are_the_fundamental_measures(self):
        readout = _fmt(radii=(0.5, 0.3))
        rho = torch.tensor([0.2, 0.1], dtype=torch.float64)
        grid = rho.reshape(1, 1, 1, 1, 2).expand(1, GRID, GRID, GRID, 2).movedim(-1, 1).contiguous()
        weighted = readout.weighted_densities(grid, (GRID, GRID, GRID), torch.full((3,), SPACING, dtype=torch.float64))
        radii = torch.tensor(readout.radii, dtype=torch.float64)
        expected = {
            "n3": (rho * 4 * math.pi * radii ** 3 / 3).sum(), "n2": (rho * 4 * math.pi * radii ** 2).sum(),
            "n1": (rho * radii).sum(), "n0": rho.sum(),
        }
        for name, value in expected.items():
            torch.testing.assert_close(weighted[name], torch.full_like(weighted[name], value.item()))
        for name in ("n1v", "n2v"):
            self.assertLess(weighted[name].abs().max().item(), 1e-12)
        with_tensor = readout.weighted_densities(grid, (GRID, GRID, GRID), torch.full((3,), SPACING, dtype=torch.float64), tensor=True)
        expected_tensor = expected["n2"].item() / 3 * torch.eye(3, dtype=torch.float64)
        torch.testing.assert_close(with_tensor["n2t"], expected_tensor.expand_as(with_tensor["n2t"]))

    def test_low_density_limit_is_the_second_virial_coefficient(self):
        volume = (GRID * SPACING) ** 3
        for version in ("rosenfeld", "white_bear_i"):
            with self.subTest(version=version):
                energy = _model([_fmt(version=version)], mean_density=1e-4)(_homogeneous_batch([5e-5]), compute_c1=False)["beta_F_exc"]
                rho = 1e-4
                self.assertAlmostEqual(energy.item() / (volume * rho ** 2) / (2 * math.pi / 3), 1.0, places=3)

    def test_white_bear_reduces_to_rosenfeld_at_low_packing(self):
        # Isolate the third term: n0 = n1 = 0, n2 = 1, no vectors.
        n3 = torch.tensor([0.3, 0.1, 1e-3, 1e-6, 0.0], dtype=torch.float64)
        zero = torch.zeros_like(n3)
        one = torch.ones_like(n3)
        vector = torch.zeros(5, 3, dtype=torch.float64)
        rosenfeld = fmt_free_energy_density(zero, zero, one, n3, vector, vector, "rosenfeld")
        white_bear = fmt_free_energy_density(zero, zero, one, n3, vector, vector, "white_bear_i")
        ratio = white_bear / rosenfeld
        expected = 2.0 / 3.0 * (n3[:2] + (1 - n3[:2]) ** 2 * torch.log1p(-n3[:2])) / n3[:2] ** 2
        torch.testing.assert_close(ratio[:2], expected)
        self.assertLess(ratio[0].item(), 0.95)
        torch.testing.assert_close(ratio[2:], torch.ones(3, dtype=torch.float64), atol=1e-3, rtol=0)
        self.assertTrue(torch.isfinite(white_bear).all())
        with self.assertRaisesRegex(ValueError, "version"):
            fmt_free_energy_density(zero, zero, one, n3, vector, vector, "other")
        with self.assertRaisesRegex(ValueError, "third_term"):
            fmt_free_energy_density(zero, zero, one, n3, vector, vector, "rosenfeld", "other")
        with self.assertRaisesRegex(ValueError, "n2t"):
            fmt_free_energy_density(zero, zero, one, n3, vector, vector, "rosenfeld", "tensor")


class TestFMTReadout(unittest.TestCase):
    K_INDICES = ((1, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0), (8, 0, 0), (12, 0, 0))

    def test_rosenfeld_second_derivative_is_percus_yevick_at_every_density(self):
        batch = _homogeneous_batch([0.2, 0.3])
        indices = torch.tensor([list(self.K_INDICES)] * 2)
        for third_term in FMT_THIRD_TERMS:
            with self.subTest(third_term=third_term):
                curvature, valid = FourierResponse(relative_amplitude=0.01)(
                    _model([_fmt(third_term=third_term)]), batch, indices, torch.tensor([[1.0, 1.0]]),
                )
                self.assertTrue(bool(valid[:, :, 0, :].all()))
                for field, per_species in enumerate((0.2, 0.3)):
                    rho = 2 * per_species
                    expected = [_percus_yevick_inverse_structure(2 * math.pi * i[0] / (GRID * SPACING), rho) for i in self.K_INDICES]
                    np.testing.assert_allclose(curvature[field, :, 0, 0].numpy(), expected, rtol=1e-3)

    @staticmethod
    def _peak_batch(grid, spacing, width, offset, occupancy=0.95):
        """A Gaussian of ``occupancy`` particles centred on a grid point (``offset=0``) or a cell centre (``0.5``)."""

        positions = torch.tensor(np.indices((grid, grid, grid)).reshape(3, -1).T)
        center = torch.full((3,), grid / 2, dtype=torch.float64) + offset
        distance_squared = ((positions.double() - center) * spacing).square().sum(-1)
        rho = torch.exp(-0.5 * distance_squared / width ** 2)
        rho = occupancy * rho / (rho.sum() * spacing ** 3)
        return {
            "rho": rho[None, :, None], "temperature": torch.ones(1, dtype=torch.float64),
            "grid_spacing": torch.full((1, 3), spacing, dtype=torch.float64),
            "grid_size": torch.tensor([[grid, grid, grid]]), "grid_positions": positions[None],
        }

    def test_resolved_peaks_are_translation_invariant_on_the_grid(self):
        """A peak at least one spacing wide gives the same energy on a grid point and between grid points."""

        for grid, spacing in ((16, 0.25), (32, 0.125)):
            for term in FMT_THIRD_TERMS:
                model = _model([_fmt(radii=(RADIUS,), third_term=term)], spacing=spacing)
                for width in (spacing, 1.5 * spacing):
                    with self.subTest(spacing=spacing, term=term, width=width):
                        on_point, between = (
                            model(self._peak_batch(grid, spacing, width, offset), compute_c1=False)["beta_F_exc"].item()
                            for offset in (0.0, 0.5)
                        )
                        if term != "original":  # the 1989 form is negative for a peak narrower than R
                            self.assertGreater(on_point, 0.0)
                        self.assertLess(abs(on_point - between), 1.0e-2 * abs(between))

    def test_unresolved_peaks_stay_nonnegative_with_the_crossover_term(self):
        """Below the grid spacing the density is not represented: the energy depends on the alignment (as the ideal-gas term's does) but never turns negative with ``"q3"``, while the original third term diverges."""

        for grid, spacing in ((16, 0.25), (32, 0.125)):
            models = {term: _model([_fmt(radii=(RADIUS,), third_term=term)], spacing=spacing) for term in ("q3", "original")}
            for width in (0.4 * spacing, 0.2 * spacing):
                for offset in (0.0, 0.5):
                    with self.subTest(spacing=spacing, width=width, offset=offset):
                        batch = self._peak_batch(grid, spacing, width, offset)
                        readout = models["q3"].readout[0]
                        rho_grid = batch["rho"][:, :, 0].reshape(1, 1, grid, grid, grid)
                        weighted = readout.weighted_densities(rho_grid, (grid, grid, grid), batch["grid_spacing"][0])
                        phi = fmt_free_energy_density(
                            weighted["n0"], weighted["n1"], weighted["n2"], readout._guarded_packing(weighted["n3"]),
                            weighted["n1v"], weighted["n2v"], "white_bear_i", "q3",
                        )
                        # Nonnegative pointwise up to round-off of the clamped bounds.
                        self.assertGreaterEqual(phi.min().item(), -1.0e-12 * phi.max().item())
                        energy = models["q3"](batch, compute_c1=False)["beta_F_exc"].item()
                        self.assertTrue(math.isfinite(energy) and energy >= 0.0, energy)
            # The original scalar-vector form has no lower bound: it is
            # negative and still falling at the sharpest peak between grid points.
            values = [
                models["original"](self._peak_batch(grid, spacing, width, 0.5), compute_c1=False)["beta_F_exc"].item()
                for width in (0.4 * spacing, 0.2 * spacing)
            ]
            self.assertLess(values[-1], 0.0)
            self.assertLess(values[-1], values[0])

    def test_guard_leaves_a_resolved_field_untouched(self):
        """The continuum bounds are inactive for a smooth density: guarded and unguarded weighted densities are identical."""

        torch.manual_seed(0)
        grid, spacing = 16, 0.25
        readout = _fmt(radii=(RADIUS,))
        noise = torch.fft.rfftn(torch.randn(grid, grid, grid, dtype=torch.float64))
        axes = [torch.fft.fftfreq(grid, d=spacing), torch.fft.fftfreq(grid, d=spacing), torch.fft.rfftfreq(grid, d=spacing)]
        k_squared = (2 * math.pi) ** 2 * (axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2 + axes[2][None, None, :] ** 2)
        smooth = torch.fft.irfftn(noise * torch.exp(-0.5 * k_squared * 0.3 ** 2), s=(grid, grid, grid))
        rho_grid = (0.4 * (1 + 0.5 * smooth / smooth.abs().max())).reshape(1, 1, grid, grid, grid)
        spacing_tensor = torch.full((3,), spacing, dtype=torch.float64)
        guarded = readout.weighted_densities(rho_grid, (grid, grid, grid), spacing_tensor, tensor=True)
        unguarded = readout.weighted_densities(rho_grid, (grid, grid, grid), spacing_tensor, tensor=True, guard=False)
        self.assertIn("n2t", guarded)
        for key in guarded:
            self.assertTrue(torch.equal(guarded[key], unguarded[key]), key)

    def test_guard_keeps_the_tensor_trace_identity(self):
        """``Tr n2t = n2`` follows from the tensor weight and survives the guard where the guard is active."""

        for grid, spacing in ((16, 0.25), (32, 0.125)):
            readout = _fmt(radii=(RADIUS,), third_term="tensor")
            for offset in (0.0, 0.5):
                with self.subTest(spacing=spacing, offset=offset):
                    batch = self._peak_batch(grid, spacing, 0.2 * spacing, offset)
                    rho_grid = batch["rho"][:, :, 0].reshape(1, 1, grid, grid, grid)
                    raw = readout.weighted_densities(rho_grid, (grid, grid, grid), batch["grid_spacing"][0], guard=False)
                    self.assertGreater((raw["n2"] < 0).sum().item(), 100)  # the guard is active
                    guarded = readout.weighted_densities(rho_grid, (grid, grid, grid), batch["grid_spacing"][0])
                    trace = torch.diagonal(guarded["n2t"], dim1=-2, dim2=-1).sum(dim=-1)
                    torch.testing.assert_close(trace, guarded["n2"], rtol=0.0, atol=1.0e-12)
                    self.assertTrue(torch.all(guarded["n2t"][raw["n2"] < 0] == 0.0).item())
                    energy = _model([readout], spacing=spacing)(batch, compute_c1=False)["beta_F_exc"].item()
                    self.assertTrue(math.isfinite(energy))

    def test_uniform_curvature_is_the_compressibility_of_each_version(self):
        volume = (GRID * SPACING) ** 3
        rho = 0.4
        eta = math.pi * rho / 6
        expected = {
            "rosenfeld": (1 + 2 * eta) ** 2 / (1 - eta) ** 4,
            "white_bear_i": (1 + 4 * eta + 4 * eta ** 2 - 4 * eta ** 3 + eta ** 4) / (1 - eta) ** 4,
        }
        batch = _homogeneous_batch([rho / 2])
        for version, value in expected.items():
            with self.subTest(version=version):
                model = _model([_fmt(version=version)])

                def slope(scale):
                    field = (batch["rho"] * scale).requires_grad_(True)
                    energy = model({**batch, "rho": field}, compute_c1=False)["beta_F_exc"].sum()
                    gradient, = torch.autograd.grad(energy, field)
                    return (gradient * batch["rho"]).sum().item()

                step = 1e-4
                curvature = (slope(1 + step) - slope(1 - step)) / (2 * step)
                self.assertAlmostEqual(1 + curvature / (volume * rho), value, places=4)

    def test_hard_wall_contact_theorem(self):
        """A slab between hard walls: contact density ~ beta P, bulk recovered mid-slab."""

        spacing = 0.125
        nx, ny, nz = 8, 8, 96
        rho_bulk = 0.4
        for version in ("white_bear_i", "rosenfeld"):
            with self.subTest(version=version):
                model = _model([_fmt(radii=(RADIUS,), version=version)], spacing=spacing, mean_density=rho_bulk,
                               thermal_wavelength=1.0, compute_c1=True)
                data = GridData.from_dict(
                    {"grid_size": [nx, ny, nz], "n_types": 1, "grid_spacing": spacing, "temperature": 1.0},
                    cutoff_grid=0, boltzmann_constant=1.0,
                )
                z = data["grid_positions"][:, 2]
                data["excluded_mask"] = (z < 8) | (z >= nz - 8)
                data["V_ext"] = torch.zeros(nx * ny * nz, 1, dtype=torch.float64)
                density = torch.tensor(rho_bulk, dtype=torch.float64, requires_grad=True)
                free_energy = _bulk(version, density)
                mu_excess, = torch.autograd.grad(free_energy, density)
                pressure = rho_bulk + rho_bulk * mu_excess.item() - free_energy.item()
                data["mu"] = torch.tensor([math.log(rho_bulk) + mu_excess.item()], dtype=torch.float64)
                result = GridSolver(model).solve(
                    data, method="euler",
                    convergence=ConvergenceOptions(max_iter=6000, tolerance_residual=1e-6),
                    euler=EulerOptions(mixing=0.05),
                )
                self.assertTrue(result["converged"])
                profile = result["rho"][:, 0].reshape(nx, ny, nz)[0, 0]
                self.assertTrue(torch.all(profile[:8] == 0))
                self.assertAlmostEqual(profile[nz // 2].item(), rho_bulk, places=4)
                # Contact enhancement, then the depletion minimum of a hard-sphere fluid.
                self.assertGreater(profile[8].item(), 2 * rho_bulk)
                self.assertLess(profile[8:24].min().item(), rho_bulk)
                contact = profile[8].item() + 0.5 * (profile[8] - profile[9]).item()  # extrapolated to the wall
                self.assertLess(abs(contact / pressure - 1), 0.05)

    def test_permuted_grid_rows_give_the_canonical_energy(self):
        torch.manual_seed(1)
        model = _model([_fmt()])
        batch = _homogeneous_batch([0.2], grid=8)
        batch["rho"] = batch["rho"] * (1 + 0.5 * torch.rand_like(batch["rho"]))
        canonical = model(batch, compute_c1=True)
        order = torch.randperm(8 ** 3)
        permuted = dict(batch, rho=batch["rho"][:, order], grid_positions=batch["grid_positions"][:, order])
        result = model(permuted, compute_c1=True)
        torch.testing.assert_close(result["beta_F_exc"], canonical["beta_F_exc"])
        torch.testing.assert_close(result["c1"], canonical["c1"][:, order])

    def test_batches_gradients_and_physical_mode(self):
        torch.manual_seed(0)
        model = _model([_fmt()])
        batch = _homogeneous_batch([0.2, 0.25])
        batch["rho"] = batch["rho"] * (1 + 0.3 * torch.rand_like(batch["rho"]))
        batch["rho"].requires_grad_(True)
        energy = model(batch, compute_c1=False)["beta_F_exc"]
        separate = [model({key: value[[field]] for key, value in batch.items()}, compute_c1=False)["beta_F_exc"] for field in range(2)]
        torch.testing.assert_close(energy, torch.cat(separate))
        gradient, = torch.autograd.grad(energy.sum(), batch["rho"])
        self.assertTrue(torch.isfinite(gradient).all())
        # The functional is entropic: physical mode at another temperature gives the same beta F.
        physical = _model([_fmt()], free_energy_mode="physical")
        warm = dict(batch, rho=batch["rho"].detach(), temperature=torch.full((2,), 1.7, dtype=torch.float64))
        torch.testing.assert_close(physical(warm, compute_c1=False)["beta_F_exc"], energy.detach())

    def test_packing_guard(self):
        dense = _homogeneous_batch([1.0])  # eta = 1.05 for the two species together
        clamped = _model([_fmt()])(dense, compute_c1=False)["beta_F_exc"]
        self.assertTrue(torch.isfinite(clamped).all())
        with self.assertRaisesRegex(ValueError, "maximum_packing"):
            _model([_fmt(packing_guard="raise")])(dense, compute_c1=False)
        # The clamp returns n3 itself, in single precision too, wherever n3 is
        # well below maximum_packing: no cancellation in the dilute limit.
        for dtype in (torch.float32, torch.float64):
            dilute = torch.tensor([1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 0.5], dtype=dtype)
            self.assertTrue(torch.equal(_fmt()._guarded_packing(dilute), dilute))
            self.assertLessEqual(_fmt()._guarded_packing(torch.tensor([2.0], dtype=dtype)).item(), 0.95 + 1.0e-6)
        # Below the guard the two behave identically.
        normal = _homogeneous_batch([0.2])
        torch.testing.assert_close(
            _model([_fmt(packing_guard="raise")])(normal, compute_c1=False)["beta_F_exc"],
            _model([_fmt()])(normal, compute_c1=False)["beta_F_exc"],
        )

    def test_composition_with_the_msa_reference(self):
        msa = dict(charges=(1.0, -1.0), diameter=2 * RADIUS, coulomb_amplitude=15.0, long_range_alpha=0.2)
        with self.assertRaisesRegex(ValueError, "include_hard_sphere=False"):
            _model([_fmt(), MSAReferenceReadout(2, **msa)])
        _model([_fmt(), MSAReferenceReadout(2, include_hard_sphere=False, **msa)])
        with self.assertRaisesRegex(ValueError, "one FMTReadout"):
            _model([_fmt(), _fmt()])

    def test_configuration_round_trip(self):
        self.assertIs(registered_types()["FMTReadout"], FMTReadout)
        readout = _fmt(radii=(0.5, 0.4), version="white_bear_i", third_term="tensor", maximum_packing=0.9, packing_guard="raise")
        self.assertEqual(build(readout.to_config()).to_config(), readout.to_config())
        model = _model([readout])
        batch = _homogeneous_batch([0.2])
        batch["rho"] = batch["rho"] * (1 + 0.1 * torch.linspace(0, 1, GRID ** 3)[None, :, None])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            save_model(model, path)
            restored = load_model(path)
        torch.testing.assert_close(
            restored(batch, compute_c1=False)["beta_F_exc"], model(batch, compute_c1=False)["beta_F_exc"],
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 0)

    def test_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "n_types entries"):
            FMTReadout(2, radii=(0.5,))
        with self.assertRaisesRegex(ValueError, "radii"):
            _fmt(radii=(0.5, 0.0))
        with self.assertRaisesRegex(ValueError, "version"):
            _fmt(version="white_bear_ii")
        with self.assertRaisesRegex(ValueError, "third_term"):
            _fmt(third_term="tarazona")
        with self.assertRaisesRegex(ValueError, "maximum_packing"):
            _fmt(maximum_packing=1.0)
        with self.assertRaisesRegex(ValueError, "packing_guard"):
            _fmt(packing_guard="ignore")
        negative = _homogeneous_batch([0.2, 0.2])
        negative["rho"][0, 0, 0] = -1.0e-6
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _model([_fmt()])(negative, compute_c1=False)
        with self.assertRaisesRegex(ValueError, "n_types"):
            _model([_fmt()])({**_homogeneous_batch([0.2]), "rho": torch.full((1, GRID ** 3, 3), 0.2, dtype=torch.float64)}, compute_c1=False)


if __name__ == "__main__":
    unittest.main()
