"""Independent Fourier checks for fixed, explicitly positioned metal sites."""

import io
import itertools
import math
import unittest

import torch
from metal_helpers import _run, liquid_coulomb, metal_sites

from equicdft import GridSolver, MetalWall


DTYPE = torch.float64


def _wall(charges=(1., -1.), sites=None):
    return MetalWall(metal_sites=metal_sites(_data()) if sites is None else sites,
                     liquid_coulomb=liquid_coulomb(charges, 1.7),
                     metal_sigma=.35,
                     ).double()


def _data(shape=(3, 3, 3), spacing=(.7, .9, 1.1)):
    spacing = torch.tensor(spacing, dtype=DTYPE)
    cell = torch.tensor(shape, dtype=DTYPE) * spacing
    rho = torch.full((math.prod(shape), 2), .4, dtype=DTYPE)
    rho[:, 0] += torch.linspace(-.08, .08, len(rho), dtype=DTYPE)
    positions = torch.tensor([[.08, .13, .17], [.27, .67, .74],
                              [.62, .31, .43], [.88, .82, .91]], dtype=DTYPE) * cell
    return dict(rho=rho, grid_size=torch.tensor(shape), grid_spacing=spacing,
                metal_positions=positions, metal_site_groups=torch.tensor([2, 2, 7, 7]),
                metal_group_ids=torch.tensor([7, 2]),
                metal_total_charge=torch.tensor([-.15, .15], dtype=DTYPE),
                metal_charge_units="e")


def _one_electrode(data):
    data = dict(data)
    data["metal_site_groups"] = torch.zeros_like(data["metal_site_groups"])
    data["metal_group_ids"] = torch.tensor([0])
    data["metal_total_charge"] = data["metal_total_charge"].sum().reshape(1)
    return data


def _field_coordinates(sites, cell):
    """Independent compact periodic branch used by the test oracle."""
    wrapped = torch.remainder(sites, cell)
    coordinates = torch.empty_like(wrapped)
    for axis in range(3):
        values = torch.sort(wrapped[:, axis]).values
        gaps = torch.cat((values[1:] - values[:-1],
                          values[:1] + cell[axis] - values[-1:]))
        cut = values[torch.argmax(gaps)] + 0.5 * gaps.max()
        center = torch.remainder(cut + 0.5 * cell[axis], cell[axis])
        displacement = wrapped[:, axis] - center
        coordinates[:, axis] = displacement - cell[axis] * torch.round(
            displacement / cell[axis]
        )
    return coordinates


def _reference(data, charges=(1., -1.), sigma=.35, coulomb_amplitude=1.7):
    """Pairwise cosine sums, independent of production FFT/site machinery.

    Even axes retain both Nyquist signs with half weights on each endpoint;
    products of those weights also cover edges and corners of the cutoff.
    All matrices act on integrated charges, not density values.
    """
    shape = tuple(data["grid_size"].tolist())
    spacing = data["grid_spacing"]
    cell = torch.tensor(shape, dtype=DTYPE) * spacing
    grid = torch.tensor(list(itertools.product(*(range(n) for n in shape))),
                        dtype=DTYPE) * spacing
    grid = data.get("grid_center", grid)
    sites = torch.as_tensor(data["metal_positions"], dtype=DTYPE)
    axes = []
    for n in shape:
        modes = range(-(n // 2), n // 2 + 1)
        axes.append([(m, .5 if n % 2 == 0 and abs(m) == n // 2 else 1.)
                     for m in modes])
    entries = list(itertools.product(*axes))
    integers = torch.tensor([[a[0] for a in row] for row in entries], dtype=DTYPE)
    weights = torch.tensor([math.prod(a[1] for a in row) for row in entries], dtype=DTYPE)
    wavevectors = 2 * math.pi * integers / cell
    k2 = wavevectors.square().sum(-1)
    nonzero = k2 > 0
    wavevectors, k2 = wavevectors[nonzero], k2[nonzero]
    coefficients = coulomb_amplitude * 4 * math.pi * weights[nonzero] / (cell.prod() * k2)

    def kernel(target, source, exponent):
        phases = (target[:, None, :] - source[None, :, :]) @ wavevectors.T
        return (torch.cos(phases) * (coefficients * torch.exp(-exponent * k2))).sum(-1)

    bmatrix = kernel(sites, grid, sigma**2 / 2)
    a = kernel(sites, sites, sigma**2)
    valencies = torch.tensor(charges, dtype=DTYPE)
    liquid = data["rho"] @ valencies * spacing.prod()
    constraints = (data["metal_site_groups"][:, None]
                   == data["metal_group_ids"][None, :]).to(DTYPE)
    field = torch.as_tensor(data.get("metal_external_field", [0., 0., 0.]), dtype=DTYPE)
    external = -(_field_coordinates(sites, cell) * field).sum(-1)
    b = bmatrix @ liquid
    m, groups = constraints.shape
    kkt = torch.cat((torch.cat((a, constraints), dim=1),
                     torch.cat((constraints.T, torch.zeros((groups, groups), dtype=DTYPE)), dim=1)))
    solved = torch.linalg.solve(kkt, torch.cat((-b - external, data["metal_total_charge"])))
    q = solved[:m]
    response = torch.linalg.solve(kkt, torch.eye(m + groups, dtype=DTYPE))[:m, :m]
    elm, emm = q @ b, .5 * q @ a @ q
    return dict(metal_site_q=q, q_liquid=liquid, liquid_charge_density=data["rho"] @ valencies,
                metal_charge=constraints.T @ q, metal_potential=-solved[m:],
                charge_residual=(constraints.T @ q - data["metal_total_charge"]).abs().max(),
                potential_residual=(a @ q + b + external + constraints @ solved[m:]).abs().max(),
                coulomb_cross_energy=elm,
                coulomb_metal_energy=emm, electrode_coulomb_energy=elm + emm,
                metal_external_energy=q @ external, B=bmatrix, S=response)


def _correction(result):
    return sum(result[key] for key in ("coulomb_cross_energy", "coulomb_metal_energy",
                                       "metal_external_energy"))


def _model_fixture(mode="beta"):
    # Import locally to reuse the established weak-coupling fixture without
    # collecting its TestCase a second time through this module.
    from test_metal_model import TestMetalModel

    model = TestMetalModel._model(mode=mode).eval()
    data = TestMetalModel._data()
    data["metal_positions"] += torch.tensor([.075, .11, .09], dtype=DTYPE)
    # Off-grid sites preserve the independently specified inaccessible volume.
    data["excluded_mask"][7] = True
    data["rho"][data["excluded_mask"]] = 0
    data["rho"][:, 1] *= data["rho"][:, 0].sum() / data["rho"][:, 1].sum()
    data = _one_electrode(data)
    data["metal_external_field"] = torch.tensor([.03, 0., 0.], dtype=DTYPE)
    return model, data


def _model_reference(data):
    return _reference(data, sigma=.6, coulomb_amplitude=.2)


def _inverse_fixture():
    model, data = _model_fixture()
    c1 = _run(model, data)["c1"].detach()
    accessible = ~data["excluded_mask"]
    data["V_ext"] = torch.zeros_like(data["rho"])
    data["V_ext"][accessible] = (c1[accessible] - torch.log(data["rho"][accessible])) / data["beta"]
    return model, data


class TestExplicitMetalSites(unittest.TestCase):
    def assert_outputs_close(self, actual, expected):
        for key in expected:
            if key in ("B", "S", "q_liquid", "liquid_charge_density"):
                continue
            with self.subTest(output=key):
                torch.testing.assert_close(actual[key], expected[key], atol=3e-11, rtol=3e-11)

    def test_on_grid_sites_match_independent_reference(self):
        data = _data(shape=(3, 2, 4), spacing=(.8, 1.1, .65))
        selected = torch.tensor([0, 3, 16, 23])
        grid = torch.tensor(list(itertools.product(range(3), range(2), range(4))), dtype=DTYPE)
        data["metal_positions"] = grid[selected] * data["grid_spacing"]
        self.assert_outputs_close(_run(_wall(), data), _reference(data))

    def test_arbitrary_positions_match_independent_odd_even_anisotropic_fourier_sums(self):
        for shape, spacing in (((3, 3, 3), (.7, .9, 1.1)),
                               ((4, 2, 4), (.8, 1.3, .6)),
                               ((3, 4, 2), (.91, .73, 1.27))):
            with self.subTest(shape=shape):
                data = _data(shape, spacing)
                data = _one_electrode(data)
                data["metal_external_field"] = torch.tensor([.12, -.05, .08], dtype=DTYPE)
                self.assert_outputs_close(_run(_wall(), data), _reference(data))
                # Python coordinates must not first round through default float32.
                listed = dict(data, metal_positions=data["metal_positions"].tolist())
                self.assert_outputs_close(_run(_wall(), listed), _reference(data))

    def test_general_species_and_nonzero_liquid_charge_use_integrated_units(self):
        data = _data()
        data["rho"] = torch.cat((data["rho"][:, :1] / 2,
                                 data["rho"][:, 1:], torch.full((27, 1), .23, dtype=DTYPE)), dim=1)
        data["rho"][:, 0] += .025
        liquid_total = data["grid_spacing"].prod() * (2 * data["rho"][:, 0] - data["rho"][:, 1]).sum()
        data["metal_total_charge"][0] -= liquid_total
        actual = _run(_wall(charges=(2., -1., 0.)), data)
        self.assert_outputs_close(actual, _reference(data, charges=(2., -1., 0.)))
        self.assertLess(abs(float(liquid_total + actual["metal_site_q"].sum())), 1e-12)
        self.assertNotIn("q_liquid", actual)
        self.assertNotIn("liquid_charge_density", actual)

    def test_density_gradient_matches_independent_envelope_and_voxel_factor(self):
        data = _one_electrode(_data())
        data["metal_external_field"] = torch.tensor([.12, -.05, .08], dtype=DTYPE)
        reference = _reference(data)
        data["rho"].requires_grad_()
        actual = _run(_wall(), data)
        gradient = torch.autograd.grad(_correction(actual), data["rho"], retain_graph=True)[0]
        expected = ((reference["B"].T @ reference["metal_site_q"])[:, None]
                    * data["grid_spacing"].prod() * torch.tensor([1., -1.], dtype=DTYPE))
        torch.testing.assert_close(gradient, expected, atol=3e-11, rtol=3e-11)

    def test_second_density_derivative_includes_constrained_electrode_response(self):
        data = _one_electrode(_data())
        data["metal_external_field"] = torch.tensor([.12, -.05, .08], dtype=DTYPE)
        reference = _reference(data)
        directions = torch.zeros((27, 2, 2), dtype=DTYPE)
        directions[1, 0, 0], directions[8, 0, 0] = 1., -1.
        directions[5, 1, 1], directions[19, 1, 1] = 1., -1.
        wall = _wall()

        def energy(coefficients):
            rho = data["rho"] + torch.einsum("gsi,i->gs", directions, coefficients)
            return _correction(_run(wall, dict(data, rho=rho)))

        coordinates = torch.zeros(2, dtype=DTYPE, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(energy, (coordinates,), eps=1e-6, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.autograd.gradgradcheck(energy, (coordinates,), eps=1e-6, atol=1e-6, rtol=1e-5))
        actual = torch.autograd.functional.hessian(energy, coordinates)
        source_directions = data["grid_spacing"].prod() * (directions[:, 0] - directions[:, 1])
        driven = reference["B"] @ source_directions
        expected = -driven.T @ reference["S"] @ driven
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        self.assertGreater(float(actual.abs().max()), 1e-4)

    def test_empty_neutral_pair_has_zero_limit_and_correct_constant_field_sign(self):
        data = _data(shape=(4, 2, 3), spacing=(.8, .9, .7))
        data.update(rho=torch.zeros_like(data["rho"]),
                    metal_positions=torch.tensor([[.23, .1, .2], [2.97, .1, .2]], dtype=DTYPE),
                    metal_site_groups=torch.tensor([5, 5]), metal_group_ids=torch.tensor([5]),
                    metal_total_charge=torch.zeros(1, dtype=DTYPE))
        wall = _wall()
        zero = _run(wall, data)
        for key in ("metal_site_q", "metal_potential", "electrode_coulomb_energy", "metal_external_energy"):
            torch.testing.assert_close(zero[key], torch.zeros_like(zero[key]), atol=0, rtol=0)
        field = torch.tensor([.17, 0., 0.], dtype=DTYPE)
        positive = _run(wall, dict(data, metal_external_field=field))
        negative = _run(wall, dict(data, metal_external_field=-field))
        self.assertGreater(float(positive["metal_site_q"][0]), 0)
        self.assertLess(float(positive["metal_site_q"][1]), 0)
        self.assertLess(float(positive["metal_external_energy"]), 0)
        torch.testing.assert_close(negative["metal_site_q"], -positive["metal_site_q"])
        torch.testing.assert_close(_correction(negative), _correction(positive))

    def test_automatic_branch_and_individual_periodic_site_translations(self):
        data = _one_electrode(_data())
        field = torch.tensor([.12, -.05, .08], dtype=DTYPE)
        data["metal_external_field"] = field
        expected = _run(_wall(), data)
        cell = data["grid_size"] * data["grid_spacing"]
        translated = data["metal_positions"] + cell * torch.tensor([[1, 0, -1], [0, 2, 0], [-2, 0, 1], [1, 1, 1]])
        self.assert_outputs_close(
            _run(_wall(), dict(data, metal_positions=translated)), expected,
        )

    def test_joint_lattice_translation_of_source_and_sites(self):
        data = _one_electrode(_data())
        data["metal_external_field"] = torch.tensor([.12, -.05, .08], dtype=DTYPE)
        shift = torch.tensor([float(data["grid_spacing"][0]), 0., 0.], dtype=DTYPE)
        translated_rho = torch.roll(data["rho"].reshape(3, 3, 3, 2), 1, 0).reshape(-1, 2)
        moved = dict(data, rho=translated_rho, metal_positions=data["metal_positions"] + shift)
        old, new = _run(_wall(), data), _run(_wall(), moved)
        for key in old:
            torch.testing.assert_close(
                new[key], old[key], atol=3e-11, rtol=3e-11,
            )

    def test_shared_site_geometry_broadcasts_across_multiple_batch_axes(self):
        data = _one_electrode(_data())
        factors = torch.tensor([[1., .8], [1.1, .9]], dtype=DTYPE)
        fields = torch.tensor([[[.1, 0., 0.], [.2, 0., 0.]],
                               [[-.1, .05, 0.], [0., 0., .07]]], dtype=DTYPE)
        batch = dict(data, rho=factors[..., None, None] * data["rho"], metal_external_field=fields)
        actual = _run(_wall(), batch)
        self.assertEqual(actual["metal_site_q"].shape, (2, 2, 4))
        for i, j in itertools.product(range(2), repeat=2):
            single = _run(_wall(), dict(data, rho=batch["rho"][i, j], metal_external_field=fields[i, j]))
            for key in single:
                torch.testing.assert_close(actual[key][i, j], single[key], atol=3e-11, rtol=3e-11)

    def test_cache_updates_for_position_and_group_changes_not_density_or_field(self):
        data = _one_electrode(_data())
        wall = _wall()
        _run(wall, data)
        cached = wall._cache
        _run(wall, dict(data, rho=data["rho"] * 1.1, metal_external_field=torch.tensor([.1, 0., 0.], dtype=DTYPE)))
        self.assertIs(wall._cache, cached)
        moved = dict(data, metal_positions=data["metal_positions"].clone())
        moved["metal_positions"][0, 0] += .051
        self.assert_outputs_close(_run(wall, moved), _reference(moved))
        self.assertIsNot(wall._cache, cached)
        regrouped = dict(
            moved,
            metal_site_groups=torch.tensor([2, 7, 2, 7]),
            metal_group_ids=torch.tensor([2, 7]),
            metal_total_charge=torch.zeros(2, dtype=DTYPE),
        )
        self.assert_outputs_close(_run(wall, regrouped), _reference(regrouped))

    def test_cache_serialization_dtype_migration_and_inference_then_derivatives(self):
        data = _data()
        wall = _wall()
        with torch.inference_mode():
            _run(wall, data)
        rho = data["rho"].clone().requires_grad_()
        gradient = torch.autograd.grad(_correction(_run(wall, dict(data, rho=rho))), rho, create_graph=True)[0]
        second = torch.autograd.grad(gradient[1, 0], rho)[0]
        self.assertTrue(torch.isfinite(second).all())
        expected = _run(wall, data)
        buffer = io.BytesIO()
        torch.save(wall, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        self.assertIsNone(restored._cache)
        self.assertIsNone(restored._cache_key)
        self.assert_outputs_close(_run(restored, data), expected)
        restored.float()
        self.assertIsNone(restored._cache)
        single = {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
                  for key, value in data.items()}
        actual = _run(restored, single)
        self.assertEqual(actual["metal_site_q"].dtype, torch.float32)
        torch.testing.assert_close(actual["metal_site_q"].double(), expected["metal_site_q"], atol=3e-6, rtol=3e-5)

    def test_invalid_positions_labels_and_periodic_duplicates_are_rejected(self):
        data = _data()
        coordinates = data["metal_positions"]
        bad_positions = [coordinates[:, :2], coordinates[:3], coordinates.unsqueeze(0),
                         coordinates.to(torch.complex128), coordinates > 0,
                         coordinates.clone().requires_grad_()]
        nonfinite = coordinates.clone()
        nonfinite[1, 0] = float("nan")
        bad_positions.append(nonfinite)
        for shift in (torch.zeros(3), data["grid_size"] * data["grid_spacing"]):
            duplicate = coordinates.clone()
            duplicate[1] = duplicate[0] + shift
            bad_positions.append(duplicate)
        for value in bad_positions:
            with self.subTest(positions=value):
                with self.assertRaises((TypeError, ValueError)):
                    _run(_wall(), dict(data, metal_positions=value))
        for labels in (torch.tensor([2, 2, 7]), torch.tensor([2, -1, 7, 7]),
                       torch.tensor([2., 2.5, 7., 7.]), torch.tensor([True] * 4),
                       torch.tensor([2, 2, 7, 9])):
            with self.subTest(labels=labels):
                with self.assertRaises((TypeError, ValueError)):
                    _run(_wall(), dict(data, metal_site_groups=labels))

    def test_exclusion_is_independent_and_forward_does_not_mutate_inputs(self):
        data = _data()
        data["rho"][0] = 0
        data["rho"][:, 1] = data["rho"][:, 0]
        data["excluded_mask"] = torch.zeros(27, dtype=torch.bool)
        data["excluded_mask"][0] = True
        before = {key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}
        actual = _run(_wall(), data)
        self.assert_outputs_close(actual, _reference(data))
        self.assertEqual(data.keys(), before.keys())
        for key in data:
            if torch.is_tensor(data[key]):
                self.assertTrue(torch.equal(data[key], before[key]), key)
            else:
                self.assertEqual(data[key], before[key])

    def test_model_explicit_state_c1_and_c2_match_independent_response(self):
        for mode in ("beta", "physical"):
            with self.subTest(mode=mode):
                model, data = _model_fixture(mode)
                reference = _model_reference(data)
                result = _run(model, data, compute_c2=True, c2_reference=(2, 0))
                self.assert_outputs_close(result, reference)
                for key in (
                    "metal_positions", "metal_site_groups",
                    "metal_group_ids", "metal_total_charge",
                ):
                    self.assertNotIn(key, result)
                wall_energy = _run(model.readout[1].energy, data)
                expected_energy = (
                    wall_energy
                    if mode == "beta" else
                    wall_energy / (data["temperature"] / model.mean_temperature)
                )
                torch.testing.assert_close(
                    result["beta_F_exc"], expected_energy,
                    atol=3e-11, rtol=3e-11,
                )
                direction = torch.zeros_like(data["rho"])
                direction[2, 0], direction[4, 0] = 1., -1.
                step = 1e-5
                plus = _run(model, dict(data, rho=data["rho"].detach() + step * direction))
                minus = _run(model, dict(data, rho=data["rho"].detach() - step * direction))
                torch.testing.assert_close((plus["beta_F_exc"] - minus["beta_F_exc"]) / (2 * step),
                                           -model.voxel_volume * (result["c1"] * direction).sum(),
                                           atol=1e-10, rtol=1e-8)
                torch.testing.assert_close((plus["c1"][2, 0] - minus["c1"][2, 0]) / (2 * step),
                                           model.voxel_volume * (result["c2"] * direction).sum(),
                                           atol=1e-10, rtol=1e-8)

    def test_solver_evaluate_preserves_explicit_metadata_exclusions_and_fresh_state(self):
        model, data = _inverse_fixture()
        model.compute_local_mu = True
        data["mu"] = torch.zeros(2, dtype=DTYPE)
        solver = GridSolver(model)
        result = _run(solver.evaluate, data)
        self.assert_outputs_close(result, _model_reference(data))
        torch.testing.assert_close(result["excluded_mask"], data["excluded_mask"])
        for key in (
            "metal_positions", "metal_site_groups",
            "metal_group_ids", "metal_total_charge",
        ):
            self.assertNotIn(key, result)
        torch.testing.assert_close(result["euler_lagrange_residual"], torch.zeros_like(data["rho"]),
                                   atol=3e-11, rtol=0)
        expected_external = data["beta"] * model.voxel_volume * (data["rho"] * data["V_ext"]).sum()
        torch.testing.assert_close(result["beta_V_ext"], expected_external)
        shifted = dict(data, rho=data["rho"].detach().clone())
        shifted["rho"][2, 0] += .005
        shifted["rho"][4, 0] -= .005
        fresh = _run(solver.evaluate, shifted)
        self.assert_outputs_close(fresh, _model_reference(shifted))
        self.assertGreater(float((fresh["metal_site_q"] - result["metal_site_q"]).abs().max()), 1e-8)
        invalid = dict(data, rho=data["rho"].detach().clone())
        invalid["rho"][7] = .01
        with self.assertRaisesRegex(ValueError, "excluded"):
            _run(solver.evaluate, invalid)

    def test_explicit_inverse_two_starts_and_truncated_final_state(self):
        model, data = _inverse_fixture()
        target = data.pop("rho")
        excluded = data["excluded_mask"]
        numbers = model.voxel_volume * target.sum(0)
        solutions = []
        for initialization in (None, torch.flip(target, dims=(0,)) + .05):
            result = _run(GridSolver(model).solve,
                data, initial_rho=initialization, particle_numbers=numbers,
                method="euler", max_iter=250, tolerance_residual=1e-9,
                tolerance_change=1e-12, mixing=.2,
            )
            self.assertTrue(result["converged"])
            self.assertTrue(torch.equal(result["excluded_mask"], excluded))
            self.assertTrue(torch.equal(result["rho"][excluded], torch.zeros_like(target[excluded])))
            self.assertTrue(torch.all(result["rho"][~excluded] > 0))
            torch.testing.assert_close(model.voxel_volume * result["rho"].sum(0), numbers, atol=1e-12, rtol=0)
            torch.testing.assert_close(result["rho"], target, atol=1e-9, rtol=0)
            self.assertLess(result["max_euler_lagrange_residual"], 1e-9)
            self.assert_outputs_close(result, _model_reference(dict(data, rho=result["rho"])))
            solutions.append(result["rho"])
        torch.testing.assert_close(solutions[0], solutions[1], atol=1e-9, rtol=0)
        # A deliberately unfinished one-step unit solve leaves a large enough
        # density update that returning the initial electrode state would fail.
        baseline = torch.zeros_like(target)
        baseline[~excluded] = numbers / (model.voxel_volume * (~excluded).sum())
        initial_q = _model_reference(dict(data, rho=baseline))["metal_site_q"]
        truncated = _run(GridSolver(model).solve,
            data, initial_rho=baseline, particle_numbers=numbers, method="euler",
            max_iter=1, tolerance_residual=1e-9, mixing=.2,
        )
        self.assertFalse(truncated["converged"])
        self.assertGreater(float((truncated["metal_site_q"] - initial_q).abs().max()), 1e-8)
        self.assert_outputs_close(truncated, _model_reference(dict(data, rho=truncated["rho"])))


if __name__ == "__main__":
    unittest.main()
