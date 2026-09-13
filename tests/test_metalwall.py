"""Independent discrete-electrostatics checks for polarizable metal cells."""

import itertools
import io
import math
import unittest
from unittest.mock import patch

import torch
from metal_helpers import _run, liquid_coulomb, metal_sites

from equicdft import MetalWall


DTYPE = torch.float64


def _field(shape=(3, 2, 2), spacing=(0.8, 1.1, 1.3), group_ids=(2, 7)):
    positions = torch.tensor(list(itertools.product(*(range(n) for n in shape))))
    mask = torch.full((math.prod(shape),), -1, dtype=torch.long)
    mask[positions[:, 0] == 0] = group_ids[0]
    mask[positions[:, 0] == shape[0] - 1] = group_ids[1]
    accessible = mask < 0
    rho = torch.zeros((len(mask), 2), dtype=DTYPE)
    rho[accessible] = 0.4
    variation = torch.linspace(-0.08, 0.08, int(accessible.sum()), dtype=DTYPE)
    rho[accessible, 0] += variation
    return {
        "rho": rho,
        "grid_size": torch.tensor(shape),
        "grid_spacing": torch.tensor(spacing, dtype=DTYPE),
        "excluded_mask": mask >= 0,
        "metal_positions": positions[mask >= 0].to(DTYPE) * torch.tensor(spacing, dtype=DTYPE),
        "metal_site_groups": mask[mask >= 0],
        "metal_group_ids": torch.tensor(group_ids),
        "metal_total_charge": torch.tensor((0.15, -0.15), dtype=DTYPE),
        "metal_charge_units": "e",
    }


def _site_rows(data):
    """Canonical indices for this test's explicitly positioned on-grid sites."""
    coordinates = (data["metal_positions"] / data["grid_spacing"]).round().long()
    shape = data["grid_size"]
    return (coordinates[:, 0] * shape[1] + coordinates[:, 1]) * shape[2] + coordinates[:, 2]


def _liquid_charge(data, charges=(1.0, -1.0)):
    return (
        data["grid_spacing"].prod()
        * (data["rho"] @ torch.tensor(charges, dtype=data["rho"].dtype))
    )


def _module(sigma=0.35, amplitude=1.7, charges=(1.0, -1.0)):
    return MetalWall(
        metal_sites=metal_sites(_field()),
        liquid_coulomb=liquid_coulomb(charges, amplitude),
        metal_sigma=sigma,
    ).double()


def _dense_kernels(data, sigma, amplitude):
    """Direct Fourier sum over pairs; does not use FFT convolution."""
    shape = tuple(data["grid_size"].tolist())
    spacing = data["grid_spacing"]
    positions = torch.tensor(
        list(itertools.product(*(range(n) for n in shape))), dtype=DTYPE,
    ) * spacing
    frequencies = [
        torch.fft.fftfreq(n, d=float(dx), dtype=DTYPE) * (2 * math.pi)
        for n, dx in zip(shape, spacing)
    ]
    wavevectors = torch.tensor(list(itertools.product(*frequencies)), dtype=DTYPE)
    squared = wavevectors.square().sum(-1)
    nonzero = squared > 0
    squared = squared[nonzero]
    phase = positions @ wavevectors[nonzero].T
    pair_cosine = torch.cos(phase[:, None, :] - phase[None, :, :])
    volume = math.prod(shape) * spacing.prod()
    coulomb = amplitude * 4 * math.pi / squared / volume

    def kernel(width_squared):
        return (pair_cosine * (coulomb * torch.exp(-0.5 * width_squared * squared))).sum(-1)

    return (
        kernel(sigma**2),
        kernel(2 * sigma**2),
    )


def _dense_solution(data, sigma=0.35, amplitude=1.7, charges=(1.0, -1.0)):
    klm, kmm = _dense_kernels(data, sigma, amplitude)
    liquid = data["rho"] @ torch.tensor(charges, dtype=DTYPE) * data["grid_spacing"].prod()
    selected = _site_rows(data)
    groups = data["metal_group_ids"]
    constraints = (data["metal_site_groups"][None, :] == groups[:, None]).to(DTYPE)
    a = kmm[selected][:, selected]
    b = klm[selected] @ liquid
    nmetal = selected.numel()
    ngroup = len(groups)
    kkt = torch.cat((
        torch.cat((a, constraints.T), dim=1),
        torch.cat((constraints, torch.zeros((ngroup, ngroup), dtype=DTYPE)), dim=1),
    ), dim=0)
    solved = torch.linalg.solve(kkt, torch.cat((-b, data["metal_total_charge"])))
    q = solved[:nmetal]
    metal = torch.zeros_like(liquid)
    metal[selected] = q
    potentials = a @ q + b
    elm = liquid @ klm @ metal
    emm = 0.5 * metal @ kmm @ metal
    return {
        "q_liquid": liquid,
        "metal_site_q": q,
        "metal_potential": torch.stack([potentials[data["metal_site_groups"] == group].mean() for group in groups]),
        "coulomb_cross_energy": elm,
        "coulomb_metal_energy": emm,
        "electrode_coulomb_energy": elm + emm,
        "effective_kernel": -klm[:, selected] @ torch.linalg.inv(kkt)[:nmetal, :nmetal] @ klm[selected],
    }


class TestMetalWall(unittest.TestCase):
    @staticmethod
    def _biased_field():
        data = _field(shape=(4, 2, 2), group_ids=(0, 0))
        data["metal_group_ids"] = torch.tensor([0])
        data["metal_total_charge"] = torch.zeros(1, dtype=DTYPE)
        data["metal_external_field"] = torch.tensor([0.17, 0.0, 0.0], dtype=DTYPE)
        return data

    def test_constant_field_matches_independent_dense_kkt_and_energy(self):
        data = self._biased_field()
        actual = _run(_module(), data)
        klm, kmm = _dense_kernels(data, 0.35, 1.7)
        selected = _site_rows(data)
        # Centers are +/- dx/2 across the boundary, independent of FFT code.
        x = torch.tensor([0.4] * 4 + [-0.4] * 4, dtype=DTYPE)
        v = -0.17 * x
        a = kmm[selected][:, selected]
        b = klm[selected] @ _liquid_charge(data)
        ones = torch.ones((8, 1), dtype=DTYPE)
        kkt = torch.cat((torch.cat((a, ones), dim=1),
                         torch.cat((ones.T, torch.zeros((1, 1), dtype=DTYPE)), dim=1)))
        expected = torch.linalg.solve(kkt, torch.cat((-b - v, torch.zeros(1, dtype=DTYPE))))
        torch.testing.assert_close(actual["metal_site_q"], expected[:8], atol=2e-8, rtol=2e-8)
        torch.testing.assert_close(actual["metal_potential"], -expected[8:], atol=2e-8, rtol=2e-8)
        torch.testing.assert_close(actual["metal_external_energy"], expected[:8] @ v)
        self.assertLess(actual["metal_site_q"].sum().abs().item(), 1e-12)
        self.assertLess(actual["potential_residual"].item(), 1e-12)

    def test_constant_field_sign_zero_limit_and_geometry_cache(self):
        data = self._biased_field()
        data["rho"].zero_()
        wall = _module()
        positive = _run(wall, data)
        cache = wall._cache
        negative = _run(wall, dict(data, metal_external_field=-data["metal_external_field"]))
        self.assertIs(cache, wall._cache)
        torch.testing.assert_close(negative["metal_site_q"], -positive["metal_site_q"])
        self.assertGreater(positive["metal_site_q"][:4].sum().item(), 0)
        self.assertLess(positive["metal_site_q"][-4:].sum().item(), 0)
        self.assertLess(positive["metal_external_energy"].item(), 0)
        plain = {key: value for key, value in data.items()
                 if key != "metal_external_field"}
        zero = _run(wall, dict(data, metal_external_field=torch.zeros(3, dtype=DTYPE)))
        for key, value in _run(wall, plain).items():
            torch.testing.assert_close(zero[key], value, atol=0, rtol=0)

    def test_automatic_branch_is_invariant_to_periodic_site_shifts(self):
        data = self._biased_field()
        wall = _module()
        expected = _run(wall, data)
        cell = data["grid_spacing"] * data["grid_size"]
        translated = data["metal_positions"] + cell * torch.tensor(
            [[1, 0, -1], [0, 2, 0], [-2, 0, 1], [1, 1, 1],
             [0, -1, 0], [2, 0, -2], [-1, 1, 0], [0, 0, 2]],
        )
        periodic = _run(wall, dict(data, metal_positions=translated))
        for key in expected:
            if key == "metal_positions":
                torch.testing.assert_close(periodic[key], translated)
            else:
                torch.testing.assert_close(periodic[key], expected[key], atol=1e-12, rtol=1e-12)

    def test_field_energy_envelope_derivative_and_voxel_volume(self):
        data = self._biased_field()
        data["rho"].requires_grad_()
        result = _run(_module(), data)
        energy = result["coulomb_cross_energy"] + result["coulomb_metal_energy"] + result["metal_external_energy"]
        gradient = torch.autograd.grad(energy, data["rho"])[0]
        klm, _ = _dense_kernels(data, 0.35, 1.7)
        # Stationarity cancels dq/drho terms only if metal field work is
        # included. This is stronger than differentiating the same wrong energy.
        expected = ((klm[:, _site_rows(data)] @ result["metal_site_q"])[:, None]
                    * data["grid_spacing"].prod() * torch.tensor([1., -1.], dtype=DTYPE))
        torch.testing.assert_close(gradient, expected, atol=2e-8, rtol=2e-8)

    def test_field_vectors_batch_and_validation(self):
        first = self._biased_field()
        second = dict(first, metal_external_field=-first["metal_external_field"])
        batch = dict(first, rho=torch.stack([first["rho"], second["rho"]]),
                     metal_external_field=torch.stack([first["metal_external_field"], second["metal_external_field"]]))
        wall = _module()
        outputs = _run(wall, batch)
        for i, field in enumerate((first, second)):
            for key, value in _run(wall, field).items():
                torch.testing.assert_close(outputs[key][i], value)
        for value in (0.1, [1, 2], [1, 2, float("nan")], [True]*3,
                      [1j]*3, torch.zeros((2, 3))):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "external_field"):
                    _run(wall, dict(first, metal_external_field=value))
        del first["metal_external_field"]
        self.assertEqual(_run(wall, first)["metal_external_energy"].item(), 0.)

    def test_constant_field_float32(self):
        data = self._biased_field()
        expected = _run(_module(), data)
        single = {key: value.float() if torch.is_tensor(value) and value.is_floating_point()
                  else value for key, value in data.items()}
        result = _run(_module().float(), single)
        for key in ("metal_site_q", "metal_potential", "metal_external_energy"):
            torch.testing.assert_close(result[key].double(), expected[key], atol=2e-6, rtol=2e-6)

    def test_physical_parameters_and_boundary_must_be_valid(self):
        arguments = {
            "metal_sites": metal_sites(_field()),
            "liquid_coulomb": liquid_coulomb(),
            "metal_sigma": 0.35,
        }
        for key, value in (
            ("metal_sigma", 0.0),
            ("metal_sigma", -0.2),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises((TypeError, ValueError)):
                    MetalWall(**dict(arguments, **{key: value}))
        with self.assertRaises(TypeError):
            MetalWall(**dict(arguments, liquid_coulomb=object()))

    def test_dense_fourier_solution_and_energy_decomposition(self):
        data = _field()
        expected = _dense_solution(data)
        actual = _run(_module(), data)
        for key in (
            "metal_site_q", "metal_potential", "electrode_coulomb_energy",
            "coulomb_cross_energy", "coulomb_metal_energy",
        ):
            with self.subTest(key=key):
                torch.testing.assert_close(actual[key], expected[key], atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(
            actual["electrode_coulomb_energy"],
            actual["coulomb_cross_energy"] + actual["coulomb_metal_energy"],
        )

    def test_group_totals_and_equipotential_allow_nonuniform_metal_charges(self):
        data = _field()
        result = _run(_module(), data)
        selected = _site_rows(data)
        klm, kmm = _dense_kernels(data, 0.35, 1.7)
        potential = klm[selected] @ _liquid_charge(data) + kmm[selected][:, selected] @ result["metal_site_q"]
        for index, group in enumerate(data["metal_group_ids"]):
            cells = data["metal_site_groups"] == group
            torch.testing.assert_close(result["metal_site_q"][cells].sum(), data["metal_total_charge"][index])
            torch.testing.assert_close(potential[cells], result["metal_potential"][index].expand(int(cells.sum())), atol=2e-7, rtol=2e-7)
        self.assertGreater(result["metal_site_q"][data["metal_site_groups"] == 2].std().item(), 1e-5)
        self.assertEqual(result["metal_site_q"].numel(), selected.numel())
        torch.testing.assert_close(result["metal_charge"], data["metal_total_charge"])
        self.assertLess(result["charge_residual"].abs().max().item(), 1e-10)
        self.assertLess(result["potential_residual"].abs().max().item(), 2e-7)

    def test_liquid_charge_is_internal_and_still_drives_the_electrode(self):
        data = _field()
        data["rho"] = torch.cat((data["rho"][:, :1] / 2, data["rho"][:, 1:], 0.7 * (~data["excluded_mask"])[:, None]), dim=-1)
        result = _run(_module(charges=(2.0, -1.0, 0.0)), data)
        reference = _dense_solution(data, charges=(2.0, -1.0, 0.0))
        torch.testing.assert_close(result["metal_site_q"], reference["metal_site_q"])
        self.assertNotIn("q_liquid", result)
        self.assertNotIn("liquid_charge_density", result)

    def test_length_unit_conversion_preserves_integrated_charges_and_energy(self):
        data = _field(spacing=(0.5, 0.75, 1.25))
        # Include nonzero net liquid charge, exactly compensated by the metal.
        data["rho"][~data["excluded_mask"], 0] += 0.1
        volume = data["grid_spacing"].prod()
        data["metal_total_charge"][0] -= volume * (data["rho"][:, 0] - data["rho"][:, 1]).sum()
        actual = _run(_module(), data)
        reference = _dense_solution(data)
        for key in ("metal_site_q",
                    "coulomb_cross_energy", "coulomb_metal_energy", "electrode_coulomb_energy"):
            torch.testing.assert_close(actual[key], reference[key], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(
            _liquid_charge(data).sum() + actual["metal_site_q"].sum(),
            torch.zeros((), dtype=DTYPE), atol=1e-12, rtol=0,
        )
        # Change the coordinate unit, keeping the energy and charge units fixed.
        # Lengths and Coulomb amplitude scale by s; densities by s**(-3).
        scale = 2.0
        converted = dict(data, grid_spacing=data["grid_spacing"] * scale,
                         rho=data["rho"] / scale**3,
                         metal_positions=data["metal_positions"] * scale)
        result = _run(_module(sigma=0.35 * scale,
                         amplitude=1.7 * scale), converted)
        torch.testing.assert_close(_liquid_charge(converted), _liquid_charge(data))
        for key in ("metal_site_q", "metal_charge", "metal_potential",
                    "coulomb_cross_energy",
                    "coulomb_metal_energy", "electrode_coulomb_energy"):
            torch.testing.assert_close(result[key], actual[key], atol=1e-10, rtol=1e-10)

    def test_electrode_only_outputs_and_no_liquid_energy_fft(self):
        data = _field()
        wall = _module()
        _run(wall, data)  # Populate geometry cache before counting density FFTs.
        with patch.object(torch.fft, "fftn", wraps=torch.fft.fftn) as fft:
            result = _run(wall, data)
        self.assertEqual(fft.call_count, 1)  # Liquid source -> electrode potential only.
        self.assertEqual(wall._cache[0].shape, (8, 8))
        self.assertNotIn("coulomb_liquid_energy", result)
        self.assertNotIn("coulomb_energy", result)
        self.assertNotIn("liquid_charge_density", result)
        self.assertNotIn("q_liquid", result)
        torch.testing.assert_close(
            result["electrode_coulomb_energy"],
            result["coulomb_cross_energy"] + result["coulomb_metal_energy"],
        )

    def test_zero_field_uncharged_electrodes_have_zero_solution(self):
        data = _field()
        data["rho"] = torch.zeros_like(data["rho"])
        data["metal_total_charge"] = torch.zeros_like(data["metal_total_charge"])
        result = _run(_module(), data)
        for key in ("metal_site_q", "metal_potential", "electrode_coulomb_energy"):
            self.assertTrue(torch.equal(result[key], torch.zeros_like(result[key])), key)

    def test_forward_does_not_mutate_input(self):
        data = _field()
        before = {key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}
        _run(_module(), data)
        self.assertEqual(set(data), set(before))
        for key in data:
            if torch.is_tensor(data[key]):
                self.assertTrue(torch.equal(data[key], before[key]), key)
            else:
                self.assertEqual(data[key], before[key])

    def test_combined_neutrality_allows_a_charged_fluid_with_compensating_metal(self):
        data = _field()
        data["rho"][~data["excluded_mask"], 0] += 0.03
        module = _module()
        with self.assertRaisesRegex(ValueError, "neutral"):
            _run(module, data)
        fluid_charge = data["grid_spacing"].prod() * (data["rho"][:, 0] - data["rho"][:, 1]).sum()
        data["metal_total_charge"][0] -= fluid_charge
        result = _run(module, data)
        torch.testing.assert_close(result["metal_charge"], data["metal_total_charge"])
        torch.testing.assert_close(_liquid_charge(data).sum() + result["metal_site_q"].sum(), torch.zeros((), dtype=DTYPE), atol=1e-10, rtol=0)

    def test_invalid_or_occupied_metal_densities_are_rejected(self):
        for location, value in ((0, 0.1), (4, -0.1), (4, float("nan"))):
            with self.subTest(location=location, value=value):
                data = _field()
                data["rho"][location, 0] = value
                with self.assertRaises(ValueError):
                    _run(_module(), data)
        data = _field()
        data["rho"] = data["rho"].to(torch.long)
        with self.assertRaises(TypeError):
            _run(_module(), data)

    def test_invalid_or_incomplete_metal_sites_are_rejected_at_construction(self):
        sites = metal_sites(_field())
        for key in ("metal_group_ids", "metal_total_charge", "metal_charge_units"):
            with self.subTest(key=key):
                incomplete = dict(sites)
                del incomplete[key]
                with self.assertRaisesRegex(ValueError, "missing"):
                    MetalWall(
                        liquid_coulomb=liquid_coulomb(), metal_sites=incomplete,
                        metal_sigma=.35,
                    )
        for key, value in (
            ("metal_charge_units", "C"),
            ("metal_group_ids", torch.tensor((2, 2))),
            ("metal_group_ids", torch.tensor((2, 8))),
            ("metal_total_charge", torch.tensor((0.1,))),
        ):
            with self.subTest(key=key):
                invalid = dict(sites, **{key: value})
                with self.assertRaises(ValueError):
                    MetalWall(
                        liquid_coulomb=liquid_coulomb(), metal_sites=invalid,
                        metal_sigma=.35,
                    )

    def test_geometry_cache_is_excluded_from_serialization_and_dtype_migration(self):
        data = _field()
        module = _module()
        expected = _run(module, data)
        self.assertIsNotNone(module._cache)
        self.assertFalse(any("cache" in key for key in module.state_dict()))
        buffer = io.BytesIO()
        torch.save(module, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        self.assertIsNone(restored._cache)
        self.assertIsNone(restored._cache_key)
        torch.testing.assert_close(_run(restored, data)["metal_site_q"], expected["metal_site_q"])
        restored.float()
        self.assertIsNone(restored._cache)
        single = {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value for key, value in data.items()}
        result = _run(restored, single)
        self.assertEqual(result["metal_site_q"].dtype, torch.float32)
        torch.testing.assert_close(result["metal_site_q"].double(), expected["metal_site_q"], atol=2e-6, rtol=2e-5)

    def test_multiaxis_batch_matches_single_field_outputs(self):
        data = _field()
        factors = torch.tensor([[1.0, 1.1], [0.9, 1.2]], dtype=DTYPE)
        densities = factors[..., None, None] * data["rho"]
        module = _module()
        # One module-owned electrode is shared across the density batch.
        actual = _run(module, dict(data, rho=densities))
        for i, j in itertools.product(range(2), repeat=2):
            expected = _run(_module(), dict(data, rho=densities[i, j]))
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assertEqual(actual[key].shape, (2, 2) + expected[key].shape)
                torch.testing.assert_close(actual[key][i, j], expected[key], atol=0, rtol=0)

        varying_spacing = factors[..., None] * data["grid_spacing"]
        with self.assertRaisesRegex(ValueError, "grid_spacing"):
            _run(module, dict(data, rho=densities, grid_spacing=varying_spacing))

    def test_multiaxis_liquid_batch_uses_one_fixed_wall(self):
        data = {key: value for key, value in _field().items()
                if not key.startswith("metal_")}
        densities = data["rho"].expand(2, 2, -1, -1).clone().requires_grad_(True)
        actual = _run(_module(), dict(data, rho=densities))
        gradient = torch.autograd.grad(actual["electrode_coulomb_energy"].sum(), densities)[0]
        single_density = data["rho"].clone().requires_grad_(True)
        expected = _run(_module(), dict(data, rho=single_density))
        expected_gradient = torch.autograd.grad(expected["electrode_coulomb_energy"], single_density)[0]
        for key in expected:
            self.assertEqual(actual[key].shape, (2, 2) + expected[key].shape)
            for i, j in itertools.product(range(2), repeat=2):
                torch.testing.assert_close(actual[key][i, j], expected[key], atol=0, rtol=0)
                torch.testing.assert_close(gradient[i, j], expected_gradient, atol=0, rtol=0)

    def test_cached_geometry_handles_spacing_sites_and_group_changes(self):
        module = _module()
        fields = [
            _field(),
            _field(spacing=(0.9, 1.3, 0.7)),
            _field(group_ids=(11, 0)),
            _field(shape=(2, 3, 2)),
        ]
        for data in fields:
            # The final shape has only metal cells; avoid a singular uniform
            # charge mode by retaining one nonmetal cell in each x plane.
            if tuple(data["grid_size"].tolist()) == (2, 3, 2):
                keep = ~torch.isin(_site_rows(data), torch.tensor([1, 7]))
                data["metal_positions"] = data["metal_positions"][keep]
                data["metal_site_groups"] = data["metal_site_groups"][keep]
            expected = _dense_solution(data)
            actual = _run(module, data)
            torch.testing.assert_close(actual["metal_site_q"], expected["metal_site_q"], atol=2e-7, rtol=2e-7)
            torch.testing.assert_close(actual["electrode_coulomb_energy"], expected["electrode_coulomb_energy"], atol=2e-7, rtol=2e-7)

    def test_first_and_second_derivatives_include_electrode_response(self):
        data = _field()
        module = _module()
        accessible = torch.nonzero(~data["excluded_mask"]).flatten()
        basis = torch.zeros((len(data["rho"]), len(accessible) - 1), dtype=DTYPE)
        basis[accessible[:-1], torch.arange(len(accessible) - 1)] = 1
        basis[accessible[-1], :] = -1
        direction = torch.tensor((1.0, 0.0), dtype=DTYPE)

        def energy(coordinates):
            perturbed = dict(data)
            perturbed["rho"] = data["rho"] + (basis @ coordinates)[:, None] * direction
            return _run(module, perturbed)["electrode_coulomb_energy"]

        coordinates = torch.zeros(basis.shape[1], dtype=DTYPE, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(energy, (coordinates,), eps=1e-6, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.autograd.gradgradcheck(energy, (coordinates,), eps=1e-6, atol=1e-5, rtol=1e-4))
        hessian = torch.autograd.functional.hessian(energy, coordinates)
        reference = _dense_solution(data)["effective_kernel"]
        expected = data["grid_spacing"].prod().square() * basis.T @ reference @ basis
        torch.testing.assert_close(hessian, expected, atol=2e-7, rtol=2e-7)
        # At fixed electrode charge the correction is linear in liquid density.
        frozen_metal_hessian = torch.zeros_like(hessian)
        self.assertGreater((hessian - frozen_metal_hessian).abs().max().item(), 1e-4)

    def test_inference_created_cache_supports_later_density_derivatives(self):
        data = _field()
        module = _module()
        with torch.inference_mode():
            _run(module, data)
        data["rho"].requires_grad_(True)
        energy = _run(module, data)["electrode_coulomb_energy"]
        gradient = torch.autograd.grad(energy, data["rho"], create_graph=True)[0]
        hessian_row = torch.autograd.grad(gradient[4, 0], data["rho"])[0]
        self.assertTrue(torch.all(torch.isfinite(hessian_row)))
        fresh = _run(_module(), data)["electrode_coulomb_energy"]
        expected = torch.autograd.grad(fresh, data["rho"])[0]
        torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)

    def test_python_spacing_preserves_double_precision(self):
        spacing = (0.8123456789, 1.123456789, 1.3123456789)
        data = _field(spacing=spacing)
        expected = _run(_module(), data)
        actual = _run(_module(), dict(data, grid_spacing=spacing))
        for key in ("metal_site_q", "electrode_coulomb_energy"):
            torch.testing.assert_close(actual[key], expected[key], atol=1e-14, rtol=1e-14)

    def test_legacy_torch_lu_solve_preserves_density_gradients(self):
        data = _field()
        data["rho"].requires_grad_(True)
        expected = _run(_module(), data)
        expected_gradient = torch.autograd.grad(
            expected["electrode_coulomb_energy"], data["rho"],
        )[0]
        with patch.object(torch.linalg, "lu_solve", None):
            actual = _run(_module(), data)
        gradient = torch.autograd.grad(actual["electrode_coulomb_energy"], data["rho"])[0]
        torch.testing.assert_close(actual["metal_site_q"], expected["metal_site_q"])
        torch.testing.assert_close(gradient, expected_gradient)


if __name__ == "__main__":
    unittest.main()
