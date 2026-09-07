"""Independent discrete-electrostatics checks for polarizable metal cells."""

import itertools
import io
import math
import unittest
from unittest.mock import patch

import torch

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
        "metal_mask": mask,
        "metal_group_ids": torch.tensor(group_ids),
        "metal_total_charge": torch.tensor((0.15, -0.15), dtype=DTYPE),
        "metal_charge_units": "e",
    }


def _module(sigma=0.35, liquid_sigma=0.12, amplitude=1.7, charges=(1.0, -1.0)):
    return MetalWall(
        charges=charges,
        sigma=sigma,
        liquid_sigma=liquid_sigma,
        coulomb_amplitude=amplitude,
        boundary="periodic",
    ).double()


def _dense_kernels(data, sigma, liquid_sigma, amplitude):
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
        kernel(2 * liquid_sigma**2),
        kernel(liquid_sigma**2 + sigma**2),
        kernel(2 * sigma**2),
    )


def _dense_solution(data, sigma=0.35, liquid_sigma=0.12, amplitude=1.7, charges=(1.0, -1.0)):
    kll, klm, kmm = _dense_kernels(data, sigma, liquid_sigma, amplitude)
    liquid = data["rho"] @ torch.tensor(charges, dtype=DTYPE) * data["grid_spacing"].prod()
    selected = data["metal_mask"] >= 0
    groups = data["metal_group_ids"]
    constraints = (data["metal_mask"][selected][None, :] == groups[:, None]).to(DTYPE)
    a = kmm[selected][:, selected]
    b = klm[selected] @ liquid
    nmetal = int(selected.sum())
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
    ell = 0.5 * liquid @ kll @ liquid
    elm = liquid @ klm @ metal
    emm = 0.5 * metal @ kmm @ metal
    return {
        "q_liquid": liquid,
        "metal_q": metal,
        "q_mw": liquid + metal,
        "metal_potential": torch.stack([potentials[data["metal_mask"][selected] == group].mean() for group in groups]),
        "coulomb_liquid_energy": ell,
        "coulomb_cross_energy": elm,
        "coulomb_metal_energy": emm,
        "coulomb_energy": ell + elm + emm,
        "effective_kernel": kll - klm[:, selected] @ torch.linalg.inv(kkt)[:nmetal, :nmetal] @ klm[selected],
    }


class TestMetalWall(unittest.TestCase):
    def test_physical_parameters_and_boundary_must_be_valid(self):
        arguments = {
            "charges": (1.0, -1.0), "sigma": 0.35, "liquid_sigma": 0.0,
            "coulomb_amplitude": 1.7, "boundary": "periodic",
        }
        for key, value in (
            ("charges", ()), ("charges", (float("nan"), -1.0)),
            ("sigma", 0.0), ("sigma", -0.2), ("liquid_sigma", -0.1),
            ("coulomb_amplitude", 0.0), ("coulomb_amplitude", float("inf")),
            ("boundary", "slab"),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises((TypeError, ValueError)):
                    MetalWall(**dict(arguments, **{key: value}))
        with self.assertRaises(TypeError):
            MetalWall(**{key: value for key, value in arguments.items() if key != "boundary"})

    def test_dense_fourier_solution_and_energy_decomposition(self):
        data = _field()
        expected = _dense_solution(data)
        actual = _module()(data)
        for key in (
            "q_liquid", "metal_q", "q_mw", "metal_potential", "coulomb_energy",
            "coulomb_liquid_energy", "coulomb_cross_energy", "coulomb_metal_energy",
        ):
            with self.subTest(key=key):
                torch.testing.assert_close(actual[key], expected[key], atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(
            actual["coulomb_energy"],
            actual["coulomb_liquid_energy"] + actual["coulomb_cross_energy"] + actual["coulomb_metal_energy"],
        )

    def test_group_totals_and_equipotential_allow_nonuniform_metal_charges(self):
        data = _field()
        result = _module()(data)
        selected = data["metal_mask"] >= 0
        _, klm, kmm = _dense_kernels(data, 0.35, 0.12, 1.7)
        potential = klm @ result["q_liquid"] + kmm @ result["metal_q"]
        for index, group in enumerate(data["metal_group_ids"]):
            cells = data["metal_mask"] == group
            torch.testing.assert_close(result["metal_q"][cells].sum(), data["metal_total_charge"][index])
            torch.testing.assert_close(potential[cells], result["metal_potential"][index].expand(int(cells.sum())), atol=2e-7, rtol=2e-7)
        self.assertGreater(result["metal_q"][data["metal_mask"] == 2].std().item(), 1e-5)
        self.assertTrue(torch.equal(result["metal_q"][~selected], torch.zeros_like(result["metal_q"][~selected])))
        torch.testing.assert_close(result["metal_charge"], data["metal_total_charge"])
        self.assertLess(result["charge_residual"].abs().max().item(), 1e-10)
        self.assertLess(result["potential_residual"].abs().max().item(), 2e-7)

    def test_component_valencies_and_voxel_volume_determine_integrated_charge(self):
        data = _field()
        data["rho"] = torch.cat((data["rho"][:, :1] / 2, data["rho"][:, 1:], 0.7 * (data["metal_mask"] < 0)[:, None]), dim=-1)
        result = _module(charges=(2.0, -1.0, 0.0))(data)
        expected = data["grid_spacing"].prod() * (2 * data["rho"][:, 0] - data["rho"][:, 1])
        torch.testing.assert_close(result["q_liquid"], expected)
        torch.testing.assert_close(
            result["liquid_charge_density"], 2 * data["rho"][:, 0] - data["rho"][:, 1],
        )

    def test_length_unit_conversion_preserves_integrated_charges_and_energy(self):
        data = _field(spacing=(0.5, 0.75, 1.25))
        # Include nonzero net liquid charge, exactly compensated by the metal.
        data["rho"][data["metal_mask"] < 0, 0] += 0.1
        volume = data["grid_spacing"].prod()
        data["metal_total_charge"][0] -= volume * (data["rho"][:, 0] - data["rho"][:, 1]).sum()
        actual = _module()(data)
        reference = _dense_solution(data)
        for key in ("q_liquid", "metal_q", "coulomb_liquid_energy",
                    "coulomb_cross_energy", "coulomb_metal_energy", "coulomb_energy"):
            torch.testing.assert_close(actual[key], reference[key], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(actual["q_liquid"], volume * actual["liquid_charge_density"])
        torch.testing.assert_close(
            volume * actual["liquid_charge_density"].sum() + actual["metal_q"].sum(),
            torch.zeros((), dtype=DTYPE), atol=1e-12, rtol=0,
        )
        # Change the coordinate unit, keeping the energy and charge units fixed.
        # Lengths and Coulomb amplitude scale by s; densities by s**(-3).
        scale = 2.0
        converted = dict(data, grid_spacing=data["grid_spacing"] * scale,
                         rho=data["rho"] / scale**3)
        result = _module(sigma=0.35 * scale, liquid_sigma=0.12 * scale,
                         amplitude=1.7 * scale)(converted)
        torch.testing.assert_close(
            result["liquid_charge_density"], actual["liquid_charge_density"] / scale**3,
        )
        for key in ("q_liquid", "metal_q", "q_mw", "metal_charge", "metal_potential",
                    "coulomb_liquid_energy", "coulomb_cross_energy",
                    "coulomb_metal_energy", "coulomb_energy"):
            torch.testing.assert_close(result[key], actual[key], atol=1e-10, rtol=1e-10)

    def test_no_metal_reduces_to_liquid_fourier_energy(self):
        data = _field()
        data["metal_mask"] = torch.full_like(data["metal_mask"], -3)
        data["metal_group_ids"] = torch.empty(0, dtype=torch.long)
        data["metal_total_charge"] = torch.empty(0, dtype=DTYPE)
        for liquid_sigma in (0.0, 0.2):
            with self.subTest(liquid_sigma=liquid_sigma):
                result = _module(liquid_sigma=liquid_sigma)(data)
                liquid = data["rho"] @ torch.tensor((1.0, -1.0), dtype=DTYPE) * data["grid_spacing"].prod()
                kll, _, _ = _dense_kernels(data, 0.35, liquid_sigma, 1.7)
                torch.testing.assert_close(result["coulomb_energy"], 0.5 * liquid @ kll @ liquid, atol=2e-7, rtol=2e-7)
                torch.testing.assert_close(result["q_mw"], liquid)
                self.assertEqual(result["metal_charge"].numel(), 0)
                self.assertTrue(torch.equal(result["metal_q"], torch.zeros_like(liquid)))

    def test_zero_field_uncharged_electrodes_have_zero_solution(self):
        data = _field()
        data["rho"] = torch.zeros_like(data["rho"])
        data["metal_total_charge"] = torch.zeros_like(data["metal_total_charge"])
        result = _module()(data)
        for key in ("q_liquid", "metal_q", "q_mw", "metal_potential", "coulomb_energy"):
            self.assertTrue(torch.equal(result[key], torch.zeros_like(result[key])), key)

    def test_forward_does_not_mutate_input(self):
        data = _field()
        before = {key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}
        _module()(data)
        self.assertEqual(set(data), set(before))
        for key in data:
            if torch.is_tensor(data[key]):
                self.assertTrue(torch.equal(data[key], before[key]), key)
            else:
                self.assertEqual(data[key], before[key])

    def test_combined_neutrality_allows_a_charged_fluid_with_compensating_metal(self):
        data = _field()
        data["rho"][data["metal_mask"] < 0, 0] += 0.03
        module = _module()
        with self.assertRaisesRegex(ValueError, "neutral"):
            module(data)
        fluid_charge = data["grid_spacing"].prod() * (data["rho"][:, 0] - data["rho"][:, 1]).sum()
        data["metal_total_charge"][0] -= fluid_charge
        result = module(data)
        torch.testing.assert_close(result["metal_charge"], data["metal_total_charge"])
        torch.testing.assert_close(result["q_mw"].sum(), torch.zeros((), dtype=DTYPE), atol=1e-10, rtol=0)

    def test_invalid_or_occupied_metal_densities_are_rejected(self):
        for location, value in ((0, 0.1), (4, -0.1), (4, float("nan"))):
            with self.subTest(location=location, value=value):
                data = _field()
                data["rho"][location, 0] = value
                with self.assertRaises(ValueError):
                    _module()(data)
        data = _field()
        data["rho"] = data["rho"].to(torch.long)
        with self.assertRaises(TypeError):
            _module()(data)

    def test_missing_charge_metadata_is_not_silently_zero(self):
        for key in ("metal_group_ids", "metal_total_charge", "metal_charge_units"):
            with self.subTest(key=key):
                data = _field()
                del data[key]
                with self.assertRaisesRegex(ValueError, "explicit"):
                    _module()(data)
        for key, value in (
            ("metal_charge_units", "C"),
            ("metal_group_ids", torch.tensor((2, 2))),
            ("metal_group_ids", torch.tensor((2, 8))),
            ("metal_total_charge", torch.tensor((0.1,))),
        ):
            with self.subTest(key=key):
                data = _field()
                data[key] = value
                with self.assertRaises(ValueError):
                    _module()(data)

    def test_geometry_cache_is_excluded_from_serialization_and_dtype_migration(self):
        data = _field()
        module = _module()
        expected = module(data)
        self.assertIsNotNone(module._cache)
        self.assertFalse(any("cache" in key for key in module.state_dict()))
        buffer = io.BytesIO()
        torch.save(module, buffer)
        buffer.seek(0)
        restored = torch.load(buffer)
        self.assertIsNone(restored._cache)
        self.assertIsNone(restored._cache_key)
        torch.testing.assert_close(restored(data)["metal_q"], expected["metal_q"])
        restored.float()
        self.assertIsNone(restored._cache)
        single = {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value for key, value in data.items()}
        result = restored(single)
        self.assertEqual(result["metal_q"].dtype, torch.float32)
        torch.testing.assert_close(result["metal_q"].double(), expected["metal_q"], atol=2e-6, rtol=2e-5)

    def test_absent_metal_metadata_preserves_the_liquid_only_path(self):
        data = _field()
        data = {key: value for key, value in data.items() if not key.startswith("metal_")}
        data["rho"][:, 0] += 0.1
        result = _module(liquid_sigma=0.0)(data)
        kll, _, _ = _dense_kernels(data, 0.35, 0.0, 1.7)
        liquid = data["grid_spacing"].prod() * (data["rho"][:, 0] - data["rho"][:, 1])
        self.assertGreater(liquid.sum().item(), 0)
        torch.testing.assert_close(result["coulomb_energy"], 0.5 * liquid @ kll @ liquid, atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(result["q_mw"], liquid)

    def test_multiaxis_batch_matches_single_field_outputs(self):
        data = _field()
        factors = torch.tensor([[1.0, 1.1], [0.9, 1.2]], dtype=DTYPE)
        densities = factors[..., None, None] * data["rho"]
        spacings = factors[..., None] * data["grid_spacing"]
        totals = factors[..., None] * data["metal_total_charge"]
        module = _module()
        # The mask, IDs and grid size are shared; spacing and totals vary.
        for spacing in (data["grid_spacing"], spacings):
            with self.subTest(shared_spacing=spacing.ndim == 1):
                actual = module(dict(data, rho=densities, grid_spacing=spacing,
                                     metal_total_charge=totals))
                for i, j in itertools.product(range(2), repeat=2):
                    expected = _module()(dict(
                        data, rho=densities[i, j], metal_total_charge=totals[i, j],
                        grid_spacing=spacing if spacing.ndim == 1 else spacing[i, j],
                    ))
                    self.assertEqual(actual.keys(), expected.keys())
                    for key in expected:
                        self.assertEqual(actual[key].shape, (2, 2) + expected[key].shape)
                        torch.testing.assert_close(actual[key][i, j], expected[key], atol=0, rtol=0)

    def test_multiaxis_liquid_only_batch_preserves_outputs_and_gradients(self):
        data = {key: value for key, value in _field().items()
                if not key.startswith("metal_")}
        densities = data["rho"].expand(2, 2, -1, -1).clone().requires_grad_(True)
        actual = _module()(dict(data, rho=densities))
        gradient = torch.autograd.grad(actual["coulomb_energy"].sum(), densities)[0]
        single_density = data["rho"].clone().requires_grad_(True)
        expected = _module()(dict(data, rho=single_density))
        expected_gradient = torch.autograd.grad(expected["coulomb_energy"], single_density)[0]
        for key in expected:
            self.assertEqual(actual[key].shape, (2, 2) + expected[key].shape)
            for i, j in itertools.product(range(2), repeat=2):
                torch.testing.assert_close(actual[key][i, j], expected[key], atol=0, rtol=0)
                torch.testing.assert_close(gradient[i, j], expected_gradient, atol=0, rtol=0)

    def test_cached_geometry_handles_spacing_mask_and_group_changes(self):
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
                data["metal_mask"][[1, 7]] = -1
            expected = _dense_solution(data)
            actual = module(data)
            torch.testing.assert_close(actual["metal_q"], expected["metal_q"], atol=2e-7, rtol=2e-7)
            torch.testing.assert_close(actual["coulomb_energy"], expected["coulomb_energy"], atol=2e-7, rtol=2e-7)

    def test_batched_fields_keep_distinct_group_labels_and_constraints(self):
        fields = [_field(), _field(group_ids=(5, 3))]
        fields[1]["metal_total_charge"] *= -2
        batched = {
            key: torch.stack([data[key] for data in fields]) if torch.is_tensor(fields[0][key]) else fields[0][key]
            for key in fields[0]
        }
        module = _module()
        actual = module(batched)
        for index, data in enumerate(fields):
            expected = _dense_solution(data)
            for key in ("metal_q", "metal_potential", "coulomb_energy"):
                torch.testing.assert_close(actual[key][index], expected[key], atol=2e-7, rtol=2e-7)

    def test_first_and_second_derivatives_include_electrode_response(self):
        data = _field()
        module = _module()
        accessible = torch.nonzero(data["metal_mask"] < 0).flatten()
        basis = torch.zeros((len(data["rho"]), len(accessible) - 1), dtype=DTYPE)
        basis[accessible[:-1], torch.arange(len(accessible) - 1)] = 1
        basis[accessible[-1], :] = -1
        direction = torch.tensor((1.0, 0.0), dtype=DTYPE)

        def energy(coordinates):
            perturbed = dict(data)
            perturbed["rho"] = data["rho"] + (basis @ coordinates)[:, None] * direction
            return module(perturbed)["coulomb_energy"]

        coordinates = torch.zeros(basis.shape[1], dtype=DTYPE, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(energy, (coordinates,), eps=1e-6, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.autograd.gradgradcheck(energy, (coordinates,), eps=1e-6, atol=1e-5, rtol=1e-4))
        hessian = torch.autograd.functional.hessian(energy, coordinates)
        reference = _dense_solution(data)["effective_kernel"]
        expected = data["grid_spacing"].prod().square() * basis.T @ reference @ basis
        torch.testing.assert_close(hessian, expected, atol=2e-7, rtol=2e-7)
        kll, _, _ = _dense_kernels(data, 0.35, 0.12, 1.7)
        frozen_metal_hessian = data["grid_spacing"].prod().square() * basis.T @ kll @ basis
        self.assertGreater((hessian - frozen_metal_hessian).abs().max().item(), 1e-4)

    def test_inference_created_cache_supports_later_density_derivatives(self):
        data = _field()
        module = _module()
        with torch.inference_mode():
            module(data)
        data["rho"].requires_grad_(True)
        energy = module(data)["coulomb_energy"]
        gradient = torch.autograd.grad(energy, data["rho"], create_graph=True)[0]
        hessian_row = torch.autograd.grad(gradient[4, 0], data["rho"])[0]
        self.assertTrue(torch.all(torch.isfinite(hessian_row)))
        fresh = _module()(data)["coulomb_energy"]
        expected = torch.autograd.grad(fresh, data["rho"])[0]
        torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)

    def test_python_spacing_preserves_double_precision(self):
        spacing = (0.8123456789, 1.123456789, 1.3123456789)
        data = _field(spacing=spacing)
        expected = _module()(data)
        actual = _module()(dict(data, grid_spacing=spacing))
        for key in ("metal_q", "coulomb_energy"):
            torch.testing.assert_close(actual[key], expected[key], atol=1e-14, rtol=1e-14)

    def test_legacy_torch_lu_solve_preserves_density_gradients(self):
        data = _field()
        data["rho"].requires_grad_(True)
        expected = _module()(data)
        expected_gradient = torch.autograd.grad(
            expected["coulomb_energy"], data["rho"],
        )[0]
        with patch.object(torch.linalg, "lu_solve", None):
            actual = _module()(data)
        gradient = torch.autograd.grad(actual["coulomb_energy"], data["rho"])[0]
        torch.testing.assert_close(actual["metal_q"], expected["metal_q"])
        torch.testing.assert_close(gradient, expected_gradient)


if __name__ == "__main__":
    unittest.main()
