"""Variational and inverse integration checks for coordinate-based electrodes."""

import itertools
import unittest
from unittest.mock import patch

import torch
from metal_helpers import _run, liquid_data, metal_sites

from equicdft import (
    GridCACEModel,
    GridSolver,
    LDAReadout,
    MetalElectrodeReadout,
    MetalWall,
)


class TestMetalModel(unittest.TestCase):
    def test_field_energy_readout_modes_and_c1_envelope(self):
        for mode in ("beta", "physical"):
            with self.subTest(mode=mode):
                model = self._model(mode=mode)
                data = self._data()
                data["metal_site_groups"].zero_()
                data["metal_group_ids"] = torch.tensor([0])
                data["metal_total_charge"] = torch.zeros(1, dtype=torch.float64)
                data["metal_external_field"] = torch.tensor([0.07, 0., 0.], dtype=torch.float64)
                data["metal_field_origin"] = torch.tensor([-0.375, 0., 0.], dtype=torch.float64)
                result = _run(model, data, compute_c2=True, c2_reference=(2, 0))
                state = _run(model.readout[1].metalwall, data)
                energy = state["coulomb_cross_energy"] + state["coulomb_metal_energy"] + state["metal_external_energy"]
                torch.testing.assert_close(result["beta_F_exc"], data["beta"] * energy)
                direction = torch.zeros_like(data["rho"])
                direction[2, 0], direction[4, 0] = 1., -1.
                step = 1e-5
                plus = _run(model, dict(data, rho=data["rho"].detach() + step * direction))
                minus = _run(model, dict(data, rho=data["rho"].detach() - step * direction))
                torch.testing.assert_close((plus["beta_F_exc"] - minus["beta_F_exc"]) / (2*step),
                                           -model.voxel_volume * (result["c1"] * direction).sum(), atol=1e-10, rtol=1e-8)
                torch.testing.assert_close((plus["c1"][2, 0] - minus["c1"][2, 0]) / (2*step),
                                           model.voxel_volume * (result["c2"] * direction).sum(), atol=1e-10, rtol=1e-8)

    @staticmethod
    def _wall():
        return MetalWall(
            metal_sites=metal_sites(TestMetalModel._data()),
            liquid_charges=[1.0, -1.0],
            metal_sigma=0.6,
            liquid_sigma=0.0,
            coulomb_amplitude=0.2,
            boundary="periodic",
        ).double()

    @classmethod
    def _model(cls, mode="beta", with_metal=True):
        readouts = [
            LDAReadout(
                mean_density=1.0, n_types=2, hidden_sizes=(), zero_init=True
            )
        ]
        if with_metal:
            readouts.append(
                MetalElectrodeReadout(cls._wall())
            )
        return GridCACEModel(
            a_features=None,
            b_features=None,
            readout=readouts,
            grid_spacing=0.75,
            mean_temperature=2.0,
            boltzmann_constant=1.5,
            free_energy_mode=mode,
        ).double()

    @staticmethod
    def _data():
        # Noncontiguous labels and multiple sites per electrode exercise the
        # charge-redistribution response, not just fixed single-site charges.
        mask = torch.full((16,), -1, dtype=torch.long)
        mask[:2], mask[-2:] = 0, 3
        rho = torch.zeros((16, 2), dtype=torch.float64)
        modulation = torch.linspace(-0.03, 0.03, 12, dtype=torch.float64)
        rho[2:14, 0] = 0.2 + modulation
        rho[2:14, 1] = 0.2 - modulation
        return {
            "rho": rho,
            "grid_size": torch.tensor([4, 2, 2]),
            "grid_spacing": torch.full((3,), 0.75, dtype=torch.float64),
            "temperature": torch.tensor(2.0, dtype=torch.float64),
            "beta": torch.tensor(1.0 / 3.0, dtype=torch.float64),
            "excluded_mask": mask >= 0,
            "metal_positions": torch.tensor(list(itertools.product(range(4), range(2), range(2))),
                                            dtype=torch.float64)[mask >= 0] * 0.75,
            "metal_site_groups": mask[mask >= 0],
            "metal_group_ids": torch.tensor([0, 3]),
            "metal_total_charge": torch.zeros(2, dtype=torch.float64),
            "metal_charge_units": "e",
        }

    def test_readout_has_no_contribution_mode(self):
        for mode in ("total", "correction"):
            with self.assertRaises(TypeError):
                MetalElectrodeReadout(self._wall(), contribution=mode)

    def test_model_and_wall_respect_independent_exclusions(self):
        data = self._data()
        model = self._model()
        wall = model.readout[1].metalwall
        for evaluate in (model, wall):
            with self.subTest(evaluator=type(evaluate).__name__):
                occupied = dict(data, rho=data["rho"].clone())
                occupied["rho"][0] = 0.01
                with self.assertRaisesRegex(ValueError, "zero at excluded"):
                    _run(evaluate, occupied)
                for invalid in (torch.zeros(16), torch.zeros(15, dtype=torch.bool),
                                torch.zeros((2, 16), dtype=torch.bool)):
                    with self.assertRaisesRegex(ValueError, "excluded_mask"):
                        _run(evaluate, dict(data, excluded_mask=invalid))

    def test_c1_and_c2_include_relaxed_electrode_response(self):
        model = self._model()
        data = self._data()
        outputs = _run(model, data, compute_c2=True, c2_reference=(2, 0))
        electrode = _run(model.readout[1].metalwall, data)
        expected_energy = electrode["electrode_coulomb_energy"] * data["beta"]
        self.assertTrue(torch.allclose(outputs["beta_F_exc"], expected_energy, atol=1e-12))
        self.assertGreater(electrode["metal_site_q"].abs().max().item(), 1e-6)
        # A neutral, particle-number-preserving variation avoids changing the
        # prescribed periodic electrostatic ensemble during finite differences.
        direction = torch.zeros_like(data["rho"])
        direction[2, 0], direction[3, 0] = 1.0, -1.0
        step = 1.0e-5
        perturbed = []
        for sign in (1.0, -1.0):
            shifted = dict(data, rho=data["rho"].detach() + sign * step * direction)
            perturbed.append(_run(model, shifted))
        finite_energy = (
            perturbed[0]["beta_F_exc"] - perturbed[1]["beta_F_exc"]
        ) / (2.0 * step)
        gradient_energy = -model.voxel_volume * (outputs["c1"] * direction).sum()
        self.assertTrue(torch.allclose(finite_energy, gradient_energy, atol=1e-10))
        finite_c1 = (
            perturbed[0]["c1"][2, 0] - perturbed[1]["c1"][2, 0]
        ) / (2.0 * step)
        gradient_c1 = model.voxel_volume * (outputs["c2"] * direction).sum()
        self.assertTrue(torch.allclose(finite_c1, gradient_c1, atol=1e-10))
        self.assertGreater(abs(gradient_c1.item()), 1.0e-5)

    def test_model_returns_metal_state_from_one_solve(self):
        for mode in ("beta", "physical"):
            with self.subTest(mode=mode):
                model = self._model(mode=mode)
                data = self._data()
                wall = model.readout[1].metalwall
                with patch.object(wall, "forward", wraps=wall.forward) as call:
                    outputs = _run(model, data, compute_c2=True, c2_reference=(2, 0))
                    self.assertEqual(call.call_count, 1)
                state = _run(wall, data)
                for key, expected in state.items():
                    torch.testing.assert_close(outputs[key], expected)
                self.assertEqual(outputs["metal_site_q"].shape, data["metal_site_groups"].shape)
                self.assertTrue(torch.all(outputs["liquid_charge_density"][data["excluded_mask"]] == 0))

    def test_model_outputs_obey_anisotropic_voxel_normalization(self):
        spacing = torch.tensor([0.5, 0.75, 1.25], dtype=torch.float64)
        model = GridCACEModel(
            None, None, [MetalElectrodeReadout(self._wall())],
            grid_spacing=spacing, mean_temperature=2.0, boltzmann_constant=1.5,
        ).double()
        data = self._data()
        data["grid_spacing"] = spacing
        data["rho"][2, 0] += 0.4
        expected_density = data["rho"][:, 0] - data["rho"][:, 1]
        expected_charge = spacing.prod() * expected_density
        data["metal_total_charge"][0] = -expected_charge.sum()
        result = _run(model, data, compute_c1=False)
        torch.testing.assert_close(result["liquid_charge_density"], expected_density)
        torch.testing.assert_close(result["q_liquid"], expected_charge)
        for group, total in zip(data["metal_group_ids"], data["metal_total_charge"]):
            torch.testing.assert_close(result["metal_site_q"][data["metal_site_groups"] == group].sum(), total, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            result["q_liquid"].sum() + result["metal_site_q"].sum(), torch.zeros((), dtype=torch.float64), atol=1e-12, rtol=0,
        )

    def test_charge_outputs_keep_gradients_for_joint_response_loss(self):
        data = self._data()
        model = self._model()
        result = _run(model, data, compute_c2=True, c2_reference=(2, 0))
        charge_gradient = torch.autograd.grad(
            result["metal_site_q"].square().sum(), data["rho"], retain_graph=True,
        )[0]
        self.assertGreater(charge_gradient.abs().max().item(), 1e-8)
        density_gradient = torch.autograd.grad(
            result["liquid_charge_density"].sum(), data["rho"], retain_graph=True,
        )[0]
        torch.testing.assert_close(density_gradient, torch.tensor([1.0, -1.0]).to(data["rho"]).expand_as(data["rho"]))
        loss = result["beta_F_exc"] + result["c1"].square().sum() + result["metal_site_q"].square().sum()
        gradient = torch.autograd.grad(loss, data["rho"])[0]
        self.assertTrue(torch.all(torch.isfinite(gradient)))

    def test_energy_only_outputs_are_fresh_and_respect_no_grad(self):
        model = self._model().eval()
        data = self._data()
        with torch.no_grad():
            first = _run(model, data, compute_c1=False)
            snapshot = {key: value.clone() for key, value in first.items()}
            self.assertFalse(any(value.requires_grad for value in first.values()))
            data["rho"][2, 0] += 0.03
            data["rho"][3, 0] -= 0.03
            second = _run(model, data, compute_c1=False)
        for key in snapshot:
            torch.testing.assert_close(first[key], snapshot[key])
        self.assertGreater((second["metal_site_q"] - first["metal_site_q"]).abs().max().item(), 1e-8)

    def test_metal_readout_alone_supplies_component_count(self):
        model = GridCACEModel(
            a_features=None, b_features=None,
            readout=[MetalElectrodeReadout(self._wall())],
            grid_spacing=0.75, mean_temperature=2.0, boltzmann_constant=1.5,
        ).double()
        data = self._data()
        self.assertEqual(model.n_types, 2)
        actual = _run(model, data, compute_c2=True, c2_reference=(2, 0))
        expected = _run(self._model(), data, compute_c2=True, c2_reference=(2, 0))
        for key in ("beta_F_exc", "c1", "c2"):
            torch.testing.assert_close(actual[key], expected[key])

    def test_beta_and_physical_modes_use_model_energy_convention(self):
        beta_model = self._model(mode="beta")
        physical_model = self._model(mode="physical")
        for temperature in (1.0, 2.5):
            with self.subTest(temperature=temperature):
                data = self._data()
                data["temperature"] = torch.tensor(temperature, dtype=torch.float64)
                # beta_F_exc must not inherit an inconsistent cached beta.
                data["beta"] = torch.tensor(123.0, dtype=torch.float64)
                beta_outputs = _run(beta_model, data)
                physical_outputs = _run(physical_model, data)
                beta = 1.0 / (beta_model.boltzmann_constant * temperature)
                self.assertTrue(torch.allclose(
                    beta_outputs["beta_F_exc"], beta * physical_outputs["F_exc"],
                    atol=1e-12,
                ))
                for key in ("beta_F_exc", "c1"):
                    self.assertTrue(torch.allclose(
                        beta_outputs[key], physical_outputs[key], atol=1e-12,
                    ))

    def test_electrode_readout_adds_to_existing_liquid_functional(self):
        for mode in ("beta", "physical"):
            with self.subTest(mode=mode):
                liquid = self._model(mode=mode, with_metal=False)
                combined = self._model(mode=mode)
                electrode_only = self._model(mode=mode)
                for model in (liquid, combined):
                    with torch.no_grad():
                        model.readout[0].mlp[-1].weight.fill_(0.2)
                        model.readout[0].mlp[-1].bias.fill_(0.1)
                data = self._data()
                data["metal_external_field"] = torch.tensor([.03, 0., 0.], dtype=torch.float64)
                outputs = [_run(model, dict(data, rho=data["rho"].clone()),
                                 compute_c2=True, c2_reference=(2, 0))
                           for model in (liquid, combined, electrode_only)]
                for key in ("beta_F_exc", "c1", "c2"):
                    torch.testing.assert_close(outputs[1][key], outputs[0][key] + outputs[2][key],
                                               atol=1e-12, rtol=1e-12)

    def test_batched_fields_preserve_independent_temperature(self):
        fields = [self._data(), self._data()]
        fields[1]["temperature"] = torch.tensor(3.0, dtype=torch.float64)
        batch = liquid_data({
            key: torch.stack([item[key] for item in fields])
            if torch.is_tensor(fields[0][key]) else fields[0][key]
            for key in fields[0]
        })
        model = self._model()
        outputs = _run(model, batch, compute_c2=True, c2_reference=(2, 0))
        for index, data in enumerate(fields):
            expected = _run(model, data, compute_c2=True, c2_reference=(2, 0))
            for key in outputs:
                self.assertTrue(torch.allclose(outputs[key][index], expected[key], atol=1e-12))

    def test_inverse_baseline_and_perturbed_start_obey_exclusions(self):
        self._check_inverse(self._data())

    def _check_inverse(self, data):
        model = self._model().eval()
        # An insulating exclusion is distinct from either electrode group.
        data["excluded_mask"][7] = True
        excluded = data["excluded_mask"]
        accessible = ~excluded
        target = data["rho"].clone()
        target[excluded] = 0.0
        # Keep the two species neutral after removing the insulating cell.
        target[:, 1] *= target[:, 0].sum() / target[:, 1].sum()
        data["rho"] = target
        c1 = _run(model, data)["c1"].detach()
        data["V_ext"] = torch.zeros_like(target)
        data["V_ext"][accessible] = (
            c1[accessible] - torch.log(target[accessible])
        ) / data["beta"]
        numbers = target.sum(dim=0) * model.voxel_volume
        data.pop("rho")
        solutions = []
        for initialization in (None, torch.flip(target, dims=(0,)) + 0.05):
            result = _run(GridSolver(model).solve,
                data,
                initial_rho=initialization,
                particle_numbers=numbers,
                method="euler",
                max_iter=250,
                tolerance_residual=1e-9,
                tolerance_change=1e-12,
                mixing=0.2,
            )
            self.assertTrue(result["converged"])
            self.assertTrue(torch.equal(
                result["rho"][excluded], torch.zeros_like(target[excluded]),
            ))
            self.assertTrue(torch.all(result["rho"][accessible] > 0.0))
            self.assertTrue(torch.allclose(
                result["rho"].sum(dim=0) * model.voxel_volume, numbers, atol=1e-12,
            ))
            self.assertTrue(torch.allclose(result["rho"], target, atol=1e-9, rtol=0.0))
            self.assertLess(result["max_euler_lagrange_residual"], 1e-9)
            electrode = _run(model.readout[1].metalwall, dict(data, rho=result["rho"]))
            for key, expected in electrode.items():
                torch.testing.assert_close(result[key], expected)
            self.assertLess(electrode["charge_residual"].abs().max().item(), 1e-12)
            self.assertLess(electrode["potential_residual"].abs().max().item(), 1e-12)
            solutions.append(result["rho"])
        self.assertTrue(torch.allclose(solutions[0], solutions[1], atol=1e-9))

    def test_constant_field_inverse_baseline_and_perturbed_start(self):
        # Reuse the manufactured inverse fixture with a nonzero field;
        # V_ext is derived independently of initialization, never from MD.
        data = self._data()
        data["metal_site_groups"].zero_()
        data["metal_group_ids"] = torch.tensor([0])
        data["metal_total_charge"] = torch.zeros(1, dtype=torch.float64)
        data["metal_external_field"] = torch.tensor([0.03, 0., 0.], dtype=torch.float64)
        data["metal_field_origin"] = torch.tensor([-0.375, 0., 0.], dtype=torch.float64)
        self._check_inverse(data)

    def test_liquid_external_field_counted_only_by_solver(self):
        model = self._model()
        data = self._data()
        data["metal_external_field"] = torch.tensor([0.03, 0., 0.], dtype=torch.float64)
        data["metal_field_origin"] = torch.tensor([-0.375, 0., 0.], dtype=torch.float64)
        original = _run(model, data)
        x = torch.arange(4, dtype=torch.float64).repeat_interleave(4) * 0.75
        data["V_ext"] = -0.03 * x[:, None] * torch.tensor([1., -1.], dtype=torch.float64)
        result = _run(GridSolver(model).evaluate, data)
        for key in ("beta_F_exc", "metal_site_q", "metal_external_energy", "c1"):
            torch.testing.assert_close(result[key], original[key])
        expected = data["beta"] * model.voxel_volume * (data["rho"] * data["V_ext"]).sum()
        torch.testing.assert_close(result["beta_V_ext"], expected)
        torch.testing.assert_close(model.readout[1].metalwall.external_field, data["metal_external_field"])


if __name__ == "__main__":
    unittest.main()
