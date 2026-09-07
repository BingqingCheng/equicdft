"""Variational and inverse integration checks for grid metal electrodes."""

import unittest
from unittest.mock import patch

import torch

from equicdft import (
    GridCACEModel,
    GridSolver,
    LDAReadout,
    MetalElectrodeReadout,
    MetalWall,
)


class TestMetalModel(unittest.TestCase):
    @staticmethod
    def _wall():
        return MetalWall(
            charges=[1.0, -1.0],
            sigma=0.6,
            liquid_sigma=0.0,
            coulomb_amplitude=0.2,
            boundary="periodic",
        ).double()

    @classmethod
    def _model(cls, mode="beta", contribution="total", with_metal=True):
        readouts = [
            LDAReadout(
                mean_density=1.0, n_types=2, hidden_sizes=(), zero_init=True
            )
        ]
        if with_metal:
            readouts.append(
                MetalElectrodeReadout(cls._wall(), contribution=contribution)
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
            "metal_mask": mask,
            "metal_group_ids": torch.tensor([0, 3]),
            "metal_total_charge": torch.zeros(2, dtype=torch.float64),
            "metal_charge_units": "e",
        }

    def test_c1_and_c2_include_relaxed_electrode_response(self):
        model = self._model()
        data = self._data()
        outputs = model(data, compute_c2=True, c2_reference=(2, 0))
        electrode = model.readout[1].metalwall(data)
        expected_energy = electrode["coulomb_energy"] * data["beta"]
        self.assertTrue(torch.allclose(outputs["beta_F_exc"], expected_energy, atol=1e-12))
        self.assertGreater(electrode["metal_q"].abs().max().item(), 1e-6)
        # A neutral, particle-number-preserving variation avoids changing the
        # prescribed periodic electrostatic ensemble during finite differences.
        direction = torch.zeros_like(data["rho"])
        direction[2, 0], direction[3, 0] = 1.0, -1.0
        step = 1.0e-5
        perturbed = []
        for sign in (1.0, -1.0):
            shifted = dict(data, rho=data["rho"].detach() + sign * step * direction)
            perturbed.append(model(shifted))
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
            for contribution in ("correction", "total"):
                with self.subTest(mode=mode, contribution=contribution):
                    model = self._model(mode=mode, contribution=contribution)
                    data = self._data()
                    wall = model.readout[1].metalwall
                    with patch.object(wall, "forward", wraps=wall.forward) as call:
                        outputs = model(data, compute_c2=True, c2_reference=(2, 0))
                        self.assertEqual(call.call_count, 1)
                    state = wall(data)
                    for key, expected in state.items():
                        torch.testing.assert_close(outputs[key], expected)
                    self.assertEqual(outputs["metal_q"].shape, data["rho"].shape[:-1])
                    self.assertTrue(torch.all(outputs["metal_q"][data["metal_mask"] < 0] == 0))
                    self.assertTrue(torch.all(outputs["liquid_charge_density"][data["metal_mask"] >= 0] == 0))

    def test_model_outputs_obey_anisotropic_voxel_normalization(self):
        spacing = torch.tensor([0.5, 0.75, 1.25], dtype=torch.float64)
        model = GridCACEModel(
            None, None, [MetalElectrodeReadout(self._wall(), contribution="total")],
            grid_spacing=spacing, mean_temperature=2.0, boltzmann_constant=1.5,
        ).double()
        data = self._data()
        data["grid_spacing"] = spacing
        data["rho"][2, 0] += 0.4
        expected_density = data["rho"][:, 0] - data["rho"][:, 1]
        expected_charge = spacing.prod() * expected_density
        data["metal_total_charge"][0] = -expected_charge.sum()
        result = model(data, compute_c1=False)
        torch.testing.assert_close(result["liquid_charge_density"], expected_density)
        torch.testing.assert_close(result["q_liquid"], expected_charge)
        for group, total in zip(data["metal_group_ids"], data["metal_total_charge"]):
            torch.testing.assert_close(result["metal_q"][data["metal_mask"] == group].sum(), total, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            result["q_mw"].sum(), torch.zeros((), dtype=torch.float64), atol=1e-12, rtol=0,
        )

    def test_charge_outputs_keep_gradients_for_joint_response_loss(self):
        data = self._data()
        model = self._model()
        result = model(data, compute_c2=True, c2_reference=(2, 0))
        charge_gradient = torch.autograd.grad(
            result["metal_q"].square().sum(), data["rho"], retain_graph=True,
        )[0]
        self.assertGreater(charge_gradient.abs().max().item(), 1e-8)
        density_gradient = torch.autograd.grad(
            result["liquid_charge_density"].sum(), data["rho"], retain_graph=True,
        )[0]
        torch.testing.assert_close(density_gradient, torch.tensor([1.0, -1.0]).to(data["rho"]).expand_as(data["rho"]))
        loss = result["beta_F_exc"] + result["c1"].square().sum() + result["metal_q"].square().sum()
        gradient = torch.autograd.grad(loss, data["rho"])[0]
        self.assertTrue(torch.all(torch.isfinite(gradient)))

    def test_energy_only_outputs_are_fresh_and_respect_no_grad(self):
        model = self._model().eval()
        data = self._data()
        with torch.no_grad():
            first = model(data, compute_c1=False)
            snapshot = {key: value.clone() for key, value in first.items()}
            self.assertFalse(any(value.requires_grad for value in first.values()))
            data["rho"][2, 0] += 0.03
            data["rho"][3, 0] -= 0.03
            second = model(data, compute_c1=False)
        for key in snapshot:
            torch.testing.assert_close(first[key], snapshot[key])
        self.assertGreater((second["metal_q"] - first["metal_q"]).abs().max().item(), 1e-8)

    def test_metal_readout_alone_supplies_component_count(self):
        model = GridCACEModel(
            a_features=None, b_features=None,
            readout=[MetalElectrodeReadout(self._wall(), contribution="total")],
            grid_spacing=0.75, mean_temperature=2.0, boltzmann_constant=1.5,
        ).double()
        data = self._data()
        self.assertEqual(model.n_types, 2)
        actual = model(data, compute_c2=True, c2_reference=(2, 0))
        expected = self._model()(data, compute_c2=True, c2_reference=(2, 0))
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
                beta_outputs = beta_model(data)
                physical_outputs = physical_model(data)
                beta = 1.0 / (beta_model.boltzmann_constant * temperature)
                self.assertTrue(torch.allclose(
                    beta_outputs["beta_F_exc"], beta * physical_outputs["F_exc"],
                    atol=1e-12,
                ))
                for key in ("beta_F_exc", "c1"):
                    self.assertTrue(torch.allclose(
                        beta_outputs[key], physical_outputs[key], atol=1e-12,
                    ))

    def test_correction_without_metal_preserves_existing_readout(self):
        augmented = self._model(contribution="correction")
        previous = self._model(with_metal=False)
        # Make the original LDA nonzero so equality is not just a zero test.
        for model in (augmented, previous):
            with torch.no_grad():
                model.readout[0].mlp[-1].weight.fill_(0.2)
                model.readout[0].mlp[-1].bias.fill_(0.1)
        for mask_present in (False, True):
            with self.subTest(mask_present=mask_present):
                data = self._data()
                for key in ("metal_mask", "metal_group_ids", "metal_total_charge",
                            "metal_charge_units"):
                    data.pop(key)
                if mask_present:
                    data["metal_mask"] = torch.full((16,), -3, dtype=torch.long)
                expected = previous(data, compute_c2=True)
                actual = augmented(data, compute_c2=True)
                for key in ("beta_F_exc", "c1", "c2"):
                    self.assertTrue(torch.equal(actual[key], expected[key]))
                self.assertTrue(torch.all(actual["metal_q"] == 0))
                self.assertEqual(actual["metal_charge"].numel(), 0)
                torch.testing.assert_close(
                    actual["q_liquid"], augmented.voxel_volume * actual["liquid_charge_density"],
                )

    def test_batched_fields_preserve_independent_temperature_and_constraints(self):
        fields = [self._data(), self._data()]
        fields[1]["temperature"] = torch.tensor(3.0, dtype=torch.float64)
        fields[1]["metal_total_charge"] = torch.tensor([0.03, -0.03], dtype=torch.float64)
        batch = {
            key: torch.stack([item[key] for item in fields])
            if torch.is_tensor(fields[0][key]) else fields[0][key]
            for key in fields[0]
        }
        model = self._model()
        outputs = model(batch, compute_c2=True, c2_reference=(2, 0))
        for index, data in enumerate(fields):
            expected = model(data, compute_c2=True, c2_reference=(2, 0))
            for key in outputs:
                self.assertTrue(torch.allclose(outputs[key][index], expected[key], atol=1e-12))

    def test_inverse_baseline_and_perturbed_start_obey_both_masks(self):
        model = self._model().eval()
        data = self._data()
        # An insulating exclusion is distinct from either electrode group.
        data["excluded_mask"] = torch.zeros(16, dtype=torch.bool)
        data["excluded_mask"][7] = True
        excluded = data["excluded_mask"] | (data["metal_mask"] >= 0)
        accessible = ~excluded
        target = data["rho"].clone()
        target[excluded] = 0.0
        # Keep the two species neutral after removing the insulating cell.
        target[:, 1] *= target[:, 0].sum() / target[:, 1].sum()
        data["rho"] = target
        c1 = model(data)["c1"].detach()
        data["V_ext"] = torch.zeros_like(target)
        data["V_ext"][accessible] = (
            c1[accessible] - torch.log(target[accessible])
        ) / data["beta"]
        numbers = target.sum(dim=0) * model.voxel_volume
        data.pop("rho")
        solutions = []
        for initialization in (None, torch.flip(target, dims=(0,)) + 0.05):
            result = GridSolver(model).solve(
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
            electrode = model.readout[1].metalwall(dict(data, rho=result["rho"]))
            for key, expected in electrode.items():
                torch.testing.assert_close(result[key], expected)
            self.assertLess(electrode["charge_residual"].abs().max().item(), 1e-12)
            self.assertLess(electrode["potential_residual"].abs().max().item(), 1e-12)
            solutions.append(result["rho"])
        self.assertTrue(torch.allclose(solutions[0], solutions[1], atol=1e-9))


if __name__ == "__main__":
    unittest.main()
