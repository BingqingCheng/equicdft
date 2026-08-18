import unittest

import torch
from torch import nn

from equicdft import (
    FourierStabilityLoss,
    GridCACEModel,
    LDAReadout,
    Loss,
    TensorLoss,
)
from equicdft.stability import _feasible_modes


class _PredictionPenalty(nn.Module):
    """Minimal specialized term used to test loss aggregation."""

    name = "prediction_penalty"

    def forward(self, outputs, batch):
        return outputs["c1"].abs().mean()


class _QuadraticExcessModel(nn.Module):
    """Analytic local functional used to test projected curvature."""

    def __init__(self, coefficient):
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(float(coefficient)))
        self.last_rho = None

    def forward(self, data, compute_c1=None):
        rho = data["rho"]
        cell_volume = torch.prod(data["grid_spacing"].to(rho), dim=-1)
        self.last_rho = rho.detach()
        return {
            "beta_F_exc": (
                0.5
                * self.coefficient.to(rho)
                * cell_volume
                * torch.sum(rho.square(), dim=(-2, -1))
            )
        }


class TestTensorLoss(unittest.TestCase):
    def test_weighted_loss_and_gradient(self):
        prediction = torch.tensor([1.0, 3.0], requires_grad=True)
        term = TensorLoss(
            name="c1",
            prediction_key="predicted_c1",
            target_key="c1",
            loss_fn=nn.MSELoss(),
            weight=2.0,
        )

        value = term(
            {"predicted_c1": prediction},
            {"c1": torch.tensor([0.0, 1.0])},
        )
        value.backward()

        self.assertAlmostEqual(value.item(), 5.0)
        self.assertTrue(
            torch.allclose(prediction.grad, torch.tensor([2.0, 4.0]))
        )

    def test_default_is_mean_squared_error(self):
        term = TensorLoss("c1", "c1", "c1")
        value = term(
            {"c1": torch.tensor([1.0, 3.0])},
            {"c1": torch.tensor([0.0, 1.0])},
        )
        self.assertAlmostEqual(value.item(), 2.5)

    def test_shapes_must_match_exactly(self):
        term = TensorLoss("c1", "c1", "c1")
        with self.assertRaisesRegex(ValueError, "shape"):
            term(
                {"c1": torch.ones(2, 3, 1)},
                {"c1": torch.ones(6, 1)},
            )

    def test_missing_prediction_and_target_are_reported(self):
        term = TensorLoss("c1", "predicted_c1", "target_c1")
        with self.assertRaisesRegex(KeyError, "prediction"):
            term({}, {"target_c1": torch.tensor(0.0)})
        with self.assertRaisesRegex(KeyError, "target"):
            term({"predicted_c1": torch.tensor(0.0)}, {})

    def test_loss_function_must_return_scalar(self):
        term = TensorLoss(
            "c1",
            "c1",
            "c1",
            loss_fn=nn.MSELoss(reduction="none"),
        )
        with self.assertRaisesRegex(ValueError, "scalar"):
            term({"c1": torch.ones(2)}, {"c1": torch.zeros(2)})

    def test_weight_must_be_finite_and_nonnegative(self):
        for weight in (-1.0, float("inf"), float("nan"), "invalid"):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(ValueError, "weight"):
                    TensorLoss("c1", "c1", "c1", weight=weight)


class TestWeightedTensorLoss(unittest.TestCase):
    @staticmethod
    def _example():
        rho = torch.tensor([[[1.0], [2.0], [0.0], [4.0]]])
        local = torch.tensor(
            [[[1.0], [3.0], [0.0], [5.0]]],
            requires_grad=True,
        )
        outputs = {
            "local_chemical_potential": local,
            "average_chemical_potential": local[:, [0, 1, 3], :].mean(
                dim=-2
            ),
            "chemical_potential_weights": (rho > 0.5).to(rho.dtype),
        }
        return outputs, {}

    @staticmethod
    def _loss():
        return TensorLoss(
            name="local_chemical_potential",
            prediction_key="local_chemical_potential",
            target_key="average_chemical_potential",
            weights_key="chemical_potential_weights",
        )

    def test_unknown_mu_is_masked_spatial_mean(self):
        outputs, batch = self._example()
        loss = self._loss()(outputs, batch)
        self.assertAlmostEqual(loss.item(), 8.0 / 3.0, places=6)
        loss.backward()
        local = outputs["local_chemical_potential"]
        self.assertTrue(torch.all(torch.isfinite(local.grad)))
        self.assertEqual(local.grad[0, 2, 0].item(), 0.0)

    def test_known_beta_mu_is_selected_from_batch(self):
        outputs, _ = self._example()
        loss = TensorLoss(
            name="local_chemical_potential",
            prediction_key="local_chemical_potential",
            target_key="beta_mu",
            weights_key="chemical_potential_weights",
        )(outputs, {"beta_mu": torch.tensor([[2.0]])})
        self.assertAlmostEqual(loss.item(), 11.0 / 3.0, places=6)

    def test_ordered_targets_mix_known_and_inferred_mu(self):
        prediction = torch.tensor(
            [[[1.0], [3.0]], [[4.0], [6.0]]],
            requires_grad=True,
        )
        outputs = {
            "local_chemical_potential": prediction,
            "average_chemical_potential": torch.tensor([[2.0], [5.0]]),
            "chemical_potential_weights": torch.ones_like(prediction),
        }
        batch = {"beta_mu": torch.tensor([[float("nan")], [4.0]])}
        term = TensorLoss(
            name="local_chemical_potential",
            prediction_key="local_chemical_potential",
            target_key=("beta_mu", "average_chemical_potential"),
            weights_key="chemical_potential_weights",
        )
        loss = term(outputs, batch)
        loss.backward()
        self.assertAlmostEqual(loss.item(), 1.5, places=6)
        self.assertTrue(torch.all(torch.isfinite(prediction.grad)))

    def test_ordered_targets_require_a_complete_finite_fallback(self):
        prediction = torch.ones(1, 2, 1)
        term = TensorLoss(
            "mu",
            "local_mu",
            ("beta_mu", "average_mu"),
        )
        with self.assertRaisesRegex(KeyError, "target candidates"):
            term({"local_mu": prediction}, {})
        with self.assertRaisesRegex(ValueError, "unresolved nonfinite"):
            term(
                {"local_mu": prediction},
                {"beta_mu": torch.tensor([[float("nan")]])},
            )

    def test_ordered_target_configuration_is_validated(self):
        for target_key in ((), ("beta_mu", "beta_mu"), ("beta_mu", "")):
            with self.subTest(target_key=target_key):
                with self.assertRaisesRegex(ValueError, "target_key"):
                    TensorLoss("mu", "local_mu", target_key)

    def test_zero_total_element_weight_is_rejected(self):
        outputs, batch = self._example()
        outputs["chemical_potential_weights"].zero_()
        with self.assertRaisesRegex(ValueError, "positive sum"):
            self._loss()(outputs, batch)


class TestFourierStabilityLoss(unittest.TestCase):
    @staticmethod
    def _batch(dtype=torch.float64, n_fields=1):
        n_grid = 8
        return {
            "rho": torch.full(
                (n_fields, n_grid, 1),
                0.5,
                dtype=dtype,
            ),
            "temperature": torch.ones(n_fields, dtype=dtype),
            "grid_spacing": torch.ones(n_fields, 3, dtype=dtype),
            "grid_size": torch.tensor(
                [[n_grid, 1, 1]] * n_fields
            ),
            "grid_positions": torch.tensor(
                [
                    [[index, 0, 0] for index in range(n_grid)]
                    for _ in range(n_fields)
                ]
            ),
        }

    def test_stable_ideal_curvature_has_zero_penalty(self):
        batch = self._batch()
        model = _QuadraticExcessModel(0.0).to(dtype=torch.float64)
        outputs = model(batch)
        term = FourierStabilityLoss(
            modes=((1, 0, 0),),
            relative_amplitude=0.01,
        )

        value = term(outputs, batch, model=model)

        self.assertEqual(value.item(), 0.0)

    def test_unstable_curvature_is_penalized_and_differentiable(self):
        batch = self._batch()
        model = _QuadraticExcessModel(-4.0).to(dtype=torch.float64)
        outputs = model(batch)
        term = FourierStabilityLoss(
            modes=((1, 0, 0),),
            relative_amplitude=0.01,
        )

        value = term(outputs, batch, model=model)
        value.backward()

        self.assertGreater(value.item(), 0.9)
        self.assertLess(value.item(), 1.1)
        self.assertIsNotNone(model.coefficient.grad)
        self.assertTrue(torch.isfinite(model.coefficient.grad).item())
        self.assertNotEqual(model.coefficient.grad.item(), 0.0)

    def test_perturbations_preserve_particle_number(self):
        batch = self._batch()
        batch["rho"] = torch.cat(
            (batch["rho"], 0.5 * batch["rho"]),
            dim=-1,
        )
        model = _QuadraticExcessModel(-4.0).to(dtype=torch.float64)
        outputs = model(batch)
        term = FourierStabilityLoss(modes=((1, 0, 0), (2, 0, 0)))

        term(outputs, batch, model=model)

        reference_sum = batch["rho"].sum(dim=-2)
        perturbed_sums = model.last_rho.sum(dim=-2)
        self.assertTrue(
            torch.allclose(
                perturbed_sums,
                reference_sum[:, None, :].expand_as(perturbed_sums),
                atol=1.0e-12,
                rtol=0.0,
            )
        )

    def test_multicomponent_loss_averages_independent_component_curvatures(self):
        batch = self._batch()
        batch["rho"] = torch.cat(
            (batch["rho"], 0.5 * batch["rho"]),
            dim=-1,
        )
        model = _QuadraticExcessModel(-8.0).to(dtype=torch.float64)
        outputs = model(batch)
        term = FourierStabilityLoss(
            modes=((1, 0, 0),),
            relative_amplitude=0.01,
        )

        value = term(outputs, batch, model=model)
        value.backward()

        # For this local quadratic functional, the two normalized component
        # curvatures are 1 + rho_a * coefficient = -3 and -1. Their squared
        # stability hinges therefore average to (9 + 1) / 2 = 5.
        self.assertAlmostEqual(value.item(), 5.0, places=3)
        self.assertIsNotNone(model.coefficient.grad)

    def test_multicomponent_loss_integrates_with_grid_model(self):
        batch = self._batch()
        batch["rho"] = torch.cat(
            (batch["rho"], 0.5 * batch["rho"]),
            dim=-1,
        )
        readout = LDAReadout(
            mean_density=1.0,
            n_types=2,
            hidden_sizes=(),
        ).to(dtype=torch.float64)
        with torch.no_grad():
            readout.mlp[-1].weight.copy_(
                torch.tensor([[-4.0, -4.0, 0.0]], dtype=torch.float64)
            )
            readout.mlp[-1].bias.zero_()
        model = GridCACEModel(
            a_features=None,
            b_features=None,
            readout=[readout],
            grid_spacing=1.0,
            compute_c1=False,
            free_energy_mode="beta",
        ).to(dtype=torch.float64)
        outputs = model(batch, compute_c1=False)
        term = FourierStabilityLoss(
            modes=((1, 0, 0),),
            relative_amplitude=0.01,
        )

        value = term(outputs, batch, model=model)
        value.backward()

        self.assertAlmostEqual(value.item(), 5.0, places=3)
        self.assertTrue(torch.all(torch.isfinite(readout.mlp[-1].weight.grad)))

    def test_absent_component_is_excluded_from_average(self):
        batch = self._batch()
        batch["rho"] = torch.cat(
            (batch["rho"], torch.zeros_like(batch["rho"])),
            dim=-1,
        )
        model = _QuadraticExcessModel(-4.0).to(dtype=torch.float64)
        outputs = model(batch)
        term = FourierStabilityLoss(
            modes=((1, 0, 0),),
            relative_amplitude=0.01,
        )

        value = term(outputs, batch, model=model)

        self.assertAlmostEqual(value.item(), 1.0, places=3)

    def test_loss_aggregator_supplies_model(self):
        batch = self._batch()
        model = _QuadraticExcessModel(-4.0).to(dtype=torch.float64)
        outputs = model(batch)
        loss = Loss(
            [FourierStabilityLoss(modes=((1, 0, 0),), weight=2.0)]
        )

        values = loss(outputs, batch, model=model)

        self.assertEqual(list(values), ["fourier_stability", "total"])
        self.assertAlmostEqual(
            values["total"].item(),
            values["fourier_stability"].item(),
        )

    def test_nyquist_cosine_is_kept_and_zero_alias_is_rejected(self):
        batch = self._batch(dtype=torch.float32)
        model = _QuadraticExcessModel(0.0)
        outputs = model(batch)

        value = FourierStabilityLoss(((4, 0, 0),))(
            outputs,
            batch,
            model=model,
        )
        self.assertEqual(value.item(), 0.0)
        with self.assertRaisesRegex(ValueError, "Nyquist sphere"):
            FourierStabilityLoss(((8, 0, 0),))(
                outputs,
                batch,
                model=model,
            )

    def test_random_triplets_are_sampled_per_field_inside_nyquist_sphere(self):
        n_grid = 8
        positions = torch.tensor(
            [
                [x, y, z]
                for x in range(n_grid)
                for y in range(n_grid)
                for z in range(n_grid)
            ]
        )
        batch = {
            "rho": torch.full((2, n_grid**3, 1), 0.5),
            "temperature": torch.ones(2),
            "grid_spacing": torch.ones(2, 3),
            "grid_size": torch.tensor([[n_grid] * 3] * 2),
            "grid_positions": positions[None].expand(2, -1, -1),
        }
        term = FourierStabilityLoss(random_modes_per_field=3)
        torch.manual_seed(11)

        modes = term._select_modes(batch, batch["rho"])

        self.assertEqual(modes.shape, (2, 3, 3))
        scaled_squared_norm = torch.sum(
            (2.0 * modes.to(torch.float32) / n_grid) ** 2,
            dim=-1,
        )
        self.assertTrue(
            torch.all(scaled_squared_norm <= 1.0 + 1.0e-6).item()
        )
        self.assertTrue(torch.all(torch.any(modes != 0, dim=-1)).item())

    def test_anisotropic_spacing_uses_physical_nyquist_sphere(self):
        modes = set(_feasible_modes((8, 8, 8), (0.5, 1.0, 1.0)))

        # The x direction has a larger axis-specific Nyquist limit, but the
        # isotropically complete sphere is limited by y and z to k <= pi.
        self.assertIn((2, 0, 0), modes)
        self.assertNotIn((4, 0, 0), modes)
        for mode in modes:
            wavevector = torch.tensor(
                [
                    2.0 * torch.pi * index / (size * spacing)
                    for index, size, spacing in zip(
                        mode,
                        (8, 8, 8),
                        (0.5, 1.0, 1.0),
                    )
                ]
            )
            self.assertLessEqual(
                torch.linalg.vector_norm(wavevector).item(),
                torch.pi + 1.0e-6,
            )

    def test_training_only_random_loss_is_zero_during_validation(self):
        batch = self._batch()
        model = _QuadraticExcessModel(-4.0).to(dtype=torch.float64)
        outputs = model(batch)
        term = FourierStabilityLoss(
            modes=((1, 0, 0),),
            training_only=True,
        ).eval()

        value = term(outputs, batch, model=model)

        self.assertEqual(value.item(), 0.0)

    def test_configuration_and_scope_are_validated(self):
        for modes in ((), ((0, 0, 0),), ((1.5, 0, 0),)):
            with self.subTest(modes=modes):
                with self.assertRaises(ValueError):
                    FourierStabilityLoss(modes=modes)
        with self.assertRaisesRegex(ValueError, "relative_amplitude"):
            FourierStabilityLoss(((1, 0, 0),), relative_amplitude=1.0)
        with self.assertRaisesRegex(ValueError, "random_modes_per_field"):
            FourierStabilityLoss(random_modes_per_field=0)
        with self.assertRaisesRegex(ValueError, "must be zero"):
            FourierStabilityLoss(
                modes=((1, 0, 0),),
                random_modes_per_field=1,
            )
        batch = self._batch()
        model = _QuadraticExcessModel(0.0).to(dtype=torch.float64)
        del batch["grid_size"]
        with self.assertRaisesRegex(KeyError, "grid_size"):
            FourierStabilityLoss(((1, 0, 0),))(
                model(batch),
                batch,
                model=model,
            )


class TestLossAggregation(unittest.TestCase):
    def test_aggregates_named_weighted_terms(self):
        c1_prediction = torch.tensor([1.0, 3.0], requires_grad=True)
        energy_prediction = torch.tensor([3.0], requires_grad=True)
        loss = Loss(
            [
                TensorLoss(
                    "c1",
                    "c1",
                    "c1",
                    loss_fn=nn.MSELoss(),
                    weight=2.0,
                ),
                TensorLoss(
                    "free_energy",
                    "beta_F_exc",
                    "beta_F_exc",
                    loss_fn=nn.L1Loss(),
                    weight=0.5,
                ),
            ]
        )
        values = loss(
            {"c1": c1_prediction, "beta_F_exc": energy_prediction},
            {
                "c1": torch.tensor([0.0, 1.0]),
                "beta_F_exc": torch.tensor([1.0]),
            },
        )
        values["total"].backward()
        self.assertEqual(list(values), ["c1", "free_energy", "total"])
        self.assertAlmostEqual(values["c1"].item(), 5.0)
        self.assertAlmostEqual(values["free_energy"].item(), 1.0)
        self.assertAlmostEqual(values["total"].item(), 6.0)
        self.assertIsNotNone(c1_prediction.grad)
        self.assertIsNotNone(energy_prediction.grad)

    def test_accepts_specialized_term_with_same_interface(self):
        values = Loss([_PredictionPenalty()])(
            {"c1": torch.tensor([-2.0, 4.0])},
            {},
        )
        self.assertAlmostEqual(values["prediction_penalty"].item(), 3.0)
        self.assertAlmostEqual(values["total"].item(), 3.0)

    def test_terms_must_be_nonempty_named_and_unique(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            Loss([])
        with self.assertRaisesRegex(ValueError, "unique"):
            Loss(
                [
                    TensorLoss("c1", "c1", "c1"),
                    TensorLoss("c1", "c1_other", "c1_other"),
                ]
            )
        with self.assertRaisesRegex(ValueError, "reserved"):
            Loss([TensorLoss("total", "c1", "c1")])


if __name__ == "__main__":
    unittest.main()
