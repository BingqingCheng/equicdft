import unittest

import torch
from torch import nn

from equicdft import (
    DensityPerturbationStabilityLoss,
    GlobalDensityStabilityLoss,
    Loss,
    TensorLoss,
)


class _PredictionPenalty(nn.Module):
    """Minimal specialized term used to test the aggregation interface."""

    name = "prediction_penalty"

    def forward(self, outputs, batch):
        return outputs["c1"].abs().mean()


class _UnstableQuadraticFunctional(nn.Module):
    """Small energy model whose concentrated fields are too favorable."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(3.0))
        self.register_buffer("thermal_wavelength", torch.ones(1))

    def forward(self, data, compute_c1=False):
        self.last_rho = data["rho"].detach().clone()
        beta_F_exc = -self.scale * torch.sum(
            data["rho"].square(),
            dim=(-2, -1),
        )
        return {"beta_F_exc": beta_F_exc}


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
            term(
                {"c1": torch.ones(2)},
                {"c1": torch.zeros(2)},
            )

    def test_weight_must_be_finite_and_nonnegative(self):
        for weight in (-1.0, float("inf"), float("nan"), "invalid"):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(ValueError, "weight"):
                    TensorLoss("c1", "c1", "c1", weight=weight)


class TestWeightedTensorLoss(unittest.TestCase):
    @staticmethod
    def _example():
        rho = torch.tensor([[[1.0], [2.0], [0.0], [4.0]]])

        # On the three unmasked grid points, construct local values
        # [1, 3, 5]. Their spatial mean is 3.
        desired_local = torch.tensor([[[1.0], [3.0], [0.0], [5.0]]])
        local_chemical_potential = desired_local.clone().requires_grad_(True)
        average_chemical_potential = (
            local_chemical_potential[:, [0, 1, 3], :].mean(dim=-2)
        )
        chemical_potential_weights = (rho > 0.5).to(rho.dtype)
        batch = {}
        outputs = {
            "local_chemical_potential": local_chemical_potential,
            "average_chemical_potential": average_chemical_potential,
            "chemical_potential_weights": chemical_potential_weights,
        }
        return outputs, batch

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
        batch = {"beta_mu": torch.tensor([[2.0]])}
        loss = TensorLoss(
            name="local_chemical_potential",
            prediction_key="local_chemical_potential",
            target_key="beta_mu",
            weights_key="chemical_potential_weights",
        )(outputs, batch)

        # The known componentwise beta*mu is expanded over the grid axis;
        # the empty voxel remains excluded by the same hard mask.
        self.assertAlmostEqual(loss.item(), 11.0 / 3.0, places=6)

    def test_zero_total_element_weight_is_rejected(self):
        outputs, batch = self._example()
        outputs["chemical_potential_weights"].zero_()

        with self.assertRaisesRegex(ValueError, "positive sum"):
            self._loss()(outputs, batch)


class TestDensityPerturbationStabilityLoss(unittest.TestCase):
    @staticmethod
    def _batch():
        return {
            "rho": torch.full((1, 4, 1), 0.5),
            "V_ext": torch.zeros(1, 4, 1),
            "beta": torch.ones(1),
            "temperature": torch.ones(1),
            "grid_spacing": torch.ones(1, 3),
            "grid_positions": torch.tensor(
                [[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]]
            ),
            "local_density_index": torch.arange(4).reshape(1, 4, 1),
        }

    def test_penalizes_lower_concentrated_objective_and_backpropagates(self):
        model = _UnstableQuadraticFunctional()
        batch = self._batch()
        outputs = model(batch)
        term = DensityPerturbationStabilityLoss(
            maximum_density=1.0,
            relative_amplitudes=(0.05,),
        )

        value = term(outputs, batch, model=model)
        value.backward()

        self.assertGreater(value.item(), 0.0)
        self.assertGreater(model.scale.grad.item(), 0.0)
        perturbed_sums = torch.sum(model.last_rho, dim=-2)
        self.assertTrue(
            torch.allclose(perturbed_sums, torch.full_like(perturbed_sums, 2.0))
        )
        relative_change = torch.abs(
            model.last_rho - batch["rho"][:, None, :, :]
        ) / batch["rho"][:, None, :, :]
        self.assertLessEqual(relative_change.max().item(), 0.0501)

    def test_aggregate_passes_model_only_to_model_dependent_term(self):
        model = _UnstableQuadraticFunctional()
        batch = self._batch()
        outputs = model(batch)
        loss = Loss(
            [
                DensityPerturbationStabilityLoss(
                    maximum_density=1.0,
                    relative_amplitudes=(0.05,),
                )
            ]
        )

        values = loss(outputs, batch, model=model)

        self.assertGreater(
            values["density_perturbation_stability"].item(),
            0.0,
        )
        self.assertEqual(
            values["total"].item(),
            values["density_perturbation_stability"].item(),
        )

    def test_rejects_reference_density_above_cap(self):
        model = _UnstableQuadraticFunctional()
        batch = self._batch()
        outputs = model(batch)
        term = DensityPerturbationStabilityLoss(
            maximum_density=0.4,
            relative_amplitudes=(0.05,),
        )

        with self.assertRaisesRegex(ValueError, "exceeds"):
            term(outputs, batch, model=model)


class TestGlobalDensityStabilityLoss(unittest.TestCase):
    @staticmethod
    def _batch():
        return TestDensityPerturbationStabilityLoss._batch()

    def test_penalizes_lower_finite_candidate_and_preserves_particles(self):
        model = _UnstableQuadraticFunctional()
        batch = self._batch()
        outputs = model(batch)
        term = GlobalDensityStabilityLoss(
            maximum_density=1.0,
            mixing_fractions=(0.25, 0.5, 1.0),
        )
        term.eval()

        value = term(outputs, batch, model=model)
        value.backward()

        self.assertGreater(value.item(), 0.0)
        self.assertGreater(model.scale.grad.item(), 0.0)
        candidate_sums = torch.sum(model.last_rho, dim=-2)
        self.assertTrue(
            torch.allclose(
                candidate_sums,
                torch.full_like(candidate_sums, 2.0),
            )
        )
        self.assertGreaterEqual(model.last_rho.min().item(), 0.0)
        self.assertLessEqual(model.last_rho.max().item(), 1.0)

    def test_validates_mixing_fractions(self):
        for fractions in ((), (0.0,), (1.1,), (0.5, 0.25)):
            with self.subTest(fractions=fractions):
                with self.assertRaisesRegex(ValueError, "mixing_fractions"):
                    GlobalDensityStabilityLoss(
                        maximum_density=1.0,
                        mixing_fractions=fractions,
                    )

    def test_validates_trial_strategy(self):
        with self.assertRaisesRegex(ValueError, "trial_strategy"):
            GlobalDensityStabilityLoss(
                maximum_density=1.0,
                trial_strategy="unknown",
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
            {
                "c1": c1_prediction,
                "beta_F_exc": energy_prediction,
            },
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
        loss = Loss([_PredictionPenalty()])

        values = loss({"c1": torch.tensor([-2.0, 4.0])}, {})

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
