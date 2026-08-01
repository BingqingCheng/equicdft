import unittest

import torch
from torch import nn

from equicdft import (
    Loss,
    TensorLoss,
)


class _PredictionPenalty(nn.Module):
    """Minimal specialized term used to test the aggregation interface."""

    name = "prediction_penalty"

    def forward(self, outputs, batch):
        return outputs["c1"].abs().mean()


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
