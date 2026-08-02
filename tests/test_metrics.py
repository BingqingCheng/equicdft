import math
import unittest

import torch

from equicdft import Metrics, compute_metric
from equicdft.metrics import format_metric_value, metric_label


class TestComputeMetric(unittest.TestCase):
    def test_supported_metrics(self):
        target = torch.tensor([1.0, 2.0, 3.0])
        prediction = torch.tensor([1.0, 3.0, 2.0])

        self.assertAlmostEqual(
            compute_metric("mae", target, prediction).item(),
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            compute_metric("mse", target, prediction).item(),
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            compute_metric("rmse", target, prediction).item(),
            math.sqrt(2.0 / 3.0),
        )
        self.assertAlmostEqual(
            compute_metric("rmse_percent", target, prediction).item(),
            100.0,
        )
        self.assertAlmostEqual(
            compute_metric("pearson_r", target, prediction).item(),
            0.5,
        )
        self.assertAlmostEqual(
            compute_metric("r2", target, prediction).item(),
            0.0,
        )

    def test_constant_data_have_undefined_correlation(self):
        target = torch.ones(4)
        prediction = torch.ones(4)

        value = compute_metric("pearson_r", target, prediction)

        self.assertTrue(torch.isnan(value))

    def test_rejects_shape_mismatch_empty_and_unknown_metric(self):
        with self.assertRaisesRegex(ValueError, "shapes"):
            compute_metric("mae", torch.ones(2), torch.ones(2, 1))
        with self.assertRaisesRegex(ValueError, "empty"):
            compute_metric("mae", torch.empty(0), torch.empty(0))
        with self.assertRaisesRegex(ValueError, "unknown metric"):
            compute_metric("invalid", torch.ones(2), torch.ones(2))


class TestMetricFormatting(unittest.TestCase):
    def test_metric_labels(self):
        self.assertEqual(metric_label("mae"), "MAE")
        self.assertEqual(
            metric_label("rmse_percent"),
            "RMSE / sigma (%)",
        )
        self.assertEqual(metric_label("custom"), "custom")

    def test_metric_specific_precision(self):
        self.assertEqual(format_metric_value("rmse", 1.25), "1.250000e+00")
        self.assertEqual(format_metric_value("rmse_percent", 12.345), "12.35")
        self.assertEqual(format_metric_value("pearson_r", 0.123456), "0.12346")


class TestMetrics(unittest.TestCase):
    def test_default_metrics_use_r2(self):
        metrics = Metrics("c1")

        self.assertEqual(
            metrics.metric_keys,
            ("mae", "rmse", "rmse_percent", "r2"),
        )

    def test_forward_computes_unweighted_batch_metrics(self):
        metrics = Metrics(
            target_key="target_c1",
            prediction_key="predicted_c1",
            name="c1",
            metric_keys=("mae", "rmse", "pearson_r"),
        )

        values = metrics(
            {"predicted_c1": torch.tensor([1.0, 3.0, 2.0])},
            {"target_c1": torch.tensor([1.0, 2.0, 3.0])},
        )

        self.assertEqual(list(values), ["mae", "rmse", "pearson_r"])
        self.assertAlmostEqual(values["mae"].item(), 2.0 / 3.0)
        self.assertEqual(metrics.name, "c1")

    def test_retrieval_concatenates_batches_before_metrics(self):
        metrics = Metrics("c1", metric_keys=("mae", "rmse"))
        metrics.update_metrics(
            "train",
            {"c1": torch.tensor([[0.0], [0.0]])},
            {"c1": torch.tensor([[0.0], [0.0]])},
        )
        metrics.update_metrics(
            "train",
            {"c1": torch.tensor([[3.0]])},
            {"c1": torch.tensor([[0.0]])},
        )

        values = metrics.retrieve_metrics("train", clear=False)

        self.assertAlmostEqual(values["mae"].item(), 1.0)
        self.assertAlmostEqual(values["rmse"].item(), math.sqrt(3.0))
        self.assertEqual(len(metrics.logs["train"]["prediction"]), 2)

        metrics.clear_metrics("train")
        with self.assertRaisesRegex(ValueError, "no metric data"):
            metrics.retrieve_metrics("train")

    def test_retrieval_clears_by_default(self):
        metrics = Metrics("c1")
        metrics.update_metrics(
            "valid",
            {"c1": torch.tensor([1.0, 2.0])},
            {"c1": torch.tensor([1.0, 2.0])},
        )

        metrics.retrieve_metrics("valid")

        self.assertEqual(metrics.logs["valid"]["prediction"], [])
        self.assertEqual(metrics.logs["valid"]["target"], [])

    def test_output_target_is_expanded_and_hard_masked(self):
        metrics = Metrics(
            target_key="average_mu",
            prediction_key="local_mu",
            metric_keys=("mae", "rmse"),
            mask_key="rho_mask",
        )
        outputs = {
            "local_mu": torch.tensor(
                [
                    [[1.0], [3.0], [100.0]],
                    [[4.0], [6.0], [8.0]],
                ]
            ),
            "average_mu": torch.tensor([[2.0], [6.0]]),
            "rho_mask": torch.tensor(
                [
                    [[1.0], [1.0], [0.0]],
                    [[1.0], [1.0], [1.0]],
                ]
            ),
        }

        values = metrics(outputs, {})

        self.assertAlmostEqual(values["mae"].item(), 1.2)
        self.assertAlmostEqual(values["rmse"].item(), math.sqrt(2.0))

    def test_missing_and_empty_metric_masks_are_reported(self):
        metrics = Metrics("target", prediction_key="prediction", mask_key="mask")
        outputs = {"prediction": torch.ones(2)}
        batch = {"target": torch.ones(2)}

        with self.assertRaisesRegex(KeyError, "mask"):
            metrics(outputs, batch)
        with self.assertRaisesRegex(ValueError, "retain"):
            metrics(
                {"prediction": torch.ones(2), "mask": torch.zeros(2)},
                batch,
            )

    def test_detaches_recorded_tensors(self):
        metrics = Metrics("c1")
        prediction = torch.tensor([1.0], requires_grad=True)

        metrics.update_metrics(
            "test",
            {"c1": prediction},
            {"c1": torch.tensor([0.0])},
        )

        recorded = metrics.logs["test"]["prediction"][0]
        self.assertFalse(recorded.requires_grad)
        self.assertEqual(recorded.device.type, "cpu")

    def test_missing_keys_shape_and_subset_are_reported(self):
        metrics = Metrics("target_c1", prediction_key="predicted_c1")

        with self.assertRaisesRegex(KeyError, "prediction"):
            metrics({}, {"target_c1": torch.ones(1)})
        with self.assertRaisesRegex(KeyError, "target"):
            metrics({"predicted_c1": torch.ones(1)}, {})
        with self.assertRaisesRegex(ValueError, "shape"):
            metrics(
                {"predicted_c1": torch.ones(2)},
                {"target_c1": torch.ones(2, 1)},
            )
        with self.assertRaisesRegex(KeyError, "subset"):
            metrics.update_metrics(
                "unknown",
                {"predicted_c1": torch.ones(1)},
                {"target_c1": torch.ones(1)},
            )

    def test_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            Metrics("c1", metric_keys=("mae", "invalid"))
        with self.assertRaisesRegex(ValueError, "unique"):
            Metrics("c1", metric_keys=("mae", "mae"))
        with self.assertRaisesRegex(ValueError, "subsets"):
            Metrics("c1", subsets=())


if __name__ == "__main__":
    unittest.main()
