import math
import unittest

import torch

from equicdft import FourierResponseMetrics, Metrics
from equicdft.metrics import compute_metric
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
        self.assertEqual(values["rmse"].dtype, torch.float32)
        self.assertEqual(metrics.logs["train"]["count"], 3)

        metrics.clear_metrics("train")
        with self.assertRaisesRegex(ValueError, "no metric data"):
            metrics.retrieve_metrics("train")

    def test_streaming_matches_concatenated_metrics(self):
        metric_keys = (
            "mae",
            "rmse",
            "rmse_percent",
            "mse",
            "pearson_r",
            "r2",
        )
        metrics = Metrics("c1", metric_keys=metric_keys)
        targets = (
            torch.tensor([1.0e8 + 1.0, 1.0e8 + 2.0], dtype=torch.float64),
            torch.tensor(
                [1.0e8 - 3.0, 1.0e8 + 5.0, 1.0e8 + 7.0],
                dtype=torch.float64,
            ),
        )
        predictions = (
            targets[0] + torch.tensor([0.5, -1.5], dtype=torch.float64),
            targets[1] + torch.tensor([2.0, -0.5, 1.0], dtype=torch.float64),
        )
        for target, prediction in zip(targets, predictions):
            metrics.update_metrics(
                "valid",
                {"c1": prediction},
                {"c1": target},
            )

        values = metrics.retrieve_metrics("valid", clear=False)
        target = torch.cat(targets)
        prediction = torch.cat(predictions)

        self.assertEqual(tuple(values), metric_keys)
        for key in metric_keys:
            torch.testing.assert_close(
                values[key],
                compute_metric(key, target, prediction),
            )

    def test_streaming_storage_is_fixed_size(self):
        metrics = Metrics("c1")
        for offset in range(20):
            metrics.update_metrics(
                "train",
                {"c1": torch.arange(5.0) + offset},
                {"c1": torch.arange(5.0)},
            )

        log = metrics.logs["train"]
        self.assertEqual(log["count"], 100)
        self.assertEqual(len(log), 10)
        self.assertTrue(
            all(
                value is None
                or torch.is_tensor(value)
                or isinstance(value, torch.dtype)
                for key, value in log.items()
                if key not in ("count", "trailing_shape")
            )
        )
        self.assertEqual(log["trailing_shape"], ())
        self.assertLessEqual(
            sum(value.numel() for value in log.values() if torch.is_tensor(value)),
            7,
        )

    def test_each_metric_streams_across_multiple_batches(self):
        target_batches = (
            torch.tensor([1.0, -2.0], dtype=torch.float32),
            torch.tensor([4.0], dtype=torch.float32),
        )
        prediction_batches = (
            torch.tensor([2.0, -1.5], dtype=torch.float32),
            torch.tensor([3.0], dtype=torch.float32),
        )
        target = torch.cat(target_batches)
        prediction = torch.cat(prediction_batches)

        for key in (
            "mae",
            "mse",
            "rmse",
            "rmse_percent",
            "pearson_r",
            "r2",
        ):
            with self.subTest(metric=key):
                metrics = Metrics("c1", metric_keys=(key,))
                for target_batch, prediction_batch in zip(
                    target_batches,
                    prediction_batches,
                ):
                    metrics.update_metrics(
                        "train",
                        {"c1": prediction_batch},
                        {"c1": target_batch},
                    )
                value = metrics.retrieve_metrics("train")[key]
                torch.testing.assert_close(
                    value,
                    compute_metric(key, target, prediction),
                )

    def test_streaming_statistics_are_not_serialized(self):
        metrics = Metrics("c1")
        self.assertEqual(metrics.state_dict(), {})

        metrics.update_metrics(
            "train",
            {"c1": torch.tensor([2.0])},
            {"c1": torch.tensor([1.0])},
        )

        self.assertEqual(metrics.state_dict(), {})

    def test_retrieval_clears_by_default(self):
        metrics = Metrics("c1")
        metrics.update_metrics(
            "valid",
            {"c1": torch.tensor([1.0, 2.0])},
            {"c1": torch.tensor([1.0, 2.0])},
        )

        metrics.retrieve_metrics("valid")

        self.assertEqual(metrics.logs["valid"]["count"], 0)
        self.assertTrue(
            all(
                value is None
                for key, value in metrics.logs["valid"].items()
                if key not in ("count", "result_dtype", "trailing_shape")
            )
        )
        self.assertIsNone(metrics.logs["valid"]["trailing_shape"])

    def test_streaming_rejects_nonfloating_and_incompatible_batches(self):
        metrics = Metrics("c1")
        with self.assertRaisesRegex(TypeError, "floating-point"):
            metrics.update_metrics(
                "train",
                {"c1": torch.ones(2, dtype=torch.int64)},
                {"c1": torch.ones(2, dtype=torch.int64)},
            )

        metrics.update_metrics(
            "train",
            {"c1": torch.ones(2, 3)},
            {"c1": torch.ones(2, 3)},
        )
        with self.assertRaisesRegex(ValueError, "trailing shapes"):
            metrics.update_metrics(
                "train",
                {"c1": torch.ones(2, 4)},
                {"c1": torch.ones(2, 4)},
            )
        self.assertIsNone(metrics.logs["valid"]["result_dtype"])

    def test_output_target_is_expanded_and_selection_masked(self):
        metrics = Metrics(
            target_key="average_mu",
            prediction_key="local_mu",
            metric_keys=("mae", "rmse"),
            selection_mask_key="rho_selection",
        )
        outputs = {
            "local_mu": torch.tensor(
                [
                    [[1.0], [3.0], [100.0]],
                    [[4.0], [6.0], [8.0]],
                ]
            ),
            "average_mu": torch.tensor([[2.0], [6.0]]),
            "rho_selection": torch.tensor(
                [
                    [[1.0], [1.0], [0.0]],
                    [[1.0], [1.0], [1.0]],
                ]
            ),
        }

        values = metrics(outputs, {})

        self.assertAlmostEqual(values["mae"].item(), 1.2)
        self.assertAlmostEqual(values["rmse"].item(), math.sqrt(2.0))

        metrics.update_metrics("test", outputs, {})
        streamed = metrics.retrieve_metrics("test")
        self.assertAlmostEqual(streamed["mae"].item(), 1.2)
        self.assertAlmostEqual(streamed["rmse"].item(), math.sqrt(2.0))

    def test_ordered_targets_mix_known_and_inferred_values(self):
        metrics = Metrics(
            target_key=("beta_mu", "average_mu"),
            prediction_key="local_mu",
            metric_keys=("mae", "rmse"),
        )
        outputs = {
            "local_mu": torch.tensor(
                [
                    [[1.0], [3.0]],
                    [[4.0], [6.0]],
                ]
            ),
            "average_mu": torch.tensor([[2.0], [5.0]]),
        }
        batch = {"beta_mu": torch.tensor([[float("nan")], [4.0]])}

        values = metrics(outputs, batch)

        self.assertAlmostEqual(values["mae"].item(), 1.0)
        self.assertAlmostEqual(values["rmse"].item(), math.sqrt(1.5))

    def test_missing_and_empty_selection_masks_are_reported(self):
        metrics = Metrics(
            "target",
            prediction_key="prediction",
            selection_mask_key="selection",
        )
        outputs = {"prediction": torch.ones(2)}
        batch = {"target": torch.ones(2)}

        with self.assertRaisesRegex(KeyError, "selection mask"):
            metrics(outputs, batch)
        with self.assertRaisesRegex(ValueError, "retain"):
            metrics(
                {
                    "prediction": torch.ones(2),
                    "selection": torch.zeros(2),
                },
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

        recorded = metrics.logs["test"]
        self.assertTrue(
            all(
                not value.requires_grad
                for value in recorded.values()
                if torch.is_tensor(value)
            )
        )
        self.assertEqual(recorded["count"], 1)

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

    def test_non_real_or_non_floating_tensors_are_rejected(self):
        metrics = Metrics("c1")
        with self.assertRaisesRegex(TypeError, "real floating-point"):
            metrics(
                {"c1": torch.tensor([1])},
                {"c1": torch.tensor([1])},
            )
        with self.assertRaisesRegex(TypeError, "real floating-point"):
            metrics.update_metrics(
                "train",
                {"c1": torch.tensor([1.0 + 1.0j])},
                {"c1": torch.tensor([1.0 + 0.0j])},
            )

    def test_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            Metrics("c1", metric_keys=("mae", "invalid"))
        with self.assertRaisesRegex(ValueError, "unique"):
            Metrics("c1", metric_keys=("mae", "mae"))
        with self.assertRaisesRegex(ValueError, "subsets"):
            Metrics("c1", subsets=())


class TestFourierResponseMetrics(unittest.TestCase):
    @staticmethod
    def _details():
        prediction = torch.tensor([[[2.0, 4.0]], [[-1.0, 8.0]]])
        target = torch.tensor([[[1.0, 2.0]], [[1.0, 4.0]]])
        scale = target.clone()
        weight = torch.tensor([[[1.0, 1.0]], [[0.0, 2.0]]])
        scaled_error = (prediction - target) / scale
        element_loss = torch.where(
            scaled_error.abs() < 1.0,
            0.5 * scaled_error.square(),
            scaled_error.abs() - 0.5,
        )
        return {
            "prediction": prediction,
            "target": target,
            "scale": scale,
            "element_weight": weight,
            "element_loss": element_loss,
        }

    def test_reports_weighted_curvature_and_response_metrics(self):
        metrics = FourierResponseMetrics(
            ("number", "charge"),
            subsets=("valid",),
        )
        metrics.update_metrics("valid", self._details())

        values = metrics.retrieve_metrics("valid")

        self.assertAlmostEqual(values["loss"].item(), 0.5)
        self.assertAlmostEqual(values["K_number_rmse"].item(), 1.0)
        self.assertAlmostEqual(
            values["K_charge_rmse"].item(),
            (12.0 ** 0.5),
            places=6,
        )
        self.assertEqual(values["K_number_nonpositive"].item(), 0)
        self.assertEqual(values["K_charge_nonpositive"].item(), 0)
        self.assertAlmostEqual(
            values["S_number_positive_fraction"].item(), 1.0
        )
        self.assertEqual(metrics.logs["valid"]["prediction"], [])

    def test_details_and_subset_are_validated(self):
        metrics = FourierResponseMetrics(("number", "charge"))
        details = self._details()
        details.pop("scale")
        with self.assertRaisesRegex(KeyError, "missing"):
            metrics.update_metrics("train", details)
        with self.assertRaisesRegex(KeyError, "subset"):
            metrics.update_metrics("unknown", self._details())


if __name__ == "__main__":
    unittest.main()
