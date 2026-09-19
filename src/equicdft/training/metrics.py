"""Accumulate and evaluate prediction metrics across complete grid fields."""

from typing import Dict, Optional, Sequence, Tuple
from .._deprecation import keyword_only

import torch
from torch import nn

from .._validation import (
    boolean,
    nonempty_string,
    unique_strings,
)
from ._targets import TargetKeys, normalize_target_keys, resolve_target

__all__ = [
    "SUPPORTED_METRICS",
    "METRIC_LABELS",
    "metric_label",
    "format_metric_value",
    "compute_metric",
    "MetricCollection",
    "Metrics",
]


SUPPORTED_METRICS = (
    "mae",
    "rmse",
    "rmse_percent",
    "mse",
    "pearson_r",
    "r2",
)

METRIC_LABELS = {
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "rmse_percent": "RMSE / sigma (%)",
    "pearson_r": "Pearson r",
    "r2": "R2",
}


def metric_label(name: str) -> str:
    """Return the concise screen label associated with a metric key."""

    return METRIC_LABELS.get(name, name)


def format_metric_value(name: str, value: float) -> str:
    """Format a scalar using precision appropriate to its metric."""

    if name == "rmse_percent":
        return "{:.2f}".format(value)
    if name in ("pearson_r", "r2"):
        return "{:.5f}".format(value)
    return "{:.6e}".format(value)


def compute_metric(
    metric: str,
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """Compute one scalar metric over all supplied tensor entries."""

    if target.shape != prediction.shape:
        raise ValueError("prediction and target shapes must match exactly")
    if target.numel() == 0:
        raise ValueError("prediction and target tensors must not be empty")

    error = prediction - target
    if metric == "mae":
        return torch.mean(torch.abs(error))
    if metric == "mse":
        return torch.mean(error.square())
    rmse = torch.sqrt(torch.mean(error.square()))
    if metric == "rmse":
        return rmse

    target_centered = target.reshape(-1) - torch.mean(target)
    if metric == "rmse_percent":
        target_standard_deviation = torch.sqrt(
            torch.mean(target_centered.square())
        )
        return 100.0 * rmse / target_standard_deviation
    if metric == "pearson_r":
        prediction_centered = prediction.reshape(-1) - torch.mean(prediction)
        denominator = torch.sqrt(
            torch.sum(target_centered.square())
            * torch.sum(prediction_centered.square())
        )
        return torch.sum(target_centered * prediction_centered) / denominator
    if metric == "r2":
        return 1.0 - torch.sum(error.square()) / torch.sum(
            target_centered.square()
        )
    raise ValueError(
        "unknown metric '{}'; choose from {}".format(
            metric,
            SUPPORTED_METRICS,
        )
    )


class MetricCollection(nn.Module):
    """Base class of streaming metric accumulators used by the trainer.

    A collection owns one accumulator per named subset. The trainer calls
    :meth:`update_metrics` after every batch with the model outputs, the
    batch, and the per-term details returned by :meth:`equicdft.Loss.evaluate`,
    then :meth:`retrieve_metrics` once per epoch. Subclasses decide which of
    those inputs they consume.
    """

    def __init__(
        self,
        name: str,
        subsets: Sequence[str] = ("train", "valid", "test"),
    ) -> None:
        super().__init__()

        self.name = nonempty_string(name, "name")
        self.subsets = unique_strings(subsets, "subsets")

    def update_metrics(
        self,
        subset: str,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        details: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """Accumulate one detached batch into a subset's statistics."""

        raise NotImplementedError

    def retrieve_metrics(
        self,
        subset: str,
        clear: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return metrics over all recorded batches for one subset."""

        raise NotImplementedError

    def clear_metrics(self, subset: str) -> None:
        """Discard all batches recorded for one subset."""

        raise NotImplementedError

    def _require_subset(self, subset: str) -> None:
        if subset not in self.subsets:
            raise KeyError(
                "unknown metric subset '{}'; choose from {}".format(
                    subset,
                    tuple(self.subsets),
                )
            )


class Metrics(MetricCollection):
    """Stream predictions and targets into dataset-level metrics.

    Call :meth:`update_metrics` for every batch, then
    :meth:`retrieve_metrics` once at the end of an epoch. Fixed-size
    sufficient statistics are accumulated in float64, so RMSE and R2 are true
    dataset-level values rather than averages of per-batch values without
    retaining complete fields.

    Parameters
    ----------
    target_key
        One key or an ordered sequence of reference keys. For each key the
        batch is checked first, followed by model outputs. The first available
        finite value is used. A componentwise target without the prediction's
        grid axis is expanded over that axis.
    prediction_key
        Key selecting the predicted tensor from the model outputs. Defaults to
        ``target_key``.
    name
        Name used by calling code when reporting this metric collection.
        Defaults to ``target_key``.
    metric_keys
        Metrics to evaluate. Supported values are ``mae``, ``rmse``, ``mse``,
        ``rmse_percent``, ``pearson_r``, and ``r2``. ``rmse_percent`` is
        ``100 * RMSE / sigma(target)``, where ``sigma`` is the population
        standard deviation over every recorded target value.
    subsets
        Independent accumulation buffers to create. The defaults are
        ``train``, ``valid``, and ``test``.
    selection_mask_key
        Optional key selecting an inclusion mask from the model outputs or
        batch. Positive entries are retained. The selection mask must have the
        same shape as the prediction.
    """

    @keyword_only(
        1,
        (
            "target_key",
            "prediction_key",
            "name",
            "metric_keys",
            "subsets",
            "selection_mask_key",
        ),
    )
    def __init__(
        self,
        target_key: TargetKeys,
        *,
        prediction_key: Optional[str] = None,
        name: Optional[str] = None,
        metric_keys: Sequence[str] = (
            "mae",
            "rmse",
            "rmse_percent",
            "r2",
        ),
        subsets: Sequence[str] = ("train", "valid", "test"),
        selection_mask_key: Optional[str] = None,
    ) -> None:
        target_keys = normalize_target_keys(target_key)
        default_target_name = target_keys[0]
        super().__init__(
            default_target_name if name is None else name,
            subsets,
        )

        self.target_keys = target_keys
        self.target_key = (
            self.target_keys[0]
            if len(self.target_keys) == 1
            else self.target_keys
        )
        self.prediction_key = nonempty_string(
            default_target_name if prediction_key is None else prediction_key,
            "prediction_key",
        )
        self.selection_mask_key = (
            None
            if selection_mask_key is None
            else nonempty_string(
                selection_mask_key,
                "selection_mask_key",
            )
        )
        self.metric_keys = unique_strings(metric_keys, "metric_keys")
        unknown_metrics = set(self.metric_keys) - set(SUPPORTED_METRICS)
        if unknown_metrics:
            raise ValueError(
                "unsupported metrics: {}".format(sorted(unknown_metrics))
            )

        self.logs = {
            subset: self._empty_statistics()
            for subset in self.subsets
        }

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics for one batch without changing stored logs."""

        prediction, target = self._collect_tensors(outputs, batch)
        return {
            metric: compute_metric(metric, target, prediction)
            for metric in self.metric_keys
        }

    def update_metrics(
        self,
        subset: str,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        details: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """Accumulate one detached batch; loss details are not needed."""

        self._require_subset(subset)
        prediction, target = self._select_tensors(outputs, batch)
        batch_statistics = self._batch_statistics(prediction, target)
        trailing_shape = self.logs[subset]["trailing_shape"]
        if (
            trailing_shape is not None
            and trailing_shape != batch_statistics["trailing_shape"]
        ):
            raise ValueError(
                "recorded prediction batches must have matching trailing "
                "shapes"
            )
        self.logs[subset] = self._merge_statistics(
            self.logs[subset],
            batch_statistics,
        )

    def retrieve_metrics(
        self,
        subset: str,
        clear: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return metrics over all recorded batches for one subset."""

        clear = boolean(clear, "clear")
        self._require_subset(subset)
        if self.logs[subset]["count"] == 0:
            raise ValueError("no metric data recorded for subset '{}'".format(subset))

        statistics = {
            key: value.cpu() if torch.is_tensor(value) else value
            for key, value in self.logs[subset].items()
        }
        values = self._metrics_from_statistics(statistics)
        if clear:
            self.clear_metrics(subset)
        return values

    def clear_metrics(self, subset: str) -> None:
        """Discard all batches recorded for one subset."""

        self._require_subset(subset)
        self.logs[subset] = self._empty_statistics()

    def _collect_tensors(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select, validate, detach, and move one tensor pair to CPU."""

        prediction, target = self._select_tensors(outputs, batch)
        return (
            prediction.detach().to(device="cpu").clone(),
            target.detach().to(device="cpu").clone(),
        )

    def _select_tensors(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select and validate one prediction-target tensor pair."""

        if self.prediction_key not in outputs:
            raise KeyError(
                "model outputs are missing prediction '{}'".format(
                    self.prediction_key
                )
            )
        prediction = outputs[self.prediction_key]
        if not prediction.is_floating_point():
            raise TypeError("prediction must be a real floating-point tensor")
        target = resolve_target(
            prediction=prediction,
            prediction_key=self.prediction_key,
            target_keys=self.target_keys,
            outputs=outputs,
            batch=batch,
        )
        for label, value in (("prediction", prediction), ("target", target)):
            if not value.is_floating_point() or value.is_complex():
                raise TypeError(
                    "{} must be a real floating-point tensor".format(label)
                )

        if self.selection_mask_key is not None:
            if self.selection_mask_key in outputs:
                selection_mask = outputs[self.selection_mask_key]
            elif self.selection_mask_key in batch:
                selection_mask = batch[self.selection_mask_key]
            else:
                raise KeyError(
                    "batch and model outputs are missing selection mask "
                    "'{}'".format(
                        self.selection_mask_key
                    )
                )
            if selection_mask.shape != prediction.shape:
                raise ValueError(
                    "selection mask and prediction must have same shape"
                )
            selection_mask = selection_mask.detach().to(dtype=torch.bool)
            prediction = prediction[selection_mask]
            target = target[selection_mask]
            if prediction.numel() == 0:
                raise ValueError(
                    "selection mask must retain at least one value"
                )

        if prediction.ndim == 0:
            prediction = prediction.unsqueeze(0)
            target = target.unsqueeze(0)
        return prediction, target

    def _empty_statistics(self) -> Dict[str, object]:
        """Return an empty fixed-size streaming accumulator."""

        return {
            "count": 0,
            "result_dtype": None,
            "trailing_shape": None,
            "absolute_error_sum": None,
            "squared_error_sum": None,
            "target_mean": None,
            "target_m2": None,
            "prediction_mean": None,
            "prediction_m2": None,
            "target_prediction_covariance": None,
        }

    def _batch_statistics(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, object]:
        """Reduce one batch to float64 error and centered-moment statistics."""

        result_dtype = torch.promote_types(prediction.dtype, target.dtype)
        prediction = prediction.detach()
        target = target.detach()
        error = prediction - target
        count = target.numel()

        needs_absolute_error = "mae" in self.metric_keys
        needs_squared_error = any(
            key in self.metric_keys
            for key in ("mse", "rmse", "rmse_percent", "r2")
        )
        needs_target_moments = any(
            key in self.metric_keys
            for key in ("rmse_percent", "pearson_r", "r2")
        )
        needs_prediction_moments = "pearson_r" in self.metric_keys

        target_mean = None
        target_m2 = None
        if needs_target_moments:
            target_mean = torch.mean(target)
            target_m2 = torch.sum((target - target_mean).square())

        prediction_mean = None
        prediction_m2 = None
        covariance = None
        if needs_prediction_moments:
            prediction_mean = torch.mean(prediction)
            prediction_centered = prediction - prediction_mean
            prediction_m2 = torch.sum(prediction_centered.square())
            covariance = torch.sum(
                (target - target_mean) * prediction_centered
            )

        return {
            "count": count,
            "result_dtype": result_dtype,
            "trailing_shape": tuple(prediction.shape[1:]),
            "absolute_error_sum": self._float64(
                torch.sum(torch.abs(error)) if needs_absolute_error else None
            ),
            "squared_error_sum": self._float64(
                torch.sum(error.square()) if needs_squared_error else None
            ),
            "target_mean": self._float64(target_mean),
            "target_m2": self._float64(target_m2),
            "prediction_mean": self._float64(prediction_mean),
            "prediction_m2": self._float64(prediction_m2),
            "target_prediction_covariance": self._float64(covariance),
        }

    @staticmethod
    def _float64(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Cast one reduced scalar to the accumulator dtype."""

        return None if value is None else value.to(dtype=torch.float64)

    @staticmethod
    def _merge_statistics(
        accumulated: Dict[str, object],
        batch: Dict[str, object],
    ) -> Dict[str, object]:
        """Merge two centered-moment accumulators with Chan's formulas."""

        if accumulated["count"] == 0:
            return batch

        device = next(
            value.device for value in batch.values() if torch.is_tensor(value)
        )
        accumulated = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in accumulated.items()
        }
        old_count = accumulated["count"]
        batch_count = batch["count"]
        count = old_count + batch_count
        result = {
            "count": count,
            "result_dtype": torch.promote_types(
                accumulated["result_dtype"],
                batch["result_dtype"],
            ),
            "trailing_shape": accumulated["trailing_shape"],
        }

        for key in ("absolute_error_sum", "squared_error_sum"):
            old_value = accumulated[key]
            batch_value = batch[key]
            result[key] = (
                None
                if old_value is None
                else old_value + batch_value
            )

        merge_factor = old_count * batch_count / count
        target_mean = accumulated["target_mean"]
        if target_mean is None:
            result["target_mean"] = None
            result["target_m2"] = None
        else:
            target_delta = batch["target_mean"] - target_mean
            result["target_mean"] = (
                target_mean + target_delta * batch_count / count
            )
            result["target_m2"] = (
                accumulated["target_m2"]
                + batch["target_m2"]
                + target_delta.square() * merge_factor
            )

        prediction_mean = accumulated["prediction_mean"]
        if prediction_mean is None:
            result["prediction_mean"] = None
            result["prediction_m2"] = None
            result["target_prediction_covariance"] = None
        else:
            prediction_delta = batch["prediction_mean"] - prediction_mean
            result["prediction_mean"] = (
                prediction_mean + prediction_delta * batch_count / count
            )
            result["prediction_m2"] = (
                accumulated["prediction_m2"]
                + batch["prediction_m2"]
                + prediction_delta.square() * merge_factor
            )
            result["target_prediction_covariance"] = (
                accumulated["target_prediction_covariance"]
                + batch["target_prediction_covariance"]
                + target_delta * prediction_delta * merge_factor
            )
        return result

    def _metrics_from_statistics(
        self,
        statistics: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        """Evaluate configured metrics from one complete accumulator."""

        count = statistics["count"]
        squared_error_sum = statistics["squared_error_sum"]
        target_m2 = statistics["target_m2"]
        values = {}
        for metric in self.metric_keys:
            if metric == "mae":
                value = statistics["absolute_error_sum"] / count
            elif metric == "mse":
                value = squared_error_sum / count
            elif metric == "rmse":
                value = torch.sqrt(squared_error_sum / count)
            elif metric == "rmse_percent":
                value = 100.0 * torch.sqrt(squared_error_sum / target_m2)
            elif metric == "pearson_r":
                value = statistics["target_prediction_covariance"] / torch.sqrt(
                    target_m2 * statistics["prediction_m2"]
                )
            elif metric == "r2":
                value = 1.0 - squared_error_sum / target_m2
            values[metric] = value.to(dtype=statistics["result_dtype"])
        return values


