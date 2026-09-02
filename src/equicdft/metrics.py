"""Accumulate and evaluate prediction metrics across complete grid fields."""

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from ._argument_checks import (
    boolean,
    nonempty_string,
    unique_strings,
)
from ._targets import TargetKeys, normalize_target_keys, resolve_target


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


class Metrics(nn.Module):
    """Record predictions and targets and report dataset-level metrics.

    Call :meth:`update_metrics` for every batch, then
    :meth:`retrieve_metrics` once at the end of an epoch. Metrics are evaluated
    after concatenating all batches, so RMSE and R2 are true dataset-level
    values rather than averages of per-batch values.

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

    def __init__(
        self,
        target_key: TargetKeys,
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
        super().__init__()

        self.target_keys = normalize_target_keys(target_key)
        self.target_key = (
            self.target_keys[0]
            if len(self.target_keys) == 1
            else self.target_keys
        )
        default_target_name = self.target_keys[0]
        self.prediction_key = nonempty_string(
            default_target_name if prediction_key is None else prediction_key,
            "prediction_key",
        )
        self.name = nonempty_string(
            default_target_name if name is None else name,
            "name",
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

        subset_names = unique_strings(subsets, "subsets")
        self.logs = {
            subset: {"prediction": [], "target": []}
            for subset in subset_names
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
    ) -> None:
        """Append one detached CPU batch to a subset's metric log."""

        self._require_subset(subset)
        prediction, target = self._collect_tensors(outputs, batch)
        self.logs[subset]["prediction"].append(prediction)
        self.logs[subset]["target"].append(target)

    def retrieve_metrics(
        self,
        subset: str,
        clear: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return metrics over all recorded batches for one subset."""

        clear = boolean(clear, "clear")
        self._require_subset(subset)
        if not self.logs[subset]["prediction"]:
            raise ValueError("no metric data recorded for subset '{}'".format(subset))

        prediction = torch.cat(self.logs[subset]["prediction"], dim=0)
        target = torch.cat(self.logs[subset]["target"], dim=0)
        values = {
            metric: compute_metric(metric, target, prediction)
            for metric in self.metric_keys
        }
        if clear:
            self.clear_metrics(subset)
        return values

    def clear_metrics(self, subset: str) -> None:
        """Discard all batches recorded for one subset."""

        self._require_subset(subset)
        self.logs[subset]["prediction"] = []
        self.logs[subset]["target"] = []

    def _collect_tensors(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select, validate, detach, and move one tensor pair to CPU."""

        if self.prediction_key not in outputs:
            raise KeyError(
                "model outputs are missing prediction '{}'".format(
                    self.prediction_key
                )
            )
        prediction = outputs[self.prediction_key]
        target = resolve_target(
            prediction=prediction,
            prediction_key=self.prediction_key,
            target_keys=self.target_keys,
            outputs=outputs,
            batch=batch,
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
        return (
            prediction.detach().to(device="cpu").clone(),
            target.detach().to(device="cpu").clone(),
        )

    def _require_subset(self, subset: str) -> None:
        if subset not in self.logs:
            raise KeyError(
                "unknown metric subset '{}'; choose from {}".format(
                    subset,
                    tuple(self.logs),
                )
            )


class FourierResponseMetrics(nn.Module):
    """Accumulate projected-curvature and diagonal-response diagnostics.

    The input to :meth:`update_metrics` is the dictionary returned by
    :meth:`equicdft.FourierResponseLoss.evaluate`. Curvature metrics use the
    supplied response weights. ``S`` diagnostics report ``1 / K`` separately
    for each projected direction and therefore describe diagonal projected
    responses, not an inversion of a full coupled response matrix.
    """

    requires_loss_details = True

    def __init__(
        self,
        direction_names: Sequence[str],
        name: str = "fourier_response",
        subsets: Sequence[str] = ("train", "valid", "test"),
    ) -> None:
        super().__init__()

        self.name = nonempty_string(name, "name")
        self.direction_names = unique_strings(
            direction_names,
            "direction_names",
        )
        subset_names = unique_strings(subsets, "subsets")
        self.logs = {
            subset: {
                key: []
                for key in (
                    "prediction",
                    "target",
                    "scale",
                    "element_weight",
                    "element_loss",
                )
            }
            for subset in subset_names
        }

    def update_metrics(
        self,
        subset: str,
        details: Dict[str, torch.Tensor],
    ) -> None:
        """Append one detached response-loss evaluation."""

        self._require_subset(subset)
        required = set(self.logs[subset])
        missing = required - set(details)
        if missing:
            raise KeyError(
                "response details are missing: {}".format(sorted(missing))
            )
        n_directions = len(self.direction_names)
        reference_shape = None
        for key in required:
            value = details[key]
            if not torch.is_tensor(value):
                raise TypeError("response detail '{}' must be a tensor".format(key))
            if value.ndim < 1 or value.shape[-1] != n_directions:
                raise ValueError(
                    "response details must end with n_directions={}".format(
                        n_directions
                    )
                )
            if reference_shape is None:
                reference_shape = value.shape
            elif value.shape != reference_shape:
                raise ValueError("all response details must have the same shape")
            if not torch.all(torch.isfinite(value)).item():
                raise ValueError("response details must be finite")
            self.logs[subset][key].append(
                value.detach().reshape(-1, n_directions).cpu().clone()
            )

    def retrieve_metrics(
        self,
        subset: str,
        clear: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return globally weighted response metrics for one subset."""

        clear = boolean(clear, "clear")
        self._require_subset(subset)
        if not self.logs[subset]["prediction"]:
            raise ValueError("no metric data recorded for subset '{}'".format(subset))
        values = {
            key: torch.cat(items, dim=0)
            for key, items in self.logs[subset].items()
        }
        prediction = values["prediction"]
        target = values["target"]
        scale = values["scale"]
        weight = values["element_weight"]
        element_loss = values["element_loss"]
        if torch.any(weight < 0.0).item():
            raise ValueError("element_weight must be nonnegative")
        total_weight = weight.sum()
        if total_weight.item() <= 0.0:
            raise ValueError("element_weight must have a positive sum")

        result = {"loss": torch.sum(weight * element_loss) / total_weight}
        for channel, label in enumerate(self.direction_names):
            channel_weight = weight[:, channel]
            valid = channel_weight > 0.0
            denominator = channel_weight.sum()
            if denominator.item() <= 0.0:
                raise ValueError(
                    "direction '{}' has no positive response weight".format(label)
                )
            error = prediction[:, channel] - target[:, channel]
            result["K_{}_rmse".format(label)] = torch.sqrt(
                torch.sum(channel_weight * error.square()) / denominator
            )
            relative_error = error / scale[:, channel]
            result["K_{}_relative_rmse_percent".format(label)] = 100.0 * torch.sqrt(
                torch.sum(channel_weight * relative_error.square()) / denominator
            )

            positive = valid & (prediction[:, channel] > 0.0)
            result["K_{}_nonpositive".format(label)] = torch.sum(
                valid & ~positive
            )
            result["S_{}_positive_fraction".format(label)] = (
                channel_weight[positive].sum() / denominator
            )
            positive_weight = channel_weight[positive]
            positive_denominator = positive_weight.sum()
            if positive_denominator.item() > 0.0:
                predicted_s = 1.0 / prediction[positive, channel]
                target_s = 1.0 / target[positive, channel]
                s_error = predicted_s - target_s
                result["S_{}_rmse_positive".format(label)] = torch.sqrt(
                    torch.sum(positive_weight * s_error.square())
                    / positive_denominator
                )
                result[
                    "S_{}_relative_rmse_positive_percent".format(label)
                ] = 100.0 * torch.sqrt(
                    torch.sum(
                        positive_weight * (s_error / target_s).square()
                    )
                    / positive_denominator
                )
            else:
                nan = torch.tensor(float("nan"))
                result["S_{}_rmse_positive".format(label)] = nan
                result[
                    "S_{}_relative_rmse_positive_percent".format(label)
                ] = nan

        if clear:
            self.clear_metrics(subset)
        return result

    def clear_metrics(self, subset: str) -> None:
        """Discard all recorded response batches for one subset."""

        self._require_subset(subset)
        for key in self.logs[subset]:
            self.logs[subset][key] = []

    def _require_subset(self, subset: str) -> None:
        if subset not in self.logs:
            raise KeyError(
                "unknown metric subset '{}'; choose from {}".format(
                    subset,
                    tuple(self.logs),
                )
            )
