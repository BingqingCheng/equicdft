"""Streaming metrics of the Fourier-response loss details."""

from typing import Dict, Optional, Sequence
import torch
from .._validation import (
    boolean,
    nonempty_string,
    unique_strings,
)
from ..training.metrics import MetricCollection
from .._deprecation import keyword_only

__all__ = [
    "FourierResponseMetrics",
]


class FourierResponseMetrics(MetricCollection):
    """Accumulate projected-curvature and diagonal-response diagnostics.

    The collection reads the details that the :class:`FourierResponseLoss`
    term named ``loss_name`` (default: this collection's ``name``) returned
    through :meth:`equicdft.Loss.evaluate`, so the expensive perturbed
    evaluations are never repeated. Curvature metrics use the supplied
    response weights. ``S`` diagnostics report ``1 / K`` separately for each
    projected direction and therefore describe diagonal projected responses,
    not an inversion of a full coupled response matrix.
    """

    @keyword_only(1, ("direction_names", "name", "subsets", "loss_name"))
    def __init__(
        self,
        direction_names: Sequence[str],
        *,
        name: str = "fourier_response",
        subsets: Sequence[str] = ("train", "valid", "test"),
        loss_name: Optional[str] = None,
    ) -> None:
        super().__init__(name, subsets)

        self.direction_names = unique_strings(
            direction_names,
            "direction_names",
        )
        self.loss_name = (
            self.name if loss_name is None else nonempty_string(loss_name, "loss_name")
        )
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
            for subset in self.subsets
        }

    def update_metrics(
        self,
        subset: str,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        details: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """Append the response-loss evaluation recorded under ``loss_name``."""

        self._require_subset(subset)
        if details is None or self.loss_name not in details:
            raise KeyError(
                "metric '{}' requires the details of loss term '{}'".format(
                    self.name,
                    self.loss_name,
                )
            )
        response = details[self.loss_name]
        required = set(self.logs[subset])
        missing = required - set(response)
        if missing:
            raise KeyError(
                "response details are missing: {}".format(sorted(missing))
            )
        n_directions = len(self.direction_names)
        reference_shape = None
        for key in required:
            value = response[key]
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
