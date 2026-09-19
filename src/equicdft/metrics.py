"""Deprecated location; use :mod:`equicdft.training.metrics`."""

from ._deprecation import warn_moved_module
from .training.metrics import (
    SUPPORTED_METRICS,
    METRIC_LABELS,
    metric_label,
    format_metric_value,
    compute_metric,
    MetricCollection,
    Metrics,
)
from .response.metrics import FourierResponseMetrics

__all__ = [
    "FourierResponseMetrics",
    "SUPPORTED_METRICS",
    "METRIC_LABELS",
    "metric_label",
    "format_metric_value",
    "compute_metric",
    "MetricCollection",
    "Metrics",
]

warn_moved_module(__name__, "equicdft.training.metrics")
