"""The two base classes a training objective is built from.

``LossTerm`` and ``MetricCollection`` are protocol definitions: they say
what the trainer calls and what a term must return, and they depend on
nothing else in the package. They live here rather than in
:mod:`equicdft.training` because :mod:`equicdft.response` subclasses both,
while the trainer needs to know about response terms -- a cycle when the
bases sit on the training side of it.

:mod:`equicdft.training.loss` and :mod:`equicdft.training.metrics` re-export
them, so ``from equicdft import LossTerm`` and the module paths that existed
before keep working.
"""

from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

from ._validation import nonempty_string, nonnegative_scalar, unique_strings

__all__ = [
    "LossTerm",
    "MetricCollection",
]


class LossTerm(nn.Module):
    """Base class of every named, weighted scalar loss term.

    Subclasses implement :meth:`evaluate`, which returns a mapping holding
    the weighted scalar ``"loss"`` together with any diagnostics that
    a :class:`MetricCollection` may reuse. :meth:`forward` returns the
    scalar alone. :class:`equicdft.Loss` accepts only ``LossTerm``
    instances and passes the model to every term, so a term that needs the
    model documents that by raising when ``model`` is ``None``.
    """

    def __init__(self, name: str, weight: float = 1.0) -> None:
        super().__init__()

        self.name = nonempty_string(name, "loss term name")
        self.weight = nonnegative_scalar(weight, "weight")

    def evaluate(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Return ``{"loss": weighted scalar, ...diagnostics}``."""

        raise NotImplementedError

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Return this term's weighted scalar loss."""

        return self.evaluate(outputs, batch, model)["loss"]


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
