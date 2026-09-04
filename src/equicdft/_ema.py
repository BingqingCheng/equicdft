"""Small exponential-moving-average helper for model parameters."""

from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping

import torch
from torch import nn


def _trainable_parameters(model: nn.Module) -> "OrderedDict[str, nn.Parameter]":
    return OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


class _ParameterEMA:
    """Track and temporarily apply an EMA of trainable model parameters."""

    def __init__(self, decay: float) -> None:
        self.decay = decay
        self.num_updates = 0
        self.shadow_parameters = OrderedDict()

    @property
    def initialized(self) -> bool:
        return bool(self.shadow_parameters)

    def update(self, model: nn.Module) -> None:
        """Update shadows, copying the model exactly on the first update."""

        parameters = _trainable_parameters(model)
        if not parameters:
            raise ValueError("EMA requires at least one trainable model parameter")
        if not self.initialized:
            self.shadow_parameters = OrderedDict(
                (name, parameter.detach().clone())
                for name, parameter in parameters.items()
            )
        else:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    try:
                        shadow = self.shadow_parameters[name]
                    except KeyError as error:
                        raise RuntimeError(
                            "trainable model parameters changed after EMA setup"
                        ) from error
                    if shadow.shape != parameter.shape:
                        raise RuntimeError(
                            "trainable model parameter shape changed during EMA"
                        )
                    shadow.lerp_(
                        parameter.detach(),
                        1.0 - self.decay,
                    )
        self.num_updates += 1

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily replace trainable parameters by their EMA values."""

        if not self.initialized:
            yield
            return
        parameters = _trainable_parameters(model)
        original = OrderedDict(
            (name, parameter.detach().clone())
            for name, parameter in parameters.items()
        )
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(self.shadow_parameters[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(original[name])

    def evaluation_state_dict(
        self,
        model: nn.Module,
    ) -> "OrderedDict[str, torch.Tensor]":
        """Return a detached model state using EMA parameters when available."""

        with self.average_parameters(model):
            return OrderedDict(
                (name, value.detach().clone())
                for name, value in model.state_dict().items()
            )

    def state_dict(self) -> Dict[str, Any]:
        """Return scalar restart metadata; EMA weights live in the model state."""

        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        evaluation_model_state: Mapping[str, torch.Tensor],
        model: nn.Module,
    ) -> None:
        """Restore shadows from the evaluated model stored in a checkpoint."""

        if not isinstance(state, Mapping):
            raise TypeError("EMA state must be a mapping")
        if state.get("decay") != self.decay:
            raise ValueError("checkpoint EMA decay does not match the trainer")
        num_updates = state.get("num_updates")
        if (
            isinstance(num_updates, bool)
            or not isinstance(num_updates, int)
            or num_updates < 0
        ):
            raise ValueError("checkpoint EMA update count must be nonnegative")
        parameters = _trainable_parameters(model)
        if num_updates == 0:
            self.shadow_parameters = OrderedDict()
        else:
            shadows = OrderedDict()
            for name, parameter in parameters.items():
                value = evaluation_model_state.get(name)
                if not torch.is_tensor(value):
                    raise ValueError(
                        "EMA model state is missing trainable parameter '{}'".format(
                            name
                        )
                    )
                if value.shape != parameter.shape:
                    raise ValueError(
                        "checkpoint EMA parameter shapes do not match model"
                    )
                shadows[name] = value.detach().to(parameter).clone()
            self.shadow_parameters = shadows
        self.num_updates = num_updates
