"""Private helpers for declared optimizer parameter groups."""

from numbers import Real
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch
from torch import nn

from ._argument_checks import nonempty_string, nonnegative_scalar


def trainable_parameter_mapping(
    candidates: Mapping[str, Optional[nn.Parameter]],
) -> Dict[str, nn.Parameter]:
    """Return the trainable parameters from an owner's named candidates."""

    return {
        name: parameter
        for name, parameter in candidates.items()
        if isinstance(parameter, nn.Parameter) and parameter.requires_grad
    }


def build_optimizer_parameters(
    model: nn.Module,
    parameters: Sequence[nn.Parameter],
    optimizer_args: Mapping[str, Any],
    feature_multiplier: float,
) -> Tuple[
    Union[Sequence[nn.Parameter], List[Dict[str, Any]]],
    Tuple[str, ...],
]:
    """Build legacy parameters or base/feature groups declared by the model."""

    if feature_multiplier == 1.0:
        return parameters, ()

    declared = getattr(model, "feature_parameters", None)
    if declared is None:
        raise ValueError(
            "feature_learning_rate_multiplier requires the model to publish "
            "feature_parameters"
        )
    if not isinstance(declared, Mapping):
        raise TypeError("model.feature_parameters must be a mapping")

    registered = {id(parameter) for parameter in parameters}
    feature_parameters = []
    feature_names = []
    feature_ids = set()
    for name, parameter in declared.items():
        nonempty_string(name, "feature parameter name")
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(
                "model.feature_parameters values must be parameters"
            )
        if not parameter.requires_grad:
            raise ValueError(
                "model.feature_parameters must contain only trainable "
                "parameters"
            )
        identifier = id(parameter)
        if identifier not in registered:
            raise ValueError(
                "model.feature_parameters contains an unregistered parameter"
            )
        if identifier in feature_ids:
            raise ValueError(
                "model.feature_parameters contains duplicate parameters"
            )
        feature_ids.add(identifier)
        feature_names.append(name)
        feature_parameters.append(parameter)
    if not feature_parameters:
        raise ValueError(
            "feature_learning_rate_multiplier requires at least one "
            "trainable feature parameter"
        )
    if "lr" not in optimizer_args:
        raise ValueError(
            "feature_learning_rate_multiplier requires optimizer_args to "
            "contain lr"
        )

    base_learning_rate = nonnegative_scalar(
        optimizer_args["lr"],
        "optimizer learning rate",
    )
    groups = []
    base_parameters = [
        parameter
        for parameter in parameters
        if id(parameter) not in feature_ids
    ]
    if base_parameters:
        groups.append(
            {
                "params": base_parameters,
                "lr": base_learning_rate,
                "group_name": "base",
            }
        )
    groups.append(
        {
            "params": feature_parameters,
            "lr": base_learning_rate * feature_multiplier,
            "group_name": "feature",
        }
    )
    return groups, tuple(feature_names)


def scheduler_arguments_for_groups(
    scheduler_cls: Optional[Type[Any]],
    scheduler_args: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    feature_multiplier: float,
) -> Dict[str, Any]:
    """Scale a scalar plateau floor consistently across LR groups."""

    arguments = dict(scheduler_args)
    minimum = arguments.get("min_lr")
    try:
        is_plateau = issubclass(
            scheduler_cls,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        )
    except TypeError:
        is_plateau = False
    if (
        feature_multiplier == 1.0
        or not is_plateau
        or minimum is None
        or not isinstance(minimum, Real)
    ):
        return arguments

    minimum = nonnegative_scalar(minimum, "scheduler min_lr")
    arguments["min_lr"] = [
        minimum * feature_multiplier
        if group.get("group_name") == "feature"
        else minimum
        for group in optimizer.param_groups
    ]
    return arguments


def learning_rate_record(
    optimizer: torch.optim.Optimizer,
    feature_multiplier: float,
) -> Dict[str, float]:
    """Return backward-compatible base and optional feature rates."""

    rates = {
        group.get("group_name"): float(group["lr"])
        for group in optimizer.param_groups
    }
    feature_learning_rate = rates.get("feature")
    if "base" in rates:
        base_learning_rate = rates["base"]
    elif feature_learning_rate is not None:
        base_learning_rate = feature_learning_rate / feature_multiplier
    else:
        base_learning_rate = float(optimizer.param_groups[0]["lr"])

    record = {"learning_rate": base_learning_rate}
    if feature_learning_rate is not None:
        record["feature_learning_rate"] = feature_learning_rate
    return record
