"""Constructor-argument configurations for serializable model modules.

A configurable module describes its *structure* as a plain, JSON-compatible
dictionary of constructor keyword arguments plus a ``"type"`` tag, and can be
rebuilt from that dictionary. Learned values (parameters and persistent
buffers) are not part of the configuration; they travel separately in the
ordinary ``state_dict``. Reconstruction therefore always runs the regular
constructor, after which ``load_state_dict`` restores the fitted state.

The registry maps type tags to classes so that a saved configuration never
records Python module paths. Moving or renaming a module inside the package
does not invalidate saved models as long as the type tag stays registered.
"""

from typing import Any, Dict, Mapping, Optional, Type, TypeVar

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter


_REGISTRY: Dict[str, Type["Configurable"]] = {}

TYPE_KEY = "type"

ConfigurableType = TypeVar("ConfigurableType", bound="Configurable")


class Configurable:
    """Mixin for modules that can be rebuilt from constructor arguments.

    Subclasses implement :meth:`to_config`. The default :meth:`from_config`
    removes the ``"type"`` tag and passes the remaining entries to the class
    constructor; classes whose constructor receives other configurable
    modules override it and build those children with :func:`build`.
    """

    def to_config(self) -> Dict[str, Any]:
        """Return the JSON-compatible constructor arguments of this module."""

        raise NotImplementedError

    @classmethod
    def from_config(
        cls: Type[ConfigurableType],
        config: Mapping[str, Any],
    ) -> ConfigurableType:
        """Construct an instance from a configuration dictionary."""

        return cls(**constructor_arguments(cls, config))


def register(cls: Type[ConfigurableType]) -> Type[ConfigurableType]:
    """Register a configurable class under its class name."""

    if not isinstance(cls, type) or not issubclass(cls, Configurable):
        raise TypeError("only Configurable classes can be registered")
    name = cls.__name__
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            "configurable type '{}' is already registered".format(name)
        )
    _REGISTRY[name] = cls
    return cls


def registered_types() -> Dict[str, Type[Configurable]]:
    """Return a copy of the type-tag registry."""

    return dict(_REGISTRY)


def resolve(type_name: str) -> Type[Configurable]:
    """Return the registered class for one type tag."""

    if not isinstance(type_name, str):
        raise TypeError("configuration type tags must be strings")
    try:
        return _REGISTRY[type_name]
    except KeyError:
        raise KeyError(
            "unknown configurable type '{}'; registered types are {}".format(
                type_name,
                sorted(_REGISTRY),
            )
        ) from None


def build(config: Mapping[str, Any]) -> Configurable:
    """Construct the module described by a configuration dictionary."""

    if not isinstance(config, Mapping):
        raise TypeError("a module configuration must be a mapping")
    if TYPE_KEY not in config:
        raise ValueError(
            "a module configuration requires a '{}' entry".format(TYPE_KEY)
        )
    return resolve(config[TYPE_KEY]).from_config(config)


def build_optional(
    config: Optional[Mapping[str, Any]],
) -> Optional[Configurable]:
    """Construct a module, passing ``None`` through unchanged."""

    return None if config is None else build(config)


def constructor_arguments(
    cls: type,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return ``config`` without its type tag after checking the tag."""

    if not isinstance(config, Mapping):
        raise TypeError("a module configuration must be a mapping")
    arguments = dict(config)
    type_name = arguments.pop(TYPE_KEY, None)
    if type_name is not None and type_name != cls.__name__:
        raise ValueError(
            "configuration type '{}' does not match {}".format(
                type_name,
                cls.__name__,
            )
        )
    return arguments


def make_config(module: Configurable, **fields: Any) -> Dict[str, Any]:
    """Assemble a configuration with the type tag first.

    Tensors become nested lists or Python scalars, tuples become lists, and
    nested configurable modules are left to the caller. The result is
    checked to contain only JSON-compatible values.
    """

    config: Dict[str, Any] = {TYPE_KEY: type(module).__name__}
    for name, value in fields.items():
        config[name] = plain_value(value)
    return config


def plain_value(value: Any) -> Any:
    """Convert tensors and tuples into JSON-compatible Python values."""

    if torch.is_tensor(value):
        detached = value.detach().cpu()
        if detached.ndim == 0:
            return detached.item()
        return detached.tolist()
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): plain_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "configuration values must be JSON-compatible, got {}".format(
            type(value).__name__
        )
    )


def optional_config(module: Optional[nn.Module]) -> Optional[Dict[str, Any]]:
    """Return ``module.to_config()`` or ``None`` for an absent module."""

    if module is None:
        return None
    if not isinstance(module, Configurable):
        raise TypeError(
            "{} does not provide a configuration".format(
                type(module).__name__
            )
        )
    return module.to_config()


def mlp_hidden_sizes(mlp: nn.Sequential) -> list:
    """Return the hidden widths of an MLP built by ``build_mlp``."""

    widths = [
        layer.out_features
        for layer in mlp
        if isinstance(layer, (nn.Linear, nn.LazyLinear))
    ]
    return [int(width) for width in widths[:-1]]


def mlp_input_size(mlp: nn.Sequential) -> Optional[int]:
    """Return the input width of an MLP, or ``None`` while it is lazy."""

    first_layer = mlp[0]
    if isinstance(first_layer, nn.LazyLinear) or isinstance(
        first_layer.weight,
        UninitializedParameter,
    ):
        return None
    return int(first_layer.in_features)


def optional_buffer_value(value: Optional[torch.Tensor]) -> Any:
    """Return a plain value for an optional registered buffer."""

    return None if value is None else plain_value(value)


def scalar_value(value: torch.Tensor) -> float:
    """Return one Python float from a scalar tensor."""

    return float(value.detach().cpu().reshape(()).item())
