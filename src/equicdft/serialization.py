"""Versioned model files that separate structure from fitted state.

A model file written by :func:`save_model` is one ``torch.save`` archive
holding a plain dictionary::

    {
        "format": "equicdft-model",
        "format_version": 1,
        "equicdft_version": "...",
        "torch_version": "...",
        "default_dtype": "float32",
        "config": {...},       # model.to_config(): structure, no module paths
        "state_dict": {...},   # fitted parameters and persistent buffers
    }

The archive contains only dictionaries, strings, numbers, and CPU tensors,
so it loads under PyTorch's restricted ``weights_only`` unpickler and never
references package module paths. :func:`load_model` rebuilds the module from
its configuration through the type registry and then restores the state.

Whole-object ``torch.save(model)`` files predating this format are converted
by :mod:`equicdft.legacy`.
"""

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from ._version import __version__
from ._config import Configurable, build
from ._trainer_io import atomic_torch_save


MODEL_FORMAT = "equicdft-model"
MODEL_FORMAT_VERSION = 1

_DTYPE_NAMES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

PathLike = Union[str, Path]


def dtype_name(dtype: torch.dtype) -> str:
    """Return the short name of a floating-point dtype, e.g. ``float32``."""

    for name, candidate in _DTYPE_NAMES.items():
        if candidate == dtype:
            return name
    raise ValueError("unsupported default dtype {}".format(dtype))


def dtype_from_name(name: str) -> torch.dtype:
    """Return the dtype for a short name written by :func:`dtype_name`."""

    try:
        return _DTYPE_NAMES[name]
    except KeyError:
        raise ValueError("unsupported default dtype name {!r}".format(name))


@contextmanager
def default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Temporarily set the global default floating-point dtype."""

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def model_dtype(model: nn.Module) -> torch.dtype:
    """Return the floating-point dtype of a model's fitted state.

    Constructors create every floating tensor with the default dtype active
    at construction time, so the parameters and floating buffers share one
    dtype. A model without floating state reports the current default.
    """

    dtypes = {
        tensor.dtype
        for tensor in list(model.parameters()) + list(model.buffers())
        if tensor.is_floating_point()
    }
    if not dtypes:
        return torch.get_default_dtype()
    if len(dtypes) != 1:
        raise ValueError(
            "model mixes floating dtypes {}; cast it to one dtype before "
            "saving".format(sorted(str(dtype) for dtype in dtypes))
        )
    return dtypes.pop()


def model_payload(model: nn.Module) -> Dict[str, Any]:
    """Return the dictionary that :func:`save_model` writes for ``model``.

    The configuration must be JSON-compatible and the state tensors are
    detached and moved to the CPU so the file does not depend on the device
    the model was trained on. The recorded default dtype is the dtype of the
    model's floating state, so reconstruction does not depend on the global
    default active when the file was written.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(model, Configurable):
        raise TypeError(
            "{} cannot be saved: it does not provide to_config()".format(
                type(model).__name__
            )
        )
    if any(
        isinstance(parameter, UninitializedParameter)
        for parameter in model.parameters()
    ):
        raise ValueError(
            "model has uninitialized lazy parameters; run one forward pass "
            "(or construct readouts with explicit n_features) before saving"
        )
    config = model.to_config()
    try:
        json.dumps(config)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "model configuration is not JSON-compatible: {}".format(error)
        ) from error
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    return {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "equicdft_version": str(__version__),
        # torch.__version__ is a str subclass that the restricted unpickler
        # rejects; store a plain string.
        "torch_version": str(torch.__version__),
        "default_dtype": dtype_name(model_dtype(model)),
        "config": config,
        "state_dict": state_dict,
    }


def save_model(model: nn.Module, path: PathLike) -> Path:
    """Write ``model`` as a versioned configuration-plus-state file."""

    destination = Path(path).expanduser()
    atomic_torch_save(model_payload(model), destination)
    return destination


def _torch_load_plain(path: Path, map_location: Any) -> Any:
    """Load a file with the restricted unpickler where PyTorch offers it."""

    try:
        return torch.load(
            str(path),
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        # PyTorch releases before the weights_only keyword.
        return torch.load(str(path), map_location=map_location)


def read_model_payload(path: PathLike, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load and validate a model file without constructing the model."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError("model file does not exist: {}".format(source))
    try:
        payload = _torch_load_plain(source, map_location)
    except Exception as error:  # noqa: BLE001 - re-raised with guidance
        raise ValueError(
            "{} is not an {} file. A whole-object torch.save(model) file "
            "can be converted with `python -m equicdft.convert`.".format(
                source,
                MODEL_FORMAT,
            )
        ) from error
    return validate_model_payload(payload, source)


def validate_model_payload(payload: Any, source: Any = "<payload>") -> Dict[str, Any]:
    """Check the envelope fields of a loaded payload and return it."""

    if not isinstance(payload, dict) or payload.get("format") != MODEL_FORMAT:
        raise ValueError(
            "{} is not an {} file. A whole-object torch.save(model) file "
            "can be converted with `python -m equicdft.convert`.".format(
                source,
                MODEL_FORMAT,
            )
        )
    version = payload.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("{} has no integer format_version".format(source))
    if version > MODEL_FORMAT_VERSION:
        raise ValueError(
            "{} uses {} format version {}, but this equicdft supports "
            "versions up to {}".format(
                source,
                MODEL_FORMAT,
                version,
                MODEL_FORMAT_VERSION,
            )
        )
    for key in ("config", "state_dict", "default_dtype"):
        if key not in payload:
            raise ValueError("{} is missing '{}'".format(source, key))
    if not isinstance(payload["config"], dict):
        raise ValueError("{} has a malformed config".format(source))
    if not isinstance(payload["state_dict"], dict):
        raise ValueError("{} has a malformed state_dict".format(source))
    return payload


def read_model_config(path: PathLike) -> Dict[str, Any]:
    """Return the configuration stored in a model file."""

    return read_model_payload(path)["config"]


def build_model(payload: Dict[str, Any]) -> nn.Module:
    """Construct a model from a validated payload and load its state."""

    payload = validate_model_payload(payload)
    with default_dtype(dtype_from_name(payload["default_dtype"])):
        model = build(payload["config"])
    if not isinstance(model, nn.Module):
        raise TypeError("model configuration did not build a torch module")
    # Some fixed-geometry constructors deliberately use float64. Restore the
    # saved model's uniform dtype before copying its fitted state into buffers.
    model.to(dtype=dtype_from_name(payload["default_dtype"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model


def load_model(
    path: PathLike,
    map_location: Optional[Union[str, torch.device]] = None,
    eval_mode: bool = True,
) -> nn.Module:
    """Rebuild a model saved by :func:`save_model`.

    The module is constructed under the default dtype recorded in the file,
    its state is restored strictly, and it is moved to ``map_location`` when
    one is given. The returned module is in evaluation mode unless
    ``eval_mode=False``.
    """

    payload = read_model_payload(path, map_location="cpu")
    model = build_model(payload)
    if map_location is not None:
        model = model.to(torch.device(map_location))
    return model.eval() if eval_mode else model.train()
