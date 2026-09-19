"""Versioned model files that separate structure from fitted state.

A model file written by :func:`save_model` is one ``torch.save`` archive
holding a plain dictionary::

    {
        "format": "equicdft-model",
        "format_version": 2,
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

Format history
--------------
1
    Initial configuration-plus-state format.
2
    The containing ``GridCACEModel`` owns the density scale ``mean_density``,
    which version 1 kept on the feature module and repeated on every
    ``LDAReadout``. :func:`migrate_model_payload` upgrades a version-1 payload
    in memory and :func:`migrate_model_state_dict` a bare model state (as a
    training checkpoint holds it), so old files keep loading.

Whole-object ``torch.save(model)`` files predating this format are converted
by :mod:`equicdft.legacy`.
"""

from contextlib import contextmanager
import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import torch
from torch import nn

from ._version import __version__
from ._config import TYPE_KEY, Configurable, build
from ._trainer_io import atomic_torch_save
from .functional.base import EnergyReadout

__all__ = [
    "MODEL_FORMAT",
    "MODEL_FORMAT_VERSION",
    "dtype_name",
    "dtype_from_name",
    "default_dtype",
    "model_dtype",
    "model_payload",
    "save_model",
    "read_model_payload",
    "validate_model_payload",
    "read_model_config",
    "migrate_model_state_dict",
    "migrate_model_payload",
    "build_model",
    "load_model",
]


MODEL_FORMAT = "equicdft-model"
MODEL_FORMAT_VERSION = 2

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
    unbound = [
        type(module).__name__
        for module in model.modules()
        if isinstance(module, EnergyReadout) and not module.is_bound
    ]
    if unbound:
        raise ValueError(
            "model has unbound readouts ({}); pass n_features or construct "
            "them inside GridCACEModel before saving".format(
                ", ".join(unbound)
            )
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
    """Return the configuration stored in a model file, in the current format."""

    return migrate_model_payload(read_model_payload(path))["config"]


_READOUT_SCALE_KEY = re.compile(r"readout\.(\d+)\.mean_density")


def _one_density_scale(scales: Dict[str, Any], source: str) -> Any:
    """Return the first value in ``scales`` or reject disagreeing ones.

    The feature module's scale is listed first by every caller, because the
    version-1 constructor made each readout scale match it (within
    ``allclose``) and used it for the feature normalization.
    """

    values = list(scales.values())
    reference = values[0]
    for value in values[1:]:
        same = torch.allclose(
            torch.as_tensor(reference, dtype=torch.float64),
            torch.as_tensor(value, dtype=torch.float64),
        )
        if not same:
            raise ValueError(
                "cannot migrate {}: it records different density scales {} "
                "but the model owns one mean_density".format(
                    source,
                    ", ".join(
                        "{}={}".format(name, float(value))
                        for name, value in scales.items()
                    ),
                )
            )
    return reference


def migrate_model_state_dict(
    state_dict: Dict[str, torch.Tensor],
    expects_model_scale: bool,
) -> Dict[str, torch.Tensor]:
    """Return ``state_dict`` with the density scale where the model keeps it.

    Before model file format 2 the model had no ``mean_density`` buffer: the
    feature module carried the scale as ``a_features.mean_density`` and every
    LDA readout repeated it as ``readout.<i>.mean_density``. The readout
    entries are always removed. When the current model expects a
    ``mean_density`` buffer and the state has none, the feature module's
    scale supplies it, whether or not the model has LDA readouts; a state
    without a feature module falls back to the readouts' common scale. Every
    scale the state records must agree, so a state whose copies disagree is
    rejected rather than silently altered. Training checkpoints written
    before format version 2 go through this before they are loaded strictly.
    """

    state = dict(state_dict)
    scales = {}
    if "a_features.mean_density" in state:
        scales["a_features.mean_density"] = state["a_features.mean_density"]
    for key in list(state):
        if _READOUT_SCALE_KEY.fullmatch(key):
            scales[key] = state.pop(key)
    if "mean_density" in state:
        scales["mean_density"] = state["mean_density"]
    if not scales:
        return state
    scale = _one_density_scale(scales, "the model state")
    if expects_model_scale and "mean_density" not in state:
        state["mean_density"] = scale.detach().clone().reshape(())
    return state


def _migrate_v1_to_v2(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Give the model the density scale that version 1 kept elsewhere.

    The configuration takes the feature module's scale (the readouts' common
    scale when there is no feature module) and loses the per-readout copies;
    the state is migrated by :func:`migrate_model_state_dict`.
    """

    config = payload["config"]
    if config.get(TYPE_KEY) != "GridCACEModel":
        return payload
    scales = {}
    a_features = config.get("a_features")
    if isinstance(a_features, dict) and a_features.get("mean_density") is not None:
        scales["a_features"] = a_features["mean_density"]
    for index, item in enumerate(config.get("readout") or ()):
        if isinstance(item, dict) and item.get(TYPE_KEY) == "LDAReadout":
            value = item.pop("mean_density", None)
            if value is not None:
                scales["readout.{}".format(index)] = value
    if config.get("mean_density") is None:
        config["mean_density"] = (
            _one_density_scale(scales, "this model file") if scales else None
        )
    payload["state_dict"] = migrate_model_state_dict(
        payload["state_dict"],
        expects_model_scale=config["mean_density"] is not None,
    )
    if config["mean_density"] is not None and "mean_density" not in payload["state_dict"]:
        payload["state_dict"]["mean_density"] = torch.tensor(
            config["mean_density"],
            dtype=dtype_from_name(payload["default_dtype"]),
        )
    return payload


_MIGRATIONS = {1: _migrate_v1_to_v2}


def migrate_model_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``payload`` upgraded to ``MODEL_FORMAT_VERSION``."""

    payload = validate_model_payload(payload)
    version = payload["format_version"]
    if version == MODEL_FORMAT_VERSION:
        return payload
    upgraded = dict(payload)
    upgraded["config"] = copy.deepcopy(payload["config"])
    upgraded["state_dict"] = dict(payload["state_dict"])
    while version < MODEL_FORMAT_VERSION:
        upgraded = _MIGRATIONS[version](upgraded)
        version += 1
        upgraded["format_version"] = version
    return upgraded


def build_model(payload: Dict[str, Any]) -> nn.Module:
    """Construct a model from a validated payload and load its state.

    Payloads written by earlier format versions are migrated in memory first.
    """

    payload = migrate_model_payload(payload)
    with default_dtype(dtype_from_name(payload["default_dtype"])):
        model = build(payload["config"])
    if not isinstance(model, nn.Module):
        raise TypeError("model configuration did not build a torch module")
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
