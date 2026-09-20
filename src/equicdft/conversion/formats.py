"""Explicit migrations of historical versioned model files.

Versions 1/2 predate model-owned ideal references; polarized files require
the physical molecular moment from the original run, never descriptor scales.
Saved fitting policy is preserved; absent settings use current defaults.
Migration leaves inputs intact.
"""

import copy
import re
from typing import Any, Dict

import torch

from .._config import TYPE_KEY
from ..serialization import (
    MODEL_FORMAT_VERSION, build_model, dtype_from_name, validate_model_payload,
)

__all__ = ["migrate_model_payload", "migrate_model_state_dict", "build_converted_model"]


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
    rejected rather than silently altered.
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


def _migrate_v2_to_v3(payload):
    """Move the ideal reference into the model; never guess molecular moments."""
    config = payload["config"]
    if config.get("ideal") is not None:
        return payload
    wavelength = config.pop("thermal_wavelength", 1.0)

    def polarized(value):
        if isinstance(value, dict):
            return (value.get("include_polarization", False)
                    or value.get("variable") == "dipole_density"
                    or (value.get(TYPE_KEY) == "LDAReadout"
                        and value.get("dipole_density_scale") is not None)
                    or any(polarized(v) for v in value.values()))
        return isinstance(value, list) and any(polarized(v) for v in value)

    if polarized(config):
        config["ideal"] = None
        # This field is migration metadata, consumed by build_converted_model rather
        # than passed to the constructor. An explicit ideal is required.
        payload["missing_ideal_wavelength"] = wavelength
        payload["state_dict"].pop("thermal_wavelength", None)
        payload["state_dict"].pop("ideal.thermal_wavelength", None)
    else:
        config["ideal"] = {"type": "IdealGas", "thermal_wavelength": wavelength}
        state = payload["state_dict"]
        value = state.pop("thermal_wavelength", state.get("ideal.thermal_wavelength"))
        if value is None:
            value = torch.as_tensor(wavelength, dtype=dtype_from_name(payload["default_dtype"]))
        state["ideal.thermal_wavelength"] = value
    return payload


def _migrate_v3_to_v4(payload):
    """Carry the fitting policy forward; fitted parameters are unchanged.

    ``compute_local_mu`` and ``rho_min`` were briefly absent from the model
    and are now optional diagnostics again, computing the same quantities
    from the same mask, so an old file's settings transfer rather than
    reset: a converted functional evaluates the chemical potential its fit
    used. A file written while they were absent has neither key and takes
    the current defaults.
    """
    config = payload["config"]
    if config.get(TYPE_KEY) == "GridCACEModel":
        config.setdefault("compute_local_mu", False)
        config.setdefault("rho_min", 0.0)
    return payload


def _migrate_v4_to_v5(payload):
    """Move polarization normalization to the model without changing its energy.

    The feature scale wins when present. Historical LDA scales can differ;
    absorb that difference into the first linear layer's squared-P columns.
    Equal scales require no arithmetic on fitted weights.
    """
    config, state = payload["config"], payload["state_dict"]
    if config.get(TYPE_KEY) != "GridCACEModel":
        return payload
    scale = state.get("polarization_scale", config.get("polarization_scale"))
    features = config.get("a_features")
    if isinstance(features, dict):
        old = features.pop("dipole_density_scale", None)
        old = state.pop("a_features.dipole_density_scale", old)
        if old is not None:
            if scale is not None and not torch.equal(torch.as_tensor(scale).to(torch.as_tensor(old)), torch.as_tensor(old)):
                raise ValueError("model and historical feature polarization scales disagree")
            scale = old
    lda_scales = []
    for index, readout in enumerate(config.get("readout") or ()):
        if not isinstance(readout, dict) or readout.get(TYPE_KEY) != "LDAReadout":
            continue
        if "dipole_density_scale" in readout:
            old = readout.pop("dipole_density_scale")
            old = state.pop("readout.{}.dipole_density_scale".format(index), old)
            readout["include_polarization"] = old is not None
            if old is not None:
                lda_scales.append((index, int(readout.get("n_types", 1)), old))
                if scale is None:
                    scale = old
    if scale is not None:
        scale = torch.as_tensor(scale, dtype=dtype_from_name(payload["default_dtype"])).detach().clone().reshape(())
        config["polarization_scale"] = scale.item()
        state["polarization_scale"] = scale
        for index, n_types, old in lda_scales:
            old = torch.as_tensor(old).to(scale)
            if not torch.equal(scale, old):
                key = "readout.{}.mlp.0.weight".format(index)
                weight = state[key].clone()
                weight[:, n_types:2*n_types] *= (scale / old).square().to(weight)
                state[key] = weight
    else:
        config["polarization_scale"] = None
    return payload


_MIGRATIONS = {1: _migrate_v1_to_v2, 2: _migrate_v2_to_v3,
               3: _migrate_v3_to_v4, 4: _migrate_v4_to_v5}


def migrate_model_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``payload`` upgraded to ``MODEL_FORMAT_VERSION``."""

    payload = validate_model_payload(payload, require_current=False)
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



def build_converted_model(payload, *, ideal=None):
    """Migrate a versioned payload and construct a current model explicitly."""
    payload = copy.deepcopy(migrate_model_payload(payload))
    if "missing_ideal_wavelength" in payload and ideal is None:
        raise ValueError(
            "old polarization model has no molecular dipole magnitude; "
            "pass ideal=FixedDipoleIdeal(...) using the original run metadata"
        )
    if ideal is not None:
        if payload["config"].get("ideal") is not None:
            raise ValueError("this model already records an ideal reference")
        payload["config"]["ideal"] = ideal.to_config()
        payload["state_dict"].update({
            "ideal." + name: value for name, value in ideal.state_dict().items()
        })
    payload.pop("missing_ideal_wavelength", None)
    return build_model(payload)
