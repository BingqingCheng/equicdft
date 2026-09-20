"""Convert trusted whole-object or versioned models to the current format."""

import copy
import pickle
from pathlib import Path

import torch
from torch import nn

from ..serialization import build_model, model_payload, save_model
from .formats import build_converted_model
from .legacy import _remapping_pickle_module, legacy_model_to_current, verify_equivalent


def _conversion_destination(source, destination, overwrite):
    """Resolve paths before loading; conversion never replaces its input."""
    source = Path(source).expanduser()
    destination = Path(destination).expanduser()
    if not source.is_file():
        raise FileNotFoundError("source file does not exist: {}".format(source))
    if source.resolve() == destination.resolve() or (
        destination.exists() and source.samefile(destination)
    ):
        raise ValueError("conversion destination must differ from the source")
    if destination.exists() and not overwrite:
        raise FileExistsError("{} exists; pass overwrite=True to replace it".format(destination))
    return source, destination


def load_converted_model(source, *, ideal=None, verify=True, verification_data=None):
    """Explicitly convert a trusted old file in memory, without modifying it.

    All inputs must be trusted, including files believed to be versioned.
    Restricted loading is tried first; an unpickling rejection falls back
    to the historical object loader, which can execute code from the file.
    That fallback is not a file-type or safety check. Versioned files transfer
    their state strictly; file conversion can also verify rebuilt predictions.
    """
    source = Path(source).expanduser()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Trusted historical objects need the remapping unpickler. Do not
        # retry unrelated I/O, runtime or resource failures unrestricted.
        payload = torch.load(
            source, map_location="cpu",
            pickle_module=_remapping_pickle_module(),
            weights_only=False,
        )
    if isinstance(payload, nn.Module):
        model = legacy_model_to_current(
            payload, ideal=ideal,
            verify=verify, verification_data=verification_data,
        )
    else:
        model = build_converted_model(payload, ideal=ideal)
    return model.eval()


def convert_model(source, destination, *, ideal=None, verify=True,
                  verification_data=None, overwrite=False):
    """Convert a trusted file to a separate current-format file; keep the source."""
    source, destination = _conversion_destination(source, destination, overwrite)
    model = load_converted_model(
        source, ideal=ideal, verify=verify, verification_data=verification_data,
    )
    # Exercise current serialization before publishing the converted file.
    restored = build_model(model_payload(model)).eval()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    if verify and verification_data is not None:
        verify_equivalent(model, restored, copy.deepcopy(verification_data))
    return save_model(model, destination)
