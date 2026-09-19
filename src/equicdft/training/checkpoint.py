"""Reading a training checkpoint: format, migration and restore helpers.

:mod:`equicdft.training.trainer` owns optimization and epoch orchestration;
what a checkpoint file contains, which of its entries older versions wrote
under other names, and how its saved state is put back belong here, next to
:mod:`equicdft.training.io`, which owns the history and log files.

Helpers take explicit checkpoint, model, loss or random-generator inputs
rather than a trainer, keeping the restore phases explicit at the call site
in ``Trainer.load_checkpoint``.
"""

import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from ..objectives import LossTerm
from .noise import NoiseOptions

__all__ = [
    "REQUIRED_KEYS",
    "checkpoint_noise",
    "checkpoint_stream_noise",
    "load_loss_state",
    "read_checkpoint",
    "restore_history",
    "restore_random_state",
]


REQUIRED_KEYS = frozenset((
    "epoch",
    "model_state_dict",
    "loss_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "history",
))


def read_checkpoint(path, map_location) -> Dict[str, Any]:
    """Load a checkpoint file and check that it holds the required entries."""

    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "checkpoint does not exist: {}".format(checkpoint_path)
        )
    checkpoint = torch.load(
        str(checkpoint_path), map_location=map_location, weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    missing_keys = REQUIRED_KEYS - set(checkpoint)
    if missing_keys:
        raise ValueError(
            "checkpoint is missing keys: {}".format(sorted(missing_keys))
        )
    return checkpoint


def checkpoint_noise(checkpoint: Dict[str, Any]) -> NoiseOptions:
    """Read the trainer noise of a version-3 or earlier checkpoint."""

    if "noise" in checkpoint:
        return NoiseOptions.from_dict(checkpoint["noise"])
    return NoiseOptions.from_legacy(
        density_noise=checkpoint.get("density_noise", False),
        density_noise_floor=checkpoint.get("density_noise_floor", 0.0),
        polarization_noise=checkpoint.get("polarization_noise", False),
        noise_dipole_magnitude=checkpoint.get("noise_dipole_magnitude"),
    )


def checkpoint_stream_noise(checkpoint: Dict[str, Any]) -> Dict[str, NoiseOptions]:
    """Read the per-stream noise of a version-3 or earlier checkpoint."""

    if "stream_noise" in checkpoint:
        return {
            name: NoiseOptions.from_dict(values)
            for name, values in checkpoint["stream_noise"].items()
        }
    density = checkpoint.get("stream_density_noise", {})
    polarization = checkpoint.get("stream_polarization_noise", {})
    return {
        name: NoiseOptions.from_legacy(
            density_noise=density.get(name, {}).get("enabled", False),
            density_noise_floor=density.get(name, {}).get("floor", 0.0),
            polarization_noise=polarization.get(name, {}).get("enabled", False),
            noise_dipole_magnitude=polarization.get(name, {}).get("dipole_magnitude"),
        )
        for name in set(density) | set(polarization)
    }


def load_loss_state(loss, state, model) -> None:
    """Load loss state, letting each term retire its own old buffers first.

    A term that stopped registering a buffer knows what the saved value
    meant and where the quantity lives now, so it drops the entry through
    :meth:`equicdft.objectives.LossTerm.load_legacy_state` rather than
    having the trainer special-case it. Without that, a strict load would
    reject the whole file.
    """

    state = copy.copy(state)
    for name, term in loss.named_modules():
        if isinstance(term, LossTerm):
            state = term.load_legacy_state(
                state, name + "." if name else "", model,
            )
    loss.load_state_dict(state, strict=True)


def restore_history(checkpoint: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]], float, int]:
    """Return the completed epoch, history, best loss and stall count.

    A checkpoint written before the counters were saved carries neither, so
    both are recomputed from the history it does carry.
    """

    epoch = checkpoint["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("checkpoint epoch must be a positive integer")
    history = checkpoint["history"]
    if not isinstance(history, list):
        raise ValueError("checkpoint history must be a list")
    history_epochs = [record.get("epoch") for record in history]
    expected_epochs = list(range(1, epoch + 1))
    if history_epochs not in (expected_epochs, [0] + expected_epochs):
        raise ValueError(
            "checkpoint history must contain epochs 1 through its "
            "completed epoch, with at most one leading epoch 0 baseline"
        )

    if "best_valid_loss" in checkpoint:
        best_valid_loss = float(checkpoint["best_valid_loss"])
    else:
        best_valid_loss = min(
            record["valid_losses"]["total"]
            for record in history
            if record["epoch"] > 0
        )
    if "epochs_without_improvement" in checkpoint:
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
    else:
        best_loss = math.inf
        epochs_without_improvement = 0
        for record in history:
            if record["epoch"] == 0:
                continue
            valid_loss = record["valid_losses"]["total"]
            if valid_loss < best_loss:
                best_loss = valid_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
    return epoch, history, best_valid_loss, epochs_without_improvement


def restore_random_state(
    checkpoint: Dict[str, Any],
    train_loader_generator,
    stream_loader_generators: Dict[str, Any],
) -> None:
    """Put back the global, CUDA and loader random states.

    This runs last in a restore, so draws made while building the
    replacement process do not survive into the resumed run.
    """

    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_states = checkpoint.get("cuda_rng_state_all")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])

    loader_state = checkpoint.get("train_loader_generator_state")
    if loader_state is not None:
        if train_loader_generator is None:
            raise ValueError(
                "checkpoint contains a train-loader random state, but the "
                "current loader has no generator"
            )
        train_loader_generator.set_state(loader_state.cpu())

    for name, state in checkpoint.get("stream_loader_generator_states", {}).items():
        if name not in stream_loader_generators:
            raise ValueError(
                "checkpoint contains an unknown stream loader '{}'".format(name)
            )
        generator = stream_loader_generators[name]
        if state is not None:
            if generator is None:
                raise ValueError(
                    "checkpoint contains a loader state for stream '{}', "
                    "but its current loader has no generator".format(name)
                )
            generator.set_state(state.cpu())
