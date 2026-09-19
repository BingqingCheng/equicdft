"""Current training-checkpoint validation and state restoration.

Historical training checkpoints are not supported.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


__all__ = [
    "REQUIRED_KEYS",
    "CHECKPOINT_VERSION",
    "validate_checkpoint",
    "read_checkpoint",
    "restore_history",
    "restore_random_state",
]


CHECKPOINT_VERSION = 4

REQUIRED_KEYS = frozenset((
    "noise", "stream_noise", "best_valid_loss", "epochs_without_improvement",
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
    return validate_checkpoint(checkpoint)


def validate_checkpoint(checkpoint):
    """Require a current checkpoint; historical resume is unsupported."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    version = checkpoint.get("checkpoint_version")
    if type(version) is not int or version != CHECKPOINT_VERSION:
        raise ValueError(
            "old or unsupported checkpoint version; "
            "legacy training-checkpoint resume is not supported"
        )
    missing = REQUIRED_KEYS - set(checkpoint)
    if missing:
        raise ValueError("checkpoint is missing keys: {}".format(sorted(missing)))
    return checkpoint


def restore_history(checkpoint: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]], float, int]:
    """Return the recorded epoch, history, best loss and stall count."""

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

    return (
        epoch, history, float(checkpoint["best_valid_loss"]),
        int(checkpoint["epochs_without_improvement"]),
    )


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
