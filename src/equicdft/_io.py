"""Atomic file writes shared by model files and training checkpoints."""

import os
from pathlib import Path

import torch


def atomic_torch_save(value: object, path: Path) -> None:
    """Write a Torch object atomically through a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(".{}.tmp".format(path.name))
    try:
        torch.save(value, str(temporary_path))
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
