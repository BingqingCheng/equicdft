"""Moving batches and modules between devices, shared by trainer and solver."""

from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from torch import nn

__all__ = ["GeometryDeviceCache", "module_device", "move_batch"]


def module_device(module: nn.Module) -> torch.device:
    """Return the device of a module's first parameter or buffer (CPU if none)."""

    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device


class GeometryDeviceCache:
    """Device copies of grid-geometry tensors shared by many batches.

    Loaders built with ``reuse_grid_geometry=True`` attach the shared source
    tensors of a batch as ``batch._shared_geometry``; moving such a batch
    reuses one device copy per source tensor instead of transferring the
    expanded geometry every step. The cache keeps at most ``capacity``
    sources and drops a copy whose source tensor was modified in place.
    """

    def __init__(self, capacity: int = 2) -> None:
        self.capacity = int(capacity)
        self._entries: "OrderedDict[int, Any]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def fetch(self, source: torch.Tensor, device: torch.device) -> torch.Tensor:
        identity = id(source)
        cached = self._entries.pop(identity, None)
        if (
            cached is None
            or cached[0] is not source
            or cached[1] != source._version
        ):
            while len(self._entries) >= self.capacity:
                self._entries.popitem(last=False)
            cached = (source, source._version, source.to(device))
        self._entries[identity] = cached
        return cached[2]


def move_batch(
    batch: Dict[str, Any],
    device: Optional[torch.device],
    geometry_cache: Optional[GeometryDeviceCache] = None,
) -> Dict[str, Any]:
    """Return a shallow copy of ``batch`` with its tensors on ``device``.

    Non-tensor entries pass through. With a ``geometry_cache``, entries the
    loader marked as identity-shared grid geometry are expanded from one
    cached device copy. ``device=None`` copies the dictionary only.
    """

    if device is None:
        return dict(batch)
    shared_geometry = (
        getattr(batch, "_shared_geometry", {}) if geometry_cache is not None else {}
    )
    moved = {}
    for key, value in batch.items():
        source = shared_geometry.get(key)
        if source is None:
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        else:
            moved[key] = geometry_cache.fetch(source, device).unsqueeze(0).expand(value.shape)
    return moved
