"""Orthogonal projection onto densities homogeneous along selected grid axes."""

from numbers import Integral

import torch


def _normalize_homogeneous_axes(axes):
    """Translate axis names to indices; grid-dependent validation follows."""
    if axes is None:
        return ()
    if isinstance(axes, str):
        axes = (axes,)
    try:
        axes = tuple(axes)
    except TypeError as exc:
        raise ValueError("homogeneous_axes must be an axis name or a sequence of axes") from exc
    names = {"x": 0, "y": 1, "z": 2}
    indices = []
    for axis in axes:
        if isinstance(axis, str):
            if axis not in names:
                raise ValueError("homogeneous_axes names must be 'x', 'y' or 'z'")
            axis = names[axis]
        indices.append(axis)
    return tuple(indices)


class _HomogeneousDensityProjection:
    def __init__(self, data, axes, accessible):
        self.axes = tuple(axes)
        shape = torch.as_tensor(data["grid_size"], device=accessible.device)
        if shape.ndim != 1 or not torch.all(shape > 0) or not torch.equal(shape, shape.long()):
            raise ValueError("homogeneous_axes requires positive integer grid_size")
        ndim = len(shape)
        if (any(isinstance(a, bool) or not isinstance(a, Integral) for a in self.axes)
                or len(set(self.axes)) != len(self.axes)
                or any(a < 0 or a >= ndim for a in self.axes)):
            raise ValueError("homogeneous_axes must contain distinct valid integer grid axes")
        positions = data.get("grid_positions")
        if positions is None:
            raise ValueError("homogeneous_axes requires integer grid_positions")
        positions = torch.as_tensor(positions, device=accessible.device)
        if positions.shape != (len(accessible), ndim) or not torch.equal(positions, positions.long()):
            raise ValueError("homogeneous_axes requires integer grid_positions of shape [n_grid, ndim]")
        positions = positions.long()
        shape = shape.long()
        if (int(shape.prod()) != len(accessible) or torch.any(positions < 0)
                or torch.any(positions >= shape)):
            raise ValueError("homogeneous_axes requires a complete rectangular grid")
        flat = torch.zeros(len(accessible), device=accessible.device, dtype=torch.long)
        self.shape = tuple(shape.tolist())
        for axis, size in enumerate(self.shape):
            flat = flat * size + positions[:, axis]
        if torch.unique(flat).numel() != len(accessible):
            raise ValueError("homogeneous_axes requires unique grid_positions")
        # flat maps input rows to C-order grid indices; argsort reverses it.
        self.grid_order = torch.argsort(flat)
        self.input_order = flat
        mask = accessible[self.grid_order].reshape(self.shape)
        for axis in self.axes:
            first_slice = mask.select(axis, 0).unsqueeze(axis)
            if not torch.equal(mask, first_slice.expand_as(mask)):
                raise ValueError("accessibility must be homogeneous along homogeneous_axes")

    def average(self, values):
        """Average and broadcast without changing row order or channel count."""
        if not self.axes:
            return values
        grid = values[self.grid_order].reshape(*self.shape, values.shape[-1])
        averaged = grid.mean(dim=self.axes, keepdim=True).expand_as(grid)
        return averaged.reshape_as(values)[self.input_order]
