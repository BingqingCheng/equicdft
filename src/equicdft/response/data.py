"""Dataset of Fourier-response curvature targets on one grid geometry."""

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
)
import numpy as np
import torch
from torch.utils.data import Dataset
from .._validation import nonempty_string
from .modes import canonical_wavevector_indices, integer_wavevector_indices

__all__ = [
    "FourierResponseData",
]


class FourierResponseData(Dataset):
    """Homogeneous grid fields paired with projected Fourier curvatures.

    ``template`` supplies one periodic grid geometry. ``density`` contains
    the component densities for each response item. ``wavevector_indices``
    and ``curvature`` contain its integer reciprocal-lattice indices and
    projected response targets. The dataset is deliberately agnostic to
    component names, directions, thermodynamic units, and how response targets
    were obtained.

    A single wavevector per item may be supplied as ``[n_items, 3]``;
    otherwise indices have shape ``[n_items, n_wavevectors, 3]``. Curvature,
    scale, and weight have shape
    ``[n_items, n_wavevectors, n_directions]``. Optional ``indices`` selects
    an existing, externally defined split without regenerating it.
    """

    def __init__(
        self,
        template: Mapping[str, Any],
        density: Any,
        wavevector_indices: Any,
        curvature: Any,
        scale: Optional[Any] = None,
        weight: Optional[Any] = None,
        indices: Optional[Sequence[int]] = None,
        wavevector_indices_key: str = "fourier_wavevector_indices",
        target_key: str = "fourier_curvature",
        scale_key: str = "fourier_scale",
        weights_key: str = "fourier_weight",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if not isinstance(template, Mapping):
            raise TypeError("template must be a mapping")
        self.template = dict(template)
        required_template_keys = {
            "index",
            "n_types",
            "grid_size",
            "grid_spacing",
        }
        missing_template_keys = required_template_keys - set(self.template)
        if missing_template_keys:
            raise ValueError(
                "template is missing keys: {}".format(
                    sorted(missing_template_keys)
                )
            )
        self.wavevector_indices_key = nonempty_string(
            wavevector_indices_key,
            "wavevector_indices_key",
        )
        self.target_key = nonempty_string(target_key, "target_key")
        self.scale_key = nonempty_string(scale_key, "scale_key")
        self.weights_key = nonempty_string(weights_key, "weights_key")

        if dtype is None:
            dtype = torch.get_default_dtype()
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch dtype")
        self.density = torch.as_tensor(density, dtype=dtype)
        self.wavevector_indices = integer_wavevector_indices(wavevector_indices)
        self.curvature = torch.as_tensor(curvature, dtype=dtype)
        if self.wavevector_indices.ndim == 2:
            self.wavevector_indices = self.wavevector_indices.unsqueeze(1)
        if self.curvature.ndim == 2:
            self.curvature = self.curvature.unsqueeze(1)

        if self.density.ndim != 2:
            raise ValueError("density must have shape [n_items, n_types]")
        n_items, n_types = self.density.shape
        if int(torch.as_tensor(self.template["n_types"]).item()) != n_types:
            raise ValueError("density n_types does not match template")
        if (
            self.wavevector_indices.shape[:1] != (n_items,)
            or self.wavevector_indices.ndim != 3
        ):
            raise ValueError(
                "wavevector_indices must have shape "
                "[n_items, n_wavevectors, 3]"
            )
        if self.wavevector_indices.shape[-1] != 3:
            raise ValueError(
                "wavevector_indices must contain three integer components"
            )
        self.wavevector_indices = torch.stack(
            [
                canonical_wavevector_indices(
                    item_indices,
                    self.template["grid_size"],
                    self.template["grid_spacing"],
                )
                for item_indices in self.wavevector_indices
            ]
        )
        expected = (n_items, self.wavevector_indices.shape[1])
        if self.curvature.ndim != 3 or self.curvature.shape[:2] != expected:
            raise ValueError(
                "curvature must have shape "
                "[n_items, n_wavevectors, n_directions]"
            )

        self.scale = self._optional_response_tensor(
            scale,
            "scale",
            positive=True,
        )
        self.weight = self._optional_response_tensor(
            weight,
            "weight",
            nonnegative=True,
        )
        if not torch.all(torch.isfinite(self.density)).item():
            raise ValueError("density must be finite")
        if torch.any(self.density <= 0.0).item():
            raise ValueError("density must be positive")
        if not torch.all(torch.isfinite(self.curvature)).item():
            raise ValueError("curvature must be finite")

        if indices is None:
            self.indices = tuple(range(n_items))
        else:
            values = tuple(indices)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                for value in values
            ):
                raise TypeError("indices must contain only integers")
            if any(value < 0 or value >= n_items for value in values):
                raise IndexError("indices are outside the response data")
            self.indices = tuple(int(value) for value in values)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        frame = dict(self.template)
        n_grid = int(torch.as_tensor(frame["index"]).numel())
        frame["rho"] = self.density[index].expand(n_grid, -1).clone()
        frame[self.wavevector_indices_key] = self.wavevector_indices[index]
        frame[self.target_key] = self.curvature[index]
        if self.scale is not None:
            frame[self.scale_key] = self.scale[index]
        if self.weight is not None:
            frame[self.weights_key] = self.weight[index]
        return frame

    def _optional_response_tensor(
        self,
        value: Optional[Any],
        name: str,
        positive: bool = False,
        nonnegative: bool = False,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=self.curvature.dtype)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        if tensor.shape != self.curvature.shape:
            raise ValueError("{} must have the curvature shape".format(name))
        if not torch.all(torch.isfinite(tensor)).item():
            raise ValueError("{} must be finite".format(name))
        if positive and torch.any(tensor <= 0.0).item():
            raise ValueError("{} must be positive".format(name))
        if nonnegative and torch.any(tensor < 0.0).item():
            raise ValueError("{} must be nonnegative".format(name))
        return tensor
