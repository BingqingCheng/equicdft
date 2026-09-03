"""Projected Fourier response evaluation for grid density functionals."""

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from ._argument_checks import boolean, finite_scalar, optional_positive_integer
from ._fourier import (
    fourier_curvature_matrix,
    fourier_directions,
    projected_fourier_curvature,
    validate_explicit_modes,
    validate_response,
)


class FourierResponse(nn.Module):
    """Evaluate phase-resolved projected Fourier curvatures.

    Integer reciprocal-grid modes and component-space directions are supplied
    at evaluation time. ``require_uniform`` distinguishes bulk-response
    fitting from stability tests around general inhomogeneous fields.
    """

    def __init__(
        self,
        relative_amplitude: float = 0.01,
        perturbations_per_forward: Optional[int] = None,
        require_uniform: bool = False,
    ) -> None:
        super().__init__()

        relative_amplitude = finite_scalar(
            relative_amplitude,
            "relative_amplitude",
        )
        if not 0.0 < relative_amplitude < 1.0:
            raise ValueError("relative_amplitude must lie in (0, 1)")
        self.relative_amplitude = relative_amplitude
        self.perturbations_per_forward = optional_positive_integer(
            perturbations_per_forward,
            "perturbations_per_forward",
        )
        self.require_uniform = boolean(require_uniform, "require_uniform")

    def forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        modes: torch.Tensor,
        directions: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return curvature and validity as ``[field, mode, phase, direction]``."""

        if outputs is None:
            outputs = model(batch, compute_c1=False)
        n_directions = directions.shape[0]
        rho = validate_response(
            outputs,
            batch,
            n_types=directions.shape[-1],
            require_uniform=self.require_uniform,
        )
        modes = validate_explicit_modes(batch, rho, modes)
        perturbations, valid, mean_densities = fourier_directions(
            batch,
            rho,
            modes,
            directions,
        )
        curvature, valid = projected_fourier_curvature(
            model=model,
            outputs=outputs,
            batch=batch,
            rho=rho,
            directions=perturbations,
            valid_directions=valid,
            mean_densities=mean_densities,
            relative_amplitude=self.relative_amplitude,
            perturbations_per_forward=self.perturbations_per_forward,
        )
        shape = (rho.shape[0], modes.shape[1], 2, n_directions)
        return curvature.reshape(shape), valid.reshape(shape)

    def matrix(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        modes: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return the phase-resolved physical-component curvature matrix.

        The matrix has shape ``[field, mode, phase, type, type]`` and uses
        the ideal-gas metric for component normalization. The active-component
        mask has shape ``[field, mode, phase, type]``. For homogeneous fields,
        the matrix is the dimensionless inverse OZ response
        ``I - sqrt(R) c(k) sqrt(R)``.
        """

        if outputs is None:
            outputs = model(batch, compute_c1=False)
        if "rho" not in batch:
            raise KeyError("batch is missing 'rho'")
        supplied_rho = batch["rho"]
        if not torch.is_tensor(supplied_rho) or supplied_rho.ndim != 3:
            raise ValueError(
                "rho must have shape [n_fields, n_grid, n_types]"
            )
        if supplied_rho.shape[-1] == 0:
            raise ValueError("rho must contain at least one density type")
        rho = validate_response(
            outputs,
            batch,
            n_types=supplied_rho.shape[-1],
            require_uniform=self.require_uniform,
        )
        modes = validate_explicit_modes(batch, rho, modes)
        return fourier_curvature_matrix(
            model=model,
            outputs=outputs,
            batch=batch,
            rho=rho,
            modes=modes,
            relative_amplitude=self.relative_amplitude,
            perturbations_per_forward=self.perturbations_per_forward,
        )
