"""Projected Fourier response evaluation for grid density functionals."""

from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn

from ._argument_checks import boolean, finite_scalar, optional_positive_integer
from ._fourier import (
    expand_mode_amplitudes,
    fourier_curvature_matrix,
    fourier_directions,
    projected_fourier_curvature,
    real_mode_shape,
    validate_explicit_modes,
    validate_response,
    validated_mode_domain,
)


class FourierResponse(nn.Module):
    """Evaluate phase-resolved projected Fourier curvatures.

    Integer reciprocal-grid modes and component-space directions are supplied
    at evaluation time. ``require_uniform`` distinguishes bulk-response
    fitting from stability tests around general inhomogeneous fields.
    ``mode_domain="sphere"`` retains the physical isotropic Nyquist sphere;
    ``"cube"`` admits all componentwise grid-representable wavevectors,
    including the high-wavevector corners outside that sphere.
    """

    def __init__(
        self,
        relative_amplitude: float = 0.01,
        perturbations_per_forward: Optional[int] = None,
        require_uniform: bool = False,
        mode_domain: str = "sphere",
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
        self.mode_domain = validated_mode_domain(mode_domain)

    def forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        modes: torch.Tensor,
        directions: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        *,
        relative_amplitude: Optional[Union[float, torch.Tensor]] = None,
        mode_phases: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return curvature and validity as ``[field, mode, phase, direction]``.

        An optional scalar or [field, mode] amplitude overrides the fixed
        constructor value for this call only, without mutating the module.
        With ``mode_phases`` supplied as radians [field, mode], sum the waves
        into one pattern before projection/normalization. The output shape
        becomes [field, 1, 1, direction] and amplitude tensors use [field, 1].
        """

        if outputs is None:
            outputs = model(batch, compute_c1=False)
        n_directions = directions.shape[0]
        rho = validate_response(
            outputs,
            batch,
            n_types=directions.shape[-1],
            require_uniform=self.require_uniform,
        )
        modes = validate_explicit_modes(
            batch, rho, modes, mode_domain=self.mode_domain,
        )
        perturbations, valid, mean_densities = fourier_directions(
            batch,
            rho,
            modes,
            directions,
            mode_phases=mode_phases,
        )
        n_patterns, n_phases = real_mode_shape(modes, mode_phases)
        curvature, valid = projected_fourier_curvature(
            model=model,
            outputs=outputs,
            batch=batch,
            rho=rho,
            directions=perturbations,
            valid_directions=valid,
            mean_densities=mean_densities,
            relative_amplitude=expand_mode_amplitudes(
                self.relative_amplitude if relative_amplitude is None else relative_amplitude,
                rho, modes[:, :n_patterns], n_directions, n_phases,
            ),
            perturbations_per_forward=self.perturbations_per_forward,
        )
        shape = (rho.shape[0], n_patterns, n_phases, n_directions)
        return curvature.reshape(shape), valid.reshape(shape)

    def matrix(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        modes: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        *,
        relative_amplitude: Optional[Union[float, torch.Tensor]] = None,
        mode_phases: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return the phase-resolved physical-component curvature matrix.

        The matrix has shape ``[field, mode, phase, type, type]`` and uses
        the ideal-gas metric for component normalization. The active-component
        mask has shape ``[field, mode, phase, type]``. For homogeneous fields,
        the matrix is the dimensionless inverse OZ response
        ``I - sqrt(R) c(k) sqrt(R)``.
        The optional scalar or [field, mode] amplitude override is shared by
        all probes reconstructing the same matrix and does not change state.
        With radians ``mode_phases`` [field, mode], all component probes share
        one sum of waves; output is [field, 1, 1, type, type], amplitudes are
        scalar or [field, 1]. This is a projected composite Hessian, not S(k)
        at any single wavevector.
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
        modes = validate_explicit_modes(
            batch, rho, modes, mode_domain=self.mode_domain,
        )
        return fourier_curvature_matrix(
            model=model,
            outputs=outputs,
            batch=batch,
            rho=rho,
            modes=modes,
            relative_amplitude=(
                self.relative_amplitude if relative_amplitude is None else relative_amplitude
            ),
            perturbations_per_forward=self.perturbations_per_forward,
            mode_phases=mode_phases,
        )
