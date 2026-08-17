"""Physics-based stability objectives for learned density functionals."""

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn


class FourierStabilityLoss(nn.Module):
    r"""Penalize negative fixed-particle-number Fourier curvature.

    For each nonzero integer reciprocal-grid mode ``m``, cosine and sine
    directions are constructed on the periodic grid. On an inhomogeneous
    reference field, a density-weighted mean is removed so that

    ``delta_rho = epsilon * rho * (wave - <wave>_rho)``

    has exactly zero grid sum. Symmetric energy evaluations at
    ``rho +/- delta_rho`` estimate the projected curvature of the total
    intrinsic dimensionless free energy ``beta * (F_id + F_exc)``. External-
    potential and reservoir terms are linear in density and cancel from the
    second difference.

    For a homogeneous one-component fluid, the normalized curvature tends to

    ``rho * delta^2(beta*F) / (DeltaV * sum(delta_rho**2)) = 1 / S(k)``.

    A squared hinge penalizes values below ``minimum_curvature``. This initial
    implementation intentionally supports one density component only; mixture
    stability requires coupled component-space eigenmodes.

    Parameters
    ----------
    modes
        Nonzero integer triplets ``(nx, ny, nz)`` defining lattice-
        commensurate waves. Both cosine and sine phases are evaluated.
    relative_amplitude
        Maximum pointwise fractional density change after fixed-number
        projection. It must lie strictly between zero and one.
    minimum_curvature
        Smallest accepted normalized curvature. Zero penalizes only locally
        unstable directions.
    weight
        Nonnegative multiplier applied to the mean squared hinge.
    name
        Unique name used by :class:`equicdft.loss.Loss`.
    """

    requires_model = True

    def __init__(
        self,
        modes: Sequence[Sequence[int]],
        relative_amplitude: float = 0.05,
        minimum_curvature: float = 0.0,
        weight: float = 1.0,
        name: str = "fourier_stability",
    ) -> None:
        super().__init__()

        self.name = _validate_name(name)
        modes_tensor = torch.as_tensor(modes)
        if (
            modes_tensor.ndim != 2
            or modes_tensor.shape[0] == 0
            or modes_tensor.shape[1] != 3
        ):
            raise ValueError("modes must have shape [n_modes, 3]")
        integer_modes = modes_tensor.to(torch.long)
        if not torch.equal(modes_tensor, integer_modes.to(modes_tensor.dtype)):
            raise ValueError("modes must contain integers")
        if torch.any(torch.all(integer_modes == 0, dim=-1)).item():
            raise ValueError("modes must not contain the zero mode")
        if torch.unique(integer_modes, dim=0).shape[0] != len(integer_modes):
            raise ValueError("modes must not contain duplicates")

        relative_amplitude = _finite_scalar(
            relative_amplitude,
            "relative_amplitude",
        )
        if not 0.0 < relative_amplitude < 1.0:
            raise ValueError("relative_amplitude must lie in (0, 1)")

        self.relative_amplitude = relative_amplitude
        self.minimum_curvature = _finite_nonnegative_scalar(
            minimum_curvature,
            "minimum_curvature",
        )
        self.weight = _finite_nonnegative_scalar(weight, "weight")
        self.register_buffer("modes", integer_modes)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Return the mean squared stability hinge over fields and modes."""

        if model is None:
            raise ValueError("FourierStabilityLoss requires the model")
        if "beta_F_exc" not in outputs:
            raise KeyError("model outputs are missing 'beta_F_exc'")
        for key in ("rho", "grid_positions", "grid_spacing", "temperature"):
            if key not in batch:
                raise KeyError("batch is missing '{}'".format(key))

        rho = batch["rho"]
        if rho.ndim != 3:
            raise ValueError(
                "rho must have shape [n_fields, n_grid, n_types]"
            )
        if rho.shape[-1] != 1:
            raise ValueError(
                "FourierStabilityLoss currently supports one density type"
            )
        if torch.any(rho < 0.0).item():
            raise ValueError("rho must be nonnegative")
        if outputs["beta_F_exc"].shape != rho.shape[:-2]:
            raise ValueError("beta_F_exc must contain one value per field")

        directions, valid_directions = self._directions(batch, rho)
        delta_rho = self.relative_amplitude * directions
        rho_plus = rho[:, None, :, :] + delta_rho
        rho_minus = rho[:, None, :, :] - delta_rho

        # |delta_rho / rho| < 1 by construction, including vacuum points where
        # both rho and the perturbation are exactly zero.
        if torch.any(rho_plus < -1.0e-7).item() or torch.any(
            rho_minus < -1.0e-7
        ).item():
            raise RuntimeError("Fourier perturbation produced negative density")

        n_directions = directions.shape[1]
        perturbed_rho = torch.cat((rho_plus, rho_minus), dim=1)
        perturbed_batch = self._expand_batch(batch, perturbed_rho)
        perturbed_outputs = model(perturbed_batch, compute_c1=False)
        if "beta_F_exc" not in perturbed_outputs:
            raise KeyError("model outputs are missing 'beta_F_exc'")

        cell_volume = torch.prod(batch["grid_spacing"].to(rho), dim=-1)
        reference_energy = (
            self._ideal_free_energy(rho, cell_volume)
            + outputs["beta_F_exc"]
        )
        perturbed_energy = (
            self._ideal_free_energy(
                perturbed_rho,
                cell_volume[:, None].expand(-1, 2 * n_directions),
            )
            + perturbed_outputs["beta_F_exc"]
        )
        plus_energy = perturbed_energy[:, :n_directions]
        minus_energy = perturbed_energy[:, n_directions:]
        second_difference = (
            plus_energy + minus_energy - 2.0 * reference_energy[:, None]
        )

        perturbation_norm = cell_volume[:, None] * torch.sum(
            delta_rho.square(),
            dim=(-2, -1),
        )
        normalized_curvature = (
            torch.mean(rho, dim=(-2, -1))[:, None]
            * second_difference
            / torch.clamp(perturbation_norm, min=1.0e-12)
        )
        valid = valid_directions & (perturbation_norm > 1.0e-12)
        if not torch.any(valid).item():
            raise ValueError("requested modes have no nonconstant real phase")
        hinge = torch.relu(
            self.minimum_curvature - normalized_curvature
        ).square()
        return self.weight * torch.sum(hinge * valid.to(hinge)) / valid.sum()

    def _directions(
        self,
        batch: Dict[str, torch.Tensor],
        rho: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized fixed-number cosine/sine directions and mask."""

        positions = batch["grid_positions"].to(rho)
        if positions.shape != (rho.shape[0], rho.shape[1], 3):
            raise ValueError(
                "grid_positions must have shape [n_fields, n_grid, 3]"
            )
        grid_size = batch.get("grid_size")
        if grid_size is None:
            grid_size = torch.amax(positions, dim=-2) + 1.0
        else:
            grid_size = grid_size.to(rho)
        if grid_size.shape != (rho.shape[0], 3):
            raise ValueError("grid_size must have shape [n_fields, 3]")
        if torch.any(grid_size <= 0.0).item():
            raise ValueError("grid_size must be positive")

        phase = 2.0 * torch.pi * torch.sum(
            positions[:, None, :, :]
            * self.modes.to(rho)[None, :, None, :]
            / grid_size[:, None, None, :],
            dim=-1,
        )
        waves = torch.stack((torch.cos(phase), torch.sin(phase)), dim=2)
        waves = waves.flatten(start_dim=1, end_dim=2)

        density = rho[..., 0]
        total_density = torch.sum(density, dim=-1)
        if torch.any(total_density <= 0.0).item():
            raise ValueError("every field must have positive particle number")
        weighted_mean = torch.sum(
            density[:, None, :] * waves,
            dim=-1,
        ) / total_density[:, None]
        relative_direction = waves - weighted_mean[..., None]
        relative_norm = torch.amax(torch.abs(relative_direction), dim=-1)
        # A sine at an even-grid Nyquist mode is analytically zero but may be
        # of order 1e-6 in float32 after evaluating sin(pi*n).
        valid = relative_norm > 1.0e-5
        relative_direction = relative_direction / torch.clamp(
            relative_norm[..., None],
            min=1.0e-12,
        )
        directions = (density[:, None, :] * relative_direction)[..., None]

        if not torch.all(valid.reshape(rho.shape[0], -1, 2).any(dim=-1)).item():
            raise ValueError(
                "a requested mode aliases to a constant on at least one grid"
            )
        return directions.detach(), valid

    @staticmethod
    def _expand_batch(
        batch: Dict[str, torch.Tensor],
        rho: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Insert the perturbation axis into every field-wise batch tensor."""

        n_fields, n_perturbations = rho.shape[:2]
        expanded = {}
        for key, value in batch.items():
            if key == "rho":
                expanded[key] = rho
            elif (
                torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] == n_fields
            ):
                expanded[key] = value.detach().unsqueeze(1).expand(
                    n_fields,
                    n_perturbations,
                    *value.shape[1:]
                )
            else:
                expanded[key] = value
        return expanded

    @staticmethod
    def _ideal_free_energy(
        rho: torch.Tensor,
        cell_volume: torch.Tensor,
    ) -> torch.Tensor:
        """Return discrete ``beta*F_id``; omitted linear terms cancel."""

        positive = rho > 0.0
        safe_density = torch.where(positive, rho, torch.ones_like(rho))
        integrand = torch.where(
            positive,
            rho * (torch.log(safe_density) - 1.0),
            torch.zeros_like(rho),
        )
        return cell_volume * torch.sum(integrand, dim=(-2, -1))


def _validate_name(name: str) -> str:
    """Return a nonempty loss-term name."""

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a nonempty string")
    return name


def _finite_scalar(value: float, name: str) -> float:
    """Return a validated finite scalar."""

    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a finite scalar".format(name))
    if not math.isfinite(value):
        raise ValueError("{} must be a finite scalar".format(name))
    return value


def _finite_nonnegative_scalar(value: float, name: str) -> float:
    """Return a validated finite nonnegative scalar."""

    value = _finite_scalar(value, name)
    if value < 0.0:
        raise ValueError("{} must be a finite nonnegative scalar".format(name))
    return value
