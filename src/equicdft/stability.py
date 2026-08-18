"""Physics-based stability objectives for learned density functionals."""

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn


class FourierStabilityLoss(nn.Module):
    r"""Penalize negative fixed-particle-number Fourier curvature.

    For each nonzero integer reciprocal-grid mode, cosine and sine directions
    are constructed on the periodic grid. Each density component is perturbed
    independently according to

    ``delta_rho_a = epsilon * rho_a * (wave - <wave>_rho_a)``,

    so the particle number of that component is unchanged while every other
    component is held fixed. Symmetric evaluations at ``rho +/- delta_rho``
    estimate the projected curvature of the total intrinsic dimensionless free
    energy ``beta * (F_id + F_exc)``. External-potential and reservoir terms
    are linear in density and therefore cancel from the second difference.

    For a homogeneous one-component fluid, the normalized curvature tends to

    ``rho * delta^2(beta*F) / (DeltaV * sum(delta_rho**2)) = 1 / S(k)``.

    For mixtures, the loss averages the independently perturbed component
    directions. It constrains diagonal species curvatures but not coupled
    composition eigenmodes of the full component-space Hessian.

    Parameters
    ----------
    modes
        Optional fixed nonzero integer triplets ``(nx, ny, nz)``. When omitted,
        ``random_modes_per_field`` triplets are sampled independently for every
        field and batch.
    random_modes_per_field
        Number of distinct reciprocal triplets sampled per field. Feasible
        modes lie inside the physically isotropic Nyquist sphere. It must be
        zero when explicit ``modes`` are supplied.
    relative_amplitude
        Maximum pointwise fractional change of the perturbed component after
        its fixed-number projection. It must lie strictly between zero and one.
    minimum_curvature
        Smallest accepted normalized curvature. Zero penalizes only locally
        unstable directions.
    weight
        Nonnegative multiplier applied after averaging the squared hinge over
        fields, modes, real phases, and density components.
    training_only
        Return an exact zero during evaluation. This keeps validation model
        selection tied to the data objective rather than a random regularizer.
    name
        Unique name used by :class:`equicdft.loss.Loss`.
    """

    requires_model = True

    def __init__(
        self,
        modes: Optional[Sequence[Sequence[int]]] = None,
        random_modes_per_field: int = 0,
        relative_amplitude: float = 0.05,
        minimum_curvature: float = 0.0,
        weight: float = 1.0,
        training_only: bool = True,
        name: str = "fourier_stability",
    ) -> None:
        super().__init__()

        self.name = _validate_name(name)
        if modes is None:
            integer_modes = torch.empty((0, 3), dtype=torch.long)
        else:
            modes_tensor = torch.as_tensor(modes)
            if (
                modes_tensor.ndim != 2
                or modes_tensor.shape[0] == 0
                or modes_tensor.shape[1] != 3
            ):
                raise ValueError("modes must have shape [n_modes, 3]")
            integer_modes = modes_tensor.to(torch.long)
            if not torch.equal(
                modes_tensor,
                integer_modes.to(modes_tensor.dtype),
            ):
                raise ValueError("modes must contain integers")
            if torch.any(torch.all(integer_modes == 0, dim=-1)).item():
                raise ValueError("modes must not contain the zero mode")
            if torch.unique(integer_modes, dim=0).shape[0] != len(
                integer_modes
            ):
                raise ValueError("modes must not contain duplicates")

        if (
            isinstance(random_modes_per_field, bool)
            or not isinstance(random_modes_per_field, int)
            or random_modes_per_field < 0
        ):
            raise ValueError(
                "random_modes_per_field must be a nonnegative integer"
            )
        if modes is None and random_modes_per_field == 0:
            raise ValueError(
                "supply modes or a positive random_modes_per_field"
            )
        if modes is not None and random_modes_per_field != 0:
            raise ValueError(
                "random_modes_per_field must be zero when modes are supplied"
            )
        if not isinstance(training_only, bool):
            raise TypeError("training_only must be a boolean")

        relative_amplitude = _finite_scalar(
            relative_amplitude,
            "relative_amplitude",
        )
        if not 0.0 < relative_amplitude < 1.0:
            raise ValueError("relative_amplitude must lie in (0, 1)")

        self.relative_amplitude = relative_amplitude
        self.random_modes_per_field = random_modes_per_field
        self.training_only = training_only
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
        """Return the mean squared stability hinge over valid directions."""

        if model is None:
            raise ValueError("FourierStabilityLoss requires the model")
        if "beta_F_exc" not in outputs:
            raise KeyError("model outputs are missing 'beta_F_exc'")
        if self.training_only and not self.training:
            return outputs["beta_F_exc"].sum() * 0.0
        for key in (
            "rho",
            "grid_positions",
            "grid_size",
            "grid_spacing",
            "temperature",
        ):
            if key not in batch:
                raise KeyError("batch is missing '{}'".format(key))

        rho = batch["rho"]
        if rho.ndim != 3:
            raise ValueError(
                "rho must have shape [n_fields, n_grid, n_types]"
            )
        if rho.shape[-1] == 0:
            raise ValueError("rho must contain at least one density type")
        if torch.any(rho < 0.0).item():
            raise ValueError("rho must be nonnegative")
        if outputs["beta_F_exc"].shape != rho.shape[:-2]:
            raise ValueError("beta_F_exc must contain one value per field")

        modes = self._select_modes(batch, rho)
        directions, valid_directions, mean_densities = self._directions(
            batch,
            rho,
            modes,
        )
        delta_rho = self.relative_amplitude * directions
        rho_plus = rho[:, None, :, :] + delta_rho
        rho_minus = rho[:, None, :, :] - delta_rho

        # |delta_rho_a / rho_a| < 1 for the selected component. Vacuum points
        # have exactly zero perturbation.
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
            mean_densities
            * second_difference
            / torch.clamp(perturbation_norm, min=1.0e-12)
        )
        valid = valid_directions & (perturbation_norm > 1.0e-12)
        if not torch.any(valid).item():
            raise ValueError("batch contains no valid component-mode direction")
        hinge = torch.relu(
            self.minimum_curvature - normalized_curvature
        ).square()
        return self.weight * torch.sum(hinge * valid.to(hinge)) / valid.sum()

    def _directions(
        self,
        batch: Dict[str, torch.Tensor],
        rho: torch.Tensor,
        modes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return componentwise fixed-number directions, validity, and density."""

        n_fields, n_grid, n_types = rho.shape
        positions = batch["grid_positions"].to(rho)
        grid_size = batch["grid_size"].to(rho)
        if positions.shape != (n_fields, n_grid, 3):
            raise ValueError(
                "grid_positions must have shape [n_fields, n_grid, 3]"
            )
        if grid_size.shape != (n_fields, 3):
            raise ValueError("grid_size must have shape [n_fields, 3]")
        if modes.ndim != 3 or modes.shape[0] != n_fields or modes.shape[2] != 3:
            raise ValueError("modes must have shape [n_fields, n_modes, 3]")

        phase = 2.0 * torch.pi * torch.sum(
            positions[:, None, :, :]
            * modes.to(rho)[:, :, None, :]
            / grid_size[:, None, None, :],
            dim=-1,
        )
        waves = torch.stack((torch.cos(phase), torch.sin(phase)), dim=2)
        waves = waves.flatten(start_dim=1, end_dim=2)

        total_density = torch.sum(rho, dim=-2)
        component_present = total_density > 1.0e-12
        weighted_mean = torch.sum(
            rho[:, None, :, :] * waves[..., None],
            dim=-2,
        ) / torch.clamp(total_density[:, None, :], min=1.0e-12)
        relative_direction = waves[..., None] - weighted_mean[:, :, None, :]
        relative_norm = torch.amax(torch.abs(relative_direction), dim=-2)
        # A sine at an even-grid Nyquist mode is analytically zero but may be
        # of order 1e-6 in float32 after evaluating sin(pi*n).
        valid = (relative_norm > 1.0e-5) & component_present[:, None, :]
        relative_direction = relative_direction / torch.clamp(
            relative_norm[:, :, None, :],
            min=1.0e-12,
        )
        component_directions = rho[:, None, :, :] * relative_direction

        # Introduce a direction axis for the perturbed component while keeping
        # all other density components exactly fixed.
        identity = torch.eye(n_types, device=rho.device, dtype=rho.dtype)
        directions = torch.einsum(
            "bdgc,ct->bdcgt",
            component_directions,
            identity,
        ).flatten(start_dim=1, end_dim=2)
        valid = valid.flatten(start_dim=1, end_dim=2)
        mean_densities = (
            (total_density / n_grid)[:, None, :]
            .expand(-1, waves.shape[1], -1)
            .reshape(n_fields, -1)
        )

        valid_by_mode = valid.reshape(
            n_fields,
            modes.shape[1],
            2,
            n_types,
        ).any(dim=2)
        invalid_present = component_present[:, None, :] & ~valid_by_mode
        if torch.any(invalid_present).item():
            raise ValueError(
                "a requested mode aliases to a constant for a present component"
            )
        return directions.detach(), valid, mean_densities.detach()

    def _select_modes(
        self,
        batch: Dict[str, torch.Tensor],
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Return fixed or randomly sampled physical modes for every field."""

        n_fields = rho.shape[0]
        grid_size = batch["grid_size"].detach().cpu()
        grid_spacing = batch["grid_spacing"].detach().cpu()
        if grid_size.shape != (n_fields, 3):
            raise ValueError("grid_size must have shape [n_fields, 3]")
        if grid_spacing.shape != (n_fields, 3):
            raise ValueError("grid_spacing must have shape [n_fields, 3]")
        integer_grid_size = grid_size.to(torch.long)
        if not torch.equal(grid_size, integer_grid_size.to(grid_size.dtype)):
            raise ValueError("grid_size must contain integers")

        selected_by_field = []
        for field in range(n_fields):
            size = tuple(integer_grid_size[field].tolist())
            spacing = tuple(grid_spacing[field].tolist())
            candidates = _feasible_modes(size, spacing)
            if self.modes.shape[0] > 0:
                candidate_set = set(candidates)
                selected = [
                    _canonical_grid_mode(mode, size)
                    for mode in self.modes.detach().cpu().tolist()
                ]
                if len(set(selected)) != len(selected):
                    raise ValueError(
                        "modes contain equivalent Fourier directions"
                    )
                if any(mode not in candidate_set for mode in selected):
                    raise ValueError(
                        "a requested mode lies outside the isotropic Nyquist sphere"
                    )
            else:
                if self.random_modes_per_field > len(candidates):
                    raise ValueError(
                        "random_modes_per_field exceeds the feasible modes"
                    )
                indices = torch.randperm(len(candidates))[
                    : self.random_modes_per_field
                ].tolist()
                selected = [candidates[index] for index in indices]
            selected_by_field.append(
                torch.tensor(selected, dtype=torch.long)
            )

        return torch.stack(selected_by_field).to(device=rho.device)

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


def _feasible_modes(
    grid_size: Sequence[int],
    grid_spacing: Sequence[float],
) -> Sequence[Tuple[int, int, int]]:
    """Return unique modes inside the physical isotropic Nyquist sphere."""

    if len(grid_size) != 3 or any(size <= 0 for size in grid_size):
        raise ValueError("grid_size must contain three positive integers")
    if (
        len(grid_spacing) != 3
        or any(not math.isfinite(spacing) for spacing in grid_spacing)
        or any(spacing <= 0.0 for spacing in grid_spacing)
    ):
        raise ValueError("grid_spacing must contain three positive values")

    box_lengths = [
        size * spacing for size, spacing in zip(grid_size, grid_spacing)
    ]
    isotropic_nyquist = min(math.pi / spacing for spacing in grid_spacing)
    maximum_squared = isotropic_nyquist**2 * (1.0 + 1.0e-12)
    half_sizes = [size // 2 for size in grid_size]
    feasible = set()
    for nx in range(-half_sizes[0], half_sizes[0] + 1):
        for ny in range(-half_sizes[1], half_sizes[1] + 1):
            for nz in range(-half_sizes[2], half_sizes[2] + 1):
                mode = (nx, ny, nz)
                if mode == (0, 0, 0):
                    continue
                squared_wavevector = sum(
                    (2.0 * math.pi * component / length) ** 2
                    for component, length in zip(mode, box_lengths)
                )
                if squared_wavevector <= maximum_squared:
                    feasible.add(_canonical_grid_mode(mode, grid_size))
    return sorted(feasible)


def _canonical_grid_mode(
    mode: Sequence[int],
    grid_size: Sequence[int],
) -> Tuple[int, int, int]:
    """Remove Nyquist-sign and global-sign duplicates of a real Fourier mode."""

    canonical = []
    for component, size in zip(mode, grid_size):
        if size % 2 == 0 and abs(component) == size // 2:
            component = abs(component)
        canonical.append(component)
    first_nonzero = next(component for component in canonical if component != 0)
    if first_nonzero < 0:
        canonical = [-component for component in canonical]
    return tuple(canonical)


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
