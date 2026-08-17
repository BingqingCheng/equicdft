"""Physics-based stability objectives for learned density functionals."""

import math
from itertools import permutations, product
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
        Optional nonzero integer triplets ``(nx, ny, nz)`` defining lattice-
        commensurate waves. They are fixed when supplied. Use ``None`` with a
        positive ``random_modes_per_field`` to sample feasible triplets.
    random_modes_per_field
        Number of distinct reciprocal triplets sampled for each field on every
        training batch. Feasible modes are nonzero and lie inside the
        isotropically complete Nyquist sphere. It must be zero when explicit
        ``modes`` are supplied.
    expand_cubic_orbits
        For explicitly supplied fixed modes, expand each mode under signed axis
        permutations, removing the redundant global sign because cosine and
        sine already span ``+/- k``. Randomly sampled triplets are deliberately
        not symmetry-expanded.
    relative_amplitude
        Maximum pointwise fractional density change after fixed-number
        projection. It must lie strictly between zero and one.
    minimum_curvature
        Smallest accepted normalized curvature. Zero penalizes only locally
        unstable directions.
    weight
        Nonnegative multiplier applied to the mean squared hinge.
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
        expand_cubic_orbits: bool = False,
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
        if random_modes_per_field != 0 and expand_cubic_orbits:
            raise ValueError(
                "randomly sampled modes cannot use cubic-orbit expansion"
            )
        if not isinstance(expand_cubic_orbits, bool):
            raise TypeError("expand_cubic_orbits must be a boolean")
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
        self.expand_cubic_orbits = expand_cubic_orbits
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
        """Return the mean squared stability hinge over fields and modes."""

        if model is None:
            raise ValueError("FourierStabilityLoss requires the model")
        if "beta_F_exc" not in outputs:
            raise KeyError("model outputs are missing 'beta_F_exc'")
        if self.training_only and not self.training:
            return outputs["beta_F_exc"].sum() * 0.0
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

        modes, valid_modes = self._select_modes(batch, rho)
        directions, valid_directions = self._directions(
            batch,
            rho,
            modes,
            valid_modes,
        )
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
        modes: torch.Tensor,
        valid_modes: torch.Tensor,
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
            * modes.to(rho)[:, :, None, :]
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
        phase_valid = valid_modes[..., None].expand(-1, -1, 2).flatten(1)
        valid = (relative_norm > 1.0e-5) & phase_valid
        relative_direction = relative_direction / torch.clamp(
            relative_norm[..., None],
            min=1.0e-12,
        )
        directions = (density[:, None, :] * relative_direction)[..., None]

        valid_by_mode = valid.reshape(rho.shape[0], -1, 2).any(dim=-1)
        if torch.any(valid_modes & ~valid_by_mode).item():
            raise ValueError(
                "a requested mode aliases to a constant on at least one grid"
            )
        return directions.detach(), valid

    def _select_modes(
        self,
        batch: Dict[str, torch.Tensor],
        rho: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return fixed or randomly sampled modes for the current batch."""

        n_fields = rho.shape[0]
        if self.modes.shape[0] > 0:
            modes = self.modes
            if self.expand_cubic_orbits:
                self._validate_cubic_grid(batch)
                modes = _expand_cubic_orbits(modes)
            modes = modes.to(device=rho.device)
            modes = modes[None, :, :].expand(n_fields, -1, -1)
            valid = torch.ones(
                modes.shape[:2],
                device=rho.device,
                dtype=torch.bool,
            )
            return modes, valid

        grid_size = batch.get("grid_size")
        if grid_size is None:
            positions = batch["grid_positions"]
            grid_size = torch.amax(positions, dim=-2) + 1
        grid_size = grid_size.detach().cpu().to(torch.long)

        selected_by_field = []
        for field in range(n_fields):
            candidates = _feasible_modes(
                grid_size[field].tolist(),
                orbit_representatives=False,
            )
            if self.random_modes_per_field > len(candidates):
                raise ValueError(
                    "random_modes_per_field exceeds the feasible modes"
                )
            selection = torch.randperm(len(candidates))[
                : self.random_modes_per_field
            ].tolist()
            selected = [candidates[index] for index in selection]
            selected_by_field.append(selected)

        maximum_modes = max(len(selected) for selected in selected_by_field)
        modes = torch.zeros(
            (n_fields, maximum_modes, 3),
            device=rho.device,
            dtype=torch.long,
        )
        valid = torch.zeros(
            (n_fields, maximum_modes),
            device=rho.device,
            dtype=torch.bool,
        )
        for field, selected in enumerate(selected_by_field):
            count = len(selected)
            modes[field, :count] = torch.tensor(
                selected,
                device=rho.device,
                dtype=torch.long,
            )
            valid[field, :count] = True
        return modes, valid

    @staticmethod
    def _validate_cubic_grid(batch: Dict[str, torch.Tensor]) -> None:
        """Require cubic geometry before treating axis permutations equally."""

        if "grid_size" in batch:
            grid_size = batch["grid_size"]
            if not torch.all(grid_size == grid_size[..., :1]).item():
                raise ValueError(
                    "cubic-orbit expansion requires equal grid dimensions"
                )
        spacing = batch["grid_spacing"]
        if not torch.allclose(spacing, spacing[..., :1].expand_as(spacing)):
            raise ValueError(
                "cubic-orbit expansion requires equal grid spacings"
            )

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


def _expand_cubic_orbits(modes: torch.Tensor) -> torch.Tensor:
    """Return unique signed permutations, identifying global ``+/-`` pairs."""

    expanded = set()
    for mode in modes.detach().cpu().tolist():
        for permuted in set(permutations(mode)):
            for signs in product((-1, 1), repeat=3):
                candidate = tuple(
                    sign * component
                    for sign, component in zip(signs, permuted)
                )
                first_nonzero = next(
                    component for component in candidate if component != 0
                )
                if first_nonzero < 0:
                    candidate = tuple(-component for component in candidate)
                expanded.add(candidate)
    return torch.tensor(
        sorted(expanded),
        dtype=torch.long,
        device=modes.device,
    )


def _feasible_modes(
    grid_size: Sequence[int],
    orbit_representatives: bool,
) -> Sequence[Tuple[int, int, int]]:
    """Return nonzero modes inside the isotropic Nyquist sphere."""

    if len(grid_size) != 3 or any(size <= 0 for size in grid_size):
        raise ValueError("grid_size must contain three positive integers")
    half_sizes = [size // 2 for size in grid_size]
    feasible = set()
    for nx in range(-half_sizes[0], half_sizes[0] + 1):
        for ny in range(-half_sizes[1], half_sizes[1] + 1):
            for nz in range(-half_sizes[2], half_sizes[2] + 1):
                mode = (nx, ny, nz)
                if mode == (0, 0, 0):
                    continue
                scaled_squared_norm = sum(
                    (2.0 * component / size) ** 2
                    for component, size in zip(mode, grid_size)
                )
                if scaled_squared_norm > 1.0 + 1.0e-12:
                    continue
                if orbit_representatives:
                    mode = tuple(sorted(map(abs, mode), reverse=True))
                else:
                    mode = _canonical_grid_mode(mode, grid_size)
                feasible.add(mode)
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


def _finite_nonnegative_scalar(value: float, name: str) -> float:
    """Return a validated finite nonnegative scalar."""

    value = _finite_scalar(value, name)
    if value < 0.0:
        raise ValueError("{} must be a finite nonnegative scalar".format(name))
    return value
