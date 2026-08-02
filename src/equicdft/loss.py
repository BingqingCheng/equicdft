"""Composable objectives for training grid density-functional models."""

import math
from typing import Dict, Optional, Sequence, Union

import torch
from torch import nn


class TensorLoss(nn.Module):
    """Apply one weighted scalar loss to a prediction/target tensor pair.

    Parameters
    ----------
    name
        Unique name used in the dictionary returned by :class:`Loss`.
    prediction_key
        Key selecting the predicted tensor from the model outputs.
    target_key
        Key selecting the reference tensor. The training batch is checked
        first, followed by the model outputs.
    weights_key
        Optional key selecting element weights from the model outputs or batch.
        With weights, the default is elementwise squared error followed by a
        weighted mean.
    loss_fn
        PyTorch loss module. It must reduce the selected tensors to one scalar.
        The default is mean-squared error.
    weight
        Nonnegative multiplier applied to this loss term.

    Notes
    -----
    A componentwise target lacking only the prediction's grid axis is expanded
    over that axis. All other shape mismatches are rejected.
    """

    def __init__(
        self,
        name: str,
        prediction_key: str,
        target_key: str,
        loss_fn: Optional[nn.Module] = None,
        weights_key: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.name = _validate_name(name)
        self.prediction_key = _validate_key(prediction_key, "prediction_key")
        self.target_key = _validate_key(target_key, "target_key")
        self.weights_key = (
            None
            if weights_key is None
            else _validate_key(weights_key, "weights_key")
        )
        if loss_fn is None:
            loss_fn = nn.MSELoss(
                reduction="none" if self.weights_key else "mean"
            )
        if not isinstance(loss_fn, nn.Module):
            raise TypeError("loss_fn must be a torch.nn.Module")
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise ValueError("weight must be a finite nonnegative scalar")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be a finite nonnegative scalar")

        self.loss_fn = loss_fn
        self.weight = weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return this term's weighted scalar loss."""

        if self.prediction_key not in outputs:
            raise KeyError(
                "model outputs are missing prediction '{}'".format(
                    self.prediction_key
                )
            )
        if self.target_key in batch:
            target = batch[self.target_key]
        elif self.target_key in outputs:
            target = outputs[self.target_key]
        else:
            raise KeyError(
                "batch and model outputs are missing target '{}'".format(
                    self.target_key
                )
            )

        prediction = outputs[self.prediction_key]
        if prediction.shape != target.shape:
            component_target_shape = (
                prediction.shape[:-2] + prediction.shape[-1:]
            )
            if target.shape == component_target_shape:
                target = target.unsqueeze(-2).expand_as(prediction)
            else:
                raise ValueError(
                    "prediction '{}' has shape {}, but target '{}' has shape "
                    "{}".format(
                        self.prediction_key,
                        tuple(prediction.shape),
                        self.target_key,
                        tuple(target.shape),
                    )
                )

        value = self.loss_fn(prediction, target)
        if self.weights_key is None:
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError("loss_fn must return one scalar tensor")
        else:
            if self.weights_key in outputs:
                weights = outputs[self.weights_key]
            elif self.weights_key in batch:
                weights = batch[self.weights_key]
            else:
                raise KeyError(
                    "batch and model outputs are missing weights '{}'".format(
                        self.weights_key
                    )
                )
            if weights.shape != prediction.shape:
                raise ValueError("weights and prediction must have same shape")
            if value.shape != prediction.shape:
                raise ValueError(
                    "weighted loss_fn must return one value per prediction"
                )
            weights = weights.detach().to(prediction)
            total_weight = weights.sum()
            if total_weight.item() <= 0.0:
                raise ValueError("weights must have a positive sum")
            value = (weights * value).sum() / total_weight
        return self.weight * value


class DensityPerturbationStabilityLoss(nn.Module):
    """Penalize negative curvature around equilibrium fixed-N fields.

    For every equilibrium field, this term draws a bounded multiplicative
    random direction tangent to each component particle-number constraint. It
    evaluates small symmetric perturbations ``rho +/- delta_rho`` and compares
    their mean
    dimensionless thermodynamic objective with the reference,

    ``beta*Phi = beta*F_id + beta*F_exc + beta*integral(rho*V_ext)``

    The symmetric second difference cancels the first-order contribution and
    is negative only when the learned functional has negative curvature along
    the sampled direction. It is divided by particle number and squared
    relative amplitude so its scale remains comparable between fields and
    perturbation sizes.

    Each component particle number is unchanged, so the reservoir contribution
    cancels exactly: ``Delta(beta*Omega) = Delta(beta*Phi) - sum_i
    beta*mu_i*Delta N_i = Delta(beta*Phi)``.

    The term uses energy-only model evaluations for perturbed fields. It does
    not require direct-correlation targets for those artificial fields.

    Parameters
    ----------
    maximum_density
        Positive scalar or one upper density per physical component. Random
        directions are shortened when needed so both signs remain below it.
    relative_amplitudes
        Small positive relative density amplitudes. A value of 0.05 gives
        changes of order five percent before particle-number projection.
    curvature_tolerance
        Allowed negative symmetric curvature per particle before the hinge
        activates.
    weight
        Nonnegative multiplier applied to the mean hinge value.
    random_seed
        Integer seed used by the reproducible evaluation-time lattice hash.
    name
        Unique name used by :class:`Loss`.
    """

    requires_model = True

    def __init__(
        self,
        maximum_density: Union[
            float,
            Sequence[float],
            torch.Tensor,
        ],
        relative_amplitudes: Sequence[float] = (0.01, 0.02, 0.05),
        curvature_tolerance: float = 0.0,
        weight: float = 1.0,
        random_seed: int = 17,
        name: str = "density_perturbation_stability",
    ) -> None:
        super().__init__()

        self.name = _validate_name(name)
        density_cap = torch.as_tensor(
            maximum_density,
            dtype=torch.get_default_dtype(),
        ).detach().clone().reshape(-1)
        if (
            density_cap.numel() == 0
            or not torch.all(torch.isfinite(density_cap)).item()
            or torch.any(density_cap <= 0.0).item()
        ):
            raise ValueError(
                "maximum_density must contain finite positive values"
            )

        amplitudes_tensor = torch.as_tensor(
            list(relative_amplitudes),
            dtype=torch.get_default_dtype(),
        ).detach().clone().reshape(-1)
        if (
            amplitudes_tensor.numel() == 0
            or not torch.all(torch.isfinite(amplitudes_tensor)).item()
            or torch.any(amplitudes_tensor <= 0.0).item()
            or torch.any(amplitudes_tensor >= 0.5).item()
        ):
            raise ValueError(
                "relative_amplitudes must contain values in (0, 0.5)"
            )
        if torch.any(amplitudes_tensor[1:] <= amplitudes_tensor[:-1]).item():
            raise ValueError(
                "relative_amplitudes must be strictly increasing"
            )

        curvature_tolerance = _finite_nonnegative_scalar(
            curvature_tolerance,
            "curvature_tolerance",
        )
        weight = _finite_nonnegative_scalar(weight, "weight")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer")

        self.weight = weight
        self.curvature_tolerance = curvature_tolerance
        self.random_seed = random_seed
        self.register_buffer("maximum_density", density_cap)
        self.register_buffer("relative_amplitudes", amplitudes_tensor)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Return the mean fixed-N thermodynamic stability hinge."""

        if model is None:
            raise ValueError(
                "DensityPerturbationStabilityLoss requires the model"
            )
        required_batch = (
            "rho",
            "V_ext",
            "beta",
            "temperature",
            "grid_spacing",
            "grid_positions",
            "local_density_index",
        )
        missing = [key for key in required_batch if key not in batch]
        if missing:
            raise KeyError(
                "batch is missing required stability keys: {}".format(
                    ", ".join(missing)
                )
            )
        if "beta_F_exc" not in outputs:
            raise KeyError("model outputs are missing 'beta_F_exc'")

        rho_reference = batch["rho"].detach()
        V_ext = batch["V_ext"].detach().to(rho_reference)
        if rho_reference.ndim != 3:
            raise ValueError(
                "DensityPerturbationStabilityLoss expects batched rho with shape "
                "[n_fields, n_grid, n_types]"
            )
        if V_ext.shape != rho_reference.shape:
            raise ValueError("V_ext and rho must have the same shape")

        n_fields, n_grid, n_types = rho_reference.shape
        density_cap = self.maximum_density.to(rho_reference)
        if density_cap.numel() == 1:
            density_cap = density_cap.repeat(n_types)
        if density_cap.shape != (n_types,):
            raise ValueError(
                "maximum_density must contain one value or one per type"
            )
        if torch.any(
            rho_reference > density_cap[None, None, :] + 1.0e-6
        ).item():
            raise ValueError(
                "a reference density exceeds maximum_density"
            )

        target_sums = torch.sum(rho_reference, dim=-2)
        if torch.any(target_sums > n_grid * density_cap[None, :]).item():
            raise ValueError(
                "reference particle numbers are infeasible under "
                "maximum_density"
            )
        grid_positions = batch["grid_positions"].detach()
        if grid_positions.shape != (n_fields, n_grid, 3):
            raise ValueError(
                "grid_positions must have shape [n_fields, n_grid, 3]"
            )
        if self.training:
            random_field = 2.0 * torch.rand_like(rho_reference) - 1.0
        else:
            grid_size = torch.max(grid_positions, dim=1).values + 1
            linear_index = (
                grid_positions[..., 0]
                + grid_size[:, None, 0]
                * (
                    grid_positions[..., 1]
                    + grid_size[:, None, 1] * grid_positions[..., 2]
                )
            )
            component_offset = torch.arange(
                n_types,
                device=rho_reference.device,
                dtype=torch.long,
            )[None, None, :]
            hashed = torch.remainder(
                linear_index[:, :, None] * 1103515245
                + component_offset * 12345
                + self.random_seed,
                2147483647,
            )
            random_field = (
                hashed.to(rho_reference.dtype)
                / 1073741823.5
                - 1.0
            )

        # Center the dimensionless random field using rho as its measure. Thus
        # sum_grid direction = 0 independently for every field and component.
        # Multiplication by rho prevents a dilute voxel from setting the scale
        # of the complete perturbation. The final normalization makes eps the
        # maximum pointwise fractional density change.
        density_weighted_mean = torch.sum(
            rho_reference * random_field,
            dim=-2,
        ) / torch.clamp(target_sums, min=1.0e-12)
        relative_direction = (
            random_field - density_weighted_mean[:, None, :]
        )
        relative_norm = torch.amax(
            torch.abs(relative_direction),
            dim=-2,
        )
        relative_direction = relative_direction / torch.clamp(
            relative_norm[:, None, :],
            min=1.0e-12,
        )
        direction = rho_reference * relative_direction

        # Shorten a requested relative amplitude only when needed to keep both
        # signs positive and below maximum_density.
        absolute_direction = torch.abs(direction)
        available_change = torch.clamp(
            torch.minimum(
                rho_reference,
                density_cap[None, None, :] - rho_reference,
            ),
            min=0.0,
        )
        amplitude_limits = torch.where(
            absolute_direction > 0.0,
            available_change / absolute_direction,
            torch.full_like(absolute_direction, float("inf")),
        )
        amplitude_limits = 0.95 * torch.amin(amplitude_limits, dim=-2)
        requested_amplitudes = self.relative_amplitudes.to(rho_reference)
        n_amplitudes = requested_amplitudes.numel()
        actual_amplitudes = torch.minimum(
            requested_amplitudes[None, :, None],
            amplitude_limits[:, None, :],
        )
        delta_rho = (
            actual_amplitudes[:, :, None, :] * direction[:, None, :, :]
        )
        rho_plus = rho_reference[:, None, :, :] + delta_rho
        rho_minus = rho_reference[:, None, :, :] - delta_rho
        rho_plus = rho_plus * (
            target_sums[:, None, None, :]
            / torch.sum(rho_plus, dim=-2)[:, :, None, :]
        )
        rho_minus = rho_minus * (
            target_sums[:, None, None, :]
            / torch.sum(rho_minus, dim=-2)[:, :, None, :]
        )
        rho_plus = _correct_component_grid_sums(
            rho_plus,
            target_sums,
            density_cap,
        )
        rho_minus = _correct_component_grid_sums(
            rho_minus,
            target_sums,
            density_cap,
        )
        accumulation_dtype = _accumulation_dtype(rho_reference.dtype)
        particle_number_error = torch.maximum(
            torch.amax(
                torch.abs(
                    torch.sum(
                        rho_plus.to(accumulation_dtype),
                        dim=-2,
                    )
                    - target_sums[:, None, :].to(accumulation_dtype)
                )
            ),
            torch.amax(
                torch.abs(
                    torch.sum(
                        rho_minus.to(accumulation_dtype),
                        dim=-2,
                    )
                    - target_sums[:, None, :].to(accumulation_dtype)
                )
            ),
        )
        if particle_number_error.item() > 1.0e-4:
            raise RuntimeError(
                "density perturbations failed to preserve particle numbers"
            )
        rho_perturbed = torch.cat((rho_plus, rho_minus), dim=1)
        n_perturbations = 2 * n_amplitudes

        # The model supports arbitrary leading batch dimensions. Expand only
        # the fields needed by the energy-only representation/readout pass.
        perturbed_data = {
            "rho": rho_perturbed,
            "local_density_index": batch["local_density_index"]
            .detach()
            .unsqueeze(1)
            .expand(-1, n_perturbations, -1, -1),
            "temperature": batch["temperature"]
            .detach()
            .unsqueeze(1)
            .expand(-1, n_perturbations),
            "grid_spacing": batch["grid_spacing"]
            .detach()
            .unsqueeze(1)
            .expand(-1, n_perturbations, -1),
        }
        perturbed_outputs = model(perturbed_data, compute_c1=False)

        cell_volume = torch.prod(
            batch["grid_spacing"].detach().to(rho_reference),
            dim=-1,
        )
        beta = batch["beta"].detach().to(rho_reference)
        wavelength = model.thermal_wavelength.detach().to(rho_reference)
        if wavelength.shape != (n_types,):
            raise ValueError(
                "model thermal_wavelength must contain one value per type"
            )

        reference_ideal = _ideal_free_energy(
            rho_reference,
            wavelength,
            cell_volume,
        )
        perturbed_ideal = _ideal_free_energy(
            rho_perturbed,
            wavelength,
            cell_volume[:, None].expand(-1, n_perturbations),
        )
        reference_external = cell_volume * torch.sum(
            rho_reference * beta[:, None, None] * V_ext,
            dim=(-2, -1),
        )
        perturbed_external = cell_volume[:, None] * torch.sum(
            rho_perturbed
            * beta[:, None, None, None]
            * V_ext[:, None, :, :],
            dim=(-2, -1),
        )

        reference_objective = (
            reference_ideal
            + outputs["beta_F_exc"]
            + reference_external
        )
        perturbed_objective = (
            perturbed_ideal
            + perturbed_outputs["beta_F_exc"]
            + perturbed_external
        )
        total_particles = (
            cell_volume * torch.sum(rho_reference, dim=(-2, -1))
        )
        plus_objective = perturbed_objective[:, :n_amplitudes]
        minus_objective = perturbed_objective[:, n_amplitudes:]
        symmetric_objective = 0.5 * (
            plus_objective + minus_objective
        )
        component_grid_sums = target_sums
        squared_relative_amplitude = torch.sum(
            component_grid_sums[:, None, :] * actual_amplitudes.square(),
            dim=-1,
        ) / torch.sum(component_grid_sums, dim=-1)[:, None]
        valid_scale = squared_relative_amplitude > 1.0e-12
        curvature_violation = (
            reference_objective[:, None] - symmetric_objective
        ) / (
            total_particles[:, None]
            * torch.clamp(squared_relative_amplitude, min=1.0e-12)
        )
        hinge = torch.where(
            valid_scale,
            torch.relu(curvature_violation - self.curvature_tolerance),
            torch.zeros_like(curvature_violation),
        )
        return self.weight * torch.mean(hinge)


class GlobalDensityStabilityLoss(nn.Module):
    """Enforce finite fixed-N candidates above an equilibrium objective.

    For each reference field, a random cap-limited trial density is built with
    exactly the same particle number of every component. Candidate fields lie
    on the concentration path

    ``rho_lambda = (1 - lambda) * rho + lambda * rho_trial``.

    The loss penalizes every sampled candidate whose dimensionless canonical
    objective is below that of the equilibrium reference. Unlike
    :class:`DensityPerturbationStabilityLoss`, this is a finite-displacement
    ordering test rather than a local curvature test. Sampling cannot prove a
    global minimum, but it directly supplies hard negative density fields.

    Since every candidate preserves each ``N_i``, the chemical-potential term
    cancels: ``Delta(beta*Omega) = Delta(beta*Phi)``.

    Parameters
    ----------
    maximum_density
        Positive scalar or one upper density per physical component. The
        random trial field is constructed inside this bound.
    mixing_fractions
        Increasing path fractions in ``(0, 1]``. A value of one evaluates the
        complete random trial density.
    energy_tolerance
        Allowed objective decrease per particle before the hinge activates.
    weight
        Nonnegative multiplier applied to the mean hinge value.
    random_seed
        Integer seed used by the reproducible evaluation-time lattice hash.
    trial_strategy
        ``"random"`` selects cap-filled cells randomly. The more adversarial
        ``"lowest_external_potential"`` fills cells in increasing
        ``V_ext`` order.
    name
        Unique name used by :class:`Loss`.
    """

    requires_model = True

    def __init__(
        self,
        maximum_density: Union[
            float,
            Sequence[float],
            torch.Tensor,
        ],
        mixing_fractions: Sequence[float] = (0.25, 0.5, 1.0),
        energy_tolerance: float = 0.0,
        weight: float = 1.0,
        random_seed: int = 29,
        trial_strategy: str = "random",
        name: str = "global_density_stability",
    ) -> None:
        super().__init__()

        self.name = _validate_name(name)
        density_cap = torch.as_tensor(
            maximum_density,
            dtype=torch.get_default_dtype(),
        ).detach().clone().reshape(-1)
        if (
            density_cap.numel() == 0
            or not torch.all(torch.isfinite(density_cap)).item()
            or torch.any(density_cap <= 0.0).item()
        ):
            raise ValueError(
                "maximum_density must contain finite positive values"
            )

        fractions = torch.as_tensor(
            list(mixing_fractions),
            dtype=torch.get_default_dtype(),
        ).detach().clone().reshape(-1)
        if (
            fractions.numel() == 0
            or not torch.all(torch.isfinite(fractions)).item()
            or torch.any(fractions <= 0.0).item()
            or torch.any(fractions > 1.0).item()
        ):
            raise ValueError("mixing_fractions must contain values in (0, 1]")
        if torch.any(fractions[1:] <= fractions[:-1]).item():
            raise ValueError("mixing_fractions must be strictly increasing")

        self.energy_tolerance = _finite_nonnegative_scalar(
            energy_tolerance,
            "energy_tolerance",
        )
        self.weight = _finite_nonnegative_scalar(weight, "weight")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer")
        if trial_strategy not in (
            "random",
            "lowest_external_potential",
        ):
            raise ValueError(
                "trial_strategy must be 'random' or "
                "'lowest_external_potential'"
            )
        self.random_seed = random_seed
        self.trial_strategy = trial_strategy
        self.register_buffer("maximum_density", density_cap)
        self.register_buffer("mixing_fractions", fractions)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Return the sampled finite-displacement objective-ordering hinge."""

        if model is None:
            raise ValueError("GlobalDensityStabilityLoss requires the model")
        required_batch = (
            "rho",
            "V_ext",
            "beta",
            "temperature",
            "grid_spacing",
            "grid_positions",
            "local_density_index",
        )
        missing = [key for key in required_batch if key not in batch]
        if missing:
            raise KeyError(
                "batch is missing required stability keys: {}".format(
                    ", ".join(missing)
                )
            )
        if "beta_F_exc" not in outputs:
            raise KeyError("model outputs are missing 'beta_F_exc'")

        rho_reference = batch["rho"].detach()
        V_ext = batch["V_ext"].detach().to(rho_reference)
        if rho_reference.ndim != 3:
            raise ValueError(
                "GlobalDensityStabilityLoss expects batched rho with shape "
                "[n_fields, n_grid, n_types]"
            )
        if V_ext.shape != rho_reference.shape:
            raise ValueError("V_ext and rho must have the same shape")

        n_fields, n_grid, n_types = rho_reference.shape
        density_cap = self.maximum_density.to(rho_reference)
        if density_cap.numel() == 1:
            density_cap = density_cap.repeat(n_types)
        if density_cap.shape != (n_types,):
            raise ValueError(
                "maximum_density must contain one value or one per type"
            )
        if torch.any(
            rho_reference > density_cap[None, None, :] + 1.0e-6
        ).item():
            raise ValueError("a reference density exceeds maximum_density")

        target_sums = torch.sum(rho_reference, dim=-2)
        if torch.any(target_sums > n_grid * density_cap[None, :]).item():
            raise ValueError(
                "reference particle numbers are infeasible under "
                "maximum_density"
            )
        grid_positions = batch["grid_positions"].detach()
        if grid_positions.shape != (n_fields, n_grid, 3):
            raise ValueError(
                "grid_positions must have shape [n_fields, n_grid, 3]"
            )
        if self.trial_strategy == "lowest_external_potential":
            selection_scores = V_ext
        elif self.training:
            selection_scores = torch.rand_like(rho_reference)
        else:
            grid_size = torch.max(grid_positions, dim=1).values + 1
            linear_index = (
                grid_positions[..., 0]
                + grid_size[:, None, 0]
                * (
                    grid_positions[..., 1]
                    + grid_size[:, None, 1] * grid_positions[..., 2]
                )
            )
            component_offset = torch.arange(
                n_types,
                device=rho_reference.device,
                dtype=torch.long,
            )[None, None, :]
            selection_scores = torch.remainder(
                linear_index[:, :, None] * 1103515245
                + component_offset * 12345
                + self.random_seed,
                2147483647,
            ).to(rho_reference.dtype)

        # Fill randomly selected cells up to the density cap, with at most one
        # partially filled cell. This gives a deliberately difficult trial
        # field while preserving every component grid sum.
        ranks = torch.arange(
            n_grid,
            device=rho_reference.device,
            dtype=rho_reference.dtype,
        )[None, :, None]
        sorted_values = torch.minimum(
            torch.clamp(
                target_sums[:, None, :]
                - ranks * density_cap[None, None, :],
                min=0.0,
            ),
            density_cap[None, None, :],
        )
        concentration_order = torch.argsort(selection_scores, dim=-2)
        trial_density = torch.zeros_like(rho_reference).scatter(
            dim=-2,
            index=concentration_order,
            src=sorted_values,
        )

        fractions = self.mixing_fractions.to(rho_reference)
        n_candidates = fractions.numel()
        candidate_density = (
            (1.0 - fractions[None, :, None, None])
            * rho_reference[:, None, :, :]
            + fractions[None, :, None, None]
            * trial_density[:, None, :, :]
        )
        candidate_density = _correct_component_grid_sums(
            candidate_density,
            target_sums,
            density_cap,
        )
        accumulation_dtype = _accumulation_dtype(rho_reference.dtype)
        particle_number_error = torch.amax(
            torch.abs(
                torch.sum(
                    candidate_density.to(accumulation_dtype),
                    dim=-2,
                )
                - target_sums[:, None, :].to(accumulation_dtype)
            )
        )
        if particle_number_error.item() > 1.0e-4:
            raise RuntimeError(
                "density perturbations failed to preserve particle numbers"
            )

        candidate_data = {
            "rho": candidate_density,
            "local_density_index": batch["local_density_index"]
            .detach()
            .unsqueeze(1)
            .expand(-1, n_candidates, -1, -1),
            "temperature": batch["temperature"]
            .detach()
            .unsqueeze(1)
            .expand(-1, n_candidates),
            "grid_spacing": batch["grid_spacing"]
            .detach()
            .unsqueeze(1)
            .expand(-1, n_candidates, -1),
        }
        candidate_outputs = model(candidate_data, compute_c1=False)

        cell_volume = torch.prod(
            batch["grid_spacing"].detach().to(rho_reference),
            dim=-1,
        )
        beta = batch["beta"].detach().to(rho_reference)
        wavelength = model.thermal_wavelength.detach().to(rho_reference)
        if wavelength.shape != (n_types,):
            raise ValueError(
                "model thermal_wavelength must contain one value per type"
            )
        reference_objective = (
            _ideal_free_energy(rho_reference, wavelength, cell_volume)
            + outputs["beta_F_exc"]
            + cell_volume
            * torch.sum(
                rho_reference * beta[:, None, None] * V_ext,
                dim=(-2, -1),
            )
        )
        candidate_objective = (
            _ideal_free_energy(
                candidate_density,
                wavelength,
                cell_volume[:, None].expand(-1, n_candidates),
            )
            + candidate_outputs["beta_F_exc"]
            + cell_volume[:, None]
            * torch.sum(
                candidate_density
                * beta[:, None, None, None]
                * V_ext[:, None, :, :],
                dim=(-2, -1),
            )
        )
        total_particles = (
            cell_volume * torch.sum(rho_reference, dim=(-2, -1))
        )
        objective_decrease_per_particle = (
            reference_objective[:, None] - candidate_objective
        ) / total_particles[:, None]
        hinge = torch.relu(
            objective_decrease_per_particle - self.energy_tolerance
        )
        return self.weight * torch.mean(hinge)


class Loss(nn.Module):
    """Aggregate named scalar loss terms into one training objective.

    Each registered term must be an ``nn.Module`` with a unique string
    ``name`` attribute and must return one scalar tensor from
    ``term(outputs, batch)``. This permits specialized future terms to coexist
    with :class:`TensorLoss` without changing the trainer.
    """

    def __init__(self, terms: Sequence[nn.Module]) -> None:
        super().__init__()

        terms = list(terms)
        if not terms:
            raise ValueError("Loss requires at least one loss term")

        names = []
        for term in terms:
            if not isinstance(term, nn.Module):
                raise TypeError("every loss term must be a torch.nn.Module")
            name = _validate_name(getattr(term, "name", None))
            if name == "total":
                raise ValueError("'total' is reserved for the aggregate loss")
            if name in names:
                raise ValueError("loss term names must be unique")
            names.append(name)

        self.terms = nn.ModuleList(terms)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return weighted named terms and their scalar sum as ``total``."""

        values = {}
        total = None
        for term in self.terms:
            if getattr(term, "requires_model", False):
                value = term(outputs, batch, model=model)
            else:
                value = term(outputs, batch)
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError(
                    "loss term '{}' must return one scalar tensor".format(
                        term.name
                    )
                )
            values[term.name] = value
            total = value if total is None else total + value

        values["total"] = total
        return values


def _validate_name(name: Optional[str]) -> str:
    """Return a nonempty loss-term name."""

    if not isinstance(name, str) or not name:
        raise ValueError("loss term name must be a nonempty string")
    return name


def _validate_key(key: str, field: str) -> str:
    """Return a nonempty prediction or target key."""

    if not isinstance(key, str) or not key:
        raise ValueError("{} must be a nonempty string".format(field))
    return key


def _finite_nonnegative_scalar(value: float, name: str) -> float:
    """Return a validated finite nonnegative scalar."""

    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a finite nonnegative scalar".format(name))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("{} must be a finite nonnegative scalar".format(name))
    return value


def _ideal_free_energy(
    rho: torch.Tensor,
    thermal_wavelength: torch.Tensor,
    cell_volume: torch.Tensor,
) -> torch.Tensor:
    """Return beta*F_id for fields with arbitrary leading batch dimensions."""

    wavelength_shape = [1] * (rho.ndim - 1) + [rho.shape[-1]]
    dimensionless_density = (
        rho * thermal_wavelength.reshape(wavelength_shape).pow(3)
    )
    logarithm = torch.log(
        torch.clamp(
            dimensionless_density,
            min=torch.finfo(rho.dtype).tiny,
        )
    )
    ideal_density = torch.where(
        rho > 0.0,
        rho * (logarithm - 1.0),
        torch.zeros_like(rho),
    )
    return cell_volume * torch.sum(ideal_density, dim=(-2, -1))


def _correct_component_grid_sums(
    rho: torch.Tensor,
    target_sums: torch.Tensor,
    maximum_density: torch.Tensor,
) -> torch.Tensor:
    """Correct float summation error without violating positivity or the cap."""

    accumulation_dtype = _accumulation_dtype(rho.dtype)
    current_sums = torch.sum(rho.to(accumulation_dtype), dim=-2)
    correction = (
        target_sums[:, None, :].to(accumulation_dtype) - current_sums
    ).to(rho.dtype)
    addition_index = torch.argmax(
        maximum_density[None, None, None, :] - rho,
        dim=-2,
        keepdim=True,
    )
    subtraction_index = torch.argmax(rho, dim=-2, keepdim=True)
    correction_index = torch.where(
        correction[:, :, None, :] >= 0.0,
        addition_index,
        subtraction_index,
    )
    return rho.scatter_add(
        dim=-2,
        index=correction_index,
        src=correction[:, :, None, :],
    )


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use double precision for reductions of low-precision grid fields."""

    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float64
    return dtype
