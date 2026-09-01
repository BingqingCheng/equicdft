"""Physics-based stability objectives for learned density functionals."""

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from ._argument_checks import (
    boolean,
    finite_scalar,
    nonempty_string,
    nonnegative_integer,
    nonnegative_scalar,
)
from ._fourier import (
    canonical_grid_mode as _canonical_grid_mode,
    feasible_modes as _feasible_modes,
    fourier_directions as _fourier_directions,
    projected_fourier_curvature as _normalized_fourier_curvature,
    wavevector_magnitude as _wavevector_magnitude,
)


class FourierStabilityLoss(nn.Module):
    r"""Penalize negative fixed-particle-number Fourier curvature.

    For each nonzero integer reciprocal-grid mode, cosine and sine waves are
    constructed on the periodic grid. A fixed-number base direction is built
    for each density component according to

    ``delta_rho_a = epsilon * rho_a * (wave - <wave>_rho_a)``,

    so the particle number of every component is unchanged. The selected
    mixture mode either keeps these component directions separate or combines
    them. Symmetric evaluations at ``rho +/- delta_rho`` estimate the projected
    curvature of the total intrinsic dimensionless free energy
    ``beta * (F_id + F_exc)``. External-potential and reservoir terms are
    linear in density and therefore cancel from the second difference.

    For a homogeneous one-component fluid, the normalized curvature tends to

    ``rho * delta^2(beta*F) / (DeltaV * sum(delta_rho**2)) = 1 / S(k)``.

    For mixtures, ``mixture_mode`` selects the component-space direction.
    ``"independent"`` averages independently perturbed physical components,
    ``"total_density"`` perturbs every component in phase, and ``"charge"``
    weights the component perturbations by explicit charges. The latter two
    modes probe coupled directions of the component-space Hessian. For a
    symmetric binary mixture with equal component densities and charges
    ``(+1, -1)``, they are the number-number and charge-charge directions.

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
    wavevector_range
        Optional inclusive ``(minimum, maximum)`` magnitude used to restrict
        random mode sampling. Values use the reciprocal units implied by
        ``grid_spacing``. It cannot be combined with explicit ``modes``.
    relative_amplitude
        Maximum pointwise fractional change of the perturbed component after
        its fixed-number projection. It must lie strictly between zero and one.
    minimum_curvature
        Smallest accepted normalized curvature. Zero penalizes only locally
        unstable directions.
    weight
        Nonnegative multiplier applied after averaging the squared hinge over
        fields, modes, real phases, and selected mixture directions.
    training_only
        Return an exact zero during evaluation. This keeps validation model
        selection tied to the data objective rather than a random regularizer.
    name
        Unique name used by :class:`equicdft.loss.Loss`.
    mixture_mode
        Component-space perturbation. It must be ``"independent"``,
        ``"total_density"``, or ``"charge"``. The default preserves the
        original independent-component behavior.
    charges
        Explicit finite charge weight for every density component. Required
        only by ``mixture_mode="charge"``. A common scale is immaterial because
        the weights are normalized by their largest absolute value.
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
        mixture_mode: str = "independent",
        charges: Optional[Sequence[float]] = None,
        wavevector_range: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()

        self.name = nonempty_string(name, "name")
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

        random_modes_per_field = nonnegative_integer(
            random_modes_per_field,
            "random_modes_per_field",
        )
        if modes is None and random_modes_per_field == 0:
            raise ValueError(
                "supply modes or a positive random_modes_per_field"
            )
        if modes is not None and random_modes_per_field != 0:
            raise ValueError(
                "random_modes_per_field must be zero when modes are supplied"
            )
        if wavevector_range is None:
            selected_wavevector_range = None
        else:
            if modes is not None:
                raise ValueError(
                    "wavevector_range cannot be combined with explicit modes"
                )
            try:
                wavevector_limits = tuple(wavevector_range)
            except TypeError as error:
                raise ValueError(
                    "wavevector_range must contain two values"
                ) from error
            if len(wavevector_limits) != 2:
                raise ValueError("wavevector_range must contain two values")
            minimum_wavevector = finite_scalar(
                wavevector_limits[0],
                "wavevector_range minimum",
            )
            maximum_wavevector = finite_scalar(
                wavevector_limits[1],
                "wavevector_range maximum",
            )
            if (
                minimum_wavevector < 0.0
                or maximum_wavevector <= 0.0
                or minimum_wavevector > maximum_wavevector
            ):
                raise ValueError(
                    "wavevector_range must satisfy 0 <= minimum <= maximum "
                    "and maximum > 0"
                )
            selected_wavevector_range = (
                minimum_wavevector,
                maximum_wavevector,
            )
        training_only = boolean(training_only, "training_only")
        mixture_mode = nonempty_string(mixture_mode, "mixture_mode")
        if mixture_mode not in ("independent", "total_density", "charge"):
            raise ValueError(
                "mixture_mode must be 'independent', 'total_density', or "
                "'charge'"
            )
        if mixture_mode == "charge":
            if charges is None:
                raise ValueError("charges are required for charge mixture_mode")
            charge_tensor = torch.as_tensor(
                charges,
                dtype=torch.get_default_dtype(),
            ).detach().clone().reshape(-1)
            if charge_tensor.numel() == 0:
                raise ValueError("charges must not be empty")
            if not torch.all(torch.isfinite(charge_tensor)).item():
                raise ValueError("charges must be finite")
            if not torch.any(charge_tensor != 0.0).item():
                raise ValueError("charges must contain a nonzero value")
        else:
            if charges is not None:
                raise ValueError("charges require charge mixture_mode")
            charge_tensor = None

        relative_amplitude = finite_scalar(
            relative_amplitude,
            "relative_amplitude",
        )
        if not 0.0 < relative_amplitude < 1.0:
            raise ValueError("relative_amplitude must lie in (0, 1)")

        self.relative_amplitude = relative_amplitude
        self.random_modes_per_field = random_modes_per_field
        self.wavevector_range = selected_wavevector_range
        self.training_only = training_only
        self.mixture_mode = mixture_mode
        self.minimum_curvature = nonnegative_scalar(
            minimum_curvature,
            "minimum_curvature",
        )
        self.weight = nonnegative_scalar(weight, "weight")
        self.register_buffer("modes", integer_modes)
        self.register_buffer("charges", charge_tensor)

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
        normalized_curvature, valid = _normalized_fourier_curvature(
            model=model,
            outputs=outputs,
            batch=batch,
            rho=rho,
            directions=directions,
            valid_directions=valid_directions,
            mean_densities=mean_densities,
            relative_amplitude=self.relative_amplitude,
        )
        if not torch.any(valid).item():
            raise ValueError("batch contains no valid mixture-mode direction")
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
        """Return fixed-number mixture directions, validity, and density."""

        n_types = rho.shape[-1]
        mixture_weights = self._mixture_weights(n_types, rho)
        return _fourier_directions(batch, rho, modes, mixture_weights)

    def _mixture_weights(
        self,
        n_types: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return one component-weight row per mixture direction."""

        if self.mixture_mode == "independent":
            return torch.eye(
                n_types,
                device=reference.device,
                dtype=reference.dtype,
            )
        if self.mixture_mode == "total_density":
            return torch.ones(
                (1, n_types),
                device=reference.device,
                dtype=reference.dtype,
            )
        if self.charges.shape != (n_types,):
            raise ValueError("charges must contain one value per density type")
        charges = self.charges.to(reference)
        return (charges / torch.amax(torch.abs(charges)))[None, :]

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
                if self.wavevector_range is not None:
                    box_lengths = tuple(
                        axis_size * axis_spacing
                        for axis_size, axis_spacing in zip(size, spacing)
                    )
                    minimum_wavevector, maximum_wavevector = (
                        self.wavevector_range
                    )
                    candidates = [
                        mode
                        for mode in candidates
                        if minimum_wavevector
                        <= _wavevector_magnitude(mode, box_lengths)
                        <= maximum_wavevector
                    ]
                if self.random_modes_per_field > len(candidates):
                    selection_scope = (
                        " in wavevector_range"
                        if self.wavevector_range is not None
                        else ""
                    )
                    raise ValueError(
                        "random_modes_per_field exceeds the feasible modes"
                        + selection_scope
                    )
                indices = torch.randperm(len(candidates))[
                    : self.random_modes_per_field
                ].tolist()
                selected = [candidates[index] for index in indices]
            selected_by_field.append(
                torch.tensor(selected, dtype=torch.long)
            )

        return torch.stack(selected_by_field).to(device=rho.device)
