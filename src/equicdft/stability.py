"""Physics-based stability objectives for learned density functionals."""

from typing import Dict, Optional, Sequence, Union

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
    _validated_grid,
    canonical_mode_triplets,
    feasible_modes as _feasible_modes,
    mode_triplets,
    polarization_fourier_curvature,
    wavevector_magnitude as _wavevector_magnitude,
)
from .response import FourierResponse


class FourierStabilityLoss(nn.Module):
    r"""Penalize negative density or fixed-dipole Fourier curvature.

    In density mode, a single mode uses separate cosine and sine waves;
    multiple randomly sampled modes are summed with random phases into one
    spatial pattern. A fixed-number base direction is built for each density
    component from

    ``delta_rho_a = epsilon * rho_a * (wave - <wave>_rho_a)``,

    so the particle number of every component is unchanged. The selected
    mixture mode either keeps these component directions separate or combines
    them. Symmetric evaluations at ``rho +/- delta_rho`` estimate the projected
    curvature of the total intrinsic dimensionless free energy
    ``beta * (F_id + F_exc)``. External-potential and reservoir terms are
    linear in density and therefore cancel from the second difference.

    For a single mode in a homogeneous one-component fluid, curvature tends to

    ``rho * delta^2(beta*F) / (DeltaV * sum(delta_rho**2)) = 1 / S(k)``.

    For mixtures, ``mixture_mode`` selects the component-space treatment.
    ``"independent"`` averages independently perturbed physical components,
    ``"total_density"`` perturbs every component in phase, and ``"charge"``
    weights the component perturbations by explicit charges. The latter two
    modes probe coupled directions of the component-space Hessian. For a
    symmetric binary mixture with equal component densities and charges
    ``(+1, -1)``, they are the number-number and charge-charge directions.
    ``"full_matrix"`` reconstructs the complete physical-component Hessian
    in the ideal-gas metric and penalizes every eigenvalue below the requested
    minimum. At a single mode in a homogeneous field this is the inverse OZ
    response matrix. A summed pattern instead probes a projected Hessian with
    cross-wavevector contributions in inhomogeneous fields, not S(k) at one k
    or the complete spatial Hessian. Matrix polarization makes small finite-
    difference amplitudes susceptible to floating-point cancellation.

    ``variable="dipole_density"`` instead holds density fixed and probes the
    one randomly selected polarization direction ``u`` per field, drawn
    uniformly on the unit sphere. Its shared species displacement is

    ``delta_P_a = epsilon * m_a * rho_a * wave * u``,

    so ``epsilon`` is a dimensionless change in local alignment fraction.
    Every selected polarization wavevector keeps its cosine and sine probes
    separate; random wavevectors are not superposed in polarization mode.
    The same ``u`` is used for all wavevectors and phases of one field.
    The selected direction is normalized by its exact fixed-dipole ideal
    curvature, so the ideal functional has unit curvature. Polarization has no
    fixed-integral constraint, and its zero wavevector is included by default.

    Parameters
    ----------
    modes
        Optional fixed nonzero integer triplets ``(nx, ny, nz)``. When omitted,
        ``random_modes_per_field`` triplets are sampled independently for every
        field and batch.
    random_modes_per_field
        Number of distinct reciprocal triplets sampled per field from
        ``mode_domain``. In density mode they are SUMMED into one pattern; in
        polarization mode each remains a separate cosine/sine probe. An integer pair
        ``(lower, upper)`` draws an inclusive uniform count once per batch.
        In density mode every field uses independent wavevectors and uniform
        phases in [0, 2*pi). The sum is projected to fixed component counts
        and then normalized; one amplitude per field controls the combined perturbation.
        Both endpoints must be positive; the upper endpoint must fit every
        field's feasible set. Equal endpoints fix the number of summed waves
        without a count draw, identical to the corresponding integer argument.
        A selected count of one retains the original separate cosine/sine
        evaluation; counts of two or more use one summed spatial pattern.
        Must be zero when explicit ``modes`` are supplied.
    mode_domain
        ``"sphere"`` keeps the physical isotropic Nyquist sphere (default).
        ``"cube"`` includes every grid-representable mode, bounded separately
        by each axis's Nyquist limit. This includes high-wavevector corners
        omitted by the sphere. Both domains remove the zero mode, global-sign
        duplicates, and equivalent signs of even-grid Nyquist components.
    wavevector_range
        Optional inclusive ``(minimum, maximum)`` magnitude used to restrict
        random mode sampling. Values use the reciprocal units implied by
        ``grid_spacing``. It cannot be combined with explicit ``modes``.
    relative_amplitude
        In density mode, the maximum pointwise fractional change after the
        fixed-number projection. In polarization mode, the requested maximum
        change in ``P/(m*rho)``; it is reduced if needed to keep both symmetric
        probes inside ``|P| < m*rho``. A scalar preserves fixed-amplitude
        evaluation. A pair ``(lower, upper)`` samples uniformly per field and
        spatial pattern on every call, with ``0 < lower <= upper < 1``.
        Cosine/sine phases share one draw for each explicit wavevector.
        Equal endpoints behave as a scalar and consume no random draws.
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
        Component-space treatment. It must be ``"independent"``,
        ``"total_density"``, ``"charge"``, or ``"full_matrix"``. The default
        preserves the original independent-component density behavior.
        Polarization mode currently accepts only ``"independent"`` because it
        samples one shared three-dimensional polarization direction.
    charges
        Explicit finite charge weight for every density component. Required
        only by ``mixture_mode="charge"``. A common scale is immaterial because
        the weights are normalized by their largest absolute value.
    perturbations_per_forward
        Optional maximum number of perturbed fields evaluated together. This
        can limit memory use for the quadratic number of full-matrix probes.
    variable
        ``"rho"`` (default) preserves the fixed-particle-number density loss.
        ``"dipole_density"`` probes one random unit-vector polarization
        curvature per field at fixed density.
    dipole_magnitude
        Positive scalar or one value per species. Required only for
        ``variable="dipole_density"``. This is the physical fixed molecular
        dipole magnitude, not a statistical loss weight.
    include_zero_mode
        Whether to add the uniform polarization mode. It defaults to true for
        polarization and false for density. Density cannot include the zero
        mode because its perturbations preserve particle number.
    """

    requires_model = True

    def __init__(
        self,
        modes: Optional[Sequence[Sequence[int]]] = None,
        random_modes_per_field: Union[int, Sequence[int]] = 0,
        relative_amplitude: Union[float, Sequence[float]] = 0.05,
        minimum_curvature: float = 0.0,
        weight: float = 1.0,
        training_only: bool = True,
        name: str = "fourier_stability",
        mixture_mode: str = "independent",
        charges: Optional[Sequence[float]] = None,
        wavevector_range: Optional[Sequence[float]] = None,
        perturbations_per_forward: Optional[int] = None,
        mode_domain: str = "sphere",
        variable: str = "rho",
        dipole_magnitude: Optional[Union[float, Sequence[float]]] = None,
        include_zero_mode: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.name = nonempty_string(name, "name")
        variable = nonempty_string(variable, "variable")
        if variable not in ("rho", "dipole_density"):
            raise ValueError("variable must be 'rho' or 'dipole_density'")
        self.variable = variable
        if include_zero_mode is None:
            include_zero_mode = variable == "dipole_density"
        self.include_zero_mode = boolean(include_zero_mode, "include_zero_mode")
        if variable == "rho" and self.include_zero_mode:
            raise ValueError("density stability cannot include the zero mode")

        if modes is None:
            integer_modes = torch.empty((0, 3), dtype=torch.long)
        else:
            integer_modes = mode_triplets(modes)

        if isinstance(random_modes_per_field, (tuple, list)):
            if len(random_modes_per_field) != 2:
                raise ValueError("random_modes_per_field interval must have two endpoints")
            lower, upper = (
                nonnegative_integer(value, "random_modes_per_field endpoint")
                for value in random_modes_per_field
            )
            if not 1 <= lower <= upper:
                raise ValueError("random_modes_per_field requires 1 <= lower <= upper")
            random_modes_per_field = lower if lower == upper else (lower, upper)
        else:
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
        if mixture_mode not in (
            "independent",
            "total_density",
            "charge",
            "full_matrix",
        ):
            raise ValueError(
                "mixture_mode must be 'independent', 'total_density', "
                "'charge', or 'full_matrix'"
            )
        if variable == "dipole_density" and mixture_mode != "independent":
            raise ValueError(
                "dipole-density stability currently requires "
                "mixture_mode='independent'"
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

        if variable == "rho":
            if dipole_magnitude is not None:
                raise ValueError(
                    "dipole_magnitude requires variable='dipole_density'"
                )
            dipole_tensor = None
        else:
            if dipole_magnitude is None:
                raise ValueError(
                    "dipole_magnitude is required for dipole-density stability"
                )
            dipole_tensor = (
                dipole_magnitude
                if isinstance(dipole_magnitude, torch.Tensor)
                else torch.as_tensor(dipole_magnitude, dtype=torch.float64)
            )
            if (
                dipole_tensor.dtype == torch.bool
                or torch.is_complex(dipole_tensor)
            ):
                raise ValueError("dipole_magnitude must be finite and positive")
            if not dipole_tensor.is_floating_point():
                dipole_tensor = dipole_tensor.to(torch.get_default_dtype())
            if (
                dipole_tensor.ndim > 1
                or dipole_tensor.numel() == 0
                or not torch.all(torch.isfinite(dipole_tensor)).item()
                or torch.any(dipole_tensor <= 0.0).item()
            ):
                raise ValueError("dipole_magnitude must be finite and positive")
            dipole_tensor = dipole_tensor.reshape(-1)

        if isinstance(relative_amplitude, (tuple, list)):
            if len(relative_amplitude) != 2:
                raise ValueError("relative_amplitude interval must have two endpoints")
            lower, upper = (
                finite_scalar(value, "relative_amplitude endpoint")
                for value in relative_amplitude
            )
            if not 0.0 < lower <= upper < 1.0:
                raise ValueError("relative_amplitude requires 0 < lower <= upper < 1")
            relative_amplitude = lower if lower == upper else (lower, upper)
        else:
            relative_amplitude = finite_scalar(relative_amplitude, "relative_amplitude")
            if not 0.0 < relative_amplitude < 1.0:
                raise ValueError("relative_amplitude must lie in (0, 1)")

        self.relative_amplitude = relative_amplitude
        self.response = FourierResponse(
            relative_amplitude=(
                relative_amplitude[0]
                if isinstance(relative_amplitude, tuple) else relative_amplitude
            ),
            perturbations_per_forward=perturbations_per_forward,
            mode_domain=mode_domain,
        )
        self.mode_domain = self.response.mode_domain
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
        self.register_buffer("dipole_magnitude", dipole_tensor)

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
        variable = getattr(self, "variable", "rho")
        if variable == "dipole_density":
            return self._polarization_loss(model, outputs, batch, rho, modes)

        mode_phases = None
        amplitude_shape = modes.shape[:2]
        if self.modes.shape[0] == 0 and modes.shape[1] > 1:
            mode_phases = rho.new_empty(modes.shape[:2]).uniform_(0.0, 2.0 * torch.pi)
            amplitude_shape = (rho.shape[0], 1)
        amplitude = self._sample_amplitude(amplitude_shape, rho)
        if self.mixture_mode == "full_matrix":
            matrix, active = self.response.matrix(
                model=model,
                batch=batch,
                modes=modes,
                outputs=outputs,
                relative_amplitude=amplitude,
                mode_phases=mode_phases,
            )
            return self._matrix_loss(matrix, active)

        mixture_weights = self._mixture_weights(rho.shape[-1], rho)
        normalized_curvature, valid = self.response(
            model=model,
            batch=batch,
            modes=modes,
            directions=mixture_weights,
            outputs=outputs,
            relative_amplitude=amplitude,
            mode_phases=mode_phases,
        )
        return self._directional_loss(
            normalized_curvature,
            valid,
            "batch contains no valid mixture-mode direction",
        )

    def _polarization_loss(self, model, outputs, batch, rho, modes):
        """Return the fixed-density polarization stability penalty."""

        if getattr(self, "include_zero_mode", True):
            modes = torch.cat((torch.zeros_like(modes[:, :1]), modes), dim=1)
        directions = torch.randn(
            (rho.shape[0], 3), dtype=rho.dtype, device=rho.device,
        )
        directions /= torch.linalg.vector_norm(
            directions, dim=-1, keepdim=True,
        )
        curvature, valid = polarization_fourier_curvature(
            model=model,
            outputs=outputs,
            batch=batch,
            rho=rho,
            modes=modes,
            dipole_magnitude=self.dipole_magnitude,
            relative_amplitude=self._sample_amplitude(modes.shape[:2], rho),
            polarization_directions=directions,
            perturbations_per_forward=self.response.perturbations_per_forward,
        )
        return self._directional_loss(
            curvature,
            valid,
            "batch contains no valid polarization direction",
        )

    def _sample_amplitude(self, shape, reference):
        """Draw per-pattern amplitudes only for an interval configuration."""

        if isinstance(self.relative_amplitude, tuple):
            return reference.new_empty(shape).uniform_(*self.relative_amplitude)
        return self.relative_amplitude

    def _directional_loss(self, curvature, valid, empty_message):
        """Average the weighted squared hinge over valid directions."""

        if not torch.any(valid).item():
            raise ValueError(empty_message)
        hinge = torch.relu(self.minimum_curvature - curvature).square()
        return self.weight * torch.sum(hinge * valid.to(hinge)) / valid.sum()

    def _matrix_loss(
        self,
        matrix: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean spectral hinge over active component submatrices."""

        n_types = matrix.shape[-1]
        if matrix.shape[-2:] != (n_types, n_types):
            raise ValueError("curvature matrix must be square")
        if active.shape != matrix.shape[:-1]:
            raise ValueError("active mask must match the matrix components")

        penalties = []
        for value, selected in zip(
            matrix.reshape(-1, n_types, n_types),
            active.reshape(-1, n_types),
        ):
            if torch.any(selected).item():
                value = value[selected][:, selected]
                value = 0.5 * (value + value.transpose(-1, -2))
                eigenvalues = torch.linalg.eigvalsh(value)
                penalties.append(
                    torch.relu(
                        self.minimum_curvature - eigenvalues
                    ).square()
                )
        if not penalties:
            raise ValueError("batch contains no valid full-matrix direction")
        return self.weight * torch.cat(penalties).mean()

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
        if self.mixture_mode == "full_matrix":
            raise RuntimeError("full_matrix does not use mixture weights")
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
        grid_size, grid_spacing = _validated_grid(
            batch,
            n_fields,
            n_grid=rho.shape[1],
        )

        count = maximum_count = self.random_modes_per_field
        if isinstance(count, tuple):
            lower, maximum_count = count
            # One common count keeps the response tensor rectangular. Mode
            # identities are still sampled independently for each field.
            count = (
                lower if lower == maximum_count else
                int(torch.randint(lower, maximum_count + 1, ()).item())
            )

        selected_by_field = []
        for field in range(n_fields):
            size = tuple(grid_size[field].tolist())
            spacing = tuple(grid_spacing[field].tolist())
            if self.modes.shape[0] > 0:
                selected = canonical_mode_triplets(
                    self.modes,
                    size,
                    spacing,
                    mode_domain=self.mode_domain,
                ).detach().cpu().tolist()
            else:
                candidates = _feasible_modes(
                    size, spacing, mode_domain=self.mode_domain,
                )
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
                if maximum_count > len(candidates):
                    selection_scope = (
                        " in wavevector_range"
                        if self.wavevector_range is not None
                        else ""
                    )
                    raise ValueError(
                        "random_modes_per_field exceeds the feasible modes"
                        + selection_scope
                    )
                indices = torch.randperm(len(candidates))[:count].tolist()
                selected = [candidates[index] for index in indices]
            selected_by_field.append(
                torch.tensor(selected, dtype=torch.long)
            )

        return torch.stack(selected_by_field).to(device=rho.device)
