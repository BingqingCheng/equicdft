"""Physics-based stability objectives for learned density functionals."""

from typing import Dict, Optional, Sequence, Union

import torch
from torch import nn

from ._argument_checks import (
    boolean,
    finite_scalar,
    nonempty_string,
    nonnegative_scalar,
    positive_integer,
)
from ._fourier import (
    _validated_grid,
    canonical_wavevector_indices,
    feasible_wavevector_indices as _feasible_wavevector_indices,
    wavevector_index_triplets,
    polarization_fourier_curvature,
    wavevector_magnitude as _wavevector_magnitude,
)
from .response import FourierResponse


class FourierStabilityLoss(nn.Module):
    r"""Penalize negative density or fixed-dipole Fourier curvature.

    Selected wavevectors can be evaluated as independent cosine/sine probes
    or superposed with random phases into one spatial pattern. A fixed-number
    base direction is built for each density component from

    ``delta_rho_a = epsilon * rho_a * (wave - <wave>_rho_a)``,

    so the particle number of every component is unchanged. The selected
    component treatment either keeps these component directions independent
    or combines them. Symmetric evaluations at ``rho +/- delta_rho`` estimate
    the projected curvature of the total intrinsic dimensionless free energy
    ``beta * (F_id + F_exc)``. External-potential and reservoir terms are linear
    in density and therefore cancel from the second difference.

    For a single wavevector in a homogeneous one-component fluid, curvature
    tends to

    ``rho * delta^2(beta*F) / (DeltaV * sum(delta_rho**2)) = 1 / S(k)``.

    For mixtures, ``component_treatment`` selects the component-space treatment.
    ``"independent"`` averages independently perturbed physical components,
    ``"total_density"`` perturbs every component in phase, and ``"charge"``
    weights the component perturbations by explicit charges. The latter two
    choices probe coupled directions of the component-space Hessian. For a
    symmetric binary mixture with equal component densities and charges
    ``(+1, -1)``, they are the number-number and charge-charge directions.
    ``"full_matrix"`` reconstructs the complete physical-component Hessian
    in the ideal-gas metric and penalizes every eigenvalue below the requested
    minimum. At a single wavevector in a homogeneous field this is the inverse
    OZ response matrix. A superposed pattern instead probes a projected
    Hessian with cross-wavevector contributions in inhomogeneous fields, not
    S(k) at one k or the complete spatial Hessian. Matrix polarization makes
    small finite-difference amplitudes susceptible to floating-point
    cancellation.

    ``variable="dipole_density"`` instead holds density fixed and probes the
    one randomly selected polarization direction ``u`` per field, drawn
    uniformly on the unit sphere. Its shared species displacement is

    ``delta_P_a = epsilon * m_a * rho_a * wave * u``,

    so ``epsilon`` is a dimensionless change in local alignment fraction.
    The same ``u`` is used for all wavevectors and phases of one field.
    The selected direction is normalized by the isotropic small-alignment
    fixed-dipole ideal curvature, so the ideal reference has unit curvature.
    The probe does not impose the local ``|P| < m*rho`` bound; this keeps the
    stochastic regularizer usable with gridded reference fields whose local
    polarization can reach that sampling bound. Polarization has no fixed-
    integral constraint, and its zero wavevector is included by default.

    Parameters
    ----------
    wavevector_indices
        Fixed nonzero integer triplets ``(nx, ny, nz)``. Supply either this or
        ``random_wavevectors_per_field``, but not both.
    random_wavevectors_per_field
        Number of distinct reciprocal triplets sampled per field from
        ``wavevector_domain``. An integer pair ``(lower, upper)`` draws an
        inclusive uniform count once per batch.
        Both endpoints must be positive; the upper endpoint must fit every
        field's feasible set. Equal endpoints fix the number of selected waves
        without a count draw, identical to the corresponding integer argument.
        Supply either this or ``wavevector_indices``, but not both.
    wavevector_treatment
        ``"independent"`` evaluates the cosine and sine wave for every selected
        triplet as independent curvature probes. ``"superposition"`` assigns
        every selected triplet an independent uniform phase in ``[0, 2*pi)``
        and sums them into one spatial pattern before projection and
        normalization. For polarization, the optional zero wavevector
        remains an independent probe. The default is ``"independent"``.
    wavevector_domain
        ``"sphere"`` keeps the physical isotropic Nyquist sphere (default).
        ``"cube"`` includes every grid-representable wavevector, bounded
        separately by each axis's Nyquist limit. This includes high-wavevector
        corners omitted by the sphere. Both domains remove the zero
        wavevector, global-sign duplicates, and equivalent signs of even-grid
        Nyquist components.
    wavevector_range
        Optional inclusive ``(minimum, maximum)`` magnitude used to restrict
        random wavevector sampling. Values use the reciprocal units implied by
        ``grid_spacing``. It cannot be combined with explicit
        ``wavevector_indices``.
    relative_amplitude
        For density, the maximum pointwise fractional change after the
        fixed-number projection. For polarization, the requested maximum
        change in ``P/(m*rho)``. The stability probe does not constrain this
        local alignment fraction. A scalar preserves fixed-amplitude
        evaluation. A pair ``(lower, upper)`` samples uniformly per field and
        spatial pattern on every call, with ``0 < lower <= upper < 1``.
        Cosine/sine phases share one draw for each explicit wavevector.
        Equal endpoints behave as a scalar and consume no random draws.
    minimum_curvature
        Smallest accepted normalized curvature. Zero penalizes only locally
        unstable directions.
    weight
        Nonnegative multiplier applied after averaging the squared hinge over
        fields, wavevectors, real phases, and selected component directions.
    training_only
        Return an exact zero during evaluation. This keeps validation model
        selection tied to the data objective rather than a random regularizer.
    name
        Unique name used by :class:`equicdft.loss.Loss`.
    component_treatment
        Component-space treatment. It must be ``"independent"``,
        ``"total_density"``, ``"charge"``, or ``"full_matrix"``. The default
        preserves the original independent-component density behavior.
        Polarization currently accepts only ``"independent"`` because it
        samples one shared three-dimensional polarization direction.
    charges
        Explicit finite charge weight for every density component. Required
        only by ``component_treatment="charge"``. A common scale is immaterial
        because the weights are normalized by their largest absolute value.
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
    include_zero_wavevector
        Whether to add the uniform polarization wavevector. It defaults to
        true for polarization and false for density. Density cannot include
        the zero wavevector because its perturbations preserve particle number.
    """

    requires_model = True

    def __init__(
        self,
        *,
        wavevector_indices: Optional[Sequence[Sequence[int]]] = None,
        random_wavevectors_per_field: Optional[
            Union[int, Sequence[int]]
        ] = None,
        wavevector_treatment: str = "independent",
        relative_amplitude: Union[float, Sequence[float]] = 0.05,
        minimum_curvature: float = 0.0,
        weight: float = 1.0,
        training_only: bool = True,
        name: str = "fourier_stability",
        component_treatment: str = "independent",
        charges: Optional[Sequence[float]] = None,
        wavevector_range: Optional[Sequence[float]] = None,
        perturbations_per_forward: Optional[int] = None,
        wavevector_domain: str = "sphere",
        variable: str = "rho",
        dipole_magnitude: Optional[Union[float, Sequence[float]]] = None,
        include_zero_wavevector: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.name = nonempty_string(name, "name")
        variable = nonempty_string(variable, "variable")
        if variable not in ("rho", "dipole_density"):
            raise ValueError("variable must be 'rho' or 'dipole_density'")
        self.variable = variable
        if include_zero_wavevector is None:
            include_zero_wavevector = variable == "dipole_density"
        self.include_zero_wavevector = boolean(
            include_zero_wavevector,
            "include_zero_wavevector",
        )
        if variable == "rho" and self.include_zero_wavevector:
            raise ValueError(
                "density stability cannot include the zero wavevector"
            )
        wavevector_treatment = nonempty_string(
            wavevector_treatment,
            "wavevector_treatment",
        )
        if wavevector_treatment not in ("independent", "superposition"):
            raise ValueError(
                "wavevector_treatment must be 'independent' or "
                "'superposition'"
            )
        self.wavevector_treatment = wavevector_treatment

        if (wavevector_indices is None) == (
            random_wavevectors_per_field is None
        ):
            raise ValueError(
                "supply exactly one of wavevector_indices or "
                "random_wavevectors_per_field"
            )
        if wavevector_indices is None:
            selected_indices = torch.empty((0, 3), dtype=torch.long)
        else:
            selected_indices = wavevector_index_triplets(
                wavevector_indices,
                name="wavevector_indices",
            )

        if isinstance(random_wavevectors_per_field, (tuple, list)):
            if len(random_wavevectors_per_field) != 2:
                raise ValueError(
                    "random_wavevectors_per_field interval must have two "
                    "endpoints"
                )
            lower, upper = (
                positive_integer(
                    value,
                    "random_wavevectors_per_field endpoint",
                )
                for value in random_wavevectors_per_field
            )
            if not 1 <= lower <= upper:
                raise ValueError(
                    "random_wavevectors_per_field requires "
                    "1 <= lower <= upper"
                )
            random_wavevectors_per_field = (
                lower if lower == upper else (lower, upper)
            )
        elif random_wavevectors_per_field is not None:
            random_wavevectors_per_field = positive_integer(
                random_wavevectors_per_field,
                "random_wavevectors_per_field",
            )
        if wavevector_range is None:
            selected_wavevector_range = None
        else:
            if wavevector_indices is not None:
                raise ValueError(
                    "wavevector_range cannot be combined with explicit "
                    "wavevector_indices"
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
        component_treatment = nonempty_string(
            component_treatment,
            "component_treatment",
        )
        if component_treatment not in (
            "independent",
            "total_density",
            "charge",
            "full_matrix",
        ):
            raise ValueError(
                "component_treatment must be 'independent', 'total_density', "
                "'charge', or 'full_matrix'"
            )
        if (
            variable == "dipole_density"
            and component_treatment != "independent"
        ):
            raise ValueError(
                "dipole-density stability currently requires "
                "component_treatment='independent'"
            )
        if component_treatment == "charge":
            if charges is None:
                raise ValueError(
                    "charges are required for charge component_treatment"
                )
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
                raise ValueError("charges require charge component_treatment")
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
                raise ValueError(
                    "relative_amplitude interval must have two endpoints"
                )
            lower, upper = (
                finite_scalar(value, "relative_amplitude endpoint")
                for value in relative_amplitude
            )
            if not 0.0 < lower <= upper < 1.0:
                raise ValueError(
                    "relative_amplitude requires 0 < lower <= upper < 1"
                )
            relative_amplitude = lower if lower == upper else (lower, upper)
        else:
            relative_amplitude = finite_scalar(
                relative_amplitude,
                "relative_amplitude",
            )
            if not 0.0 < relative_amplitude < 1.0:
                raise ValueError("relative_amplitude must lie in (0, 1)")

        self.relative_amplitude = relative_amplitude
        self.response = FourierResponse(
            relative_amplitude=(
                relative_amplitude[0]
                if isinstance(relative_amplitude, tuple) else relative_amplitude
            ),
            perturbations_per_forward=perturbations_per_forward,
            wavevector_domain=wavevector_domain,
        )
        self.wavevector_domain = self.response.wavevector_domain
        self.random_wavevectors_per_field = random_wavevectors_per_field
        self.wavevector_range = selected_wavevector_range
        self.training_only = training_only
        self.component_treatment = component_treatment
        self.minimum_curvature = nonnegative_scalar(
            minimum_curvature,
            "minimum_curvature",
        )
        self.weight = nonnegative_scalar(weight, "weight")
        self.register_buffer("wavevector_indices", selected_indices)
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

        wavevector_indices = self._select_wavevector_indices(batch, rho)
        if self.variable == "dipole_density":
            return self._polarization_loss(
                model,
                outputs,
                batch,
                rho,
                wavevector_indices,
            )

        wavevector_phases, amplitude = self._sample_wavevector_probe(
            wavevector_indices,
            rho,
        )
        if self.component_treatment == "full_matrix":
            matrix, active = self.response.matrix(
                model=model,
                batch=batch,
                wavevector_indices=wavevector_indices,
                outputs=outputs,
                relative_amplitude=amplitude,
                wavevector_phases=wavevector_phases,
            )
            return self._matrix_loss(matrix, active)

        component_weights = self._component_weights(rho.shape[-1], rho)
        normalized_curvature, valid = self.response(
            model=model,
            batch=batch,
            wavevector_indices=wavevector_indices,
            directions=component_weights,
            outputs=outputs,
            relative_amplitude=amplitude,
            wavevector_phases=wavevector_phases,
        )
        return self._directional_loss(
            normalized_curvature,
            valid,
            "batch contains no valid component direction",
        )

    def _polarization_loss(self, model, outputs, batch, rho, wavevector_indices):
        """Return the fixed-density polarization stability penalty."""

        directions = torch.randn(
            (rho.shape[0], 3), dtype=rho.dtype, device=rho.device,
        )
        directions /= torch.linalg.vector_norm(
            directions, dim=-1, keepdim=True,
        )
        common = {
            "model": model,
            "outputs": outputs,
            "batch": batch,
            "rho": rho,
            "dipole_magnitude": self.dipole_magnitude,
            "polarization_directions": directions,
            "perturbations_per_forward": self.response.perturbations_per_forward,
        }
        if self.wavevector_treatment == "independent":
            if self.include_zero_wavevector:
                wavevector_indices = torch.cat(
                    (
                        torch.zeros_like(wavevector_indices[:, :1]),
                        wavevector_indices,
                    ),
                    dim=1,
                )
            _, amplitude = self._sample_wavevector_probe(
                wavevector_indices,
                rho,
            )
            curvature, valid = polarization_fourier_curvature(
                wavevector_indices=wavevector_indices,
                relative_amplitude=amplitude,
                **common,
            )
        else:
            phases, amplitude = self._sample_wavevector_probe(
                wavevector_indices,
                rho,
            )
            curvature, valid = polarization_fourier_curvature(
                wavevector_indices=wavevector_indices,
                relative_amplitude=amplitude,
                wavevector_phases=phases,
                **common,
            )
            if self.include_zero_wavevector:
                zero_indices = torch.zeros_like(wavevector_indices[:, :1])
                zero_curvature, zero_valid = polarization_fourier_curvature(
                    wavevector_indices=zero_indices,
                    relative_amplitude=self._sample_amplitude(
                        zero_indices.shape[:2], rho
                    ),
                    **common,
                )
                curvature = torch.cat(
                    (zero_curvature.flatten(1), curvature.flatten(1)), dim=1
                )
                valid = torch.cat(
                    (zero_valid.flatten(1), valid.flatten(1)), dim=1
                )
        return self._directional_loss(
            curvature,
            valid,
            "batch contains no valid polarization direction",
        )

    def _sample_wavevector_probe(self, wavevector_indices, reference):
        """Return phases and amplitudes for the selected probe treatment."""

        if self.wavevector_treatment == "independent":
            return (
                None,
                self._sample_amplitude(
                    wavevector_indices.shape[:2],
                    reference,
                ),
            )
        phases = reference.new_empty(
            wavevector_indices.shape[:2]
        ).uniform_(0.0, 2.0 * torch.pi)
        amplitude = self._sample_amplitude(
            (reference.shape[0], 1),
            reference,
        )
        return phases, amplitude

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

    def _component_weights(
        self,
        n_types: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return one component-weight row per selected treatment."""

        if self.component_treatment == "independent":
            return torch.eye(
                n_types,
                device=reference.device,
                dtype=reference.dtype,
            )
        if self.component_treatment == "total_density":
            return torch.ones(
                (1, n_types),
                device=reference.device,
                dtype=reference.dtype,
            )
        if self.component_treatment == "full_matrix":
            raise RuntimeError("full_matrix does not use component weights")
        if self.charges.shape != (n_types,):
            raise ValueError("charges must contain one value per density type")
        charges = self.charges.to(reference)
        return (charges / torch.amax(torch.abs(charges)))[None, :]

    def _select_wavevector_indices(
        self,
        batch: Dict[str, torch.Tensor],
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Return fixed or randomly sampled indices for every field."""

        n_fields = rho.shape[0]
        grid_size, grid_spacing = _validated_grid(
            batch,
            n_fields,
            n_grid=rho.shape[1],
        )

        count = maximum_count = self.random_wavevectors_per_field
        if isinstance(count, tuple):
            lower, maximum_count = count
            # One common count keeps the response tensor rectangular.
            # Wavevector identities remain independent between fields.
            count = (
                lower if lower == maximum_count else
                int(torch.randint(lower, maximum_count + 1, ()).item())
            )

        selected_by_field = []
        for field in range(n_fields):
            size = tuple(grid_size[field].tolist())
            spacing = tuple(grid_spacing[field].tolist())
            if self.wavevector_indices.shape[0] > 0:
                selected = canonical_wavevector_indices(
                    self.wavevector_indices,
                    size,
                    spacing,
                    wavevector_domain=self.wavevector_domain,
                ).detach().cpu().tolist()
            else:
                candidates = _feasible_wavevector_indices(
                    size,
                    spacing,
                    wavevector_domain=self.wavevector_domain,
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
                        index_triplet
                        for index_triplet in candidates
                        if minimum_wavevector
                        <= _wavevector_magnitude(index_triplet, box_lengths)
                        <= maximum_wavevector
                    ]
                if maximum_count > len(candidates):
                    selection_scope = (
                        " in wavevector_range"
                        if self.wavevector_range is not None
                        else ""
                    )
                    raise ValueError(
                        "random_wavevectors_per_field exceeds the feasible "
                        "wavevector indices"
                        + selection_scope
                    )
                indices = torch.randperm(len(candidates))[:count].tolist()
                selected = [candidates[index] for index in indices]
            selected_by_field.append(
                torch.tensor(selected, dtype=torch.long)
            )

        return torch.stack(selected_by_field).to(device=rho.device)
