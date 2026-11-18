"""Cartesian moment features for density fields on fixed integer grids."""

import copy
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .._validation import (
    boolean,
    nonnegative_integer,
    nonnegative_scalar,
    optional_positive_integer,
    positive_integer,
    positive_scalar,
)
from .._config import Configurable, make_config, register, scalar_value
from .._polarization import (
    polarization_input_name,
    regularized_reduced_polarization,
)
from ..geometry.convolution import (
    gather_neighbors,
    periodic_density_convolution,
)
from .radial import (
    _RadialTransform,
    _cartesian_stencil_basis,
    _conditioned_bessel_stencil_basis,
    prepare_radial_centers,
    prepare_radial_exponents,
)
from ..geometry.stencil import make_stencil

__all__ = [
    "CartesianAFeatures",
]


def _make_powers(max_power: int) -> torch.Tensor:
    """Enumerate ``(a, b, c)`` by increasing total Cartesian power.

    For example, ``max_power=2`` gives the monomials
    ``1, x, y, z, x^2, xy, xz, y^2, yz, z^2``.
    """

    max_power = nonnegative_integer(max_power, "max_power")

    powers = []
    for total_power in range(max_power + 1):
        for x_power in range(total_power, -1, -1):
            remaining_power = total_power - x_power
            for y_power in range(remaining_power, -1, -1):
                z_power = remaining_power - y_power
                powers.append((x_power, y_power, z_power))
    return torch.tensor(powers, dtype=torch.long)


def _normalization_reference_config(reference):
    """Validate a JSON-compatible original stencil record, without fitted state."""
    if reference is None:
        return None
    if not isinstance(reference, Mapping):
        raise TypeError("normalization_reference must be a mapping or None")
    required = {"grid_spacing", "offsets", "neighbor_mask", "cutoff_grid",
                "stencil_radius", "monomial_scale", "coordinate_scaling", "spacing_ratio"}
    if set(reference) != required:
        raise ValueError("normalization_reference must contain exactly {}".format(sorted(required)))
    cutoff = nonnegative_integer(reference["cutoff_grid"], "reference cutoff_grid")
    radius = nonnegative_scalar(reference["stencil_radius"], "reference stencil_radius")
    if radius > cutoff:
        raise ValueError("reference stencil_radius cannot exceed cutoff_grid")
    offsets = torch.as_tensor(reference["offsets"])
    expected = torch.from_numpy(make_stencil(cutoff))
    if (offsets.dtype == torch.bool or offsets.is_complex()
            or offsets.shape != expected.shape or not torch.equal(offsets, expected.to(offsets))):
        raise ValueError("reference offsets must be the canonical integer stencil")
    mask = torch.as_tensor(reference["neighbor_mask"])
    if mask.dtype != torch.bool or mask.shape != expected.shape[:1]:
        raise ValueError("reference neighbor_mask must contain one Boolean per offset")
    squared = expected.square().sum(dim=-1)
    expected_mask = (squared > 0) & (squared <= radius ** 2 + 1e-12 * max(1., radius ** 2))
    if not torch.equal(mask, expected_mask) or not mask.any().item():
        raise ValueError("reference mask must be a nonempty spherical, center-excluded stencil")
    scaling = reference["coordinate_scaling"]
    if scaling not in ("none", "cutoff"):
        raise ValueError("reference coordinate_scaling must be 'none' or 'cutoff'")
    return dict(
        grid_spacing=positive_scalar(reference["grid_spacing"], "reference grid_spacing"),
        offsets=expected.tolist(), neighbor_mask=mask.tolist(), cutoff_grid=cutoff,
        stencil_radius=radius,
        monomial_scale=positive_scalar(reference["monomial_scale"], "reference monomial_scale"),
        coordinate_scaling=scaling,
        spacing_ratio=positive_scalar(reference["spacing_ratio"], "reference spacing_ratio"),
    )


@register
class CartesianAFeatures(nn.Module, Configurable):
    """Compute normalized Cartesian ``A`` features on a fixed stencil.

    For central grid point ``g``, radial channel ``n``, monomial
    ``k = (a, b, c)``, and component ``t``, the module computes

    ``A[g, n, k, t] = sum_j (rho[g, j, t] / mean_density)``
    ``* w[n, j] * x_j^a * y_j^b * z_j^c``,

    where ``j`` runs over the canonically ordered integer stencil. With
    ``radial_basis="gaussian"``, the normalized weights are

    ``w[n, j] = exp(-alpha[n] * (|q_j| - u[n])^2)``
    ``/ sum_i exp(-alpha[n] * (|q_i| - u[n])^2)``.

    With ``radial_basis="none"``, the module uses the same expression with one
    fixed exponent ``alpha[0]=0``, exactly recovering the uniform normalized
    polynomial moment. With ``radial_basis="bessel"``, primitive ``n`` is

    ``sqrt(2/R_c) * sin(n*pi*r/R_c) / r``, ``n=1,...,N``,

    inside the cutoff, with its finite limit at ``r=0`` and exact zero at the
    boundary. The primitive radial-Cartesian products are discretely whitened
    separately for each total Cartesian degree on the actual integer stencil.
    An explicitly requested radial transform then mixes primitive channels
    before invariant products are formed. When ``separate_center=True``, the
    central point is excluded from every neighbor channel and supplied
    directly to the local readout by the model.
    Grid-volume quadrature is deliberately left to the free-energy readout
    rather than included in these local descriptors.

    Parameters
    ----------
    cutoff_grid
        Inclusive spherical cutoff in integer grid steps. The default is
        three, matching :class:`equicdft.data.GridData`.
    max_power
        Maximum total Cartesian power ``a + b + c``.
    radial_basis
        ``"none"`` selects uniform normalized weights. ``"gaussian"``
        selects normalized exponential damping channels. ``"bessel"``
        selects fixed spherical-Bessel-like functions conditioned on the
        exact discrete Cartesian stencil.
    radial_exponents
        Optional nonempty sequence of nonnegative damping coefficients
        ``alpha_n`` in inverse squared grid units. Its length determines the
        number of primitive functions; ``n_radial_channels`` may subsequently
        reduce the output count. For ``radial_basis="gaussian"``, ``None``
        uses one primitive with ``alpha=0.125``. It is unavailable for
        ``radial_basis="none"``, which always uses one fixed zero exponent,
        and for ``radial_basis="bessel"``.
    n_radial_channels
        Optional number of channels after a learned bias-free radial
        transform. For Gaussian and Bessel bases, explicitly supplying this
        argument enables one transform matrix per total Cartesian degree;
        omitting it retains all primitive functions without a transform. The
        output count cannot exceed the primitive count. With
        ``radial_basis="none"``, only the retained value ``1`` is accepted
        and no transform is applied.
    trainable_radial_exponents
        If ``True``, optimize positive ``radial_exponents`` in logarithmic
        form. Zero exponents are allowed only when the list is fixed.
    radial_centers
        Optional Gaussian centers ``u_n`` in grid units. Its length must
        match ``radial_exponents``. ``None`` uses zero for every channel,
        exactly retaining the original zero-centered Gaussian basis. It is
        unavailable for ``radial_basis="none"`` and
        ``radial_basis="bessel"``. A center ``u`` corresponds to physical
        radius ``u * grid_spacing``; ``coordinate_scaling`` does not alter
        this radial convention.
    trainable_radial_centers
        If ``True``, optimize the Gaussian centers directly. The default is
        ``False``. Initial centers must be nonnegative, but their optimized
        values are unconstrained.
    coordinate_scaling
        Cartesian-coordinate convention used in the monomials. ``"none"``
        uses the raw integer stencil offsets. ``"cutoff"`` divides each
        coordinate by ``cutoff_grid``, keeping polynomial moments similarly
        scaled when comparing different cutoffs. The damping distance remains
        in raw squared grid units. For the Bessel basis, degreewise whitening
        cancels this uniform scale factor up to numerical roundoff; the option
        is retained there only for a common Cartesian-feature interface.
    separate_center
        If ``True``, remove the zero offset from all neighbor moments. The
        model then concatenates the normalized central density to the
        invariant neighbor features exactly once.
    n_types
        Number of physical density components in the input field.
    n_channels
        Number of output density channels after a pointwise linear transform.
        When ``density_transform`` is omitted, this retains the
        original Xavier-initialized learned transform. ``None`` retains the
        physical density channels unless explicit transform weights are
        supplied.
    density_transform
        Optional matrix with shape ``[n_channels, n_types]``. Before any
        neighborhood gathering or Cartesian-moment construction, physical
        density channels are replaced by ``rho_transformed[..., q] =
        sum_t weights[q, t] * rho[..., t]``. The transformed densities replace
        the physical channels in both the neighborhood moments and direct
        center descriptors. Negative and rectangular transforms are allowed.
        The number of rows determines ``n_channels`` when it is omitted.
    trainable_density_transform
        Whether a configured density transform is optimized. The default is
        ``True``, retaining the existing learned ``n_channels`` behavior.
        Set it to ``False`` with explicit ``density_transform`` weights for a
        fixed physical basis such as number and charge densities.
    n_radial_functions
        Number of primitive Bessel functions. This is required with
        ``radial_basis="bessel"`` and unavailable for other radial bases.
        Frequencies are the fixed integer sequence ``n*pi/cutoff_grid``.
    mean_density
        Positive scalar used for scale-only density normalization. For fitting,
        set this to a precomputed reference density. It is
        stored as a fixed buffer; no mean is subtracted, so zero density
        remains zero.
    convolution_backend
        ``"gather"`` retains explicit periodic neighborhoods. ``"fft"``
        evaluates the same circular stencil contraction in reciprocal space
        without materializing ``[..., G, J, C]``.
    stencil_radius
        Optional active spherical radius in grid steps, no larger than
        ``cutoff_grid``. Regridding uses this to preserve a physical cutoff
        that falls between integer shells. The default is ``cutoff_grid``.
    monomial_scale
        Positive multiplier on the Cartesian coordinates after the selected
        ``coordinate_scaling``. Regridding uses it to preserve moment units.
    center_normalization
        ``"strict"`` retains ordinary center exclusion. The exploratory
        ``"reference"`` policy compensates each primitive Gaussian channel
        relative to ``normalization_reference`` before convolution. Its
        scalar center weight may be negative; it is not a positive kernel.
    normalization_reference
        Original stencil geometry recorded by :meth:`GridCACEModel.regrid`.
        It contains no frozen exponents or correction factors. The current
        exponents determine both reference and target normalizations.
    include_polarization
        If True, require electric dipole density ``[..., G, n_types, 3]``.
        Use the same spatial basis and species transform for rho and each
        vector component, retaining the vector's transformation law in B.
        The component axis contains blocks [rho, Px, Py, Pz], each ordered
        by spatial monomial. With separate_center, append the normalized
        center to each block before symmetrization, rather than feeding a
        raw vector to the readout. Radial/type axes remain independent.
    Polarization normalization is supplied by the containing model through
    the forward argument ``polarization_scale`` (dipole moment per volume).
    ``polarization_input='alignment'`` instead uses the model-owned smooth
    molecular alignment and its separate dimensionless scale.

    Notes
    -----
    The stencil and all monomials are model constants. They are constructed
    once using the same canonical ordering as ``GridData`` and registered as
    buffers, so they follow the module across devices without being optimized.
    """

    def __init__(
        self,
        max_power: int,
        mean_density: Union[float, torch.Tensor],
        *,
        cutoff_grid: int = 3,
        radial_basis: str = "none",
        radial_exponents: Optional[
            Union[Sequence[float], torch.Tensor]
        ] = None,
        n_radial_channels: Optional[int] = None,
        trainable_radial_exponents: bool = False,
        coordinate_scaling: str = "none",
        separate_center: bool = True,
        n_types: int = 1,
        n_channels: Optional[int] = None,
        radial_centers: Optional[
            Union[Sequence[float], torch.Tensor]
        ] = None,
        trainable_radial_centers: bool = False,
        density_transform: Optional[
            Union[Sequence[Sequence[float]], torch.Tensor]
        ] = None,
        trainable_density_transform: bool = True,
        n_radial_functions: Optional[int] = None,
        convolution_backend: str = "gather",
        include_polarization: bool = False,
        polarization_input: str = "raw",
        stencil_radius: Optional[float] = None,
        monomial_scale: float = 1.0,
        center_normalization: str = "strict",
        normalization_reference: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.include_polarization = boolean(include_polarization, "include_polarization")
        self.polarization_input = polarization_input_name(
            polarization_input,
            include_polarization=self.include_polarization,
        )
        if center_normalization not in ("strict", "reference"):
            raise ValueError("center_normalization must be 'strict' or 'reference'")
        radius = float(cutoff_grid) if stencil_radius is None else nonnegative_scalar(
            stencil_radius, "stencil_radius",
        )
        if radius > cutoff_grid:
            raise ValueError("stencil_radius cannot exceed cutoff_grid")
        monomial_scale = positive_scalar(monomial_scale, "monomial_scale")
        reference = _normalization_reference_config(normalization_reference)
        if center_normalization == "reference" and reference is None:
            raise ValueError("reference center normalization requires normalization_reference")

        if not isinstance(radial_basis, str):
            raise TypeError(
                "radial_basis must be 'bessel', 'gaussian', or 'none'"
            )
        radial_basis = radial_basis.lower()
        if radial_basis not in ("bessel", "gaussian", "none"):
            raise ValueError(
                "radial_basis must be 'bessel', 'gaussian', or 'none'"
            )
        if n_radial_channels is not None:
            n_radial_channels = positive_integer(
                n_radial_channels,
                "n_radial_channels",
            )
            if radial_basis == "none" and n_radial_channels != 1:
                raise ValueError(
                    "n_radial_channels must be 1 with "
                    "radial_basis='none'"
                )
        n_radial_functions = optional_positive_integer(
            n_radial_functions,
            "n_radial_functions",
        )
        if radial_basis == "bessel":
            if n_radial_functions is None:
                raise ValueError(
                    "n_radial_functions is required with "
                    "radial_basis='bessel'"
                )
        elif n_radial_functions is not None:
            raise ValueError(
                "n_radial_functions requires radial_basis='bessel'"
            )
        trainable_radial_exponents = boolean(
            trainable_radial_exponents,
            "trainable_radial_exponents",
        )
        trainable_radial_centers = boolean(
            trainable_radial_centers,
            "trainable_radial_centers",
        )
        if radial_basis != "gaussian" and trainable_radial_centers:
            raise ValueError(
                "trainable_radial_centers requires radial_basis='gaussian'"
            )
        if radial_basis == "bessel":
            if radial_exponents is not None:
                raise ValueError(
                    "radial_exponents are unavailable with "
                    "radial_basis='bessel'"
                )
            if trainable_radial_exponents:
                raise ValueError(
                    "trainable_radial_exponents requires "
                    "radial_basis='gaussian'"
                )
            if radial_centers is not None:
                raise ValueError(
                    "radial_centers are unavailable with "
                    "radial_basis='bessel'"
                )
        if not isinstance(coordinate_scaling, str):
            raise TypeError("coordinate_scaling must be 'none' or 'cutoff'")
        coordinate_scaling = coordinate_scaling.lower()
        if coordinate_scaling not in ("none", "cutoff"):
            raise ValueError("coordinate_scaling must be 'none' or 'cutoff'")
        if not isinstance(convolution_backend, str):
            raise TypeError("convolution_backend must be 'gather' or 'fft'")
        convolution_backend = convolution_backend.lower()
        if convolution_backend not in ("gather", "fft"):
            raise ValueError("convolution_backend must be 'gather' or 'fft'")
        separate_center = boolean(separate_center, "separate_center")
        n_types = positive_integer(n_types, "n_types")
        trainable_density_transform = boolean(
            trainable_density_transform,
            "trainable_density_transform",
        )
        if density_transform is None:
            transform_weights = None
        else:
            transform_weights = torch.as_tensor(
                density_transform,
                dtype=torch.get_default_dtype(),
            ).detach().clone()
            if transform_weights.ndim != 2:
                raise ValueError(
                    "density_transform must be a two-dimensional "
                    "matrix"
                )
            if transform_weights.shape[0] == 0:
                raise ValueError(
                    "density_transform must contain at least one row"
                )
            if transform_weights.shape[1] != n_types:
                raise ValueError(
                    "density_transform must contain one column per "
                    "physical density type"
                )
            if not torch.all(torch.isfinite(transform_weights)).item():
                raise ValueError("density_transform must be finite")
        n_channels = optional_positive_integer(n_channels, "n_channels")
        if transform_weights is not None:
            inferred_channels = int(transform_weights.shape[0])
            if n_channels is None:
                n_channels = inferred_channels
            elif n_channels != inferred_channels:
                raise ValueError(
                    "n_channels must equal the number of rows in "
                    "density_transform"
                )
        if n_channels is not None and n_types == 1:
            raise ValueError(
                "density transforms are disabled for one-component fields"
            )

        mean_density_tensor = torch.as_tensor(
            mean_density,
            dtype=torch.get_default_dtype(),
        ).detach().clone()
        if mean_density_tensor.numel() != 1:
            raise ValueError("mean_density must be a positive scalar")
        mean_density_tensor = mean_density_tensor.reshape(())
        if not torch.isfinite(mean_density_tensor).item():
            raise ValueError("mean_density must be finite")
        if mean_density_tensor.item() <= 0.0:
            raise ValueError("mean_density must be positive")

        # Both GridData and this module call make_stencil, so neighbor j in
        # local_density always corresponds to row j of the monomial table.
        local_density_positions = torch.from_numpy(make_stencil(cutoff_grid))
        powers = _make_powers(max_power)

        # Evaluate every Cartesian monomial once on the shared integer
        # stencil. These values are independent of the central grid point and
        # of the density configuration.
        positions = local_density_positions.to(dtype=torch.get_default_dtype())
        squared_distances = torch.sum(positions**2, dim=1)
        monomial_positions = positions
        if coordinate_scaling == "cutoff" and int(cutoff_grid) > 0:
            monomial_positions = positions / float(cutoff_grid)
        if monomial_scale != 1.0:
            monomial_positions = monomial_positions * monomial_scale
        monomial_values = torch.ones(
            (positions.shape[0], powers.shape[0]),
            dtype=torch.get_default_dtype(),
        )
        for axis in range(3):
            monomial_values = monomial_values * monomial_positions[
                :, axis, None
            ].pow(powers[None, :, axis])

        center_mask = squared_distances == 0
        neighbor_mask = ~center_mask if separate_center else torch.ones_like(
            center_mask
        )
        if radius != cutoff_grid:
            neighbor_mask = neighbor_mask & (
                squared_distances <= radius ** 2 + 1e-12 * max(1., radius ** 2)
            )

        if radial_basis == "bessel":
            radial_function_count = int(n_radial_functions)
            bessel_basis, bessel_gram_eigenvalues = (
                _conditioned_bessel_stencil_basis(
                    squared_distances,
                    monomial_values,
                    powers,
                    int(cutoff_grid),
                    neighbor_mask,
                    radial_function_count,
                )
            )
        else:
            # Preserve the established unit-sum Gaussian/none construction.
            initial_radial_exponents = prepare_radial_exponents(
                radial_basis,
                radial_exponents,
                trainable_radial_exponents,
            )
            initial_radial_centers = prepare_radial_centers(
                radial_basis,
                radial_centers,
                int(initial_radial_exponents.numel()),
            )

            if radial_basis == "gaussian" and trainable_radial_exponents:
                self.log_radial_exponents = nn.Parameter(
                    torch.log(initial_radial_exponents)
                )
            else:
                self.register_buffer(
                    "fixed_radial_exponents",
                    initial_radial_exponents,
                )
            if radial_basis == "gaussian" and trainable_radial_centers:
                self.learned_radial_centers = nn.Parameter(
                    initial_radial_centers
                )
            else:
                # Fixed centers are deterministic constructor input, not
                # fitted state, so they stay out of the state_dict.
                self.register_buffer(
                    "fixed_radial_centers",
                    initial_radial_centers,
                    persistent=False,
                )
            radial_function_count = int(initial_radial_exponents.numel())

        if radial_basis == "none" or n_radial_channels is None:
            radial_channel_count = radial_function_count
            radial_transform = None
        else:
            radial_transform = _RadialTransform(
                max_power=int(max_power),
                n_radial_functions=radial_function_count,
                n_radial_channels=n_radial_channels,
            )
            radial_channel_count = n_radial_channels

        if radial_basis == "bessel":
            self.register_buffer(
                "fixed_bessel_stencil_basis",
                bessel_basis,
            )
            self.register_buffer(
                "bessel_gram_eigenvalues",
                bessel_gram_eigenvalues,
            )
        self.cutoff_grid = int(cutoff_grid)
        self.max_power = int(max_power)
        self.radial_basis = radial_basis
        self.trainable_radial_exponents = trainable_radial_exponents
        self.trainable_radial_centers = trainable_radial_centers
        self.coordinate_scaling = coordinate_scaling
        self.convolution_backend = convolution_backend
        self.separate_center = separate_center
        self.n_types = n_types
        self.n_channels = n_channels
        self.trainable_density_transform = trainable_density_transform
        self.n_radial_functions = radial_function_count
        self.n_radial_channels = radial_channel_count
        self.n_output_channels = n_types if n_channels is None else n_channels
        self.radial_transform = radial_transform
        self.density_transform = (
            None
            if n_channels is None
            else _DensityMixing(
                n_types=n_types,
                n_channels=n_channels,
                weights=transform_weights,
                trainable=trainable_density_transform,
            )
        )
        self.register_buffer("local_density_positions", local_density_positions)
        self.register_buffer("squared_distances", squared_distances)
        self.register_buffer("powers", powers)
        self.register_buffer("monomial_values", monomial_values)
        self.register_buffer("mean_density", mean_density_tensor)
        # This deterministic constructor product is not fitted state.
        self.register_buffer("neighbor_mask", neighbor_mask, persistent=False)
        self.stencil_radius = radius
        self.monomial_scale = monomial_scale
        self.center_normalization = center_normalization
        self.normalization_reference = reference
        if reference is not None:
            ratio = reference["spacing_ratio"]
            expected_scale = reference["monomial_scale"] * ratio
            if coordinate_scaling == "cutoff":
                expected_scale *= self.cutoff_grid / reference["cutoff_grid"]
            if (reference["coordinate_scaling"] != coordinate_scaling
                    or not math.isclose(radius, reference["stencil_radius"] / ratio, rel_tol=1e-12)
                    or not math.isclose(monomial_scale, expected_scale, rel_tol=1e-12)):
                raise ValueError("normalization_reference conflicts with current radius or monomial units")
        self.register_buffer(
            "normalization_reference_squared_distances",
            None if reference is None else torch.as_tensor(
                reference["offsets"], dtype=mean_density_tensor.dtype,
            ).square().sum(dim=-1), persistent=False,
        )
        self.register_buffer(
            "normalization_reference_neighbor_mask",
            None if reference is None else torch.tensor(reference["neighbor_mask"], dtype=torch.bool),
            persistent=False,
        )
        self._normalization_reference_is_native = reference is not None and (
            reference["spacing_ratio"] == 1.0
            and reference["coordinate_scaling"] == coordinate_scaling
            and reference["monomial_scale"] == monomial_scale
            and reference["stencil_radius"] == radius
            and reference["offsets"] == local_density_positions.tolist()
            and reference["neighbor_mask"] == neighbor_mask.tolist()
        )
        if center_normalization == "reference":
            self._validate_reference_normalization()

    def to_config(self) -> Dict[str, Any]:
        """Return the constructor arguments describing this module.

        Learned values live in the ``state_dict`` and are not repeated here,
        with two deliberate exceptions. Gaussian exponents are recorded
        because their count fixes the number of primitive channels, and the
        current values always satisfy the constructor's positivity check.
        Fixed Gaussian centers are recorded because they are not persistent
        state; trainable centers may have left the nonnegative initial
        domain, so they are described as ``None`` and restored from state.
        """

        gaussian = self.radial_basis == "gaussian"
        bessel = self.radial_basis == "bessel"
        density_transform = self.density_transform
        config = make_config(
            self,
            max_power=self.max_power,
            mean_density=scalar_value(self.mean_density),
            cutoff_grid=self.cutoff_grid,
            radial_basis=self.radial_basis,
            radial_exponents=self.radial_exponents if gaussian else None,
            n_radial_channels=(
                None
                if self.radial_transform is None
                else self.n_radial_channels
            ),
            trainable_radial_exponents=self.trainable_radial_exponents,
            coordinate_scaling=self.coordinate_scaling,
            separate_center=self.separate_center,
            n_types=self.n_types,
            n_channels=self.n_channels,
            radial_centers=(
                self.radial_centers
                if gaussian and not self.trainable_radial_centers
                else None
            ),
            trainable_radial_centers=self.trainable_radial_centers,
            density_transform=(
                None
                if density_transform is None
                else density_transform.weight
            ),
            trainable_density_transform=self.trainable_density_transform,
            n_radial_functions=self.n_radial_functions if bessel else None,
            convolution_backend=self.convolution_backend,
            include_polarization=self.include_polarization,
            polarization_input=self.polarization_input,
        )
        # Keep the exact legacy configuration/state structure for default
        # models. New geometry is structural config, not mandatory state keys.
        radius = getattr(self, "stencil_radius", float(self.cutoff_grid))
        scale = getattr(self, "monomial_scale", 1.0)
        if radius != self.cutoff_grid:
            config["stencil_radius"] = radius
        if scale != 1.0:
            config["monomial_scale"] = scale
        policy = getattr(self, "center_normalization", "strict")
        if policy != "strict":
            config["center_normalization"] = policy
        reference = getattr(self, "normalization_reference", None)
        if reference is not None:
            config["normalization_reference"] = copy.deepcopy(reference)
        return config

    def _validate_reference_normalization(self):
        """Fail closed for kernels outside the tested primitive Gaussian rule."""
        if self.radial_basis not in ("none", "gaussian") or self.radial_transform is not None:
            raise ValueError("reference normalization supports only untransformed Gaussian/none primitives")
        if self.include_polarization or not self.separate_center:
            raise ValueError("reference normalization requires scalar separate_center features")
        if self.trainable_radial_centers or torch.any(self.radial_centers != 0).item():
            raise ValueError("reference normalization requires fixed zero-centered Gaussians")
        if not self.neighbor_mask.any().item():
            raise ValueError("reference normalization requires active noncentral neighbors")

    def _primitive_stencil_basis(self, radial_exponents, radial_centers):
        """One primitive path for base and independent Gaussian messages."""
        arguments = {}
        if getattr(self, "center_normalization", "strict") == "reference":
            if torch.any(radial_centers != 0).item():
                raise ValueError("reference normalization requires zero-centered Gaussian overrides")
        if (getattr(self, "center_normalization", "strict") == "reference"
                and not self._normalization_reference_is_native):
            arguments = dict(
                reference_squared_distances=self.normalization_reference_squared_distances,
                reference_neighbor_mask=self.normalization_reference_neighbor_mask,
                spacing_ratio=self.normalization_reference["spacing_ratio"],
            )
        return _cartesian_stencil_basis(
            self.squared_distances, self.monomial_values, radial_exponents,
            radial_centers, self.stencil_neighbor_mask(), **arguments,
        )

    @property
    def radial_exponents(self) -> torch.Tensor:
        """Gaussian damping coefficients for all primitive channels.

        The ``"none"`` basis reports its single fixed zero exponent.
        """

        if self.radial_basis == "bessel":
            raise RuntimeError(
                "radial_exponents are unavailable for the Bessel basis"
            )
        if self.trainable_radial_exponents:
            return torch.exp(self.log_radial_exponents)
        return self.fixed_radial_exponents

    @property
    def radial_centers(self) -> torch.Tensor:
        """Gaussian centers in grid units; zero for the ``"none"`` basis."""

        if self.radial_basis == "bessel":
            raise RuntimeError(
                "radial_centers are unavailable for the Bessel basis"
            )
        if self.trainable_radial_centers:
            return self.learned_radial_centers
        return self.fixed_radial_centers

    def stencil_basis(
        self,
        radial_exponents: Optional[torch.Tensor] = None,
        radial_centers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a radial-Cartesian basis with shape ``[J, N, K]``.

        ``J`` is the number of stencil points, ``N`` the number of output
        radial channels, and ``K`` the number of Cartesian monomials. With no
        arguments, this returns the configured basis including any learned
        pre-invariant radial transform. Message layers may supply independent
        Gaussian exponents while reusing the fixed stencil geometry; such an
        explicit override bypasses the initial basis and transform.
        """

        if radial_centers is not None and radial_exponents is None:
            if self.radial_basis == "bessel":
                raise ValueError(
                    "radial_centers require explicit Gaussian "
                    "radial_exponents"
                )
            radial_exponents = self.radial_exponents
        independent_gaussian = radial_exponents is not None
        if independent_gaussian:
            if radial_centers is None:
                if self.radial_basis == "bessel":
                    radial_centers = torch.zeros_like(radial_exponents)
                else:
                    radial_centers = self.radial_centers
            if radial_centers.shape != radial_exponents.shape:
                raise ValueError(
                    "radial_centers shape must match radial_exponents shape"
                )
            return self._primitive_stencil_basis(radial_exponents, radial_centers)

        if self.radial_basis == "bessel":
            basis = self.fixed_bessel_stencil_basis
        else:
            basis = self._primitive_stencil_basis(self.radial_exponents, self.radial_centers)

        if self.radial_transform is not None:
            basis = self.radial_transform(basis, self.powers)
        return basis

    def bessel_stencil_basis(
        self,
        n_radial_functions: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a conditioned Bessel basis and Gram eigenvalues on this stencil.

        Message layers with their own Bessel basis call this through
        :meth:`equicdft.interaction.BChiMessage.bind_geometry` so that every
        layer shares the exact integer stencil and center convention.
        """

        return _conditioned_bessel_stencil_basis(
            self.squared_distances,
            self.monomial_values,
            self.powers,
            self.cutoff_grid,
            self.stencil_neighbor_mask(),
            n_radial_functions,
        )

    def stencil_neighbor_mask(self) -> torch.Tensor:
        """Return the Boolean center-inclusion mask over stencil points."""

        return self.neighbor_mask

    def transform_density(self, rho: torch.Tensor) -> torch.Tensor:
        """Return the physical or configured transformed density channels.

        The transform is pointwise and remains connected to ``rho`` for
        functional derivatives.
        """

        if rho.ndim < 2 or rho.shape[-1] != self.n_types:
            raise ValueError(
                "rho must have shape [..., n_grid, n_types] with the "
                "configured number of physical density types"
            )
        if self.n_channels is None:
            return rho
        return self.density_transform(rho)

    @property
    def requires_dipole_density(self) -> bool:
        return self.include_polarization

    @property
    def requires_polarization_scale(self) -> bool:
        return self.include_polarization and self.polarization_input == "raw"

    @property
    def requires_alignment_scale(self) -> bool:
        return self.include_polarization and self.polarization_input == "alignment"

    def forward(
        self, data: Mapping[str, torch.Tensor], *,
        polarization_scale: Optional[Union[float, torch.Tensor]] = None,
        alignment_scale: Optional[Union[float, torch.Tensor]] = None,
        dipole_magnitude: Optional[Union[float, torch.Tensor]] = None,
        alignment_density_floor: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return density-weighted Cartesian ``A`` features.

        Parameters
        ----------
        data
            GridData-like mapping containing:

            ``rho``
                Live density tensor with shape ``[..., n_grid, n_types]``.
            ``local_density_index``
                Periodic neighbor rows with shape
                ``[..., n_grid, n_neighbors]``, required by the default
                ``"gather"`` backend.
            ``grid_positions``, ``grid_size``
                Complete regular-grid geometry required by the ``"fft"``
                backend.
        polarization_scale
            Positive model-owned scale for all raw P components. Required only
            when ``polarization_input='raw'``; standalone calls supply it
            explicitly.
        alignment_scale, dipole_magnitude, alignment_density_floor
            Model-owned values required only when
            ``polarization_input='alignment'``.
        Returns
        -------
        torch.Tensor
            ``A`` with shape
            ``[..., n_grid, n_radial_channels, n_monomials, n_output_channels]``.
            The final dimension equals ``n_types`` when mixing is disabled and
            ``n_channels`` when mixing is enabled.
            With polarization, replace n_monomials by
            ``4 * (n_monomials + int(separate_center))``.
        """

        stencil_basis = self.stencil_basis()
        if not self.requires_dipole_density:
            descriptor_density = self.transform_density(data["rho"])
            return self._convolve(descriptor_density, self.mean_density, stencil_basis, data)

        fields, scales = self.polarization_fields(
            data,
            polarization_scale=polarization_scale,
            alignment_scale=alignment_scale,
            dipole_magnitude=dipole_magnitude,
            alignment_density_floor=alignment_density_floor,
        )
        moments = self._convolve(fields, scales, stencil_basis, data)
        if self.separate_center:
            center = (fields / scales).unsqueeze(-2).unsqueeze(-2)
            center = center.expand(*moments.shape[:-2], 1, fields.shape[-1])
            moments = torch.cat((moments, center), dim=-2)
        # [..., G, N, K, 4*C] -> [..., G, N, 4*K, C], field-major.
        moments = moments.unflatten(-1, (4, self.n_output_channels))
        return moments.transpose(-3, -2).flatten(-3, -2)

    def polarization_fields(
        self,
        data,
        *,
        polarization_scale=None,
        alignment_scale=None,
        dipole_magnitude=None,
        alignment_density_floor=None,
    ):
        """Pack [rho, Vx, Vy, Vz] for raw P or reduced alignment V.

        Initial moments and joint messages use the same species mixing and
        normalization. These carriers remain connected to the physical inputs.
        """
        rho = data["rho"]
        if "dipole_density" not in data:
            raise ValueError("dipole_density is required when include_polarization=True")
        dipole = data["dipole_density"]
        if dipole.shape != (*rho.shape, 3):
            raise ValueError("dipole_density must have shape [..., n_grid, n_types, 3]")
        if dipole.dtype != rho.dtype or dipole.device != rho.device:
            raise ValueError("rho and dipole_density must share dtype and device")
        if not torch.isfinite(dipole).all():
            raise ValueError("dipole_density must be finite")
        if self.polarization_input == "raw":
            if polarization_scale is None:
                raise ValueError(
                    "raw polarized features require polarization_scale from the model"
                )
            scale = torch.as_tensor(
                polarization_scale, dtype=rho.dtype, device=rho.device,
            )
            if scale.numel() != 1 or not torch.isfinite(scale).all() or torch.any(scale <= 0):
                raise ValueError("polarization_scale must be a positive scalar")
            vector_source = dipole
        else:
            if alignment_scale is None:
                raise ValueError(
                    "alignment features require alignment_scale from the model"
                )
            scale = torch.as_tensor(
                alignment_scale, dtype=rho.dtype, device=rho.device,
            )
            if scale.numel() != 1 or not torch.isfinite(scale).all() or torch.any(scale <= 0):
                raise ValueError("alignment_scale must be a positive scalar")
            vector_source = regularized_reduced_polarization(
                rho,
                dipole,
                dipole_magnitude,
                alignment_density_floor,
            )
        scale = scale.reshape(())
        # Apply the same species mixing to each component, without mixing
        # scalar and vector fields or treating vector axes as species.
        vector = self.transform_density(vector_source.transpose(-1, -2))
        channels = self.n_output_channels
        descriptor_density = self.transform_density(rho)
        fields = torch.cat((descriptor_density, vector.flatten(-2)), dim=-1)
        scales = torch.cat((
            self.mean_density.expand(channels),
            scale.expand(3 * channels),
        ))
        return fields, scales

    def _convolve(self, fields, scales, stencil_basis, data):
        """Shared scalar/vector field convolution with scale-only normalization."""
        if self.convolution_backend == "fft":
            if "grid_positions" not in data or "grid_size" not in data:
                raise ValueError(
                    "fft features require grid_positions and grid_size"
                )
            return periodic_density_convolution(
                fields / scales,
                stencil_basis,
                data["grid_positions"],
                data["grid_size"],
                self.local_density_positions,
            )
        if self.convolution_backend != "gather":
            raise RuntimeError(
                "convolution_backend must be 'gather' or 'fft'"
            )
        if "local_density_index" not in data:
            raise ValueError(
                "gather features require local_density_index; reload the "
                "data with include_local_density_index=True or use the fft "
                "feature backend"
            )
        local_density = gather_neighbors(
            fields,
            data["local_density_index"],
        )
        if local_density.shape[-2] != self.monomial_values.shape[0]:
            raise ValueError(
                "local_density neighbor count does not match cutoff_grid={}".format(
                    self.cutoff_grid
                )
            )

        # Contract only the neighbor axis. Grid points, radial channels,
        # monomials, component channels, and leading batch dimensions remain
        # separate.
        features = torch.einsum(
            "...gjt,jnk->...gnkt",
            local_density / scales,
            stencil_basis,
        )
        return features


class _DensityMixing(nn.Module):
    """Internal pointwise map from physical to descriptor densities.

    ``rho_mixed[..., q] = sum_t weight[q, t] * rho[..., t]``.

    Here ``t`` labels physical density components and ``q`` labels descriptor
    density channels. The map acts independently at every spatial point, so
    spatial rotations and reflections are unaffected.

    Notes
    -----
    The map intentionally has no additive bias, so an empty voxel remains
    empty after the transform.
    """

    def __init__(
        self,
        n_types: int,
        n_channels: int,
        weights: Optional[torch.Tensor] = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()

        self.n_types = positive_integer(n_types, "n_types")
        self.n_channels = positive_integer(n_channels, "n_channels")
        if weights is None:
            initial_weights = torch.empty(
                self.n_channels,
                self.n_types,
                dtype=torch.get_default_dtype(),
            )
            nn.init.xavier_uniform_(initial_weights)
        else:
            initial_weights = weights.detach().clone()
        self.weight = nn.Parameter(initial_weights, requires_grad=trainable)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """Return ``rho`` with its physical-type axis linearly transformed."""

        if rho.ndim < 2:
            raise ValueError(
                "rho must have shape [..., n_grid, n_types]"
            )
        if rho.shape[-1] != self.n_types:
            raise ValueError(
                "rho has {} type channels but this module expects {}".format(
                    rho.shape[-1],
                    self.n_types,
                )
            )

        return torch.einsum("...t,qt->...q", rho, self.weight)
