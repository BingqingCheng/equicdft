"""Prescribed external potentials evaluated at absolute simulation time."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
import math

import torch

from ..geometry.grid import grid_spacing_tensor


class ExternalField(ABC):
    """Return the complete V_ext, in energy units and rho's voxel/species order.

    Evaluation must be deterministic, independent of rho, and must not mutate
    data or advance internal state. To drive conducting electrodes too, return
    a dictionary with ``V_ext`` and ``metal_external_field`` (Cartesian [3],
    in energy per charge per length). List all switches/kinks in ``breakpoints``;
    values at a switch are right-continuous. Drivers may repeat evaluations
    when a step is rejected. Override ``metadata`` to describe custom protocols.
    """

    breakpoints = ()

    @abstractmethod
    def __call__(self, time, data):
        """Return a finite tensor with the same shape, dtype and device as rho."""

    def metadata(self):
        return {"class": type(self).__name__, "breakpoints": list(self.breakpoints)}


class RampedField(ExternalField):
    """Linearly interpolate two complete potentials; hold endpoints outside the ramp."""

    def __init__(self, initial, final, duration, start_time=0.0, *,
                 initial_metal_field=None, final_metal_field=None):
        self.start_time = _time(start_time)
        self.duration = _time(duration)
        if self.duration <= 0:
            raise ValueError("duration must be positive.")
        self.initial = torch.as_tensor(initial).detach().clone()
        self.final = torch.as_tensor(final).detach().clone()
        if self.initial.shape != self.final.shape:
            raise ValueError("initial and final potentials must have the same shape.")
        if not torch.isfinite(self.initial).all() or not torch.isfinite(self.final).all():
            raise ValueError("Ramp potentials must be finite.")
        self.breakpoints = (self.start_time, _time(self.start_time + self.duration))
        self.metal_fields = None
        if initial_metal_field is not None or final_metal_field is not None:
            from ..functional.metalwall import _field_vector
            if initial_metal_field is None or final_metal_field is None:
                raise ValueError("Supply both initial_metal_field and final_metal_field.")
            self.metal_fields = tuple(_field_vector(v, "metal field") for v in
                                      (initial_metal_field, final_metal_field))
            if any(v.shape != (3,) for v in self.metal_fields):
                raise ValueError("Ramp metal fields must have shape [3].")

    def __call__(self, time, data):
        fraction = min(max((time - self.start_time) / self.duration, 0.0), 1.0)
        initial, final = self.initial.to(data["rho"]), self.final.to(data["rho"])
        potential = initial + fraction * (final - initial)
        if self.metal_fields is None:
            return potential
        a, b = (v.to(data["rho"]) for v in self.metal_fields)
        return dict(V_ext=potential, metal_external_field=a+fraction*(b-a))

    def metadata(self):
        return dict(super().metadata(), initial=self.initial.cpu().clone(),
                    final=self.final.cpu().clone(), duration=self.duration,
                    start_time=self.start_time,
                    metal_fields=(None if self.metal_fields is None else
                                  tuple(v.cpu().clone() for v in self.metal_fields)))

    @classmethod
    def from_electric_field(cls, base_potential, grid_center, charges, duration, *,
                            final_field, initial_field=(0., 0., 0.),
                            start_time=0., metal=True):
        """Build V_i = V_base,i - q_i E dot r and the matching electrode field.

        Coordinates, charges and fields use the model's physical input units.
        The caller must place the liquid branch cut across blocked transport faces.
        Set metal=False for a liquid-only ramp. No potential clipping is applied.
        """
        from ..functional.metalwall import _field_vector
        base = torch.as_tensor(base_potential).detach().clone()
        if base.ndim != 2 or not base.is_floating_point():
            raise ValueError("base_potential must be a floating [voxels, species] tensor.")
        positions = torch.as_tensor(grid_center, dtype=base.dtype, device=base.device)
        charge = torch.as_tensor(charges, dtype=base.dtype, device=base.device)
        if (positions.shape != (base.shape[0], 3) or charge.shape != (base.shape[1],)
                or not torch.isfinite(positions).all() or not torch.isfinite(charge).all()):
            raise ValueError("grid_center and charges must be finite [voxels, 3] and [species].")
        fields = tuple(_field_vector(v, "electric field").to(base)
                       for v in (initial_field, final_field))
        if any(v.shape != (3,) for v in fields) or not isinstance(metal, bool):
            raise ValueError("Fields must have shape [3] and metal must be boolean.")
        potentials = tuple(base - (positions @ v)[:, None] * charge for v in fields)
        options = dict(initial_metal_field=fields[0], final_metal_field=fields[1]) if metal else {}
        return cls(*potentials, duration, start_time, **options)


def _time(value):
    if isinstance(value, bool):
        raise ValueError("Time must be finite and nonnegative.")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Time must be finite and nonnegative.")
    return value


def _breakpoints(field):
    if field is None:
        return ()
    if not isinstance(field, ExternalField):
        raise TypeError("field must be an ExternalField or None.")
    points = tuple(_time(t) for t in field.breakpoints)
    if any(b <= a for a, b in zip(points, points[1:])):
        raise ValueError("Field breakpoints must be strictly increasing.")
    return points


class _FieldEvaluator:
    """Driver-owned field validation, reset for each run rather than stored in data."""

    def __init__(self, field):
        self.field = field
        self.keys = None

    def __call__(self, time, data, left=False):
        return _at_time(self.field, time, data, left, check_keys=self._check_keys)

    def _check_keys(self, keys):
        if self.keys is not None and self.keys != keys:
            raise ValueError("ExternalField must return the same keys throughout a trajectory.")
        self.keys = keys


def _at_time(field, time, data, left=False, *, check_keys=None):
    if field is None:
        return data
    # Evaluate the left limit only at an explicitly declared breakpoint.
    sample_time = (torch.nextafter(torch.tensor(time, dtype=torch.float64),
                                  torch.tensor(-math.inf, dtype=torch.float64)).item()
                   if left and time in field.breakpoints else time)
    with torch.no_grad():
        potential = field(sample_time, data)
    updates = dict(potential) if isinstance(potential, Mapping) else dict(V_ext=potential)
    if set(updates) not in ({"V_ext"}, {"V_ext", "metal_external_field"}):
        raise ValueError("ExternalField returns V_ext and optionally metal_external_field only.")
    if check_keys is not None:
        check_keys(tuple(sorted(updates)))
    potential = updates["V_ext"]
    rho = data["rho"]
    if (not torch.is_tensor(potential) or potential.shape != rho.shape
            or potential.dtype != rho.dtype or potential.device != rho.device
            or not torch.isfinite(potential).all()):
        raise ValueError("ExternalField must return finite V_ext matching rho's shape, dtype and device.")
    if "metal_external_field" in updates:
        value = updates["metal_external_field"]
        if (not torch.is_tensor(value) or value.shape != (3,) or value.dtype != rho.dtype
                or value.device != rho.device or not torch.isfinite(value).all()):
            raise ValueError("metal_external_field must be finite [3], matching rho's dtype and device.")
    return dict(data, **{key: value.detach().clone() for key,value in updates.items()})


def _metal_walls(model):
    from ..functional.metalwall import MetalWall
    if not isinstance(model, torch.nn.Module):
        raise ValueError("metal_external_field requires a torch.nn.Module containing MetalWall.")
    walls = [m for m in model.modules() if isinstance(m, MetalWall)]
    if not walls:
        raise ValueError("metal_external_field requires a model containing MetalWall.")
    return walls


def _metal_context(model, data):
    from ..functional.base import ReadoutContext
    _metal_walls(model)
    names = ("grid_spacing", "mean_temperature", "boltzmann_constant")
    if any(not hasattr(model, name) for name in names):
        raise ValueError("A driven metal model must declare grid_spacing, mean_temperature and boltzmann_constant.")
    context = ReadoutContext.from_data(data, **{name: getattr(model, name) for name in names},
                                       free_energy_mode="beta")
    beta = torch.as_tensor(data["beta"]).to(context.beta)
    if not torch.allclose(beta, context.beta, rtol=1e-6, atol=0.):
        raise ValueError("data beta must match the driven metal model's temperature/k_B convention.")
    return context


def _metal_dipole(model, output):
    walls = _metal_walls(model)
    # Full-model electrode output is unambiguous only for one wall module.
    # Other custom compositions retain the midpoint evaluation fallback.
    if len(walls) != 1 or "metal_site_q" not in output:
        return None
    return walls[0].field_dipole(output["metal_site_q"]).detach().clone()


def _work(start, end, model=None, *, start_dipole=None, end_dipole=None):
    """Trapezoidal external work in beta units (exact for a jump at fixed rho)."""
    beta = float(start["beta"])
    if not math.isfinite(beta) or beta <= 0 or beta != float(end["beta"]):
        raise ValueError("External work requires one fixed, positive beta at both endpoints.")
    volume = grid_spacing_tensor(start["grid_spacing"], dtype=start["rho"].dtype,
                                 device=start["rho"].device).prod()
    work = (float(start["beta"]) * volume *
            (0.5 * (start["rho"] + end["rho"]) * (end["V_ext"] - start["V_ext"])).sum()).detach()
    if "metal_external_field" not in start and "metal_external_field" not in end:
        return work
    if "metal_external_field" not in start or "metal_external_field" not in end:
        raise ValueError("A driven metal field must be present at both endpoints.")
    walls = _metal_walls(model)
    _metal_context(model, start)
    _metal_context(model, end)
    delta = end["metal_external_field"] - start["metal_external_field"]
    if not torch.any(delta != 0):
        return work
    if start_dipole is not None and end_dipole is not None:
        dipole = .5 * (start_dipole + end_dipole)
        return work - beta * (dipole * delta.to(dipole)).sum().detach()
    # Only the electrode solve is needed for its conjugate dipole: no learned
    # short-range evaluation or functional differentiation for this work term.
    midpoint = dict(start, rho=.5*(start["rho"]+end["rho"]),
                    metal_external_field=.5*(start["metal_external_field"]+end["metal_external_field"]))
    with torch.no_grad():
        context = _metal_context(model, midpoint)
        for wall in walls:
            work = work + wall.external_field_work(context, delta).sum().detach()
    return work


def _same_fields(a, b):
    if ("metal_external_field" in a) != ("metal_external_field" in b):
        raise ValueError("A driven metal field must be present at both endpoints.")
    return all(torch.equal(a[key], b[key]) for key in
               ("V_ext", "metal_external_field") if key in a or key in b)
