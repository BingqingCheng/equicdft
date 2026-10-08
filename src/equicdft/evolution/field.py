"""Prescribed external potentials evaluated at absolute simulation time."""

from abc import ABC, abstractmethod
import math

import torch

from ..geometry.grid import grid_spacing_tensor


class ExternalField(ABC):
    """Return the complete V_ext, in energy units and rho's voxel/species order.

    Evaluation must be deterministic, independent of rho, and must not mutate
    data or advance internal state. List all switches/kinks in ``breakpoints``;
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

    def __init__(self, initial, final, duration, start_time=0.0):
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

    def __call__(self, time, data):
        fraction = min(max((time - self.start_time) / self.duration, 0.0), 1.0)
        initial, final = self.initial.to(data["rho"]), self.final.to(data["rho"])
        return initial + fraction * (final - initial)

    def metadata(self):
        return dict(super().metadata(), initial=self.initial.cpu().clone(),
                    final=self.final.cpu().clone(), duration=self.duration,
                    start_time=self.start_time)


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


def _at_time(field, time, data, left=False):
    if field is None:
        return data
    # Evaluate the left limit only at an explicitly declared breakpoint.
    sample_time = (torch.nextafter(torch.tensor(time, dtype=torch.float64),
                                  torch.tensor(-math.inf, dtype=torch.float64)).item()
                   if left and time in field.breakpoints else time)
    with torch.no_grad():
        potential = field(sample_time, data)
    rho = data["rho"]
    if (not torch.is_tensor(potential) or potential.shape != rho.shape
            or potential.dtype != rho.dtype or potential.device != rho.device
            or not torch.isfinite(potential).all()):
        raise ValueError("ExternalField must return finite V_ext matching rho's shape, dtype and device.")
    return dict(data, V_ext=potential.detach().clone())


def _work(start, end):
    """Trapezoidal external work in beta units (exact for a jump at fixed rho)."""
    volume = grid_spacing_tensor(start["grid_spacing"], dtype=start["rho"].dtype,
                                 device=start["rho"].device).prod()
    return (float(start["beta"]) * volume *
            (0.5 * (start["rho"] + end["rho"]) * (end["V_ext"] - start["V_ext"])).sum()).detach()
