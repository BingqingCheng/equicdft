"""Canonical minimization for an explicit fixed-magnitude point-dipole model."""

import torch
from torch import nn

from ._argument_checks import positive_integer, positive_scalar
from ._grid import voxel_volume
from .polarization_ideal import (
    FixedDipoleIdeal,
    _positive_components,
    dipole_alignment,
    log_sinhc,
)


def _accept_trial(current, trial, volume):
    """Armijo energy descent, with residual improvement at energy roundoff."""
    rho_change = trial["rho"] - current["rho"]
    polarization_change = trial["dipole_density"] - current["dipole_density"]
    linear_change = volume * (
        (current["density_residual"] * rho_change).sum()
        + (current["polarization_residual"] * polarization_change).sum()
    )
    epsilon = torch.finfo(current["rho"].dtype).eps
    roundoff = 32 * epsilon * max(1., abs(float(current["beta_A"])))
    energy_bound = current["beta_A"] + 1e-4 * linear_change + roundoff
    energy_ok = trial["beta_A"] <= energy_bound
    resolved_descent = linear_change < -roundoff
    residual_improves = trial["maximum_residual"] < current["maximum_residual"]
    # Small objective changes alone do not establish physical convergence.
    return bool(
        torch.isfinite(trial["maximum_residual"])
        and linear_change <= 0 and energy_ok
        and (resolved_descent or residual_improves)
    )


class PolarizationSolver:
    """Minimize ideal + optional excess + integral(rho*V_ext - P.E_ext).

    One unbatched, fully accessible field is supported, with any number of
    fixed-dipole species. V_ext [grid,species] is energy/particle and E_ext
    [grid,species,3] is energy/dipole. beta is scalar inverse energy. The model,
    if supplied, uses the GridCACEModel energy-only API and returns beta_F_exc.
    The solver does not differentiate through its optimization trajectory.
    """

    def __init__(self, dipole_magnitude, model=None, thermal_wavelength=1.0):
        if model is not None and not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module or None")
        self.ideal = FixedDipoleIdeal(dipole_magnitude, thermal_wavelength)
        self.model = model

    def _inputs(self, data):
        potential = data["V_ext"]
        field = data["E_ext"]
        if (not isinstance(potential, torch.Tensor) or potential.ndim != 2
                or not potential.is_floating_point() or min(potential.shape) < 1):
            raise ValueError("V_ext must be floating [grid,species]")
        if (not isinstance(field, torch.Tensor)
                or field.shape != (*potential.shape, 3)
                or field.dtype != potential.dtype or field.device != potential.device):
            raise ValueError("E_ext must match V_ext dtype/device with shape [grid,species,3]")
        if not torch.isfinite(potential).all() or not torch.isfinite(field).all():
            raise ValueError("external fields must be finite")
        if "excluded_mask" in data and torch.as_tensor(data["excluded_mask"]).any():
            raise ValueError("PolarizationSolver currently requires fully accessible grids")
        beta = torch.as_tensor(
            data["beta"], dtype=potential.dtype, device=potential.device
        )
        if beta.ndim or not torch.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be a finite positive scalar")
        if self.model is not None and hasattr(self.model, "boltzmann_constant"):
            temperature = torch.as_tensor(
                data["temperature"], dtype=potential.dtype, device=potential.device
            )
            k_b = self.model.boltzmann_constant.to(potential)
            if (temperature.ndim or not torch.isfinite(temperature) or temperature <= 0
                    or not torch.allclose(beta*k_b*temperature, beta.new_ones(()),
                                          atol=0, rtol=1e-6)):
                raise ValueError("beta must agree with model k_B and data temperature")
        spacing = torch.as_tensor(
            data["grid_spacing"], dtype=potential.dtype, device=potential.device
        )
        if spacing.numel() != 3 or not torch.isfinite(spacing).all() or torch.any(spacing <= 0):
            raise ValueError("grid_spacing must contain three positive finite values")
        if not torch.allclose(
            spacing, spacing.flatten()[0].expand_as(spacing), rtol=1e-7, atol=0
        ):
            raise ValueError("PolarizationSolver requires cubic voxels")
        volume = voxel_volume(spacing.reshape(3))
        moment = _positive_components(
            self.ideal.dipole_magnitude, potential, "dipole_magnitude"
        )
        return beta, volume, moment

    def _energy(self, data, rho, polarization, beta, volume):
        ideal = self.ideal(rho, polarization, volume)["beta_F_id"]
        excess = rho.new_zeros(())
        if self.model is not None:
            values = dict(data, rho=rho, dipole_density=polarization)
            excess = self.model(
                values, compute_c1=False, compute_c2=False,
                compute_polarization_derivative=False,
            )["beta_F_exc"]
            if excess.ndim:
                raise ValueError("model must return scalar beta_F_exc for one field")
        external = beta * volume * (
            (rho * data["V_ext"]).sum() - (polarization * data["E_ext"]).sum()
        )
        return ideal + excess + external

    @torch.enable_grad()
    def evaluate(self, data):
        """Return physical Euler residuals at independently variable rho and P.

        Density residual subtracts a spatial constant for each species
        (canonical gauge). Polarization residual is also reported multiplied
        by m, giving a dimensionless vector comparable across dipole units.
        """
        beta, volume, moment = self._inputs(data)
        rho = data["rho"].detach().clone().requires_grad_(True)
        polarization = data["dipole_density"].detach().clone().requires_grad_(True)
        if rho.shape != data["V_ext"].shape:
            raise ValueError("rho must match V_ext shape")
        energy = self._energy(data, rho, polarization, beta, volume)
        if not torch.isfinite(energy):
            raise FloatingPointError("nonfinite polarization objective")
        derivative_rho, derivative_p = torch.autograd.grad(energy, (rho, polarization))
        local_mu = derivative_rho / volume
        chemical_potential = local_mu.mean(dim=0)
        density_residual = local_mu - chemical_potential
        polarization_residual = derivative_p / volume
        scaled = polarization_residual * moment[..., None]
        maximum = torch.maximum(density_residual.abs().max(), scaled.abs().max())
        result = {
            "rho": rho,
            "dipole_density": polarization,
            "beta_A": energy,
            "beta_mu": chemical_potential,
            "density_residual": density_residual,
            "polarization_residual": polarization_residual,
            "scaled_polarization_residual": scaled,
            "maximum_residual": maximum,
            "particle_numbers": volume * rho.sum(dim=0),
            "maximum_alignment": (polarization.norm(dim=-1) / (rho * moment)).max(),
        }
        return {key: value.detach() for key, value in result.items()}

    @torch.enable_grad()
    def solve(
        self, data, particle_numbers, initial_rho=None, initial_polarization=None,
        max_iter=400, tolerance_residual=1e-7, step_size=1.0, max_backtracks=30,
    ):
        """Orientational mirror descent with exact N and |P|<m*rho.

        Default start is uniform/unpolarized. Explicit initial fields take
        priority over data's rho/dipole_density. Density must already have the
        requested N; invalid starts are rejected rather than silently rescaled.
        Convergence requires both physical residuals to pass their tolerance.
        """
        max_iter = positive_integer(max_iter, "max_iter")
        max_backtracks = positive_integer(max_backtracks, "max_backtracks")
        step_size = positive_scalar(step_size, "step_size")
        tolerance_residual = positive_scalar(tolerance_residual, "tolerance_residual")
        _, volume, moment = self._inputs(data)
        potential = data["V_ext"]
        numbers = _positive_components(
            particle_numbers, potential, "particle_numbers"
        ).expand(potential.shape[-1])
        rho0 = initial_rho if initial_rho is not None else data.get("rho")
        if rho0 is None:
            rho0 = torch.ones_like(potential) * numbers / (volume * potential.shape[0])
        else:
            rho0 = torch.as_tensor(
                rho0, dtype=potential.dtype, device=potential.device
            ).detach().clone()
        if rho0.shape != potential.shape:
            raise ValueError("initial_rho must match V_ext shape")
        count_tolerance = 100 * torch.finfo(potential.dtype).eps
        if not torch.allclose(
            volume * rho0.sum(0), numbers, rtol=count_tolerance, atol=0
        ):
            raise ValueError("initial_rho must have the requested particle_numbers")
        p0 = (initial_polarization if initial_polarization is not None
              else data.get("dipole_density"))
        if p0 is None:
            p0 = potential.new_zeros(*potential.shape, 3)
        else:
            p0 = torch.as_tensor(
                p0, dtype=potential.dtype, device=potential.device
            ).detach().clone()
        initial_ideal = self.ideal(rho0, p0, volume)  # validates both fields
        # m * d(beta f_id)/dP is precisely the conjugate alignment vector.
        alignment = (initial_ideal["polarization_derivative"] * moment[..., None]).detach()
        rho, polarization = rho0, p0
        result = self.evaluate(dict(data, rho=rho, dipole_density=polarization))
        history = []
        status = "max_iter"
        for iteration in range(max_iter + 1):
            history.append({
                "iteration": iteration,
                "beta_A": float(result["beta_A"]),
                "maximum_residual": float(result["maximum_residual"]),
            })
            converged = bool(result["maximum_residual"] <= tolerance_residual)
            if converged or iteration == max_iter:
                status = "converged" if converged else "max_iter"
                break
            # The molecular orientation distribution is proportional to
            # rho*exp(a.u)/Z(a). Multiplication by exp(-step*functional_gradient)
            # updates a and log(rho)-log(Z) together. Renormalize each species.
            # This is the same gradient update with or without excess energy.
            step = step_size
            log_z = log_sinhc(alignment.norm(dim=-1))
            accepted = False
            for _ in range(max_backtracks):
                trial_alignment = alignment - step * result["scaled_polarization_residual"]
                log_weights = (
                    rho.log() - step * result["density_residual"]
                    + log_sinhc(trial_alignment.norm(dim=-1)) - log_z
                )
                trial_rho = torch.softmax(log_weights, dim=0) * numbers / volume
                trial_p = (trial_rho * moment)[..., None] * dipole_alignment(trial_alignment)
                # Very large trial steps can underflow rho or round |P|/(m*rho)
                # to one. Backtrack these inadmissible trials without clipping.
                interior = (
                    torch.isfinite(trial_rho).all() and torch.all(trial_rho > 0)
                    and torch.isfinite(trial_p).all()
                    and torch.all(trial_p.norm(dim=-1) < trial_rho * moment)
                )
                if interior:
                    trial = self.evaluate(dict(
                        data, rho=trial_rho, dipole_density=trial_p
                    ))
                    if _accept_trial(result, trial, volume):
                        accepted = True
                        break
                step *= 0.5
            if not accepted:
                status = "line_search_failed"
                break
            history[-1]["accepted_step"] = step
            rho, polarization, alignment = trial_rho, trial_p, trial_alignment
            result = trial
        result.update(
            converged=converged, iterations=iteration, history=history, status=status
        )
        result["particle_number_error"] = result["particle_numbers"] - numbers
        return result
