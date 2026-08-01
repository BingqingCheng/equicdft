"""Evaluate and minimize learned grid density functionals."""

from typing import Any, Dict, Optional, Sequence, Union

import torch
from torch import nn


class GridSolver:
    """Evaluate prescribed densities or solve for an equilibrium density.

    The wrapped model supplies the intrinsic excess functional. This class
    adds the exact ideal-gas, external-potential, and chemical-potential terms.

    ``evaluate(data)`` requires ``rho`` and returns all quantities supported by
    the available fields. ``solve(data)`` requires ``V_ext`` and either ``mu``
    or fixed ``particle_numbers``. The initial solver supports one complete,
    unbatched field; prescribed-density evaluation also supports batches.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self.device = (
            _module_device(model)
            if device is None
            else torch.device(device)
        )
        self.model.to(self.device)

    def evaluate(
        self,
        data: Dict[str, Any],
        compute_c1: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a supplied density and its available thermodynamics."""

        data = self._move_data(data)
        rho = data["rho"]
        if torch.any(rho < 0.0).item():
            raise ValueError("rho must be nonnegative")

        outputs = self.model(data, compute_c1=compute_c1)
        result = {
            key: data[key]
            for key in ("rho", "V_ext", "mu", "temperature", "beta")
            if key in data
        }
        result.update(outputs)

        cell_volume = torch.prod(data["grid_spacing"], dim=-1)
        thermal_wavelength = data.get(
            "thermal_wavelength",
            torch.ones(
                rho.shape[-1],
                dtype=rho.dtype,
                device=rho.device,
            ),
        )
        log_density = _log_dimensionless_density(
            rho,
            thermal_wavelength,
        )
        ideal_density = torch.where(
            rho > 0.0,
            rho * (log_density - 1.0),
            torch.zeros_like(rho),
        )
        result["beta_F_id"] = cell_volume * torch.sum(
            ideal_density,
            dim=(-2, -1),
        )
        result["beta_F"] = result["beta_F_id"] + result["beta_F_exc"]

        if "V_ext" in data:
            beta = data["beta"][..., None, None]
            result["beta_V_ext"] = cell_volume * torch.sum(
                rho * beta * data["V_ext"],
                dim=(-2, -1),
            )

        if "V_ext" in data and "mu" in data:
            beta = data["beta"][..., None, None]
            beta_mu_N = cell_volume * torch.sum(
                rho * beta * data["mu"][..., None, :],
                dim=(-2, -1),
            )
            result["beta_mu_N"] = beta_mu_N
            result["beta_Omega"] = (
                result["beta_F"] + result["beta_V_ext"] - beta_mu_N
            )
            if "local_chemical_potential" in result:
                result["euler_lagrange_residual"] = (
                    result["local_chemical_potential"]
                    - beta * data["mu"][..., None, :]
                )

        return result

    def solve(
        self,
        data: Dict[str, Any],
        initial_rho: Optional[torch.Tensor] = None,
        particle_numbers: Optional[
            Union[float, Sequence[float], torch.Tensor]
        ] = None,
        max_iter: int = 200,
        tolerance_grad: float = 1.0e-7,
        tolerance_change: float = 1.0e-9,
    ) -> Dict[str, Any]:
        """Minimize the grand-canonical or fixed-particle-number objective."""

        data = self._move_data(data)
        V_ext = data["V_ext"]
        if V_ext.ndim != 2:
            raise ValueError("solve currently accepts one unbatched field")

        n_types = V_ext.shape[-1]
        thermal_wavelength = _component_tensor(
            data.get("thermal_wavelength", 1.0),
            n_types,
            V_ext,
            "thermal_wavelength",
        )
        cell_volume = torch.prod(data["grid_spacing"])

        fixed_N = None
        if particle_numbers is None:
            mu = _component_tensor(data["mu"], n_types, V_ext, "mu")
        else:
            fixed_N = _component_tensor(
                particle_numbers,
                n_types,
                V_ext,
                "particle_numbers",
            )
            if torch.any(fixed_N <= 0.0).item():
                raise ValueError("particle_numbers must be positive")
            mu = None

        if initial_rho is None:
            initial_rho = data.get("rho")
        if initial_rho is not None:
            initial_rho = torch.as_tensor(
                initial_rho,
                dtype=V_ext.dtype,
                device=V_ext.device,
            )
            if initial_rho.shape != V_ext.shape:
                raise ValueError("initial_rho must have the same shape as V_ext")
            if torch.any(initial_rho <= 0.0).item():
                raise ValueError("initial_rho must be positive")
            initial_u = torch.log(
                initial_rho * thermal_wavelength[None, :] ** 3
            )
        elif fixed_N is None:
            initial_u = data["beta"] * (mu[None, :] - V_ext)
        else:
            initial_u = -data["beta"] * V_ext

        u = nn.Parameter(initial_u.detach().clone())
        optimizer = torch.optim.LBFGS(
            [u],
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn="strong_wolfe",
        )
        objective_history = []

        def density_from_u() -> torch.Tensor:
            if fixed_N is None:
                return torch.exp(u) / thermal_wavelength[None, :] ** 3
            return (
                fixed_N[None, :]
                * torch.softmax(u, dim=0)
                / cell_volume
            )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            trial_data = dict(data)
            trial_data["rho"] = density_from_u()
            evaluation = self.evaluate(trial_data, compute_c1=False)
            objective = (
                evaluation["beta_Omega"]
                if fixed_N is None
                else evaluation["beta_F"] + evaluation["beta_V_ext"]
            )
            if not torch.isfinite(objective).item():
                raise ValueError("the equilibrium objective became non-finite")
            u.grad = torch.autograd.grad(objective, u)[0]
            objective_history.append(objective.detach().item())
            return objective

        was_training = self.model.training
        self.model.eval()
        try:
            optimizer.step(closure)
            final_data = dict(data)
            final_data["rho"] = density_from_u().detach()
            result = self.evaluate(final_data, compute_c1=True)
        finally:
            self.model.train(was_training)

        if fixed_N is not None:
            result["euler_lagrange_residual"] = (
                result["local_chemical_potential"]
                - result["average_chemical_potential"][..., None, :]
            )

        state = optimizer.state[u]
        result["converged"] = (
            u.grad is not None
            and torch.max(torch.abs(u.grad)).item() <= tolerance_grad
        )
        result["n_iter"] = int(state.get("n_iter", 0))
        result["n_evaluations"] = int(state.get("func_evals", 0))
        result["objective_history"] = objective_history
        return result

    def _move_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in data.items()
        }


def _log_dimensionless_density(
    rho: torch.Tensor,
    thermal_wavelength: torch.Tensor,
) -> torch.Tensor:
    tiny = torch.finfo(rho.dtype).tiny
    return torch.log(
        torch.clamp(rho, min=tiny)
        * thermal_wavelength[..., None, :] ** 3
    )


def _component_tensor(
    values: Union[float, Sequence[float], torch.Tensor],
    n_types: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    values = torch.as_tensor(
        values,
        dtype=reference.dtype,
        device=reference.device,
    ).reshape(-1)
    if values.numel() == 1:
        values = values.repeat(n_types)
    if values.shape != (n_types,):
        raise ValueError("{} must contain one value per type".format(name))
    return values


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device
