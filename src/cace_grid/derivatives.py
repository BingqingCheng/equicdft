"""Functional derivatives of the learned excess free energy."""

import torch


def compute_c1(
    beta_F_exc: torch.Tensor,
    rho: torch.Tensor,
    grid_spacing: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """Return ``c1 = -(1 / cell_volume) * d(beta_F_exc) / d(rho)``."""

    derivative = torch.autograd.grad(
        beta_F_exc.sum(),
        rho,
        create_graph=create_graph,
    )[0]
    cell_volume = torch.prod(grid_spacing, dim=-1)
    return -derivative / cell_volume[..., None, None]
