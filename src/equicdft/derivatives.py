"""Generic derivatives of scalar outputs with respect to grid fields."""

import torch


def compute_grid_derivative(
    scalar_output: torch.Tensor,
    grid_field: torch.Tensor,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> torch.Tensor:
    """Differentiate a scalar output with respect to a grid field.

    This function applies no physical sign or grid-volume convention. Such
    transformations belong to the model output that gives the derivative its
    physical meaning. ``scalar_output`` must already be a scalar; batch or
    component reductions are the caller's responsibility. If ``allow_unused``
    is ``True``, an output independent of ``grid_field`` has a zero derivative
    instead of raising an exception. With ``create_graph=True``, that zero
    retains a zero-valued connection to a differentiable scalar output.
    """

    if allow_unused and not scalar_output.requires_grad:
        return torch.zeros_like(grid_field)

    derivative = torch.autograd.grad(
        scalar_output,
        grid_field,
        create_graph=create_graph,
        allow_unused=allow_unused,
    )[0]
    if derivative is not None:
        return derivative

    derivative = torch.zeros_like(grid_field)
    if create_graph:
        derivative = derivative + 0.0 * scalar_output
    return derivative
