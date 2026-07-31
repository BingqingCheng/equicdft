"""Generic derivatives of scalar outputs with respect to grid fields."""

import torch


def compute_grid_derivative(
    scalar_output: torch.Tensor,
    grid_field: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """Differentiate a scalar output with respect to a grid field.

    This function applies no physical sign or grid-volume convention. Such
    transformations belong to the model output that gives the derivative its
    physical meaning. ``scalar_output`` must already be a scalar; batch or
    component reductions are the caller's responsibility.
    """

    return torch.autograd.grad(
        scalar_output,
        grid_field,
        create_graph=create_graph,
    )[0]
