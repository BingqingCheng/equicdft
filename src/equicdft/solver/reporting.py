"""Common final solve metadata; physical residuals stay with the drivers."""


def finalize_result(result, state, *, method, beta_multiplier, converged):
    """Attach counters/history without recomputing energies or residuals.

    Drivers supply their final convergence decision after any projections.
    A small-change stop before the iteration budget is exhausted is stalled;
    max_iter means the driver actually exhausted its budget.
    """

    if converged:
        status = "converged"
    elif "status" in state:
        status = state["status"]
    elif state["line_search_failures"]:
        status = "line_search_failed"
    elif state.get("stalled", False):
        status = "stalled"
    else:
        status = "max_iter"
    result.update(
        converged=converged,
        status=status,
        solver_method=method,
        solver_beta_multiplier=beta_multiplier,
    )
    for key in (
        "n_iter", "n_evaluations", "objective_history", "line_search_failures",
        "final_relative_density_change",
    ):
        result[key] = state[key]
    for key, default in (
        ("mixing_backtracks", 0),
        ("final_mixing", None),
        ("solver_anderson", False),
        ("anderson_attempts", 0),
        ("anderson_accepted", 0),
        ("anderson_rejected", 0),
        ("anderson_resets", 0),
    ):
        result[key] = state.get(key, default)
    for key in (
        "history",
        "final_relative_polarization_change",
        "final_relative_alignment_change",
    ):
        if key in state:
            result[key] = state[key]
    return result
