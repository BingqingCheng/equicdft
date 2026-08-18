"""Run the frozen LJ-paper-v1 fixed-N V_ext -> rho regression."""

import torch

from .common import check_thresholds, load_case, print_metrics, r2, rmse


def evaluate_solve():
    model, solver, frame = load_case()
    solve_data = dict(frame)
    target = solve_data.pop("rho")
    volume_element = model.voxel_volume.to(target)
    particle_numbers = volume_element * torch.sum(target, dim=0)
    result = solver.solve(
        solve_data,
        particle_numbers=particle_numbers,
        method="euler",
        beta_multiplier=0.0,
        max_iter=200,
        tolerance_residual=1.0e-2,
        tolerance_change=1.0e-6,
        residual_density_threshold=model.rho_min,
        maximum_density=2.0,
    )
    prediction = result["rho"].detach()
    predicted_particles = volume_element * torch.sum(prediction, dim=0)
    metrics = {
        "reverse_rmse": rmse(target, prediction),
        "reverse_r2": r2(target, prediction),
        "converged": bool(result["converged"]),
        "iterations": int(result["n_iter"]),
        "evaluations": int(result["n_evaluations"]),
        "max_residual": float(result["max_euler_lagrange_residual"]),
        "particle_number": float(particle_numbers.item()),
        "particle_error": float(
            torch.max(torch.abs(predicted_particles - particle_numbers))
        ),
        "minimum_density": float(torch.min(prediction)),
        "maximum_density": float(torch.max(prediction)),
        "finite_density": bool(torch.all(torch.isfinite(prediction))),
    }
    return metrics


def main():
    metrics = evaluate_solve()
    print_metrics(metrics)
    check_thresholds("solve", metrics)


if __name__ == "__main__":
    main()
