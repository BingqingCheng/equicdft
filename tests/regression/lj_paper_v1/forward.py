"""Run the frozen LJ-paper-v1 rho -> beta*V_ext regression."""

import torch

from equicdft.functional.thermodynamics import beta_external_potential

from .common import check_thresholds, load_case, print_metrics, r2, rmse


def evaluate_forward():
    model, solver, frame = load_case()
    result = solver.evaluate(frame, compute_derivatives=True)
    rho = result["rho"].detach()
    target = frame["beta"].to(rho) * frame["V_ext"].to(rho)
    # beta*V_ext up to the chemical-potential gauge, which is fixed below by
    # matching the masked mean of the target.
    raw_prediction = beta_external_potential(
        rho,
        result["c1"].detach(),
        model.thermal_wavelength.to(rho),
    )
    mask = rho > 1.0e-3  # Frozen regression mask, not a model parameter.
    gauge_offset = torch.mean(target[mask] - raw_prediction[mask])
    prediction = raw_prediction + gauge_offset
    metrics = {
        "forward_rmse": rmse(target[mask], prediction[mask]),
        "forward_r2": r2(target[mask], prediction[mask]),
        "gauge_offset": float(gauge_offset),
        "masked_points": int(torch.sum(mask)),
        "total_points": int(rho.numel()),
    }
    return metrics


def main():
    metrics = evaluate_forward()
    print_metrics(metrics)
    check_thresholds("forward", metrics)


if __name__ == "__main__":
    main()
