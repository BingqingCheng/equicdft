"""Run the frozen LJ-paper-v1 rho -> beta*V_ext regression."""

import torch

from .common import check_thresholds, load_case, print_metrics, r2, rmse


def evaluate_forward():
    model, solver, frame = load_case()
    result = solver.evaluate(frame, compute_c1=True)
    rho = result["rho"].detach()
    target = frame["beta"].to(rho) * frame["V_ext"].to(rho)
    wavelength = model.thermal_wavelength.to(rho)
    raw_prediction = result["c1"].detach() - torch.log(
        torch.clamp(
            rho * wavelength.pow(3),
            min=torch.finfo(rho.dtype).tiny,
        )
    )
    mask = rho > model.rho_min
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
