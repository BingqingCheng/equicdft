"""Deterministic canonical fixed-dipole benchmark; no simulation or fitting.

Run from repository root:
  PYTHONPATH=src python examples/polarization_density/noninteracting.py
The JSON printed to stdout includes all cases and both initializations.
"""

import json

import numpy as np
import torch

from equicdft import PolarizationSolver


def exact_solution(v, h_vector, number, volume, moment):
    """Independent NumPy orientation-integral solution on this discrete grid."""
    h = np.linalg.norm(h_vector, axis=-1)
    safe = np.where(h == 0, 1., h)
    log_z = np.where(h == 0, 0., safe + np.log(-np.expm1(-2*safe)) - np.log(2*safe))
    mean = np.where(h == 0, 0., 1/np.tanh(safe) - 1/safe)
    # Avoid cancellation of the two 1/h terms in the independent weak-field
    # reference using direct angular quadrature, rather than library series.
    weak = (h > 0) & (h < 1e-3)
    nodes, angular_weights = np.polynomial.legendre.leggauss(64)
    boltzmann = np.exp(h[weak, None]*nodes)
    mean[weak] = (boltzmann @ (angular_weights*nodes))/(boltzmann @ angular_weights)
    log_weight = -v + log_z
    weight = np.exp(log_weight - log_weight.max())
    rho = number*weight/(volume*weight.sum())
    polar = (moment*rho*mean/safe)[..., None]*h_vector
    return torch.from_numpy(rho), torch.from_numpy(polar)


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(37)
    shape = (8, 6, 4)
    spacing, beta, moment, number = 0.8, 1.4, 1.7, 60.
    xyz = torch.cartesian_prod(*(torch.arange(size) for size in shape))
    phase = 2*torch.pi*(xyz + 0.5)/torch.tensor(shape)
    v = (0.7*phase[:, 0].cos() + 0.3*(phase[:, 1] + phase[:, 2]).sin())[:, None]
    # Gradient of a periodic scalar potential with cross-coordinate modes.
    h = torch.stack((phase[:, 0].sin(),
                     0.7*(phase[:, 1] + phase[:, 2]).cos(),
                     0.7*shape[1]/shape[2]*(phase[:, 1] + phase[:, 2]).cos()), dim=-1)[:, None, :]
    zero_v, zero_h = torch.zeros_like(v), torch.zeros_like(h)
    uniform = torch.ones_like(h)*torch.tensor([0.5, -0.8, 1.1])
    cases = [("zero", zero_v, zero_h), ("scalar", v, zero_h),
             ("uniform-electric", zero_v, uniform), ("electric", zero_v, h),
             ("coupled", v, h), ("weak", v, 1e-5*h), ("strong", v, 20*h)]
    records = []
    for name, potential, field in cases:
        data = {"V_ext": potential/beta, "E_ext": field/(beta*moment),
                "beta": torch.tensor(beta), "grid_spacing": torch.full((3,), spacing)}
        reference_rho, reference_p = exact_solution(potential.numpy(), field.numpy(), number, spacing**3, moment)
        pair = []
        for start in ("uniform", "perturbed"):
            initial = {}
            if start == "perturbed":
                rho = torch.exp(0.4*torch.randn_like(potential))
                rho *= number/(spacing**3*rho.sum())
                fraction = 0.15*torch.randn_like(field)
                fraction /= 1 + fraction.norm(dim=-1, keepdim=True)
                initial = {"initial_rho": rho, "initial_polarization": moment*rho[..., None]*fraction}
            result = PolarizationSolver(moment).solve(data, number, tolerance_residual=1e-8, **initial)
            density_error = float((result["rho"]-reference_rho).abs().max()/reference_rho.mean())
            polar_error = float((result["dipole_density"]-reference_p).abs().max()/(moment*reference_rho.mean()))
            record = {"case": name, "start": start, "converged": result["converged"],
                      "iterations": result["iterations"],
                      "maximum_residual": float(result["maximum_residual"]),
                      "scaled_density_error": density_error,
                      "scaled_polarization_error": polar_error,
                      "particle_number_error": float(result["particle_number_error"].abs().max()),
                      "maximum_alignment": float(result["maximum_alignment"])}
            records.append(record)
            pair.append(result)
            if not result["converged"] or max(density_error, polar_error) > 1e-7:
                raise AssertionError(record)
        torch.testing.assert_close(pair[0]["rho"], pair[1]["rho"], rtol=0, atol=1e-7*reference_rho.mean().item())
        torch.testing.assert_close(pair[0]["dipole_density"], pair[1]["dipole_density"], rtol=0, atol=1e-7*moment*reference_rho.mean().item())
    print(json.dumps({"grid_size": shape, "spacing": spacing, "beta": beta,
                      "dipole_magnitude": moment, "particle_number": number,
                      "seed": 37, "dtype": "float64", "results": records}, indent=2))


if __name__ == "__main__":
    main()
