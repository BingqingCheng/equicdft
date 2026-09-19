"""Analytic model fixtures using the intrinsic model/solver contract."""

import torch
from torch import nn
from equicdft import IdealGas


class IdealOnly(nn.Module):
    """No fitted parameters; isolates solver numerics from representation code."""

    def __init__(self, ideal=None):
        super().__init__()
        self.ideal = IdealGas() if ideal is None else ideal

    def forward(self, data, compute_c1=True, compute_polarization_derivative=False,
                compute_c2=False):
        rho = data["rho"]
        outputs = self.ideal.evaluate(data, data["grid_spacing"].prod())
        outputs["beta_F_exc"] = rho.sum(dim=(-2, -1)) * 0
        outputs["beta_F"] = outputs["beta_F_id"]
        if compute_c1:
            outputs["c1"] = torch.zeros_like(rho)
            outputs["density_derivative_excess"] = -outputs["c1"]
            outputs["density_derivative"] = outputs["ideal_density_derivative"]
        if compute_polarization_derivative:
            outputs["polarization_derivative_excess"] = torch.zeros_like(data["dipole_density"])
            outputs["polarization_derivative"] = outputs["ideal_polarization_derivative"]
        return outputs


def with_ideal(model, ideal):
    """Attach the independently declared ideal reference to an excess fixture."""
    if model is None:
        return IdealOnly(ideal)
    model.ideal = ideal
    return model
