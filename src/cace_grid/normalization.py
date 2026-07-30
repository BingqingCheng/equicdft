"""Fixed feature normalization computed from the training split."""

import torch


def compute_rms_feature_scale(
    B: torch.Tensor,
    minimum_scale: float = 1.0e-12,
) -> torch.Tensor:
    """Return one detached RMS scale for each flattened ``B`` feature."""

    B_flat = B.detach().flatten(start_dim=-3)
    sample_dimensions = tuple(range(B_flat.ndim - 1))
    scale = torch.sqrt(torch.mean(B_flat.square(), dim=sample_dimensions))
    return torch.clamp(scale, min=minimum_scale)
