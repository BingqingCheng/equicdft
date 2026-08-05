"""Common interface for additive free-energy readouts."""

from typing import Dict

import torch
from torch import nn


class EnergyReadout(nn.Module):
    """Neural readout that supplies one scalar functional contribution."""

    requires_local_features = False
    requires_state_features = False
    def energy(
        self,
        context: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return one scalar energy per complete density field."""

        raise NotImplementedError
