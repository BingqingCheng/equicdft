"""Configure modules from the independent electrostatics oracle case records.

The reference cases store imposed fields alongside geometry for their direct
Fourier calculations. Production calls receive only density/geometry; fields
are configured on MetalWall. Public-constructor tests do not use this helper.
"""

import torch
from equicdft import MetalWall
from equicdft.metalwall import _field_vector


def _run(call, data, *args, **kwargs):
    module = call if isinstance(call, torch.nn.Module) else call.__self__.model
    for wall in module.modules():
        if isinstance(wall, MetalWall):
            for name in ("external_field", "field_origin"):
                value = data.get("metal_" + name, (0., 0., 0.))
                setattr(wall, name, _field_vector(value, name).to(wall.liquid_charges))
    inputs = {k: v for k, v in data.items()
              if k not in ("metal_external_field", "metal_field_origin")}
    return call(inputs, *args, **kwargs)
