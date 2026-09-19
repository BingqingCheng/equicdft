"""One place for quantities given per density component.

Several physical inputs are "a positive scalar, or one value per species":
thermal wavelengths, molecular dipole magnitudes, density caps, particle
numbers, chemical potentials. They are consumed in three different ways, so
this module offers exactly three helpers rather than one per call site:

``species_parameter``
    construction time, with no field to measure against. Used for buffers,
    so it preserves the precision of a Python float until the module is
    cast, and returns a detached copy.
``species_vector``
    call time, with a known component count and a reference tensor whose
    dtype and device the result must share.
``broadcast_species_field``
    call time, for a value that may also vary per field in a batch, checked
    by broadcasting against a density's batch and species axes.
"""

import torch

__all__ = [
    "broadcast_species_field",
    "species_parameter",
    "species_vector",
]


def species_parameter(value, name: str) -> torch.Tensor:
    """Return a detached positive scalar or species vector for a buffer.

    A Python number is kept in double precision so a physical constant is
    not rounded before the owning module is cast to its working dtype; a
    tensor keeps the precision it was given.
    """

    tensor = (
        value if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=torch.float64)
    )
    if tensor.is_complex():
        raise ValueError(name + " must be real")
    if (tensor.ndim > 1 or not tensor.numel()
            or not torch.isfinite(tensor).all() or torch.any(tensor <= 0)):
        raise ValueError(name + " must be a positive scalar or species vector")
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    return tensor.detach().clone()


def species_vector(
    value,
    n_types: int,
    reference: torch.Tensor,
    name: str,
    *,
    positive: bool = False,
) -> torch.Tensor:
    """Return ``[n_types]`` on ``reference``'s dtype and device.

    A scalar is expanded to every component. ``positive=True`` additionally
    requires every value to be finite and strictly greater than zero, which
    physical magnitudes need and signed quantities such as a chemical
    potential do not.
    """

    values = torch.as_tensor(
        value, dtype=reference.dtype, device=reference.device,
    ).reshape(-1)
    if values.numel() == 1:
        values = values.repeat(n_types)
    if values.shape != (n_types,):
        raise ValueError("{} must contain one value per type".format(name))
    if positive and (
        not torch.isfinite(values).all() or torch.any(values <= 0)
    ):
        raise ValueError(name + " must be finite and positive")
    return values


def broadcast_species_field(value, rho: torch.Tensor, name: str) -> torch.Tensor:
    """Return a value broadcastable over ``rho``'s field and species axes.

    Unlike :func:`species_vector` this keeps a leading batch shape, so one
    batch may carry a different value per field. The result is not expanded,
    only checked, so the caller's arithmetic broadcasts as before.
    """

    raw = torch.as_tensor(value, device=rho.device)
    if raw.is_complex() or not torch.isfinite(raw).all() or torch.any(raw <= 0):
        raise ValueError(name + " must be finite, positive and real")
    values = raw.to(rho)
    if values.ndim == 0:
        values = values.reshape(1)
    target = rho.shape[:-2] + (rho.shape[-1],)
    if torch.broadcast_shapes(values.shape, target) != target:
        raise ValueError(name + " must match field and species axes")
    return values
