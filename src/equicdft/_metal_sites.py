"""Finite-Fourier sampling at fixed sites independent of the liquid grid."""

import math

import torch


class ExplicitSiteGrid:
    """Reuse shifted FFTs for sites sharing a fractional grid offset.

    Coordinates are fixed, double-precision geometry relative to grid zero.
    Only roundoff-equivalent offsets share a transform; arbitrary distinct
    offsets are supported but require more transforms. No nearest-grid charge
    assignment or real-space interpolation is used.
    """

    def __init__(self, positions, shape, spacing, dtype, device):
        self.shape = shape
        self.spacing = spacing
        self.dtype, self.device = dtype, device
        self.voxel_volume = spacing.prod()
        self.n_sites = positions.shape[0]
        # Group in float64 even when density/FFT arithmetic uses float32.
        scaled = positions.detach().cpu().double() / spacing.detach().cpu().double()
        scaled = scaled.remainder(torch.tensor(shape))
        base = scaled.floor().long()
        fraction = scaled - base
        self.offset_tolerance = 32 * torch.finfo(torch.float64).eps * max(shape)
        near_next = fraction > 1 - self.offset_tolerance
        base += near_next.long()
        fraction -= near_next.double()
        base %= torch.tensor(shape)
        flat = (base[:, 0]*shape[1] + base[:, 1])*shape[2] + base[:, 2]
        representatives, members = [], []
        max_error = 0.0
        for site, offset in enumerate(fraction.tolist()):
            for group, representative in enumerate(representatives):
                error = max(abs(a-b) for a, b in zip(offset, representative))
                if error <= self.offset_tolerance:
                    members[group].append(site)
                    max_error = max(max_error, error)
                    break
            else:
                representatives.append(offset)
                members.append([site])
        self.max_offset_error = max_error
        self.groups = []
        for offset, sites in zip(representatives, members):
            index = torch.tensor(sites, dtype=torch.long)
            if torch.unique(flat[index]).numel() != index.numel():
                raise ValueError("duplicate or numerically coincident periodic metal_positions")
            self.groups.append((
                index.to(device), base[index].to(device), flat[index].to(device),
                torch.tensor(offset, dtype=torch.float64, device=device)
                * spacing.to(dtype=torch.float64),
            ))
        self.phases = [self.phase(group[3]) for group in self.groups]

    def phase(self, offset):
        """Symmetric +/- Nyquist extension, equal to native FFT on grid nodes."""
        result = None
        for axis, (size, step) in enumerate(zip(self.shape, self.spacing)):
            wave = 2*math.pi*torch.fft.fftfreq(
                size, d=float(step), dtype=torch.float64, device=self.device,
            )
            angle = wave*offset[axis]
            values = torch.exp(1j*angle)
            if size % 2 == 0:
                # Half of each +/- endpoint: do not retain its imaginary part.
                values[size//2] = torch.cos(angle[size//2]).to(values.dtype)
            view = [1, 1, 1]
            view[axis] = size
            values = values.reshape(view)
            result = values if result is None else result*values
        return result.to(torch.complex128 if self.dtype == torch.float64 else torch.complex64)

    def potential(self, charge, kernel):
        """Liquid voxel charges -> potential conjugate to metal site charges."""
        spectrum = torch.fft.fftn(charge.reshape(self.shape))*kernel
        result = charge.new_zeros(self.n_sites)
        for group, phase in zip(self.groups, self.phases):
            values = torch.fft.ifftn(spectrum*phase).real.reshape(-1)/self.voxel_volume
            result = result.index_copy(0, group[0], values[group[2]])
        return result

    def matrix(self, kernel):
        """Site pair Green function, including the finite Gaussian self term."""
        result = kernel.new_empty((self.n_sites, self.n_sites))
        for a, (ia, base_a, _, delta_a) in enumerate(self.groups):
            for ib, base_b, _, delta_b in self.groups[a:]:
                # At Nyquist phase(a-b) != phase(a)*conj(phase(b)). Using
                # the latter would spuriously change self terms with position.
                response = torch.fft.ifftn(
                    kernel*self.phase(delta_a-delta_b),
                ).real/self.voxel_volume
                offsets = tuple(
                    (base_a[:, axis, None]-base_b[None, :, axis]) % size
                    for axis, size in enumerate(self.shape)
                )
                block = response[offsets]
                result[ia[:, None], ib[None, :]] = block
                result[ib[:, None], ia[None, :]] = block.T
        return result
