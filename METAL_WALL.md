# MetalWall

`MetalWall` adds polarizable Gaussian electrode sites to a liquid model that
already contains a charge-factorized Coulomb `LongRangeReadout`. The liquid
Coulomb readout remains the single owner of species charges, the Coulomb
amplitude, reciprocal split, and liquid-generated potential. `MetalWall`
passes it the electrode positions, receives the liquid potential at those
positions, relaxes the electrode charges, and contributes the electrode energy
to the functional.

## Adding a conducting metal

First load the electrode coordinates and charge constraints. For the usual
constant-field cell, all sites belong to one neutral constrained group:

```python
metal_sites = read_metal_sites(
    "metal.extxyz",
    total_charge=0.0,
)
```

Here `metal.extxyz` only needs atomic positions. A scalar `total_charge`
assigns every site to group 0. The group is a charge constraint, not a
geometrical electrode label: sites in two separate slabs may belong to this
same group. Their lower- and upper-slab charges can later be summed from
`metal_site_q` using their positions.

For several independently constrained groups, label every site and give one
total charge per label. Labels may be stored in the EXTXYZ
`metal_site_groups` array:

```python
metal_sites = read_metal_sites(
    "metal.extxyz",
    total_charge={0: -0.5, 1: 0.5},
)
```

If the file contains labels 0 and 1, these totals constrain groups 0 and 1,
respectively. The dictionary maps each label directly to its total charge, so
there is no separate group ordering. Labels can instead be supplied directly,
for example `site_groups=[0, 0, 1, 1]`. Multiple independently constrained
groups are currently supported with zero applied field; constant-field mode
requires one constrained group.

Then replace the model's liquid Coulomb readout with `MetalWall`:

```python
liquid_coulomb = model.readout[2]
model.readout[2] = MetalWall(
    liquid_coulomb=liquid_coulomb,
    metal_sites=metal_sites,
    metal_sigma=metal_sigma,
    external_field=(0.0, 0.0, E_z),
)
```

`MetalWall` wraps and evaluates the existing `liquid_coulomb` readout. Replace
the original entry as shown above. If the original entry is left in the model
and `MetalWall` is appended as another entry, the liquid long-range energy is
evaluated twice.

Constructor arguments:

| Argument | Meaning |
|---|---|
| `liquid_coulomb` | One charge-factorized `LongRangeReadout` with a Coulomb kernel |
| `metal_sites` | Coordinates, site groups, and prescribed group charges from `read_metal_sites` |
| `metal_sigma` | Gaussian standard deviation of each electrode site, in model length units |
| `external_field` | Applied field acting on electrode charges, in model energy per charge per length; default zero |
| `tolerance` | Charge and equipotential residual tolerance; default `1e-6` |

Electrode coordinates and liquid `grid_center` use the same physical coordinate
system. Electrode sites need not coincide with liquid grid points. A separate
`excluded_mask` marks grid points at which liquid density must remain zero.

A nonzero `external_field` currently requires exactly one constrained electrode
group. `MetalWall` automatically places the periodic sawtooth discontinuity in
the largest site-free gap along each field direction and unwraps all electrode
sites onto the remaining contiguous branch. No field origin is supplied by the
caller. With zero field, multiple independently constrained groups remain
supported.

## Electrostatic coupling

For a set of positions, the liquid Coulomb readout forms one charge spectrum.
Its damped kernel supplies the liquid–liquid long-range free energy. The full
Coulomb kernel is used only to evaluate the liquid potential at electrode
positions:

$$
\widehat{\phi}_{\mathrm{LR}} = K_{\mathrm{LR}}\widehat{q}_{\mathrm{liq}},
\qquad
\widehat{\phi}_{\mathrm{SR}}
= (K_{\mathrm{full}}-K_{\mathrm{LR}})\widehat{q}_{\mathrm{liq}}.
$$

The complementary analytical kernel does not enter the liquid–liquid free
energy because the learned short-range branch represents that physics. It is
used only to complete the liquid–metal site potential
$\phi_{\mathrm{LR}}+\phi_{\mathrm{SR}}$. The electrode positions and their
`metal_sigma` are arguments to this evaluation; MetalWall does not define a
separate liquid charge width.

Each voxel carries integrated charge
$p_g=\Delta V\sum_i z_i\rho_{gi}$. The relaxed electrode charges solve

$$
Aq+C\lambda=-(b+v), \qquad C^{\mathsf T}q=Q,
$$

where `A` is the metal–metal interaction matrix, `b` is the liquid potential at
the sites, `v` is the applied-field potential, and `C` imposes the prescribed
group totals `Q`. The geometry-dependent system is factorized once and reused
as the liquid density changes.

The corresponding electrode contribution is

$$
U_{\mathrm{electrode}}
=q^{\mathsf T}b+\frac{1}{2}q^{\mathsf T}Aq+q^{\mathsf T}v.
$$

The liquid charge field and its integrated scalar remain owned by the Coulomb
readout. MetalWall receives
$Q_{\mathrm{liquid}}=\Delta V\sum_{g,i}z_i\rho_{gi}$ to require
$Q_{\mathrm{liquid}}+\sum_a Q_a=0$ in the periodic cell; it does not add a
neutralizing background or return liquid charge fields.

Liquid–metal and metal–metal interactions use the same finite reciprocal grid.
The zero mode is omitted, the Gaussian metal self interaction is retained, and
off-grid electrode coordinates are evaluated spectrally rather than rounded or
interpolated onto liquid voxels. Electrode relaxation remains differentiable
with respect to liquid density. Check reciprocal-grid and `metal_sigma`
convergence for scientific calculations.

## Using a 2 V finite field in a density solve

For a cell of length `Lz`, convert the desired voltage to the model's field
units and apply the same field to the liquid and electrode charges:

```python
delta_phi_V = 2.0
Lz = float(data["grid_size"][2] * data["grid_spacing"][2])
E_z = -delta_phi_V / (energy_unit_eV * Lz)

charges = liquid_coulomb.charges.to(data["V_ext"])
z = data["grid_center"][:, 2, None]
data["V_ext"] = data["V_ext"] - z * E_z * charges

model.readout[2] = MetalWall(
    liquid_coulomb=liquid_coulomb,
    metal_sites=metal_sites,
    metal_sigma=metal_sigma,
    external_field=(0.0, 0.0, E_z),
)

result = GridSolver(model).solve(
    data,
    method="minimize",
    particle_numbers=particle_numbers,
    initial_rho=initial_rho,
    homogeneous_axes=["x", "y"],
)
```
