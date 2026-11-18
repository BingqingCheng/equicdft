# Evaluate a model on another grid

From the repository root:

```sh
python examples/grid_transfer/evaluate.py model.pt fine.extxyz \
    --spacing 0.125 --center-normalization reference --output evaluation.pt
```

`model.pt` must be a versioned `save_model` export. The first EXTXYZ frame
must already contain a density on the requested grid, with temperature and
grid metadata. Nothing is interpolated and no equilibrium solve is run.

The script returns `beta_F_exc` and `c1` (shape `[n_grid, n_types]`), together
with grid and input metadata. It works with multiple density components and
supported Gaussian radial channels. FFT avoids an explicit neighbor table.
Add `--device cuda` for large fields on a GPU.

`--center-normalization reference` selects the exploratory reference-grid
compensation. Omit it for ordinary strict center exclusion. Physical cutoff,
Gaussian widths and moment units are preserved in either case; the original
model is not modified. See [the regridding guide](../../docs/regridding.md)
for supported models and limitations. Existing output files are not overwritten.
