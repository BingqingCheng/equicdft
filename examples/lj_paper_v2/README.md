# Fitted Lennard--Jones functional: revised-paper example

This directory bundles the fitted Equi-cDFT model used in the revised paper
and a single held-out equilibrium density field. No training or download is
needed. From the repository root, after installing the package (`pip install -e .`):

```sh
python examples/lj_paper_v2/evaluate.py
```

The script evaluates the dimensionless excess free energy `beta_F_exc` and
the first direct correlation `c1`, then reports a density-to-external-potential
check. It uses one CPU thread and double precision; `--device cuda` is optional.
No density solve or second derivative is computed. The native example uses
well below 1 GB of memory in the retained CPU check.

Expected values (roundoff may vary slightly):

```text
beta_F_exc:                  -604.7519242625531
c1_shape:                   [4096, 1]
particle_number:            208
gauge_aligned_beta_V_RMSE:   0.026188112132846143
```

The field has `T=1.5`, `N=208`, `L=8 sigma`, and `h=0.5 sigma`. It is the
same field as the older `tests/regression/lj_paper_v1` example, but **the model
is different**. The fluid is truncated-and-shifted Lennard--Jones with cutoff
`2.5 sigma`. The external-potential error uses `rho > 1e-3` and one fitted
constant over that mask, reflecting the unavoidable canonical potential gauge.
The classical ideal-term wavelength convention is `Lambda=1`.

## Model and provenance

The reference fit is `fit-beta-p3nu2-rbf1normtrain-c4`, without stability
training: LDA plus Cartesian CACE, degree 3, product order 2, a `2 sigma`
receptive field, and one normalized, trained Gaussian radial channel. It was
trained on `h=0.5 sigma` density fields. These are reference-study choices,
not universal settings or a claim of accuracy for arbitrary density fields.

`model.pt` is a current versioned `save_model` archive, loadable directly with
`equicdft.load_model`. It was converted from the published fitted model and
promoted from float32 to float64; **no parameters were refitted**. Conversion
was checked against the original model on this field. The original model is
preserved in the [data release](https://github.com/BingqingCheng/equicdft-lj-data/releases/tag/lj-paper-v2).
[provenance.json](provenance.json) records source/export hashes, code identity,
training configuration, and the retained numerical check.

For an already sampled density on a different grid, use the separate
[grid-transfer example](../grid_transfer/README.md):

```sh
python examples/grid_transfer/evaluate.py examples/lj_paper_v2/model.pt fine.extxyz \
    --spacing 0.25 --center-normalization reference --output evaluation.pt
```

This preserves the physical descriptor scales and the training-grid center
convention. It does not interpolate the supplied density, retrain the model,
or perform an equilibrium solve; `fine.extxyz` must already be on the target
grid. The bundled field itself is only on the native `h=0.5 sigma` grid.

Paper: [Equivariant learning of a transferable three-dimensional classical
density functional](https://arxiv.org/abs/2608.13506).
The [LJ data repository](https://github.com/BingqingCheng/equicdft-lj-data)
contains the full training/test corpus and application/figure data.
