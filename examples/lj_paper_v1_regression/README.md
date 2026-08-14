# LJ-paper-v1 production regression

This self-contained regression freezes one representative held-out NVT field
and the published LJ model. It checks compatibility of the complete saved
model with the `equicdft` forward derivative and fixed-particle-number solver.

The field is zero-based frame 120 from the `T=1.5` general-3D held-out file in
`equicdft-lj-data/reference_data/heldout_nvt_general.zip`. It has `N=208`, mean
density 0.40625, and nontrivial density standard deviation 0.14991. It was
selected from spread-out candidates because both directions are representative
and the reverse solve meets the production strict-convergence criterion.

After installing the checkout with `python -m pip install -e .`, run from the
repository root:

```bash
python -m examples.lj_paper_v1_regression.forward
python -m examples.lj_paper_v1_regression.solve
```

Each command prints machine-readable metrics and exits nonzero when an
acceptance threshold in `expected_metrics.json` is violated. The forward test
aligns the additive NVT chemical-potential gauge over points with
`rho > 1e-3`. The reverse test uses the paper benchmark settings: uniform
initialization, adaptive Euler iteration, exact `N`, 200 iterations, residual
tolerance `1e-2`, density-change tolerance `1e-6`, and maximum density 2.

The default unit-test discovery also runs both regressions:

```bash
python -m unittest discover -s tests -p "test_lj_paper_v1_regression.py"
```

`provenance.json` records the production tag/commit, source archive member,
frame index, and SHA-256 hashes. The tolerances are scientific compatibility
gates rather than bitwise-output requirements, allowing harmless floating-point
variation across supported PyTorch versions.
