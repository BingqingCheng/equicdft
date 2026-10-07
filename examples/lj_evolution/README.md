# LJ sine-wave evolution

`evolve.py` loads the retained LJ-paper-v1 model and runs ten deterministic
density timesteps. It constructs the grid directly; no external-field data
file, simulation, or retraining is needed.

Run from the EquiCDFT repository root:

```sh
PYTHONPATH=src python examples/lj_evolution/evolve.py --output /tmp/lj-semi-implicit.pt
PYTHONPATH=src python examples/lj_evolution/evolve.py --method euler --output /tmp/lj-euler.pt
```

Settings are together at the top of the script:

- Periodic `16 x 16 x 16` grid, spacing `0.5 sigma` from the fitted model.
- `T*=1.5`, `rho*(x)=0.4+0.05 sin(2 pi x/Lx)`, uniform in y and z.
- `V_ext=0`; constant physical mobility `M=0.01`, timestep `0.01`.
- Ten steps in CPU float64; no noise, clipping, or renormalization.

Mobility sets the time scale. These numerical settings are an example, not
a measured LJ mobility or calibrated MD kinetics. The default semi-implicit
method and optional Euler method are both first order in time.

The output is a PyTorch file containing all eleven density frames, their
times, free energies, particle numbers, minimum densities, and metadata
including the checkpoint hash. Existing files are not overwritten.

```python
trajectory = torch.load("/tmp/lj-semi-implicit.pt", weights_only=True)
rho_final = trajectory["rho"][-1]  # [4096, 1]
```

Checkpoint: `tests/regression/lj_paper_v1/model.pt`, SHA256
`44419dd3d15ce0e330fe476f62431465fc32f9ad923cca3af29fbf405c13e6a1`.
Its original source and conversion are recorded in that fixture directory's
`provenance.json`.
