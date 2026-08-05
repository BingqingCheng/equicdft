# equicdft

`equicdft` develops equivariant neural models for learning classical density
functionals. The current architecture uses Cartesian symmetry-adapted features
for density fields on periodic three-dimensional grids.

One `GridData` object represents one complete density configuration and
contains the grid fields together with periodic neighborhood indices for every
grid point.

Split complete fields and construct reproducible data loaders with:

```python
from equicdft import GridData, make_dataloaders

dataset = GridData.from_xyz("density.extxyz", cutoff_grid=3)
data = make_dataloaders(
    train_dataset=dataset,
    valid_fraction=0.1,
    test_dataset=None,
    batch_size=2,
    seed=1,
    compute_mean_density=True,
    compute_mean_temperature=True,
)

train_loader = data["train"]
valid_loader = data["valid"]
test_loader = data["test"]
mean_density = data["mean_density"]
mean_temperature = data["mean_temperature"]
```

The random split acts on complete fields and does not inspect temperature or
chemical potential. A test dataset should be constructed separately to probe
the intended thermodynamic states. The optional `mean_density` and
`mean_temperature` are computed from train plus validation and exclude the
test set. They can be used as fixed input scales without leaking test-set
statistics.

Pass both scales into the representation/model. Density normalization is
applied inside `CartesianAFeatures`, while the readout is conditioned on the
scale-only normalized temperature `temperature / mean_temperature`:

```python
a_features = CartesianAFeatures(mean_density=mean_density, ...)
model = GridCACEModel(
    a_features=a_features,
    ...,
    mean_temperature=mean_temperature,
)
```

Choose the local-chemical-potential target in the training configuration. Use
the dimensionless reservoir value `beta_mu` when it is known, or the model's
masked spatial `average_chemical_potential` when it is latent:

```python
from equicdft import Loss, TensorLoss

chemical_potential_target_key = "average_chemical_potential"
# chemical_potential_target_key = "beta_mu"  # known reservoir value

loss_module = Loss(
    terms=[
        TensorLoss(
            name="local_chemical_potential",
            prediction_key="local_chemical_potential",
            target_key=chemical_potential_target_key,
            weights_key="chemical_potential_weights",
            weight=1.0,
        ),
    ]
)
loss_values = loss_module(outputs, batch)
loss_values["total"].backward()
```

When `compute_local_mu=True` and `V_ext` is supplied, `GridCACEModel`
constructs the dimensionless physical output
`local_chemical_potential = log(rho * thermal_wavelength**3) + beta * V_ext - c1`.
Its `average_chemical_potential` method returns the componentwise spatial
average weighted by the hard mask `rho > rho_min`. The model also returns this
mask as `chemical_potential_weights`. `GridData` stores `beta_mu = beta * mu`
when reservoir chemical potentials are present. A weighted `TensorLoss`
therefore handles either supervised or latent chemical potentials solely by
changing `target_key`; the model does not select the target. In the latent
case, the unknown chemical potential is the first masked spatial cumulant and
the loss is its second central cumulant.

Record unweighted dataset-level prediction metrics across batches:

```python
from equicdft import Metrics

c1_metrics = Metrics(
    target_key="c1",
    prediction_key="c1",
    metric_keys=("mae", "rmse", "rmse_percent", "pearson_r"),
)
for batch in train_loader:
    outputs = model(batch)
    c1_metrics.update_metrics("train", outputs, batch)

epoch_metrics = c1_metrics.retrieve_metrics("train")
```

`retrieve_metrics` concatenates the recorded batches before evaluating each
metric. Thus RMSE and Pearson correlation are computed over the complete
subset rather than averaged from batch-level values. `rmse_percent` is
`100 * RMSE / sigma(target)` using the population standard deviation of all
recorded target values.

Keep epoch orchestration outside the model and training script with
`Trainer`:

```python
from equicdft import Trainer

trainer = Trainer(
    model=model,
    loss=loss_module,
    metrics=[c1_metrics],
    optimizer_cls=torch.optim.Adam,
    optimizer_args={"lr": 1.0e-4},
    scheduler_cls=None,
    scheduler_args=None,
    device="cpu",
    checkpoint_dir=None,
    checkpoint_interval=10,
    save_best=True,
    early_stopping_patience=5,
)
history = trainer.fit(
    train_loader,
    valid_loader,
    epochs=1000,
    print_interval=10,
)
```

Set `scheduler_cls` and `scheduler_args` to use an epoch scheduler. A
`ReduceLROnPlateau` scheduler receives validation loss automatically; other
schedulers are stepped once per epoch. Setting `checkpoint_dir` writes
periodic `checkpoint_epoch_XXXX.pt` files and updates `last.pt`; with
`save_best=True`, it also maintains `best.pt`.
`early_stopping_patience` optionally stops after the requested number of
consecutive epochs without validation-loss improvement. The patience counter
is stored in checkpoints and therefore continues correctly after a restart.

For manual preprocessing, compute one mean density per frame with:

```bash
python scripts/mean_density.py density.extxyz
```

The script prints a JSON list. Select the training-plus-validation entries and
average them to obtain the scalar `mean_density` used below.

The same calculation can be called from a training script:

```python
from scripts.mean_density import compute_mean_densities

frame_mean_densities = compute_mean_densities(
    "density.extxyz",
    index="0:50",  # training frames only
)
mean_density = sum(frame_mean_densities) / len(frame_mean_densities)
```

```python
from equicdft import (
    CartesianAFeatures,
    CartesianBFeatures,
    GridCACEModel,
    GridData,
    LocalReadout,
)

dataset = GridData.from_xyz(
    "density.extxyz",
    cutoff_grid=3,
    boltzmann_constant=1.0,  # reduced units; default is eV/K
)
data = dataset[0]
n_types = int(data["n_types"].item())
mean_density = 0.7  # precomputed from the training frames

a_features = CartesianAFeatures(
    mean_density=mean_density,
    cutoff_grid=3,
    max_power=4,
    n_radial_channels=4,
    trainable_radial_exponents=False,
    n_types=n_types,
)
b_features = CartesianBFeatures(max_power=4, max_product_order=3)
readout = LocalReadout(
    n_types=n_types,
)
model = GridCACEModel(
    a_features=a_features,
    b_features=b_features,
    readout=readout,
    grid_spacing=data["grid_spacing"],
    boltzmann_constant=1.0,
    thermal_wavelength=data["thermal_wavelength"],
    compute_c1=True,
    compute_local_mu=True,
    rho_min=1.0e-3,
)

outputs = model(data)
beta_F_exc = outputs["beta_F_exc"]
c1 = outputs["c1"]
local_chemical_potential = outputs.get("local_chemical_potential")
average_chemical_potential = outputs.get("average_chemical_potential")
chemical_potential_weights = outputs.get("chemical_potential_weights")
```

Set `radial_basis="none"` and `n_radial_channels=1` to replace the Gaussian
channels by one equal-weight neighborhood average. The resulting descriptors
use only Cartesian polynomial moments; the constant radial weight is divided
by the stencil size so that invariant products remain well scaled.

With `separate_center=True`, the zero offset is removed from every radial
channel and `GridCACEModel` inserts `rho / mean_density` exactly once before
the invariant neighbor features:

```python
a_features = CartesianAFeatures(
    mean_density=mean_density,
    cutoff_grid=3,
    max_power=2,
    radial_basis="none",
    n_radial_channels=1,
    separate_center=True,
    n_types=n_types,
)
```

Center separation is opt-in. The default retains the original basis with the
center included, and previously saved checkpoints and full models remain
loadable.

Temperature is required, and `GridData` always stores `beta = 1 / (k_B T)`.
Both `rho` and `beta` passed to `GridCACEModel` retain their physical values;
density normalization is confined to `CartesianAFeatures`.
`GridCACEModel` appends the scalar temperature once to the flattened local
`B` feature vector at every grid point before applying `LocalReadout`.
The trained grid spacing and thermodynamic conventions are persistent model
buffers. The stencil cutoff, number of components, and density scale are also
available directly as `model.cutoff_grid`, `model.n_types`, and
`model.mean_density`. For inference, `model.grid_info` collects the grid and
thermodynamic metadata in the native forms accepted by `GridData`:

```python
inference_data = GridData.from_xyz(
    "density.extxyz",
    grid_info=model.grid_info,
)
```
External potential and chemical potential are optional fields rather than
inputs to the learned intrinsic functional. When `V_ext` is available, the
model additionally returns the requested local and averaged chemical
potentials. `GridData` constructs
`c1_plus_beta_mu = log(rho * thermal_wavelength**3) + beta * V_ext`; if `mu`
is also present, it stores `c1 = c1_plus_beta_mu - beta * mu`. These pointwise
targets are stored only when every selected frame has strictly positive
density values; zero-density voxels are retained for the model's
`chemical_potential_weights` to mask. The default
Boltzmann constant is `8.617333262e-5` eV/K and the default thermal wavelength
is one. Pass `boltzmann_constant=1.0` for reduced-unit data. An inference
EXTXYZ requires temperature, grid metadata, and at least one of `rho` and
`V_ext`. A density-only record can be evaluated directly; an external-field-
only record supplies the geometry and thermodynamic controls needed to solve
for an equilibrium density.

For inference, the regular grid can instead be constructed directly:

```python
import torch

external_data = GridData.from_dict(
    {
        "grid_size": [10, 10, 10],
        "temperature": 1.5,
    },
    grid_info=model.grid_info,
)
external_data["V_ext"] = V_ext.reshape(1000, 1)
external_data["mu"] = torch.tensor([0.0])
```

`grid_size` always contains only `[nx, ny, nz]`. Without `grid_info`,
`n_types` and `grid_spacing` are separate required entries. Density,
external-potential, and chemical-potential tensors are assigned afterward in
their canonical flattened shapes.

Use `GridSolver` for prescribed-density thermodynamics or equilibrium
solving:

```python
from equicdft import GridSolver

solver = GridSolver(model, device="cuda")

# data contains rho; V_ext and mu are optional.
evaluated = solver.evaluate(data)

# external_data contains V_ext and mu but need not contain rho.
equilibrium = solver.solve(external_data)
rho = equilibrium["rho"]

# Alternatively impose one particle number per component.
canonical = solver.solve(external_data, particle_numbers=[128.0])
```

The default solver is the adaptively mixed Euler fixed-point iteration. Use
`method="minimize"` for positivity-preserving mirror descent with an Armijo
line search on the free energy or grand potential. The minimization uses
energy-only trial evaluations and computes `c1` for its update and convergence
diagnostics. `GridCACEModel.forward(..., compute_c1=False)` can likewise be
used directly when only `beta_F_exc` is needed.

For a multicomponent density field, an optional learned, bias-free channel map
can mix the physical component channels of `A` before symmetrization:

```python
a_features = CartesianAFeatures(
    mean_density=mean_density,
    cutoff_grid=3,
    max_power=4,
    n_radial_channels=4,
    n_types=n_types,
    n_channels=4,
)
```

`CartesianAFeatures` applies the same mixing matrix over all grid points,
radial channels, and Cartesian components, so it commutes with cubic rotations
and reflections. The nonlinear products in `B` then contain cross-component
correlations. For a one-component field, leave `n_channels=None`; no mixing
module or mixing parameters are created.

A fine regular grid can be block-averaged in memory before its local
environments are constructed:

```python
dataset = GridData.from_xyz(
    "density-grid-0.25.extxyz",
    cutoff_grid=3,
    target_grid_spacing=0.5,
)
```
