# cace_grid

`cace_grid` develops Cartesian symmetry-adapted models for classical
density-functional theory on periodic three-dimensional grids.

One `GridData` object represents one complete density configuration and
contains the grid fields together with periodic neighborhood indices for every
grid point.

Split complete fields and construct reproducible data loaders with:

```python
from cace_grid import GridData, make_dataloaders

dataset = GridData.from_xyz("density.extxyz", cutoff_grid=3)
train_loader, valid_loader, test_loader, mean_density = make_dataloaders(
    train_dataset=dataset,
    valid_fraction=0.1,
    test_dataset=None,
    batch_size=2,
    seed=1,
    compute_mean_density=True,
)
```

The random split acts on complete fields and does not inspect temperature or
chemical potential. A test dataset should be constructed separately to probe
the intended thermodynamic states. The optional `mean_density` is computed
from train plus validation and excludes the test set.

Compose named training objectives independently of the training loop:

```python
from torch import nn
from cace_grid import Loss, TensorLoss

loss_module = Loss(
    terms=[
        TensorLoss(
            name="c1",
            prediction_key="c1",
            target_key="c1",
            loss_fn=nn.MSELoss(),
            weight=1.0,
        ),
    ]
)
loss_values = loss_module(outputs, batch)
loss_values["total"].backward()
```

Each term returns a weighted scalar. `Loss` returns the named terms for
logging and their sum as `total`. Predictions and targets must have exactly
matching shapes; the loss code never reshapes grid fields implicitly.

Record unweighted dataset-level prediction metrics across batches:

```python
from cace_grid import Metrics

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
from cace_grid import Trainer

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
from cace_grid import (
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
    compute_c1=True,
)

outputs = model(data)
beta_F_exc = outputs["beta_F_exc"]
c1 = outputs["c1"]
```

Temperature is required, and `GridData` always stores `beta = 1 / (k_B T)`.
`GridCACEModel` appends the scalar temperature once to the flattened local
`B` feature vector at every grid point before applying `LocalReadout`.
External potential and chemical potential are optional annotations rather
than model inputs. When `V_ext` is available, `GridData` constructs
`c1_plus_beta_mu = log(rho * thermal_wavelength**3) + beta * V_ext`; if `mu`
is also present, it stores `c1 = c1_plus_beta_mu - beta * mu`. The default
Boltzmann constant is `8.617333262e-5` eV/K and the default thermal wavelength
is one. Pass `boltzmann_constant=1.0` for reduced-unit data. An inference
EXTXYZ therefore requires density, temperature, and grid metadata, but no
dummy external potential or chemical potential.

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
