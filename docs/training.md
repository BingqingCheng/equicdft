# Running a fit: the trainer's operational surface

`Trainer` owns optimization and epoch orchestration; the model, the loss and
the metric collections keep their own responsibilities and `Trainer.to(device)`
moves all of them together. This page covers what a fit writes, how it resumes
and which option answers which operational need. The physics of the objective
belongs with the model; see the [README](../README.md) for the current recipe
and [`extending.md`](extending.md) for writing a loss term of your own.

```python
import torch
from equicdft import Loss, Metrics, TensorLoss, Trainer

trainer = Trainer(
    model,
    Loss(terms=[TensorLoss(
        name="beta_mu_local",
        prediction_key="beta_mu_local",
        target_key="beta_mu_average",
        weights_key="voxel_weights",
    )]),
    metrics=[Metrics(
        name="local beta mu",
        prediction_key="beta_mu_local",
        target_key="beta_mu_average",
        selection_mask_key="voxel_weights",
    )],
    optimizer_cls=torch.optim.Adam,
    optimizer_args={"lr": 1.0e-4},
    scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_args={"factor": 0.5, "patience": 3},
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir="fit",
    log_dir="fit",
    early_stopping_patience=10,
)
history = trainer.fit(train_loader, valid_loader, epochs=20)
```

The optimizer and scheduler are constructed on the first `fit`, `fit_streams`
or `load_checkpoint` call, from the parameters the model and loss modules
already own. A loss term with its own parameters is therefore optimized
without any extra registration.

## What a fit writes

With `checkpoint_dir` set:

| file | written |
|---|---|
| `last.pt` | after every epoch |
| `best.pt` | whenever validation loss improves, unless `save_best=False` |
| `checkpoint_epoch_XXXX.pt` | every `checkpoint_interval` epochs |

With `log_dir` set, `history.csv` and `training.log`. The CSV is rebuilt
atomically from the complete in-memory history after every epoch and after a
checkpoint load, so it never contains a partial row and never diverges from a
resumed run. `training.log` receives, with timestamps, exactly what the trainer
prints, so `verbose=False` leaves it unwritten; `trainer.log_message("...")`
appends an application message to it from outside the epoch loop. Leaving
either directory as `None` disables that output entirely.

`fit` returns the same history it accumulates: one record per epoch with
`epoch`, `learning_rate`, `train_losses`, `valid_losses`, `train_metrics` and
`valid_metrics`.

## Resuming

```python
start_epoch = trainer.load_checkpoint("fit/last.pt", train_loader)
trainer.fit(train_loader, valid_loader, epochs=20)
```

`load_checkpoint` takes the training loader because it restores that loader's
`generator` state alongside the model, optimizer, scheduler, history and
early-stopping counter. A loader built with an explicit
`generator=torch.Generator().manual_seed(...)` therefore continues the same
sequence of batches rather than restarting it; a loader without one is
reshuffled as usual. Only current-format checkpoints load;
older ones are not migrated, and a fitted model can be carried forward as
initialization with `python -m equicdft.conversion` instead.

`early_stopping_patience` counts consecutive epochs without validation
improvement, and its counter survives a restart, so stopping behaves the same
whether or not a run was interrupted.

## Scheduling and validation

`scheduler_cls=ReduceLROnPlateau` receives the validation loss; every other
scheduler is stepped once per epoch with no argument. `record_initial_validation=True`
evaluates the untrained model before the first epoch, which gives a baseline
row in the history at the cost of one validation pass.

`verbose=False` silences the per-epoch table; `print_interval=5` prints every
fifth epoch. Neither affects `history.csv`.

## Masked fitting forwards

A model exposing `voxel_weights(data)` is evaluated with
`use_voxel_mask=True` for both training and validation, so a fit sees the
selected ideal, LDA and CACE-centered energy while an ordinary
`model(data)` and every solve keep the full physical functional. A stream
that should be fitted against the full functional sets
`model_kwargs={"use_voxel_mask": False}`.

## Standardizing local readout inputs

The invariants a CACE pipeline feeds its `LocalReadout` are not of one size.
For a polarized `p = 2, nu = 2` branch on water, the smeared density and the
polarization centre channel have a spread of order $10^{-2}$ over the
training voxels while the gradient-type moments of the same fields, the
invariants that tell a divergence from a curl, sit two to three orders of
magnitude below; on a four-shell branch the spread covers five decades. A
network with default initialization fits along the large inputs first and
reaches the small ones only slowly, so a fit can leave whole families of
invariants unused.

`standardize_local_features` measures the population standard deviation
$\sqrt{\langle(x-\langle x\rangle)^2\rangle}$ of every readout input, with
the variance divided by the voxel count. It covers the whole vector the
readout sees: an optional separate centre density, the invariants and the
temperature feature. It installs the result as the readout's
`feature_scale`, so each varying input enters the network with unit spread.
Mean subtraction in the forward transform is independently controlled by
`center`:

```python
from torch.utils.data import ConcatDataset, DataLoader
from equicdft import standardize_local_features

def standardize_model(model, train_dataset, validation_dataset):
    """Standardize a freshly initialized model before constructing Trainer."""
    normalization_data = ConcatDataset([train_dataset, validation_dataset])
    normalization_loader = DataLoader(normalization_data, batch_size=1)
    transforms = standardize_local_features(
        model, normalization_loader, center=True,
    )
    return transforms, model.to_config()
```

`transforms` maps each local readout's module name (as in
`model.named_modules()`, e.g. `readout.1.readout`) to the `scale` and
`offset` it now carries; nothing else has to be done before `Trainer`.
`center=True` also subtracts each feature's population mean
(`feature_offset`). A constant shift is absorbed exactly by the first
layer's bias, so centring changes conditioning only: without it, an input
with a large mean-to-spread ratio (the smeared density channels sit at
ρ/ρ_ref ≈ 1 with a spread of a few tenths in bulk cases) enters every
hidden unit with the same large offset, far from the SiLU knee. Features
excluded by the relative-spread floor keep a unit scale and zero offset.

Use the same membership as the other data-derived normalizations (train plus
validation, never test). The statistics depend on the feature modules'
initial parameters, so compute them on the model that will be fitted, before
the fit; the scale is then a fixed diagonal preconditioner. Features whose
spread is below `relative_floor` (default $10^{-4}$) of the largest spread of
the same readout keep a scale of one, so a constant input, such as the
temperature feature at the reference temperature, is not amplified into
noise. `local_feature_statistics` and `local_feature_scale` expose the two
steps separately.

The scale and offset belong to the fitted weights: they are persistent
buffers of the readout, so every checkpoint and state dict carries them, and a scaled state
refuses to load into an unscaled readout instead of silently computing a
different function. `model.to_config()` records it as well, so a model rebuilt
from its configuration alone is already scaled. Models without a scale have
no state entry and load exactly as before. The scale lives in the dtype of
the readout's weights. To standardize an already fitted model without
changing it, pass `keep_function=True` (or call
`readout.set_feature_scale(scale, offset, keep_function=True)`): the first
linear layer's weights absorb the ratio of the scales, its bias the shift of
the offsets, and the outputs are unchanged to round-off.

## Noise augmentation

`Trainer(noise=NoiseOptions(density=True, density_floor=0.0))` perturbs each
training forward and leaves validation and explicit evaluation clean. The
batch must carry `rho_std`; polarized fields additionally use the model's
ideal reference for the local bound.

Apply noise first, then let the model compute `rho_noisy > model.rho_min`
intersected with the unchanged `valid` and nonexcluded support. This realized
selection controls local energy, the chemical-potential spatial mean, and
losses using output `voxel_weights`. Supported nonvacuum voxels may enter or
leave the selection on each visit; validation uses the clean density.
`TensorLoss(weights_key="voxel_weights")` also works for polarization:
weights are expanded over the final vector-component axis and the weighted
mean includes all components. With hard zero/one weights,
`residual_smoothing` Gaussian-smooths the masked residual on the grid before
the criterion (see the polarization guide); continuous weights are rejected
for this nonlocal objective. Smoothing changes the objective only, never the
model or the metrics. On a fixed grid, `fixed_grid=True` reads the batch
geometry once instead of synchronizing with the GPU on every step.

For this protocol, `valid` must be a **cutoff-independent support mask**.
Do not cache clean-cutoff `voxel_weights` or separate `polarization_weights`
in the input. Explicit `voxel_weights` remain an intentional override of
the cutoff, not something the trainer can safely reinterpret. The library
cannot infer how a Boolean support mask was constructed. A legacy cache
whose `valid` includes `rho_clean > rho_min`, or which discarded targets
outside that selection, must be rebuilt before using this protocol; it
cannot support upward crossings. Keep external-field targets on the full
support. Standalone augmentation callers use the same order: augment, then
`model(noisy_batch, use_voxel_mask=True)` without frozen cutoff weights.

This changes the noisy training objective from the former frozen-mask
behavior. Do not silently resume an existing frozen-mask campaign.

## Per-case loss weights from signal and noise

An equal-case field loss averages squared residuals whose size follows the
strength of each case's external field: a strongly driven case leaves a
residual orders of magnitude above a weakly driven one, so the strong cases
set the fit and the weak ones are fitted only in absolute terms. Two scales
describe each case in the units of a loss term:

- the **signal** `S`: the term at a fixed reference functional, normally the
  freshly initialised model, i.e. what the learned terms have to supply;
- the **noise** `N`: the term evaluated on the model's linear response to
  the sampling error of the input fields, with the targets set to zero. For a
  draw `xi` scaled by `rho_std` (and `dipole_density_std`), every floating
  output is replaced by its central difference
  `[o(x + h xi) - o(x - h xi)] / (2h)`; for a quadratic term that is
  `|J xi|^2`, the residual a functional without model error would still
  leave on these fields, in the term's own metric (masks, gauges, residual
  filters and long-range kernels included).

`case_loss_weights` turns them into frozen weights
`w = c (noise_scale * N + eta * S)^(-power)`. `eta`
(`relative_model_error`) is the fraction of the signal a good fit is still
expected to miss: small values approach chi-squared weighting by the noise
alone, large values relative-error weighting by the signal. `power` in
(0, 1] tempers the spread (0.5 takes the square root). With
`normalization="mean"` the weights of the given cases average to one;
`"reference_loss"` keeps the objective at the reference unchanged. The
weights enter the loss through `Loss(case_weight_keys=...)`, which multiplies
each listed term by its batch weight; case weights need one case per batch.

```python
from equicdft import Loss, case_loss_statistics, case_loss_weights

loss = Loss(terms, case_weight_keys={"density": "density_case_weight",
                                     "polarization": "polarization_case_weight"})
reference = case_loss_statistics(
    model, loss, DataLoader(train_frames, batch_size=1),   # training cases only
    terms=["density", "polarization"],
    target_keys=["beta_E_target"],   # batch targets of custom terms
    case_key="case_id",              # every frame carries its identity
    noise_samples=4, noise_step=0.1,
)
train = case_loss_weights(reference, relative_model_error=0.1, power=0.5)
for frame in train_frames:
    weights = train["by_case"][frame["case_id"]]
    frame["density_case_weight"] = torch.tensor(weights["density"])
    frame["polarization_case_weight"] = torch.tensor(weights["polarization"])
validation = case_loss_weights(
    case_loss_statistics(model, loss, validation_loader, terms=[...], case_key="case_id", ...),
    relative_model_error=0.1, power=0.5,
    noise_scale=train["noise_scale"],       # same chosen noise scale
    normalization=train["normalization"],   # same scale as training
)
```

Assign weights by `by_case`, never by position: rows follow the batch order,
and without `case_key` a shuffled `DataLoader` is refused. `Loss` checks
every case weight it reads (finite and nonnegative) on each evaluation, so a
weight loaded from a cache or built by hand cannot turn the objective
negative or nonfinite.

Compute the statistics once, on the model before the fit, and record the
weights with the run (like the feature standardization); recomputing them on
resume would change the objective. Test cases never enter. Batch targets of
`TensorLoss` terms are zeroed automatically; a custom term's batch targets
are listed in `target_keys`, and every term is checked to vanish for zero
outputs, so an undeclared target raises.

Every Gaussian draw is kept, so the propagated distribution is the one the
SEM states. Only the finite-difference step adapts: it starts at
`noise_step` and is halved for that draw until `x ± h xi` stays inside
`rho > 0`, `|P| < m rho` and leaves every selection unchanged (the
`frozen_outputs`, by default `voxel_weights`, and every Boolean output such
as the energy-centre mask). The response is then the derivative on the
clean selection branch. `selection_changes` and `selection_activations`
count the voxels that flipped, and that entered the selection, at the first
step of each draw; `step_reductions` and `minimum_step` show how often and
how far the step had to shrink. Many activations mean the case sits close to
the density cutoff, where the training noise augmentation would also move
voxels across it; the linear estimate does not include that jump.
The noise of a non-quadratic criterion (smooth L1) is the quadratic estimate.

The noise model is diagonal: independent errors per voxel and component, as
`rho_std` and `dipole_density_std` state them. Real sampling errors can be
spatially correlated. Such correlations can increase or decrease the
propagated noise, particularly where the functional weights wavelengths or
directions differently (for example, the longitudinal Coulomb response of
polarization). The default `noise_scale=1` is therefore a working choice, not
a bound on the true noise. `noise_scale="signal_quantile"` is an optional
exploratory heuristic: it uses the `signal_quantile` (default 0.05) of `S/N`
over cases with measured noise. An individual squared residual can be below
its expected noise even for an exact reference functional, so this quantile
does not estimate or bound the true correlated-noise scale. Estimating that
scale requires independent MD blocks/replicas or covariance-aware noise draws.
Pass any chosen `noise_scale` to the other splits.

A weight scales a case's gradient, not its information: a case whose
residual already sits near its estimated noise level contributes noise-fitting gradient
in proportion to its weight. Prefer `power < 1` or a larger `eta` when many
cases are noise-dominated, and compare against the unweighted control as one
switch.

## Joint training over several datasets

`fit_streams` optimizes several `TrainingStream`s together, each with its own
loaders, loss, metrics and noise:

```python
from equicdft.training.trainer import TrainingStream

streams = [
    TrainingStream(
        name="fields",
        train_loader=field_train,
        valid_loader=field_valid,
        loss=field_loss,
    ),
    TrainingStream(
        name="response",
        train_loader=response_train,
        valid_loader=response_valid,
        loss=response_loss,
        model_kwargs={"compute_c1": False, "compute_local_mu": False},
        cycle=True,
    ),
]
trainer.fit_streams(streams, epochs=20, stream_weights={"fields": 1.0, "response": 0.1})
```

`batches_per_step` averages several batches of one stream into a single
optimizer update. `cycle=True` restarts a short stream when it is exhausted and
lets the non-cycling streams set the epoch length; validation is always a
single pass. `model_kwargs` makes a response-only forward explicit for that
stream without mutating flags on the model, which matters because those flags
are saved in the model file.

`stream_weights` accepts a number or a callable of the one-based epoch
number, so a schedule that introduces a regularizer gradually needs no custom
loop. With `record_initial_validation=True` the epoch-zero pass evaluates
those callables at zero.
