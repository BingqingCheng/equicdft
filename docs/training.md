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

## Noise augmentation

`Trainer(noise=NoiseOptions(density=True, density_floor=0.0))` perturbs each
training forward and leaves validation and explicit evaluation clean. The
batch must carry `rho_std`; polarized fields additionally use the model's
ideal reference for the local bound.

The weights are computed from the clean field and frozen into the batch
before the perturbation, so the energy selection and the averaging weights
cannot drift between noisy visits to the same field; the chemical potentials
themselves, and their weighted mean, are recomputed from the perturbed
fields. Training and validation therefore select the same voxels of a field,
which is what makes the two comparable. A caller that augments outside the
trainer has to freeze the weights itself — putting weights in the batch does
not by itself mask an ordinary forward.

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
