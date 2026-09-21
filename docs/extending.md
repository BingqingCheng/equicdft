# Writing your own readout, loss term or metric

The package exports three base classes for extending it: `EnergyReadout`,
`LossTerm` and `MetricCollection`. Each is a small explicit protocol — the
model, the loss and the trainer call declared methods rather than inspecting
your object — so a new component drops into an existing model or fit without
touching the package. Every example below is complete and runs as written.

## A custom energy readout

A readout receives a `ReadoutContext` and returns one scalar per complete
density field. The model sums every readout's energy and differentiates the
sum, so a readout never computes a derivative itself.

```python
import torch
from torch import nn
from equicdft import EnergyReadout
from equicdft._config import Configurable, make_config, register


@register
class MeanFieldReadout(EnergyReadout, Configurable):
    """Analytic mean-field excess ``a * sum (rho/rho_ref)^2 * dV``."""

    def __init__(self, n_types, *, initial_coefficient=0.0):
        super().__init__()
        self.n_types = int(n_types)
        self.initial_coefficient = float(initial_coefficient)
        self.coefficient = nn.Parameter(
            torch.tensor(self.initial_coefficient, dtype=torch.get_default_dtype())
        )

    def energy(self, context):
        normalized = context.normalized_rho()
        return self.coefficient * (
            normalized.square().sum(dim=(-2, -1)) * context.voxel_volume
        )

    def to_config(self):
        return make_config(
            self, n_types=self.n_types,
            initial_coefficient=self.initial_coefficient,
        )
```

Put it in a model like any other readout:

```python
from equicdft import GridCACEModel, IdealGas

model = GridCACEModel(
    readout=[MeanFieldReadout(n_types=1, initial_coefficient=0.1)],
    n_types=1,
    grid_spacing=1.0,
    mean_density=0.5,
    boltzmann_constant=1.0,
    ideal=IdealGas(),
)
```

### What the context offers

The model builds the context with `ReadoutContext.from_data(...)`, which is
also the way to construct one when evaluating a readout outside a model.
Primitives are set once per forward: `rho`, `normalized_temperature`,
`voxel_volume`, `mean_density`, `polarization_scale`, `alignment_scale`,
`alignment_density_floor`, `dipole_magnitude`, `grid_spacing`, `grid_size`,
`beta`, `reference_energy`, `free_energy_mode`, `dipole_density`,
`center_density`, `grid_positions`, `grid_center`, `excluded_mask`,
`local_density_index` and `local_energy_mask`. Optional ones the model did not supply are `None`;
`context.require("mean_density")` raises a `KeyError` naming the missing
quantity instead of failing later in arithmetic.

Composed inputs are cached on first use, so several readouts asking for the
same thing compute it once: `normalized_rho()`, `temperature_feature()`,
`normalized_polarization()`, `reduced_polarization()` (the smooth molecular
alignment, computed by the exported `regularized_reduced_polarization`),
`normalized_alignment()` (that field over `alignment_scale`),
`pointwise_state()` (normalized density plus temperature), `state_features()`
(the field-level state) and `local_features()` (the invariant environment
vector).

`context.integrate_local(per_particle)` integrates `rho * per_particle` over
the grid while honouring `local_energy_mask`, the selection a fitting forward
sets with `use_voxel_mask=True`. A readout whose energy is a sum over local
centers should integrate through it rather than reducing by hand, so that
`use_voxel_mask` reaches it; a readout whose energy is not center-local — a
long-range or pairwise term — should not, and the model does not expect it
to. The example above reduces directly and is therefore never selected.

### Declaring what you need

Class attributes tell the model what to prepare and what to check:

| attribute | meaning |
|---|---|
| `requires_local_features` | you call `context.local_features()`; the model then evaluates the invariant features, and `a_features`/`b_features` become mandatory |
| `requires_local_density_index` | you gather explicit periodic neighborhoods, so the data must carry a neighbor list |
| `requires_dipole_density` | you read the polarization field |
| `requires_polarization_scale` | you normalize polarization, so the model must own a `polarization_scale` |
| `requires_alignment_scale` | you use the molecular-alignment inputs, so the model must own `alignment_scale`, `alignment_density_floor` and an ideal with a `dipole_magnitude` |
| `cutoff_grid` | your stencil radius in grid steps, or `None` |
| `shares_grid_stencil` | your `cutoff_grid` is the model's shared one and must agree with it; leave `False` if you build your own offsets |
| `n_types` | the component count you are built for, or `None` for any |

Two more hooks matter occasionally. `energy_and_outputs(context)` returns
`(energy, outputs)` when a readout also reports observables; the names must be
unique across readouts and must not shadow model outputs. `wrapped_readouts()`
returns readouts whose energy yours already includes, which is how
`MetalWall` stops the model from counting its wrapped Coulomb readout twice.

If your first layer consumes the model's local feature vector, implement
`bind_input_size(n_features)` and `is_bound`; the model calls the first at
construction so every parameter exists with its final shape before an
optimizer or a checkpoint sees it.

### Serialization

`Configurable` plus `@register` is what lets `save_model` write your readout
and `load_model` rebuild it. `to_config` returns the *constructor* arguments,
not the learned values: trained numbers travel in the `state_dict`, so
recording `initial_coefficient` rather than the current coefficient is the
rule, not an oversight. The default `from_config` passes the remaining entries
to the constructor; override it only when your constructor takes other
configurable modules, which it then builds with `equicdft._config.build`.

With the two together, a model holding your readout round-trips exactly:

```python
from equicdft import load_model, save_model

save_model(model, "model.pt")
assert torch.equal(
    load_model("model.pt")(dict(data))["beta_F_exc"].detach(),
    model(dict(data))["beta_F_exc"].detach(),
)
```

`Configurable`, `make_config` and `register` currently live in
`equicdft._config`, a private module. Importing them from there is the
supported way to write a serializable component today.

## A custom loss term

A `LossTerm` returns a mapping containing the weighted scalar `"loss"`,
plus any diagnostics worth reusing. `Loss` collects those mappings and hands
them to the metric collections, so an expensive intermediate is computed once
per batch rather than once per consumer.

```python
import torch
from equicdft import LossTerm


class HuberSpread(LossTerm):
    """Huber penalty on the spread of a field about its weighted mean."""

    def __init__(self, name, key, weights_key, *, delta=1.0, weight=1.0):
        super().__init__(name, weight)
        self.key, self.weights_key, self.delta = key, weights_key, float(delta)

    def evaluate(self, outputs, batch, model=None):
        values = outputs[self.key]
        weights = outputs.get(self.weights_key, batch.get(self.weights_key))
        if weights is None:
            raise KeyError("missing weights '{}'".format(self.weights_key))
        weights = weights.detach().to(values)
        mean = ((weights * values).sum(dim=-2, keepdim=True)
                / weights.sum(dim=-2, keepdim=True))
        residual = values - mean
        absolute = residual.abs()
        quadratic = torch.minimum(absolute, torch.full_like(absolute, self.delta))
        elementwise = 0.5 * quadratic.square() + self.delta * (absolute - quadratic)
        loss = self.weight * (weights * elementwise).sum() / weights.sum()
        return {"loss": loss, "residual": residual.detach(), "weights": weights}
```

`name` must be unique within a `Loss` and may not be `"total"`, which is
reserved for the sum. The scalar must be a zero-dimensional tensor; `Loss`
checks that rather than letting a shape mistake propagate into `backward`.

A term is an `nn.Module`, so any parameter it owns is trained: the trainer
constructs its optimizer from the model's and the loss's parameters together.

Every term receives the `model`. A term that genuinely needs it — to evaluate
a reference state, or to read a scale — should raise when it is `None` rather
than silently degrade, which is the convention the response terms follow.

## A custom metric collection

A `MetricCollection` accumulates detached statistics per subset and reports
them once an epoch. It receives the same `outputs` and `batch` the loss saw,
plus the `details` mapping keyed by loss-term name.

```python
import torch
from equicdft import MetricCollection


class ExceedanceFraction(MetricCollection):
    """Weighted fraction of voxels whose residual exceeds a threshold."""

    def __init__(self, name, loss_name, *, threshold=0.1, subsets=("train", "valid")):
        super().__init__(name, subsets)
        self.loss_name, self.threshold = loss_name, float(threshold)
        self._totals = {subset: [0.0, 0.0] for subset in self.subsets}

    def update_metrics(self, subset, outputs, batch, details=None):
        self._require_subset(subset)
        if details is None or self.loss_name not in details:
            raise KeyError("no details from loss term '{}'".format(self.loss_name))
        record = details[self.loss_name]
        weights, residual = record["weights"], record["residual"]
        exceeded = (residual.abs() > self.threshold).to(weights)
        accumulator = self._totals[subset]
        accumulator[0] += float((weights * exceeded).sum())
        accumulator[1] += float(weights.sum())

    def retrieve_metrics(self, subset, clear=True):
        self._require_subset(subset)
        exceeded, total = self._totals[subset]
        value = torch.tensor(exceeded / total if total else float("nan"))
        if clear:
            self.clear_metrics(subset)
        return {"exceedance": value}

    def clear_metrics(self, subset):
        self._require_subset(subset)
        self._totals[subset] = [0.0, 0.0]
```

`_require_subset` raises a `KeyError` naming the valid subsets, so a typo in
`subsets` fails where it is made. Accumulate detached Python or tensor
scalars, never graph-connected values: a collection that keeps autograd
history will hold the whole epoch's graph alive.

The three pieces compose directly:

```python
from equicdft import Loss, Trainer

trainer = Trainer(
    model,
    Loss(terms=[HuberSpread("spread", "beta_mu_local", "voxel_weights", delta=0.5)]),
    metrics=[ExceedanceFraction("exceedance", "spread", threshold=0.2)],
    optimizer_args={"lr": 1.0e-3},
)
history = trainer.fit(train_loader, valid_loader, epochs=2)
```

## Feature modules

A custom A or B feature module is checked at construction against a name
contract rather than a base class, so it need not inherit from anything. The
model raises `TypeError` naming every missing member. The lists live in
`representation/local.py`:

- A features: `n_types`, `cutoff_grid`, `mean_density`, `n_radial_channels`,
  `n_output_channels`, `separate_center`, `convolution_backend`,
  `include_polarization`, `requires_dipole_density`, `transform_density`.
- A features, additionally when message layers are configured:
  `stencil_basis`, `local_density_positions`, `polarization_fields`,
  `bessel_stencil_basis`, `max_power`, `powers`, `coordinate_scaling`.
- B features: `n_features`, `validate_a_features`.

`b_features.validate_a_features(a_features)` is called with the paired module,
which is where a B module rejects an A module it cannot consume.

The check tests for the presence of names, not their signatures, so a module
written against an older signature passes construction and fails later. If you
maintain a feature module outside the package, pin it with a test that
constructs a model from it.
