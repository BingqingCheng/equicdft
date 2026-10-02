"""Per-case loss weights from each case's signal and its sampling noise.

An equal-case field loss averages squared residuals whose natural size
differs by orders of magnitude between cases: a case driven by a strong
external potential needs a large response from the functional and leaves a
large residual, a weakly driven case needs almost none, so the strong cases
set the fit. Two scales describe a case in the units of a given loss term:

* the *signal* ``S``: the term at a fixed reference functional, usually the
  freshly initialised model, i.e. what the learned terms still have to supply;
* the *noise* ``N``: a linear-response estimate of how much the term moves
  when the input fields are perturbed by their reported sampling errors
  (``rho_std``, ``dipole_density_std``), assuming independent errors.

:func:`case_loss_statistics` measures both for every case and
:func:`case_loss_weights` turns them into frozen per-case weights
``w ∝ (N + η S)^-p``: ``η`` is the expected relative model error, so a case
whose residual is as large as its noise plus that fraction of its signal
contributes the same as any other; ``p < 1`` tempers the spread. The weights
enter the objective through ``Loss(case_weight_keys=...)``.
"""

import math
from numbers import Real
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from .._device import module_device, move_batch
from .._validation import nonnegative_integer, positive_scalar, unique_strings
from ..functional.polarization_ideal import FixedDipoleIdeal
from .loss import Loss

__all__ = ["case_loss_statistics", "case_loss_weights"]


def _term_value(term, outputs, batch, model) -> torch.Tensor:
    """Return a term's unweighted scalar value."""

    details = term.evaluate(outputs, batch, model)
    value = details.get("unweighted")
    if value is None:
        if term.weight == 0:
            raise ValueError(
                "loss term '{}' has zero weight and reports no 'unweighted' "
                "value".format(term.name)
            )
        value = details["loss"] / term.weight
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ValueError("loss term '{}' must return a scalar".format(term.name))
    return value.detach().to(torch.float64).cpu()


def _single_case(batch: Mapping[str, Any]) -> None:
    rho = batch.get("rho")
    if rho is None:
        raise KeyError("case statistics require rho in every batch")
    if rho.ndim >= 3 and rho.shape[0] != 1:
        raise ValueError(
            "case statistics require one case per batch (batch_size=1)"
        )


def _perturbation(batch, keys, generator):
    """Draw one Gaussian perturbation per field from its standard error."""

    draws = {}
    for field, std_key in keys.items():
        if std_key is None or std_key not in batch:
            continue
        if field not in batch:
            raise KeyError("batch has {} but no {}".format(std_key, field))
        std = batch[std_key].detach()
        reference = batch[field]
        if std.shape != reference.shape:
            raise ValueError("{} must have the shape of {}".format(std_key, field))
        if not torch.all(torch.isfinite(std) & (std >= 0)).item():
            raise ValueError(std_key + " must be finite and nonnegative")
        noise = torch.randn(
            std.shape, generator=generator, dtype=torch.float64, device="cpu",
        ).to(reference)
        draws[field] = noise * std.to(reference)
    return draws


def _perturbed(batch, draws, step):
    values = dict(batch)
    for field, draw in draws.items():
        values[field] = batch[field] + step * draw
    return values


def _check_domain(model, clean, perturbed):
    """Reject perturbations that leave the ideal domain of a valid voxel."""

    rho = perturbed["rho"]
    clean_rho = clean["rho"]
    valid = clean_rho > 0
    if torch.any(valid & ~(rho > 0)).item():
        return False
    ideal = getattr(model, "ideal", None)
    if "dipole_density" in perturbed and isinstance(ideal, FixedDipoleIdeal):
        moment = ideal.dipole_magnitude.to(rho)
        moment = moment.reshape(*((1,) * (rho.ndim - 1)), -1) if moment.ndim else moment
        bound = rho * moment
        clean_bound = clean_rho * moment
        clean_inside = valid & (
            clean["dipole_density"].square().sum(-1) < clean_bound.square()
        )
        inside = perturbed["dipole_density"].square().sum(-1) < bound.square()
        if torch.any(clean_inside & ~inside).item():
            return False
    return True


def _difference_outputs(plus, minus, clean, step, frozen):
    """Central difference of every floating output; masks and frozen keys stay clean."""

    result = {}
    for key, value in clean.items():
        if (
            key in frozen
            or not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
        ):
            result[key] = value
        else:
            result[key] = (plus[key] - minus[key]) / (2.0 * step)
    return result


def _target_free_batch(batch, target_keys):
    values = dict(batch)
    for key in target_keys:
        if key in values and isinstance(values[key], torch.Tensor):
            values[key] = torch.zeros_like(values[key])
    return values


def _batch_targets(terms, batch, target_keys):
    keys = set(target_keys)
    for term in terms:
        for key in getattr(term, "target_keys", ()):
            if key in batch:
                keys.add(key)
    return keys


def _selection_masks(outputs, frozen):
    """Boolean selections whose flips break the linear response."""

    masks = {}
    for key, value in outputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key in frozen:
            masks[key] = value != 0
        elif value.dtype == torch.bool:
            masks[key] = value
    return masks


def _selection_flips(clean_masks, outputs):
    """Return (all flips, activations) of the clean selections in ``outputs``."""

    flips = activations = 0
    for key, clean in clean_masks.items():
        perturbed = outputs[key]
        perturbed = perturbed if perturbed.dtype == torch.bool else perturbed != 0
        flips += int((perturbed != clean).sum())
        activations += int((perturbed & ~clean).sum())
    return flips, activations


def _require_ordered(batches, case_key):
    sampler = getattr(batches, "sampler", None)
    if (
        case_key is None
        and isinstance(batches, DataLoader)
        and not isinstance(sampler, SequentialSampler)
    ):
        raise ValueError(
            "case statistics follow the order of the batches: use an unshuffled "
            "DataLoader, or give case_key so every result row carries its case"
        )


def _case_identity(batch, case_key):
    value = batch.get(case_key)
    if value is None:
        raise KeyError("batch is missing case key '{}'".format(case_key))
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("case key '{}' must hold one case".format(case_key))
        value = value[0]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("case key '{}' must hold one case".format(case_key))
        value = value.reshape(()).item()
    return value


@torch.no_grad()
def case_loss_statistics(
    model: nn.Module,
    loss: Loss,
    batches: Iterable[Mapping[str, Any]],
    *,
    terms: Sequence[str],
    noise_samples: int = 4,
    noise_step: float = 0.1,
    target_keys: Sequence[str] = (),
    case_key: Optional[str] = None,
    density_std_key: Optional[str] = "rho_std",
    polarization_std_key: Optional[str] = "dipole_density_std",
    frozen_outputs: Sequence[str] = ("voxel_weights",),
    max_step_halvings: int = 40,
    seed: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    model_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return every case's signal and noise scale for the named loss terms.

    ``batches`` yields one case per batch (``batch_size=1``). Result rows
    follow the batch order; with ``case_key`` (a batch key such as
    ``"case_id"``) the identities are returned as ``cases``, and a shuffled
    ``DataLoader`` is refused without one. For each case and term:

    * ``signal`` is the term's unweighted value at ``model`` as it is now
      (pass the reference functional, normally the freshly initialised
      model before the fit);
    * ``noise`` is the term evaluated on the *linear response* of the model
      outputs to the sampling error of the input fields, with every target
      set to zero: for a Gaussian draw ``xi`` scaled by ``rho_std`` (and
      ``dipole_density_std`` where present), each floating output ``o`` is
      replaced by the directional derivative ``[o(x + h xi) - o(x - h xi)] /
      (2 h)``. For a quadratic term this is ``|J xi|^2`` in the term's own
      metric (masks, filters, gauges and long-range kernels included), the
      residual a functional with no model error would leave on these fields;
      it does not depend on the reference's own residual. It is averaged over
      ``noise_samples`` draws; ``noise_sem`` is the standard error.

    Every draw is kept: the Gaussian whose spread is the stated SEM is not
    truncated. ``h`` starts at ``noise_step`` and is halved for that draw
    (at most ``max_step_halvings`` times) until ``x +- h xi`` stays inside
    ``rho > 0`` and ``|P| < m rho`` at every valid voxel and no selection
    changes: the outputs in ``frozen_outputs`` (nonzero pattern) and every
    Boolean output (such as the energy-centre mask) must equal their clean
    values, so the difference is the derivative on the clean selection
    branch. ``selection_changes`` and ``selection_activations`` count the
    voxels that flipped (respectively became selected) at the first step
    tried for each draw, before any reduction; ``step_reductions`` counts the
    draws that needed a smaller step and ``minimum_step`` is the smallest
    used. A draw whose selection still flips after all halvings raises.

    Targets are zeroed in the batch: every batch key in a term's
    ``target_keys`` (as for :class:`~equicdft.TensorLoss`) and any listed in
    ``target_keys`` here, for custom terms. A target computed by the model
    (such as a gauge average) is differenced like any output. Each term is
    checked to vanish for zero outputs, so an undeclared batch target raises.
    Terms inactive in evaluation mode (training-only regularizers) should
    not be listed. The model is put in evaluation mode and restored. Test
    cases must not be passed.
    """

    names = unique_strings(terms, "terms")
    available = {term.name: term for term in loss.terms}
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError("loss has no terms {}".format(missing))
    selected_terms = [available[name] for name in names]
    noise_samples = nonnegative_integer(noise_samples, "noise_samples")
    noise_step = positive_scalar(noise_step, "noise_step")
    max_step_halvings = nonnegative_integer(max_step_halvings, "max_step_halvings")
    seed = nonnegative_integer(seed, "seed")
    if isinstance(target_keys, str):
        target_keys = (target_keys,)
    target_keys = tuple(target_keys)
    _require_ordered(batches, case_key)
    std_keys = {"rho": density_std_key, "dipole_density": polarization_std_key}
    frozen_outputs = tuple(frozen_outputs)
    device = module_device(model) if device is None else torch.device(device)
    kwargs = dict(model_kwargs or {})
    if getattr(model, "voxel_weights", None) is not None:
        kwargs.setdefault("use_voxel_mask", True)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    signal, noise, noise_sem, cases = [], [], [], []
    changes, activations, reductions, minimum_steps = [], [], [], []
    was_training = model.training
    model.eval()
    try:
        for raw in batches:
            _single_case(raw)
            if case_key is not None:
                cases.append(_case_identity(raw, case_key))
            batch = move_batch(dict(raw), device)
            outputs = model(batch, **kwargs)
            signal.append(torch.stack([
                _term_value(term, outputs, batch, model) for term in selected_terms
            ]))
            target_free = _target_free_batch(
                batch, _batch_targets(selected_terms, batch, target_keys),
            )
            if noise_samples:
                zero = _difference_outputs(outputs, outputs, outputs, 1.0, frozen_outputs)
                for term in selected_terms:
                    if float(_term_value(term, zero, target_free, model)) != 0.0:
                        raise ValueError(
                            "loss term '{}' does not vanish for zero outputs: list "
                            "its batch targets in target_keys".format(term.name)
                        )
            clean_masks = _selection_masks(outputs, frozen_outputs)
            draws_values = []
            flipped = activated = reduced = 0
            smallest = noise_step if noise_samples else float("nan")
            for _ in range(noise_samples):
                draws = _perturbation(batch, std_keys, generator)
                if not draws:
                    raise KeyError(
                        "noise statistics need {} or {} in every batch".format(
                            density_std_key, polarization_std_key,
                        )
                    )
                step, first = noise_step, True
                for halving in range(max_step_halvings + 1):
                    pair = [_perturbed(batch, draws, sign * step) for sign in (1.0, -1.0)]
                    inside = all(_check_domain(model, batch, values) for values in pair)
                    if inside:
                        plus, minus = (model(values, **kwargs) for values in pair)
                        counts = [_selection_flips(clean_masks, o) for o in (plus, minus)]
                        if first:
                            flipped += sum(count[0] for count in counts)
                            activated += sum(count[1] for count in counts)
                            first = False
                        if all(count[0] == 0 for count in counts):
                            break
                    if halving == max_step_halvings:
                        raise RuntimeError(
                            "a noise draw still leaves the ideal domain or changes "
                            "the selection after {} step halvings (step {:.3g})".format(
                                max_step_halvings, step,
                            )
                        )
                    step = step / 2.0
                if step < noise_step:
                    reduced += 1
                smallest = min(smallest, step)
                response = _difference_outputs(plus, minus, outputs, step, frozen_outputs)
                draws_values.append(torch.stack([
                    _term_value(term, response, target_free, model)
                    for term in selected_terms
                ]))
            if draws_values:
                stacked = torch.stack(draws_values)
                noise.append(stacked.mean(0))
                noise_sem.append(
                    stacked.std(0, unbiased=True) / math.sqrt(len(draws_values))
                    if len(draws_values) > 1
                    else torch.full_like(stacked[0], float("nan"))
                )
            else:
                noise.append(torch.zeros(len(names), dtype=torch.float64))
                noise_sem.append(torch.zeros(len(names), dtype=torch.float64))
            changes.append(flipped)
            activations.append(activated)
            reductions.append(reduced)
            minimum_steps.append(smallest)
    finally:
        model.train(was_training)
    if not signal:
        raise ValueError("no batches were supplied")
    result = {
        "terms": names,
        "signal": torch.stack(signal),
        "noise": torch.stack(noise),
        "noise_sem": torch.stack(noise_sem),
        "selection_changes": torch.tensor(changes, dtype=torch.int64),
        "selection_activations": torch.tensor(activations, dtype=torch.int64),
        "step_reductions": torch.tensor(reductions, dtype=torch.int64),
        "minimum_step": torch.tensor(minimum_steps, dtype=torch.float64),
        "noise_samples": noise_samples,
        "noise_step": noise_step,
    }
    if case_key is not None:
        if len(set(cases)) != len(cases):
            raise ValueError("case key '{}' is not unique".format(case_key))
        result["cases"] = list(cases)
    return result


def _per_term(value, names, label):
    if isinstance(value, Mapping):
        missing = [name for name in names if name not in value]
        if missing:
            raise KeyError("{} is missing terms {}".format(label, missing))
        values = [value[name] for name in names]
    else:
        values = [value] * len(names)
    result = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(label + " must be a real number or a mapping of them")
        result.append(float(item))
    return torch.tensor(result, dtype=torch.float64)


def case_loss_weights(
    statistics: Mapping[str, Any],
    *,
    relative_model_error: Union[float, Mapping[str, float]],
    noise_scale: Union[str, float, Mapping[str, float], torch.Tensor] = 1.0,
    signal_quantile: float = 0.05,
    power: Union[float, Mapping[str, float]] = 1.0,
    normalization: Union[str, torch.Tensor, Sequence[float]] = "mean",
) -> Dict[str, Any]:
    """Return frozen per-case weights ``c (noise_scale N + eta S)^-power``.

    ``statistics`` comes from :func:`case_loss_statistics`. Per term,
    ``relative_model_error`` (``eta``) is the fraction of the signal a good
    fit is still expected to miss; small values approach noise (chi-squared)
    weighting, large ones relative-error weighting. ``power`` in ``(0, 1]``
    tempers the spread of weights (``0.5`` takes their square root). Each may
    be one number or a mapping per term.

    ``noise_scale`` multiplies the measured noise. Its default of 1 uses the
    diagonal ``*_std`` uncertainty model (independent voxel/component errors).
    Correlations can make the true propagated noise larger or smaller.
    ``noise_scale="signal_quantile"`` is an exploratory heuristic: it uses
    the ``signal_quantile`` of ``S / N`` over cases with measured noise,
    separately for each term. This is not an estimator or bound for the true
    correlated-noise scale: even with an exact reference functional, an
    individual squared residual can be below its expected noise. A defensible
    scale needs independent MD blocks/replicas or a covariance-aware noise
    estimate. The chosen values are returned as ``noise_scale`` for reuse on
    other splits.

    ``normalization="mean"`` sets ``c`` per term so the weights of these cases
    average to one; ``"reference_loss"`` so the weighted signal sums to the
    unweighted one (the objective at the reference is unchanged). Weights of
    another split, e.g. validation, are made with the training split's
    ``normalization`` tensor from the result, so both are on one scale.
    Returns ``{"terms", "weights" [case, term], "scales", "noise_scale",
    "normalization"}``; when the statistics carry ``cases``, also ``cases``
    and ``by_case`` (case identity -> {term: weight}), the assignment to use.
    """

    names = tuple(statistics["terms"])
    signal = torch.as_tensor(statistics["signal"], dtype=torch.float64)
    noise = torch.as_tensor(statistics["noise"], dtype=torch.float64)
    if signal.ndim != 2 or signal.shape != noise.shape or signal.shape[1] != len(names):
        raise ValueError("statistics must hold [case, term] signal and noise")
    eta = _per_term(relative_model_error, names, "relative_model_error")
    if isinstance(noise_scale, str):
        if noise_scale != "signal_quantile":
            raise ValueError("noise_scale must be a number, a mapping, a tensor or 'signal_quantile'")
        quantile = float(signal_quantile)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("signal_quantile must lie in [0, 1]")
        values = []
        for column in range(len(names)):
            measured = noise[:, column] > 0
            if not torch.any(measured):
                raise ValueError(
                    "term '{}' has no measured noise for the signal quantile".format(
                        names[column]
                    )
                )
            ratio = signal[measured, column] / noise[measured, column]
            values.append(torch.quantile(ratio, quantile))
        noise_scale = torch.stack(values)
    elif isinstance(noise_scale, torch.Tensor):
        noise_scale = noise_scale.to(torch.float64).reshape(-1)
        if noise_scale.shape != (len(names),):
            raise ValueError("a noise_scale tensor needs one value per term")
    else:
        noise_scale = _per_term(noise_scale, names, "noise_scale")
    power = _per_term(power, names, "power")
    if torch.any(~torch.isfinite(eta) | (eta < 0)) or torch.any(~torch.isfinite(noise_scale) | (noise_scale < 0)):
        raise ValueError("relative_model_error and noise_scale must be finite and nonnegative")
    if torch.any(~(power > 0) | (power > 1)):
        raise ValueError("power must lie in (0, 1]")
    if not torch.all(torch.isfinite(signal) & (signal >= 0)) or not torch.all(torch.isfinite(noise)):
        raise ValueError("signal must be finite and nonnegative, noise finite")
    if torch.any(noise < 0):
        raise ValueError("noise must be nonnegative")
    scales = noise_scale * noise + eta * signal
    if torch.any(scales <= 0):
        bad = torch.nonzero(scales <= 0)[0].tolist()
        raise ValueError(
            "case {} has zero scale for term '{}': give it noise statistics or "
            "a positive relative_model_error".format(bad[0], names[bad[1]])
        )
    raw = scales.pow(-power)
    if isinstance(normalization, str):
        if normalization == "mean":
            constant = 1.0 / raw.mean(0)
        elif normalization == "reference_loss":
            constant = signal.sum(0) / (raw * signal).sum(0)
            if not torch.all(torch.isfinite(constant) & (constant > 0)):
                raise ValueError("reference_loss normalization needs a positive signal")
        else:
            raise ValueError("normalization must be 'mean', 'reference_loss' or a tensor")
    else:
        constant = torch.as_tensor(normalization, dtype=torch.float64).reshape(-1)
        if constant.shape != (len(names),) or not torch.all(torch.isfinite(constant) & (constant > 0)):
            raise ValueError("a normalization tensor needs one positive value per term")
    result = {
        "terms": names,
        "weights": constant * raw,
        "scales": scales,
        "noise_scale": noise_scale,
        "normalization": constant,
    }
    if "cases" in statistics:
        result["cases"] = list(statistics["cases"])
        result["by_case"] = {
            case: {name: float(value) for name, value in zip(names, row)}
            for case, row in zip(result["cases"], result["weights"])
        }
    return result
