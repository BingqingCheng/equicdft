"""Training orchestration for grid density-functional models."""

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from ._argument_checks import (
    boolean,
    nonnegative_integer,
    nonempty_string,
    nonnegative_scalar,
    optional_positive_integer,
    positive_integer,
    positive_scalar,
)
from ._ema import _ParameterEMA
from ._parameter_groups import (
    build_optimizer_parameters,
    learning_rate_record,
    scheduler_arguments_for_groups,
)
from ._trainer_io import (
    append_log_message,
    atomic_torch_save,
    format_record,
    write_history_csv,
)
from .loss import Loss


_MAX_CACHED_GRID_GEOMETRIES = 2


@dataclass
class TrainingStream:
    """One data/loss/metric stream in a joint optimization.

    ``batches_per_step`` controls how many batches from this stream contribute
    to one optimizer update. Their losses are averaged before applying the
    stream weight. A stream with ``cycle=True`` is restarted when exhausted
    and follows the epoch length set by the non-cycling streams. Validation is
    always single-pass. ``model_kwargs`` makes response-only forwards such as
    ``compute_c1=False`` explicit without mutating persistent model flags.
    """

    name: str
    train_loader: Iterable[Dict[str, torch.Tensor]]
    valid_loader: Iterable[Dict[str, torch.Tensor]]
    loss: Loss
    metrics: Sequence[nn.Module] = ()
    batches_per_step: int = 1
    model_kwargs: Optional[Dict[str, Any]] = None
    cycle: bool = False

    def __post_init__(self) -> None:
        self.name = nonempty_string(self.name, "stream name")
        if not isinstance(self.loss, Loss):
            raise TypeError("stream loss must be a equicdft.Loss")
        self.metrics = tuple(self.metrics)
        for metric in self.metrics:
            if not isinstance(metric, nn.Module):
                raise TypeError("stream metrics must be torch modules")
            nonempty_string(getattr(metric, "name", None), "metric name")
        metric_names = [metric.name for metric in self.metrics]
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("metric collection names must be unique per stream")
        self.batches_per_step = positive_integer(
            self.batches_per_step,
            "batches_per_step",
        )
        self.model_kwargs = dict(self.model_kwargs or {})
        self.cycle = boolean(self.cycle, "cycle")


class Trainer(nn.Module):
    """Train a model with composable losses and dataset-level metrics.

    The trainer owns optimization and epoch orchestration while the model,
    loss, and metric modules retain their separate responsibilities. Optimizer
    construction is delayed until :meth:`fit`, after any lazy parameters have
    been materialized from the first training batch.

    Parameters
    ----------
    model
        Model mapping a batched grid dictionary to an output dictionary.
    loss
        Aggregate :class:`equicdft.loss.Loss` module.
    metrics
        Metric collections updated for every train and validation batch.
    optimizer_cls
        PyTorch optimizer class.
    optimizer_args
        Keyword arguments passed to ``optimizer_cls``.
    scheduler_cls
        Optional PyTorch scheduler class. ``ReduceLROnPlateau`` receives the
        validation loss; other schedulers are stepped once after each epoch.
    scheduler_args
        Keyword arguments passed to ``scheduler_cls``.
    device
        Device used for the model and each tensor in a batch.
    checkpoint_dir
        Optional output directory. ``None`` disables checkpoint writing.
    checkpoint_interval
        Write ``checkpoint_epoch_XXXX.pt`` every this many epochs.
    save_best
        When checkpointing is enabled, update ``best.pt`` whenever validation
        loss improves. ``last.pt`` is updated after every epoch regardless.
    early_stopping_patience
        Optional number of consecutive epochs without validation-loss
        improvement allowed before stopping. ``None`` disables early
        stopping. The counter is preserved across checkpoint restarts.
    log_dir
        Optional directory for ``history.csv`` and ``training.log``.
        ``history.csv`` is atomically reconstructed from the complete trainer
        history after every epoch and after loading a checkpoint.
    feature_learning_rate_multiplier
        Positive multiplier applied to parameters explicitly published by
        ``model.feature_parameters``.  The default ``1.0`` retains the
        established single optimizer group.  The model, not the trainer,
        defines which parameters belong to the feature representation.
    learning_rate_warmup_steps
        Number of optimizer updates over which every learning-rate group is
        ramped linearly from ``1 / learning_rate_warmup_steps`` of its target
        value to the target value. ``0`` disables warmup.
    ema_decay
        Optional exponential-moving-average decay for trainable model
        parameters. ``None`` preserves the established raw-parameter
        validation and checkpoint behavior.
    ema_start_step
        Number of optimizer updates to complete before EMA tracking begins.
        The first EMA update copies the current model exactly, avoiding
        initialization bias.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Loss,
        metrics: Sequence[nn.Module] = (),
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type[Any]] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        device: Union[str, torch.device] = "cpu",
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_interval: int = 1,
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None,
        log_dir: Optional[Union[str, Path]] = None,
        feature_learning_rate_multiplier: float = 1.0,
        learning_rate_warmup_steps: int = 0,
        ema_decay: Optional[float] = None,
        ema_start_step: int = 0,
    ) -> None:
        super().__init__()

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(loss, Loss):
            raise TypeError("loss must be a equicdft.Loss")
        checkpoint_interval = positive_integer(
            checkpoint_interval,
            "checkpoint_interval",
        )
        save_best = boolean(save_best, "save_best")
        early_stopping_patience = optional_positive_integer(
            early_stopping_patience,
            "early_stopping_patience",
        )
        feature_learning_rate_multiplier = positive_scalar(
            feature_learning_rate_multiplier,
            "feature_learning_rate_multiplier",
        )
        learning_rate_warmup_steps = nonnegative_integer(
            learning_rate_warmup_steps,
            "learning_rate_warmup_steps",
        )
        ema_start_step = nonnegative_integer(ema_start_step, "ema_start_step")
        if ema_decay is not None:
            ema_decay = positive_scalar(ema_decay, "ema_decay")
            if ema_decay >= 1.0:
                raise ValueError("ema_decay must be smaller than one")
        elif ema_start_step != 0:
            raise ValueError("ema_start_step requires ema_decay")

        metrics = list(metrics)
        if any(not isinstance(metric, nn.Module) for metric in metrics):
            raise TypeError("metrics must contain only torch modules")
        for metric in metrics:
            for method in ("update_metrics", "retrieve_metrics", "clear_metrics"):
                if not callable(getattr(metric, method, None)):
                    raise TypeError(
                        "every metric must define {}".format(method)
                    )
        metric_names = [metric.name for metric in metrics]
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("metric collection names must be unique")

        self.model = model
        self.loss = loss
        self.metrics = nn.ModuleList(metrics)
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = dict(optimizer_args or {"lr": 1.0e-3})
        self.scheduler_cls = scheduler_cls
        self.scheduler_args = dict(scheduler_args or {})
        self.device = torch.device(device)
        self.checkpoint_interval = checkpoint_interval
        self.save_best = save_best
        self.early_stopping_patience = early_stopping_patience
        self.feature_learning_rate_multiplier = (
            feature_learning_rate_multiplier
        )
        self.feature_parameter_names = ()
        self.learning_rate_warmup_steps = learning_rate_warmup_steps
        self._warmup_target_learning_rates = ()
        self.ema_decay = ema_decay
        self.ema_start_step = ema_start_step
        self.optimizer_steps = 0
        self._ema = None if ema_decay is None else _ParameterEMA(ema_decay)

        self.checkpoint_dir = None
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir).expanduser()
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = None
        self.history_path = None
        self.training_log_path = None
        if log_dir is not None:
            self.log_dir = Path(log_dir).expanduser()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.history_path = self.log_dir / "history.csv"
            self.training_log_path = self.log_dir / "training.log"

        self.optimizer = None
        self.scheduler = None
        self.history = []
        self.best_valid_loss = math.inf
        self.epochs_without_improvement = 0
        self._train_loader_generator = None
        self.stream_losses = nn.ModuleDict()
        self.stream_metrics = nn.ModuleDict()
        self._streams = ()
        self._stream_loader_generators = {}
        self._grid_geometry_device_cache = OrderedDict()
        self.to(self.device)

    def fit(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        valid_loader: Iterable[Dict[str, torch.Tensor]],
        epochs: int,
        verbose: bool = True,
        print_interval: int = 1,
    ) -> List[Dict[str, Any]]:
        """Run complete train/validation epochs and return their history."""

        epochs = positive_integer(epochs, "epochs")
        verbose = boolean(verbose, "verbose")
        print_interval = positive_integer(print_interval, "print_interval")

        self._train_loader_generator = getattr(train_loader, "generator", None)
        self._initialize_optimization(train_loader)
        start_epoch = self._next_epoch()
        stream = TrainingStream(
            "default",
            train_loader,
            valid_loader,
            self.loss,
            metrics=tuple(self.metrics),
        )

        if self._stop_before_training(verbose):
            return self.history

        for epoch in range(start_epoch, start_epoch + epochs):
            train_losses, train_metrics = self._run_training_streams(
                (stream,),
                {stream.name: 1.0},
            )[stream.name]
            valid_losses, valid_metrics = self._evaluate_stream(
                stream,
                subset="valid",
            )

            record = {
                "epoch": epoch,
                **self._optimization_record(),
                "train_losses": train_losses,
                "valid_losses": valid_losses,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
            }
            if self._finish_epoch(record, verbose, print_interval):
                break

        return self.history

    def fit_streams(
        self,
        streams: Sequence[TrainingStream],
        epochs: int,
        stream_weights: Optional[
            Mapping[str, Union[float, Callable[[int], float]]]
        ] = None,
        verbose: bool = True,
        print_interval: int = 1,
        record_initial_validation: bool = False,
    ) -> List[Dict[str, Any]]:
        """Jointly optimize multiple data streams without mixing datasets.

        Non-cycling streams are consumed exactly once per epoch. At each
        optimizer update, up to ``batches_per_step`` batches are drawn from
        every still-active non-cycling stream and every cycling stream. Cycling
        streams restart as needed until the non-cycling streams are exhausted.
        Losses are averaged within each stream. Stream weights may be constants
        or callables of the one-based epoch number. Validation is evaluated
        exactly once for every stream; the weighted sum of validation totals
        drives the configured scheduler and ordinary ``best.pt``.
        If requested, the untrained validation result is first recorded as
        epoch 0, using stream-weight callables evaluated at zero.
        """

        epochs = positive_integer(epochs, "epochs")
        record_initial_validation = boolean(
            record_initial_validation,
            "record_initial_validation",
        )
        verbose = boolean(verbose, "verbose")
        print_interval = positive_integer(print_interval, "print_interval")
        streams = self._configure_streams(streams)
        weight_specification = dict(stream_weights or {})
        unknown_weights = set(weight_specification) - {
            stream.name for stream in streams
        }
        if unknown_weights:
            raise KeyError(
                "stream_weights contains unknown streams: {}".format(
                    sorted(unknown_weights)
                )
            )

        self._initialize_optimization(
            streams[0].train_loader,
            model_kwargs=streams[0].model_kwargs,
        )
        if record_initial_validation and not self.history:
            self._record_stream_baseline(
                streams,
                self._resolve_stream_weights(
                    streams,
                    weight_specification,
                    epoch=0,
                ),
                verbose,
            )
        start_epoch = self._next_epoch()
        if self._stop_before_training(verbose):
            return self.history

        for epoch in range(start_epoch, start_epoch + epochs):
            weights = self._resolve_stream_weights(
                streams,
                weight_specification,
                epoch,
            )
            train_by_stream = self._run_training_streams(streams, weights)
            valid_by_stream = {
                stream.name: self._evaluate_stream(
                    stream,
                    subset="valid",
                )
                for stream in streams
            }
            train_losses, train_metrics = self._flatten_stream_results(
                train_by_stream,
                weights,
            )
            valid_losses, valid_metrics = self._flatten_stream_results(
                valid_by_stream,
                weights,
            )
            record = {
                "epoch": epoch,
                **self._optimization_record(),
                "stream_weights": weights,
                "train_losses": train_losses,
                "valid_losses": valid_losses,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
            }
            if self._finish_epoch(record, verbose, print_interval):
                break
        return self.history

    def evaluate_streams(
        self,
        streams: Sequence[TrainingStream],
        subset: str = "valid",
    ) -> Dict[str, Tuple[Dict[str, float], Dict[str, Dict[str, float]]]]:
        """Evaluate named streams independently without optimizer updates."""

        if subset not in ("train", "valid"):
            raise ValueError("subset must be 'train' or 'valid'")
        streams = self._validate_streams(streams)
        return {
            stream.name: self._evaluate_stream(
                stream,
                subset=subset,
            )
            for stream in streams
        }

    def _record_stream_baseline(
        self,
        streams: Sequence[TrainingStream],
        weights: Mapping[str, float],
        verbose: bool,
    ) -> None:
        """Append validation-only epoch 0 before the first optimizer update."""

        evaluated = {
            stream.name: self._evaluate_stream(
                stream,
                subset="valid",
            )
            for stream in streams
        }
        valid_losses, valid_metrics = self._flatten_stream_results(
            evaluated,
            weights,
        )
        record = {
            "epoch": 0,
            **self._optimization_record(),
            "stream_weights": weights,
            "train_losses": {},
            "valid_losses": valid_losses,
            "train_metrics": {},
            "valid_metrics": valid_metrics,
        }
        self.history.append(record)
        self._write_history_csv()
        if verbose:
            self.log_message(
                "Untrained validation baseline\n" + format_record(record)
            )

    def save_checkpoint(
        self,
        path: Union[str, Path],
        record: Dict[str, Any],
    ) -> None:
        """Write model, optimization, scheduler, and history state."""

        evaluation_model_state = self.evaluation_model_state_dict()
        checkpoint = {
            "checkpoint_version": 1,
            "epoch": record["epoch"],
            "model_state_dict": evaluation_model_state,
            "loss_state_dict": self.loss.state_dict(),
            "stream_loss_state_dict": (
                self.stream_losses.state_dict() if self._streams else None
            ),
            "stream_names": [stream.name for stream in self._streams],
            "optimizer_state_dict": self.optimizer.state_dict(),
            "feature_learning_rate_multiplier": (
                self.feature_learning_rate_multiplier
            ),
            "feature_parameter_names": self.feature_parameter_names,
            "scheduler_state_dict": (
                None if self.scheduler is None else self.scheduler.state_dict()
            ),
            "record": record,
            "history": self.history,
            "best_valid_loss": self.best_valid_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "train_loader_generator_state": (
                self._train_loader_generator.get_state()
                if self._train_loader_generator is not None
                else None
            ),
            "stream_loader_generator_states": {
                name: (
                    generator.get_state() if generator is not None else None
                )
                for name, generator in self._stream_loader_generators.items()
            },
        }
        if self._ema is not None:
            checkpoint.update(
                {
                    "raw_model_state_dict": self.model.state_dict(),
                    "ema_state_dict": self._ema.state_dict(),
                    "ema_start_step": self.ema_start_step,
                }
            )
        if self.learning_rate_warmup_steps:
            checkpoint["learning_rate_warmup_steps"] = (
                self.learning_rate_warmup_steps
            )
        if self._ema is not None or self.learning_rate_warmup_steps:
            checkpoint["optimizer_steps"] = self.optimizer_steps
        atomic_torch_save(checkpoint, Path(path).expanduser())

    def load_checkpoint(
        self,
        path: Union[str, Path],
        train_loader: Iterable[Dict[str, torch.Tensor]],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Restore a complete training state and return its completed epoch.

        Lazy parameters are first materialized from ``train_loader`` so that
        model, optimizer, and scheduler state dictionaries can be restored in
        their normal order. Random states are restored last, undoing any draws
        made while constructing the replacement process.
        """

        checkpoint_path = Path(path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "checkpoint does not exist: {}".format(checkpoint_path)
            )

        self._train_loader_generator = getattr(train_loader, "generator", None)
        self._initialize_optimization(train_loader, model_kwargs=model_kwargs)
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must contain a dictionary")

        required_keys = {
            "epoch",
            "model_state_dict",
            "loss_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "history",
        }
        missing_keys = required_keys - set(checkpoint)
        if missing_keys:
            raise ValueError(
                "checkpoint is missing keys: {}".format(
                    sorted(missing_keys)
                )
            )

        checkpoint_multiplier = checkpoint.get(
            "feature_learning_rate_multiplier",
            1.0,
        )
        if checkpoint_multiplier != self.feature_learning_rate_multiplier:
            raise ValueError(
                "checkpoint feature_learning_rate_multiplier does not "
                "match the configured trainer"
            )
        checkpoint_feature_names = checkpoint.get("feature_parameter_names")
        if (
            checkpoint_feature_names is not None
            and tuple(checkpoint_feature_names) != self.feature_parameter_names
        ):
            raise ValueError(
                "checkpoint feature parameter names do not match the model"
            )

        checkpoint_warmup_steps = checkpoint.get(
            "learning_rate_warmup_steps",
            0,
        )
        if checkpoint_warmup_steps != self.learning_rate_warmup_steps:
            raise ValueError(
                "checkpoint learning_rate_warmup_steps does not match trainer"
            )

        checkpoint_ema = checkpoint.get("ema_state_dict")
        if (checkpoint_ema is None) != (self._ema is None):
            raise ValueError("checkpoint EMA configuration does not match trainer")
        if self._ema is not None:
            if checkpoint.get("ema_start_step") != self.ema_start_step:
                raise ValueError(
                    "checkpoint ema_start_step does not match the trainer"
                )
        optimizer_steps = checkpoint.get("optimizer_steps")
        if self._ema is not None or self.learning_rate_warmup_steps:
            if (
                isinstance(optimizer_steps, bool)
                or not isinstance(optimizer_steps, int)
                or optimizer_steps < 0
            ):
                raise ValueError(
                    "checkpoint optimizer step count must be nonnegative"
                )

        model_state = checkpoint["model_state_dict"]
        if self._ema is not None:
            model_state = checkpoint.get("raw_model_state_dict")
            if model_state is None:
                raise ValueError(
                    "EMA checkpoint has no raw model state for restart"
                )
        self.model.load_state_dict(model_state, strict=True)
        if self._ema is not None:
            self._ema.load_state_dict(
                checkpoint_ema,
                checkpoint["model_state_dict"],
                self.model,
            )
        if self._ema is not None or self.learning_rate_warmup_steps:
            self.optimizer_steps = optimizer_steps
        self.loss.load_state_dict(checkpoint["loss_state_dict"], strict=True)
        stream_loss_state = checkpoint.get("stream_loss_state_dict")
        if stream_loss_state is not None:
            expected_names = checkpoint.get("stream_names", [])
            actual_names = [stream.name for stream in self._streams]
            if expected_names != actual_names:
                raise ValueError(
                    "checkpoint stream names do not match configured streams"
                )
            self.stream_losses.load_state_dict(stream_loss_state, strict=True)
        elif self._streams:
            raise ValueError(
                "checkpoint has no stream-loss state for configured streams"
            )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._move_optimizer_state_to_device()

        scheduler_state = checkpoint["scheduler_state_dict"]
        if self.scheduler is None and scheduler_state is not None:
            raise ValueError(
                "checkpoint contains scheduler state but no scheduler is configured"
            )
        if self.scheduler is not None and scheduler_state is None:
            raise ValueError(
                "checkpoint has no scheduler state but a scheduler is configured"
            )
        if self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        epoch = checkpoint["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError("checkpoint epoch must be a positive integer")
        history = checkpoint["history"]
        if not isinstance(history, list):
            raise ValueError("checkpoint history must be a list")
        history_epochs = [record.get("epoch") for record in history]
        expected_epochs = list(range(1, epoch + 1))
        if history_epochs not in (expected_epochs, [0] + expected_epochs):
            raise ValueError(
                "checkpoint history must contain epochs 1 through its "
                "completed epoch, with at most one leading epoch 0 baseline"
            )
        self.history = history

        if "best_valid_loss" in checkpoint:
            self.best_valid_loss = float(checkpoint["best_valid_loss"])
        else:
            self.best_valid_loss = min(
                record["valid_losses"]["total"]
                for record in history
                if record["epoch"] > 0
            )
        if "epochs_without_improvement" in checkpoint:
            self.epochs_without_improvement = int(
                checkpoint["epochs_without_improvement"]
            )
        else:
            best_loss = math.inf
            epochs_without_improvement = 0
            for record in history:
                if record["epoch"] == 0:
                    continue
                valid_loss = record["valid_losses"]["total"]
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            self.epochs_without_improvement = epochs_without_improvement

        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda_states = checkpoint.get("cuda_rng_state_all")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
        loader_state = checkpoint.get("train_loader_generator_state")
        if loader_state is not None:
            if self._train_loader_generator is None:
                raise ValueError(
                    "checkpoint contains a train-loader random state, but the "
                    "current loader has no generator"
                )
            self._train_loader_generator.set_state(loader_state.cpu())

        stream_loader_states = checkpoint.get(
            "stream_loader_generator_states",
            {},
        )
        for name, state in stream_loader_states.items():
            if name not in self._stream_loader_generators:
                raise ValueError(
                    "checkpoint contains an unknown stream loader '{}'".format(name)
                )
            generator = self._stream_loader_generators[name]
            if state is not None:
                if generator is None:
                    raise ValueError(
                        "checkpoint contains a loader state for stream '{}', "
                        "but its current loader has no generator".format(name)
                    )
                generator.set_state(state.cpu())

        for metric in self.metrics:
            for subset in metric.logs:
                metric.clear_metrics(subset)
        for metrics in self.stream_metrics.values():
            for metric in metrics:
                for subset in metric.logs:
                    metric.clear_metrics(subset)
        self._write_history_csv()
        return epoch

    def load_stream_checkpoint(
        self,
        path: Union[str, Path],
        streams: Sequence[TrainingStream],
    ) -> int:
        """Restore a joint-stream checkpoint using the same stream layout."""

        streams = self._configure_streams(streams)
        return self.load_checkpoint(
            path,
            train_loader=streams[0].train_loader,
            model_kwargs=streams[0].model_kwargs,
        )

    def log_message(self, message: str, display: bool = True) -> None:
        """Print a message and append it to the human-readable training log."""

        append_log_message(message, self.training_log_path, display)

    def _initialize_optimization(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Materialize lazy parameters and construct optimizer/scheduler."""

        if self.optimizer is not None:
            return

        if self._has_uninitialized_parameters():
            try:
                first_batch = next(iter(train_loader))
            except StopIteration:
                raise ValueError("train_loader must contain at least one batch")
            first_batch = self._move_batch(first_batch)
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                self.model(first_batch, **dict(model_kwargs or {}))
            self.model.train(was_training)

        # Loss modules may later contain trainable nuisance parameters, such
        # as latent chemical-potential offsets, so optimize them with the model.
        parameters = list(self.model.parameters()) + list(self.loss.parameters())
        parameters.extend(self.stream_losses.parameters())
        unique_parameters = []
        seen = set()
        for parameter in parameters:
            if id(parameter) not in seen:
                unique_parameters.append(parameter)
                seen.add(id(parameter))
        optimizer_parameters, self.feature_parameter_names = (
            build_optimizer_parameters(
                self.model,
                unique_parameters,
                self.optimizer_args,
                self.feature_learning_rate_multiplier,
            )
        )
        self.optimizer = self.optimizer_cls(
            optimizer_parameters,
            **self.optimizer_args,
        )
        if self.scheduler_cls is not None:
            self.scheduler = self.scheduler_cls(
                self.optimizer,
                **scheduler_arguments_for_groups(
                    self.scheduler_cls,
                    self.scheduler_args,
                    self.optimizer,
                    self.feature_learning_rate_multiplier,
                ),
            )
        if self.learning_rate_warmup_steps:
            self._warmup_target_learning_rates = tuple(
                float(group["lr"]) for group in self.optimizer.param_groups
            )

    @staticmethod
    def _validate_streams(
        streams: Sequence[TrainingStream],
        require_epoch_driver: bool = False,
    ) -> Tuple[TrainingStream, ...]:
        """Return a validated stream tuple without changing trainer state."""

        streams = tuple(streams)
        if not streams:
            raise ValueError("streams must contain at least one stream")
        if any(not isinstance(stream, TrainingStream) for stream in streams):
            raise TypeError("streams must contain only TrainingStream objects")
        names = [stream.name for stream in streams]
        if len(set(names)) != len(names):
            raise ValueError("stream names must be unique")
        if require_epoch_driver and all(stream.cycle for stream in streams):
            raise ValueError(
                "at least one training stream must have cycle=False"
            )
        return streams

    def _configure_streams(
        self,
        streams: Sequence[TrainingStream],
    ) -> Tuple[TrainingStream, ...]:
        """Validate and register a stable set of training streams."""

        streams = self._validate_streams(
            streams,
            require_epoch_driver=True,
        )
        if not self._streams and self.optimizer is not None:
            raise ValueError(
                "cannot configure streams after optimization has started"
            )
        if self._streams and (
            len(streams) != len(self._streams)
            or any(
                supplied is not configured
                for supplied, configured in zip(streams, self._streams)
            )
        ):
            raise ValueError("configured streams cannot change once registered")
        if not self._streams:
            self.stream_losses = nn.ModuleDict(
                {stream.name: stream.loss for stream in streams}
            )
            self.stream_metrics = nn.ModuleDict(
                {
                    stream.name: nn.ModuleList(stream.metrics)
                    for stream in streams
                }
            )
            self._streams = streams
            self._stream_loader_generators = {
                stream.name: getattr(stream.train_loader, "generator", None)
                for stream in streams
            }
            self.to(self.device)
        return self._streams

    @staticmethod
    def _stream_weight(
        specification: Union[float, Callable[[int], float]],
        epoch: int,
        name: str,
    ) -> float:
        value = specification(epoch) if callable(specification) else specification
        return nonnegative_scalar(value, "{} stream weight".format(name))

    def _resolve_stream_weights(
        self,
        streams: Sequence[TrainingStream],
        specifications: Mapping[
            str,
            Union[float, Callable[[int], float]],
        ],
        epoch: int,
    ) -> Dict[str, float]:
        """Evaluate all stream-weight specifications for one epoch."""

        return {
            stream.name: self._stream_weight(
                specifications.get(stream.name, 1.0),
                epoch,
                stream.name,
            )
            for stream in streams
        }

    def _run_training_streams(
        self,
        streams: Sequence[TrainingStream],
        weights: Mapping[str, float],
    ) -> Dict[str, Tuple[Dict[str, float], Dict[str, Dict[str, float]]]]:
        """Run joint updates, cycling opted-in streams to the epoch length."""

        if not any(weight > 0.0 for weight in weights.values()):
            raise ValueError("at least one training stream must have positive weight")
        self.model.train(True)
        for stream in streams:
            stream.loss.train(True)
        iterators = {stream.name: iter(stream.train_loader) for stream in streams}
        active = {stream.name for stream in streams if not stream.cycle}
        loss_sums = {stream.name: {} for stream in streams}
        field_counts = {stream.name: 0 for stream in streams}

        while active:
            groups = {}
            # Non-cycling streams define the epoch length. Gather them first so
            # cycling streams never create an extra update after the last
            # ordinary stream is exhausted.
            for stream in streams:
                if stream.cycle or stream.name not in active:
                    continue
                batches = []
                for _ in range(stream.batches_per_step):
                    try:
                        batches.append(next(iterators[stream.name]))
                    except StopIteration:
                        active.remove(stream.name)
                        break
                if batches:
                    groups[stream.name] = batches
            if not groups:
                break

            for stream in streams:
                if not stream.cycle:
                    continue
                batches = []
                for _ in range(stream.batches_per_step):
                    try:
                        batch = next(iterators[stream.name])
                    except StopIteration:
                        iterators[stream.name] = iter(stream.train_loader)
                        try:
                            batch = next(iterators[stream.name])
                        except StopIteration as error:
                            raise ValueError(
                                "cycling train loader for stream '{}' is empty".format(
                                    stream.name
                                )
                            ) from error
                    batches.append(batch)
                groups[stream.name] = batches

            self.optimizer.zero_grad(set_to_none=True)
            for stream in streams:
                batches = groups.get(stream.name, ())
                for batch in batches:
                    batch, outputs, losses, details = self._forward_loss(
                        batch,
                        stream.loss,
                        stream.model_kwargs,
                    )
                    self._update_metrics(
                        stream.metrics,
                        "train",
                        outputs,
                        batch,
                        details,
                    )
                    (
                        weights[stream.name]
                        * losses["total"]
                        / len(batches)
                    ).backward()
                    n_fields = int(batch["rho"].shape[0])
                    for loss_name, value in losses.items():
                        loss_sums[stream.name][loss_name] = (
                            loss_sums[stream.name].get(loss_name, 0.0)
                            + value.detach().item() * n_fields
                        )
                    field_counts[stream.name] += n_fields
            self._apply_learning_rate_warmup()
            self.optimizer.step()
            self._after_optimizer_step()

        results = {}
        for stream in streams:
            n_fields = field_counts[stream.name]
            if n_fields == 0:
                raise ValueError(
                    "train_loader for stream '{}' is empty".format(stream.name)
                )
            losses = {
                name: value / n_fields
                for name, value in loss_sums[stream.name].items()
            }
            metrics = self._retrieve_metrics(stream.metrics, "train")
            results[stream.name] = (losses, metrics)
        return results

    def _evaluate_stream(
        self,
        stream: TrainingStream,
        subset: str,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Evaluate one named stream without combining its statistics."""

        if subset == "train":
            loader = stream.train_loader
        elif subset == "valid":
            loader = stream.valid_loader
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        self.model.eval()
        stream.loss.to(self.device).eval()
        for metric in stream.metrics:
            metric.to(self.device)
        loss_sums = {}
        n_fields = 0
        with self._evaluation_parameters(), torch.no_grad():
            for batch in loader:
                batch, outputs, losses, details = self._forward_loss(
                    batch,
                    stream.loss,
                    stream.model_kwargs,
                )
                self._update_metrics(
                    stream.metrics,
                    subset,
                    outputs,
                    batch,
                    details,
                )
                n_batch_fields = int(batch["rho"].shape[0])
                for name, value in losses.items():
                    loss_sums[name] = loss_sums.get(name, 0.0) + (
                        value.detach().item() * n_batch_fields
                    )
                n_fields += n_batch_fields
        if n_fields == 0:
            raise ValueError(
                "{} loader for stream '{}' is empty".format(subset, stream.name)
            )
        losses = {name: value / n_fields for name, value in loss_sums.items()}
        return losses, self._retrieve_metrics(stream.metrics, subset)

    @staticmethod
    def _update_metrics(
        metrics: Sequence[nn.Module],
        subset: str,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        details: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        for metric in metrics:
            if getattr(metric, "requires_loss_details", False):
                if metric.name not in details:
                    raise KeyError(
                        "metric '{}' has no matching loss details".format(
                            metric.name
                        )
                    )
                metric.update_metrics(subset, details[metric.name])
            else:
                metric.update_metrics(subset, outputs, batch)

    @staticmethod
    def _retrieve_metrics(
        metrics: Sequence[nn.Module],
        subset: str,
    ) -> Dict[str, Dict[str, float]]:
        return {
            metric.name: {
                name: value.item()
                for name, value in metric.retrieve_metrics(subset).items()
            }
            for metric in metrics
        }

    def _forward_loss(
        self,
        batch: Dict[str, torch.Tensor],
        loss: Loss,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        """Move one batch, run the model, and evaluate one loss module."""

        batch = self._move_batch(batch)
        outputs = self.model(batch, **(model_kwargs or {}))
        losses, details = loss.evaluate(outputs, batch, model=self.model)
        return batch, outputs, losses, details

    @staticmethod
    def _flatten_stream_results(
        results: Mapping[
            str,
            Tuple[Dict[str, float], Dict[str, Dict[str, float]]],
        ],
        weights: Mapping[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        losses = {}
        metrics = {}
        total = 0.0
        for stream_name, (stream_losses, stream_metrics) in results.items():
            total += weights[stream_name] * stream_losses["total"]
            for name, value in stream_losses.items():
                losses["{}/{}".format(stream_name, name)] = value
            for name, values in stream_metrics.items():
                metrics["{}/{}".format(stream_name, name)] = values
        losses["total"] = total
        return losses, metrics

    def _step_scheduler(self, valid_loss: float) -> None:
        """Advance an optional scheduler using its expected calling style."""

        if self.scheduler is None:
            return
        if self.optimizer_steps < self.learning_rate_warmup_steps:
            return
        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            self.scheduler.step(valid_loss)
        else:
            self.scheduler.step()

    def _apply_learning_rate_warmup(self) -> None:
        """Set the learning rate for the next optimizer update."""

        if self.optimizer_steps >= self.learning_rate_warmup_steps:
            return
        factor = (self.optimizer_steps + 1) / self.learning_rate_warmup_steps
        for group, target in zip(
            self.optimizer.param_groups,
            self._warmup_target_learning_rates,
        ):
            group["lr"] = factor * target

    def _evaluation_parameters(self):
        """Return a context using EMA parameters when they are available."""

        if self._ema is None:
            return nullcontext()
        return self._ema.average_parameters(self.model)

    def _after_optimizer_step(self) -> None:
        """Advance optimizer-step bookkeeping and an optional model EMA."""

        self.optimizer_steps += 1
        if self._ema is not None and self.optimizer_steps > self.ema_start_step:
            self._ema.update(self.model)

    def evaluation_model_state_dict(self):
        """Return the model state used for validation and model selection."""

        if self._ema is None:
            return OrderedDict(
                (name, value.detach().clone())
                for name, value in self.model.state_dict().items()
            )
        return self._ema.evaluation_state_dict(self.model)

    def _optimization_record(self) -> Dict[str, Any]:
        """Return learning-rate and optional EMA progress metadata."""

        record = learning_rate_record(
            self.optimizer,
            self.feature_learning_rate_multiplier,
        )
        if self._ema is not None or self.learning_rate_warmup_steps:
            record["optimizer_steps"] = self.optimizer_steps
        if self._ema is not None:
            record.update(
                {
                    "ema_updates": self._ema.num_updates,
                    "validation_parameter_source": (
                        "ema" if self._ema.initialized else "raw"
                    ),
                }
            )
        return record

    def _finish_epoch(
        self,
        record: Dict[str, Any],
        verbose: bool,
        print_interval: int,
    ) -> bool:
        """Record one epoch and return whether early stopping was reached."""

        self.history.append(record)
        self._step_scheduler(record["valid_losses"]["total"])
        self._write_epoch_checkpoints(record)
        self._write_history_csv()
        if verbose and record["epoch"] % print_interval == 0:
            self.log_message(format_record(record))
        if not self._early_stopping_reached():
            return False
        self.log_message(
            "Early stopping at epoch {} after {} epochs without "
            "validation-loss improvement.".format(
                record["epoch"],
                self.epochs_without_improvement,
            ),
            display=verbose,
        )
        return True

    def _stop_before_training(self, verbose: bool) -> bool:
        """Report an already exhausted patience counter before another fit."""

        if not self._early_stopping_reached():
            return False
        self.log_message(
            "Early stopping already reached after {} epochs without "
            "validation-loss improvement.".format(
                self.epochs_without_improvement
            ),
            display=verbose,
        )
        return True

    def _write_epoch_checkpoints(self, record: Dict[str, Any]) -> None:
        """Write configured periodic, best, and last checkpoints."""

        epoch = record["epoch"]
        valid_loss = record["valid_losses"]["total"]
        improved = valid_loss < self.best_valid_loss
        if improved:
            self.best_valid_loss = valid_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        if self.checkpoint_dir is None:
            return

        is_best = self.save_best and improved
        if epoch % self.checkpoint_interval == 0:
            self.save_checkpoint(
                self.checkpoint_dir
                / "checkpoint_epoch_{:04d}.pt".format(epoch),
                record,
            )
        if is_best:
            self.save_checkpoint(self.checkpoint_dir / "best.pt", record)
        self.save_checkpoint(self.checkpoint_dir / "last.pt", record)

    def _early_stopping_reached(self) -> bool:
        """Return whether the configured validation patience is exhausted."""

        return (
            self.early_stopping_patience is not None
            and self.epochs_without_improvement
            >= self.early_stopping_patience
        )

    def _move_optimizer_state_to_device(self) -> None:
        """Move restored optimizer tensors onto the configured device."""

        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _write_history_csv(self) -> None:
        """Atomically write one flattened numerical row per completed epoch."""

        if self.history_path is None or not self.history:
            return

        write_history_csv(self.history, self.history_path)

    def _move_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Move batch tensors while reusing identity-shared grid geometry."""

        shared_geometry = getattr(batch, "_shared_geometry", {})
        moved = {}
        for key, value in batch.items():
            source = shared_geometry.get(key)
            if source is None:
                moved[key] = (
                    value.to(self.device) if torch.is_tensor(value) else value
                )
                continue

            identity = id(source)
            cached = self._grid_geometry_device_cache.pop(identity, None)
            if (
                cached is None
                or cached[0] is not source
                or cached[1] != source._version
            ):
                cached = None
                while (
                    len(self._grid_geometry_device_cache)
                    >= _MAX_CACHED_GRID_GEOMETRIES
                ):
                    self._grid_geometry_device_cache.popitem(last=False)
                cached = (source, source._version, source.to(self.device))
            self._grid_geometry_device_cache[identity] = cached
            moved[key] = cached[2].unsqueeze(0).expand(value.shape)
        return moved

    def _has_uninitialized_parameters(self) -> bool:
        """Return whether model or loss contains lazy parameters."""

        return any(
            isinstance(parameter, UninitializedParameter)
            for parameter in list(self.model.parameters())
            + list(self.loss.parameters())
        )

    def _next_epoch(self) -> int:
        """Return the next positive training epoch after optional epoch 0."""

        return self.history[-1]["epoch"] + 1 if self.history else 1

    _format_record = staticmethod(format_record)
