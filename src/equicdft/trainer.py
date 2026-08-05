"""Training orchestration for grid density-functional models."""

import math
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from ._trainer_io import (
    append_log_message,
    atomic_torch_save,
    format_record,
    write_history_csv,
)
from .loss import Loss
from .metrics import Metrics


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
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Loss,
        metrics: Sequence[Metrics] = (),
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
    ) -> None:
        super().__init__()

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(loss, Loss):
            raise TypeError("loss must be a equicdft.Loss")
        if (
            not isinstance(checkpoint_interval, int)
            or checkpoint_interval <= 0
        ):
            raise ValueError("checkpoint_interval must be a positive integer")
        if not isinstance(save_best, bool):
            raise TypeError("save_best must be a boolean")
        if early_stopping_patience is not None:
            if (
                isinstance(early_stopping_patience, bool)
                or not isinstance(early_stopping_patience, int)
                or early_stopping_patience <= 0
            ):
                raise ValueError(
                    "early_stopping_patience must be a positive integer or None"
                )

        metrics = list(metrics)
        if any(not isinstance(metric, Metrics) for metric in metrics):
            raise TypeError("metrics must contain only equicdft.Metrics")
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

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(print_interval, int) or print_interval <= 0:
            raise ValueError("print_interval must be a positive integer")

        self._train_loader_generator = getattr(train_loader, "generator", None)
        self._initialize_optimization(train_loader)
        start_epoch = len(self.history) + 1

        if self._early_stopping_reached():
            self.log_message(
                "Early stopping already reached after {} epochs without "
                "validation-loss improvement.".format(
                    self.epochs_without_improvement
                ),
                display=verbose,
            )
            return self.history

        for epoch in range(start_epoch, start_epoch + epochs):
            train_losses, train_metrics = self._run_loader(
                train_loader,
                subset="train",
                training=True,
            )
            valid_losses, valid_metrics = self._run_loader(
                valid_loader,
                subset="valid",
                training=False,
            )

            learning_rate = self.optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_losses": train_losses,
                "valid_losses": valid_losses,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
            }
            self.history.append(record)

            self._step_scheduler(valid_losses["total"])
            self._write_epoch_checkpoints(record)
            self._write_history_csv()
            if verbose and epoch % print_interval == 0:
                self.log_message(format_record(record))
            if self._early_stopping_reached():
                self.log_message(
                    "Early stopping at epoch {} after {} epochs without "
                    "validation-loss improvement.".format(
                        epoch,
                        self.epochs_without_improvement,
                    ),
                    display=verbose,
                )
                break

        return self.history

    def save_checkpoint(
        self,
        path: Union[str, Path],
        record: Dict[str, Any],
    ) -> None:
        """Write model, optimization, scheduler, and history state."""

        checkpoint = {
            "checkpoint_version": 1,
            "epoch": record["epoch"],
            "model_state_dict": self.model.state_dict(),
            "loss_state_dict": self.loss.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
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
        }
        atomic_torch_save(checkpoint, Path(path).expanduser())

    def load_checkpoint(
        self,
        path: Union[str, Path],
        train_loader: Iterable[Dict[str, torch.Tensor]],
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
        self._initialize_optimization(train_loader)
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

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.loss.load_state_dict(checkpoint["loss_state_dict"], strict=True)
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
        if not isinstance(history, list) or len(history) != epoch:
            raise ValueError(
                "checkpoint history length must equal its completed epoch"
            )
        self.history = history

        if "best_valid_loss" in checkpoint:
            self.best_valid_loss = float(checkpoint["best_valid_loss"])
        else:
            self.best_valid_loss = min(
                record["valid_losses"]["total"] for record in history
            )
        if "epochs_without_improvement" in checkpoint:
            self.epochs_without_improvement = int(
                checkpoint["epochs_without_improvement"]
            )
        else:
            best_loss = math.inf
            epochs_without_improvement = 0
            for record in history:
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

        for metric in self.metrics:
            for subset in metric.logs:
                metric.clear_metrics(subset)
        self._write_history_csv()
        return epoch

    def log_message(self, message: str, display: bool = True) -> None:
        """Print a message and append it to the human-readable training log."""

        append_log_message(message, self.training_log_path, display)

    def _initialize_optimization(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
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
                self.model(first_batch)
            self.model.train(was_training)

        # Loss modules may later contain trainable nuisance parameters, such
        # as latent chemical-potential offsets, so optimize them with the model.
        parameters = list(self.model.parameters()) + list(self.loss.parameters())
        self.optimizer = self.optimizer_cls(parameters, **self.optimizer_args)
        if self.scheduler_cls is not None:
            self.scheduler = self.scheduler_cls(
                self.optimizer,
                **self.scheduler_args,
            )

    def _run_loader(
        self,
        loader: Iterable[Dict[str, torch.Tensor]],
        subset: str,
        training: bool,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Run one loader and return field-averaged losses and metrics."""

        self.model.train(training)
        self.loss.train(training)
        loss_sums = {}
        n_fields = 0

        gradient_context = torch.enable_grad() if training else torch.no_grad()
        with gradient_context:
            for batch in loader:
                batch = self._move_batch(batch)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(batch)
                loss_values = self.loss(outputs, batch)
                for metric in self.metrics:
                    metric.update_metrics(subset, outputs, batch)

                if training:
                    loss_values["total"].backward()
                    self.optimizer.step()

                n_batch_fields = int(batch["rho"].shape[0])
                for name, value in loss_values.items():
                    loss_sums[name] = loss_sums.get(name, 0.0) + (
                        value.detach().item() * n_batch_fields
                    )
                n_fields += n_batch_fields

        if n_fields == 0:
            raise ValueError(
                "{}_loader must contain at least one batch".format(subset)
            )

        losses = {
            name: value / n_fields
            for name, value in loss_sums.items()
        }
        metric_values = {
            metric.name: {
                name: value.item()
                for name, value in metric.retrieve_metrics(subset).items()
            }
            for metric in self.metrics
        }
        return losses, metric_values

    def _step_scheduler(self, valid_loss: float) -> None:
        """Advance an optional scheduler using its expected calling style."""

        if self.scheduler is None:
            return
        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            self.scheduler.step(valid_loss)
        else:
            self.scheduler.step()

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
        """Move every tensor value in a grid-data dictionary to the device."""

        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _has_uninitialized_parameters(self) -> bool:
        """Return whether model or loss contains lazy parameters."""

        return any(
            isinstance(parameter, UninitializedParameter)
            for parameter in list(self.model.parameters())
            + list(self.loss.parameters())
        )

    _format_record = staticmethod(format_record)
