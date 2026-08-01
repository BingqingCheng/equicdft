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

from .loss import Loss
from .metrics import Metrics, format_metric_value, metric_label


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

        self.checkpoint_dir = None
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir).expanduser()
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = None
        self.scheduler = None
        self.history = []
        self.best_valid_loss = math.inf
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

        self._initialize_optimization(train_loader)
        start_epoch = len(self.history) + 1

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
            if verbose and epoch % print_interval == 0:
                print(self._format_record(record))

        return self.history

    def save_checkpoint(
        self,
        path: Union[str, Path],
        record: Dict[str, Any],
    ) -> None:
        """Write model, optimization, scheduler, and history state."""

        checkpoint = {
            "epoch": record["epoch"],
            "model_state_dict": self.model.state_dict(),
            "loss_state_dict": self.loss.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                None if self.scheduler is None else self.scheduler.state_dict()
            ),
            "record": record,
            "history": self.history,
        }
        torch.save(checkpoint, str(Path(path).expanduser()))

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
                loss_values = self.loss(outputs, batch, model=self.model)
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

        if self.checkpoint_dir is None:
            return

        epoch = record["epoch"]
        valid_loss = record["valid_losses"]["total"]
        if epoch % self.checkpoint_interval == 0:
            self.save_checkpoint(
                self.checkpoint_dir
                / "checkpoint_epoch_{:04d}.pt".format(epoch),
                record,
            )
        if self.save_best and valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.save_checkpoint(self.checkpoint_dir / "best.pt", record)
        self.save_checkpoint(self.checkpoint_dir / "last.pt", record)

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

    @staticmethod
    def _format_record(record: Dict[str, Any]) -> str:
        """Format one readable, terminal-safe multi-line epoch summary."""

        lines = [
            "Epoch {:4d} | learning rate {:.3e}".format(
                record["epoch"],
                record["learning_rate"],
            )
        ]

        loss_names = ["total"] + [
            name
            for name in record["train_losses"]
            if name != "total"
        ]
        loss_rows = []
        for subset in ("train", "valid"):
            values = record["{}_losses".format(subset)]
            loss_rows.append(
                [subset]
                + ["{:.6e}".format(values[name]) for name in loss_names]
            )
        lines.extend(
            [
                "",
                Trainer._format_table(
                    "Losses",
                    ["subset"] + loss_names,
                    loss_rows,
                ),
            ]
        )

        for collection_name, train_values in record["train_metrics"].items():
            metric_names = list(train_values)
            metric_rows = []
            for subset in ("train", "valid"):
                values = record["{}_metrics".format(subset)][
                    collection_name
                ]
                metric_rows.append(
                    [subset]
                    + [
                        format_metric_value(name, values[name])
                        for name in metric_names
                    ]
                )
            lines.extend(
                [
                    "",
                    Trainer._format_table(
                        "{} metrics".format(collection_name),
                        ["subset"]
                        + [
                            metric_label(name)
                            for name in metric_names
                        ],
                        metric_rows,
                    ),
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _format_table(
        title: str,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
    ) -> str:
        """Format a small aligned ASCII table."""

        widths = [len(header) for header in headers]
        for row in rows:
            widths = [
                max(width, len(value))
                for width, value in zip(widths, row)
            ]

        def format_row(row: Sequence[str]) -> str:
            cells = [row[0].ljust(widths[0])]
            cells.extend(
                value.rjust(width)
                for value, width in zip(row[1:], widths[1:])
            )
            return "  " + "  ".join(cells)

        separator = "  " + "  ".join("-" * width for width in widths)
        table_lines = [title, format_row(headers), separator]
        table_lines.extend(format_row(row) for row in rows)
        return "\n".join(table_lines)
