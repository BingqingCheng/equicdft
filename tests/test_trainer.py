import csv
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from equicdft import (
    FourierResponseLoss,
    FourierResponseMetrics,
    FourierStabilityLoss,
    Loss,
    Metrics,
    TensorLoss,
    Trainer,
    TrainingStream,
    make_dataloaders,
)
from equicdft._grid import voxel_volume


class _LinearDictionaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        return {"prediction": self.weight * batch["x"]}


class _RecordingLinearDictionaryModel(_LinearDictionaryModel):
    def __init__(self):
        super().__init__()
        self.training_values = []

    def forward(self, batch):
        if self.training:
            self.training_values.extend(batch["x"].flatten().tolist())
        return super().forward(batch)


class _QuadraticDictionaryFunctional(nn.Module):
    """Small functional for exercising model-dependent trainer losses."""

    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(-4.0))

    def forward(self, batch, compute_c1=None):
        rho = batch["rho"]
        volume_element = voxel_volume(batch["grid_spacing"].to(rho))
        return {
            "beta_F_exc": (
                0.5
                * self.coefficient
                * volume_element
                * torch.sum(rho.square(), dim=(-2, -1))
            )
        }


def _dataset(values):
    return [
        {
            "x": torch.tensor([float(value)]),
            "target": torch.tensor([2.0 * float(value)]),
            "rho": torch.ones(1, 1),
        }
        for value in values
    ]


def _functional_dataset(n_fields=2):
    n_grid = 8
    return [
        {
            "rho": torch.full((n_grid, 1), 0.5),
            "temperature": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
            "grid_size": torch.tensor([n_grid, 1, 1]),
            "grid_positions": torch.tensor(
                [[index, 0, 0] for index in range(n_grid)]
            ),
        }
        for _ in range(n_fields)
    ]


class TestTrainer(unittest.TestCase):
    def test_shared_grid_geometry_device_copy_is_cached_but_not_serialized(self):
        geometry = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        dataset = _dataset([1, 2])
        for frame in dataset:
            frame["local_density_index"] = geometry
        loader = make_dataloaders(
            dataset,
            valid_dataset=dataset,
            batch_size=1,
            reuse_grid_geometry=True,
        )["valid"]
        trainer = self._make_trainer()

        batches = list(loader)
        first = trainer._move_batch(batches[0])
        second = trainer._move_batch(batches[1])

        self.assertEqual(len(trainer._grid_geometry_device_cache), 1)
        self.assertEqual(
            first["local_density_index"].data_ptr(),
            second["local_density_index"].data_ptr(),
        )
        self.assertFalse(
            any("geometry" in key for key in trainer.state_dict())
        )

        cached_version = next(
            iter(trainer._grid_geometry_device_cache.values())
        )[1]
        geometry.add_(1)
        trainer._move_batch(batches[0])
        refreshed_version = next(
            iter(trainer._grid_geometry_device_cache.values())
        )[1]
        self.assertEqual(refreshed_version, cached_version + 1)

    def test_shared_grid_geometry_device_cache_is_bounded(self):
        trainer = self._make_trainer()
        for offset in range(3):
            geometry = torch.tensor(
                [[0, 1], [1, 0]],
                dtype=torch.long,
            ) + offset
            dataset = _dataset([1, 2])
            for frame in dataset:
                frame["local_density_index"] = geometry
            batch = next(
                iter(
                    make_dataloaders(
                        dataset,
                        valid_dataset=dataset,
                        batch_size=2,
                        reuse_grid_geometry=True,
                    )["valid"]
                )
            )
            trainer._move_batch(batch)

        self.assertEqual(len(trainer._grid_geometry_device_cache), 2)

    def _make_trainer(
        self,
        checkpoint_dir=None,
        scheduler=False,
        log_dir=None,
        early_stopping_patience=None,
    ):
        loss = Loss(
            [
                TensorLoss(
                    "target",
                    "prediction",
                    "target",
                    loss_fn=nn.MSELoss(),
                )
            ]
        )
        metrics = Metrics(
            "target",
            prediction_key="prediction",
            metric_keys=("mae", "rmse", "pearson_r"),
        )
        scheduler_cls = torch.optim.lr_scheduler.StepLR if scheduler else None
        scheduler_args = {"step_size": 1, "gamma": 0.5} if scheduler else None
        return Trainer(
            model=_LinearDictionaryModel(),
            loss=loss,
            metrics=[metrics],
            optimizer_cls=torch.optim.SGD,
            optimizer_args={"lr": 0.1},
            scheduler_cls=scheduler_cls,
            scheduler_args=scheduler_args,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=1,
            save_best=True,
            early_stopping_patience=early_stopping_patience,
            log_dir=log_dir,
        )

    def test_fit_updates_model_and_returns_history(self):
        trainer = self._make_trainer()
        initial_weight = trainer.model.weight.detach().clone()
        train_loader = DataLoader(_dataset([1, 2, 3, 4]), batch_size=2)
        valid_loader = DataLoader(_dataset([5, 6]), batch_size=2)

        history = trainer.fit(
            train_loader,
            valid_loader,
            epochs=2,
            verbose=False,
        )

        self.assertEqual(len(history), 2)
        self.assertFalse(torch.equal(trainer.model.weight, initial_weight))
        self.assertEqual(
            set(history[0]),
            {
                "epoch",
                "learning_rate",
                "train_losses",
                "valid_losses",
                "train_metrics",
                "valid_metrics",
            },
        )
        self.assertIn("total", history[0]["train_losses"])
        self.assertIn("mae", history[0]["valid_metrics"]["target"])
        self.assertEqual(trainer.metrics[0].logs["train"]["count"], 0)
        self.assertEqual(trainer.metrics[0].logs["valid"]["count"], 0)

    def test_model_dependent_loss_runs_through_trainer(self):
        model = _QuadraticDictionaryFunctional()
        trainer = Trainer(
            model=model,
            loss=Loss(
                [
                    FourierStabilityLoss(
                        ((1, 0, 0),),
                        training_only=False,
                    )
                ]
            ),
            optimizer_cls=torch.optim.SGD,
            optimizer_args={"lr": 0.01},
        )
        loader = DataLoader(_functional_dataset(), batch_size=2)
        initial_coefficient = model.coefficient.detach().clone()

        history = trainer.fit(loader, loader, epochs=1, verbose=False)

        self.assertIn(
            "fourier_stability",
            history[0]["train_losses"],
        )
        self.assertIn(
            "fourier_stability",
            history[0]["valid_losses"],
        )
        self.assertGreater(
            history[0]["valid_losses"]["fourier_stability"],
            0.0,
        )
        self.assertFalse(torch.equal(model.coefficient, initial_coefficient))

    def test_fourier_response_loss_runs_through_trainer(self):
        dataset = _functional_dataset()
        for frame in dataset:
            frame["fourier_modes"] = torch.tensor([[1, 0, 0]])
            frame["fourier_curvature"] = torch.ones((1, 1))
        model = _QuadraticDictionaryFunctional()
        trainer = Trainer(
            model=model,
            loss=Loss(
                [
                    FourierResponseLoss(
                        directions=((1.0,),),
                        relative_amplitude=1.0e-3,
                    )
                ]
            ),
            optimizer_cls=torch.optim.SGD,
            optimizer_args={"lr": 0.01},
        )
        loader = DataLoader(dataset, batch_size=2)
        initial_coefficient = model.coefficient.detach().clone()

        history = trainer.fit(loader, loader, epochs=1, verbose=False)

        self.assertIn("fourier_response", history[0]["train_losses"])
        self.assertIn("fourier_response", history[0]["valid_losses"])
        self.assertGreater(
            history[0]["valid_losses"]["fourier_response"],
            0.0,
        )
        self.assertFalse(torch.equal(model.coefficient, initial_coefficient))

    def test_multiple_streams_keep_losses_and_metrics_separate(self):
        model = _LinearDictionaryModel()
        first_loss = Loss(
            [TensorLoss("target", "prediction", "target")]
        )
        second_loss = Loss(
            [TensorLoss("target", "prediction", "target")]
        )
        streams = [
            TrainingStream(
                name="field",
                train_loader=DataLoader(_dataset([1, 2, 3, 4]), batch_size=1),
                valid_loader=DataLoader(_dataset([5, 6]), batch_size=1),
                loss=first_loss,
                metrics=(
                    Metrics(
                        "target",
                        prediction_key="prediction",
                        metric_keys=("rmse",),
                        subsets=("train", "valid"),
                    ),
                ),
                batches_per_step=2,
            ),
            TrainingStream(
                name="response",
                train_loader=DataLoader(_dataset([7, 8]), batch_size=1),
                valid_loader=DataLoader(_dataset([9]), batch_size=1),
                loss=second_loss,
                metrics=(
                    Metrics(
                        "target",
                        prediction_key="prediction",
                        metric_keys=("rmse",),
                        subsets=("train", "valid"),
                    ),
                ),
            ),
        ]
        trainer = Trainer(
            model=model,
            loss=first_loss,
            optimizer_cls=torch.optim.SGD,
            optimizer_args={"lr": 0.001},
        )

        history = trainer.fit_streams(
            streams,
            epochs=2,
            stream_weights={"response": lambda epoch: 0.25 * epoch},
            verbose=False,
        )

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["stream_weights"]["field"], 1.0)
        self.assertEqual(history[1]["stream_weights"]["response"], 0.5)
        self.assertIn("field/target", history[0]["train_losses"])
        self.assertIn("response/target", history[0]["valid_losses"])
        self.assertIn("field/target", history[0]["train_metrics"])
        self.assertIn("response/target", history[0]["valid_metrics"])
        self.assertFalse(torch.equal(model.weight, torch.tensor(0.0)))

    def test_shorter_stream_can_cycle_to_noncycling_epoch_length(self):
        model = _RecordingLinearDictionaryModel()
        loss = Loss([TensorLoss("target", "prediction", "target")])
        streams = (
            TrainingStream(
                "field",
                DataLoader(_dataset([1, 2, 3, 4, 5]), batch_size=1),
                DataLoader(_dataset([6]), batch_size=1),
                loss,
                batches_per_step=2,
            ),
            TrainingStream(
                "response",
                DataLoader(_dataset([10, 11]), batch_size=1),
                DataLoader(_dataset([12]), batch_size=1),
                loss,
                cycle=True,
            ),
        )
        trainer = Trainer(
            model,
            loss,
            optimizer_cls=torch.optim.SGD,
            optimizer_args={"lr": 0.001},
        )

        trainer.fit_streams(streams, epochs=1, verbose=False)

        self.assertEqual(
            model.training_values,
            [1.0, 2.0, 10.0, 3.0, 4.0, 11.0, 5.0, 10.0],
        )

    def test_at_least_one_training_stream_must_not_cycle(self):
        loss = Loss([TensorLoss("target", "prediction", "target")])
        loader = DataLoader(_dataset([1]), batch_size=1)
        stream = TrainingStream(
            "response",
            loader,
            loader,
            loss,
            cycle=True,
        )
        trainer = Trainer(_LinearDictionaryModel(), loss)

        with self.assertRaisesRegex(ValueError, "cycle=False"):
            trainer.fit_streams((stream,), epochs=1, verbose=False)

    def test_multiple_stream_checkpoint_can_resume(self):
        def build(checkpoint_dir):
            model = _LinearDictionaryModel()
            field_loss = Loss(
                [TensorLoss("target", "prediction", "target")]
            )
            response_loss = Loss(
                [TensorLoss("target", "prediction", "target")]
            )
            streams = (
                TrainingStream(
                    "field",
                    DataLoader(_dataset([1, 2]), batch_size=1),
                    DataLoader(_dataset([3]), batch_size=1),
                    field_loss,
                ),
                TrainingStream(
                    "response",
                    DataLoader(_dataset([4]), batch_size=1),
                    DataLoader(_dataset([5]), batch_size=1),
                    response_loss,
                ),
            )
            return (
                Trainer(
                    model,
                    field_loss,
                    optimizer_cls=torch.optim.SGD,
                    optimizer_args={"lr": 0.001},
                    checkpoint_dir=checkpoint_dir,
                ),
                streams,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer, streams = build(temporary_directory)
            trainer.fit_streams(streams, epochs=1, verbose=False)

            resumed, resumed_streams = build(temporary_directory)
            completed = resumed.load_stream_checkpoint(
                Path(temporary_directory) / "last.pt",
                resumed_streams,
            )
            resumed.fit_streams(resumed_streams, epochs=1, verbose=False)

        self.assertEqual(completed, 1)
        self.assertEqual([row["epoch"] for row in resumed.history], [1, 2])

    def test_stream_baseline_is_printed_recorded_and_resumable(self):
        def build(directory):
            loss = Loss(
                [TensorLoss("target", "prediction", "target")]
            )
            stream = TrainingStream(
                "field",
                DataLoader(_dataset([1, 2]), batch_size=1),
                DataLoader(_dataset([3]), batch_size=1),
                loss,
                metrics=(
                    Metrics(
                        "target",
                        prediction_key="prediction",
                        metric_keys=("rmse",),
                        subsets=("train", "valid"),
                    ),
                ),
            )
            return (
                Trainer(
                    _LinearDictionaryModel(),
                    loss,
                    optimizer_cls=torch.optim.SGD,
                    optimizer_args={"lr": 0.001},
                    checkpoint_dir=Path(directory) / "checkpoints",
                    log_dir=Path(directory) / "logs",
                ),
                (stream,),
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer, streams = build(temporary_directory)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                history = trainer.fit_streams(
                    streams,
                    epochs=1,
                    stream_weights={"field": lambda epoch: epoch / 10.0},
                    record_initial_validation=True,
                    verbose=True,
                )
            with (Path(temporary_directory) / "logs/history.csv").open(
                newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            checkpoint = torch.load(
                Path(temporary_directory) / "checkpoints/last.pt"
            )

            resumed, resumed_streams = build(temporary_directory)
            completed = resumed.load_stream_checkpoint(
                Path(temporary_directory) / "checkpoints/last.pt",
                resumed_streams,
            )

        baseline = history[0]
        self.assertEqual(baseline["epoch"], 0)
        self.assertEqual(baseline["stream_weights"], {"field": 0.0})
        self.assertEqual(baseline["train_losses"], {})
        self.assertIn("field/target", baseline["valid_metrics"])
        self.assertIn("Untrained validation baseline", output.getvalue())
        self.assertEqual([row["epoch"] for row in rows], ["0", "1"])
        self.assertEqual(rows[0]["train_field_target_rmse"], "")
        self.assertNotEqual(rows[0]["valid_field_target_rmse"], "")
        self.assertEqual(
            [record["epoch"] for record in checkpoint["history"]],
            [0, 1],
        )
        self.assertEqual(completed, 1)
        self.assertEqual(
            [record["epoch"] for record in resumed.history],
            [0, 1],
        )

    def test_response_stream_reuses_loss_details_for_metrics(self):
        dataset = _functional_dataset()
        for frame in dataset:
            frame["fourier_modes"] = torch.tensor([[1, 0, 0]])
            frame["fourier_curvature"] = torch.ones((1, 1))
            frame["fourier_scale"] = torch.ones((1, 1))
        term = FourierResponseLoss(
            directions=((1.0,),),
            scale_key="fourier_scale",
            relative_amplitude=1.0e-3,
        )
        loss = Loss([term])
        loader = DataLoader(dataset, batch_size=2)
        stream = TrainingStream(
            "response",
            loader,
            loader,
            loss,
            metrics=(
                FourierResponseMetrics(
                    ("density",),
                    subsets=("train", "valid"),
                ),
            ),
            model_kwargs={"compute_c1": False},
        )
        trainer = Trainer(_QuadraticDictionaryFunctional(), loss)

        evaluated = trainer.evaluate_streams((stream,))

        response_metrics = evaluated["response"][1]["fourier_response"]
        self.assertIn("K_density_rmse", response_metrics)
        self.assertIn("S_density_relative_rmse_positive_percent", response_metrics)

    def test_evaluate_streams_uses_the_requested_subset_loader(self):
        trainer = self._make_trainer()
        stream = TrainingStream(
            "field",
            DataLoader(_dataset([1]), batch_size=1),
            DataLoader(_dataset([3]), batch_size=1),
            trainer.loss,
        )

        train = trainer.evaluate_streams((stream,), subset="train")
        valid = trainer.evaluate_streams((stream,), subset="valid")

        self.assertEqual(train["field"][0]["total"], 4.0)
        self.assertEqual(valid["field"][0]["total"], 36.0)
        self.assertFalse(trainer.model.training)
        self.assertFalse(stream.loss.training)

    def test_evaluate_streams_is_pure_and_allows_cycling_streams(self):
        trainer = self._make_trainer()
        loader = DataLoader(_dataset([2]), batch_size=1)
        stream = TrainingStream(
            "response",
            loader,
            loader,
            trainer.loss,
            cycle=True,
        )

        evaluated = trainer.evaluate_streams((stream,))

        self.assertEqual(evaluated["response"][0]["total"], 16.0)
        self.assertEqual(trainer._streams, ())
        self.assertEqual(len(trainer.stream_losses), 0)
        self.assertEqual(len(trainer.stream_metrics), 0)

    def test_registered_stream_is_never_silently_replaced(self):
        trainer = self._make_trainer()
        trainer.optimizer_args = {"lr": 0.0}
        original_loader = DataLoader(_dataset([1]), batch_size=1)
        original = TrainingStream(
            "field",
            original_loader,
            original_loader,
            trainer.loss,
        )
        trainer.fit_streams((original,), epochs=1, verbose=False)
        replacement_loader = DataLoader(_dataset([4]), batch_size=1)
        replacement = TrainingStream(
            "field",
            replacement_loader,
            replacement_loader,
            trainer.loss,
        )

        evaluated = trainer.evaluate_streams((replacement,))

        self.assertEqual(evaluated["field"][0]["total"], 64.0)
        self.assertIs(trainer._streams[0], original)
        with self.assertRaisesRegex(ValueError, "registered"):
            trainer.fit_streams((replacement,), epochs=1, verbose=False)

    def test_stream_fit_cannot_start_after_ordinary_optimization(self):
        trainer = self._make_trainer()
        loader = DataLoader(_dataset([1]), batch_size=1)
        trainer.fit(loader, loader, epochs=1, verbose=False)
        stream = TrainingStream("field", loader, loader, trainer.loss)

        with self.assertRaisesRegex(ValueError, "after optimization"):
            trainer.fit_streams((stream,), epochs=1, verbose=False)

    def test_stream_fit_honors_exhausted_early_stopping_before_updates(self):
        trainer = self._make_trainer(early_stopping_patience=1)
        trainer.optimizer_args = {"lr": 0.0}
        loader = DataLoader(_dataset([1]), batch_size=1)
        stream = TrainingStream("field", loader, loader, trainer.loss)

        trainer.fit_streams((stream,), epochs=5, verbose=False)
        completed_epochs = len(trainer.history)
        trainer.fit_streams((stream,), epochs=2, verbose=False)

        self.assertEqual(completed_epochs, 2)
        self.assertEqual(len(trainer.history), completed_epochs)

    def test_optional_scheduler_steps_each_epoch(self):
        trainer = self._make_trainer(scheduler=True)

        trainer.fit(
            DataLoader(_dataset([1, 2]), batch_size=2),
            DataLoader(_dataset([3, 4]), batch_size=2),
            epochs=2,
            verbose=False,
        )

        self.assertAlmostEqual(trainer.history[0]["learning_rate"], 0.1)
        self.assertAlmostEqual(trainer.history[1]["learning_rate"], 0.05)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.025)

    def test_optional_checkpoints_contain_training_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = self._make_trainer(checkpoint_dir=temporary_directory)
            trainer.fit(
                DataLoader(_dataset([1, 2]), batch_size=2),
                DataLoader(_dataset([3, 4]), batch_size=2),
                epochs=2,
                verbose=False,
            )

            directory = Path(temporary_directory)
            expected = {
                "checkpoint_epoch_0001.pt",
                "checkpoint_epoch_0002.pt",
                "best.pt",
                "last.pt",
            }
            self.assertTrue(
                expected.issubset(
                    {path.name for path in directory.iterdir()}
                )
            )
            checkpoint = torch.load(directory / "last.pt")

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("loss_state_dict", checkpoint)
        self.assertIn("optimizer_state_dict", checkpoint)
        self.assertIn("scheduler_state_dict", checkpoint)
        self.assertIn("best_valid_loss", checkpoint)
        self.assertIn("epochs_without_improvement", checkpoint)
        self.assertIn("torch_rng_state", checkpoint)
        self.assertIn("train_loader_generator_state", checkpoint)
        self.assertEqual(len(checkpoint["history"]), 2)

    def test_checkpoint_resume_restores_complete_training_state(self):
        train_values = [1, 2, 3, 4]
        valid_loader = DataLoader(_dataset([5, 6]), batch_size=2)

        continuous = self._make_trainer(scheduler=True)
        continuous_loader = DataLoader(
            _dataset(train_values),
            batch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
        )
        continuous.fit(
            continuous_loader,
            valid_loader,
            epochs=4,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            partial = self._make_trainer(
                checkpoint_dir=temporary_directory,
                scheduler=True,
            )
            partial_loader = DataLoader(
                _dataset(train_values),
                batch_size=2,
                shuffle=True,
                generator=torch.Generator().manual_seed(7),
            )
            partial.fit(
                partial_loader,
                valid_loader,
                epochs=2,
                verbose=False,
            )

            resumed = self._make_trainer(
                checkpoint_dir=temporary_directory,
                scheduler=True,
            )
            resumed_loader = DataLoader(
                _dataset(train_values),
                batch_size=2,
                shuffle=True,
                generator=torch.Generator().manual_seed(7),
            )
            completed_epoch = resumed.load_checkpoint(
                Path(temporary_directory) / "last.pt",
                train_loader=resumed_loader,
            )
            resumed.fit(
                resumed_loader,
                valid_loader,
                epochs=2,
                verbose=False,
            )

        self.assertEqual(completed_epoch, 2)
        self.assertEqual(
            [record["epoch"] for record in resumed.history],
            [1, 2, 3, 4],
        )
        self.assertTrue(
            torch.equal(resumed.model.weight, continuous.model.weight)
        )
        self.assertEqual(
            resumed.optimizer.param_groups[0]["lr"],
            continuous.optimizer.param_groups[0]["lr"],
        )
        self.assertEqual(
            resumed.best_valid_loss,
            continuous.best_valid_loss,
        )

    def test_csv_history_and_text_log_are_written(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_dir = Path(temporary_directory) / "logs"
            trainer = self._make_trainer(log_dir=log_dir)
            trainer.log_message("Starting test fit", display=False)
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.fit(
                    DataLoader(_dataset([1, 2]), batch_size=2),
                    DataLoader(_dataset([3, 4]), batch_size=2),
                    epochs=2,
                    print_interval=1,
                )

            with (log_dir / "history.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            training_log = (log_dir / "training.log").read_text()

        self.assertEqual([row["epoch"] for row in rows], ["1", "2"])
        self.assertIn("train_loss_total", rows[0])
        self.assertIn("valid_loss_target", rows[0])
        self.assertIn("train_target_rmse", rows[0])
        self.assertIn("Starting test fit", training_log)
        self.assertIn("Epoch    1", training_log)

    def test_resume_reconstructs_csv_without_duplicate_epochs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory) / "checkpoints"
            log_dir = Path(temporary_directory) / "logs"
            partial = self._make_trainer(
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
            )
            partial.fit(
                DataLoader(_dataset([1, 2]), batch_size=2),
                DataLoader(_dataset([3, 4]), batch_size=2),
                epochs=2,
                verbose=False,
            )

            resumed = self._make_trainer(
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
            )
            train_loader = DataLoader(_dataset([1, 2]), batch_size=2)
            resumed.load_checkpoint(
                checkpoint_dir / "last.pt",
                train_loader=train_loader,
            )
            resumed.fit(
                train_loader,
                DataLoader(_dataset([3, 4]), batch_size=2),
                epochs=1,
                verbose=False,
            )

            with (log_dir / "history.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual([row["epoch"] for row in rows], ["1", "2", "3"])

    def test_configuration_and_empty_loaders_are_validated(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_interval"):
            Trainer(
                _LinearDictionaryModel(),
                Loss([TensorLoss("target", "prediction", "target")]),
                checkpoint_interval=0,
            )
        for invalid_patience in (0, -1):
            with self.subTest(invalid_patience=invalid_patience):
                with self.assertRaises(ValueError):
                    Trainer(
                        _LinearDictionaryModel(),
                        Loss([TensorLoss("target", "prediction", "target")]),
                        early_stopping_patience=invalid_patience,
                    )
        for invalid_patience in (True, 1.5):
            with self.subTest(invalid_patience=invalid_patience):
                with self.assertRaises(TypeError):
                    Trainer(
                        _LinearDictionaryModel(),
                        Loss([TensorLoss("target", "prediction", "target")]),
                        early_stopping_patience=invalid_patience,
                    )

        trainer = self._make_trainer()
        with self.assertRaisesRegex(ValueError, "train_loader"):
            trainer.fit(
                DataLoader([], batch_size=1),
                DataLoader(_dataset([1]), batch_size=1),
                epochs=1,
                verbose=False,
            )

    def test_early_stopping_and_resume_preserve_patience(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory) / "checkpoints"
            trainer = self._make_trainer(
                checkpoint_dir=checkpoint_dir,
                early_stopping_patience=2,
            )
            trainer.optimizer_args = {"lr": 0.0}
            train_loader = DataLoader(_dataset([1, 2]), batch_size=2)
            valid_loader = DataLoader(_dataset([3, 4]), batch_size=2)
            history = trainer.fit(
                train_loader,
                valid_loader,
                epochs=10,
                verbose=False,
            )
            self.assertEqual(len(history), 3)
            self.assertEqual(trainer.epochs_without_improvement, 2)

            resumed = self._make_trainer(
                checkpoint_dir=checkpoint_dir,
                early_stopping_patience=2,
            )
            resumed.optimizer_args = {"lr": 0.0}
            resumed.load_checkpoint(
                checkpoint_dir / "last.pt",
                train_loader=train_loader,
            )
            resumed.fit(
                train_loader,
                valid_loader,
                epochs=5,
                verbose=False,
            )
            self.assertEqual(len(resumed.history), 3)

    def test_epoch_summary_uses_aligned_tables(self):
        record = {
            "epoch": 10,
            "learning_rate": 1.0e-4,
            "train_losses": {"c1": 4.0, "total": 4.0},
            "valid_losses": {"c1": 5.0, "total": 5.0},
            "train_metrics": {
                "c1": {
                    "mae": 1.0,
                    "rmse": 2.0,
                    "rmse_percent": 300.0,
                    "pearson_r": 0.75,
                }
            },
            "valid_metrics": {
                "c1": {
                    "mae": 1.5,
                    "rmse": 2.5,
                    "rmse_percent": 350.0,
                    "pearson_r": 0.5,
                }
            },
        }

        summary = Trainer._format_record(record)

        self.assertIn("Epoch   10 | learning rate 1.000e-04", summary)
        self.assertIn("Losses", summary)
        self.assertIn("c1 metrics", summary)
        self.assertIn("RMSE / sigma (%)", summary)
        self.assertIn("Pearson r", summary)
        self.assertGreaterEqual(summary.count("\n"), 8)


if __name__ == "__main__":
    unittest.main()
