import csv
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from equicdft import FourierStabilityLoss, Loss, Metrics, TensorLoss, Trainer


class _LinearDictionaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        return {"prediction": self.weight * batch["x"]}


class _QuadraticDictionaryFunctional(nn.Module):
    """Small functional for exercising model-dependent trainer losses."""

    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(-4.0))

    def forward(self, batch, compute_c1=None):
        rho = batch["rho"]
        cell_volume = torch.prod(batch["grid_spacing"].to(rho), dim=-1)
        return {
            "beta_F_exc": (
                0.5
                * self.coefficient
                * cell_volume
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
        self.assertEqual(trainer.metrics[0].logs["train"]["prediction"], [])
        self.assertEqual(trainer.metrics[0].logs["valid"]["prediction"], [])

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
        for invalid_patience in (0, -1, True, 1.5):
            with self.subTest(invalid_patience=invalid_patience):
                with self.assertRaises(ValueError):
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
