import unittest

import torch
from torch.utils.data import RandomSampler, SequentialSampler, Subset

from equicdft import make_dataloaders


def _dataset(values, n_grid=4, temperatures=None):
    """Return minimal complete-field dictionaries for loader tests."""

    frames = []
    for frame_id, value in enumerate(values):
        frame = {
            "frame_id": torch.tensor(frame_id),
            "rho": torch.full((n_grid, 1), float(value)),
        }
        if temperatures is not None:
            frame["temperature"] = torch.tensor(float(temperatures[frame_id]))
        frames.append(frame)
    return frames


class TestMakeDataloaders(unittest.TestCase):
    def test_fractional_split_is_deterministic_disjoint_and_complete(self):
        dataset = _dataset(range(10))
        loaders_a = make_dataloaders(
            dataset,
            valid_fraction=0.3,
            batch_size=2,
            seed=17,
        )
        loaders_b = make_dataloaders(
            dataset,
            valid_fraction=0.3,
            batch_size=2,
            seed=17,
        )
        train_a = loaders_a["train"]
        valid_a = loaders_a["valid"]
        test_a = loaders_a["test"]
        train_b = loaders_b["train"]
        valid_b = loaders_b["valid"]
        test_b = loaders_b["test"]

        self.assertEqual(set(loaders_a), {"train", "valid", "test"})
        self.assertIsInstance(train_a.dataset, Subset)
        self.assertIsInstance(valid_a.dataset, Subset)
        self.assertEqual(train_a.dataset.indices, train_b.dataset.indices)
        self.assertEqual(valid_a.dataset.indices, valid_b.dataset.indices)
        self.assertEqual(len(train_a.dataset), 7)
        self.assertEqual(len(valid_a.dataset), 3)
        self.assertEqual(
            set(train_a.dataset.indices) | set(valid_a.dataset.indices),
            set(range(10)),
        )
        self.assertTrue(
            set(train_a.dataset.indices).isdisjoint(valid_a.dataset.indices)
        )
        train_order_a = torch.cat(
            [batch["frame_id"] for batch in train_a]
        ).tolist()
        train_order_b = torch.cat(
            [batch["frame_id"] for batch in train_b]
        ).tolist()
        self.assertEqual(train_order_a, train_order_b)
        self.assertIsNone(test_a)
        self.assertIsNone(test_b)

    def test_explicit_validation_and_test_datasets(self):
        train_dataset = _dataset([1.0, 2.0, 3.0, 4.0])
        valid_dataset = _dataset([5.0, 6.0])
        test_dataset = _dataset([100.0, 200.0])

        loaders = make_dataloaders(
            train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            batch_size=3,
        )
        train_loader = loaders["train"]
        valid_loader = loaders["valid"]
        test_loader = loaders["test"]

        self.assertIs(train_loader.dataset, train_dataset)
        self.assertIs(valid_loader.dataset, valid_dataset)
        self.assertIs(test_loader.dataset, test_dataset)
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(valid_loader.sampler, SequentialSampler)
        self.assertIsInstance(test_loader.sampler, SequentialSampler)
        self.assertFalse(train_loader.drop_last)
        self.assertEqual(next(iter(train_loader))["rho"].shape, (3, 4, 1))
        self.assertEqual(next(iter(valid_loader))["rho"].shape, (2, 4, 1))

    def test_mean_density_uses_train_and_validation_but_not_test(self):
        train_dataset = _dataset([1.0, 3.0])
        valid_dataset = _dataset([5.0])
        test_dataset = _dataset([100.0])

        result = make_dataloaders(
            train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            compute_mean_density=True,
        )

        self.assertEqual(
            set(result),
            {"train", "valid", "test", "mean_density"},
        )
        self.assertAlmostEqual(result["mean_density"], 3.0)

    def test_fractional_mean_density_uses_complete_input_pool(self):
        dataset = _dataset([1.0, 3.0, 5.0, 7.0])

        result = make_dataloaders(
            dataset,
            valid_fraction=0.5,
            seed=8,
            compute_mean_density=True,
        )

        self.assertAlmostEqual(result["mean_density"], 4.0)

    def test_mean_temperature_uses_train_and_validation_but_not_test(self):
        train_dataset = _dataset(
            [1.0, 3.0],
            temperatures=[0.6, 0.9],
        )
        valid_dataset = _dataset([5.0], temperatures=[1.2])
        test_dataset = _dataset([100.0], temperatures=[100.0])

        result = make_dataloaders(
            train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            compute_mean_density=True,
            compute_mean_temperature=True,
        )

        self.assertEqual(
            set(result),
            {
                "train",
                "valid",
                "test",
                "mean_density",
                "mean_temperature",
            },
        )
        self.assertAlmostEqual(result["mean_density"], 3.0)
        self.assertAlmostEqual(result["mean_temperature"], 0.9)

    def test_mean_temperature_can_be_returned_without_mean_density(self):
        dataset = _dataset(
            [1.0, 3.0, 5.0, 7.0],
            temperatures=[0.6, 0.9, 1.2, 1.5],
        )

        result = make_dataloaders(
            dataset,
            valid_fraction=0.5,
            seed=8,
            compute_mean_temperature=True,
        )

        self.assertEqual(
            set(result),
            {"train", "valid", "test", "mean_temperature"},
        )
        self.assertAlmostEqual(result["mean_temperature"], 1.05)

    def test_requires_exactly_one_validation_source(self):
        dataset = _dataset([1.0, 2.0])

        with self.assertRaisesRegex(ValueError, "exactly one"):
            make_dataloaders(dataset)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            make_dataloaders(
                dataset,
                valid_dataset=_dataset([3.0]),
                valid_fraction=0.5,
            )

    def test_rejects_invalid_fraction_and_too_few_frames(self):
        for fraction in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "valid_fraction"):
                    make_dataloaders(
                        _dataset([1.0, 2.0]),
                        valid_fraction=fraction,
                    )

        with self.assertRaisesRegex(ValueError, "at least two"):
            make_dataloaders(_dataset([1.0]), valid_fraction=0.5)

    def test_rejects_empty_datasets_and_invalid_loader_settings(self):
        dataset = _dataset([1.0, 2.0])

        with self.assertRaisesRegex(ValueError, "train_dataset"):
            make_dataloaders([], valid_fraction=0.5)
        with self.assertRaisesRegex(ValueError, "valid_dataset"):
            make_dataloaders(dataset, valid_dataset=[])
        with self.assertRaisesRegex(ValueError, "test_dataset"):
            make_dataloaders(
                dataset,
                valid_dataset=_dataset([3.0]),
                test_dataset=[],
            )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            make_dataloaders(dataset, valid_fraction=0.5, batch_size=0)
        with self.assertRaisesRegex(ValueError, "num_workers"):
            make_dataloaders(dataset, valid_fraction=0.5, num_workers=-1)

    def test_mean_density_validates_rho(self):
        with self.assertRaisesRegex(KeyError, "rho"):
            make_dataloaders(
                [{"frame_id": torch.tensor(0)}],
                valid_dataset=_dataset([1.0]),
                compute_mean_density=True,
            )

        nonfinite = _dataset([1.0])
        nonfinite[0]["rho"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            make_dataloaders(
                nonfinite,
                valid_dataset=_dataset([1.0]),
                compute_mean_density=True,
            )

    def test_mean_temperature_validates_temperature(self):
        with self.assertRaisesRegex(KeyError, "temperature"):
            make_dataloaders(
                _dataset([1.0]),
                valid_dataset=_dataset([2.0]),
                compute_mean_temperature=True,
            )

        invalid_values = (float("nan"), 0.0, -1.0)
        for value in invalid_values:
            with self.subTest(value=value):
                invalid = _dataset([1.0], temperatures=[value])
                with self.assertRaisesRegex(ValueError, "finite|positive"):
                    make_dataloaders(
                        invalid,
                        valid_dataset=_dataset([2.0], temperatures=[1.0]),
                        compute_mean_temperature=True,
                    )


if __name__ == "__main__":
    unittest.main()
