import unittest

import torch

from tests.regression.goldens import (
    CASES,
    CROSS_VERSION_TOLERANCE,
    GOLDEN_PATH,
    TOLERANCES,
    compare,
    compare_trainer_cross_version,
)


@unittest.skipUnless(GOLDEN_PATH.is_file(), "no recorded goldens")
class TestGoldenOutputs(unittest.TestCase):
    """Every recorded case reproduces its numbers.

    Record with ``python -m tests.regression.goldens record`` at a commit
    whose numbers are trusted; a deliberate numerical change re-records in
    the same commit and explains itself in the commit message. The
    comparison is exact under the torch release that recorded the file and
    tolerant under any other.
    """

    @classmethod
    def setUpClass(cls):
        recorded = torch.load(str(GOLDEN_PATH), weights_only=False)
        cls.recorded = recorded["cases"]
        cls.same_torch = recorded["torch_version"] == str(torch.__version__)

    def test_every_case_is_recorded(self):
        self.assertEqual(set(self.recorded), set(CASES))

    def test_cases_reproduce(self):
        for name in sorted(CASES):
            with self.subTest(case=name):
                tolerance = TOLERANCES.get(name)
                actual = CASES[name]()
                if not self.same_torch and name == "trainer/two_epochs_lda_local":
                    compare_trainer_cross_version(self.recorded[name], actual)
                    continue
                if not self.same_torch:
                    tolerance = CROSS_VERSION_TOLERANCE
                compare(self.recorded[name], actual, tolerance)


class TestTrainerCrossVersionTolerance(unittest.TestCase):
    def test_constant_target_ratios_allow_roundoff_but_not_finite_variance(self):
        recorded = {
            "history[1]/valid_metrics/mu/r2": torch.tensor(-8.24e14),
            "history[1]/valid_metrics/mu/rmse_percent": torch.tensor(2.87e9),
        }
        actual = {key: value * float("inf") for key, value in recorded.items()}
        compare_trainer_cross_version(recorded, actual)
        for key, incorrect in (
            ("history[1]/valid_metrics/mu/r2", -100.0),
            ("history[1]/valid_metrics/mu/rmse_percent", 1000.0),
            ("history[1]/valid_metrics/mu/r2", float("inf")),
        ):
            with self.subTest(key=key, incorrect=incorrect):
                changed = dict(actual, **{key: torch.tensor(incorrect)})
                with self.assertRaises(AssertionError):
                    compare_trainer_cross_version(recorded, changed)

    def test_only_gauge_biases_get_the_wider_parameter_tolerance(self):
        for key in ("state/readout.0.mlp.2.bias", "state/readout.0.mlp.0.weight"):
            with self.subTest(key=key):
                recorded = {key: torch.tensor(0.1)}
                changed = {key: torch.tensor(0.1004)}
                if key.endswith(".2.bias"):
                    compare_trainer_cross_version(recorded, changed)
                    with self.assertRaises(AssertionError):
                        compare_trainer_cross_version(recorded, {key: torch.tensor(0.101)})
                else:
                    with self.assertRaises(AssertionError):
                        compare_trainer_cross_version(recorded, changed)


if __name__ == "__main__":
    unittest.main()
