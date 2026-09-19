import unittest

import torch

from tests.regression.goldens import CASES, GOLDEN_PATH, TOLERANCES, compare


@unittest.skipUnless(GOLDEN_PATH.is_file(), "no recorded goldens")
class TestGoldenOutputs(unittest.TestCase):
    """Every recorded case reproduces its numbers.

    Record with ``python -m tests.regression.goldens record`` at a commit
    whose numbers are trusted; a deliberate numerical change re-records in
    the same commit and explains itself in the commit message.
    """

    @classmethod
    def setUpClass(cls):
        cls.recorded = torch.load(str(GOLDEN_PATH), weights_only=False)["cases"]

    def test_every_case_is_recorded(self):
        self.assertEqual(set(self.recorded), set(CASES))

    def test_cases_reproduce(self):
        for name in sorted(CASES):
            with self.subTest(case=name):
                compare(self.recorded[name], CASES[name](), TOLERANCES.get(name))


if __name__ == "__main__":
    unittest.main()
