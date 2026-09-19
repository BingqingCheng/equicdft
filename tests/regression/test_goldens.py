import unittest

import torch

from tests.regression.goldens import (
    CASES,
    CROSS_VERSION_TOLERANCE,
    GOLDEN_PATH,
    TOLERANCES,
    compare,
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
                if not self.same_torch:
                    tolerance = CROSS_VERSION_TOLERANCE
                compare(self.recorded[name], CASES[name](), tolerance)


if __name__ == "__main__":
    unittest.main()
