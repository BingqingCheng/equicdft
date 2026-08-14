import unittest

from examples.lj_paper_v1_regression.common import check_thresholds
from examples.lj_paper_v1_regression.forward import evaluate_forward
from examples.lj_paper_v1_regression.solve import evaluate_solve


class TestLJPaperV1Regression(unittest.TestCase):
    def test_forward_rho_to_external_potential(self):
        check_thresholds("forward", evaluate_forward())

    def test_fixed_n_external_potential_to_density(self):
        check_thresholds("solve", evaluate_solve())


if __name__ == "__main__":
    unittest.main()
