import unittest

import numpy as np

from equicdft._argument_checks import (
    boolean,
    finite_scalar,
    nonempty_string,
    nonnegative_integer,
    nonnegative_scalar,
    optional_boolean,
    optional_positive_integer,
    positive_integer,
    positive_scalar,
    unique_strings,
)


class TestArgumentChecks(unittest.TestCase):
    def test_integer_checks_accept_python_and_numpy_integers(self):
        self.assertEqual(positive_integer(2, "value"), 2)
        self.assertEqual(positive_integer(np.int64(3), "value"), 3)
        self.assertEqual(nonnegative_integer(0, "value"), 0)
        self.assertIsNone(optional_positive_integer(None, "value"))

    def test_integer_checks_reject_boolean_float_and_invalid_values(self):
        for value in (True, 2.0, "2"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    positive_integer(value, "value")
        with self.assertRaises(ValueError):
            positive_integer(0, "value")
        with self.assertRaises(ValueError):
            nonnegative_integer(-1, "value")

    def test_scalar_checks_accept_finite_real_values(self):
        self.assertEqual(finite_scalar(np.float64(-2.0), "value"), -2.0)
        self.assertEqual(positive_scalar(2, "value"), 2.0)
        self.assertEqual(nonnegative_scalar(0.0, "value"), 0.0)

    def test_scalar_checks_reject_coercion_and_invalid_values(self):
        for value in (True, "1.0", [1.0]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    finite_scalar(value, "value")
        for value in (float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    finite_scalar(value, "value")
        with self.assertRaises(ValueError):
            positive_scalar(0.0, "value")
        with self.assertRaises(ValueError):
            nonnegative_scalar(-1.0, "value")

    def test_boolean_checks_do_not_coerce(self):
        self.assertTrue(boolean(True, "value"))
        self.assertFalse(boolean(False, "value"))
        self.assertIsNone(optional_boolean(None, "value"))
        for value in (0, 1, "true"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    boolean(value, "value")

    def test_string_checks_require_nonempty_unique_strings(self):
        self.assertEqual(nonempty_string("rho", "name"), "rho")
        self.assertEqual(
            unique_strings(["train", "valid"], "subsets"),
            ("train", "valid"),
        )
        with self.assertRaises(ValueError):
            nonempty_string("", "name")
        with self.assertRaises(TypeError):
            nonempty_string(1, "name")
        with self.assertRaises(TypeError):
            unique_strings("train", "subsets")
        with self.assertRaises(TypeError):
            unique_strings(["train", 1], "subsets")
        with self.assertRaises(ValueError):
            unique_strings([], "subsets")
        with self.assertRaises(ValueError):
            unique_strings(["train", "train"], "subsets")


if __name__ == "__main__":
    unittest.main()
