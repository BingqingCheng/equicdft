import unittest

import torch
from torch import nn

from equicdft.evolution.mobility import ConstantMobility, LocalMobility


class TestMobility(unittest.TestCase):
    def setUp(self):
        self.rho = torch.ones(2, 4, 3, dtype=torch.float64)
        self.data = {"rho": self.rho, "temperature": torch.tensor(2.0)}

    def test_constant_scalar_vector_and_matrix(self):
        cases = [
            (2.0, 2 * torch.eye(3, dtype=torch.float64)),
            ([1.0, 2.0, 0.0], torch.diag(torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64))),
            ([[1.0, -0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 0.0]],
             torch.tensor([[1.0, -0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                result = ConstantMobility(value)(self.data)
                self.assertEqual(result.shape, (2, 4, 3, 3))
                self.assertEqual(result.dtype, self.rho.dtype)
                torch.testing.assert_close(result, expected.expand_as(result))

    def test_constant_rejects_invalid_values(self):
        for value in (-1.0, [1.0, -1.0], [[1.0, 2.0], [2.0, 1.0]],
                      [[1.0, 0.1], [0.0, 1.0]], float("nan"), complex(1, 1), []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ConstantMobility(value)
        with self.assertRaises(ValueError):
            ConstantMobility([1.0, 2.0])(self.data)

    def test_tolerance_is_relative_to_mobility_scale(self):
        with self.assertRaises(ValueError):
            ConstantMobility(torch.tensor([[1e-12, 0.0], [0.0, -1e-12]]))
        # A zero eigenvalue perturbed only at floating-point roundoff is valid.
        ConstantMobility(torch.tensor([[1.0, 1.0], [1.0, 1.0 - 1e-15]], dtype=torch.float64))

    def test_constant_copies_source_and_allows_rank_deficiency(self):
        source = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
        mobility = ConstantMobility(source)
        source.zero_()
        result = mobility({"rho": torch.ones(4, 2, dtype=torch.float64)})
        expected = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
        torch.testing.assert_close(result, expected.expand_as(result))
        torch.testing.assert_close(result @ torch.ones(2, dtype=torch.float64), torch.zeros(4, 2, dtype=torch.float64))

    def test_local_receives_live_density_and_temperature(self):
        rho = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3).requires_grad_()
        temperature = torch.tensor(2.0, dtype=torch.float64)

        def local(field, temp):
            self.assertIs(field, rho)
            self.assertIs(temp, temperature)
            return field + temp

        result = LocalMobility(local)({"rho": rho, "temperature": temperature})
        torch.testing.assert_close(result, torch.diag_embed(rho + temperature))
        result.sum().backward()
        torch.testing.assert_close(rho.grad, torch.ones_like(rho))

    def test_local_full_matrix_and_shared_forms(self):
        expected = torch.tensor([[1.0, -0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 2.0]])
        result = LocalMobility(lambda rho, temp: expected.expand(rho.shape + (3,)))(self.data)
        torch.testing.assert_close(result, expected.double().expand_as(result))
        for value in (2.0, [1.0, 2.0, 3.0], expected):
            with self.subTest(value=value):
                local = LocalMobility(lambda rho, temp: value)(self.data)
                torch.testing.assert_close(local, ConstantMobility(value)(self.data))

    def test_local_rejects_invalid_matrices(self):
        invalid = torch.tensor([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            LocalMobility(lambda rho, temp: invalid.expand(rho.shape + (3,)))(self.data)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            LocalMobility(lambda rho, temp: -rho)(self.data)
        with self.assertRaisesRegex(ValueError, "finite"):
            LocalMobility(lambda rho, temp: rho * float("inf"))(self.data)

    def test_local_module_registration_and_device_dtype(self):
        class Function(nn.Module):
            def __init__(self):
                super().__init__()
                self.factor = nn.Parameter(torch.tensor(2.0))

            def forward(self, rho, temperature):
                return self.factor * rho

        mobility = LocalMobility(Function()).double()
        self.assertIn("function.factor", dict(mobility.named_parameters()))
        result = mobility(self.data)
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result.device, self.rho.device)
        if torch.cuda.is_available():
            mobility = mobility.cuda()
            result = mobility({"rho": self.rho.cuda(), "temperature": self.data["temperature"].cuda()})
            self.assertEqual(result.device.type, "cuda")

    def test_local_density_shape_has_diagonal_priority(self):
        rho = torch.ones(2, 2)
        coefficients = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = LocalMobility(lambda field, temp: coefficients)({"rho": rho, "temperature": 1.0})
        torch.testing.assert_close(result, torch.diag_embed(coefficients))


if __name__ == "__main__":
    unittest.main()
