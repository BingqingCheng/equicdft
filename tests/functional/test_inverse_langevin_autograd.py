"""Implicit inverse-Langevin differentiation without a Newton-step graph."""

import unittest

import torch

from equicdft.functional.polarization_ideal import (
    FixedDipoleIdeal,
    _langevin_prime,
    inverse_langevin,
    langevin,
)


def legacy_inverse_langevin(p):
    """Pre-optimization value and unrolled graph, also usable by benchmarks."""
    if not torch.isfinite(p).all() or torch.any(p.abs() >= 1):
        raise ValueError("inverse Langevin requires finite |p| < 1")
    x = p * (3 - p.square()) / (1 - p.square())
    for _ in range(12):
        x = x - (langevin(x) - p) / _langevin_prime(x)
    return x


def derivatives(function, p, order=3):
    values = [function(p)]
    for _ in range(order):
        values.append(torch.autograd.grad(values[-1].sum(), p, create_graph=True)[0])
    return values


class TestInverseLangevinAutograd(unittest.TestCase):
    def test_value_is_unchanged_and_preserves_dtype_device_and_shape(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    p = torch.tensor(
                        [[-.9999, -.8, -.01], [0., 1e-8, .95]],
                        dtype=dtype, device=device,
                    )
                    result = inverse_langevin(p)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.device, p.device)
                    self.assertEqual(result.shape, p.shape)
                    torch.testing.assert_close(result, legacy_inverse_langevin(p), rtol=0, atol=0)
                    self.assertFalse(result.requires_grad)
        for p in (torch.tensor(0., dtype=torch.float64), torch.empty(0, dtype=torch.float64)):
            torch.testing.assert_close(inverse_langevin(p), legacy_inverse_langevin(p), rtol=0, atol=0)

    def test_first_second_and_third_implicit_derivatives(self):
        p = torch.tensor(
            [-.9999, -.99, -.8, -.2, -.01, -1e-6, 0., 1e-6, .01, .2, .8, .99, .9999],
            dtype=torch.float64, requires_grad=True,
        )
        actual = derivatives(inverse_langevin, p)
        x = actual[0].detach().requires_grad_()
        prime = _langevin_prime(x)
        second = torch.autograd.grad(prime.sum(), x, create_graph=True)[0]
        third = torch.autograd.grad(second.sum(), x)[0]
        expected = [
            1 / prime,
            -second / prime ** 3,
            (3 * second.square() - third * prime) / prime ** 5,
        ]
        for result, reference in zip(actual[1:], expected):
            torch.testing.assert_close(result, reference, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(actual[1][6], p.new_tensor(3.), rtol=0, atol=0)
        torch.testing.assert_close(actual[2][6], p.new_tensor(0.), rtol=0, atol=0)
        torch.testing.assert_close(actual[3][6], p.new_tensor(10.8), rtol=2e-15, atol=0)

    def test_gradcheck_and_gradgradcheck(self):
        p = torch.tensor([-.91, -.4, -.01, 0., .001, .25, .82], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(inverse_langevin, (p,), eps=1e-6, atol=2e-7, rtol=2e-6))
        self.assertTrue(torch.autograd.gradgradcheck(inverse_langevin, (p,), eps=1e-6, atol=2e-6, rtol=2e-6))

    def test_matches_unrolled_higher_derivatives_away_from_branch_transition(self):
        p = torch.tensor([-.999, -.9, -.3, -.01, 0., .01, .3, .9, .999], dtype=torch.float64, requires_grad=True)
        actual = derivatives(inverse_langevin, p)
        legacy = derivatives(legacy_inverse_langevin, p)
        for result, reference in zip(actual, legacy):
            torch.testing.assert_close(result, reference, rtol=2e-9, atol=2e-10)

    def test_mixed_density_polarization_and_parameter_gradients(self):
        rho = torch.tensor([[.02], [.04]], dtype=torch.float64, requires_grad=True)
        polar = torch.tensor([[[.001, -.002, .0005]], [[.004, .001, -.002]]], dtype=torch.float64, requires_grad=True)
        exponent = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        moment = .49

        def energy(root):
            square = polar.square().sum(dim=-1) / (moment * rho).square()
            u = square.sqrt()
            x = root(u)
            # This nonlinear energy exercises root, density, polarization and
            # learned fading derivatives together, including Hessian training.
            return (rho * x.square() * torch.exp(-exponent * square)).sum()

        results = []
        for root in (inverse_langevin, legacy_inverse_langevin):
            first = torch.autograd.grad(energy(root), (rho, polar, exponent), create_graph=True)
            mixed = torch.autograd.grad(first[1].sum(), rho, create_graph=True)[0]
            hessian = torch.autograd.grad(first[1].sum(), polar, create_graph=True)[0]
            trained = torch.autograd.grad(hessian.square().sum(), exponent, retain_graph=True)[0]
            results.append((*first, mixed, hessian, trained))
        for result, reference in zip(*results):
            torch.testing.assert_close(result, reference, rtol=2e-10, atol=2e-9)
        ideal = FixedDipoleIdeal(moment)
        self.assertTrue(torch.autograd.gradgradcheck(
            lambda r, p: ideal.evaluate(dict(rho=r, dipole_density=p), 1.)["beta_F_id"],
            (rho, polar), eps=1e-7, atol=2e-5, rtol=2e-5,
        ))

    def test_forward_ad_and_func_transforms(self):
        p = torch.tensor([[-.8, -.1, 0.], [.01, .2, .9]], dtype=torch.float64)
        tangent = torch.tensor([[.2, -.1, 1.], [.3, .4, -.2]], dtype=torch.float64)
        reference = 1 / _langevin_prime(inverse_langevin(p))
        value, result = torch.func.jvp(inverse_langevin, (p,), (tangent,))
        torch.testing.assert_close(value, inverse_langevin(p), rtol=0, atol=0)
        torch.testing.assert_close(result, tangent * reference)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(p, tangent)
            _, result = torch.autograd.forward_ad.unpack_dual(inverse_langevin(dual))
            torch.testing.assert_close(result, tangent * reference)
        torch.testing.assert_close(torch.func.vmap(inverse_langevin)(p), value, rtol=0, atol=0)
        torch.testing.assert_close(torch.func.vmap(inverse_langevin, in_dims=1, out_dims=1)(p), value, rtol=0, atol=0)
        torch.testing.assert_close(torch.func.vmap(torch.func.vmap(inverse_langevin))(p), value, rtol=0, atol=0)
        expected = torch.diag(reference.flatten())
        torch.testing.assert_close(torch.func.jacrev(inverse_langevin)(p).reshape(6, 6), expected)
        torch.testing.assert_close(torch.func.jacfwd(inverse_langevin)(p).reshape(6, 6), expected)
        torch.testing.assert_close(torch.func.vmap(torch.func.jacrev(inverse_langevin))(p), torch.diag_embed(reference))
        func_hessian = torch.func.hessian(lambda q: inverse_langevin(q).sum())(p)
        autograd_hessian = torch.autograd.functional.hessian(lambda q: inverse_langevin(q).sum(), p)
        torch.testing.assert_close(func_hessian, autograd_hessian)

    def test_invalid_values_preserve_errors_in_regular_and_batched_calls(self):
        for bad in (-1., 1., -1.01, 1.01, float("nan"), float("inf"), -float("inf")):
            p = torch.tensor([[.1, bad]], dtype=torch.float64)
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "finite.*< 1"):
                    inverse_langevin(p)
                with self.assertRaisesRegex(ValueError, "finite.*< 1"):
                    torch.func.vmap(inverse_langevin)(p)


if __name__ == "__main__":
    unittest.main()
