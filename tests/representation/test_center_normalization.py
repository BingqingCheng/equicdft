"""Small tensor checks for the opt-in reference center convention."""

import copy
import unittest

import torch

from equicdft import CartesianAFeatures
from equicdft.serialization import default_dtype
from tests.model_fixtures import grid_data


def _reference(features, ratio, spacing=0.5):
    """Describe the original stencil, not a fitted p0 or gamma snapshot."""
    return {
        "grid_spacing": spacing,
        "offsets": features.local_density_positions.tolist(),
        "neighbor_mask": features.neighbor_mask.tolist(),
        "cutoff_grid": features.cutoff_grid,
        "stencil_radius": features.stencil_radius,
        "monomial_scale": features.monomial_scale,
        "coordinate_scaling": features.coordinate_scaling,
        "spacing_ratio": ratio,
    }


def _features(exponents=(0.15, 0.6), **overrides):
    options = dict(
        max_power=2,
        mean_density=0.4,
        cutoff_grid=1,
        radial_basis="gaussian",
        radial_exponents=exponents,
        trainable_radial_exponents=True,
        separate_center=True,
        n_types=2,
    )
    options.update(overrides)
    return CartesianAFeatures(**options)


def _scalar_and_center(features):
    scalar = torch.nonzero(features.powers.eq(0).all(dim=1)).item()
    center = torch.nonzero(features.local_density_positions.eq(0).all(dim=1)).item()
    return scalar, center


def _log_noncentral_fraction(r2, mask, alpha):
    log_noncentral = torch.logsumexp(-r2[mask, None] * alpha[None, :], dim=0)
    return log_noncentral - torch.logaddexp(
        torch.zeros_like(log_noncentral), log_noncentral
    )


class TestCenterNormalization(unittest.TestCase):
    def setUp(self):
        self.dtype_context = default_dtype(torch.float64)
        self.dtype_context.__enter__()

    def tearDown(self):
        self.dtype_context.__exit__(None, None, None)

    def test_strict_default_matches_existing_center_excluded_formula(self):
        features = _features()
        alpha = features.radial_exponents
        raw = torch.exp(-features.squared_distances[:, None] * alpha[None, :])
        raw = raw * features.neighbor_mask[:, None]
        expected = (raw / raw.sum(0, keepdim=True))[:, :, None]
        expected = expected * features.monomial_values[:, None, :]

        self.assertEqual(features.center_normalization, "strict")
        self.assertTrue(torch.equal(features.stencil_basis(), expected))

    def test_native_geometry_returns_original_basis_and_parameter_gradient(self):
        native = _features()
        options = native.to_config()
        options.update(
            center_normalization="reference",
            normalization_reference=_reference(native, 1.0),
        )
        controlled = CartesianAFeatures.from_config(options)
        controlled.load_state_dict(native.state_dict())
        expected = native.stencil_basis()
        actual = controlled.stencil_basis()

        self.assertTrue(torch.equal(actual, expected))
        coefficient = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual)
        gradients = [torch.autograd.grad((basis * coefficient).sum(), module.log_radial_exponents)[0]
                     for basis, module in ((expected, native), (actual, controlled))]
        self.assertTrue(torch.equal(gradients[0], gradients[1]))

    def test_three_channels_have_independent_factors_and_scalar_center_terms(self):
        native = _features((0.1, 0.35, 0.8))
        ratio = 0.5
        alpha = native.radial_exponents.detach() * ratio**2
        plain = _features(alpha, cutoff_grid=2, monomial_scale=ratio)
        controlled = _features(
            alpha,
            cutoff_grid=2,
            monomial_scale=ratio,
            center_normalization="reference",
            normalization_reference=_reference(native, ratio),
        )
        log_target = _log_noncentral_fraction(
            plain.squared_distances, plain.neighbor_mask, alpha
        )
        log_reference = _log_noncentral_fraction(
            native.squared_distances, native.neighbor_mask, alpha / ratio**2
        )
        gamma = torch.exp(log_target - log_reference)
        expected = plain.stencil_basis() * gamma[None, :, None]
        scalar, center = _scalar_and_center(controlled)
        expected[center, :, scalar] += 1.0 - gamma
        actual = controlled.stencil_basis()

        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
        self.assertTrue(torch.all(gamma > 1.0).item())
        self.assertEqual(torch.unique(gamma).numel(), 3)
        torch.testing.assert_close(actual[center, :, scalar], 1.0 - gamma)
        torch.testing.assert_close(actual[:, :, scalar].sum(0), torch.ones(3))
        nonconstant = controlled.powers.ne(0).any(dim=1)
        self.assertTrue(torch.equal(actual[center, :, nonconstant],
                                    torch.zeros_like(actual[center, :, nonconstant])))

    def test_uniform_scalar_response_is_preserved_for_every_species_and_radial(self):
        native = _features()
        controlled = _features(
            native.radial_exponents.detach() / 4.0,
            cutoff_grid=2,
            monomial_scale=0.5,
            center_normalization="reference",
            normalization_reference=_reference(native, 0.5),
        )
        data = grid_data(shape=(4, 4, 4), cutoff_grid=2, n_types=2)
        data["rho"] = torch.tensor([0.2, 0.7]).expand(64, 2).clone()
        actual = controlled(data)
        scalar, _ = _scalar_and_center(controlled)
        expected = (data["rho"] / controlled.mean_density)[:, None, :].expand(64, 2, 2)
        torch.testing.assert_close(actual[:, :, scalar, :], expected, rtol=1e-13, atol=1e-13)

    def test_live_exponent_gradient_includes_reference_normalization(self):
        native = _features()
        controlled = _features(
            native.radial_exponents.detach() / 4.0,
            cutoff_grid=2,
            monomial_scale=0.5,
            center_normalization="reference",
            normalization_reference=_reference(native, 0.5),
        )
        coefficients = torch.linspace(-0.4, 0.7, controlled.stencil_basis().numel())
        coefficients = coefficients.reshape_as(controlled.stencil_basis())
        value = (controlled.stencil_basis() * coefficients).sum()
        actual = torch.autograd.grad(value, controlled.log_radial_exponents)[0]
        step = 1e-5
        finite = []
        for index in range(actual.numel()):
            original = controlled.log_radial_exponents.detach().clone()
            values = []
            for sign in (1.0, -1.0):
                with torch.no_grad():
                    controlled.log_radial_exponents.copy_(original)
                    controlled.log_radial_exponents[index] += sign * step
                values.append((controlled.stencil_basis() * coefficients).sum().detach())
            with torch.no_grad():
                controlled.log_radial_exponents.copy_(original)
            finite.append((values[0] - values[1]) / (2 * step))
        torch.testing.assert_close(actual, torch.stack(finite), rtol=2e-8, atol=2e-9)

    def test_narrow_gaussian_uses_finite_log_normalization(self):
        # p0_ref rounds to one in ordinary subtraction, but its noncentral
        # log fraction and the ratio remain finite. Large signed coefficients
        # cannot retain machine-precision uniform sums; this is not that test.
        native = _features((40.0,))
        controlled = _features(
            (10.0,),
            cutoff_grid=2,
            monomial_scale=0.5,
            center_normalization="reference",
            normalization_reference=_reference(native, 0.5),
        )
        actual = controlled.stencil_basis()
        self.assertTrue(torch.isfinite(actual).all().item())
        scalar, center = _scalar_and_center(controlled)
        log_gamma = (
            _log_noncentral_fraction(controlled.squared_distances, controlled.neighbor_mask,
                                     controlled.radial_exponents)
            - _log_noncentral_fraction(native.squared_distances, native.neighbor_mask,
                                       controlled.radial_exponents * 4.0)
        )
        expected_center = -torch.expm1(log_gamma)
        torch.testing.assert_close(actual[center, :, scalar], expected_center, rtol=2e-13, atol=0.0)
        derivative = torch.autograd.grad(actual[center, :, scalar].sum(),
                                         controlled.log_radial_exponents)[0]
        self.assertTrue(torch.isfinite(derivative).all().item())

    def test_missing_reference_and_invalid_policy_are_rejected(self):
        for options in ({"center_normalization": "reference"},
                        {"center_normalization": "automatic"}):
            with self.subTest(options=options):
                with self.assertRaises((ValueError, TypeError)):
                    _features(**options)

    def test_reference_configuration_does_not_add_fitted_state_keys(self):
        native = _features()
        controlled = _features(
            center_normalization="reference",
            normalization_reference=_reference(native, 1.0),
        )
        self.assertEqual(set(native.state_dict()), set(controlled.state_dict()))
        restored = CartesianAFeatures.from_config(copy.deepcopy(controlled.to_config()))
        restored.load_state_dict(controlled.state_dict())
        self.assertTrue(torch.equal(restored.stencil_basis(), controlled.stencil_basis()))

    def test_returned_reference_configuration_does_not_alias_live_metadata(self):
        native = _features()
        controlled = _features(
            center_normalization="reference",
            normalization_reference=_reference(native, 1.0),
        )
        before = copy.deepcopy(controlled.normalization_reference)
        config = controlled.to_config()
        config["normalization_reference"]["offsets"][0][0] += 100
        config["normalization_reference"]["neighbor_mask"][0] = not (
            config["normalization_reference"]["neighbor_mask"][0]
        )
        self.assertEqual(controlled.normalization_reference, before)
        self.assertEqual(controlled.to_config()["normalization_reference"], before)

    def test_native_reference_policy_still_rejects_shifted_override_centers(self):
        native = _features()
        controlled = _features(
            center_normalization="reference",
            normalization_reference=_reference(native, 1.0),
        )
        with self.assertRaises((ValueError, NotImplementedError)):
            controlled.stencil_basis(
                controlled.radial_exponents, torch.tensor([0.0, 0.5])
            )

    def test_reference_rejects_inconsistent_current_radius_or_monomial_units(self):
        native = _features()
        for wrong in ({"stencil_radius": 1.5}, {"monomial_scale": 0.75}):
            options = dict(
                cutoff_grid=2, stencil_radius=2.0, monomial_scale=0.5,
                center_normalization="reference",
                normalization_reference=_reference(native, 0.5),
            )
            options.update(wrong)
            with self.subTest(wrong=wrong):
                with self.assertRaises(ValueError):
                    _features(native.radial_exponents.detach() / 4.0, **options)


if __name__ == "__main__":
    unittest.main()
