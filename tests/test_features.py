import math
import unittest

import torch
from torch import nn

from equicdft import CartesianAFeatures, CartesianBFeatures


class TestCartesianAFeatures(unittest.TestCase):
    def test_fixed_monomials_and_density_contraction(self):
        module = CartesianAFeatures(
            mean_density=2.0,
            cutoff_grid=1,
            max_power=2,
            radial_basis="gaussian",
            radial_exponents=(0.5,),
            separate_center=False,
            trainable_radial_exponents=False,
        )

        expected_powers = torch.tensor(
            [
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (2, 0, 0),
                (1, 1, 0),
                (1, 0, 1),
                (0, 2, 0),
                (0, 1, 1),
                (0, 0, 2),
            ],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(module.powers, expected_powers))
        self.assertEqual(module.monomial_values.shape, (7, 10))
        self.assertTrue(
            torch.equal(module.radial_exponents, torch.tensor([0.5]))
        )
        self.assertFalse(hasattr(module, "log_radial_exponents"))
        self.assertEqual(module.mean_density.item(), 2.0)
        self.assertTrue(
            torch.equal(
                module.monomial_values[0],
                torch.tensor(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    dtype=module.monomial_values.dtype,
                ),
            )
        )

        rho = torch.arange(
            1,
            8,
            dtype=module.monomial_values.dtype,
        ).reshape(7, 1)
        local_density_index = torch.arange(7).repeat(7, 1)
        features = module(
            {
                "rho": rho,
                "local_density_index": local_density_index,
            }
        )

        self.assertEqual(features.shape, (7, 1, 10, 1))
        # Evaluate the analytic reference in the module's float32 dtype so the
        # comparison tests the formula rather than Python-double roundoff.
        radial_weight = torch.exp(
            torch.tensor(-0.5, dtype=module.monomial_values.dtype)
        )
        normalization = 1.0 + 6.0 * radial_weight
        expected_scalar = (1.0 + 27.0 * radial_weight) / (
            2.0 * normalization
        )
        expected_vector = -radial_weight / (2.0 * normalization)
        self.assertAlmostEqual(
            features[0, 0, 0, 0].item(), expected_scalar.item()
        )
        for component in (1, 2, 3):
            self.assertAlmostEqual(
                features[0, 0, component, 0].item(),
                expected_vector.item(),
            )

    def test_batched_data(self):
        module = CartesianAFeatures(
            mean_density=2.0,
            cutoff_grid=0,
            max_power=0,
            separate_center=False,
        )
        rho = torch.tensor(
            [[[2.0]], [[3.0]]],
            dtype=module.monomial_values.dtype,
        )
        local_density_index = torch.zeros((2, 1, 1), dtype=torch.long)
        features = module(
            {
                "rho": rho,
                "local_density_index": local_density_index,
            }
        )

        self.assertEqual(features.shape, (2, 1, 1, 1, 1))
        self.assertTrue(
            torch.allclose(
                features[:, 0, :, 0, 0],
                torch.tensor(
                    [[1.0], [1.5]],
                    dtype=features.dtype,
                ),
            )
        )

    def test_damping_and_raw_integer_monomials(self):
        module = CartesianAFeatures(
            mean_density=2.0,
            cutoff_grid=2,
            max_power=2,
            radial_basis="gaussian",
            radial_exponents=(0.5,),
            separate_center=False,
        )

        position_lookup = {
            tuple(position): index
            for index, position in enumerate(
                module.local_density_positions.tolist()
            )
        }
        plus_two_z = position_lookup[(0, 0, 2)]
        self.assertEqual(module.monomial_values[plus_two_z, 3].item(), 2.0)
        self.assertEqual(module.monomial_values[plus_two_z, 9].item(), 4.0)

        n_neighbors = module.local_density_positions.shape[0]
        rho = torch.full(
            (n_neighbors, 1),
            2.0,
            dtype=module.monomial_values.dtype,
        )
        local_density_index = torch.arange(n_neighbors).repeat(
            n_neighbors,
            1,
        )
        features = module(
            {
                "rho": rho,
                "local_density_index": local_density_index,
            }
        )
        # Unit-sum damping leaves the scalar feature of a uniform field at one.
        self.assertTrue(
            torch.allclose(
                features[:, :, 0, 0],
                torch.ones_like(features[:, :, 0, 0]),
            )
        )

    def test_cutoff_scaled_cartesian_monomials(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=2,
            max_power=2,
            coordinate_scaling="cutoff",
            separate_center=False,
        )
        position_lookup = {
            tuple(position): index
            for index, position in enumerate(
                module.local_density_positions.tolist()
            )
        }
        plus_two_z = position_lookup[(0, 0, 2)]

        self.assertEqual(module.coordinate_scaling, "cutoff")
        self.assertEqual(module.monomial_values[plus_two_z, 3].item(), 1.0)
        self.assertEqual(module.monomial_values[plus_two_z, 9].item(), 1.0)
        self.assertEqual(module.squared_distances[plus_two_z].item(), 4.0)

    def test_discrete_sum_normalized_gaussian_matches_undamped_limit(self):
        common = {
            "mean_density": 2.0,
            "cutoff_grid": 2,
            "max_power": 2,
            "separate_center": True,
            "coordinate_scaling": "none",
        }
        damped = CartesianAFeatures(
            **common,
            radial_basis="gaussian",
            radial_exponents=(1.0e-8,),
        )
        undamped = CartesianAFeatures(**common)
        n_neighbors = damped.local_density_positions.shape[0]
        rho = torch.arange(
            1,
            n_neighbors + 1,
            dtype=damped.monomial_values.dtype,
        ).reshape(n_neighbors, 1)
        data = {
            "rho": rho,
            "local_density_index": torch.arange(n_neighbors).repeat(
                n_neighbors,
                1,
            ),
        }

        damped_features = damped(data)
        undamped_features = undamped(data)

        self.assertTrue(
            torch.allclose(
                damped_features,
                undamped_features,
                rtol=1.0e-5,
                atol=1.0e-5,
            )
        )

    def test_normalized_gaussian_scalar_is_one_for_uniform_density(self):
        module = CartesianAFeatures(
            mean_density=2.0,
            cutoff_grid=2,
            max_power=0,
            radial_basis="gaussian",
            radial_exponents=(0.125, 0.5),
            trainable_radial_exponents=True,
            separate_center=True,
        )
        n_neighbors = module.local_density_positions.shape[0]
        rho = torch.full(
            (n_neighbors, 1),
            2.0,
            dtype=module.monomial_values.dtype,
        )
        data = {
            "rho": rho,
            "local_density_index": torch.arange(n_neighbors).repeat(
                n_neighbors,
                1,
            ),
        }

        features = module(data)
        self.assertTrue(
            torch.allclose(
                features[:, :, 0, 0],
                torch.ones_like(features[:, :, 0, 0]),
            )
        )

        nonuniform_rho = rho.clone()
        nonuniform_rho[1] *= 2.0
        module(
            {
                "rho": nonuniform_rho,
                "local_density_index": data["local_density_index"],
            }
        ).sum().backward()
        self.assertIsNotNone(module.log_radial_exponents.grad)
        self.assertTrue(
            torch.all(torch.isfinite(module.log_radial_exponents.grad))
        )

    def test_coordinate_scaling_is_validated(self):
        with self.assertRaises(TypeError):
            CartesianAFeatures(
                max_power=1,
                mean_density=1.0,
                coordinate_scaling=True,
            )
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=1,
                mean_density=1.0,
                coordinate_scaling="unknown",
            )

    def test_undamped_polynomial_features(self):
        module = CartesianAFeatures(
            mean_density=2.0,
            cutoff_grid=1,
            max_power=1,
            radial_basis="none",
            separate_center=False,
        )
        self.assertTrue(
            torch.equal(module.radial_exponents, torch.zeros(1))
        )
        self.assertFalse(hasattr(module, "log_radial_exponents"))

        rho = torch.arange(
            1,
            8,
            dtype=module.monomial_values.dtype,
        ).reshape(7, 1)
        local_density_index = torch.arange(7).repeat(7, 1)
        features = module(
            {
                "rho": rho,
                "local_density_index": local_density_index,
            }
        )

        # The scalar moment averages the center and all six neighbors. The
        # vector moments retain the canonical signed stencil geometry.
        expected = torch.einsum(
            "jt,jk->kt",
            rho / module.mean_density,
            module.monomial_values,
        ) / module.local_density_positions.shape[0]
        self.assertEqual(features.shape, (7, 1, 4, 1))
        self.assertTrue(torch.allclose(features[0, 0], expected))

    def test_radial_exponent_options_are_validated(self):
        with self.assertRaises(TypeError):
            CartesianAFeatures(
                max_power=1,
                mean_density=1.0,
                radial_basis=True,
            )
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=1,
                mean_density=1.0,
                radial_basis="unknown",
            )
        with self.assertRaises(TypeError):
            CartesianAFeatures(
                max_power=1,
                mean_density=1.0,
                radial_basis="gaussian",
                radial_exponents=True,
            )
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=1,
                mean_density=1.0,
                radial_basis="gaussian",
                radial_exponents=0.1,
            )
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=1,
                mean_density=1.0,
                radial_basis="gaussian",
                radial_exponents=(),
            )
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=1,
                mean_density=1.0,
                radial_basis="gaussian",
                radial_exponents=(-0.1,),
            )
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=1,
                mean_density=1.0,
                trainable_radial_exponents=True,
            )
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=1,
                mean_density=1.0,
                n_radial_channels=2,
            )

    def test_gaussian_defaults_to_one_exponent(self):
        module = CartesianAFeatures(
            max_power=0,
            mean_density=1.0,
            radial_basis="gaussian",
        )

        self.assertTrue(
            torch.equal(
                module.radial_exponents,
                torch.tensor([0.125]),
            )
        )
        self.assertTrue(
            torch.equal(module.radial_centers, torch.zeros(1))
        )
        self.assertFalse(module.trainable_radial_exponents)
        self.assertFalse(module.trainable_radial_centers)

    def test_zero_centers_exactly_preserve_original_gaussian_basis(self):
        module = CartesianAFeatures(
            max_power=2,
            mean_density=1.0,
            cutoff_grid=3,
            radial_basis="gaussian",
            radial_exponents=(0.125, 0.5),
            separate_center=True,
        )

        old_radial_values = torch.exp(
            -module.squared_distances[:, None]
            * module.radial_exponents[None, :]
        )
        old_radial_values = (
            old_radial_values * module.neighbor_mask[:, None]
        )
        old_radial_values = old_radial_values / old_radial_values.sum(
            dim=0,
            keepdim=True,
        )
        old_basis = (
            old_radial_values[:, :, None]
            * module.monomial_values[:, None, :]
        )

        self.assertTrue(torch.equal(module.stencil_basis(), old_basis))

    def test_radial_centers_shift_gaussian_shells(self):
        module = CartesianAFeatures(
            max_power=0,
            mean_density=1.0,
            cutoff_grid=2,
            radial_basis="gaussian",
            radial_exponents=(1.0, 1.0),
            radial_centers=(0.0, 2.0),
            separate_center=False,
        )
        basis = module.stencil_basis()[:, :, 0]
        center = module.squared_distances == 0
        radius_two = module.squared_distances == 4

        self.assertGreater(basis[center, 0].mean(), basis[radius_two, 0].mean())
        self.assertLess(basis[center, 1].mean(), basis[radius_two, 1].mean())
        self.assertTrue(
            torch.equal(module.radial_centers, torch.tensor([0.0, 2.0]))
        )
        self.assertFalse(module.radial_centers.requires_grad)

    def test_explicit_exponents_reuse_configured_centers(self):
        module = CartesianAFeatures(
            max_power=1,
            mean_density=1.0,
            cutoff_grid=2,
            radial_basis="gaussian",
            radial_exponents=(0.5, 1.0),
            radial_centers=(0.25, 1.25),
        )
        override = torch.tensor((0.2, 0.8))

        implicit_centers = module.stencil_basis(override)
        explicit_centers = module.stencil_basis(
            override,
            module.radial_centers,
        )

        self.assertTrue(torch.equal(implicit_centers, explicit_centers))

    def test_radial_center_options_are_validated(self):
        common = dict(
            max_power=0,
            mean_density=1.0,
            radial_basis="gaussian",
            radial_exponents=(0.1, 0.2),
        )
        with self.assertRaises(TypeError):
            CartesianAFeatures(**common, radial_centers=True)
        with self.assertRaises(ValueError):
            CartesianAFeatures(**common, radial_centers=(0.0,))
        with self.assertRaises(ValueError):
            CartesianAFeatures(**common, radial_centers=(0.0, float("nan")))
        with self.assertRaises(ValueError):
            CartesianAFeatures(**common, radial_centers=(0.0, -0.1))
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=0,
                mean_density=1.0,
                radial_centers=(0.0,),
            )
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=0,
                mean_density=1.0,
                trainable_radial_centers=True,
            )

    def test_trainable_radial_centers_receive_gradients(self):
        module = CartesianAFeatures(
            max_power=1,
            mean_density=1.0,
            cutoff_grid=2,
            radial_basis="gaussian",
            radial_exponents=(0.2, 0.5),
            radial_centers=(0.0, 1.0),
            trainable_radial_centers=True,
            separate_center=False,
        )
        n_neighbors = module.local_density_positions.shape[0]
        rho = torch.linspace(0.1, 1.2, n_neighbors).reshape(-1, 1)
        data = {
            "rho": rho,
            "local_density_index": torch.arange(n_neighbors).repeat(
                n_neighbors,
                1,
            ),
        }

        module(data).square().sum().backward()

        self.assertIsInstance(module.learned_radial_centers, nn.Parameter)
        self.assertIs(module.radial_centers, module.learned_radial_centers)
        self.assertIsNotNone(module.learned_radial_centers.grad)
        self.assertTrue(
            torch.all(torch.isfinite(module.learned_radial_centers.grad))
        )
        self.assertGreater(module.learned_radial_centers.grad.abs().sum(), 0.0)
        self.assertIn("learned_radial_centers", module.state_dict())
    def test_old_zero_center_state_dict_loads_strictly(self):
        source = CartesianAFeatures(
            max_power=1,
            mean_density=1.0,
            cutoff_grid=2,
            radial_basis="gaussian",
            radial_exponents=(0.125, 0.5),
        )
        legacy_state = source.state_dict()
        self.assertNotIn("fixed_radial_centers", legacy_state)
        restored = CartesianAFeatures(
            max_power=1,
            mean_density=1.0,
            cutoff_grid=2,
            radial_basis="gaussian",
            radial_exponents=(0.125, 0.5),
        )

        restored.load_state_dict(legacy_state, strict=True)

        self.assertTrue(
            torch.equal(restored.radial_centers, torch.zeros(2))
        )

    def test_old_full_model_object_defaults_to_zero_centers(self):
        module = CartesianAFeatures(
            max_power=1,
            mean_density=1.0,
            cutoff_grid=2,
            radial_basis="gaussian",
            radial_exponents=(0.125,),
        )
        expected = module.stencil_basis()

        del module._buffers["fixed_radial_centers"]

        self.assertTrue(torch.equal(module.radial_centers, torch.zeros(1)))
        self.assertTrue(torch.equal(module.stencil_basis(), expected))

    def test_polynomial_center_separated_defaults(self):
        module = CartesianAFeatures(
            max_power=1,
            mean_density=1.0,
            cutoff_grid=1,
        )

        self.assertTrue(
            torch.equal(module.radial_exponents, torch.zeros(1))
        )
        self.assertFalse(module.trainable_radial_exponents)
        self.assertEqual(module.coordinate_scaling, "none")
        self.assertTrue(module.separate_center)

    def test_separate_center_removes_it_from_neighbor_moments(self):
        module = CartesianAFeatures(
            mean_density=2.0,
            cutoff_grid=1,
            max_power=1,
            separate_center=True,
        )
        rho = torch.arange(
            1,
            8,
            dtype=module.monomial_values.dtype,
        ).reshape(7, 1)
        local_density_index = torch.arange(7).repeat(7, 1)
        reference = module(
            {
                "rho": rho,
                "local_density_index": local_density_index,
            }
        )
        changed_center = rho.clone()
        changed_center[0] = 1000.0
        changed = module(
            {
                "rho": changed_center,
                "local_density_index": local_density_index,
            }
        )

        self.assertTrue(module.separate_center)
        self.assertFalse(module.neighbor_mask[0].item())
        self.assertTrue(torch.allclose(reference, changed))
        self.assertEqual(int(module.neighbor_mask.sum()), 6)

    def test_center_flag_does_not_break_legacy_checkpoints_or_models(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=0,
            separate_center=False,
        )
        state_keys = set(module.state_dict())
        self.assertNotIn("neighbor_mask", state_keys)

        rho = torch.arange(1.0, 8.0).reshape(7, 1)
        data = {
            "rho": rho,
            "local_density_index": torch.arange(7).repeat(7, 1),
        }
        expected = module(data)

        # Mimic an older torch.save(model) object, which has neither the flag
        # nor its deterministic center mask.
        del module.separate_center
        del module._buffers["neighbor_mask"]
        actual = module(data)
        self.assertTrue(torch.allclose(actual, expected))

    def test_trainable_channel_matches_normalized_damping_formula(self):
        module = CartesianAFeatures(
            mean_density=0.7,
            cutoff_grid=3,
            max_power=3,
            radial_basis="gaussian",
            radial_exponents=(0.125,),
            trainable_radial_exponents=True,
            coordinate_scaling="none",
            separate_center=True,
        )
        n_neighbors = module.local_density_positions.shape[0]
        rho = torch.linspace(
            0.1,
            1.3,
            steps=n_neighbors,
            dtype=module.monomial_values.dtype,
        ).reshape(n_neighbors, 1)
        data = {
            "rho": rho,
            "local_density_index": torch.arange(n_neighbors).repeat(
                n_neighbors,
                1,
            ),
        }

        actual = module(data)
        actual_gradient = torch.autograd.grad(
            actual.square().sum(),
            module.log_radial_exponents,
        )[0]

        # Independent transcription of the normalized one-channel expression:
        # raw integer monomials, excluded center, and alpha optimized in
        # logarithmic form.
        reference_log_alpha = torch.tensor(
            math.log(0.125),
            dtype=module.monomial_values.dtype,
            requires_grad=True,
        )
        reference_weights = torch.exp(
            -module.squared_distances * torch.exp(reference_log_alpha)
        )
        reference_weights = reference_weights * module.neighbor_mask
        reference_weights = reference_weights / reference_weights.sum()
        reference_basis = (
            reference_weights[:, None] * module.monomial_values
        )
        reference = torch.einsum(
            "jt,jk->kt",
            rho / module.mean_density,
            reference_basis,
        )[None, None, :, :].expand(n_neighbors, -1, -1, -1)
        reference_gradient = torch.autograd.grad(
            reference.square().sum(),
            reference_log_alpha,
        )[0]

        self.assertEqual(actual.shape, (n_neighbors, 1, 20, 1))
        self.assertTrue(torch.allclose(actual, reference))
        self.assertTrue(
            torch.allclose(actual_gradient[0], reference_gradient)
        )

    def test_mean_density_must_be_positive_scalar(self):
        with self.assertRaises(ValueError):
            CartesianAFeatures(max_power=0, mean_density=0.0)
        with self.assertRaises(ValueError):
            CartesianAFeatures(max_power=0, mean_density=[1.0, 2.0])

    def test_trainable_radial_exponents_receive_gradients(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=1,
            radial_basis="gaussian",
            radial_exponents=(0.05, 0.2),
            trainable_radial_exponents=True,
        )
        self.assertIsInstance(module.log_radial_exponents, nn.Parameter)
        self.assertTrue(
            torch.allclose(
                module.radial_exponents.detach(),
                torch.tensor([0.05, 0.2]),
            )
        )

        rho = torch.arange(
            1,
            8,
            dtype=module.monomial_values.dtype,
        ).reshape(7, 1)
        local_density_index = torch.arange(7).repeat(7, 1)
        features = module(
            {
                "rho": rho,
                "local_density_index": local_density_index,
            }
        )
        features.sum().backward()

        self.assertIsNotNone(module.log_radial_exponents.grad)
        self.assertTrue(
            torch.all(torch.isfinite(module.log_radial_exponents.grad))
        )

    def test_zero_radial_exponent_is_undamped(self):
        module = CartesianAFeatures(
            max_power=0,
            mean_density=1.0,
            radial_basis="gaussian",
            radial_exponents=(0.0,),
        )
        self.assertTrue(
            torch.equal(module.radial_exponents, torch.zeros(1))
        )

    def test_radial_exponent_must_be_finite(self):
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=0,
                mean_density=1.0,
                radial_basis="gaussian",
                radial_exponents=(float("nan"),),
            )


class TestCartesianAFeatureBesselBasis(unittest.TestCase):
    def test_new_options_do_not_change_legacy_state_keys(self):
        expected = {
            "fixed_radial_exponents",
            "local_density_positions",
            "squared_distances",
            "powers",
            "monomial_values",
            "mean_density",
        }
        undamped = CartesianAFeatures(
            max_power=2,
            mean_density=1.0,
            radial_basis="none",
            n_radial_channels=1,
        )
        gaussian = CartesianAFeatures(
            max_power=2,
            mean_density=1.0,
            radial_basis="gaussian",
            radial_exponents=(0.125, 0.5),
        )

        self.assertEqual(set(undamped.state_dict()), expected)
        self.assertEqual(set(gaussian.state_dict()), expected)

    def test_bessel_transform_shapes_and_identity_initialization(self):
        untransformed = CartesianAFeatures(
            max_power=3,
            mean_density=0.2,
            cutoff_grid=6,
            radial_basis="bessel",
            n_radial_functions=6,
            n_types=2,
        )
        transformed = CartesianAFeatures(
            max_power=3,
            mean_density=0.2,
            cutoff_grid=6,
            radial_basis="bessel",
            n_radial_functions=6,
            n_radial_channels=4,
            n_types=2,
        )

        self.assertEqual(untransformed.n_radial_functions, 6)
        self.assertEqual(untransformed.n_radial_channels, 6)
        self.assertEqual(transformed.n_radial_functions, 6)
        self.assertEqual(transformed.n_radial_channels, 4)
        self.assertEqual(transformed.stencil_basis().shape, (925, 4, 20))
        self.assertTrue(
            torch.equal(
                transformed.stencil_basis(),
                untransformed.stencil_basis()[:, :4],
            )
        )
        self.assertEqual(
            transformed.radial_transform.weight.numel(),
            96,
        )

    def test_folded_transform_matches_explicit_A_transform(self):
        module = CartesianAFeatures(
            max_power=2,
            mean_density=0.7,
            cutoff_grid=3,
            radial_basis="bessel",
            n_radial_functions=3,
            n_radial_channels=2,
            separate_center=True,
        )
        with torch.no_grad():
            module.radial_transform.weight.add_(
                0.1 * torch.randn_like(module.radial_transform.weight)
            )
        n_grid = module.local_density_positions.shape[0]
        local_density = torch.randn(n_grid, n_grid, 1)
        primitive_A = torch.einsum(
            "gjt,jnk->gnkt",
            local_density / module.mean_density,
            module.fixed_bessel_stencil_basis,
        )
        weights = module.radial_transform.weight.index_select(
            0,
            module.powers.sum(dim=-1),
        )
        explicit = torch.einsum("gnkt,knm->gmkt", primitive_A, weights)
        folded = torch.einsum(
            "gjt,jmk->gmkt",
            local_density / module.mean_density,
            module.stencil_basis(),
        )

        self.assertTrue(torch.allclose(folded, explicit, atol=1.0e-6))

    def test_gaussian_can_use_the_same_pre_invariant_transform(self):
        untransformed = CartesianAFeatures(
            max_power=2,
            mean_density=1.0,
            cutoff_grid=2,
            radial_basis="gaussian",
            radial_exponents=(0.125, 0.5),
        )
        transformed = CartesianAFeatures(
            max_power=2,
            mean_density=1.0,
            cutoff_grid=2,
            radial_basis="gaussian",
            radial_exponents=(0.125, 0.5),
            n_radial_channels=1,
        )

        self.assertEqual(transformed.n_radial_functions, 2)
        self.assertEqual(transformed.n_radial_channels, 1)
        self.assertTrue(
            torch.equal(
                transformed.stencil_basis(),
                untransformed.stencil_basis()[:, :1],
            )
        )

    def test_new_bessel_state_loads_strictly(self):
        source = CartesianAFeatures(
            max_power=2,
            mean_density=1.0,
            cutoff_grid=3,
            radial_basis="bessel",
            n_radial_functions=3,
            n_radial_channels=2,
        )
        with torch.no_grad():
            source.radial_transform.weight.add_(0.2)
        restored = CartesianAFeatures(
            max_power=2,
            mean_density=1.0,
            cutoff_grid=3,
            radial_basis="bessel",
            n_radial_functions=3,
            n_radial_channels=2,
        )

        restored.load_state_dict(source.state_dict(), strict=True)

        self.assertTrue(
            torch.equal(restored.stencil_basis(), source.stencil_basis())
        )
        self.assertIn("fixed_bessel_stencil_basis", source.state_dict())
        self.assertIn("bessel_gram_eigenvalues", source.state_dict())

    def test_old_full_model_feature_object_needs_no_migration(self):
        module = CartesianAFeatures(
            max_power=1,
            mean_density=1.0,
            cutoff_grid=2,
            radial_basis="gaussian",
            radial_exponents=(0.125,),
        )
        expected = module.stencil_basis()

        del module.n_radial_functions
        del module.radial_transform

        self.assertTrue(torch.equal(module.stencil_basis(), expected))

    def test_bessel_arguments_are_validated(self):
        common = dict(
            max_power=2,
            mean_density=1.0,
            cutoff_grid=3,
            radial_basis="bessel",
        )
        with self.assertRaisesRegex(ValueError, "required"):
            CartesianAFeatures(**common)
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                **common,
                n_radial_functions=2,
                radial_exponents=(0.1,),
            )
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                **common,
                n_radial_functions=2,
                radial_centers=(0.0, 1.0),
            )
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                **common,
                n_radial_functions=2,
                trainable_radial_exponents=True,
            )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            CartesianAFeatures(
                **common,
                n_radial_functions=2,
                n_radial_channels=3,
            )
        with self.assertRaises(ValueError):
            CartesianAFeatures(
                max_power=0,
                mean_density=1.0,
                cutoff_grid=0,
                radial_basis="bessel",
                n_radial_functions=1,
            )


class TestCartesianAFeatureDensityTransform(unittest.TestCase):
    @staticmethod
    def _single_grid_data(rho):
        return {
            "rho": rho,
            "local_density_index": torch.zeros((1, 1), dtype=torch.long),
        }

    def test_fixed_transform_replaces_physical_density_channels(self):
        weights = ((0.5, 0.5), (-0.5, 0.5))
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=0,
            separate_center=False,
            n_types=2,
            density_transform=weights,
            trainable_density_transform=False,
        )
        rho = torch.tensor([[1.0, 3.0]], requires_grad=True)

        features = module(self._single_grid_data(rho))

        self.assertEqual(features.shape, (1, 1, 1, 2))
        self.assertTrue(
            torch.allclose(features, torch.tensor([[[[2.0, 1.0]]]]))
        )
        self.assertTrue(
            torch.allclose(
                module.transform_density(rho),
                torch.tensor([[2.0, 1.0]]),
            )
        )
        self.assertFalse(module.density_transform.weight.requires_grad)
        self.assertEqual(module.n_channels, 2)
        self.assertEqual(module.n_output_channels, 2)

        number_gradient = torch.autograd.grad(
            features[..., 0].sum(), rho, retain_graph=True
        )[0]
        charge_gradient = torch.autograd.grad(features[..., 1].sum(), rho)[0]
        self.assertTrue(
            torch.allclose(number_gradient, torch.tensor([[0.5, 0.5]]))
        )
        self.assertTrue(
            torch.allclose(charge_gradient, torch.tensor([[-0.5, 0.5]]))
        )

    def test_transform_is_applied_before_cartesian_moments(self):
        module = CartesianAFeatures(
            mean_density=0.7,
            cutoff_grid=1,
            max_power=2,
            n_types=2,
            density_transform=((1.0, 1.0), (-1.0, 1.0)),
            trainable_density_transform=False,
        )
        n_grid = module.local_density_positions.shape[0]
        rho = torch.linspace(0.2, 1.6, 2 * n_grid).reshape(n_grid, 2)
        index = torch.arange(n_grid).repeat(n_grid, 1)

        actual = module({"rho": rho, "local_density_index": index})
        transformed_density = module.transform_density(rho)
        local_density = transformed_density[index]
        expected = torch.einsum(
            "gjt,jnk->gnkt",
            local_density / module.mean_density,
            module.stencil_basis(),
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_explicit_transform_can_be_trainable(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=0,
            separate_center=False,
            n_types=2,
            density_transform=((0.5, 0.5), (-0.5, 0.5)),
            trainable_density_transform=True,
        )
        rho = torch.tensor([[1.0, 3.0]], requires_grad=True)

        module(self._single_grid_data(rho)).square().sum().backward()

        self.assertTrue(module.density_transform.weight.requires_grad)
        self.assertIsNotNone(module.density_transform.weight.grad)
        self.assertIsNotNone(rho.grad)
        self.assertTrue(torch.all(torch.isfinite(rho.grad)))
        self.assertTrue(
            torch.all(torch.isfinite(module.density_transform.weight.grad))
        )

    def test_n_channels_retains_random_trainable_transform(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=0,
            n_types=2,
            n_channels=3,
        )

        self.assertEqual(module.density_transform.weight.shape, (3, 2))
        self.assertTrue(module.density_transform.weight.requires_grad)
        self.assertEqual(module.n_output_channels, 3)

    def test_transform_defaults_to_physical_density_channels(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=0,
            n_types=2,
        )
        rho = torch.tensor([[1.0, 3.0]])

        self.assertIsNone(module.density_transform)
        self.assertIs(module.transform_density(rho), rho)
        self.assertEqual(module.n_output_channels, 2)

        del module.density_transform
        self.assertIs(module.transform_density(rho), rho)

    def test_transform_arguments_are_validated(self):
        invalid_weights = (
            (0.5, 0.5),
            ((0.5,),),
            ((),),
            ((0.5, float("nan")),),
        )
        for weights in invalid_weights:
            with self.subTest(weights=weights):
                with self.assertRaises((TypeError, ValueError)):
                    CartesianAFeatures(
                        mean_density=1.0,
                        max_power=0,
                        n_types=2,
                        density_transform=weights,
                    )
        with self.assertRaisesRegex(ValueError, "number of rows"):
            CartesianAFeatures(
                mean_density=1.0,
                max_power=0,
                n_types=2,
                n_channels=3,
                density_transform=((0.5, 0.5), (-0.5, 0.5)),
            )
        with self.assertRaises(TypeError):
            CartesianAFeatures(
                mean_density=1.0,
                max_power=0,
                n_types=2,
                n_channels=2,
                trainable_density_transform="yes",
            )

    def test_one_component_disables_density_transform(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=0,
            n_types=1,
        )
        self.assertIsNone(module.density_transform)
        self.assertEqual(module.n_output_channels, 1)

        with self.assertRaises(ValueError):
            CartesianAFeatures(
                mean_density=1.0,
                cutoff_grid=0,
                max_power=0,
                n_types=1,
                density_transform=((1.0,),),
            )


if __name__ == "__main__":
    unittest.main()
