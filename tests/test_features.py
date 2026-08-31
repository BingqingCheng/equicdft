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
class TestCartesianAFeatureChannelMixing(unittest.TestCase):
    def test_known_linear_map_and_shape(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=0,
            separate_center=False,
            n_types=2,
            n_channels=3,
        )
        with torch.no_grad():
            module.channel_mixing.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]],
                    dtype=module.monomial_values.dtype,
                )
            )

        rho = torch.tensor(
            [[1.0, 3.0]],
            dtype=module.monomial_values.dtype,
        )
        mixed = module(
            {
                "rho": rho,
                "local_density_index": torch.zeros(
                    (1, 1),
                    dtype=torch.long,
                ),
            }
        )

        self.assertEqual(mixed.shape, (1, 1, 1, 3))
        self.assertTrue(
            torch.allclose(
                mixed,
                torch.tensor(
                    [[[[1.0, 3.0, -1.0]]]],
                    dtype=mixed.dtype,
                ),
            )
        )
        self.assertEqual(module.n_output_channels, 3)

    def test_mixing_commutes_with_cubic_symmetry(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=2,
            n_types=2,
            n_channels=4,
        )
        symmetrizer = CartesianBFeatures(
            max_power=2,
            max_product_order=3,
        )
        A = torch.randn(
            2,
            3,
            10,
            2,
            dtype=module.monomial_values.dtype,
        )

        mixed = module.channel_mixing(A)
        reference = symmetrizer(mixed)
        for index_map, sign_map in zip(
            symmetrizer.component_indices,
            symmetrizer.component_signs,
        ):
            sign_shape = [1] * A.ndim
            sign_shape[-2] = sign_map.shape[0]
            transformed = (
                A.index_select(-2, index_map) * sign_map.view(sign_shape)
            )

            # Mixing physical channels commutes with the signed permutation of
            # Cartesian components, and the subsequent B features are invariant.
            transformed_mixed = module.channel_mixing(transformed)
            expected_mixed = (
                mixed.index_select(-2, index_map) * sign_map.view(sign_shape)
            )
            self.assertTrue(
                torch.allclose(transformed_mixed, expected_mixed, atol=1.0e-6)
            )
            self.assertTrue(
                torch.allclose(
                    symmetrizer(transformed_mixed),
                    reference,
                    atol=1.0e-6,
                )
            )

    def test_inputs_and_weights_receive_gradients(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=2,
            n_types=2,
            n_channels=3,
        )
        A = torch.randn(
            2,
            4,
            10,
            2,
            dtype=module.monomial_values.dtype,
            requires_grad=True,
        )
        module.channel_mixing(A).square().sum().backward()

        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(module.channel_mixing.weight.grad)
        self.assertTrue(torch.all(torch.isfinite(A.grad)))
        self.assertTrue(
            torch.all(torch.isfinite(module.channel_mixing.weight.grad))
        )

    def test_one_component_disables_channel_mixing(self):
        module = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=0,
            max_power=0,
            n_types=1,
        )
        self.assertIsNone(module.channel_mixing)
        self.assertEqual(module.n_output_channels, 1)

        with self.assertRaises(ValueError):
            CartesianAFeatures(
                mean_density=1.0,
                cutoff_grid=0,
                max_power=0,
                n_types=1,
                n_channels=2,
            )


if __name__ == "__main__":
    unittest.main()
