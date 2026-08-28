import unittest

import numpy as np
import torch

from equicdft import (
    BChiMessage,
    CartesianAFeatures,
    CartesianBFeatures,
    GridCACEModel,
    LocalReadout,
)
from equicdft.stencil import get_neighbor_indices


def _grid_data(shape=(5, 5, 5), cutoff_grid=1):
    positions = np.indices(shape, dtype=int).reshape(3, -1).T
    neighbor_indices, _ = get_neighbor_indices(
        positions,
        cutoff_grid=cutoff_grid,
    )
    return {
        "rho": torch.rand(int(np.prod(shape)), 1) + 0.1,
        "local_density_index": torch.tensor(
            neighbor_indices,
            dtype=torch.long,
        ),
        "grid_spacing": torch.ones(3),
        "grid_size": torch.tensor(shape),
        "temperature": torch.tensor(1.5),
        "beta": torch.tensor(1.0 / 1.5),
    }


def _transform_cubic_field(values, shape):
    """Apply one axis permutation and reflection to a cubic scalar field."""

    n_feature_dims = values.ndim - 1
    reshaped = values.reshape(*shape, *values.shape[1:])
    feature_axes = tuple(range(3, 3 + n_feature_dims))
    transformed = reshaped.permute(2, 0, 1, *feature_axes).flip(1)
    return transformed.reshape(values.shape)


class TestBChiMessage(unittest.TestCase):
    def test_zero_invariant_field_gives_exactly_zero_message(self):
        data = _grid_data(shape=(3, 3, 3))
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=2,
            separate_center=False,
        )
        b_features = CartesianBFeatures(
            max_power=2,
            max_product_order=2,
        )
        message = BChiMessage(
            n_invariant_features=b_features.n_features,
            n_radial_channels=a_features.n_radial_channels,
            n_channels=a_features.n_output_channels,
        )
        B = torch.zeros(
            data["rho"].shape[0],
            a_features.n_radial_channels,
            b_features.n_features,
            a_features.n_output_channels,
        )

        A_next = message(
            B,
            data["local_density_index"],
            a_features.stencil_basis(),
        )

        self.assertTrue(torch.equal(A_next, torch.zeros_like(A_next)))

    def test_linear_gate_matches_direct_periodic_convolution(self):
        data = _grid_data(shape=(3, 3, 3))
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=0,
            separate_center=False,
        )
        message = BChiMessage(
            n_invariant_features=1,
            n_radial_channels=1,
            n_channels=1,
            hidden_sizes=(),
        )
        with torch.no_grad():
            message.mlp[0].weight.fill_(2.0)
            message.mlp[0].bias.fill_(0.7)
        B = torch.arange(27, dtype=torch.get_default_dtype()).reshape(
            27, 1, 1, 1
        )

        A_next = message(
            B,
            data["local_density_index"],
            a_features.stencil_basis(),
        )
        neighbor_values = B[:, 0, 0, 0][data["local_density_index"]]
        weights = a_features.stencil_basis()[:, 0, 0]
        expected = 2.0 * torch.sum(neighbor_values * weights, dim=-1)

        self.assertTrue(torch.allclose(A_next[:, 0, 0, 0], expected))

    def test_message_extends_receptive_field_by_one_stencil(self):
        shape = (7, 7, 7)
        data = _grid_data(shape=shape)
        data["rho"].zero_()
        source = np.ravel_multi_index((3, 3, 3), shape)
        target = np.ravel_multi_index((5, 3, 3), shape)
        data["rho"][source, 0] = 1.0
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=0,
            separate_center=False,
        )
        b_features = CartesianBFeatures(
            max_power=0,
            max_product_order=1,
        )
        message = BChiMessage(1, 1, 1, hidden_sizes=())
        with torch.no_grad():
            message.mlp[0].weight.fill_(1.0)
            message.mlp[0].bias.zero_()

        B0 = b_features(a_features(data))
        B1 = b_features(
            message(
                B0,
                data["local_density_index"],
                a_features.stencil_basis(),
            )
        )

        self.assertEqual(B0[target, 0, 0, 0].item(), 0.0)
        self.assertGreater(B1[target, 0, 0, 0].item(), 0.0)

    def test_message_uses_live_trainable_radial_exponent(self):
        data = _grid_data(shape=(3, 3, 3))
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=1,
            radial_basis="gaussian",
            radial_exponents=(0.5,),
            trainable_radial_exponents=True,
            separate_center=False,
        )
        message = BChiMessage(1, 1, 1, hidden_sizes=())
        with torch.no_grad():
            message.mlp[0].weight.fill_(1.0)
            message.mlp[0].bias.zero_()
        B = torch.arange(27, dtype=torch.get_default_dtype()).reshape(
            27, 1, 1, 1
        )

        A_next = message(
            B,
            data["local_density_index"],
            a_features.stencil_basis(),
        )
        derivative = torch.autograd.grad(
            A_next[0, 0, 1, 0],
            a_features.log_radial_exponents,
        )[0]

        self.assertTrue(torch.isfinite(derivative).all())
        self.assertGreater(derivative.abs().item(), 0.0)

    def test_message_can_own_an_independent_radial_exponent(self):
        data = _grid_data(shape=(3, 3, 3))
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=1,
            radial_basis="gaussian",
            radial_exponents=(0.25,),
            trainable_radial_exponents=True,
            separate_center=False,
        )
        message = BChiMessage(
            1,
            1,
            1,
            hidden_sizes=(),
            radial_exponents=(1.0,),
            trainable_radial_exponents=True,
        )
        with torch.no_grad():
            message.mlp[0].weight.fill_(1.0)
            message.mlp[0].bias.zero_()
        B = torch.arange(27, dtype=torch.get_default_dtype()).reshape(
            27, 1, 1, 1
        )

        message_basis = a_features.stencil_basis(message.radial_exponents)
        A_next = message(B, data["local_density_index"], message_basis)
        derivative = torch.autograd.grad(
            A_next[0, 0, 1, 0],
            message.log_radial_exponents,
        )[0]

        self.assertFalse(
            torch.allclose(message_basis, a_features.stencil_basis())
        )
        self.assertTrue(torch.isfinite(derivative).all())
        self.assertGreater(derivative.abs().item(), 0.0)

    def test_trainable_message_radial_requires_initial_values(self):
        with self.assertRaisesRegex(ValueError, "requires radial_exponents"):
            BChiMessage(
                1,
                1,
                1,
                trainable_radial_exponents=True,
            )

    def test_invariant_output_commutes_with_cubic_grid_symmetry(self):
        torch.manual_seed(7)
        shape = (5, 5, 5)
        data = _grid_data(shape=shape)
        a_features = CartesianAFeatures(
            mean_density=0.7,
            cutoff_grid=1,
            max_power=2,
            radial_basis="gaussian",
            radial_exponents=(0.25, 1.0),
            separate_center=False,
        )
        b_features = CartesianBFeatures(
            max_power=2,
            max_product_order=2,
        )
        message = BChiMessage(
            b_features.n_features,
            a_features.n_radial_channels,
            a_features.n_output_channels,
            hidden_sizes=(5,),
        )

        B0 = b_features(a_features(data))
        B1 = b_features(
            message(
                B0,
                data["local_density_index"],
                a_features.stencil_basis(),
            )
        )
        transformed_data = dict(data)
        transformed_data["rho"] = _transform_cubic_field(data["rho"], shape)
        transformed_B0 = b_features(a_features(transformed_data))
        transformed_B1 = b_features(
            message(
                transformed_B0,
                transformed_data["local_density_index"],
                a_features.stencil_basis(),
            )
        )

        self.assertTrue(
            torch.allclose(
                transformed_B0,
                _transform_cubic_field(B0, shape),
                rtol=2.0e-5,
                atol=2.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                transformed_B1,
                _transform_cubic_field(B1, shape),
                rtol=2.0e-5,
                atol=2.0e-6,
            )
        )

    def test_batched_message_matches_individual_fields(self):
        data = _grid_data(shape=(3, 3, 3))
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=1,
            separate_center=False,
        )
        b_features = CartesianBFeatures(1, 2)
        message = BChiMessage(b_features.n_features, 1, 1)
        B = torch.rand(2, 27, 1, b_features.n_features, 1)
        indices = data["local_density_index"].repeat(2, 1, 1)

        batched = message(B, indices, a_features.stencil_basis())
        separate = torch.stack(
            [
                message(
                    B[index],
                    data["local_density_index"],
                    a_features.stencil_basis(),
                )
                for index in range(2)
            ]
        )

        self.assertTrue(torch.allclose(batched, separate))


class TestMessagePassingModel(unittest.TestCase):
    def _make_model(self, message_layers, compute_c2=False):
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=1,
            max_power=2,
            radial_basis="gaussian",
            radial_exponents=(0.25,),
            trainable_radial_exponents=True,
            separate_center=True,
        )
        b_features = CartesianBFeatures(2, 2)
        return GridCACEModel(
            a_features=a_features,
            b_features=b_features,
            readout=[LocalReadout(n_types=1, hidden_sizes=(8,))],
            grid_spacing=1.0,
            compute_c1=True,
            compute_c2=compute_c2,
            message_layers=message_layers(a_features, b_features),
        )

    def test_one_layer_concatenates_both_invariant_levels(self):
        def messages(a_features, b_features):
            return [
                BChiMessage(
                    b_features.n_features,
                    a_features.n_radial_channels,
                    a_features.n_output_channels,
                    hidden_sizes=(6,),
                )
            ]

        model = self._make_model(messages)
        data = _grid_data(shape=(3, 3, 3))

        outputs = model(data)

        n_B = model.b_features.n_features
        expected_width = 1 + 2 * n_B + 1
        self.assertEqual(model.readout[0].mlp[0].in_features, expected_width)
        self.assertEqual(outputs["c1"].shape, data["rho"].shape)
        outputs["c1"].square().mean().backward()
        message_gradient = model.message_layers[0].mlp[0].weight.grad
        self.assertIsNotNone(message_gradient)
        self.assertTrue(torch.all(torch.isfinite(message_gradient)))
        self.assertIsNotNone(model.a_features.log_radial_exponents.grad)

    def test_selected_c2_is_finite_with_message_passing(self):
        def messages(a_features, b_features):
            return [
                BChiMessage(
                    b_features.n_features,
                    a_features.n_radial_channels,
                    a_features.n_output_channels,
                    hidden_sizes=(6,),
                )
            ]

        model = self._make_model(messages, compute_c2=True)
        data = _grid_data(shape=(3, 3, 3))

        outputs = model(data, c2_reference=(0, 0))

        self.assertEqual(outputs["c2"].shape, data["rho"].shape)
        self.assertTrue(torch.all(torch.isfinite(outputs["c2"])))

    def test_two_layers_have_independent_weights_and_radial_exponents(self):
        def messages(a_features, b_features):
            return [
                BChiMessage(
                    b_features.n_features,
                    a_features.n_radial_channels,
                    a_features.n_output_channels,
                    hidden_sizes=(6,),
                    radial_exponents=(exponent,),
                    trainable_radial_exponents=True,
                )
                for exponent in (0.5, 1.0)
            ]

        model = self._make_model(messages)
        data = _grid_data(shape=(3, 3, 3))
        outputs = model(data)
        outputs["c1"].square().mean().backward()

        first, second = model.message_layers
        self.assertIsNot(first.mlp[0].weight, second.mlp[0].weight)
        self.assertNotEqual(
            first.mlp[0].weight.data_ptr(),
            second.mlp[0].weight.data_ptr(),
        )
        self.assertIsNot(
            first.log_radial_exponents,
            second.log_radial_exponents,
        )
        self.assertIsNotNone(first.log_radial_exponents.grad)
        self.assertIsNotNone(second.log_radial_exponents.grad)
        self.assertGreater(first.log_radial_exponents.grad.abs().item(), 0.0)
        self.assertGreater(second.log_radial_exponents.grad.abs().item(), 0.0)
        self.assertIsNotNone(model.a_features.log_radial_exponents.grad)

    def test_empty_message_list_preserves_default_state_and_output(self):
        no_argument = self._make_model(lambda _a, _b: None)
        explicit_empty = self._make_model(lambda _a, _b: [])
        data_a = _grid_data(shape=(3, 3, 3))
        data_b = {key: value.clone() for key, value in data_a.items()}

        no_argument(data_a)
        explicit_empty(data_b)
        explicit_empty.load_state_dict(no_argument.state_dict())

        output_a = no_argument(data_a)["beta_F_exc"]
        output_b = explicit_empty(data_b)["beta_F_exc"]
        self.assertTrue(torch.equal(output_a, output_b))
        self.assertEqual(len(no_argument.message_layers), 0)
        self.assertEqual(len(explicit_empty.message_layers), 0)

    def test_mismatched_message_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            self._make_model(
                lambda _a, b: [
                    BChiMessage(
                        n_invariant_features=b.n_features + 1,
                        n_radial_channels=1,
                        n_channels=1,
                    )
                ]
            )


if __name__ == "__main__":
    unittest.main()
