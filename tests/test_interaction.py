import io
import unittest
from unittest import mock

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
        "grid_positions": torch.tensor(positions, dtype=torch.long),
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


def _load_whole_model(serialized):
    """Load a test model across pre- and post-weights-only PyTorch APIs."""

    try:
        return torch.load(serialized, weights_only=False)
    except TypeError:
        return torch.load(serialized)


class TestBChiMessage(unittest.TestCase):
    def test_feature_parameters_include_only_owned_radial_state(self):
        independent = BChiMessage(
            1,
            2,
            1,
            hidden_sizes=(),
            radial_exponents=(0.25, 0.5),
            trainable_radial_exponents=True,
            radial_centers=(0.0, 1.0),
            trainable_radial_centers=True,
        )
        shared = BChiMessage(1, 2, 1, hidden_sizes=())

        self.assertEqual(
            set(independent.feature_parameters),
            {"log_radial_exponents", "learned_radial_centers"},
        )
        self.assertEqual(shared.feature_parameters, {})

        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=2,
            max_power=1,
            radial_basis="gaussian",
            radial_exponents=(0.25, 0.5),
        )
        b_features = CartesianBFeatures(1, 2)
        bessel = BChiMessage(
            b_features.n_features,
            2,
            1,
            hidden_sizes=(),
            radial_basis="bessel",
            n_radial_functions=3,
        )
        GridCACEModel(
            a_features,
            b_features,
            [LocalReadout(n_types=1, hidden_sizes=(4,))],
            grid_spacing=1.0,
            message_layers=[bessel],
        )
        self.assertEqual(
            set(bessel.feature_parameters),
            {"radial_transform.weight"},
        )

    def test_convolution_backend_is_validated_without_state_keys(self):
        gather = BChiMessage(1, 1, 1, hidden_sizes=())
        convolution = BChiMessage(
            1,
            1,
            1,
            hidden_sizes=(),
            convolution_backend="conv3d",
        )
        fft = BChiMessage(
            1,
            1,
            1,
            hidden_sizes=(),
            convolution_backend="fft",
        )

        self.assertEqual(gather.convolution_backend, "gather")
        self.assertEqual(convolution.convolution_backend, "conv3d")
        self.assertEqual(fft.convolution_backend, "fft")
        self.assertEqual(set(gather.state_dict()), set(convolution.state_dict()))
        self.assertEqual(set(gather.state_dict()), set(fft.state_dict()))
        with self.assertRaisesRegex(ValueError, "gather.*conv3d.*fft"):
            BChiMessage(1, 1, 1, convolution_backend="invalid")

    def test_only_gather_backend_requires_neighborhood_index(self):
        data = _grid_data(shape=(3, 4, 5), cutoff_grid=1)
        positions = data["grid_positions"].numpy()
        _, stencil_positions = get_neighbor_indices(positions, cutoff_grid=1)
        stencil_positions = torch.tensor(stencil_positions, dtype=torch.long)
        stencil_basis = torch.randn(stencil_positions.shape[0], 1, 4)
        B = torch.randn(60, 1, 1, 1)

        gather = BChiMessage(1, 1, 1, hidden_sizes=())
        with self.assertRaisesRegex(
            ValueError,
            "gather messages require local_density_index",
        ):
            gather(B, None, stencil_basis)

        for backend in ("conv3d", "fft"):
            with self.subTest(backend=backend):
                message = BChiMessage(
                    1,
                    1,
                    1,
                    hidden_sizes=(),
                    convolution_backend=backend,
                )
                arguments = {
                    "grid_positions": data["grid_positions"],
                    "grid_size": data["grid_size"],
                    "stencil_positions": stencil_positions,
                }
                expected = message(
                    B,
                    data["local_density_index"],
                    stencil_basis,
                    **arguments,
                )
                actual = message(B, None, stencil_basis, **arguments)
                self.assertTrue(torch.equal(actual, expected))

    def test_convolution_matches_gather_without_neighbor_materialization(self):
        torch.manual_seed(31)
        shape = (4, 5, 6)
        data = _grid_data(shape=shape, cutoff_grid=1)
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=2,
            radial_basis="gaussian",
            radial_exponents=(0.25, 1.0),
            separate_center=False,
        ).double()
        gather = BChiMessage(3, 2, 2, hidden_sizes=(5,)).double()
        B = torch.randn(
            np.prod(shape),
            2,
            3,
            2,
            dtype=torch.float64,
        )
        basis = a_features.stencil_basis()

        expected = gather(
            B,
            data["local_density_index"],
            basis,
        )
        for backend in ("conv3d", "fft"):
            with self.subTest(backend=backend):
                convolution = BChiMessage(
                    3,
                    2,
                    2,
                    hidden_sizes=(5,),
                    convolution_backend=backend,
                ).double()
                convolution.load_state_dict(gather.state_dict(), strict=True)
                with mock.patch(
                    "equicdft.interaction.gather_neighbors",
                    side_effect=AssertionError("gather backend was reached"),
                ):
                    actual = convolution(
                        B,
                        data["local_density_index"],
                        basis,
                        grid_positions=data["grid_positions"],
                        grid_size=data["grid_size"],
                        stencil_positions=a_features.local_density_positions,
                    )
                self.assertTrue(
                    torch.allclose(
                        actual,
                        expected,
                        rtol=1.0e-12,
                        atol=1.0e-12,
                    )
                )

    def test_convolution_requires_geometry_arguments(self):
        B = torch.zeros(8, 1, 1, 1)
        indices = torch.zeros(8, 1, dtype=torch.long)
        basis = torch.ones(1, 1, 1)
        geometry = {
            "grid_positions": torch.tensor(
                np.indices((2, 2, 2), dtype=int).reshape(3, -1).T
            ),
            "grid_size": torch.tensor([2, 2, 2]),
            "stencil_positions": torch.zeros(1, 3, dtype=torch.long),
        }

        for backend in ("conv3d", "fft"):
            message = BChiMessage(
                1,
                1,
                1,
                hidden_sizes=(),
                convolution_backend=backend,
            )
            for missing in geometry:
                with self.subTest(backend=backend, missing=missing):
                    incomplete = geometry.copy()
                    incomplete[missing] = None
                    with self.assertRaisesRegex(
                        ValueError,
                        "grid_positions, grid_size, and stencil_positions",
                    ):
                        message(B, indices, basis, **incomplete)

    def test_zero_invariant_field_gives_numerically_zero_message(self):
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

        self.assertTrue(
            torch.allclose(
                A_next,
                torch.zeros_like(A_next),
                rtol=0.0,
                atol=2.0e-8,
            )
        )

    def test_zero_invariant_field_retains_gate_jacobian(self):
        message = BChiMessage(1, 1, 1, hidden_sizes=())
        with torch.no_grad():
            message.mlp[0].weight.fill_(2.0)
            message.mlp[0].bias.fill_(0.7)
        B = torch.zeros(1, 1, 1, 1, requires_grad=True)

        output = message(
            B,
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, 1),
        )
        derivative = torch.autograd.grad(output.sum(), B)[0]

        self.assertTrue(torch.allclose(output, torch.zeros_like(output)))
        self.assertTrue(torch.equal(derivative, torch.full_like(B, 2.0)))

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

    def test_message_can_own_radial_centers(self):
        data = _grid_data(shape=(3, 3, 3))
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=1,
            radial_basis="gaussian",
            radial_exponents=(0.25,),
            separate_center=False,
        )
        message = BChiMessage(
            1,
            1,
            1,
            hidden_sizes=(),
            radial_exponents=(1.0,),
            radial_centers=(0.5,),
        )
        with torch.no_grad():
            message.mlp[0].weight.fill_(1.0)
            message.mlp[0].bias.zero_()
        B = torch.arange(27, dtype=torch.get_default_dtype()).reshape(
            27, 1, 1, 1
        )

        message_basis = a_features.stencil_basis(
            message.radial_exponents,
            message.radial_centers,
        )
        A_next = message(B, data["local_density_index"], message_basis)

        self.assertTrue(torch.isfinite(A_next).all())
        self.assertTrue(
            torch.equal(message.radial_centers, torch.tensor([0.5]))
        )
        self.assertNotIn("fixed_radial_centers", message.state_dict())

    def test_message_can_train_independent_radial_centers(self):
        rng_state = torch.random.get_rng_state()
        data = _grid_data(shape=(3, 3, 3))
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=1,
            radial_basis="gaussian",
            radial_exponents=(0.25,),
            separate_center=False,
        )
        message = BChiMessage(
            1,
            1,
            1,
            hidden_sizes=(),
            radial_exponents=(1.0,),
            radial_centers=(0.5,),
            trainable_radial_centers=True,
        )
        with torch.no_grad():
            message.mlp[0].weight.fill_(1.0)
            message.mlp[0].bias.zero_()
        B = torch.arange(27, dtype=torch.get_default_dtype()).reshape(
            27, 1, 1, 1
        )

        basis = a_features.stencil_basis(
            message.radial_exponents,
            message.radial_centers,
        )
        output = message(B, data["local_density_index"], basis)
        gradient = torch.autograd.grad(
            output[0, 0, 1, 0],
            message.learned_radial_centers,
        )[0]
        torch.random.set_rng_state(rng_state)

        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().item(), 0.0)
        self.assertIn("learned_radial_centers", message.state_dict())

    def test_trainable_message_radial_requires_initial_values(self):
        with self.assertRaisesRegex(ValueError, "requires radial_exponents"):
            BChiMessage(
                1,
                1,
                1,
                trainable_radial_exponents=True,
            )

    def test_bessel_message_arguments_are_validated(self):
        with self.assertRaisesRegex(ValueError, "n_radial_functions is required"):
            BChiMessage(1, 1, 1, radial_basis="bessel")
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            BChiMessage(
                1,
                2,
                1,
                radial_basis="bessel",
                n_radial_functions=1,
            )
        with self.assertRaisesRegex(ValueError, "exponents are unavailable"):
            BChiMessage(
                1,
                1,
                1,
                radial_exponents=(0.5,),
                radial_basis="bessel",
                n_radial_functions=1,
            )
        with self.assertRaisesRegex(ValueError, "requires radial_basis"):
            BChiMessage(1, 1, 1, n_radial_functions=1)

    def test_legacy_message_state_keys_are_unchanged(self):
        shared = BChiMessage(1, 1, 1, hidden_sizes=())
        gaussian = BChiMessage(
            1,
            1,
            1,
            hidden_sizes=(),
            radial_exponents=(0.5,),
        )

        self.assertEqual(
            set(shared.state_dict()),
            {"mlp.0.weight", "mlp.0.bias"},
        )
        self.assertEqual(
            set(gaussian.state_dict()),
            {"fixed_radial_exponents", "mlp.0.weight", "mlp.0.bias"},
        )

    def test_legacy_message_objects_use_radial_and_convolution_defaults(self):
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=1,
            max_power=1,
            radial_basis="gaussian",
            radial_exponents=(0.5,),
        )
        b_features = CartesianBFeatures(1, 2)
        data = _grid_data(shape=(3, 3, 3))
        for exponents in (None, (1.0,)):
            with self.subTest(exponents=exponents):
                message = BChiMessage(
                    b_features.n_features,
                    1,
                    1,
                    hidden_sizes=(4,),
                    radial_exponents=exponents,
                )
                model = GridCACEModel(
                    a_features=a_features,
                    b_features=b_features,
                    readout=[LocalReadout(n_types=1, hidden_sizes=(4,))],
                    grid_spacing=1.0,
                    message_layers=[message],
                )
                expected = model(data)["beta_F_exc"].detach().clone()

                del message.radial_basis
                del message.convolution_backend
                serialized = io.BytesIO()
                torch.save(model, serialized)
                serialized.seek(0)
                restored = _load_whole_model(serialized)
                self.assertEqual(
                    restored.message_layers[0].convolution_backend,
                    "gather",
                )
                actual = restored(data)["beta_F_exc"]

                self.assertTrue(torch.equal(actual, expected))

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

    def test_bessel_transform_commutes_with_cubic_grid_symmetry(self):
        torch.manual_seed(11)
        shape = (5, 5, 5)
        data = _grid_data(shape=shape, cutoff_grid=2)
        a_features = CartesianAFeatures(
            mean_density=0.7,
            cutoff_grid=2,
            max_power=2,
            radial_basis="bessel",
            n_radial_functions=2,
            n_radial_channels=2,
            separate_center=True,
        )
        with torch.no_grad():
            a_features.radial_transform.weight.add_(
                0.1 * torch.randn_like(a_features.radial_transform.weight)
            )
        b_features = CartesianBFeatures(2, 2)
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

    def test_shared_message_backpropagates_to_radial_transform(self):
        data = _grid_data(shape=(5, 5, 5), cutoff_grid=2)
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=2,
            max_power=1,
            radial_basis="bessel",
            n_radial_functions=2,
            n_radial_channels=1,
        )
        message = BChiMessage(1, 1, 1, hidden_sizes=())
        B = torch.rand(125, 1, 1, 1)

        message(
            B,
            data["local_density_index"],
            a_features.stencil_basis(),
        ).square().sum().backward()

        gradient = a_features.radial_transform.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.all(torch.isfinite(gradient)))
        self.assertGreater(gradient.abs().sum(), 0.0)

    def test_independent_gaussian_message_bypasses_bessel_transform(self):
        bessel = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=2,
            max_power=1,
            radial_basis="bessel",
            n_radial_functions=3,
            n_radial_channels=2,
        )
        gaussian = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=2,
            max_power=1,
            radial_basis="gaussian",
            radial_exponents=(0.25, 1.0),
        )
        message = BChiMessage(
            1,
            2,
            1,
            radial_exponents=(0.25, 1.0),
        )

        actual = bessel.stencil_basis(
            message.radial_exponents,
            message.radial_centers,
        )

        self.assertEqual(actual.shape, (33, 2, 4))
        self.assertTrue(torch.equal(actual, gaussian.stencil_basis()))

    def test_independent_bessel_message_has_own_transform(self):
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=2,
            max_power=1,
            radial_basis="gaussian",
            radial_exponents=(0.25, 1.0),
        )
        b_features = CartesianBFeatures(1, 2)
        message = BChiMessage(
            b_features.n_features,
            2,
            1,
            hidden_sizes=(),
            radial_basis="bessel",
            n_radial_functions=3,
        )
        GridCACEModel(
            a_features=a_features,
            b_features=b_features,
            readout=[LocalReadout(n_types=1, hidden_sizes=(4,))],
            grid_spacing=1.0,
            message_layers=[message],
        )

        basis = message._stencil_basis(
            a_features,
            a_features.stencil_basis(),
        )

        self.assertEqual(basis.shape, (33, 2, 4))
        self.assertIsNone(message.radial_exponents)
        self.assertIsNone(message.radial_centers)
        self.assertTrue(
            torch.equal(
                basis,
                message.fixed_bessel_stencil_basis[:, :2],
            )
        )
        self.assertEqual(message.radial_transform.weight.shape, (2, 3, 2))
        self.assertIn("fixed_bessel_stencil_basis", message.state_dict())
        self.assertIn("bessel_gram_eigenvalues", message.state_dict())
        self.assertIn("radial_transform.weight", message.state_dict())

    def test_independent_bessel_message_bypasses_initial_transform(self):
        a_features = CartesianAFeatures(
            mean_density=1.0,
            cutoff_grid=2,
            max_power=1,
            radial_basis="bessel",
            n_radial_functions=3,
            n_radial_channels=2,
        )
        b_features = CartesianBFeatures(1, 2)
        message = BChiMessage(
            b_features.n_features,
            2,
            1,
            radial_basis="bessel",
            n_radial_functions=3,
        )
        GridCACEModel(
            a_features=a_features,
            b_features=b_features,
            readout=[LocalReadout(n_types=1, hidden_sizes=(4,))],
            grid_spacing=1.0,
            message_layers=[message],
        )
        before = message._stencil_basis(
            a_features,
            a_features.stencil_basis(),
        ).detach().clone()
        initial_before = a_features.stencil_basis().detach().clone()

        with torch.no_grad():
            a_features.radial_transform.weight.add_(0.2)

        after = message._stencil_basis(
            a_features,
            a_features.stencil_basis(),
        )
        self.assertTrue(torch.equal(before, after))
        self.assertFalse(
            torch.equal(initial_before, a_features.stencil_basis())
        )

    def test_independent_bessel_message_preserves_cubic_symmetry(self):
        torch.manual_seed(23)
        shape = (5, 5, 5)
        data = _grid_data(shape=shape, cutoff_grid=2)
        a_features = CartesianAFeatures(
            mean_density=0.7,
            cutoff_grid=2,
            max_power=2,
            radial_basis="gaussian",
            radial_exponents=(0.25, 1.0),
        )
        b_features = CartesianBFeatures(2, 2)
        model = GridCACEModel(
            a_features=a_features,
            b_features=b_features,
            readout=[LocalReadout(n_types=1, hidden_sizes=(8,))],
            grid_spacing=1.0,
            compute_c1=True,
            message_layers=[
                BChiMessage(
                    b_features.n_features,
                    2,
                    1,
                    hidden_sizes=(5,),
                    radial_basis="bessel",
                    n_radial_functions=3,
                )
            ],
        )
        transformed_data = dict(data)
        transformed_data["rho"] = _transform_cubic_field(data["rho"], shape)

        result = model(data)["c1"]
        transformed_result = model(transformed_data)["c1"]

        self.assertTrue(
            torch.allclose(
                transformed_result,
                _transform_cubic_field(result, shape),
                rtol=3.0e-5,
                atol=3.0e-6,
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

    def test_non_gather_model_exactly_ignores_neighborhood_index(self):
        def make_model(message_backend):
            a_features = CartesianAFeatures(
                mean_density=0.5,
                cutoff_grid=1,
                max_power=2,
                radial_basis="gaussian",
                radial_exponents=(0.25, 1.0),
                n_radial_channels=2,
                trainable_radial_exponents=True,
                separate_center=True,
                convolution_backend="fft",
            )
            b_features = CartesianBFeatures(2, 2)
            return GridCACEModel(
                a_features=a_features,
                b_features=b_features,
                readout=[LocalReadout(n_types=1, hidden_sizes=(8,))],
                grid_spacing=1.0,
                compute_c1=True,
                compute_c2=True,
                message_layers=[
                    BChiMessage(
                        b_features.n_features,
                        2,
                        1,
                        hidden_sizes=(6,),
                        radial_exponents=(0.5, 1.5),
                        trainable_radial_exponents=True,
                        convolution_backend=message_backend,
                    )
                ],
            ).double()

        source_data = _grid_data(shape=(3, 4, 5), cutoff_grid=1)

        def copy_data(include_neighbors):
            copied = {
                key: value.clone() for key, value in source_data.items()
            }
            if not include_neighbors:
                copied.pop("local_density_index")
            for key in ("rho", "grid_spacing", "temperature", "beta"):
                copied[key] = copied[key].double()
            return copied

        for backend in ("conv3d", "fft"):
            with self.subTest(backend=backend):
                torch.manual_seed(43)
                model = make_model(backend)
                self.assertFalse(model.requires_local_density_index)
                included_data = copy_data(True)
                omitted_data = copy_data(False)
                included = model(included_data, c2_reference=(1, 0))
                omitted = model(omitted_data, c2_reference=(1, 0))

                for key in ("beta_F_exc", "c1", "c2"):
                    self.assertTrue(
                        torch.equal(included[key], omitted[key]),
                        key,
                    )

                included["c1"].square().mean().backward()
                included_rho_gradient = included_data["rho"].grad.clone()
                included_parameter_gradients = {
                    name: parameter.grad.clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
                model.zero_grad(set_to_none=True)
                omitted["c1"].square().mean().backward()

                self.assertTrue(
                    torch.equal(omitted_data["rho"].grad, included_rho_gradient)
                )
                self.assertEqual(
                    set(included_parameter_gradients),
                    {
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None
                    },
                )
                for name, parameter in model.named_parameters():
                    if name in included_parameter_gradients:
                        self.assertTrue(
                            torch.equal(
                                parameter.grad,
                                included_parameter_gradients[name],
                            ),
                            name,
                        )

    def test_model_reports_when_neighborhood_index_is_required(self):
        gather_model = self._make_model(lambda a_features, b_features: [])
        self.assertTrue(gather_model.requires_local_density_index)

        def make_fft_model(with_message):
            a_features = CartesianAFeatures(
                mean_density=0.5,
                cutoff_grid=1,
                max_power=2,
                radial_basis="gaussian",
                radial_exponents=(0.25,),
                convolution_backend="fft",
            )
            b_features = CartesianBFeatures(2, 2)
            messages = []
            if with_message:
                messages.append(
                    BChiMessage(
                        b_features.n_features,
                        1,
                        1,
                        convolution_backend="fft",
                    )
                )
            return GridCACEModel(
                a_features=a_features,
                b_features=b_features,
                readout=[LocalReadout(n_types=1, hidden_sizes=(8,))],
                grid_spacing=1.0,
                message_layers=messages,
            )

        legacy_features = make_fft_model(with_message=False)
        self.assertFalse(legacy_features.requires_local_density_index)
        del legacy_features.a_features.convolution_backend
        self.assertTrue(legacy_features.requires_local_density_index)

        legacy_message = make_fft_model(with_message=True)
        self.assertFalse(legacy_message.requires_local_density_index)
        del legacy_message.message_layers[0].convolution_backend
        self.assertTrue(legacy_message.requires_local_density_index)

    def test_convolution_matches_gather_energy_c1_c2_and_gradients(self):
        def make_model(backend):
            a_features = CartesianAFeatures(
                mean_density=0.5,
                cutoff_grid=1,
                max_power=2,
                radial_basis="gaussian",
                radial_exponents=(0.25, 1.0),
                n_radial_channels=2,
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
                compute_c2=True,
                message_layers=[
                    BChiMessage(
                        b_features.n_features,
                        2,
                        1,
                        hidden_sizes=(6,),
                        radial_exponents=(0.5, 1.5),
                        trainable_radial_exponents=True,
                        radial_centers=(0.1, 0.3),
                        trainable_radial_centers=True,
                        convolution_backend=backend,
                    )
                ],
            ).double()

        source_data = _grid_data(shape=(3, 4, 5), cutoff_grid=1)

        def copy_data():
            copied = {key: value.clone() for key, value in source_data.items()}
            for key in ("rho", "grid_spacing", "temperature", "beta"):
                copied[key] = copied[key].double()
            return copied

        for backend in ("conv3d", "fft"):
            with self.subTest(backend=backend):
                torch.manual_seed(37)
                gather = make_model("gather")
                convolution = make_model(backend)
                gather(copy_data(), compute_c1=False, compute_c2=False)
                convolution(copy_data(), compute_c1=False, compute_c2=False)
                convolution.load_state_dict(gather.state_dict(), strict=True)
                gather.zero_grad(set_to_none=True)
                convolution.zero_grad(set_to_none=True)
                gather_data = copy_data()
                convolution_data = copy_data()
                expected = gather(gather_data, c2_reference=(1, 0))
                actual = convolution(
                    convolution_data,
                    c2_reference=(1, 0),
                )

                for key in ("beta_F_exc", "c1", "c2"):
                    self.assertTrue(
                        torch.allclose(
                            actual[key],
                            expected[key],
                            rtol=2.0e-9,
                            atol=2.0e-10,
                        ),
                        key,
                    )
                serialized = io.BytesIO()
                torch.save(convolution, serialized)
                serialized.seek(0)
                restored = _load_whole_model(serialized)
                self.assertEqual(
                    restored.message_layers[0].convolution_backend,
                    backend,
                )
                with mock.patch(
                    "equicdft.interaction.gather_neighbors",
                    side_effect=AssertionError("gather backend was reached"),
                ):
                    restored_output = restored(
                        copy_data(),
                        c2_reference=(1, 0),
                    )
                for key in ("beta_F_exc", "c1", "c2"):
                    self.assertTrue(
                        torch.equal(restored_output[key], actual[key])
                    )
                expected["c1"].square().mean().backward()
                actual["c1"].square().mean().backward()
                self.assertTrue(
                    torch.allclose(
                        convolution_data["rho"].grad,
                        gather_data["rho"].grad,
                        rtol=2.0e-8,
                        atol=2.0e-9,
                    )
                )
                for (expected_name, expected_parameter), (
                    actual_name,
                    actual_parameter,
                ) in zip(
                    gather.named_parameters(),
                    convolution.named_parameters(),
                ):
                    self.assertEqual(actual_name, expected_name)
                    if expected_parameter.grad is None:
                        self.assertIsNone(actual_parameter.grad)
                    else:
                        self.assertTrue(
                            torch.allclose(
                                actual_parameter.grad,
                                expected_parameter.grad,
                                rtol=2.0e-8,
                                atol=2.0e-9,
                            ),
                            expected_name,
                        )

    def test_independent_bessel_fft_has_finite_functional_derivatives(self):
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=2,
            max_power=2,
            radial_basis="gaussian",
            radial_exponents=(0.25, 1.0),
        )
        b_features = CartesianBFeatures(2, 2)
        message = BChiMessage(
            b_features.n_features,
            2,
            1,
            hidden_sizes=(6,),
            radial_basis="bessel",
            n_radial_functions=3,
            convolution_backend="fft",
        )
        model = GridCACEModel(
            a_features=a_features,
            b_features=b_features,
            readout=[LocalReadout(n_types=1, hidden_sizes=(8,))],
            grid_spacing=1.0,
            compute_c1=True,
            compute_c2=True,
            message_layers=[message],
        )
        data = _grid_data(shape=(5, 5, 5), cutoff_grid=2)

        outputs = model(data, c2_reference=(0, 0))

        for key in ("beta_F_exc", "c1", "c2"):
            self.assertTrue(torch.all(torch.isfinite(outputs[key])), key)
        outputs["c1"].square().mean().backward()

        gradient = message.radial_transform.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.all(torch.isfinite(gradient)))
        self.assertGreater(gradient.abs().sum(), 0.0)

    def test_bessel_model_has_finite_functional_derivatives(self):
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=2,
            max_power=2,
            radial_basis="bessel",
            n_radial_functions=2,
            n_radial_channels=2,
        )
        b_features = CartesianBFeatures(2, 2)
        model = GridCACEModel(
            a_features=a_features,
            b_features=b_features,
            readout=[LocalReadout(n_types=1, hidden_sizes=(8,))],
            grid_spacing=1.0,
            compute_c1=True,
            compute_c2=True,
            message_layers=[
                BChiMessage(
                    b_features.n_features,
                    a_features.n_radial_channels,
                    a_features.n_output_channels,
                    hidden_sizes=(6,),
                )
            ],
        )
        data = _grid_data(shape=(5, 5, 5), cutoff_grid=2)

        outputs = model(data, c2_reference=(0, 0))

        self.assertTrue(torch.all(torch.isfinite(outputs["beta_F_exc"])))
        self.assertTrue(torch.all(torch.isfinite(outputs["c1"])))
        self.assertTrue(torch.all(torch.isfinite(outputs["c2"])))
        outputs["c1"].square().mean().backward()
        gradient = model.a_features.radial_transform.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.all(torch.isfinite(gradient)))

    def test_independent_bessel_message_has_finite_derivatives(self):
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=2,
            max_power=2,
            radial_basis="gaussian",
            radial_exponents=(0.25, 1.0),
        )
        b_features = CartesianBFeatures(2, 2)
        message = BChiMessage(
            b_features.n_features,
            a_features.n_radial_channels,
            a_features.n_output_channels,
            hidden_sizes=(6,),
            radial_basis="bessel",
            n_radial_functions=3,
        )
        model = GridCACEModel(
            a_features=a_features,
            b_features=b_features,
            readout=[LocalReadout(n_types=1, hidden_sizes=(8,))],
            grid_spacing=1.0,
            compute_c1=True,
            compute_c2=True,
            message_layers=[message],
        )
        data = _grid_data(shape=(5, 5, 5), cutoff_grid=2)

        outputs = model(data, c2_reference=(0, 0))

        self.assertTrue(torch.all(torch.isfinite(outputs["beta_F_exc"])))
        self.assertTrue(torch.all(torch.isfinite(outputs["c1"])))
        self.assertTrue(torch.all(torch.isfinite(outputs["c2"])))
        outputs["c1"].square().mean().backward()
        gradient = message.radial_transform.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.all(torch.isfinite(gradient)))
        self.assertGreater(gradient.abs().sum(), 0.0)

    def test_rpm_scale_bessel_message_shape_is_finite(self):
        a_features = CartesianAFeatures(
            mean_density=0.2,
            cutoff_grid=6,
            max_power=3,
            radial_basis="bessel",
            n_radial_functions=6,
            n_radial_channels=4,
            separate_center=True,
            n_types=2,
        )
        b_features = CartesianBFeatures(3, 2)
        message = BChiMessage(
            b_features.n_features,
            4,
            2,
            radial_basis="bessel",
            n_radial_functions=6,
        )
        model = GridCACEModel(
            a_features=a_features,
            b_features=b_features,
            readout=[LocalReadout(n_types=2, hidden_sizes=(8,))],
            grid_spacing=0.125,
            compute_c1=True,
            message_layers=[message],
        )
        data = _grid_data(shape=(3, 3, 3), cutoff_grid=6)
        data["rho"] = torch.rand(27, 2) + 0.1
        data["grid_spacing"] = torch.full((3,), 0.125)

        outputs = model(data)

        self.assertEqual(message.radial_transform.weight.shape, (4, 6, 4))
        self.assertEqual(outputs["c1"].shape, (27, 2))
        self.assertTrue(torch.all(torch.isfinite(outputs["c1"])))

    def test_two_bessel_messages_have_independent_transforms(self):
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=2,
            max_power=1,
            radial_basis="gaussian",
            radial_exponents=(0.25, 1.0),
        )
        b_features = CartesianBFeatures(1, 2)
        messages = [
            BChiMessage(
                b_features.n_features,
                2,
                1,
                hidden_sizes=(4,),
                radial_basis="bessel",
                n_radial_functions=3,
            )
            for _ in range(2)
        ]
        model = GridCACEModel(
            a_features=a_features,
            b_features=b_features,
            readout=[LocalReadout(n_types=1, hidden_sizes=(8,))],
            grid_spacing=1.0,
            compute_c1=True,
            message_layers=messages,
        )
        data = _grid_data(shape=(5, 5, 5), cutoff_grid=2)

        model(data)["c1"].square().mean().backward()

        first, second = model.message_layers
        self.assertIsNot(
            first.radial_transform.weight,
            second.radial_transform.weight,
        )
        self.assertNotEqual(
            first.radial_transform.weight.data_ptr(),
            second.radial_transform.weight.data_ptr(),
        )
        for message in (first, second):
            gradient = message.radial_transform.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.all(torch.isfinite(gradient)))
            self.assertGreater(gradient.abs().sum(), 0.0)

    def test_independent_bessel_message_strict_state_round_trip(self):
        def make_model():
            a_features = CartesianAFeatures(
                mean_density=0.5,
                cutoff_grid=2,
                max_power=1,
                radial_basis="gaussian",
                radial_exponents=(0.25, 1.0),
            )
            b_features = CartesianBFeatures(1, 2)
            return GridCACEModel(
                a_features=a_features,
                b_features=b_features,
                readout=[LocalReadout(n_types=1, hidden_sizes=(8,))],
                grid_spacing=1.0,
                compute_c1=True,
                message_layers=[
                    BChiMessage(
                        b_features.n_features,
                        2,
                        1,
                        hidden_sizes=(4,),
                        radial_basis="bessel",
                        n_radial_functions=3,
                    )
                ],
            )

        torch.manual_seed(19)
        original = make_model()
        data = _grid_data(shape=(5, 5, 5), cutoff_grid=2)
        expected = original(data)["c1"].detach().clone()
        state = original.state_dict()

        restored = make_model()
        restored.load_state_dict(state, strict=True)
        actual = restored(data)["c1"]

        self.assertTrue(torch.equal(actual, expected))

        serialized = io.BytesIO()
        torch.save(original, serialized)
        serialized.seek(0)
        whole_model = _load_whole_model(serialized)
        self.assertTrue(torch.equal(whole_model(data)["c1"], expected))

    def test_rank_deficient_bessel_message_is_rejected_when_bound(self):
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=1,
            max_power=0,
            separate_center=True,
        )
        b_features = CartesianBFeatures(0, 1)
        message = BChiMessage(
            b_features.n_features,
            1,
            1,
            radial_basis="bessel",
            n_radial_functions=1,
        )

        with self.assertRaisesRegex(ValueError, "no active stencil points"):
            GridCACEModel(
                a_features=a_features,
                b_features=b_features,
                readout=[LocalReadout(n_types=1, hidden_sizes=(4,))],
                grid_spacing=1.0,
                message_layers=[message],
            )

    def test_bessel_message_rejects_incompatible_rebinding(self):
        first_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=2,
            max_power=1,
        )
        second_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=3,
            max_power=1,
        )
        message = BChiMessage(
            1,
            1,
            1,
            radial_basis="bessel",
            n_radial_functions=2,
        )
        message._bind_bessel_basis(first_features)

        with self.assertRaisesRegex(ValueError, "incompatible"):
            message._bind_bessel_basis(second_features)

    def test_bessel_message_binding_respects_existing_double_dtype(self):
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=2,
            max_power=1,
            radial_basis="gaussian",
            radial_exponents=(0.25, 1.0),
        ).double()
        b_features = CartesianBFeatures(1, 2).double()
        message = BChiMessage(
            b_features.n_features,
            2,
            1,
            radial_basis="bessel",
            n_radial_functions=3,
        ).double()
        model = GridCACEModel(
            a_features=a_features,
            b_features=b_features,
            readout=[LocalReadout(n_types=1, hidden_sizes=(4,)).double()],
            grid_spacing=1.0,
            compute_c1=True,
            message_layers=[message],
        )
        data = _grid_data(shape=(5, 5, 5), cutoff_grid=2)
        for key in ("rho", "grid_spacing", "temperature", "beta"):
            data[key] = data[key].double()

        output = model(data)["c1"]

        self.assertEqual(
            message.fixed_bessel_stencil_basis.dtype,
            torch.float64,
        )
        self.assertEqual(message.radial_transform.weight.dtype, torch.float64)
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_bessel_message_rebind_after_dtype_conversion_is_idempotent(self):
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=2,
            max_power=1,
        )
        message = BChiMessage(
            1,
            1,
            1,
            radial_basis="bessel",
            n_radial_functions=2,
        )
        message._bind_bessel_basis(a_features)
        message.double()
        a_features.double()

        message._bind_bessel_basis(a_features)

        self.assertEqual(message.radial_transform.weight.dtype, torch.float64)

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
