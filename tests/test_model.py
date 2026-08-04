import unittest

import numpy as np
import torch
from torch import nn

from equicdft import (
    BulkReadout,
    CartesianAFeatures,
    CartesianBFeatures,
    GridCACEModel,
    LocalReadout,
    LongRangeReadout,
    ReciprocalFeatures,
    get_neighbor_indices,
)


class _DensityFeatures(nn.Module):
    """Expose rho directly for an analytic model-integration test."""

    def __init__(self):
        super().__init__()
        self.cutoff_grid = 0
        self.n_types = 1
        self.register_buffer("mean_density", torch.tensor(1.0))

    def forward(self, data):
        return data["rho"].unsqueeze(-2).unsqueeze(-2)


class _IdentityModule(nn.Module):
    def forward(self, values):
        return values


class _FirstFeatureReadout(nn.Module):
    """Return the density feature and ignore scalar conditioning inputs."""

    def forward(self, local_features):
        return local_features[..., :1]


class _ConstantPerParticleReadout(nn.Module):
    """Return a trainable constant, making the functional linear in rho."""

    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))

    def forward(self, local_features):
        return self.value.expand(*local_features.shape[:-1], 1)


class TestGridCACEModel(unittest.TestCase):
    def _make_data(self):
        shape = (3, 3, 3)
        grid_positions = np.indices(shape, dtype=int).reshape(3, -1).T
        neighbor_indices, _ = get_neighbor_indices(
            grid_positions,
            cutoff_grid=1,
        )
        return {
            "rho": torch.linspace(0.2, 0.8, steps=27).reshape(27, 1),
            "local_density_index": torch.tensor(
                neighbor_indices,
                dtype=torch.long,
            ),
            "grid_spacing": torch.tensor([0.5, 0.5, 0.5]),
            "grid_size": torch.tensor(shape),
            "temperature": torch.tensor(1.5),
            "beta": torch.tensor(1.0 / 1.5),
        }

    def _make_model(
        self,
        compute_c1=True,
        compute_c2=False,
        compute_chemical_potential=False,
        rho_min=0.0,
        mean_temperature=1.0,
        with_bulk=False,
        with_long_range=False,
    ):
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=1,
            max_power=2,
            n_radial_channels=1,
        )
        b_features = CartesianBFeatures(max_power=2, max_product_order=3)
        readout = LocalReadout(
            n_types=1,
            hidden_sizes=(8,),
        )
        long_range_features = None
        long_range_readout = None
        bulk_readout = None
        if with_bulk:
            bulk_readout = BulkReadout(
                n_types=1,
                hidden_sizes=(4,),
            )
        if with_long_range:
            long_range_features = ReciprocalFeatures(
                radial_exponents=(0.25, 0.5),
                n_types=1,
            )
            long_range_readout = LongRangeReadout(
                n_kernels=2,
                n_types=1,
                hidden_sizes=(4,),
            )
        return GridCACEModel(
            a_features,
            b_features,
            readout,
            grid_spacing=0.5,
            mean_temperature=mean_temperature,
            boltzmann_constant=1.0,
            thermal_wavelength=1.0,
            bulk_readout=bulk_readout,
            long_range_features=long_range_features,
            long_range_readout=long_range_readout,
            compute_c1=compute_c1,
            compute_c2=compute_c2,
            compute_local_mu=compute_chemical_potential,
            rho_min=rho_min,
        )

    def test_long_range_energy_is_combined_before_local_mu_derivative(self):
        model = self._make_model(
            compute_c1=True,
            compute_chemical_potential=True,
            with_long_range=True,
        )
        with torch.no_grad():
            model.long_range_readout.mlp[-1].bias.fill_(0.2)
        data = self._make_data()
        data["V_ext"] = torch.zeros_like(data["rho"])

        outputs = model(data)
        centered_local_mu = (
            outputs["local_chemical_potential"]
            - outputs["average_chemical_potential"].unsqueeze(-2)
        )
        centered_local_mu.square().mean().backward()

        self.assertEqual(outputs["beta_F_exc_local"].shape, ())
        self.assertEqual(outputs["beta_F_exc_long_range"].shape, ())
        self.assertTrue(
            torch.allclose(
                outputs["beta_F_exc"],
                outputs["beta_F_exc_local"]
                + outputs["beta_F_exc_long_range"],
            )
        )
        final_gradient = model.long_range_readout.mlp[-1].bias.grad
        self.assertIsNotNone(final_gradient)
        self.assertTrue(torch.all(torch.isfinite(final_gradient)))
        self.assertGreater(torch.linalg.vector_norm(final_gradient).item(), 0.0)

    def test_bulk_energy_is_combined_before_functional_derivative(self):
        local_readout = _ConstantPerParticleReadout()
        with torch.no_grad():
            local_readout.value.zero_()
        bulk_readout = BulkReadout(
            n_types=1,
            hidden_sizes=(),
            zero_init=False,
        )
        # beta_a_bulk = 2 * rho_bar. Therefore
        # beta_F_bulk = 2 * V * rho_bar**2 and c1_bulk = -4 * rho_bar.
        with torch.no_grad():
            bulk_readout.mlp[-1].weight.copy_(torch.tensor([[0.0, 2.0]]))
            bulk_readout.mlp[-1].bias.zero_()
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=local_readout,
            bulk_readout=bulk_readout,
            grid_spacing=0.5,
            mean_temperature=1.0,
            compute_c1=True,
        )
        data = self._make_data()

        outputs = model(data)

        mean_density = data["rho"].mean()
        expected_energy = (
            model.cell_volume
            * data["rho"].sum()
            * 2.0
            * mean_density
        )
        self.assertTrue(
            torch.allclose(outputs["beta_F_exc_bulk"], expected_energy)
        )
        self.assertTrue(
            torch.allclose(
                outputs["c1"],
                -4.0 * mean_density * torch.ones_like(data["rho"]),
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["beta_F_exc"],
                outputs["beta_F_exc_local"]
                + outputs["beta_F_exc_bulk"],
            )
        )

    def test_bulk_and_long_range_branches_are_independently_additive(self):
        model = self._make_model(
            compute_c1=True,
            with_bulk=True,
            with_long_range=True,
        )
        with torch.no_grad():
            model.bulk_readout.mlp[-1].bias.fill_(0.3)
            model.long_range_readout.mlp[-1].bias.fill_(0.2)

        outputs = model(self._make_data())

        self.assertTrue(
            torch.allclose(
                outputs["beta_F_exc"],
                outputs["beta_F_exc_local"]
                + outputs["beta_F_exc_bulk"]
                + outputs["beta_F_exc_long_range"],
            )
        )
        outputs["c1"].square().mean().backward()
        self.assertIsNotNone(model.bulk_readout.mlp[-1].bias.grad)
        self.assertIsNotNone(model.long_range_readout.mlp[-1].bias.grad)

    def test_exposes_persistent_inference_metadata(self):
        model = self._make_model(mean_temperature=1.2)

        self.assertEqual(model.cutoff_grid, 1)
        self.assertEqual(model.n_types, 1)
        self.assertEqual(model.mean_density.item(), 0.5)
        self.assertTrue(
            torch.equal(model.grid_spacing, torch.full((3,), 0.5))
        )
        self.assertEqual(model.cell_volume.item(), 0.125)
        self.assertEqual(model.boltzmann_constant.item(), 1.0)
        self.assertAlmostEqual(model.mean_temperature.item(), 1.2)
        self.assertTrue(
            torch.equal(model.thermal_wavelength, torch.ones(1))
        )
        state = model.state_dict()
        self.assertIn("grid_spacing", state)
        self.assertIn("mean_temperature", state)
        self.assertIn("boltzmann_constant", state)
        self.assertIn("thermal_wavelength", state)
        self.assertEqual(
            model.grid_info,
            {
                "cutoff_grid": 1,
                "grid_spacing": [0.5, 0.5, 0.5],
                "n_types": 1,
                "boltzmann_constant": 1.0,
                "thermal_wavelength": [1.0],
            },
        )

    def test_rejects_mismatched_input_grid_spacing(self):
        model = self._make_model()
        data = self._make_data()
        data["grid_spacing"] = torch.tensor([0.25, 0.25, 0.25])

        with self.assertRaisesRegex(ValueError, "does not match"):
            model(data)

    def test_grid_spacing_is_optional_in_forward_data(self):
        model = self._make_model()
        data = self._make_data()
        data.pop("grid_spacing")

        outputs = model(data)

        self.assertEqual(outputs["beta_F_exc"].shape, ())

    def test_pre_bulk_pickled_model_state_remains_local_only(self):
        model = self._make_model()
        del model.bulk_readout

        outputs = model(self._make_data())

        self.assertFalse(model.has_bulk)
        self.assertNotIn("beta_F_exc_bulk", outputs)

    def test_collects_outputs_and_initializes_density_gradient(self):
        model = self._make_model()
        data = self._make_data()

        outputs = model(data)

        self.assertEqual(list(outputs), model.model_outputs)
        self.assertTrue(data["rho"].requires_grad)
        self.assertEqual(
            outputs["beta_free_energy_per_particle"].shape,
            (27, 1),
        )
        self.assertEqual(outputs["beta_free_energy_density"].shape, (27,))
        self.assertEqual(outputs["beta_F_exc"].shape, ())
        self.assertEqual(outputs["c1"].shape, (27, 1))

    def test_model_applies_c1_sign_and_cell_volume(self):
        # With per-particle free energy equal to rho,
        # beta_F_exc = Delta V * sum_g rho_g^2. Its discrete derivative is
        # 2*Delta V*rho, while the continuum c1 must be -2*rho.
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=_FirstFeatureReadout(),
            grid_spacing=0.5,
            compute_c1=True,
        )
        data = self._make_data()
        data = {
            key: torch.stack((value, value.clone()))
            for key, value in data.items()
        }

        outputs = model(data)

        self.assertEqual(outputs["beta_F_exc"].shape, (2,))
        self.assertTrue(torch.allclose(outputs["c1"], -2.0 * data["rho"]))

    def test_model_applies_c2_sign_and_cell_volume_to_one_row(self):
        # Continuing the analytic quadratic functional above,
        # c2(g0, g) = -2 delta(g0, g) / Delta V. Only one Hessian row is
        # evaluated and returned for each independent batch entry.
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=_FirstFeatureReadout(),
            grid_spacing=0.5,
            compute_c2=True,
        )
        data = self._make_data()
        data = {
            key: torch.stack((value, value.clone()))
            for key, value in data.items()
        }

        outputs = model(data, c2_reference=(5, 0))

        expected = torch.zeros_like(data["rho"])
        expected[:, 5, 0] = -2.0 / model.cell_volume
        self.assertEqual(outputs["c2"].shape, (2, 27, 1))
        self.assertTrue(torch.equal(outputs["c2"], expected))

    def test_collects_local_chemical_potential_with_external_field(self):
        model = self._make_model(
            compute_c1=True,
            compute_chemical_potential=True,
            rho_min=0.5,
        )
        data = self._make_data()
        data["V_ext"] = torch.linspace(-1.0, 1.0, steps=27).reshape(27, 1)

        outputs = model(data)

        expected = (
            torch.log(data["rho"].detach())
            + data["V_ext"] / data["temperature"]
            - outputs["c1"]
        )
        self.assertIn("local_chemical_potential", outputs)
        self.assertTrue(
            torch.allclose(outputs["local_chemical_potential"], expected)
        )
        weights = (data["rho"].detach() > 0.5).to(expected.dtype)
        expected_average = (weights * expected).sum(dim=-2) / weights.sum(
            dim=-2
        )
        self.assertTrue(
            torch.allclose(
                outputs["average_chemical_potential"],
                expected_average,
            )
        )
        self.assertTrue(
            torch.equal(outputs["chemical_potential_weights"], weights)
        )

    def test_chemical_potential_is_activated_by_compute_local_mu(self):
        model = self._make_model(compute_c1=True)
        data = self._make_data()
        data["V_ext"] = torch.zeros_like(data["rho"])

        outputs = model(data)

        self.assertNotIn("local_chemical_potential", outputs)
        self.assertNotIn("average_chemical_potential", outputs)
        self.assertNotIn("chemical_potential_weights", outputs)

    def test_zero_density_local_chemical_potential_is_finite_sentinel(self):
        model = self._make_model(
            compute_c1=True,
            compute_chemical_potential=True,
        )
        data = self._make_data()
        data["rho"][0] = 0.0
        data["V_ext"] = torch.zeros_like(data["rho"])

        outputs = model(data)

        self.assertEqual(
            outputs["local_chemical_potential"][0, 0].item(),
            0.0,
        )
        self.assertTrue(
            torch.all(torch.isfinite(outputs["local_chemical_potential"]))
        )

    def test_temperature_is_appended_once_to_each_local_feature_vector(self):
        readout = LocalReadout(
            n_types=1,
            hidden_sizes=(),
            n_features=2,
        )
        with torch.no_grad():
            readout.mlp[0].weight.copy_(torch.tensor([[0.0, 1.0]]))
            readout.mlp[0].bias.zero_()
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=readout,
            grid_spacing=0.5,
            mean_temperature=2.0,
            compute_c1=False,
        )

        first = self._make_data()
        second = self._make_data()
        second["temperature"] = torch.tensor(2.0)
        data = {
            key: torch.stack((first[key], second[key]))
            for key in first
        }

        outputs = model(data)

        expected = (
            data["temperature"][:, None, None] / 2.0
        ).expand(2, 27, 1)
        self.assertTrue(
            torch.equal(
                outputs["beta_free_energy_per_particle"],
                expected,
            )
        )

    def test_mean_temperature_must_be_positive_scalar(self):
        for value in (0.0, -1.0, float("nan"), [1.0, 2.0]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "mean_temperature"):
                    self._make_model(mean_temperature=value)

    def test_empty_density_has_zero_integrated_free_energy(self):
        model = self._make_model()
        data = self._make_data()
        data["rho"].zero_()

        outputs = model(data)

        self.assertTrue(
            torch.all(torch.isfinite(outputs["beta_free_energy_per_particle"]))
        )
        self.assertTrue(
            torch.equal(
                outputs["beta_free_energy_density"],
                torch.zeros_like(outputs["beta_free_energy_density"]),
            )
        )
        self.assertEqual(outputs["beta_F_exc"].item(), 0.0)

    def test_training_c1_loss_reaches_readout_parameters(self):
        model = self._make_model()
        model.train()

        outputs = model(self._make_data())
        outputs["c1"].square().mean().backward()

        gradients = [parameter.grad for parameter in model.readout.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.all(torch.isfinite(gradient)) for gradient in gradients)
        )

    def test_evaluation_avoids_higher_order_graph(self):
        model = self._make_model()
        model.eval()

        with torch.no_grad():
            outputs = model(self._make_data())

        self.assertFalse(outputs["c1"].requires_grad)

    def test_c2_evaluation_works_inside_no_grad(self):
        model = self._make_model(compute_c2=True)
        model.eval()

        with torch.no_grad():
            outputs = model(self._make_data(), c2_reference=(3, 0))

        self.assertEqual(outputs["c2"].shape, (27, 1))
        self.assertFalse(outputs["c1"].requires_grad)
        self.assertFalse(outputs["c2"].requires_grad)

    def test_linear_functional_has_zero_c2_with_trainable_graph(self):
        readout = _ConstantPerParticleReadout()
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=readout,
            grid_spacing=0.5,
            compute_c2=True,
        )

        outputs = model(self._make_data(), c2_reference=(0, 0))
        outputs["c2"].square().sum().backward()

        self.assertTrue(
            torch.equal(outputs["c2"], torch.zeros_like(outputs["c2"]))
        )
        self.assertIsNotNone(readout.value.grad)
        self.assertEqual(readout.value.grad.item(), 0.0)

    def test_training_c2_loss_reaches_readout_parameters(self):
        model = self._make_model(compute_c2=True)
        model.train()

        outputs = model(self._make_data(), c2_reference=(4, 0))
        outputs["c2"].square().mean().backward()

        gradients = [
            parameter.grad
            for parameter in model.readout.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(
            all(torch.all(torch.isfinite(gradient)) for gradient in gradients)
        )

    def test_c1_is_optional(self):
        model = self._make_model(compute_c1=False)
        model.eval()
        data = self._make_data()

        with torch.no_grad():
            outputs = model(data)

        self.assertEqual(
            list(outputs),
            [
                "beta_free_energy_per_particle",
                "beta_free_energy_density",
                "beta_F_exc",
            ],
        )
        self.assertEqual(model.required_derivatives, [])
        self.assertFalse(data["rho"].requires_grad)
        self.assertFalse(outputs["beta_F_exc"].requires_grad)

    def test_c1_can_be_selected_per_forward_call(self):
        model = self._make_model(compute_c1=True)
        data = self._make_data()
        data["rho"].requires_grad_(True)

        energy_outputs = model(data, compute_c1=False)
        self.assertNotIn("c1", energy_outputs)
        energy_gradient = torch.autograd.grad(
            energy_outputs["beta_F_exc"],
            data["rho"],
        )[0]
        self.assertTrue(torch.all(torch.isfinite(energy_gradient)))

        energy_default_model = self._make_model(compute_c1=False)
        response_outputs = energy_default_model(
            self._make_data(),
            compute_c1=True,
        )
        self.assertIn("c1", response_outputs)

    def test_c2_can_be_selected_per_forward_call_and_implies_c1(self):
        model = self._make_model(compute_c1=False)
        data = self._make_data()

        outputs = model(
            data,
            compute_c1=False,
            compute_c2=True,
            c2_reference=(2, 0),
        )

        self.assertIn("c1", outputs)
        self.assertIn("c2", outputs)
        self.assertTrue(data["rho"].requires_grad)

    def test_c2_reference_is_validated(self):
        model = self._make_model(compute_c2=True)

        invalid_references = (
            ((0,), TypeError),
            ((0, 0, 0), TypeError),
            ((0.0, 0), TypeError),
            ((27, 0), IndexError),
            ((0, 1), IndexError),
            ((-1, 0), IndexError),
        )
        for reference, error_type in invalid_references:
            with self.subTest(reference=reference):
                with self.assertRaises(error_type):
                    model(
                        self._make_data(),
                        c2_reference=reference,
                    )

    def test_compute_c1_must_be_boolean(self):
        with self.assertRaises(TypeError):
            GridCACEModel(
                CartesianAFeatures(
                    mean_density=0.5,
                    cutoff_grid=1,
                    max_power=2,
                    n_radial_channels=1,
                ),
                CartesianBFeatures(max_power=2, max_product_order=3),
                LocalReadout(n_types=1),
                grid_spacing=0.5,
                compute_c1=1,
            )

        with self.assertRaises(TypeError):
            GridCACEModel(
                CartesianAFeatures(
                    mean_density=0.5,
                    cutoff_grid=1,
                    max_power=2,
                    n_radial_channels=1,
                ),
                CartesianBFeatures(max_power=2, max_product_order=3),
                LocalReadout(n_types=1),
                grid_spacing=0.5,
                compute_c2=1,
            )

        with self.assertRaises(TypeError):
            GridCACEModel(
                CartesianAFeatures(
                    mean_density=0.5,
                    cutoff_grid=1,
                    max_power=2,
                    n_radial_channels=1,
                ),
                CartesianBFeatures(max_power=2, max_product_order=3),
                LocalReadout(n_types=1),
                grid_spacing=0.5,
                compute_local_mu=1,
            )

        with self.assertRaisesRegex(ValueError, "requires compute_c1"):
            GridCACEModel(
                CartesianAFeatures(
                    mean_density=0.5,
                    cutoff_grid=1,
                    max_power=2,
                    n_radial_channels=1,
                ),
                CartesianBFeatures(max_power=2, max_product_order=3),
                LocalReadout(n_types=1),
                grid_spacing=0.5,
                compute_c1=False,
                compute_local_mu=True,
            )

    def test_multicomponent_latent_channels_feed_physical_readout(self):
        data = self._make_data()
        data["rho"] = torch.cat(
            (data["rho"], 0.5 * data["rho"]),
            dim=-1,
        )
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=1,
            max_power=2,
            n_radial_channels=1,
            n_types=2,
            n_channels=3,
        )
        b_features = CartesianBFeatures(max_power=2, max_product_order=3)
        readout = LocalReadout(
            n_types=2,
            hidden_sizes=(8,),
        )
        model = GridCACEModel(
            a_features,
            b_features,
            readout,
            grid_spacing=0.5,
        )

        outputs = model(data)

        self.assertEqual(
            outputs["beta_free_energy_per_particle"].shape,
            (27, 2),
        )
        self.assertEqual(outputs["c1"].shape, (27, 2))
        outputs["c1"].square().mean().backward()
        mixing_gradient = model.a_features.channel_mixing.weight.grad
        self.assertIsNotNone(mixing_gradient)
        self.assertTrue(torch.all(torch.isfinite(mixing_gradient)))


if __name__ == "__main__":
    unittest.main()
