import unittest

import numpy as np
import torch
from torch import nn

from equicdft import (
    BulkReadout,
    CartesianAFeatures,
    CartesianBFeatures,
    GGAReadout,
    GridCACEModel,
    LDAReadout,
    LocalReadout,
    LongRangeReadout,
    ReciprocalFeatures,
)
from equicdft.stencil import get_neighbor_indices


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


class _FirstFeatureReadout(LocalReadout):
    """Return the density feature and ignore scalar conditioning inputs."""

    def forward(self, local_features):
        return local_features[..., :1]


class _ConstantPerParticleReadout(LocalReadout):
    """Return a trainable constant, making the functional linear in rho."""

    def __init__(self):
        nn.Module.__init__(self)
        self.n_types = 1
        self.value = nn.Parameter(torch.tensor(1.0))

    def forward(self, local_features):
        return self.value.expand(*local_features.shape[:-1], 1)


class TestGridLDABranch(unittest.TestCase):
    @staticmethod
    def _data():
        return {
            "rho": torch.linspace(0.2, 0.8, steps=8).reshape(8, 1),
            "temperature": torch.tensor(1.5),
            "beta": torch.tensor(1.0 / 1.5),
            "grid_spacing": torch.full((3,), 0.5),
            "grid_size": torch.tensor([2, 2, 2]),
        }

    @staticmethod
    def _quadratic_model(compute_c2=False, with_long_range=False):
        lda_readout = LDAReadout(
            mean_density=1.0,
            n_types=1,
            hidden_sizes=(),
        )
        # beta_a_exc_lda = rho, so beta_F_exc_lda = DeltaV * sum rho**2.
        with torch.no_grad():
            lda_readout.mlp[-1].weight.copy_(torch.tensor([[1.0, 0.0]]))
            lda_readout.mlp[-1].bias.zero_()

        long_range_features = None
        long_range_readout = None
        if with_long_range:
            long_range_features = ReciprocalFeatures(
                radial_exponents=(0.5,),
                n_types=1,
            )
            long_range_readout = LongRangeReadout(
                n_kernels=1,
                n_types=1,
                hidden_sizes=(),
                zero_init=False,
                features=long_range_features,
            )
            with torch.no_grad():
                long_range_readout.mlp[-1].weight.zero_()
                long_range_readout.mlp[-1].bias.fill_(0.2)

        readouts = [lda_readout]
        if long_range_readout is not None:
            readouts.append(long_range_readout)
        return GridCACEModel(
            a_features=None,
            b_features=None,
            readout=readouts,
            grid_spacing=0.5,
            mean_temperature=1.0,
            compute_c1=True,
            compute_c2=compute_c2,
            free_energy_mode="beta",
        )

    def test_lda_energy_and_c1_need_no_neighborhood_data(self):
        model = self._quadratic_model()
        data = self._data()

        outputs = model(data)

        expected_energy = model.voxel_volume * torch.sum(data["rho"].square())
        self.assertTrue(torch.allclose(outputs["beta_F_exc"], expected_energy))
        self.assertTrue(torch.allclose(outputs["c1"], -2.0 * data["rho"]))
        self.assertEqual(model.cutoff_grid, 0)
        self.assertIsInstance(model.readout[0], LDAReadout)
        self.assertNotIn("local_density_index", data)

    def test_lda_c2_has_expected_local_hessian(self):
        model = self._quadratic_model(compute_c2=True)
        data = self._data()

        outputs = model(data, c2_reference=(3, 0))

        expected = torch.zeros_like(data["rho"])
        expected[3, 0] = -2.0 / model.voxel_volume
        self.assertTrue(torch.equal(outputs["c2"], expected))

    def test_lda_vacuum_energy_is_exactly_zero(self):
        model = self._quadratic_model()
        data = self._data()
        data["rho"].zero_()

        outputs = model(data)

        self.assertEqual(outputs["beta_F_exc"].item(), 0.0)
        self.assertTrue(torch.all(torch.isfinite(outputs["c1"])))

    def test_multicomponent_lda_uses_complete_local_composition(self):
        readout = LDAReadout(
            mean_density=1.0,
            n_types=2,
            hidden_sizes=(),
        )
        with torch.no_grad():
            readout.mlp[-1].weight.copy_(torch.tensor([[1.0, 2.0, 0.0]]))
            readout.mlp[-1].bias.zero_()
        model = GridCACEModel(
            a_features=None,
            b_features=None,
            readout=[readout],
            grid_spacing=1.0,
            compute_c1=False,
        )
        rho = torch.tensor([[0.2, 0.3], [0.4, 0.1]])
        data = {
            "rho": rho,
            "temperature": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
        }

        outputs = model(data)

        beta_a = rho[:, :1] + 2.0 * rho[:, 1:]
        expected_density = torch.sum(rho, dim=-1) * beta_a[:, 0]
        self.assertTrue(
            torch.allclose(
                outputs["beta_F_exc"],
                expected_density.sum(),
            )
        )

    def test_lda_and_long_range_energies_are_additive_and_trainable(self):
        model = self._quadratic_model(with_long_range=True)
        data = self._data()

        outputs = model(data)

        self.assertTrue(torch.isfinite(outputs["beta_F_exc"]))
        outputs["c1"].square().mean().backward()
        self.assertIsNotNone(model.readout[0].mlp[-1].weight.grad)
        self.assertIsNotNone(model.readout[1].mlp[-1].bias.grad)

    def test_cace_is_an_additive_correction_to_lda(self):
        lda_readout = LDAReadout(
            mean_density=1.0,
            hidden_sizes=(),
        )
        with torch.no_grad():
            lda_readout.mlp[-1].weight.copy_(torch.tensor([[1.0, 0.0]]))
            lda_readout.mlp[-1].bias.zero_()
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=[lda_readout, _FirstFeatureReadout()],
            grid_spacing=0.5,
            free_energy_mode="beta",
        )
        data = self._data()

        outputs = model(data)

        expected_energy = 0.5**3 * torch.sum(data["rho"].square())
        self.assertTrue(
            torch.allclose(outputs["beta_F_exc"], 2.0 * expected_energy)
        )
        self.assertTrue(torch.allclose(outputs["c1"], -4.0 * data["rho"]))

    def test_at_least_one_local_energy_branch_is_required(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            GridCACEModel(
                a_features=None,
                b_features=None,
                readout=[],
                grid_spacing=1.0,
            )


class TestGridLDAGGABranch(unittest.TestCase):
    @staticmethod
    def _model(with_long_range=False):
        lda_readout = LDAReadout(
            mean_density=1.0,
            n_types=1,
            hidden_sizes=(),
            zero_init=True,
        )
        gga_readout = GGAReadout(
            hidden_sizes=(),
            n_features=2,
            initial_coefficient=0.2,
        )
        long_range_features = None
        long_range_readout = None
        if with_long_range:
            long_range_features = ReciprocalFeatures(
                radial_exponents=(0.5,),
                n_types=1,
            )
            long_range_readout = LongRangeReadout(
                n_kernels=1,
                n_types=1,
                hidden_sizes=(),
                zero_init=False,
                features=long_range_features,
            )
            with torch.no_grad():
                long_range_readout.mlp[-1].weight.zero_()
                long_range_readout.mlp[-1].bias.fill_(0.1)
        readouts = [lda_readout, gga_readout]
        if long_range_readout is not None:
            readouts.append(long_range_readout)
        return GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=readouts,
            grid_spacing=1.0,
            compute_c1=True,
        )

    @staticmethod
    def _data(rho=None):
        if rho is None:
            rho = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        return {
            "rho": rho,
            "temperature": torch.tensor(1.0),
            "grid_spacing": torch.ones(3),
            "grid_size": torch.tensor([4, 1, 1]),
        }

    def test_positive_gga_energy_and_analytic_c1(self):
        model = self._model()

        outputs = model(self._data())

        self.assertIsInstance(model.readout, nn.ModuleList)
        self.assertEqual(len(model.readout), 2)
        self.assertTrue(model.has_local_features)
        self.assertAlmostEqual(outputs["beta_F_exc"].item(), 0.4, places=6)
        expected_c1 = torch.tensor([[0.4], [-0.4], [0.4], [-0.4]])
        self.assertTrue(torch.allclose(outputs["c1"], expected_c1))

    def test_homogeneous_density_has_no_gga_contribution(self):
        model = self._model()

        outputs = model(self._data(torch.full((4, 1), 0.7)))

        self.assertEqual(outputs["beta_F_exc"].item(), 0.0)

    def test_c1_loss_trains_gga_readout(self):
        model = self._model()

        outputs = model(self._data())
        outputs["c1"].square().mean().backward()

        self.assertIsNotNone(model.readout[1].mlp[-1].weight.grad)
        self.assertIsNotNone(model.readout[1].mlp[-1].bias.grad)
        self.assertGreater(
            model.readout[1].mlp[-1].bias.grad.abs().item(),
            0.0,
        )

    def test_lda_gga_and_long_range_energies_are_additive(self):
        model = self._model(with_long_range=True)

        outputs = model(self._data())

        self.assertTrue(torch.isfinite(outputs["beta_F_exc"]))

    def test_gga_can_augment_cace_but_requires_local_features(self):
        gga_readout = GGAReadout(n_features=2)
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=[_FirstFeatureReadout(), gga_readout],
            grid_spacing=1.0,
        )
        self.assertEqual(len(model.readout), 2)
        with self.assertRaisesRegex(ValueError, "required by"):
            GridCACEModel(
                a_features=None,
                b_features=None,
                readout=[LDAReadout(mean_density=1.0), gga_readout],
                grid_spacing=1.0,
            )

    def test_lda_cace_and_gga_are_all_additive_before_autodiff(self):
        lda_readout = LDAReadout(
            mean_density=1.0,
            hidden_sizes=(),
            zero_init=True,
        )
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=[
                lda_readout,
                _FirstFeatureReadout(),
                GGAReadout(
                    hidden_sizes=(),
                    n_features=2,
                    initial_coefficient=0.2,
                ),
            ],
            grid_spacing=1.0,
            compute_c1=True,
        )

        outputs = model(self._data())

        self.assertIsInstance(model.readout, nn.ModuleList)
        self.assertEqual(
            [module.__class__.__name__ for module in model.readout],
            [
                "LDAReadout",
                "_FirstFeatureReadout",
                "GGAReadout",
            ],
        )
        self.assertAlmostEqual(outputs["beta_F_exc"].item(), 2.4, places=6)
        expected_c1 = torch.tensor([[0.4], [-2.4], [0.4], [-2.4]])
        self.assertTrue(torch.allclose(outputs["c1"], expected_c1))


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
        free_energy_mode="beta",
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
                features=long_range_features,
            )
        readouts = [readout]
        if bulk_readout is not None:
            readouts.append(bulk_readout)
        if long_range_readout is not None:
            readouts.append(long_range_readout)
        return GridCACEModel(
            a_features,
            b_features,
            readouts,
            grid_spacing=0.5,
            mean_temperature=mean_temperature,
            boltzmann_constant=1.0,
            thermal_wavelength=1.0,
            compute_c1=compute_c1,
            compute_c2=compute_c2,
            compute_local_mu=compute_chemical_potential,
            rho_min=rho_min,
            free_energy_mode=free_energy_mode,
        )

    def test_physical_free_energy_is_constructor_default(self):
        model = GridCACEModel(
            CartesianAFeatures(
                mean_density=0.5,
                cutoff_grid=1,
                max_power=2,
                n_radial_channels=1,
            ),
            CartesianBFeatures(max_power=2, max_product_order=3),
            [LocalReadout(n_types=1, hidden_sizes=(8,))],
            grid_spacing=0.5,
        )

        self.assertEqual(model.free_energy_mode, "physical")

    def test_long_range_energy_is_combined_before_local_mu_derivative(self):
        model = self._make_model(
            compute_c1=True,
            compute_chemical_potential=True,
            with_long_range=True,
        )
        with torch.no_grad():
            model.readout[1].mlp[-1].bias.fill_(0.2)
        data = self._make_data()
        data["V_ext"] = torch.zeros_like(data["rho"])

        outputs = model(data)
        centered_local_mu = (
            outputs["local_chemical_potential"]
            - outputs["average_chemical_potential"].unsqueeze(-2)
        )
        centered_local_mu.square().mean().backward()

        self.assertEqual(outputs["beta_F_exc"].shape, ())
        final_gradient = model.readout[1].mlp[-1].bias.grad
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
            readout=[local_readout, bulk_readout],
            grid_spacing=0.5,
            mean_temperature=1.0,
            compute_c1=True,
            free_energy_mode="beta",
        )
        data = self._make_data()

        outputs = model(data)

        mean_density = data["rho"].mean()
        expected_energy = (
            model.voxel_volume
            * data["rho"].sum()
            * 2.0
            * mean_density
        )
        self.assertTrue(torch.allclose(outputs["beta_F_exc"], expected_energy))
        self.assertTrue(
            torch.allclose(
                outputs["c1"],
                -4.0 * mean_density * torch.ones_like(data["rho"]),
            )
        )

    def test_bulk_and_long_range_branches_are_independently_additive(self):
        model = self._make_model(
            compute_c1=True,
            with_bulk=True,
            with_long_range=True,
        )
        with torch.no_grad():
            model.readout[1].mlp[-1].bias.fill_(0.3)
            model.readout[2].mlp[-1].bias.fill_(0.2)

        outputs = model(self._make_data())

        self.assertTrue(torch.isfinite(outputs["beta_F_exc"]))
        outputs["c1"].square().mean().backward()
        self.assertIsNotNone(model.readout[1].mlp[-1].bias.grad)
        self.assertIsNotNone(model.readout[2].mlp[-1].bias.grad)

    def test_exposes_persistent_inference_metadata(self):
        model = self._make_model(mean_temperature=1.2)

        self.assertEqual(model.cutoff_grid, 1)
        self.assertEqual(model.n_types, 1)
        self.assertEqual(model.mean_density.item(), 0.5)
        self.assertTrue(
            torch.equal(model.grid_spacing, torch.full((3,), 0.5))
        )
        self.assertEqual(model.voxel_volume.item(), 0.125)
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

    def test_collects_outputs_and_initializes_density_gradient(self):
        model = self._make_model()
        data = self._make_data()

        outputs = model(data)

        self.assertEqual(list(outputs), ["beta_F_exc", "c1"])
        self.assertTrue(data["rho"].requires_grad)
        self.assertEqual(outputs["beta_F_exc"].shape, ())
        self.assertEqual(outputs["c1"].shape, (27, 1))

    def test_model_applies_c1_sign_and_voxel_volume(self):
        # With per-particle free energy equal to rho,
        # beta_F_exc = Delta V * sum_g rho_g^2. Its discrete derivative is
        # 2*Delta V*rho, while the continuum c1 must be -2*rho.
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=[_FirstFeatureReadout()],
            grid_spacing=0.5,
            compute_c1=True,
            free_energy_mode="beta",
        )
        data = self._make_data()
        data = {
            key: torch.stack((value, value.clone()))
            for key, value in data.items()
        }

        outputs = model(data)

        self.assertEqual(outputs["beta_F_exc"].shape, (2,))
        self.assertTrue(torch.allclose(outputs["c1"], -2.0 * data["rho"]))

    def test_physical_free_energy_is_scaled_before_differentiation(self):
        # The analytic readout gives f_tilde = DeltaV*sum(rho**2), where
        # f_tilde = F_exc/(k_B*T_ref). At T_ref=2, k_B=3, and T=1.5,
        # F_exc=6*f_tilde and beta_F_exc=(2/1.5)*f_tilde.
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=[_FirstFeatureReadout()],
            grid_spacing=0.5,
            mean_temperature=2.0,
            boltzmann_constant=3.0,
            compute_c1=True,
            free_energy_mode="physical",
        )
        data = self._make_data()

        outputs = model(data)

        reduced_free_energy = model.voxel_volume * torch.sum(
            data["rho"].square()
        )
        self.assertTrue(
            torch.allclose(outputs["F_exc"], 6.0 * reduced_free_energy)
        )
        self.assertTrue(
            torch.allclose(
                outputs["beta_F_exc"],
                (2.0 / 1.5) * reduced_free_energy,
            )
        )
        self.assertTrue(
            torch.allclose(outputs["c1"], -(4.0 / 1.5) * data["rho"])
        )

    def test_physical_mode_applies_inverse_temperature_to_same_free_energy(self):
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=[_FirstFeatureReadout()],
            grid_spacing=0.5,
            mean_temperature=2.0,
            boltzmann_constant=3.0,
            compute_c1=False,
            free_energy_mode="physical",
        )
        first = self._make_data()
        second = self._make_data()
        first["temperature"] = torch.tensor(1.0)
        second["temperature"] = torch.tensor(2.0)
        data = {
            key: torch.stack((first[key], second[key]))
            for key in first
        }

        outputs = model(data)

        self.assertTrue(
            torch.allclose(outputs["F_exc"][0], outputs["F_exc"][1])
        )
        self.assertTrue(
            torch.allclose(
                outputs["beta_F_exc"][0],
                2.0 * outputs["beta_F_exc"][1],
            )
        )

    def test_free_energy_mode_is_validated(self):
        with self.assertRaisesRegex(ValueError, "free_energy_mode"):
            self._make_model(free_energy_mode="unknown")

    def test_pre_mode_checkpoint_defaults_to_beta_free_energy(self):
        model = self._make_model()
        del model.free_energy_mode

        outputs = model(self._make_data())

        self.assertEqual(list(outputs), ["beta_F_exc", "c1"])
        self.assertTrue(torch.isfinite(outputs["beta_F_exc"]))

    def test_model_applies_c2_sign_and_voxel_volume_to_one_row(self):
        # Continuing the analytic quadratic functional above,
        # c2(g0, g) = -2 delta(g0, g) / Delta V. Only one Hessian row is
        # evaluated and returned for each independent batch entry.
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=[_FirstFeatureReadout()],
            grid_spacing=0.5,
            compute_c2=True,
            free_energy_mode="beta",
        )
        data = self._make_data()
        data = {
            key: torch.stack((value, value.clone()))
            for key, value in data.items()
        }

        outputs = model(data, c2_reference=(5, 0))

        expected = torch.zeros_like(data["rho"])
        expected[:, 5, 0] = -2.0 / model.voxel_volume
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
            readout=[readout],
            grid_spacing=0.5,
            mean_temperature=2.0,
            compute_c1=False,
            free_energy_mode="beta",
        )

        first = self._make_data()
        second = self._make_data()
        second["temperature"] = torch.tensor(2.0)
        data = {
            key: torch.stack((first[key], second[key]))
            for key in first
        }

        outputs = model(data)

        per_particle = data["temperature"][:, None, None] / 2.0
        expected = model.voxel_volume * torch.sum(
            data["rho"] * per_particle,
            dim=(-2, -1),
        )
        self.assertTrue(
            torch.allclose(outputs["beta_F_exc"], expected)
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
            readout=[readout],
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
            ["beta_F_exc"],
        )
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
                [LocalReadout(n_types=1)],
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
                [LocalReadout(n_types=1)],
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
                [LocalReadout(n_types=1)],
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
                [LocalReadout(n_types=1)],
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
            [readout],
            grid_spacing=0.5,
        )

        outputs = model(data)

        self.assertEqual(outputs["beta_F_exc"].shape, ())
        self.assertEqual(outputs["c1"].shape, (27, 2))
        outputs["c1"].square().mean().backward()
        mixing_gradient = model.a_features.channel_mixing.weight.grad
        self.assertIsNotNone(mixing_gradient)
        self.assertTrue(torch.all(torch.isfinite(mixing_gradient)))

    def test_separate_center_is_added_once_to_local_readout(self):
        data = self._make_data()
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=1,
            max_power=1,
            radial_basis="none",
            n_radial_channels=1,
            separate_center=True,
        )
        b_features = CartesianBFeatures(
            max_power=1,
            max_product_order=2,
        )
        readout = LocalReadout(n_types=1, hidden_sizes=(8,))
        model = GridCACEModel(
            a_features,
            b_features,
            [readout],
            grid_spacing=0.5,
        )

        outputs = model(data)
        expected_input_width = (
            a_features.n_radial_channels * b_features.n_features + 2
        )
        self.assertEqual(readout.mlp[0].in_features, expected_input_width)
        self.assertEqual(outputs["c1"].shape, data["rho"].shape)
        outputs["c1"].square().mean().backward()
        self.assertTrue(torch.all(torch.isfinite(data["rho"].grad)))


if __name__ == "__main__":
    unittest.main()
