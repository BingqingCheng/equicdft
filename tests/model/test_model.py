import unittest

import numpy as np
import torch
from torch import nn

from equicdft.solver import ConvergenceOptions
from equicdft import (
    BChiMessage,
    BulkReadout,
    CartesianAFeatures,
    CartesianBFeatures,
    GGAReadout,
    GridCACEModel,
    GridData,
    LDAReadout,
    LocalReadout,
    LongRangeReadout,
    ReciprocalFeatures,
)
from equicdft.geometry.stencil import get_neighbor_indices


class _DensityFeatures(nn.Module):
    """Expose rho directly for an analytic model-integration test.

    The stand-in satisfies the feature-module contract the model declares
    (geometry, normalization, polarization flags and the pointwise density
    transform) without computing any Cartesian moments.
    """

    def __init__(self):
        super().__init__()
        self.cutoff_grid = 0
        self.n_types = 1
        self.n_radial_channels = 1
        self.n_output_channels = 1
        self.separate_center = False
        self.convolution_backend = "gather"
        self.include_polarization = False
        self.requires_dipole_density = False
        self.register_buffer("mean_density", torch.tensor(1.0))

    def transform_density(self, rho):
        return rho

    def forward(self, data):
        return data["rho"].unsqueeze(-2).unsqueeze(-2)


class _IdentityModule(nn.Module):
    """Stand-in symmetrizer returning the single density feature unchanged."""

    n_features = 1

    def validate_a_features(self, a_features):
        pass

    def forward(self, values):
        return values


class _FirstFeatureReadout(LocalReadout):
    """Return the density feature and ignore scalar conditioning inputs."""

    def forward(self, local_features):
        return local_features[..., :1]


class _ConstantPerParticleReadout(LocalReadout):
    """Return a trainable constant, making the functional linear in rho."""

    def __init__(self):
        # Left unbound so the containing model assigns the feature width.
        super().__init__(n_types=1, hidden_sizes=())
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
            mean_density=1.0,
            boltzmann_constant=1.0,
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
        self.assertFalse(model.requires_local_density_index)
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
            mean_density=1.0,
            boltzmann_constant=1.0,
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
        lda_readout = LDAReadout(hidden_sizes=())
        with torch.no_grad():
            lda_readout.mlp[-1].weight.copy_(torch.tensor([[1.0, 0.0]]))
            lda_readout.mlp[-1].bias.zero_()
        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=[lda_readout, _FirstFeatureReadout()],
            grid_spacing=0.5,
            free_energy_mode="beta",
            boltzmann_constant=1.0,
        )
        data = self._data()

        outputs = model(data)

        expected_energy = 0.5**3 * torch.sum(data["rho"].square())
        self.assertTrue(
            torch.allclose(outputs["beta_F_exc"], 2.0 * expected_energy)
        )
        self.assertTrue(torch.allclose(outputs["c1"], -4.0 * data["rho"]))

    def test_at_least_one_local_energy_branch_is_required(self):
        with self.assertRaisesRegex(ValueError, "n_types"):
            GridCACEModel(
                a_features=None,
                b_features=None,
                readout=[],
                grid_spacing=1.0,
                boltzmann_constant=1.0,
            )

    def test_model_owns_the_one_density_scale(self):
        features = _DensityFeatures()
        readouts = [LDAReadout(hidden_sizes=()), _FirstFeatureReadout()]
        implicit = GridCACEModel(features, _IdentityModule(), readouts, grid_spacing=0.5, boltzmann_constant=1.0)
        self.assertEqual(implicit.mean_density.item(), 1.0)
        self.assertIn("mean_density", implicit.state_dict())
        explicit = GridCACEModel(
            features, _IdentityModule(), readouts, grid_spacing=0.5, mean_density=1.0,
            boltzmann_constant=1.0,
        )
        self.assertEqual(explicit.mean_density.item(), 1.0)
        with self.assertRaisesRegex(ValueError, "mean_density must equal"):
            GridCACEModel(
                features, _IdentityModule(), readouts, grid_spacing=0.5, mean_density=2.0,
                boltzmann_constant=1.0,
            )
        with self.assertRaisesRegex(ValueError, "mean_density"):
            GridCACEModel(None, None, [LDAReadout()], grid_spacing=0.5, mean_density=-1.0, boltzmann_constant=1.0)
        unscaled = GridCACEModel(None, None, [LDAReadout()], grid_spacing=0.5, boltzmann_constant=1.0)
        self.assertIsNone(unscaled.mean_density)
        self.assertIsNone(unscaled.to_config()["mean_density"])
        rebuilt = GridCACEModel.from_config(unscaled.to_config())
        self.assertIsNone(rebuilt.mean_density)
        scaled = GridCACEModel.from_config(
            GridCACEModel(None, None, [LDAReadout()], grid_spacing=0.5, mean_density=0.7, boltzmann_constant=1.0).to_config()
        )
        self.assertAlmostEqual(scaled.mean_density.item(), 0.7)

    def test_local_readouts_are_bound_to_the_feature_width_at_construction(self):
        a_features = CartesianAFeatures(
            mean_density=0.5, cutoff_grid=1, max_power=2, radial_basis="gaussian",
            radial_exponents=(0.2, 0.5), n_types=2, separate_center=True,
        ).double()
        b_features = CartesianBFeatures(max_power=2, max_product_order=2).double()
        local = LocalReadout(n_types=2, hidden_sizes=(3,))
        gga = GGAReadout(hidden_sizes=(2,), n_types=2)
        expected_width = 2 + 2 * b_features.n_features * 2 + 1

        model = GridCACEModel(a_features, b_features, [local, gga], grid_spacing=0.5, boltzmann_constant=1.0)

        self.assertEqual(model.n_local_features, expected_width)
        for readout in (local, gga):
            self.assertTrue(readout.is_bound)
            self.assertEqual(readout.n_features, expected_width)
            # Freshly built layers follow the feature module's dtype.
            self.assertEqual(readout.mlp[0].weight.dtype, torch.float64)
        self.assertEqual(model.to_config()["readout"][0]["n_features"], expected_width)
        # A readout bound elsewhere must agree with the feature layout.
        with self.assertRaisesRegex(ValueError, "bound to 5 input features"):
            GridCACEModel(
                a_features, b_features, [LocalReadout(n_types=2, n_features=5)], grid_spacing=0.5,
                boltzmann_constant=1.0,
            )
        # Readouts bound beforehand keep their own dtype.
        prebound = LocalReadout(n_types=2, hidden_sizes=(3,), n_features=expected_width)
        GridCACEModel(a_features, b_features, [prebound], grid_spacing=0.5, boltzmann_constant=1.0)
        self.assertEqual(prebound.mlp[0].weight.dtype, torch.float32)
        self.assertEqual(
            GridCACEModel(None, None, [LDAReadout()], grid_spacing=1.0, boltzmann_constant=1.0).n_local_features, 0,
        )

    def test_feature_modules_must_satisfy_the_declared_contract(self):
        class _Incomplete(nn.Module):
            n_types = 1
            cutoff_grid = 0

            def forward(self, data):
                return data["rho"].unsqueeze(-2).unsqueeze(-2)

        with self.assertRaisesRegex(TypeError, "a_features.*mean_density.*transform_density"):
            GridCACEModel(_Incomplete(), _IdentityModule(), [_FirstFeatureReadout()], grid_spacing=1.0, boltzmann_constant=1.0)
        with self.assertRaisesRegex(TypeError, "b_features.*validate_a_features"):
            GridCACEModel(_DensityFeatures(), nn.Identity(), [_FirstFeatureReadout()], grid_spacing=1.0, boltzmann_constant=1.0)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            GridCACEModel(_DensityFeatures(), None, [_FirstFeatureReadout()], grid_spacing=1.0, boltzmann_constant=1.0)
        # Message layers rely on the stencil geometry of the feature module.
        message = BChiMessage(n_invariant_features=1, n_radial_channels=1, n_channels=1)
        with self.assertRaisesRegex(TypeError, "stencil_basis.*local_density_positions"):
            GridCACEModel(
                _DensityFeatures(), _IdentityModule(), [_FirstFeatureReadout()],
                grid_spacing=1.0, message_layers=[message],
                boltzmann_constant=1.0,
            )


class TestGridLDAGGABranch(unittest.TestCase):
    @staticmethod
    def _model(with_long_range=False):
        lda_readout = LDAReadout(
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
            boltzmann_constant=1.0,
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
            boltzmann_constant=1.0,
        )
        self.assertEqual(len(model.readout), 2)
        with self.assertRaisesRegex(ValueError, "required by"):
            GridCACEModel(
                a_features=None,
                b_features=None,
                readout=[LDAReadout(), gga_readout],
                grid_spacing=1.0,
                boltzmann_constant=1.0,
            )

    def test_lda_cace_and_gga_are_all_additive_before_autodiff(self):
        lda_readout = LDAReadout(hidden_sizes=(), zero_init=True)
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
            boltzmann_constant=1.0,
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
        mean_temperature=1.0,
        with_bulk=False,
        with_long_range=False,
        free_energy_mode="beta",
    ):
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=1,
            max_power=2,
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
            free_energy_mode=free_energy_mode,
        )

    def test_beta_free_energy_is_constructor_default(self):
        model = GridCACEModel(
            CartesianAFeatures(
                mean_density=0.5,
                cutoff_grid=1,
                max_power=2,
            ),
            CartesianBFeatures(max_power=2, max_product_order=3),
            [LocalReadout(n_types=1, hidden_sizes=(8,))],
            grid_spacing=0.5,
            boltzmann_constant=1.0,
        )

        self.assertEqual(model.free_energy_mode, "beta")

    def test_long_range_energy_is_combined_before_total_derivative(self):
        model = self._make_model(
            compute_c1=True,
            with_long_range=True,
        )
        with torch.no_grad():
            model.readout[1].mlp[-1].bias.fill_(0.2)
        data = self._make_data()
        data["V_ext"] = torch.zeros_like(data["rho"])

        outputs = model(data)
        centered_local_mu = (
            outputs["density_derivative"]
            - outputs["density_derivative"].mean(dim=-2, keepdim=True)
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
            boltzmann_constant=1.0,
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
        self.assertIn("ideal.thermal_wavelength", state)
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

        self.assertTrue({"beta_F_exc", "c1", "beta_F_id", "beta_F"}.issubset(outputs))
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
            boltzmann_constant=1.0,
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
        data["beta"] = torch.tensor(1.0 / (3.0 * 1.5))

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
        first["beta"] = torch.tensor(1.0 / 3.0)
        second["beta"] = torch.tensor(1.0 / 6.0)
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
        from equicdft.conversion.legacy import upgrade_legacy_model

        model = self._make_model()
        # Whole-object models saved before the mode flag existed lack the
        # attribute; the legacy upgrade restores the beta-F convention.
        del model.free_energy_mode
        with self.assertRaises(AttributeError):
            model(self._make_data())

        upgrade_legacy_model(model)
        self.assertEqual(model.free_energy_mode, "beta")
        outputs = model(self._make_data())

        self.assertTrue({"beta_F_exc", "c1", "beta_F_id", "beta_F"}.issubset(outputs))
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
            boltzmann_constant=1.0,
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

    def test_intrinsic_model_does_not_depend_on_external_field(self):
        model = self._make_model()
        data = self._make_data()
        first = model(dict(data))
        data["V_ext"] = torch.randn_like(data["rho"])
        second = model(dict(data))
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        for key in ("beta_mu_local", "beta_mu_average", "beta_mu_weights"):
            self.assertNotIn(key, second)
        for name in ("compute_local_mu", "rho_min", "average_chemical_potential"):
            self.assertFalse(hasattr(model, name))

    def test_vacuum_ideal_derivative_is_marked_invalid(self):
        model = self._make_model()
        data = self._make_data()
        data["rho"][0] = 0.
        outputs = model(data)
        self.assertEqual(outputs["ideal_density_derivative"][0, 0].item(), 0.)
        self.assertFalse(outputs["ideal_derivative_mask"][0, 0])

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
            boltzmann_constant=1.0,
        )

        first = self._make_data()
        second = self._make_data()
        second["temperature"] = torch.tensor(2.0)
        second["beta"] = torch.tensor(0.5)
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
            boltzmann_constant=1.0,
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
            ["beta_F_exc", "beta_F_id", "ideal_derivative_mask", "ideal_density_derivative", "beta_F"],
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
                ),
                CartesianBFeatures(max_power=2, max_product_order=3),
                [LocalReadout(n_types=1)],
                grid_spacing=0.5,
                compute_c1=1,
                boltzmann_constant=1.0,
            )

        with self.assertRaises(TypeError):
            GridCACEModel(
                CartesianAFeatures(
                    mean_density=0.5,
                    cutoff_grid=1,
                    max_power=2,
                ),
                CartesianBFeatures(max_power=2, max_product_order=3),
                [LocalReadout(n_types=1)],
                grid_spacing=0.5,
                compute_c2=1,
                boltzmann_constant=1.0,
            )

        with self.assertRaises(TypeError):
            GridCACEModel(
                CartesianAFeatures(
                    mean_density=0.5,
                    cutoff_grid=1,
                    max_power=2,
                ),
                CartesianBFeatures(max_power=2, max_product_order=3),
                [LocalReadout(n_types=1)],
                grid_spacing=0.5,
                compute_local_mu=1,
                boltzmann_constant=1.0,
            )

        with self.assertRaisesRegex(TypeError, "compute_local_mu"):
            GridCACEModel(
                CartesianAFeatures(
                    mean_density=0.5,
                    cutoff_grid=1,
                    max_power=2,
                ),
                CartesianBFeatures(max_power=2, max_product_order=3),
                [LocalReadout(n_types=1)],
                grid_spacing=0.5,
                compute_c1=False,
                compute_local_mu=True,
                boltzmann_constant=1.0,
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
            boltzmann_constant=1.0,
        )

        outputs = model(data)

        self.assertEqual(outputs["beta_F_exc"].shape, ())
        self.assertEqual(outputs["c1"].shape, (27, 2))
        outputs["c1"].square().mean().backward()
        mixing_gradient = model.a_features.density_transform.weight.grad
        self.assertIsNotNone(mixing_gradient)
        self.assertTrue(torch.all(torch.isfinite(mixing_gradient)))

    def test_transformed_density_replaces_direct_center_descriptors(self):
        data = self._make_data()
        data["rho"] = torch.cat(
            (data["rho"], 0.5 * data["rho"]),
            dim=-1,
        )
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=1,
            max_power=1,
            separate_center=True,
            n_types=2,
            density_transform=((0.5, 0.5),),
            trainable_density_transform=False,
        )
        b_features = CartesianBFeatures(
            max_power=1,
            max_product_order=1,
        )
        readout = LocalReadout(n_types=2, hidden_sizes=(8,))
        model = GridCACEModel(
            a_features,
            b_features,
            [readout],
            grid_spacing=0.5,
            boltzmann_constant=1.0,
        )

        outputs = model(data)

        expected_input_width = 1 + b_features.n_features + 1
        self.assertEqual(readout.mlp[0].in_features, expected_input_width)
        self.assertEqual(outputs["c1"].shape, data["rho"].shape)
        gradient = torch.autograd.grad(
            outputs["c1"].square().mean(), data["rho"]
        )[0]
        self.assertTrue(torch.all(torch.isfinite(gradient)))

    def test_separate_center_is_added_once_to_local_readout(self):
        data = self._make_data()
        a_features = CartesianAFeatures(
            mean_density=0.5,
            cutoff_grid=1,
            max_power=1,
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
            boltzmann_constant=1.0,
        )

        outputs = model(data)
        expected_input_width = b_features.n_features + 2
        self.assertEqual(readout.mlp[0].in_features, expected_input_width)
        self.assertEqual(outputs["c1"].shape, data["rho"].shape)
        outputs["c1"].square().mean().backward()
        self.assertTrue(torch.all(torch.isfinite(data["rho"].grad)))


class TestUnitContract(unittest.TestCase):
    """The model and the data share one Boltzmann constant, and say so."""

    def _model(self, **overrides):
        arguments = dict(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=[_FirstFeatureReadout()],
            grid_spacing=0.5,
        )
        arguments.update(overrides)
        return GridCACEModel(**arguments)

    def test_default_constant_is_the_data_default(self):
        from equicdft.units import BOLTZMANN_CONSTANT_EV_PER_K, DEFAULT_BOLTZMANN_CONSTANT
        from equicdft.data import DEFAULT_BOLTZMANN_CONSTANT as DATA_DEFAULT

        self.assertEqual(DEFAULT_BOLTZMANN_CONSTANT, BOLTZMANN_CONSTANT_EV_PER_K)
        self.assertEqual(DATA_DEFAULT, DEFAULT_BOLTZMANN_CONSTANT)
        self.assertAlmostEqual(self._model().boltzmann_constant.item(), DEFAULT_BOLTZMANN_CONSTANT)

    def test_grid_info_round_trip_passes_the_check(self):
        model = self._model(mean_temperature=300.0)
        data = GridData.from_dict(
            {"grid_size": [2, 2, 2], "n_types": 1, "grid_spacing": 0.5, "temperature": 350.0},
            cutoff_grid=0,
            grid_info=model.grid_info,
        )
        data["rho"] = torch.full((8, 1), 0.3)
        outputs = model(data)
        self.assertIn("beta_F_exc", outputs)

    def test_mismatched_constants_are_rejected_with_both_values(self):
        model = self._model(boltzmann_constant=1.0)
        data = TestGridCACEModel._make_data(self)
        model(dict(data))
        data["beta"] = torch.tensor(1.0 / (8.617333262e-5 * 1.5))
        with self.assertRaisesRegex(ValueError, "implies k_B = 8.617.*boltzmann_constant = 1"):
            model(data)

    def test_batched_beta_is_checked_per_field(self):
        model = self._model(boltzmann_constant=1.0)
        first = TestGridCACEModel._make_data(self)
        second = TestGridCACEModel._make_data(self)
        batch = {
            key: torch.stack([first[key], second[key]]) if torch.is_tensor(first[key]) else first[key]
            for key in first
        }
        batch["temperature"] = torch.tensor([1.5, 2.0])
        batch["beta"] = torch.tensor([1.0 / 1.5, 1.0 / 2.0])
        model(dict(batch))
        batch["beta"] = torch.tensor([1.0 / 1.5, 1.0 / 1.5])
        with self.assertRaisesRegex(ValueError, "unit contract"):
            model(batch)

    def test_batch_thermal_wavelength_overrides_the_stored_default(self):
        """Lambda is per-state data, unlike k_B; the buffer is only a default.

        Lambda = h / sqrt(2*pi*m*k_B*T) varies with temperature and species
        mass, so a batch may carry its own and it wins. Pinned here so the
        asymmetry with the Boltzmann constant above stays deliberate.
        """

        model = self._model(boltzmann_constant=1.0, thermal_wavelength=0.3)
        data = TestGridCACEModel._make_data(self)
        stored = model(dict(data))["beta_F_id"]
        same = model(dict(data, thermal_wavelength=torch.tensor([0.3])))["beta_F_id"]
        torch.testing.assert_close(same, stored, atol=0.0, rtol=0.0)
        other = model(dict(data, thermal_wavelength=torch.tensor([1.0])))["beta_F_id"]
        self.assertNotEqual(other.item(), stored.item())
        # The documented way to make data follow the model.
        from equicdft.data.extxyz import normalize_grid_info

        self.assertEqual(
            normalize_grid_info(model.grid_info)["thermal_wavelength"],
            model.thermal_wavelength.tolist(),
        )


class TestOutputKeys(unittest.TestCase):
    """Chemical potentials belong to solver results, not intrinsic outputs."""

    def test_canonical_keys_only(self):
        from equicdft import GridSolver

        model = GridCACEModel(
            a_features=_DensityFeatures(),
            b_features=_IdentityModule(),
            readout=[_FirstFeatureReadout()],
            grid_spacing=0.5,
            boltzmann_constant=1.0,
        )
        data = TestGridCACEModel._make_data(self)
        data["V_ext"] = 0.1 * torch.rand_like(data["rho"])
        outputs = model(dict(data))
        self.assertIs(type(outputs), dict)
        for key in ("beta_mu_local", "beta_mu_average", "beta_mu_weights"):
            self.assertNotIn(key, outputs)
        for old in ("local_chemical_potential", "average_chemical_potential", "chemical_potential_weights"):
            self.assertNotIn(old, outputs)
        data["mu"] = torch.tensor([0.0])
        result = GridSolver(model).solve(
            data,
            convergence=ConvergenceOptions(max_iter=2),
        )
        self.assertIs(type(result), dict)
        self.assertIn("beta_mu", result)
        self.assertNotIn("equilibrium_chemical_potential", result)


if __name__ == "__main__":
    unittest.main()
