"""Bounded integration checks for explicit Cartesian model regridding."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from equicdft import (
    BChiMessage, CACEBranch, CartesianAFeatures, CartesianBFeatures, GGAReadout,
    GridCACEModel, LDAReadout, LocalReadout, LongRangeReadout,
    PairwiseReadout, ReciprocalFeatures, load_model, save_model,
)
from equicdft.serialization import default_dtype
from tests.model_fixtures import grid_data


def _model(*, messages=None, backend="fft", n_types=2, cutoff_grid=1,
           long_range=False, smoothing=None, feature_options=None, extras=()):
    options = dict(
        max_power=2, mean_density=0.4, cutoff_grid=cutoff_grid,
        radial_basis="gaussian", radial_exponents=(0.15, 0.6),
        trainable_radial_exponents=True, separate_center=True,
        n_types=n_types, convolution_backend=backend,
    )
    options.update(feature_options or {})
    features = CartesianAFeatures(**options)
    invariants = CartesianBFeatures(
        max_power=2, max_product_order=2,
        include_polarization=features.include_polarization,
        separate_center=features.separate_center,
    )
    layers = []
    for kind in messages or ():
        settings = dict(
            n_invariant_features=invariants.n_features,
            n_radial_channels=features.n_radial_channels,
            n_channels=features.n_output_channels,
            hidden_sizes=(3,), convolution_backend=backend,
        )
        if kind == "gaussian":
            settings.update(radial_exponents=(0.3, 0.8),
                            trainable_radial_exponents=True)
        elif kind != "shared":
            raise ValueError(kind)
        layers.append(BChiMessage(**settings))
    readouts = [LDAReadout(n_types=n_types, hidden_sizes=(3,)),
                LocalReadout(n_types=n_types, hidden_sizes=(3,))]
    if long_range:
        readouts.append(LongRangeReadout(
            n_kernels=1, n_types=n_types, hidden_sizes=(3,),
            charges=(1.0, -1.0), coulomb_amplitude=1.5,
            features=ReciprocalFeatures(kernel="coulomb", radial_exponents=(0.3,),
                                        n_types=n_types),
        ))
    readouts.extend(extras)
    return GridCACEModel(
        a_features=features, b_features=invariants, readout=readouts,
        grid_spacing=0.5, mean_temperature=1.2, boltzmann_constant=1.0,
        thermal_wavelength=1.0, compute_c1=True, free_energy_mode="beta",
        message_layers=layers, learned_input_smoothing=smoothing,
        polarization_scale=1.0 if features.include_polarization else None,
    ).eval()


def _data(model, shape=(4, 4, 4), seed=1):
    h = float(model.grid_spacing[0])
    result = grid_data(shape=shape, cutoff_grid=model.cutoff_grid,
                       n_types=model.n_types, grid_spacing=h, seed=seed)
    return {key: value.to(dtype=model.grid_spacing.dtype)
            if value.is_floating_point() else value for key, value in result.items()}


def _outputs(model, data):
    values = model(copy.deepcopy(data))
    return {key: values[key].detach() for key in ("beta_F_exc", "c1")}


def _assert_state_equal(test, left, right):
    test.assertEqual(set(left), set(right))
    for name in left:
        test.assertTrue(torch.equal(left[name], right[name]), name)


class TestModelRegrid(unittest.TestCase):
    def setUp(self):
        self.dtype_context = default_dtype(torch.float64)
        self.dtype_context.__enter__()
        torch.manual_seed(17)

    def tearDown(self):
        self.dtype_context.__exit__(None, None, None)

    def test_native_reference_transfer_preserves_values_derivatives_and_source(self):
        native = _model()
        before = {name: value.clone() for name, value in native.state_dict().items()}
        data = _data(native)
        expected = _outputs(native, data)
        controlled = native.regrid(0.5, center_normalization="reference")

        self.assertIsNot(controlled, native)
        _assert_state_equal(self, before, native.state_dict())
        for key, actual in _outputs(controlled, data).items():
            self.assertTrue(torch.equal(actual, expected[key]), key)
        self.assertEqual(native.a_features.center_normalization, "strict")

    def test_default_transfer_keeps_strict_center_exclusion(self):
        fine = _model().regrid(0.25)
        self.assertEqual(fine.a_features.center_normalization, "strict")
        center = fine.a_features.local_density_positions.eq(0).all(1)
        self.assertTrue(torch.equal(fine.a_features.stencil_basis()[center],
                                    torch.zeros_like(fine.a_features.stencil_basis()[center])))

    def test_regrid_preserves_physical_local_exponents_and_monomial_units(self):
        native = _model()
        fine = native.regrid(0.25, center_normalization="reference")
        torch.testing.assert_close(fine.a_features.radial_exponents / 0.25**2,
                                   native.a_features.radial_exponents / 0.5**2)
        self.assertEqual(fine.a_features.cutoff_grid, 2)
        self.assertAlmostEqual(fine.a_features.stencil_radius * 0.25, 0.5)
        self.assertAlmostEqual(fine.a_features.monomial_scale, 0.5)

    def test_undeclared_manual_monomial_rescaling_is_not_silently_reset(self):
        native = _model()
        nonconstant = native.a_features.powers.sum(dim=1) > 0
        # Mimic an old script that changed the stored moments but left no
        # corresponding monomial_scale in the constructor/configuration.
        native.a_features.monomial_values[:, nonconstant] *= 0.5
        with self.assertRaises(ValueError):
            native.regrid(0.25, center_normalization="reference")

    def test_cutoff_scaled_monomials_keep_their_original_physical_units(self):
        native = _model(cutoff_grid=3, feature_options={"coordinate_scaling": "cutoff"})
        coarse = native.regrid(1.0, center_normalization="reference")
        self.assertEqual(coarse.a_features.cutoff_grid, 2)
        self.assertAlmostEqual(coarse.a_features.monomial_scale, 4.0 / 3.0)
        old_point = native.a_features.local_density_positions.eq(
            torch.tensor([2, 0, 0])
        ).all(dim=1)
        new_point = coarse.a_features.local_density_positions.eq(
            torch.tensor([1, 0, 0])
        ).all(dim=1)
        # These offsets both represent displacement (1,0,0) in physical units.
        torch.testing.assert_close(coarse.a_features.monomial_values[new_point],
                                   native.a_features.monomial_values[old_point],
                                   rtol=2e-13, atol=2e-14)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cutoff-scaled.pt"
            save_model(coarse, path)
            restored = load_model(path)
        self.assertEqual(restored.a_features.coordinate_scaling, "cutoff")
        self.assertAlmostEqual(restored.a_features.monomial_scale, 4.0 / 3.0)
        torch.testing.assert_close(restored.a_features.stencil_basis(),
                                   coarse.a_features.stencil_basis(), rtol=0.0, atol=0.0)

    def test_float32_constructed_then_double_geometry_is_accepted(self):
        with default_dtype(torch.float32):
            native = _model(cutoff_grid=3, feature_options={"coordinate_scaling": "cutoff"})
        native = native.double()
        fine = native.regrid(0.25, center_normalization="reference")
        self.assertEqual(fine.grid_spacing.dtype, torch.float64)
        self.assertAlmostEqual(fine.a_features.monomial_scale, 1.0)
        self.assertTrue(torch.isfinite(fine.a_features.stencil_basis()).all().item())

    def test_noninteger_radius_and_geometry_survive_config_and_model_roundtrip(self):
        native = _model(cutoff_grid=3)
        coarse = native.regrid(1.0, center_normalization="reference")
        self.assertEqual(coarse.a_features.cutoff_grid, 2)
        self.assertAlmostEqual(coarse.a_features.stencil_radius, 1.5)
        mask = coarse.a_features.neighbor_mask.clone()
        radii = coarse.a_features.local_density_positions.square().sum(1)
        expected_mask = (radii > 0) & (radii <= 1.5**2)
        self.assertTrue(torch.equal(mask, expected_mask))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coarse.pt"
            save_model(coarse, path)
            restored = load_model(path)
        self.assertTrue(torch.equal(restored.a_features.neighbor_mask, mask))
        self.assertEqual(restored.to_config(), coarse.to_config())
        torch.testing.assert_close(restored.a_features.stencil_basis(),
                                   coarse.a_features.stencil_basis(), rtol=0.0, atol=0.0)
        new = restored.regrid(0.25, center_normalization="reference")
        direct = native.regrid(0.25, center_normalization="reference")
        torch.testing.assert_close(new.a_features.stencil_basis(),
                                   direct.a_features.stencil_basis(), rtol=2e-13, atol=2e-14)

    def test_model_config_rejects_reference_spacing_disagreeing_with_model_grid(self):
        fine = _model().regrid(0.25, center_normalization="reference")
        config = fine.to_config()
        # Keep the internally consistent A geometry/ratio but change the
        # absolute reference spacing: reference_h*ratio no longer equals h.
        config["a_features"]["normalization_reference"]["grid_spacing"] = 1.0
        with self.assertRaises(ValueError):
            GridCACEModel.from_config(config)

    def test_padded_original_stencil_is_preserved_at_native_and_on_restoration(self):
        native = _model(cutoff_grid=4, feature_options={"stencil_radius": 1.5})
        identity = native.regrid(0.5, center_normalization="reference")
        self.assertEqual(identity.a_features.cutoff_grid, 4)
        self.assertTrue(torch.equal(identity.a_features.neighbor_mask,
                                    native.a_features.neighbor_mask))
        self.assertTrue(torch.equal(identity.a_features.stencil_basis(),
                                    native.a_features.stencil_basis()))
        fine = native.regrid(0.25, center_normalization="reference")
        restored = fine.regrid(0.5, center_normalization="reference")
        self.assertEqual(restored.a_features.cutoff_grid, 4)
        self.assertAlmostEqual(restored.a_features.stencil_radius, 1.5)
        self.assertTrue(torch.equal(restored.a_features.local_density_positions,
                                    native.a_features.local_density_positions))
        self.assertTrue(torch.equal(restored.a_features.neighbor_mask,
                                    native.a_features.neighbor_mask))
        torch.testing.assert_close(restored.a_features.stencil_basis(),
                                   native.a_features.stencil_basis(), rtol=2e-13, atol=2e-14)

    def test_chained_transfer_and_restoration_do_not_compound_correction(self):
        # Keep noncentral sites even on h=1: a radius 0.5 stencil would have
        # none and should be rejected rather than normalized artificially.
        native = _model(cutoff_grid=3)
        direct = native.regrid(1.0, center_normalization="reference")
        chained = native.regrid(0.25, center_normalization="reference").regrid(
            1.0, center_normalization="reference")
        torch.testing.assert_close(chained.a_features.stencil_basis(),
                                   direct.a_features.stencil_basis(), rtol=2e-13, atol=2e-14)
        restored = chained.regrid(0.5, center_normalization="reference")
        torch.testing.assert_close(restored.a_features.stencil_basis(),
                                   native.a_features.stencil_basis(), rtol=2e-13, atol=2e-14)
        for key, actual in _outputs(restored, _data(restored)).items():
            torch.testing.assert_close(actual, _outputs(native, _data(native))[key],
                                       rtol=2e-12, atol=2e-13)

    def test_shared_and_independent_messages_receive_correct_basis_once(self):
        native = _model(messages=("shared", "gaussian"))
        fine = native.regrid(0.25, center_normalization="reference")
        shared = fine.a_features.stencil_basis()
        shared_message, independent_message = fine.message_layers
        self.assertTrue(torch.equal(shared_message.stencil_basis(fine.a_features, shared), shared))
        torch.testing.assert_close(independent_message.radial_exponents / 0.25**2,
                                   native.message_layers[1].radial_exponents / 0.5**2)
        independent = independent_message.stencil_basis(fine.a_features, shared)
        expected = fine.a_features.stencil_basis(independent_message.radial_exponents,
                                                 independent_message.radial_centers)
        torch.testing.assert_close(independent, expected, rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(independent, shared))
        self.assertTrue(torch.isfinite(fine(_data(fine))["c1"]).all().item())

    def test_fft_and_gather_use_same_compensated_operator(self):
        native = _model()
        fft = native.regrid(0.25, center_normalization="reference", convolution_backend="fft")
        gather = native.regrid(0.25, center_normalization="reference", convolution_backend="gather")
        data = _data(fft)
        expected = _outputs(fft, data)
        for key, actual in _outputs(gather, data).items():
            torch.testing.assert_close(actual, expected[key], rtol=2e-12, atol=2e-13)

    def test_long_range_pointwise_ideal_and_input_smoothing_are_not_rescaled(self):
        native = _model(long_range=True, smoothing=0.1)
        fine = native.regrid(0.25, center_normalization="reference")
        self.assertEqual(fine.learned_input_smoothing, native.learned_input_smoothing)
        self.assertEqual(fine.ideal.to_config(), native.ideal.to_config())
        for original, actual in zip(native.readout, fine.readout):
            self.assertEqual(actual.to_config(), original.to_config())
            _assert_state_equal(self, actual.state_dict(), original.state_dict())
        torch.testing.assert_close(fine.readout[-1].features.radial_exponents,
                                   native.readout[-1].features.radial_exponents,
                                   rtol=0.0, atol=0.0)

    def test_train_mode_and_parameter_requires_grad_are_preserved(self):
        native = _model().train()
        native.a_features.log_radial_exponents.requires_grad_(False)
        next(native.readout[0].parameters()).requires_grad_(False)
        expected = {name: value.requires_grad for name, value in native.named_parameters()}
        result = native.regrid(0.25, center_normalization="reference")
        self.assertTrue(result.training)
        self.assertEqual({name: value.requires_grad for name, value in result.named_parameters()}, expected)

    def test_density_derivative_matches_directional_energy_difference(self):
        model = _model(smoothing=0.1).regrid(0.25, center_normalization="reference")
        data = _data(model)
        direction = torch.linspace(-0.15, 0.15, data["rho"].numel()).reshape_as(data["rho"])
        output = model(copy.deepcopy(data))
        # c1 = -(1/DeltaV) d(beta F_exc)/d rho for both species.
        automatic = -model.voxel_volume * (output["c1"] * direction).sum()
        step = 1e-5
        energies = []
        for sign in (1.0, -1.0):
            perturbed = copy.deepcopy(data)
            perturbed["rho"] += sign * step * direction
            energies.append(model(perturbed)["beta_F_exc"].detach().sum())
        finite = (energies[0] - energies[1]) / (2 * step)
        torch.testing.assert_close(automatic.detach(), finite, rtol=2e-7, atol=2e-9)

    def test_grid_mismatch_check_is_not_disabled_by_reference_policy(self):
        model = _model().regrid(0.25, center_normalization="reference")
        data = _data(model)
        data["grid_spacing"].fill_(0.5)
        with self.assertRaises(ValueError):
            model(data)

    def test_invalid_spacings_and_policy_are_rejected(self):
        model = _model()
        for h in (0.0, -0.25, float("nan"), float("inf"), (0.25, 0.5, 0.25),
                  (0.25, 0.25, 0.25 + 1e-14)):
            with self.subTest(h=h):
                with self.assertRaises((ValueError, TypeError, NotImplementedError)):
                    model.regrid(h, center_normalization="reference")
        with self.assertRaises((ValueError, TypeError)):
            model.regrid(0.25, center_normalization="automatic")

    def test_unsupported_local_kernels_and_readouts_are_rejected(self):
        cases = (
            dict(feature_options={"radial_centers": (0.0, 0.5)}),
            dict(feature_options={"trainable_radial_centers": True}),
            dict(feature_options={"n_radial_channels": 2}),
            dict(feature_options={"separate_center": False}),
            dict(feature_options={"include_polarization": True}),
            dict(cutoff_grid=2, feature_options={
                "radial_basis": "bessel", "radial_exponents": None,
                "trainable_radial_exponents": False, "n_radial_functions": 2,
            }),
            dict(extras=(GGAReadout(n_types=2, hidden_sizes=(3,)),)),
            dict(cutoff_grid=2, extras=(
                PairwiseReadout(n_types=2, cutoff_grid=2, hidden_sizes=(3,)),
            )),
        )
        for settings in cases:
            with self.subTest(settings=settings):
                native = _model(**settings)
                with self.assertRaises((ValueError, NotImplementedError)):
                    native.regrid(0.25, center_normalization="reference")

    def test_independent_cace_branch_is_not_silently_left_on_old_grid(self):
        branch_features = CartesianAFeatures(
            max_power=1, mean_density=0.4, cutoff_grid=1, n_types=2,
        )
        branch = CACEBranch(
            branch_features, CartesianBFeatures(max_power=1, max_product_order=2),
            LocalReadout(n_types=2, hidden_sizes=(3,)),
        )
        native = _model(extras=(branch,))
        with self.assertRaises((ValueError, NotImplementedError)):
            native.regrid(0.25, center_normalization="reference")

    def test_empty_noncentral_stencil_is_rejected(self):
        native = _model(cutoff_grid=0)
        with self.assertRaises((ValueError, NotImplementedError)):
            native.regrid(0.25, center_normalization="reference")

    def test_equal_weight_polynomial_limit_is_supported(self):
        native = _model(feature_options={"radial_basis": "none", "radial_exponents": None,
                                        "trainable_radial_exponents": False})
        fine = native.regrid(0.25, center_normalization="reference")
        self.assertEqual(fine.a_features.n_radial_channels, 1)
        scalar = fine.a_features.powers.eq(0).all(1)
        torch.testing.assert_close(fine.a_features.stencil_basis()[:, :, scalar].sum(0),
                                   torch.ones((1, 1)), rtol=1e-13, atol=1e-13)
        self.assertTrue(torch.isfinite(fine(_data(fine))["c1"]).all().item())

    def test_fixed_gaussian_exponents_are_rescaled_without_modifying_source(self):
        native = _model(feature_options={"trainable_radial_exponents": False})
        before = native.a_features.fixed_radial_exponents.clone()
        fine = native.regrid(0.25, center_normalization="reference")
        torch.testing.assert_close(fine.a_features.fixed_radial_exponents, before * 0.25,
                                   rtol=0, atol=0)
        self.assertTrue(torch.equal(native.a_features.fixed_radial_exponents, before))
        self.assertTrue(torch.isfinite(fine(_data(fine))["c1"]).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_reference_geometry_follows_cuda_device_and_dtype(self):
        cpu = _model().regrid(0.25, center_normalization="reference")
        expected = _outputs(cpu, _data(cpu))
        cuda = copy.deepcopy(cpu).cuda()
        data = {name: value.cuda() for name, value in _data(cpu).items()}
        for key, actual in _outputs(cuda, data).items():
            torch.testing.assert_close(actual.cpu(), expected[key], rtol=2e-11, atol=2e-12)


if __name__ == "__main__":
    unittest.main()
