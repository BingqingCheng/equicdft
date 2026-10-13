"""Exact zero-polarization specialization, without changing finite-P physics."""

import copy
import io
from types import MethodType
import unittest
from unittest.mock import patch

import torch
from torch import nn

from equicdft import (
    FixedDipoleIdeal, GridCACEModel, LDAReadout, LongRangeReadout,
    ReciprocalFeatures,
)
from equicdft._config import build
from equicdft.response.bulk import bulk_response_symbol
from equicdft.response.curvature import _energy_only_model


GRID = 4
SPACING = 1.5


def bulk(n_types=1):
    axes = torch.arange(GRID)
    positions = torch.stack(torch.meshgrid(axes, axes, axes, indexing="ij"), dim=-1).reshape(-1, 3)
    density = torch.linspace(0.4, 0.53, n_types)
    return {
        "rho": density.expand(1, GRID ** 3, n_types).clone(),
        "dipole_density": torch.zeros(1, GRID ** 3, n_types, 3),
        "temperature": torch.ones(1),
        "grid_size": torch.tensor([[GRID] * 3]),
        "grid_spacing": torch.full((1, 3), SPACING),
        "grid_positions": positions[None],
    }


def model(n_types=1, density_power=0.5, divergence=True, features_type=ReciprocalFeatures,
          model_type=GridCACEModel):
    torch.manual_seed(721)
    moments = torch.linspace(0.7, 0.9, n_types)
    features = features_type(
        radial_exponents=(0.7, 2.0), kernel="gaussian", n_types=n_types,
        variable="dipole_density", include_divergence=divergence,
        density_exponent=density_power,
        reference_density=0.4 if density_power else None,
        density_floor=0.01, alignment_scaling="ideal",
        saturation="gaussian", saturation_initial=0.3,
        dipole_magnitude=moments,
    )
    reciprocal = LongRangeReadout(
        n_types=n_types, hidden_sizes=(5, 3), zero_init=False,
        features=features, coefficient_scale=0.1,
    )
    coulomb = LongRangeReadout(
        n_types=n_types, charges=(0.0,) * n_types,
        include_polarization=True, coulomb_amplitude=0.15,
        features=ReciprocalFeatures(
            radial_exponents=(0.0,), kernel="coulomb", n_types=n_types,
        ),
    )
    return model_type(
        readout=[LDAReadout(n_types=n_types, hidden_sizes=(5, 3), zero_init=False), coulomb, reciprocal],
        ideal=FixedDipoleIdeal(moments), grid_spacing=SPACING,
        mean_density=0.4, polarization_scale=0.1, mean_temperature=1.0,
        boltzmann_constant=1.0, rho_min=1.0e-4,
        compute_c1=False, compute_polarization_derivative=False,
    ).double().train()


def vectors():
    # Includes k=0, even-grid Nyquist derivatives, off-axis and negative modes.
    return torch.tensor([[[0, 0, 0], [1, 0, 0], [1, 1, 0],
                          [-1, 0, 1], [2, 0, 0], [2, 1, 2]]])


def symbol_and_gradients(fitted, batch, fast):
    symbol = bulk_response_symbol(fitted, batch, vectors(), use_fast_path=fast, create_graph=True)
    weights = torch.linspace(0.1, 1.0, symbol.numel()).reshape(symbol.shape)
    objective = (symbol.real * weights).sum() + (symbol.imag * weights.flip(-1)).sum()
    objective = objective + 0.07 * symbol.abs().square().sum()
    parameters = tuple(fitted.named_parameters())
    gradients = torch.autograd.grad(objective, tuple(p for _, p in parameters), allow_unused=True)
    return symbol.detach(), dict(zip((n for n, _ in parameters), gradients))


class LegacyModel(nn.Module):
    """An arbitrary model whose forward does not accept the new keyword."""

    def __init__(self):
        super().__init__()
        self.ideal = FixedDipoleIdeal(0.7)
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, batch, compute_c1=False, compute_ideal=False):
        local = self.scale * (batch["rho"].square().sum(-1)
                              + batch["dipole_density"].square().sum((-2, -1)))
        return {"beta_F_exc": SPACING ** 3 * local.sum(-1)}


class LegacyFeatures(ReciprocalFeatures):
    """A legacy subclass with the old forward signature must retain fallback."""

    def forward(self, rho, grid_size, grid_spacing, dipole_density=None, charges=None):
        return super().forward(rho, grid_size, grid_spacing, dipole_density, charges)


class LegacyGridCACEModel(GridCACEModel):
    """An external subclass may override forward using the previous signature."""

    def forward(self, data, compute_c1=None, compute_c2=None, c2_reference=(0, 0),
                compute_polarization_derivative=None, compute_ideal=None,
                compute_local_mu=None, use_voxel_mask=False):
        return super().forward(
            data, compute_c1=compute_c1, compute_c2=compute_c2,
            c2_reference=c2_reference,
            compute_polarization_derivative=compute_polarization_derivative,
            compute_ideal=compute_ideal, compute_local_mu=compute_local_mu,
            use_voxel_mask=use_voxel_mask,
        )


class TestBulkFastPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.get_default_dtype()
        cls.threads = torch.get_num_threads()
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls.dtype)
        torch.set_num_threads(cls.threads)

    def assert_same_gradients(self, fast, reference):
        self.assertEqual(set(fast), set(reference))
        for name in fast:
            with self.subTest(parameter=name):
                self.assertEqual(fast[name] is None, reference[name] is None)
                if fast[name] is not None:
                    self.assertTrue(torch.isfinite(fast[name]).all())
                    torch.testing.assert_close(fast[name], reference[name], rtol=2e-10, atol=2e-11)

    def test_symbol_and_every_parameter_gradient_match(self):
        for n_types in (1, 2):
            for density_power in (0.0, 0.5):
                for divergence in (False, True):
                    with self.subTest(n_types=n_types, density_power=density_power, divergence=divergence):
                        fitted, batch = model(n_types, density_power, divergence), bulk(n_types)
                        fast, fast_grad = symbol_and_gradients(fitted, batch, True)
                        reference, reference_grad = symbol_and_gradients(fitted, batch, False)
                        torch.testing.assert_close(fast, reference, rtol=0, atol=2e-11)
                        self.assert_same_gradients(fast_grad, reference_grad)
                        fading = [name for name in fast_grad if name.endswith("saturation_parameter")]
                        self.assertEqual(len(fading), 1)
                        self.assertIsNotNone(fast_grad[fading[0]])
                        torch.testing.assert_close(fast_grad[fading[0]], torch.zeros_like(fast_grad[fading[0]]), rtol=0, atol=0)

    def test_fast_path_does_not_call_alignment_weight(self):
        fitted, batch = model(), bulk()
        features = fitted.readout[-1].features
        with patch.object(features, "alignment_weight", side_effect=AssertionError("slow branch called")):
            symbol_and_gradients(fitted, batch, True)
            with self.assertRaisesRegex(AssertionError, "slow branch called"):
                bulk_response_symbol(fitted, batch, vectors(), use_fast_path=False)

    def test_finite_p_normal_forward_is_unchanged_and_flags_do_not_leak(self):
        fitted, reference = model(), model()
        batch = bulk()
        batch["dipole_density"][..., 0] = 0.08 * batch["rho"]
        before = _energy_only_model(fitted, batch)["beta_F_exc"]
        with self.assertRaisesRegex(ValueError, "exactly zero polarization"):
            fitted(batch, bulk_second_variation=True)
        features = fitted.readout[-1].features
        with self.assertRaisesRegex(ValueError, "exactly zero polarization"):
            features(batch["rho"], batch["grid_size"], batch["grid_spacing"],
                     batch["dipole_density"], bulk_second_variation=True)
        bulk_response_symbol(fitted, bulk(), vectors())
        after = _energy_only_model(fitted, batch)["beta_F_exc"]
        expected = _energy_only_model(reference, batch)["beta_F_exc"]
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(after, expected, rtol=0, atol=0)
        self.assertNotIn("bulk_second_variation", fitted.to_config())
        self.assertFalse(any("bulk_second_variation" in name for name in fitted.state_dict()))

    def test_strict_checkpoint_and_config_roundtrip(self):
        fitted = model(2)
        config = copy.deepcopy(fitted.to_config())
        state = copy.deepcopy(fitted.state_dict())
        bulk_response_symbol(fitted, bulk(2), vectors())
        self.assertEqual(config, fitted.to_config())
        self.assertEqual(set(state), set(fitted.state_dict()))
        for name in state:
            torch.testing.assert_close(state[name], fitted.state_dict()[name], rtol=0, atol=0)
        stream = io.BytesIO()
        torch.save(dict(config=config, state=state), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        rebuilt = build(saved["config"])
        rebuilt.load_state_dict(saved["state"], strict=True)
        rebuilt.train()
        actual, actual_grad = symbol_and_gradients(rebuilt, bulk(2), True)
        expected, expected_grad = symbol_and_gradients(fitted, bulk(2), False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-11)
        self.assert_same_gradients(actual_grad, expected_grad)

    def test_generic_model_and_legacy_feature_subclass_fall_back(self):
        for fitted in (LegacyModel(), model(features_type=LegacyFeatures),
                       model(model_type=LegacyGridCACEModel)):
            with self.subTest(model_type=type(fitted).__name__):
                fast, fast_grad = symbol_and_gradients(fitted, bulk(), True)
                reference, reference_grad = symbol_and_gradients(fitted, bulk(), False)
                torch.testing.assert_close(fast, reference, rtol=0, atol=2e-11)
                self.assert_same_gradients(fast_grad, reference_grad)

    def test_instance_forward_overrides_preserve_old_signature_fallback(self):
        def old_model_forward(self, data, compute_c1=None, compute_c2=None,
                              c2_reference=(0, 0), compute_polarization_derivative=None,
                              compute_ideal=None, compute_local_mu=None,
                              use_voxel_mask=False):
            return GridCACEModel.forward(
                self, data, compute_c1=compute_c1, compute_c2=compute_c2,
                c2_reference=c2_reference,
                compute_polarization_derivative=compute_polarization_derivative,
                compute_ideal=compute_ideal, compute_local_mu=compute_local_mu,
                use_voxel_mask=use_voxel_mask,
            )

        def old_feature_forward(self, rho, grid_size, grid_spacing,
                                dipole_density=None, charges=None):
            return ReciprocalFeatures.forward(
                self, rho, grid_size, grid_spacing, dipole_density, charges,
            )

        for override in ("model", "features"):
            with self.subTest(instance_override=override):
                fitted = model()
                features = fitted.readout[-1].features
                if override == "model":
                    fitted.forward = MethodType(old_model_forward, fitted)
                else:
                    features.forward = MethodType(old_feature_forward, features)
                with patch.object(features, "alignment_weight", wraps=features.alignment_weight) as weight:
                    fast, fast_grad = symbol_and_gradients(fitted, bulk(), True)
                    self.assertGreater(weight.call_count, 0)
                reference, reference_grad = symbol_and_gradients(fitted, bulk(), False)
                torch.testing.assert_close(fast, reference, rtol=0, atol=2e-11)
                self.assert_same_gradients(fast_grad, reference_grad)

    def test_invalid_flag_is_not_silently_coerced(self):
        with self.assertRaisesRegex(TypeError, "Boolean"):
            bulk_response_symbol(model(), bulk(), vectors(), use_fast_path=1)


if __name__ == "__main__":
    unittest.main()
