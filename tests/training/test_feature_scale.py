"""Data-derived standardization of the local readout inputs."""

import copy
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from equicdft import (
    CACEBranch, CartesianAFeatures, CartesianBFeatures, FixedDipoleIdeal,
    GridCACEModel, LDAReadout, LocalReadout, local_feature_offset,
    local_feature_scale, local_feature_statistics, standardize_local_features,
)
from equicdft.geometry.stencil import get_neighbor_indices


class TestLocalFeatureScale(unittest.TestCase):
    def setUp(self):
        self.dtype, self.rng = torch.get_default_dtype(), torch.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(11)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)
        torch.set_rng_state(self.rng)

    @staticmethod
    def branch(polarized):
        a = CartesianAFeatures(
            max_power=1, mean_density=.5, cutoff_grid=1, include_polarization=polarized,
            separate_center=True, radial_basis="gaussian", radial_exponents=(.2,),
            trainable_radial_exponents=True, convolution_backend="fft",
        )
        b = CartesianBFeatures(1, 2, include_polarization=polarized, separate_center=True)
        return CACEBranch(a, b, LocalReadout(hidden_sizes=(5,)))

    def model(self):
        return GridCACEModel(
            readout=[self.branch(False), self.branch(True), LDAReadout()],
            ideal=FixedDipoleIdeal(1.), grid_spacing=.7, mean_density=.5,
            polarization_scale=.3, mean_temperature=1.5, boltzmann_constant=1.,
            rho_min=1.0e-2, compute_c1=False, compute_polarization_derivative=False,
        )

    @staticmethod
    def frames(count, size=4):
        positions = np.indices((size,) * 3).reshape(3, -1).T
        neighbors, _ = get_neighbor_indices(positions, cutoff_grid=1)
        frames = []
        for _ in range(count):
            rho = .4 + .2 * torch.rand(size**3, 1)
            dipole_density = .08 * torch.randn(size**3, 1, 3)
            rho[: size**2 // 2] = 1.0e-3      # a slab below rho_min: not an energy centre
            dipole_density[: size**2 // 2] = 0.0
            frames.append({
                "rho": rho,
                "dipole_density": dipole_density,
                "temperature": torch.tensor(1.5),
                "grid_size": torch.tensor([size] * 3),
                "grid_positions": torch.tensor(positions),
                "local_density_index": torch.tensor(neighbors),
            })
        return frames

    def test_statistics_match_a_direct_evaluation_over_selected_voxels(self):
        model, frames = self.model(), self.frames(3)
        loader = DataLoader(frames, batch_size=1)
        statistics = local_feature_statistics(model, loader)
        self.assertEqual(sorted(statistics), ["readout.0.readout", "readout.1.readout"])
        # Direct evaluation: capture the readout inputs and the selection.
        for name, readout in ((k, model.get_submodule(k)) for k in statistics):
            rows = []
            for frame in frames:
                batch = next(iter(DataLoader([frame], batch_size=1)))
                captured = {}
                handle = readout.register_forward_hook(
                    lambda module, inputs, output: captured.__setitem__("x", inputs[0]))
                with torch.no_grad():
                    outputs = model(batch, use_voxel_mask=True)
                handle.remove()
                selected = outputs["local_energy_mask"].any(dim=-1).reshape(-1)
                rows.append(captured["x"].reshape(-1, captured["x"].shape[-1])[selected])
            rows = torch.cat(rows)
            self.assertEqual(statistics[name]["count"], rows.shape[0])
            self.assertEqual(rows.shape[0], 3 * (64 - 8))
            torch.testing.assert_close(statistics[name]["mean"], rows.mean(0))
            torch.testing.assert_close(statistics[name]["std"], rows.std(0, correction=0))
            # The temperature feature is constant at T/T_ref = 1: zero spread.
            self.assertEqual(statistics[name]["std"][-1].item(), 0.0)
        # Without the mask every voxel counts.
        unmasked = local_feature_statistics(model, loader, use_voxel_mask=False)
        self.assertEqual(unmasked["readout.0.readout"]["count"], 3 * 64)
        self.assertTrue(model.training)

    def test_scale_uses_the_spread_and_leaves_constant_features_alone(self):
        model, loader = self.model(), DataLoader(self.frames(2), batch_size=2)
        statistics = local_feature_statistics(model, loader)
        scales = local_feature_scale(model, loader, statistics=statistics)
        for name, scale in scales.items():
            std = statistics[name]["std"]
            varying = std > 1.0e-4 * std.max()
            torch.testing.assert_close(scale[varying], std[varying])
            self.assertTrue(bool((scale[~varying] == 1.0).all()))
            self.assertEqual(scale[-1].item(), 1.0)
            self.assertGreater(int(varying.sum()), 1)
        # A larger floor demotes more features to a unit scale.
        coarse = local_feature_scale(model, loader, statistics=statistics, relative_floor=0.5)
        for name in scales:
            self.assertGreater(int((coarse[name] == 1.0).sum()), int((scales[name] == 1.0).sum()))
        with self.assertRaisesRegex(ValueError, "relative_floor"):
            local_feature_scale(model, loader, statistics=statistics, relative_floor=1.0)

    def test_standardization_is_recorded_in_the_config_and_can_keep_the_function(self):
        model, frames = self.model(), self.frames(2)
        loader = DataLoader(frames, batch_size=1)
        batch = next(iter(DataLoader(frames, batch_size=2)))
        with torch.no_grad():
            before = model(batch)["beta_F"].clone()
        # Before a fit: the scale changes the (random) function.
        fresh = copy.deepcopy(model)
        scales = standardize_local_features(fresh, loader)
        with torch.no_grad():
            self.assertFalse(torch.allclose(fresh(batch)["beta_F"], before))
        for name, entry in scales.items():
            torch.testing.assert_close(fresh.get_submodule(name).feature_scale, entry["scale"])
            self.assertIsNone(entry["offset"])
            self.assertIsNone(fresh.get_submodule(name).feature_offset)
        config = fresh.to_config()
        rebuilt = GridCACEModel.from_config(config)
        rebuilt.load_state_dict(fresh.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(rebuilt(batch)["beta_F"], fresh(batch)["beta_F"])
        # The state dict carries the scale, so it cannot load silently into
        # an unscaled model of the same architecture.
        self.assertIn("readout.1.readout.feature_scale", fresh.state_dict())
        with self.assertRaisesRegex(RuntimeError, "feature_scale"):
            GridCACEModel.from_config(model.to_config()).load_state_dict(fresh.state_dict(), strict=True)
        # The unscaled model's state loads into an unscaled config unchanged.
        legacy = GridCACEModel.from_config(model.to_config())
        legacy.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(legacy(batch)["beta_F"], before)
        # On a fitted model: keep_function reparametrizes without changing it,
        # with and without centring.
        for center in (False, True):
            kept = copy.deepcopy(model)
            result = standardize_local_features(kept, loader, keep_function=True, center=center)
            with torch.no_grad():
                torch.testing.assert_close(kept(batch)["beta_F"], before)
            readout = kept.get_submodule("readout.1.readout")
            self.assertIsNotNone(readout.feature_scale)
            self.assertEqual(readout.feature_offset is not None, center)
            if center:
                self.assertEqual(readout.feature_offset[-1].item(), 0.0)   # constant temperature feature
                self.assertNotEqual(readout.feature_offset[0].item(), 0.0)
                torch.testing.assert_close(result["readout.1.readout"]["offset"].to(readout.feature_offset), readout.feature_offset)

    def test_centring_uses_the_mean_and_makes_the_inputs_zero_mean_unit_spread(self):
        model, loader = self.model(), DataLoader(self.frames(2), batch_size=1)
        statistics = local_feature_statistics(model, loader)
        scales = local_feature_scale(model, loader, statistics=statistics)
        offsets = local_feature_offset(statistics)
        for name in scales:
            selected = statistics[name]["std"] > 1.0e-4 * statistics[name]["std"].max()
            torch.testing.assert_close(offsets[name][selected], statistics[name]["mean"][selected])
            self.assertTrue(bool((offsets[name][~selected] == 0.0).all()))
        standardize_local_features(model, loader, center=True)
        # The transformed inputs of every varying feature now have mean 0 and
        # population std 1 over the same voxels.
        for name in scales:
            readout = model.get_submodule(name)
            captured = []
            handle = readout.mlp.register_forward_hook(lambda module, inputs, output: captured.append(inputs[0]))
            with torch.no_grad():
                for batch in loader:
                    outputs = model(batch, use_voxel_mask=True)
                    selected = outputs["local_energy_mask"].any(dim=-1).reshape(-1)
                    captured[-1] = captured[-1].reshape(-1, captured[-1].shape[-1])[selected]
            handle.remove()
            rows = torch.cat(captured)
            selected = statistics[name]["std"] > 1.0e-4 * statistics[name]["std"].max()
            torch.testing.assert_close(rows.mean(0)[selected], torch.zeros(int(selected.sum())), atol=1e-10, rtol=0)
            torch.testing.assert_close(rows.std(0, correction=0)[selected], torch.ones(int(selected.sum())))

    def test_centring_does_not_confuse_unit_std_with_an_excluded_feature(self):
        statistics = {
            "readout": {
                "mean": torch.tensor([7.0, 9.0, 11.0]),
                "std": torch.tensor([1.0, 2.0, 1.0e-5]),
            }
        }
        scales = local_feature_scale(
            None, (), statistics=statistics, relative_floor=1.0e-4,
        )
        offsets = local_feature_offset(statistics, relative_floor=1.0e-4)
        torch.testing.assert_close(
            scales["readout"], torch.tensor([1.0, 2.0, 1.0]),
        )
        # The first feature is selected even though its natural scale is one;
        # only the third feature lies below the relative floor.
        torch.testing.assert_close(
            offsets["readout"], torch.tensor([7.0, 9.0, 0.0]),
        )


if __name__ == "__main__":
    unittest.main()
