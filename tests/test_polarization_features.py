"""Unified rho/vector Cartesian moments and generated cubic invariants."""

import io
import itertools
import unittest

import numpy as np
import torch

from equicdft import (
    BChiMessage, CartesianAFeatures, CartesianBFeatures, GridCACEModel, LocalReadout,
)
from equicdft.stencil import get_neighbor_indices


def periodic_field(size=3, cutoff=1, n_types=1):
    positions = np.indices((size, size, size)).reshape(3, -1).T
    neighbors, _ = get_neighbor_indices(positions, cutoff_grid=cutoff)
    generator = torch.Generator().manual_seed(38)
    return {
        "rho": 0.2 + torch.rand(len(positions), n_types, generator=generator, dtype=torch.float64),
        "dipole_density": torch.randn(len(positions), n_types, 3, generator=generator, dtype=torch.float64),
        "local_density_index": torch.tensor(neighbors),
        "grid_positions": torch.tensor(positions),
        "grid_size": torch.tensor([size, size, size]),
        "temperature": torch.tensor(1.3, dtype=torch.float64),
    }, torch.tensor(positions)


class TestPolarizationFeatures(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(38)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)

    def features(self, **kwargs):
        args = dict(mean_density=.7, dipole_density_scale=2.,
                    include_polarization=True, cutoff_grid=1, max_power=1,
                    radial_basis="gaussian", radial_exponents=(.125, .5),
                    trainable_radial_exponents=True)
        args.update(kwargs)
        a = CartesianAFeatures(**args)
        b = CartesianBFeatures(a.max_power, 3, include_polarization=True,
                               separate_center=a.separate_center)
        return a, b

    def test_all_48_actions_on_moments_and_invariants(self):
        for center, power in itertools.product((False, True), (0, 1, 2)):
            a, b = self.features(separate_center=center, max_power=power, n_types=2)
            data, positions = periodic_field(n_types=2)
            moments = a(data)
            reference = b(moments)
            group = 0
            for permutation in itertools.permutations(range(3)):
                for signs in itertools.product((1, -1), repeat=3):
                    rotation = torch.eye(3)[list(permutation)] * torch.tensor(signs)[:, None]
                    transformed_positions = (positions @ rotation.long().T) % 3
                    index = (transformed_positions[:, 0] * 3 + transformed_positions[:, 1]) * 3 + transformed_positions[:, 2]
                    rho = torch.empty_like(data["rho"])
                    dipole = torch.empty_like(data["dipole_density"])
                    rho[index] = data["rho"]
                    dipole[index] = data["dipole_density"] @ rotation.T
                    actual = a(dict(data, rho=rho, dipole_density=dipole))[index]
                    expected = moments.index_select(-2, b.component_indices[group])
                    expected = expected * b.component_signs[group][..., None]
                    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
                    torch.testing.assert_close(b(actual), reference, atol=2e-12, rtol=2e-12)
                    group += 1

    def test_generated_recipes_equal_independent_group_average(self):
        # Direct Reynolds averages, independently applying every group element
        # to each representative product rather than evaluating sparse recipes.
        b = CartesianBFeatures(1, 3, include_polarization=True)
        a = torch.randn(2, 3, 2, b.n_components, 2)
        transformed = a[..., b.component_indices, :] * b.component_signs[..., None]
        expected = []
        for order in (1, 2, 3):
            for representative in getattr(b, "representatives_" + str(order)):
                product = transformed.index_select(-2, representative).prod(-2).mean(-2)
                expected.append(product)
        expected = torch.stack(expected, dim=-2)
        torch.testing.assert_close(b(a), expected, atol=2e-12, rtol=2e-12)

    def test_mixed_products_and_no_extra_dipole_reversal_symmetry(self):
        a, _ = self.features(radial_exponents=(.125,), separate_center=False)
        b = CartesianBFeatures(1, 2, include_polarization=True, separate_center=False)
        data, _ = periodic_field()
        moments = a(data)
        result = b(moments)
        # K=4; zeroth vector slots Px/Py/Pz are 4,8,12. Spatial rho vector
        # slots are 1,2,3. The orbit of rho_x * P_x is v.P / 3.
        reps = getattr(b, "representatives_2").tolist()
        col = b.n_features_by_order[0] + reps.index([1, 4])
        expected = (moments[..., [1, 2, 3], :] * moments[..., [4, 8, 12], :]).sum(-2) / 3
        torch.testing.assert_close(result[..., col, :], expected)
        self.assertFalse(torch.allclose(b(a(dict(data, dipole_density=-data["dipole_density"]))), result))

    def test_shared_basis_direct_sums_center_and_species_transform(self):
        from unittest.mock import patch
        data, _ = periodic_field(n_types=2)
        for center in (False, True):
            a, _ = self.features(
                n_types=2, n_channels=3,
                density_transform=((1., 1.), (1., -1.), (.2, .7)),
                separate_center=center,
            )
            with patch.object(a, "stencil_basis", wraps=a.stencil_basis) as basis:
                actual = a(data)
            basis.assert_called_once()
            fields = [a.transform_density(data["rho"]) / .7]
            fields.extend(a.transform_density(data["dipole_density"][..., axis]) / 2.
                          for axis in range(3))
            expected = []
            for field in fields:
                summed = torch.einsum("gjc,jnk->gnkc", field[data["local_density_index"]], a.stencil_basis())
                if center:
                    local = field[:, None, None, :].expand(-1, a.n_radial_channels, 1, -1)
                    summed = torch.cat((summed, local), dim=-2)
                expected.append(summed)
            torch.testing.assert_close(actual, torch.cat(expected, dim=-2))
            if center:
                width = a.powers.shape[0] + 1
                for component, field in enumerate(fields):
                    torch.testing.assert_close(actual[..., component * width + width - 1, :],
                                               field[:, None, :].expand(-1, a.n_radial_channels, -1))

    def test_fft_gather_all_radial_bases_values_and_gradients(self):
        data, _ = periodic_field(size=5, cutoff=2, n_types=2)
        for basis, center in itertools.product(("none", "gaussian", "bessel"), (False, True)):
            args = dict(max_power=1, cutoff_grid=2, mean_density=.7,
                        include_polarization=True, dipole_density_scale=2.,
                        n_types=2, radial_basis=basis, separate_center=center,
                        density_transform=((1., 1.), (1., -1.)))
            if basis == "gaussian":
                args.update(radial_exponents=(.125, .4), radial_centers=(0., .3),
                            trainable_radial_exponents=True, trainable_radial_centers=True,
                            n_radial_channels=1)
            elif basis == "bessel":
                args.update(n_radial_functions=2, n_radial_channels=1)
            gather = CartesianAFeatures(**args)
            fft = CartesianAFeatures(**args, convolution_backend="fft")
            fft.load_state_dict(gather.state_dict(), strict=True)
            b = CartesianBFeatures(1, 2, include_polarization=True, separate_center=center)
            outputs, gradients = [], []
            for module in (gather, fft):
                inputs = {key: value.detach().clone() for key, value in data.items()}
                inputs["rho"].requires_grad_()
                inputs["dipole_density"].requires_grad_()
                if module is fft:
                    inputs.pop("local_density_index")
                output = b(module(inputs))
                targets = (inputs["rho"], inputs["dipole_density"], *module.parameters())
                grad = torch.autograd.grad(output.square().sum(), targets, create_graph=True)
                second = torch.autograd.grad(sum(g.square().sum() for g in grad), targets)
                outputs.append(output)
                gradients.append((*grad, *second))
            torch.testing.assert_close(*outputs, atol=2e-11, rtol=2e-11)
            for left, right in zip(*gradients):
                torch.testing.assert_close(left, right, atol=2e-8, rtol=2e-10)

    def test_translation_and_batching(self):
        data, positions = periodic_field()
        a, b = self.features()
        shifted_positions = (positions + torch.tensor([1, -1, 2])) % 3
        index = (shifted_positions[:, 0] * 3 + shifted_positions[:, 1]) * 3 + shifted_positions[:, 2]
        shifted = dict(data)
        for key in ("rho", "dipole_density"):
            shifted[key] = torch.empty_like(data[key])
            shifted[key][index] = data[key]
        torch.testing.assert_close(b(a(shifted))[index], b(a(data)))
        batch = {key: torch.stack((data[key], shifted[key])) for key in data}
        torch.testing.assert_close(b(a(batch))[0], b(a(data)))
        torch.testing.assert_close(b(a(batch))[1], b(a(shifted)))

    def test_empty_fields_finite_first_and_second_derivatives(self):
        a, b = self.features()
        data, _ = periodic_field()
        data["rho"] = torch.zeros_like(data["rho"], requires_grad=True)
        data["dipole_density"] = torch.zeros_like(data["dipole_density"], requires_grad=True)
        value = b(a(data)).sum()
        gradients = torch.autograd.grad(value, (data["rho"], data["dipole_density"]), create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in gradients),
                                     (data["rho"], data["dipole_density"]))
        for value in (*gradients, *second):
            self.assertTrue(torch.isfinite(value).all())

    def test_toggle_off_preserves_state_and_ignores_polarization(self):
        data, _ = periodic_field(n_types=2)
        for center in (False, True):
            args = dict(max_power=2, mean_density=.7, n_types=2, cutoff_grid=1,
                        radial_basis="gaussian", radial_exponents=(.125, .5),
                        trainable_radial_exponents=True, separate_center=center)
            default = CartesianAFeatures(**args)
            explicit = CartesianAFeatures(**args, include_polarization=False)
            self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
            self.assertNotIn("dipole_density_scale", explicit.state_dict())
            explicit.load_state_dict(default.state_dict(), strict=True)
            b = CartesianBFeatures(2, 3)
            off = CartesianBFeatures(2, 3, include_polarization=False)
            off.load_state_dict(b.state_dict(), strict=True)
            corrupted = dict(data, dipole_density=torch.tensor(float("nan")))
            torch.testing.assert_close(off(explicit(corrupted)), b(default(data)), atol=0, rtol=0)
            # Full objects predating the new toggle have neither new flag.
            del explicit.include_polarization
            del off.include_polarization
            stream = io.BytesIO()
            torch.save((explicit, off), stream)
            stream.seek(0)
            restored_a, restored_b = torch.load(stream, weights_only=False)
            torch.testing.assert_close(restored_b(restored_a(data)), b(default(data)), atol=0, rtol=0)

    def test_fft_model_derivatives_without_neighbor_table(self):
        a, b = self.features()
        fft_a, fft_b = self.features(convolution_backend="fft")
        models = [
            GridCACEModel(af, bf, [LocalReadout(n_features=2*bf.n_features + 1, hidden_sizes=(4,))],
                          grid_spacing=.5, compute_polarization_derivative=True)
            for af, bf in ((a, b), (fft_a, fft_b))
        ]
        models[1].load_state_dict(models[0].state_dict(), strict=True)
        self.assertFalse(models[1].requires_local_density_index)
        data, _ = periodic_field()
        outputs = []
        for i, model in enumerate(models):
            inputs = {key: value.detach().clone() for key, value in data.items()}
            if i:
                inputs.pop("local_density_index")
            outputs.append(model(inputs))
        for key in outputs[0]:
            torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=2e-11, rtol=2e-11)

    def test_shape_and_configuration_validation(self):
        for args in (dict(include_polarization=True),
                     dict(include_polarization="yes"),
                     dict(dipole_density_scale=1.),
                     dict(include_polarization=True, dipole_density_scale=0.)):
            with self.assertRaises((ValueError, TypeError)):
                CartesianAFeatures(1, .7, **args)
        a, b = self.features()
        for wrong in (CartesianBFeatures(1, 3),
                      CartesianBFeatures(2, 3, include_polarization=True),
                      CartesianBFeatures(1, 3, include_polarization=True, separate_center=False)):
            with self.assertRaisesRegex(ValueError, "A/B"):
                GridCACEModel(a, wrong, [LocalReadout()], grid_spacing=.5)
        with self.assertRaisesRegex(ValueError, "polarized Cartesian moments"):
            GridCACEModel(a, b, [LocalReadout()], grid_spacing=.5,
                          message_layers=[BChiMessage(b.n_features, 2, 1)])
        data, _ = periodic_field()
        missing = dict(data)
        del missing["dipole_density"]
        with self.assertRaisesRegex(ValueError, "dipole_density is required"):
            a(missing)
        for dipole in (data["rho"], data["dipole_density"].float(),
                       torch.full_like(data["dipole_density"], float("nan"))):
            with self.assertRaises(ValueError):
                a(dict(data, dipole_density=dipole))
        model = GridCACEModel(a, b, [LocalReadout(n_features=2*b.n_features + 1)],
                              grid_spacing=.5, compute_polarization_derivative=True)
        self.assertTrue(model.requires_dipole_density)
        self.assertTrue(model.requires_local_density_index)
        self.assertFalse(getattr(model.readout[0], "requires_dipole_density", False))


if __name__ == "__main__":
    unittest.main()
