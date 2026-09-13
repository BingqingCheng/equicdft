"""Data-free O_h regression for the complete scalar/vector free-energy model."""
import copy
import itertools
import unittest

import numpy as np
import torch

from equicdft import (
    CartesianAFeatures, CartesianBFeatures, GridCACEModel, LDAReadout,
    LocalReadout, LongRangeReadout, ReciprocalFeatures,
)
from equicdft.stencil import get_neighbor_indices


class TestFullPolarizationSymmetry(unittest.TestCase):
    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        self.rng_state = torch.random.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(53)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)
        torch.random.set_rng_state(self.rng_state)

    @staticmethod
    def data():
        # Even dimensions deliberately include populated Nyquist planes.
        size = 4
        positions = np.indices((size,) * 3).reshape(3, -1).T
        neighbors, _ = get_neighbor_indices(positions, cutoff_grid=1)
        return {
            "rho": .4 + .2 * torch.rand(size**3, 1),
            "dipole_density": .08 * torch.randn(size**3, 1, 3),
            "temperature": torch.tensor(1.2),
            "grid_size": torch.tensor([size] * 3),
            "grid_positions": torch.tensor(positions),
            "local_density_index": torch.tensor(neighbors),
        }

    @staticmethod
    def model(backend, charge):
        a = CartesianAFeatures(
            max_power=1, mean_density=.5, dipole_density_scale=.3,
            include_polarization=True, separate_center=True, cutoff_grid=1,
            radial_basis="gaussian", radial_exponents=(.125, .5),
            trainable_radial_exponents=True, convolution_backend=backend,
        )
        b = CartesianBFeatures(
            max_power=1, max_product_order=2,
            include_polarization=True, separate_center=True,
        )
        readouts = [
            LocalReadout(n_features=a.n_radial_channels * b.n_features + 1,
                         hidden_sizes=(5,)),
            LDAReadout(mean_density=.5, dipole_density_scale=.3,
                       hidden_sizes=(5,), zero_init=False),
            LongRangeReadout(
                n_kernels=1, charges=(charge,), include_polarization=True,
                coulomb_amplitude=1.8,
                # Modest damping keeps high-frequency LR modes appreciable.
                features=ReciprocalFeatures(
                    radial_exponents=(.15,), kernel="coulomb",
                ),
            ),
        ]
        return GridCACEModel(
            a, b, readouts, grid_spacing=.7, mean_temperature=1.5,
            compute_c1=True, compute_polarization_derivative=True,
        ).eval()

    @staticmethod
    def evaluate(model, data):
        inputs = {key: value.detach().clone() for key, value in data.items()}
        return {key: value.detach() for key, value in model(inputs).items()}

    def test_all_48_actions_with_lda_and_coulomb_on_even_grid(self):
        data = self.data()
        positions = data["grid_positions"].numpy()
        # Guard against accidentally replacing this by a smooth/constant
        # fixture that never exercises FFT Nyquist handling.
        for key in ("rho", "dipole_density"):
            spectrum = torch.fft.fftn(
                data[key].reshape(4, 4, 4, -1), dim=(0, 1, 2),
            )
            for axis in range(3):
                self.assertGreater(spectrum.select(axis, 2).abs().max().item(), .01)

        actions = [
            np.eye(3, dtype=int)[list(permutation)] * np.array(signs)[:, None]
            for permutation in itertools.permutations(range(3))
            for signs in itertools.product((1, -1), repeat=3)
        ]
        self.assertEqual(len({tuple(r.flatten()) for r in actions}), 48)
        self.assertEqual(sum(round(np.linalg.det(r)) == 1 for r in actions), 24)

        # Neutral dipoles and a mixed charge/dipole source; both SR backends.
        for backend, charge in itertools.product(("gather", "fft"), (0., .7)):
            with self.subTest(backend=backend, charge=charge):
                full = self.model(backend, charge)
                models = {"complete": full}
                for index, name in enumerate(("short_range", "LDA", "Coulomb_LR")):
                    branch = copy.deepcopy(full)
                    branch.readout = torch.nn.ModuleList([branch.readout[index]])
                    models[name] = branch
                inputs = dict(data)
                if backend == "fft":
                    inputs.pop("local_density_index")
                reference = {name: self.evaluate(model, inputs)
                             for name, model in models.items()}
                # Nonzero energies AND vector responses ensure no branch is
                # silently disabled (in particular, the LDA final layer).
                for name in ("short_range", "LDA", "Coulomb_LR"):
                    self.assertGreater(reference[name]["beta_F_exc"].abs().item(), 1e-8)
                    self.assertGreater(reference[name]["polarization_derivative"].abs().max().item(), 1e-8)
                for key in reference["complete"]:
                    summed = sum(reference[name][key] for name in models if name != "complete")
                    torch.testing.assert_close(reference["complete"][key], summed,
                                               atol=1e-11, rtol=1e-10)

                for action, rotation in enumerate(actions):
                    # Exact cell-center map r'=Rr modulo the periodic box.
                    centers = ((2 * positions + 1) @ rotation.T) % 8
                    destination = torch.tensor(np.ravel_multi_index(
                        ((centers - 1) // 2).T, (4, 4, 4),
                    ))
                    self.assertEqual(destination.unique().numel(), 64)
                    matrix = torch.tensor(rotation, dtype=torch.float64)
                    transformed = dict(inputs)
                    transformed["rho"] = torch.empty_like(data["rho"])
                    transformed["rho"][destination] = data["rho"]
                    transformed["dipole_density"] = torch.empty_like(data["dipole_density"])
                    transformed["dipole_density"][destination] = data["dipole_density"] @ matrix.T
                    for name, model in models.items():
                        with self.subTest(action=action, branch=name):
                            actual = self.evaluate(model, transformed)
                            for key, expected in reference[name].items():
                                value = actual[key]
                                if key != "beta_F_exc":
                                    value = value[destination]
                                if key == "polarization_derivative":
                                    expected = expected @ matrix.T
                                torch.testing.assert_close(value, expected,
                                                           atol=1e-11, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
