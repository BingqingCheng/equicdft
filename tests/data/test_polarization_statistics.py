"""Train-plus-validation, rotation-invariant population polarization scale."""

import copy
import unittest

import torch

from equicdft import make_dataloaders


def frame(vectors):
    polar = torch.tensor(vectors, dtype=torch.float64).reshape(-1, 1, 3)
    return dict(rho=torch.zeros(polar.shape[:-1]), dipole_density=polar)


class TestPolarizationStatistics(unittest.TestCase):
    def statistic(self, train, valid):
        return make_dataloaders(train, valid_dataset=valid,
                                compute_polarization_std=True)["polarization_std"]

    def test_pool_train_and_validation_but_never_test(self):
        train = [frame([[1, 2, 3], [1, 2, 3]])]
        valid = [frame([[4, 5, 6]])]
        test = [frame([[1e9, 0, 0]])]
        result = make_dataloaders(train, valid_dataset=valid, test_dataset=test,
                                  compute_polarization_std=True)
        vectors = torch.cat([f["dipole_density"].reshape(-1, 3) for f in train + valid])
        expected = vectors.var(dim=0, unbiased=False).mean().sqrt().item()
        self.assertAlmostEqual(result["polarization_std"], expected, places=14)
        self.assertNotIn("polarization_std", make_dataloaders(train, valid_dataset=valid))

    def test_fractional_split_seed_does_not_change_statistics_or_rng(self):
        data = [frame([[j, j+1, 2*j]]) for j in range(5)]
        rng = torch.get_rng_state()
        first = make_dataloaders(data, valid_fraction=.4, seed=1, compute_polarization_std=True)
        second = make_dataloaders(data, valid_fraction=.2, seed=7, compute_polarization_std=True)
        self.assertEqual(first["polarization_std"], second["polarization_std"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_rotation_translation_and_species_pooling(self):
        data = [frame([[1, 2, 3], [4, -2, 1], [3, 5, 9], [2, 7, -1]])]
        data[0]["dipole_density"] = data[0]["dipole_density"].reshape(2, 2, 3)
        data[0]["rho"] = torch.ones(2, 2)
        rotation = torch.tensor([[.6, -.8, 0], [.8, .6, 0], [0, 0, -1]], dtype=torch.float64)
        changed = copy.deepcopy(data)
        changed[0]["dipole_density"] = data[0]["dipole_density"] @ rotation.T + 100.
        self.assertAlmostEqual(self.statistic(data, data), self.statistic(changed, changed), places=13)

    def test_masks_exclude_invalid_values_without_density_cutoff(self):
        data = frame([[1, 0, 0], [3, 0, 0], [float("nan"), 0, 0], [100, 0, 0]])
        data["excluded_mask"] = torch.tensor([False, False, True, False])
        data["valid"] = torch.tensor([True, True, True, False])
        before = data["dipole_density"].clone()
        self.assertAlmostEqual(self.statistic([data], [data]), (1/3)**.5, places=14)
        torch.testing.assert_close(before, data["dipole_density"], equal_nan=True)

    def test_constant_or_zero_polarization_returns_zero(self):
        for vector in ([0, 0, 0], [1, 2, 3]):
            self.assertEqual(self.statistic([frame([vector])], [frame([vector])]), 0.)

    def test_missing_invalid_empty_and_nonfinite_data_rejected(self):
        good = frame([[1, 0, 0]])
        bad = [dict(rho=good["rho"]),
               dict(good, dipole_density=torch.zeros(1, 3)),
               dict(good, dipole_density=torch.full((1, 1, 3), float("nan"))),
               dict(good, valid=torch.ones(1)),
               dict(good, valid=torch.tensor([False]))]
        for value in bad:
            with self.subTest(value=value), self.assertRaises((KeyError, ValueError)):
                self.statistic([value], [value])
        with self.assertRaises(TypeError):
            make_dataloaders([good], valid_dataset=[good], compute_polarization_std=1)


if __name__ == "__main__":
    unittest.main()
