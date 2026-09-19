"""The compatibility helpers of the 0.1.0 deprecation cycle."""

import pickle
import unittest
import warnings

import torch

from equicdft import (
    GGAReadout,
    LDAReadout,
    LongRangeReadout,
    PairwiseReadout,
    TensorLoss,
)
from equicdft._deprecation import AliasedOutputs, EquicdftDeprecationWarning


class TestKeywordOnly(unittest.TestCase):
    def test_keyword_calls_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", EquicdftDeprecationWarning)
            LDAReadout(2, hidden_sizes=(4,), zero_init=True)
            PairwiseReadout(2, 2, hidden_sizes=(3,))
            TensorLoss("energy", "beta_F_exc", "beta_F_exc", weight=2.0)
            GGAReadout(n_types=2, hidden_sizes=(3,))

    def test_legacy_positional_calls_bind_in_the_old_order_and_warn(self):
        with self.assertWarnsRegex(EquicdftDeprecationWarning, "LDAReadout\\(\\) takes 1 positional"):
            readout = LDAReadout(2, (4,), True)
        self.assertEqual((readout.n_types, tuple(readout.hidden_sizes), readout.zero_init), (2, (4,), True))
        # GGAReadout reordered its keywords (n_types first) but the old
        # positional order (hidden_sizes, n_features, n_types) still binds.
        with self.assertWarnsRegex(EquicdftDeprecationWarning, "hidden_sizes, n_features, n_types"):
            gga = GGAReadout((3,), None, 2)
        self.assertEqual((tuple(gga.hidden_sizes), gga.n_types), ((3,), 2))
        with self.assertWarns(EquicdftDeprecationWarning):
            long_range = LongRangeReadout(1, 2, (3,), False)
        self.assertEqual((long_range.n_kernels, long_range.n_types, tuple(long_range.hidden_sizes), long_range.zero_init), (1, 2, (3,), False))

    def test_too_many_or_conflicting_positionals_raise(self):
        with self.assertRaisesRegex(TypeError, "at most"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EquicdftDeprecationWarning)
                LDAReadout(2, (4,), True, None, "extra")
        with self.assertRaisesRegex(TypeError, "multiple values for argument 'hidden_sizes'"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EquicdftDeprecationWarning)
                LDAReadout(2, (4,), hidden_sizes=(8,))

    def test_configuration_round_trip_is_unaffected(self):
        readout = GGAReadout(n_types=2, hidden_sizes=(3,), initial_coefficient=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", EquicdftDeprecationWarning)
            rebuilt = GGAReadout.from_config(readout.to_config())
        self.assertEqual(rebuilt.to_config(), readout.to_config())


class TestLegacyNoiseKeywords(unittest.TestCase):
    def test_trainer_and_stream_keywords_map_onto_noise_options(self):
        from torch import nn
        from torch.utils.data import DataLoader

        from equicdft import Loss, NoiseOptions, Trainer, TrainingStream

        loss = Loss([TensorLoss("x", "prediction", "target")])
        model = nn.Linear(1, 1)
        with self.assertWarnsRegex(EquicdftDeprecationWarning, "noise=NoiseOptions"):
            trainer = Trainer(model, loss, density_noise=True, density_noise_floor=0.1)
        self.assertEqual(trainer.noise, NoiseOptions(density=True, density_floor=0.1))
        with self.assertWarnsRegex(EquicdftDeprecationWarning, "Trainer.density_noise is deprecated"):
            self.assertTrue(trainer.density_noise)
        loader = DataLoader([{"x": torch.ones(1), "target": torch.ones(1)}])
        with self.assertWarns(EquicdftDeprecationWarning):
            stream = TrainingStream("s", loader, loader, loss, polarization_noise=True, noise_dipole_magnitude=2.0)
        self.assertEqual(stream.noise, NoiseOptions(polarization=True, dipole_magnitude=2.0))
        with self.assertRaisesRegex(TypeError, "not both"):
            Trainer(model, loss, noise=NoiseOptions(), density_noise=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", EquicdftDeprecationWarning)
            Trainer(model, loss, noise=NoiseOptions(density=True))


class TestAliasedOutputs(unittest.TestCase):
    def setUp(self):
        self.outputs = AliasedOutputs.with_aliases(
            {"beta_mu_local": torch.ones(2), "c1": torch.zeros(2)},
            {"local_chemical_potential": "beta_mu_local", "unused": "absent"},
        )

    def test_aliases_are_added_only_for_present_keys(self):
        self.assertEqual(set(self.outputs), {"beta_mu_local", "c1", "local_chemical_potential"})
        self.assertIs(dict.__getitem__(self.outputs, "local_chemical_potential"), self.outputs["beta_mu_local"])

    def test_alias_reads_warn_and_canonical_reads_do_not(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", EquicdftDeprecationWarning)
            self.outputs["beta_mu_local"]
            self.outputs.get("c1")
            self.assertIn("local_chemical_potential", self.outputs)
            dict(self.outputs)
        with self.assertWarnsRegex(EquicdftDeprecationWarning, "use 'beta_mu_local'"):
            self.outputs["local_chemical_potential"]
        with self.assertWarns(EquicdftDeprecationWarning):
            self.outputs.get("local_chemical_potential")

    def test_copy_and_pickle_keep_the_aliases(self):
        for clone in (self.outputs.copy(), pickle.loads(pickle.dumps(self.outputs))):
            self.assertIsInstance(clone, AliasedOutputs)
            with self.assertWarns(EquicdftDeprecationWarning):
                clone["local_chemical_potential"]


if __name__ == "__main__":
    unittest.main()
