"""The package layout: flat 0.0.x module paths are gone, pickles still convert.

Whole-object model files written under the flat layout record module paths
such as ``equicdft.features``; ``legacy._MODULE_ALIASES`` and
``_CLASS_ALIASES`` map every one of them to the current home so
``python -m equicdft.convert`` keeps working without shim modules.
"""

import importlib
import pathlib
import subprocess
import sys
import unittest

import equicdft
from equicdft.legacy import _CLASS_ALIASES, _MODULE_ALIASES, _RemappingUnpickler


# Flat-layout module -> public classes it held (for the pickle remapping).
OLD_MODULE_CLASSES = {
    "equicdft.data": ["GridData", "FourierResponseData"],
    "equicdft.energy": ["EnergyReadout", "ReadoutContext"],
    "equicdft.features": ["CartesianAFeatures"],
    "equicdft.interaction": ["BChiMessage"],
    "equicdft.loss": ["LossTerm", "TensorLoss", "Loss", "FourierResponseLoss"],
    "equicdft.metalwall": ["MetalWall"],
    "equicdft.metrics": ["MetricCollection", "Metrics", "FourierResponseMetrics"],
    "equicdft.pairwise": ["PairwiseReadout"],
    "equicdft.polarization_ideal": ["FixedDipoleIdeal"],
    "equicdft.readout": ["LocalReadout", "BulkReadout", "LongRangeReadout"],
    "equicdft.reciprocal": ["ReciprocalFeatures"],
    "equicdft.response": ["FourierResponse"],
    "equicdft.semilocal": ["LDAReadout", "GGAReadout"],
    "equicdft.solver": ["GridSolver"],
    "equicdft.stability": ["FourierStabilityLoss"],
    "equicdft.symmetrize": ["CartesianBFeatures"],
    "equicdft.trainer": ["Trainer", "TrainingStream"],
}

# Old paths that became packages and still import.
PACKAGES = {"equicdft.data", "equicdft.response", "equicdft.solver"}

REMOVED_MODULES = [
    "equicdft.stencil", "equicdft.features", "equicdft.symmetrize",
    "equicdft.interaction", "equicdft.energy", "equicdft.thermodynamics",
    "equicdft.readout", "equicdft.semilocal", "equicdft.pairwise",
    "equicdft.reciprocal", "equicdft.metalwall", "equicdft.polarization_ideal",
    "equicdft.stability", "equicdft.loss", "equicdft.metrics", "equicdft.trainer",
    "equicdft.loader", "equicdft._grid", "equicdft._fourier",
    "equicdft._data_helpers", "equicdft._argument_checks",
]


class TestPackageLayout(unittest.TestCase):
    def test_flat_module_paths_are_gone(self):
        for name in REMOVED_MODULES:
            with self.subTest(module=name):
                with self.assertRaises(ImportError):
                    importlib.import_module(name)

    def test_packages_keep_their_public_names(self):
        for name in PACKAGES:
            module = importlib.import_module(name)
            for attribute in OLD_MODULE_CLASSES[name]:
                self.assertTrue(hasattr(module, attribute), "{} lacks {}".format(name, attribute))

    def test_pickle_remapping_covers_every_flat_class(self):
        unpickler = _RemappingUnpickler.__new__(_RemappingUnpickler)
        for old, classes in OLD_MODULE_CLASSES.items():
            for name in classes:
                with self.subTest(reference="{}.{}".format(old, name)):
                    resolved = unpickler.find_class(old, name)
                    home = _MODULE_ALIASES.get(old, old)
                    home, attribute = _CLASS_ALIASES.get((old, name), (home, name))
                    self.assertIs(resolved, getattr(importlib.import_module(home), attribute))
                    if hasattr(equicdft, name):
                        self.assertIs(resolved, getattr(equicdft, name))

    def test_every_alias_target_exists(self):
        for old, new in _MODULE_ALIASES.items():
            with self.subTest(alias=old):
                importlib.import_module(new)
        for (old_module, old_name), (new_module, new_name) in _CLASS_ALIASES.items():
            with self.subTest(alias=old_module + "." + old_name):
                self.assertTrue(hasattr(importlib.import_module(new_module), new_name))

    def test_import_is_warning_free(self):
        code = (
            "import warnings; warnings.simplefilter('error');"
            " import equicdft; import equicdft.data, equicdft.response, equicdft.solver;"
            " print(len(equicdft.__all__))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(int(result.stdout.strip()), len(equicdft.__all__))

    def test_objective_bases_are_below_training_and_response(self):
        """The two protocol classes depend on neither package, so no cycle.

        ``response`` subclasses them and the trainer inspects response terms,
        which is a cycle whenever the bases live on the training side. Both
        original module paths still export them.
        """

        from equicdft import objectives
        from equicdft.training import loss as training_loss
        from equicdft.training import metrics as training_metrics
        from equicdft.response import stability

        source = (
            pathlib.Path(objectives.__file__).read_text()
        )
        for forbidden in ("from .training", "from .response",
                          "import equicdft.training", "import equicdft.response"):
            self.assertNotIn(forbidden, source)
        self.assertIs(training_loss.LossTerm, objectives.LossTerm)
        self.assertIs(training_metrics.MetricCollection, objectives.MetricCollection)
        self.assertIs(equicdft.LossTerm, objectives.LossTerm)
        self.assertTrue(issubclass(stability.FourierStabilityLoss, objectives.LossTerm))
        # The trainer no longer needs a function-local import to reach them.
        trainer_source = pathlib.Path(
            importlib.import_module("equicdft.training.trainer").__file__
        ).read_text()
        self.assertNotIn("    from ..response", trainer_source)

    def test_console_script_module_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "equicdft.data.mean_density", "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
