"""The step-5 package layout keeps every old import path working.

Each module that existed at the flat layout (equicdft 0.0.1) is either a
package that re-exports the same names or a deprecated shim that does so
with an ``EquicdftDeprecationWarning``. ``legacy._MODULE_ALIASES`` maps the
old paths for whole-object pickles, so the shims can be deleted in 0.2.0
without breaking those files.
"""

import importlib
import subprocess
import sys
import unittest
import warnings

import equicdft
from equicdft._deprecation import EquicdftDeprecationWarning
from equicdft.legacy import _CLASS_ALIASES, _MODULE_ALIASES


# Public names each flat-layout module offered, recorded before the moves.
OLD_MODULE_NAMES = {
    "equicdft.data": [
        "DEFAULT_BOLTZMANN_CONSTANT", "FourierResponseData", "GridData",
        "default_data_key", "read_metal_sites",
    ],
    "equicdft.energy": ["EnergyReadout", "FREE_ENERGY_MODES", "ReadoutContext"],
    "equicdft.features": ["CartesianAFeatures"],
    "equicdft.interaction": ["BChiMessage"],
    "equicdft.loader": ["LoaderResult", "make_dataloaders"],
    "equicdft.loss": ["FourierResponseLoss", "Loss", "LossTerm", "TensorLoss"],
    "equicdft.metalwall": ["MetalWall"],
    "equicdft.metrics": [
        "FourierResponseMetrics", "METRIC_LABELS", "MetricCollection", "Metrics",
        "SUPPORTED_METRICS", "compute_metric", "format_metric_value", "metric_label",
    ],
    "equicdft.pairwise": ["PairwiseReadout"],
    "equicdft.polarization_ideal": [
        "FixedDipoleIdeal", "dipole_alignment", "inverse_langevin", "langevin", "log_sinhc",
    ],
    "equicdft.readout": ["BulkReadout", "LocalReadout", "LongRangeReadout"],
    "equicdft.reciprocal": ["ReciprocalFeatures"],
    "equicdft.response": ["FourierResponse"],
    "equicdft.semilocal": ["GGAReadout", "LDAReadout", "periodic_gradient_energy_density"],
    "equicdft.solver": ["GridSolver"],
    "equicdft.stability": ["FourierStabilityLoss"],
    "equicdft.stencil": [
        "GridOrder", "canonical_grid_order", "coarsen_grid", "get_neighbor_indices",
        "grid_size_array", "grid_spacing_array", "make_stencil",
    ],
    "equicdft.symmetrize": ["CartesianBFeatures"],
    "equicdft.thermodynamics": [
        "beta_external_potential", "density_weighted_integral", "equilibrium_log_density",
        "ideal_free_energy", "local_chemical_potential", "log_dimensionless_density",
    ],
    "equicdft.trainer": ["Trainer", "TrainingStream"],
}

# Old paths that became packages keep working silently; the rest are shims.
PACKAGES = {"equicdft.data", "equicdft.response", "equicdft.solver"}


class TestPackageLayout(unittest.TestCase):
    def test_old_paths_offer_their_old_names(self):
        for old, names in OLD_MODULE_NAMES.items():
            with self.subTest(module=old):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EquicdftDeprecationWarning)
                    module = importlib.import_module(old)
                missing = [name for name in names if not hasattr(module, name)]
                self.assertEqual(missing, [], "{} lost {}".format(old, missing))

    def test_shims_warn_once_on_import_and_packages_do_not(self):
        for old in OLD_MODULE_NAMES:
            with self.subTest(module=old):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    module = importlib.import_module(old)
                    importlib.reload(module)  # a cached module would not warn again
                ours = [w for w in caught if issubclass(w.category, EquicdftDeprecationWarning)]
                if old in PACKAGES:
                    self.assertEqual(ours, [])
                else:
                    self.assertEqual(len(ours), 1, old)
                    self.assertIn("moved to", str(ours[0].message))

    def test_shim_targets_are_the_new_homes(self):
        for old in OLD_MODULE_NAMES:
            if old in PACKAGES:
                continue
            with self.subTest(module=old):
                self.assertIn(old, _MODULE_ALIASES, "no pickle alias for {}".format(old))
                new = importlib.import_module(_MODULE_ALIASES[old])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EquicdftDeprecationWarning)
                    shim = importlib.import_module(old)
                for name in shim.__all__:
                    target, attribute = _CLASS_ALIASES.get((old, name), (_MODULE_ALIASES[old], name))
                    home = importlib.import_module(target)
                    self.assertIs(getattr(shim, name), getattr(home, attribute), name)
                for name in getattr(new, "__all__", ()):
                    self.assertTrue(hasattr(shim, name), "{} lacks {}".format(old, name))

    def test_every_alias_target_exists(self):
        for old, new in _MODULE_ALIASES.items():
            with self.subTest(alias=old):
                importlib.import_module(new)
        for (old_module, old_name), (new_module, new_name) in _CLASS_ALIASES.items():
            with self.subTest(alias=old_module + "." + old_name):
                self.assertTrue(hasattr(importlib.import_module(new_module), new_name))

    def test_public_api_does_not_import_shims(self):
        # `import equicdft` must be warning-free: the package imports its
        # members from their new homes only.
        code = (
            "import warnings; warnings.simplefilter('error');"
            " import equicdft; import equicdft.data, equicdft.response, equicdft.solver;"
            " print(len(equicdft.__all__))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(int(result.stdout.strip()), len(equicdft.__all__))

    def test_console_script_module_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "equicdft.data.mean_density", "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
