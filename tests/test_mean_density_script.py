import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ase import Atoms
from ase.io import write
import numpy as np


class TestMeanDensityScript(unittest.TestCase):
    def test_framewise_mean_densities(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "density.extxyz"
            frames = []
            for values in ([1.0, 3.0], [2.0, 6.0]):
                atoms = Atoms("He2", positions=[[0, 0, 0], [1, 0, 0]])
                atoms.arrays["density"] = np.asarray(values)
                frames.append(atoms)
            write(path, frames, format="extxyz")

            script = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "mean_density.py"
            )
            result = subprocess.run(
                [sys.executable, str(script), str(path)],
                check=True,
                capture_output=True,
                text=True,
            )

            repository = Path(__file__).resolve().parents[1]
            imported = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "from scripts.mean_density import "
                        "compute_mean_densities; "
                        "print(json.dumps(compute_mean_densities({!r})))"
                    ).format(str(path)),
                ],
                cwd=str(repository),
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(json.loads(result.stdout), [2.0, 4.0])
        self.assertEqual(json.loads(imported.stdout), [2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
