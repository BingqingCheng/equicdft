"""Keep the bundled revised-paper model identifiable and numerically stable."""

import contextlib
import hashlib
import io
import json
import unittest
from unittest.mock import patch

import torch

from examples.lj_paper_v2 import evaluate


class TestLJPaperV2Example(unittest.TestCase):
    def test_export_and_field_hashes(self):
        record = json.loads((evaluate.HERE / "provenance.json").read_text())
        for name in ("model.pt", "field.extxyz"):
            digest = hashlib.sha256((evaluate.HERE / name).read_bytes()).hexdigest()
            self.assertEqual(digest, record["artifacts"][name]["sha256"])

    def test_native_forward(self):
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        capture = io.StringIO()
        with patch("sys.argv", ["evaluate.py"]), contextlib.redirect_stdout(capture):
            evaluate.main()
        result = json.loads(capture.getvalue())
        self.assertEqual(result["c1_shape"], [4096, 1])
        self.assertAlmostEqual(result["particle_number"], 208.0, delta=1e-8)
        self.assertAlmostEqual(result["beta_F_exc"], -604.7519242625531, delta=1e-8)
        self.assertAlmostEqual(result["gauge_aligned_beta_V_RMSE"],
                               0.026188112132846143, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
