import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from equicdft.evolution import ConstantMobility, Dynamics, EulerIntegrator, SemiImplicitIntegrator


class QuadraticModel(nn.Module):
    """A positive local beta-functional with an exact chemical potential."""

    def __init__(self):
        super().__init__()
        self.register_buffer("curvature", torch.tensor([[1.0, 0.2], [0.2, 1.5]], dtype=torch.float64))
        self.child = nn.Identity()
        self.observed_modes = []
        self.fail = False

    def forward(self, data, **kwargs):
        self.observed_modes.append((self.training, self.child.training))
        if self.fail:
            raise RuntimeError("Model failure for restoration test.")
        rho = data["rho"]
        derivative = rho @ self.curvature
        volume = torch.as_tensor(data["grid_spacing"]).prod()
        return {
            "beta_F": 0.5 * volume * (rho * derivative).sum(),
            "density_derivative": derivative,
        }


def field_data():
    axes = [torch.arange(size) for size in (4, 3, 2)]
    positions = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    x = 2 * torch.pi * positions[:, 0].double() / 4
    y = 2 * torch.pi * positions[:, 1].double() / 3
    rho = torch.stack((1.0 + 0.15 * x.cos(), 0.7 + 0.08 * y.sin()), dim=-1)
    spacing = torch.tensor([0.8, 1.1, 1.3], dtype=torch.float64)
    return {
        "rho": rho.requires_grad_(),
        "grid_size": torch.tensor([4, 3, 2]),
        "grid_positions": positions,
        "grid_spacing": spacing,
        "grid_center": positions.double() * spacing + spacing / 2,
        "temperature": torch.tensor(2.0, dtype=torch.float64),
        "beta": torch.tensor(0.5, dtype=torch.float64),
        "V_ext": torch.stack((0.03 * y.cos(), 0.02 * x.sin()), dim=-1),
    }


class TestDynamics(unittest.TestCase):
    def integrator(self, method=EulerIntegrator):
        self.model = QuadraticModel()
        self.model.train()
        self.model.child.eval()
        return method(self.model, ConstantMobility([0.4, 0.2]), dt=0.02)

    def test_recording_cadence_includes_initial_and_final(self):
        data = field_data()
        trajectory = Dynamics(self.integrator()).run(data, n_steps=7, save_every=3)
        torch.testing.assert_close(trajectory["time"], torch.tensor([0.0, 0.06, 0.12, 0.14], dtype=torch.float64))
        self.assertEqual(trajectory["rho"].shape, (4, 24, 2))
        torch.testing.assert_close(trajectory["rho"][0], data["rho"])
        # A final step already on the recording cadence is saved only once.
        exact = Dynamics(self.integrator()).run(data, n_steps=6, save_every=3)
        self.assertEqual(exact["time"].tolist(), [0.0, 0.06, 0.12])

    def test_zero_steps_records_one_initial_state(self):
        data = field_data()
        trajectory = Dynamics(self.integrator()).run(data, n_steps=0, save_every=100)
        self.assertEqual(trajectory["time"].tolist(), [0.0])
        self.assertEqual(trajectory["rho"].shape[0], 1)
        torch.testing.assert_close(trajectory["rho"][0], data["rho"])

    def test_conservation_energy_detach_input_and_model_modes(self):
        for method in (EulerIntegrator, SemiImplicitIntegrator):
            with self.subTest(method=method.__name__):
                data = field_data()
                original = {key: value.clone() for key, value in data.items()}
                integrator = self.integrator(method)
                trajectory = Dynamics(integrator).run(data, n_steps=12, save_every=2)
                counts = trajectory["particle_numbers"]
                torch.testing.assert_close(counts, counts[0].expand_as(counts), rtol=0, atol=1e-12)
                self.assertTrue(torch.all(torch.diff(trajectory["beta_A"]) <= 1e-12))
                self.assertTrue(torch.all(trajectory["min_density"] > 0))
                for key, value in trajectory.items():
                    if isinstance(value, torch.Tensor):
                        self.assertEqual(value.device.type, "cpu")
                        self.assertFalse(value.requires_grad)
                        self.assertIsNone(value.grad_fn)
                for key in data:
                    torch.testing.assert_close(data[key], original[key])
                self.assertTrue(data["rho"].requires_grad)
                self.assertIsNone(data["rho"].grad)
                self.assertTrue(self.model.training)
                self.assertFalse(self.model.child.training)
                self.assertTrue(self.model.observed_modes)
                self.assertTrue(all(modes == (False, False) for modes in self.model.observed_modes))

    def test_metadata_is_complete_and_independent(self):
        data = field_data()
        trajectory = Dynamics(self.integrator()).run(data, n_steps=1, save_every=7)
        metadata = trajectory["metadata"]
        self.assertEqual(metadata["dt"], 0.02)
        self.assertEqual(metadata["n_steps"], 1)
        self.assertEqual(metadata["save_every"], 7)
        self.assertEqual(metadata["method"], "EulerIntegrator")
        self.assertIs(metadata["deterministic"], True)
        self.assertEqual(metadata["mobility"]["class"], "ConstantMobility")
        torch.testing.assert_close(metadata["mobility"]["value"], torch.tensor([0.4, 0.2]))
        for key in ("grid_size", "grid_spacing", "grid_positions", "grid_center", "temperature", "beta", "V_ext"):
            torch.testing.assert_close(metadata[key], data[key])
            self.assertNotEqual(metadata[key].data_ptr(), data[key].data_ptr())
        saved_external = metadata["V_ext"].clone()
        data["V_ext"].zero_()
        torch.testing.assert_close(metadata["V_ext"], saved_external)

    def test_save_round_trip_and_refuses_overwrite(self):
        trajectory = Dynamics(self.integrator()).run(field_data(), n_steps=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.pt"
            self.assertEqual(Dynamics.save(trajectory, path), path)
            saved_bytes = path.read_bytes()
            loaded = torch.load(path, weights_only=True)
            torch.testing.assert_close(loaded["rho"], trajectory["rho"])
            self.assertEqual(loaded["metadata"]["method"], "EulerIntegrator")
            with self.assertRaises(FileExistsError):
                Dynamics.save(trajectory, path)
            self.assertEqual(path.read_bytes(), saved_bytes)

    def test_model_modes_restored_after_evaluation_failure(self):
        integrator = self.integrator()
        self.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "Model failure"):
            Dynamics(integrator).run(field_data(), n_steps=1)
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.child.training)

    def test_invalid_counts_rejected(self):
        dynamics = Dynamics(self.integrator())
        for n_steps in (-1, 1.2, True):
            with self.subTest(n_steps=n_steps), self.assertRaises(ValueError):
                dynamics.run(field_data(), n_steps=n_steps)
        for save_every in (0, -1, 1.2, True):
            with self.subTest(save_every=save_every), self.assertRaises(ValueError):
                dynamics.run(field_data(), n_steps=1, save_every=save_every)


if __name__ == "__main__":
    unittest.main()
