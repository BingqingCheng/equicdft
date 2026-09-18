"""SEM proposals, coupled ideal targets, and train-only reproducible use."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from equicdft import FixedDipoleIdeal, Loss, TensorLoss, Trainer, TrainingStream
from equicdft._density_noise import add_field_noise


class RecordingPolarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.seen = []

    def forward(self, batch):
        self.seen.append((self.training, batch["dipole_density"].detach().clone()))
        return {"prediction": self.weight * batch["dipole_density"]}


def loss():
    return Loss([TensorLoss("P", "prediction", "target")])


def loader():
    return DataLoader([
        dict(rho=torch.ones(4, 2), dipole_density=torch.full((4, 2, 3), .2),
             dipole_density_std=torch.full((4, 2, 3), .15),
             target=torch.zeros(4, 2, 3)) for _ in range(2)
    ], batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(2))


def trainer(**kwargs):
    return Trainer(RecordingPolarModel(), loss(), optimizer_cls=torch.optim.SGD,
                   optimizer_args={"lr": .01}, device="cpu", **kwargs)


class TestPolarizationNoise(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.get_default_dtype()
        self.rng = torch.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(17)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)
        torch.set_rng_state(self.rng)

    @staticmethod
    def batch():
        rho = torch.full((2, 5, 2), 2.)
        polar = torch.full((*rho.shape, 3), .1)
        return dict(rho=rho, dipole_density=polar, rho_std=rho*.05,
                    dipole_density_std=polar*.2,
                    valid=torch.ones(2, 5, dtype=torch.bool))

    def test_literal_sem_three_components_per_species_and_immutable_mask(self):
        batch = self.batch()
        batch["valid"][:, 0] = False
        batch["rho"][:, 0] = 0.0  # Non-target boundary cells stay untouched.
        batch["dipole_density"][:, 0] = 0.0
        original = {k: v.clone() for k, v in batch.items()}
        with patch("torch.randn_like", side_effect=lambda x: torch.ones_like(x)):
            out = add_field_noise(batch, polarization_noise=True, dipole_magnitude=[.5, .8])
        expected = batch["dipole_density"] + batch["dipole_density_std"]
        expected[:, 0] = 0.
        torch.testing.assert_close(out["dipole_density"], expected)
        self.assertTrue(torch.equal(out["rho"], batch["rho"]))
        self.assertIs(out["valid"], batch["valid"])
        for key in batch:
            self.assertTrue(torch.equal(batch[key], original[key]))
        self.assertFalse(torch.equal(out["dipole_density"].sum(1), batch["dipole_density"].sum(1)))

    def test_proposal_moments_and_zero_sem_no_draws(self):
        rho = torch.ones(1, 50000, 2)*10
        polar = torch.zeros(*rho.shape, 3)
        std = torch.tensor([[.1, .2, .3], [.2, .4, .5]]).expand_as(polar)
        batch = dict(rho=rho, dipole_density=polar, dipole_density_std=std)
        a = add_field_noise(batch, polarization_noise=True, dipole_magnitude=1.)
        b = add_field_noise(batch, polarization_noise=True, dipole_magnitude=1.)
        torch.testing.assert_close(a["dipole_density"].mean(1), torch.zeros(1, 2, 3), atol=.006, rtol=0)
        torch.testing.assert_close(a["dipole_density"].std(1), std[:, 0], atol=.006, rtol=0)
        self.assertFalse(torch.equal(a["dipole_density"], b["dipole_density"]))
        batch["dipole_density_std"] = torch.zeros_like(polar)
        with patch("torch.randn_like", side_effect=AssertionError("unexpected draw")):
            out = add_field_noise(batch, polarization_noise=True, dipole_magnitude=1.)
            self.assertTrue(torch.equal(out["dipole_density"], polar))
            self.assertIs(add_field_noise(batch), batch)

    def test_whole_vector_rejection_not_component_clipping(self):
        batch = dict(rho=torch.ones(1, 1, 1), dipole_density=torch.zeros(1, 1, 1, 3),
                     dipole_density_std=torch.ones(1, 1, 1, 3))
        with patch("torch.randn_like", side_effect=[torch.tensor([[2., .2, .3]]),
                                                    torch.tensor([[.1, .2, .4]])]) as draws:
            out = add_field_noise(batch, polarization_noise=True, dipole_magnitude=1.)
        self.assertEqual(draws.call_count, 2)
        torch.testing.assert_close(out["dipole_density"].flatten(), torch.tensor([.1, .2, .4]))
        with patch("torch.randn_like", side_effect=lambda x: torch.full_like(x, 2.)):
            with self.assertRaisesRegex(RuntimeError, "1000 attempts"):
                add_field_noise(batch, polarization_noise=True, dipole_magnitude=1.)

    def test_joint_density_proposal_rejected_when_it_breaks_P_bound(self):
        batch = dict(rho=torch.ones(1, 1, 1), rho_std=torch.ones(1, 1, 1),
                     dipole_density=torch.tensor([[[[.5, 0., 0.]]]]))
        with patch("torch.randn_like", side_effect=[torch.tensor([-.9]), torch.tensor([.2])]):
            out = add_field_noise(batch, density_noise=True, dipole_magnitude=1.)
        torch.testing.assert_close(out["rho"], torch.full_like(batch["rho"], 1.2))
        self.assertTrue(torch.equal(out["dipole_density"], batch["dipole_density"]))

    def test_retry_keeps_accepted_cells_and_accepts_on_last_attempt(self):
        polar = torch.zeros(1, 3, 1, 3)
        std = torch.ones_like(polar)
        std[:, 2] = 0.0
        batch = dict(rho=torch.ones(1, 3, 1), dipole_density=polar,
                     dipole_density_std=std)
        draws = [torch.tensor([[.1, .2, .3], [2., 0., 0.]]),
                 torch.tensor([[.2, .3, .4]])]
        with patch("equicdft._density_noise._MAX_NOISE_ATTEMPTS", 2):
            with patch("torch.randn_like", side_effect=draws) as random:
                out = add_field_noise(batch, polarization_noise=True, dipole_magnitude=1.)
        self.assertEqual([call.args[0].shape for call in random.call_args_list],
                         [torch.Size([2, 3]), torch.Size([1, 3])])
        torch.testing.assert_close(out["dipole_density"][0, :, 0],
            torch.tensor([[.1, .2, .3], [.2, .3, .4], [0., 0., 0.]]))

    def test_coupled_targets_refresh_for_P_only_and_joint_noise(self):
        for density_noise in (False, True):
            for packed in (False, True):
                with self.subTest(density_noise=density_noise, packed=packed):
                    batch = self.batch()
                    batch["valid"][:, 0] = False
                    moment = [.5, .8]
                    ideal = FixedDipoleIdeal(moment, thermal_wavelength=[2., 3.])
                    before = ideal(batch["rho"], batch["dipole_density"], 1.)
                    scalar = before["density_derivative"] + .7
                    vector = .3 - before["polarization_derivative"]
                    batch["c1_plus_beta_mu"] = scalar.clone()
                    batch["c1"] = scalar - .2
                    batch["polarization_derivative"] = vector.clone()
                    batch["target_c1"] = scalar[:, 1:].clone() if packed else scalar.clone()
                    batch["target_P_derivative"] = vector[:, 1:].clone() if packed else vector.clone()
                    with patch("torch.randn_like", side_effect=lambda x: torch.ones_like(x)):
                        out = add_field_noise(batch, density_noise=density_noise,
                                              polarization_noise=True, dipole_magnitude=moment)
                    after = ideal(out["rho"], out["dipole_density"], 1.)
                    expected_scalar = after["density_derivative"] + .7
                    expected_vector = .3 - after["polarization_derivative"]
                    torch.testing.assert_close(out["c1_plus_beta_mu"], expected_scalar)
                    torch.testing.assert_close(out["c1"], expected_scalar - .2)
                    torch.testing.assert_close(out["polarization_derivative"], expected_vector)
                    torch.testing.assert_close(out["target_c1"], expected_scalar[:, 1:] if packed else expected_scalar)
                    torch.testing.assert_close(out["target_P_derivative"], expected_vector[:, 1:] if packed else expected_vector)
                    self.assertFalse(torch.equal(out["target_c1"], batch["target_c1"]))
                    self.assertFalse(torch.equal(out["target_P_derivative"], batch["target_P_derivative"]))

    def test_invalid_sem_domain_moment_and_target_shapes(self):
        batch = self.batch()
        for key, value in (("dipole_density_std", torch.ones(3)),
                           ("dipole_density_std", -batch["dipole_density_std"]),
                           ("dipole_density_std", batch["dipole_density_std"]*float("nan")),
                           ("target_P_derivative", torch.ones(2, 5)),
                           ("valid", torch.ones(2)),
                           ("dipole_density", batch["dipole_density"]*100)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                add_field_noise(dict(batch, **{key: value}), polarization_noise=True, dipole_magnitude=1.)
        for value in (None, -1., [1., 2., 3.]):
            with self.assertRaises(ValueError):
                add_field_noise(batch, polarization_noise=True, dipole_magnitude=value)
        batch.pop("dipole_density_std")
        with self.assertRaisesRegex(KeyError, "dipole_density_std"):
            add_field_noise(batch, polarization_noise=True, dipole_magnitude=1.)
        batch.pop("dipole_density")
        with self.assertRaisesRegex(KeyError, "dipole_density"):
            add_field_noise(batch, polarization_noise=True, dipole_magnitude=1.)

    def test_masked_noninterior_and_excluded_zeros_preserved(self):
        batch = self.batch()
        batch["rho"][:, 0] = 0
        batch["dipole_density"][:, 0] = 0
        batch["rho_std"][:, 0] = 0
        batch["dipole_density_std"][:, 0] = 0
        batch["excluded_mask"] = torch.zeros(2, 5, dtype=torch.bool)
        batch["excluded_mask"][:, 0] = True
        batch["valid"][:, 1] = False
        batch["dipole_density"][:, 1] = 100  # Explicitly not a finite ideal target.
        out = add_field_noise(batch, polarization_noise=True, dipole_magnitude=1.)
        self.assertTrue(torch.equal(out["dipole_density"][:, :2], batch["dipole_density"][:, :2]))
        batch["dipole_density_std"][:, 0] = 1.
        with self.assertRaisesRegex(ValueError, "zero at excluded"):
            add_field_noise(batch, polarization_noise=True, dipole_magnitude=1.)

    def test_train_only_clean_evaluation_and_joint_stream_opt_in(self):
        t = trainer(polarization_noise=True, noise_dipole_magnitude=1.)
        with patch("torch.randn_like", side_effect=lambda x: torch.ones_like(x)):
            t.fit(loader(), loader(), epochs=1, verbose=False)
        for is_training, polar in t.model.seen:
            torch.testing.assert_close(polar, torch.full_like(polar, .35 if is_training else .2))
        t.model.seen.clear()
        t.evaluate_streams([TrainingStream("P", loader(), loader(), loss(),
                            polarization_noise=True, noise_dipole_magnitude=1.)], subset="train")
        for is_training, polar in t.model.seen:
            self.assertFalse(is_training)
            torch.testing.assert_close(polar, torch.full_like(polar, .2))
        t = trainer()
        with patch("torch.randn_like", side_effect=AssertionError("unexpected noise")):
            t.fit(loader(), loader(), epochs=1, verbose=False)
        t = trainer()
        streams = [TrainingStream("P", loader(), loader(), loss(), polarization_noise=True,
                                  noise_dipole_magnitude=1.),
                   TrainingStream("clean", loader(), loader(), loss())]
        with patch("torch.randn_like", side_effect=lambda x: torch.ones_like(x)) as draws:
            t.fit_streams(streams, epochs=1, verbose=False)
        self.assertEqual(draws.call_count, 2)

    def test_checkpoint_continuation_ordinary_and_joint_is_exact(self):
        for joint in (False, True):
            with self.subTest(joint=joint), tempfile.TemporaryDirectory() as tmp:
                def setup(path=None, enabled=True):
                    t = trainer(checkpoint_dir=path, polarization_noise=enabled,
                                noise_dipole_magnitude=.5 if enabled else None)
                    streams = [TrainingStream("P", loader(), loader(), loss(),
                               polarization_noise=enabled,
                               noise_dipole_magnitude=.5 if enabled else None)]
                    return t, streams

                def fit(t, streams, epochs):
                    if joint:
                        t.fit_streams(streams, epochs=epochs, verbose=False)
                    else:
                        t.fit(streams[0].train_loader, streams[0].valid_loader, epochs=epochs, verbose=False)

                torch.manual_seed(7)
                full, streams = setup()
                fit(full, streams, 3)
                torch.manual_seed(7)
                first, streams = setup(tmp)
                fit(first, streams, 1)
                resumed, streams = setup(enabled=False)
                if joint:
                    resumed.load_stream_checkpoint(Path(tmp)/"last.pt", streams)
                    self.assertTrue(streams[0].polarization_noise)
                else:
                    resumed.load_checkpoint(Path(tmp)/"last.pt", streams[0].train_loader)
                    self.assertTrue(resumed.polarization_noise)
                fit(resumed, streams, 2)
                self.assertTrue(torch.equal(full.model.weight, resumed.model.weight))
                self.assertEqual(full.history, resumed.history)
                checkpoint = torch.load(Path(tmp)/"last.pt", weights_only=False)
                for key in ("polarization_noise", "noise_dipole_magnitude", "stream_polarization_noise"):
                    checkpoint.pop(key)
                torch.save(checkpoint, Path(tmp)/"legacy.pt")
                if joint:
                    resumed.load_stream_checkpoint(Path(tmp)/"legacy.pt", streams)
                    self.assertFalse(streams[0].polarization_noise)
                else:
                    resumed.load_checkpoint(Path(tmp)/"legacy.pt", streams[0].train_loader)
                    self.assertFalse(resumed.polarization_noise)

    def test_complete_functional_training_has_finite_gradients(self):
        from test_polarization_symmetry import TestFullPolarizationSymmetry

        model = TestFullPolarizationSymmetry.model("fft", 0.)
        data = TestFullPolarizationSymmetry.data()
        data["dipole_density_std"] = torch.full_like(data["dipole_density"], .01)
        # Moment 2 leaves all original/perturbed fixture values well inside.
        data["target_c1"] = torch.zeros_like(data["rho"])
        data["target_P_derivative"] = torch.zeros_like(data["dipole_density"])
        t = Trainer(model, Loss([TensorLoss("rho", "c1", "target_c1"),
                                TensorLoss("P", "polarization_derivative", "target_P_derivative")]),
                    polarization_noise=True, noise_dipole_magnitude=2., device="cpu")
        dl = DataLoader([data])
        t.fit(dl, dl, epochs=1, verbose=False)
        self.assertTrue(torch.isfinite(torch.tensor(t.history[0]["train_losses"]["total"])))
        for readout in model.readout:
            grads = [p.grad for p in readout.parameters() if p.requires_grad and p.grad is not None]
            # The fixed Coulomb branch has no trainable coefficients.
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
