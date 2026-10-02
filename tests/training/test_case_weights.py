"""Per-case loss weights from signal and sampling noise."""

import unittest
import unittest.mock

import torch
from torch import nn
from torch.utils.data import DataLoader

from equicdft import (
    CACEBranch, CartesianAFeatures, CartesianBFeatures, FixedDipoleIdeal,
    GridCACEModel, LDAReadout, Loss, LocalReadout, LossTerm, TensorLoss,
    Trainer, TrainingStream, case_loss_statistics, case_loss_weights,
)
from equicdft.training.case_weights import _perturbation


class ThresholdModel(nn.Module):
    """``prediction = theta * rho`` on the voxels with ``rho > threshold``."""

    def __init__(self, theta=1.0, threshold=0.5):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(theta)))
        self.threshold = threshold

    def forward(self, batch):
        return {
            "prediction": self.theta * batch["rho"],
            "voxel_weights": (batch["rho"] > self.threshold).to(batch["rho"]),
        }


class SlopeModel(nn.Module):
    """``prediction = theta * rho``: one parameter shared by every case."""

    def __init__(self, theta=1.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(theta)))

    def forward(self, batch):
        return {"prediction": self.theta * batch["rho"]}


def slope_case(amplitude, slope, std, field):
    rho = amplitude * field
    return {
        "rho": rho,
        "rho_std": torch.full_like(rho, std),
        "target": slope * rho,
    }


def mse_loss(**kwargs):
    return Loss([TensorLoss("field", "prediction", "target")], **kwargs)


def fitted_slope(model, loss, cases):
    """Exact minimizer of a loss quadratic in theta (one Newton step)."""

    theta = model.theta
    total = sum(
        loss(model(case), case)["total"] for case in cases
    ) / len(cases)
    (gradient,) = torch.autograd.grad(total, theta, create_graph=True)
    (curvature,) = torch.autograd.grad(gradient, theta)
    return float((theta - gradient / curvature).detach())


class TestCaseLossWeights(unittest.TestCase):
    def setUp(self):
        self.dtype, self.rng = torch.get_default_dtype(), torch.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(5)
        self.field = torch.randn(64, 1)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)
        torch.set_rng_state(self.rng)

    def loader(self, cases):
        return DataLoader(cases, batch_size=1, shuffle=False)

    def test_noise_is_the_linear_response_to_the_standard_error(self):
        # prediction = theta rho: J xi = theta sigma xi, so the noise is
        # theta^2 sigma^2 on average and independent of the residual.
        sigma, theta = 0.02, 1.5
        model = SlopeModel(theta)
        cases = [slope_case(1.0, 1.0, sigma, self.field), slope_case(1.0, 4.0, sigma, self.field)]
        runs = [
            case_loss_statistics(
                model, mse_loss(), self.loader([case]), terms=["field"],
                noise_samples=400, seed=3,
            )
            for case in cases
        ]
        result = runs[0]
        noise = torch.cat([run["noise"][:, 0] for run in runs])
        # The noise is the response alone, independent of the residual: the
        # two cases differ only in their target and get the same value.
        torch.testing.assert_close(noise[0], noise[1], rtol=1e-12, atol=0)
        expected = theta ** 2 * sigma ** 2
        self.assertLess(abs(float(noise[0]) - expected), 4 * float(result["noise_sem"][0, 0]))
        self.assertAlmostEqual(float(result["signal"][0, 0]), 0.25 * float(self.field.square().mean()))
        self.assertTrue(torch.all(result["selection_changes"] == 0))

    def test_noise_matches_the_jacobian_of_a_grid_model(self):
        torch.manual_seed(11)

        def branch(polarized):
            a = CartesianAFeatures(
                max_power=1, mean_density=.5, cutoff_grid=1,
                include_polarization=polarized, separate_center=True,
                radial_basis="gaussian", radial_exponents=(.2,),
                convolution_backend="fft",
            )
            b = CartesianBFeatures(1, 2, include_polarization=polarized, separate_center=True)
            return CACEBranch(a, b, LocalReadout(hidden_sizes=(5,)))

        model = GridCACEModel(
            readout=[branch(False), branch(True), LDAReadout()],
            ideal=FixedDipoleIdeal(1.), grid_spacing=.7, mean_density=.5,
            polarization_scale=.3, mean_temperature=1.5, boltzmann_constant=1.,
            rho_min=1.0e-2, compute_local_mu=True,
            compute_polarization_derivative=True,
        )
        size = 4
        grid = torch.stack(torch.meshgrid(
            *(torch.arange(size),) * 3, indexing="ij"), -1).reshape(-1, 3)
        rho = .4 + .2 * torch.rand(size ** 3, 1)
        rho[:8] = 1.0e-3          # below rho_min: outside the voxel selection
        polar = .05 * torch.randn(size ** 3, 1, 3)
        polar[:8] = 0.0
        case = {
            "rho": rho, "dipole_density": polar,
            "rho_std": torch.where(rho > .01, .01, 0.0),
            "dipole_density_std": torch.where(rho[..., None] > .01, .004, 0.0).expand_as(polar).clone(),
            "V_ext": .1 * torch.randn_like(rho),
            "E_ext": .1 * torch.randn_like(polar),
            "temperature": torch.tensor(1.5), "grid_size": torch.tensor([size] * 3),
            "grid_positions": grid,
        }
        case["beta_E_target"] = case["E_ext"] / 1.5
        loss = Loss([
            TensorLoss("density", "beta_mu_local", "beta_mu_average", weights_key="voxel_weights"),
            TensorLoss("polarization", "polarization_derivative", "beta_E_target", weights_key="voxel_weights"),
        ])
        batch = {key: value[None] for key, value in case.items()}
        result = case_loss_statistics(
            model, loss, [batch], terms=["density", "polarization"],
            noise_samples=1, noise_step=0.1, seed=7,
        )
        # The same draw, propagated through the model by a central difference.
        generator = torch.Generator().manual_seed(7)
        draws = _perturbation(batch, {"rho": "rho_std", "dipole_density": "dipole_density_std"}, generator)
        model.eval()

        def residuals(step):
            values = {key: value + step * draws[key] if key in draws else value
                      for key, value in batch.items()}
            outputs = model(values, use_voxel_mask=True)
            return (
                outputs["beta_mu_local"] - outputs["beta_mu_average"][..., None, :],
                outputs["polarization_derivative"] - values["beta_E_target"],
                outputs["voxel_weights"],
            )

        step = 1.0e-5
        plus, minus = residuals(step), residuals(-step)
        weights = residuals(0.0)[2]
        expected = []
        for index in range(2):
            derivative = (plus[index] - minus[index]) / (2 * step)
            w = weights if index == 0 else weights[..., None].expand_as(derivative)
            expected.append(float((w * derivative.square()).sum().detach() / w.sum()))  # |J xi|^2
        # Central differences: O(h^2) from the exact linear response.
        torch.testing.assert_close(
            result["noise"][0], torch.tensor(expected), rtol=2e-4, atol=0,
        )
        finer = case_loss_statistics(
            model, loss, [batch], terms=["density", "polarization"],
            noise_samples=1, noise_step=0.01, seed=7,
        )
        torch.testing.assert_close(
            finer["noise"][0], torch.tensor(expected), rtol=2e-6, atol=0,
        )
        # For the ideal gas alone the density noise is (sigma / rho)^2 averaged
        # over the selection, less the gauge mean's share.
        ideal_model = GridCACEModel(
            readout=[], n_types=1, ideal=FixedDipoleIdeal(1.), grid_spacing=.7,
            mean_density=.5, mean_temperature=1.5, boltzmann_constant=1.,
            rho_min=1.0e-2, compute_local_mu=True,
            compute_polarization_derivative=True,
        )
        unpolarized = dict(batch, dipole_density=torch.zeros_like(batch["dipole_density"]))
        ideal_noise = case_loss_statistics(
            ideal_model, loss, [unpolarized], terms=["density"],
            noise_samples=200, seed=1,
        )
        selected = weights[0, :, 0] > 0
        expected_ideal = float(((.01 / rho[selected, 0]) ** 2).mean()) * (1 - 1 / int(selected.sum()))
        self.assertLess(
            abs(float(ideal_noise["noise"][0, 0]) - expected_ideal),
            3 * float(ideal_noise["noise_sem"][0, 0]),
        )
        self.assertGreater(float(result["signal"][0, 0]), 0)

    def test_weights_rebalance_a_weakly_driven_case(self):
        # Case A is strongly driven, case B weakly; one shared slope cannot fit
        # both. The plain loss fits A and leaves B 33 % off.
        cases = [
            slope_case(10.0, 2.0, 1e-3, self.field),
            slope_case(0.3, 1.5, 1e-3, self.field),
        ]
        reference = SlopeModel(1.0)
        statistics = case_loss_statistics(
            reference, mse_loss(), self.loader(cases), terms=["field"],
            noise_samples=8,
        )

        def relative_errors(theta):
            return abs(theta - 2.0) / 2.0, abs(theta - 1.5) / 1.5

        plain = fitted_slope(SlopeModel(0.0), mse_loss(), cases)
        plain_errors = relative_errors(plain)
        self.assertGreater(plain_errors[1], 0.3)
        previous = plain_errors[1]
        for power in (0.5, 1.0):
            weights = case_loss_weights(
                statistics, relative_model_error=0.1, power=power,
            )
            self.assertAlmostEqual(float(weights["weights"].mean()), 1.0)
            weighted_cases = [
                dict(case, field_case_weight=weights["weights"][index, 0])
                for index, case in enumerate(cases)
            ]
            loss = mse_loss(case_weight_keys={"field": "field_case_weight"})
            theta = fitted_slope(SlopeModel(0.0), loss, weighted_cases)
            errors = relative_errors(theta)
            self.assertLess(errors[1], previous)
            previous = errors[1]
        # Full weighting: each case is fitted relative to its own scale.
        self.assertLess(previous, plain_errors[1] / 3)

    def test_a_case_without_signal_is_weighted_by_its_noise(self):
        cases = [
            slope_case(1.0, 2.0, 1e-3, self.field),
            slope_case(1.0, 1.0, 0.05, self.field),   # exactly fitted at the reference
        ]
        statistics = case_loss_statistics(
            SlopeModel(1.0), mse_loss(), self.loader(cases), terms=["field"],
            noise_samples=16,
        )
        self.assertEqual(float(statistics["signal"][1, 0]), 0.0)
        weights = case_loss_weights(statistics, relative_model_error=0.1, normalization=[1.0])
        scales = weights["scales"][:, 0]
        torch.testing.assert_close(scales[1], statistics["noise"][1, 0])
        torch.testing.assert_close(weights["weights"][:, 0], 1 / scales)
        with self.assertRaisesRegex(ValueError, "zero scale"):
            case_loss_weights(
                dict(statistics, noise=torch.zeros_like(statistics["noise"])),
                relative_model_error=0.1,
            )

    def test_normalizations_and_validation(self):
        statistics = {
            "terms": ("a", "b"),
            "signal": torch.tensor([[1.0, 4.0], [3.0, 2.0]]),
            "noise": torch.tensor([[0.1, 0.0], [0.1, 0.0]]),
        }
        mean = case_loss_weights(statistics, relative_model_error={"a": 0.1, "b": 1.0})
        torch.testing.assert_close(mean["weights"].mean(0), torch.ones(2))
        kept = case_loss_weights(statistics, relative_model_error=0.5, normalization="reference_loss")
        torch.testing.assert_close(
            (kept["weights"] * statistics["signal"]).sum(0), statistics["signal"].sum(0),
        )
        # A second split reuses the first split's constant.
        again = case_loss_weights(
            statistics, relative_model_error={"a": 0.1, "b": 1.0},
            normalization=mean["normalization"],
        )
        torch.testing.assert_close(again["weights"], mean["weights"])
        for bad in ({"power": 0.0}, {"power": 1.5}, {"noise_scale": -1.0},
                    {"normalization": "median"}):
            with self.assertRaises(ValueError):
                case_loss_weights(statistics, relative_model_error=0.1, **bad)
        with self.assertRaises(KeyError):
            case_loss_weights(statistics, relative_model_error={"a": 0.1})

    def test_signal_quantile_noise_scale_is_an_explicit_heuristic(self):
        # The requested quantile is computed mechanically; S/N is not a
        # statistically justified bound on the correlated-noise scale.
        statistics = {
            "terms": ("p",),
            "signal": torch.tensor([[0.05], [0.06], [3.0]]),
            "noise": torch.tensor([[0.2], [0.24], [0.2]]),
        }
        heuristic = case_loss_weights(
            statistics, relative_model_error=0.1, noise_scale="signal_quantile",
            signal_quantile=0.0,
        )
        torch.testing.assert_close(heuristic["noise_scale"], torch.tensor([0.25]))
        torch.testing.assert_close(
            heuristic["scales"][:, 0], 0.25 * statistics["noise"][:, 0] + 0.1 * statistics["signal"][:, 0],
        )
        reused = case_loss_weights(
            statistics, relative_model_error=0.1, noise_scale=heuristic["noise_scale"],
            normalization=heuristic["normalization"],
        )
        torch.testing.assert_close(reused["weights"], heuristic["weights"])
        with self.assertRaises(ValueError):
            case_loss_weights(statistics, relative_model_error=0.1, noise_scale="auto")
        with self.assertRaises(ValueError):
            case_loss_weights(statistics, relative_model_error=0.1, noise_scale="calibrate")

    def test_custom_terms_declare_their_batch_targets(self):
        class Residual(LossTerm):
            def evaluate(self, outputs, batch, model=None):
                value = (outputs["prediction"] - batch["goal"]).square().mean()
                return {"loss": self.weight * value, "unweighted": value}

        case = {"rho": self.field, "rho_std": torch.full_like(self.field, .1),
                "goal": 3.0 * self.field}
        loss = Loss([Residual("custom", weight=2.0)])
        with self.assertRaisesRegex(ValueError, "target_keys"):
            case_loss_statistics(SlopeModel(1.0), loss, [case], terms=["custom"])
        result = case_loss_statistics(
            SlopeModel(1.0), loss, [case], terms=["custom"], target_keys=["goal"],
            noise_samples=200,
        )
        self.assertLess(abs(float(result["noise"][0, 0]) - 0.01), 4 * float(result["noise_sem"][0, 0]))
        self.assertAlmostEqual(float(result["signal"][0, 0]), 4 * float(self.field.square().mean()))

    def test_trainer_applies_case_weights_from_the_loader(self):
        cases = [slope_case(1.0, 2.0, 1e-3, self.field), slope_case(1.0, 3.0, 1e-3, self.field)]
        weights = (0.5, 1.5)
        weighted = [dict(case, w=torch.tensor(w)) for case, w in zip(cases, weights)]
        loader = DataLoader(weighted, batch_size=1, shuffle=False)
        loss = mse_loss(case_weight_keys={"field": "w"})
        trainer = Trainer(SlopeModel(1.0), loss, device="cpu")
        losses, _ = trainer.evaluate_streams(
            (TrainingStream("field", loader, loader, loss),),
        )["field"]
        expected = sum(
            w * float((case["rho"] - case["target"]).square().mean())
            for case, w in zip(cases, weights)
        ) / 2
        self.assertAlmostEqual(losses["total"], expected)

    def test_draws_near_the_domain_boundary_are_kept(self):
        # rho = 0.01 with SEM 0.2: almost every draw at the nominal step leaves
        # rho > 0. The noise of prediction = rho is sigma^2 = 0.04.
        case = {"rho": torch.full((1, 1), 0.01), "rho_std": torch.full((1, 1), 0.2),
                "target": torch.zeros(1, 1)}
        result = case_loss_statistics(
            SlopeModel(1.0), mse_loss(), [case], terms=["field"],
            noise_samples=2000, noise_step=1.0, seed=4,
        )
        self.assertLess(abs(float(result["noise"][0, 0]) - 0.04), 4 * float(result["noise_sem"][0, 0]))
        self.assertGreater(int(result["step_reductions"][0]), 1000)
        self.assertLess(float(result["minimum_step"][0]), 0.1)

    def test_selection_activations_are_counted_and_excluded_from_the_response(self):
        theta, sigma = 2.0, 0.1
        rho = torch.cat([torch.full((32, 1), 0.47), torch.full((32, 1), 1.0)])
        case = {"rho": rho, "rho_std": torch.full_like(rho, sigma), "target": torch.zeros_like(rho)}
        loss = Loss([TensorLoss("field", "prediction", "target", weights_key="voxel_weights")])
        result = case_loss_statistics(
            ThresholdModel(theta), loss, [case], terms=["field"],
            noise_samples=200, noise_step=1.0, seed=2,
        )
        # Voxels at 0.47 cross 0.5 upwards at the nominal step: activations.
        self.assertGreater(int(result["selection_activations"][0]), 0)
        self.assertGreaterEqual(int(result["selection_changes"][0]), int(result["selection_activations"][0]))
        self.assertGreater(int(result["step_reductions"][0]), 0)
        # The response is the derivative on the clean selection.
        expected = theta ** 2 * sigma ** 2
        self.assertLess(abs(float(result["noise"][0, 0]) - expected), 4 * float(result["noise_sem"][0, 0]))
        quiet = case_loss_statistics(
            ThresholdModel(theta), loss, [case], terms=["field"],
            noise_samples=4, noise_step=1e-3, seed=2,
        )
        self.assertEqual(int(quiet["selection_changes"][0]), 0)
        self.assertEqual(int(quiet["step_reductions"][0]), 0)

    def test_weights_follow_case_identities_not_positions(self):
        cases = [
            dict(slope_case(amplitude, 1.0, 1e-3, self.field), case_id="case-{}".format(index))
            for index, amplitude in enumerate((0.3, 1.0, 3.0, 10.0))
        ]
        shuffled = DataLoader(cases, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(1))
        with self.assertRaisesRegex(ValueError, "case_key"):
            case_loss_statistics(SlopeModel(0.0), mse_loss(), shuffled, terms=["field"])
        ordered = case_loss_weights(
            case_loss_statistics(SlopeModel(0.0), mse_loss(), self.loader(cases),
                                 terms=["field"], case_key="case_id"),
            relative_model_error=0.1,
        )
        mixed = case_loss_weights(
            case_loss_statistics(SlopeModel(0.0), mse_loss(), shuffled,
                                 terms=["field"], case_key="case_id"),
            relative_model_error=0.1,
        )
        self.assertEqual(ordered["cases"], ["case-0", "case-1", "case-2", "case-3"])
        self.assertNotEqual(mixed["cases"], ordered["cases"])
        for case in ordered["cases"]:
            self.assertAlmostEqual(mixed["by_case"][case]["field"], ordered["by_case"][case]["field"])
        # The weakest case gets the largest weight.
        self.assertEqual(max(ordered["by_case"], key=lambda c: ordered["by_case"][c]["field"]), "case-0")

    def test_loss_rejects_invalid_case_weights(self):
        case = slope_case(1.0, 2.0, 1e-3, self.field)
        model = SlopeModel(0.5)
        loss = mse_loss(case_weight_keys={"field": "w"})
        for bad in (-1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                loss(model(case), dict(case, w=torch.tensor([bad])))
        self.assertEqual(float(loss(model(case), dict(case, w=torch.tensor([0.0])))["total"]), 0.0)

    def test_tensor_loss_moves_under_a_pytorch_without_recurse(self):
        original = nn.Module._apply

        def legacy_apply(self, fn):        # Module._apply before ``recurse``
            return original(self, fn)

        term = TensorLoss("field", "prediction", "target", residual_scale=2.0)
        with unittest.mock.patch.object(nn.Module, "_apply", legacy_apply):
            moved = term.to(torch.float32)
        self.assertEqual(moved.residual_scale.dtype, torch.float32)

    def test_loss_without_case_weights_is_unchanged_and_checks_its_keys(self):
        case = slope_case(1.0, 2.0, 1e-3, self.field)
        model = SlopeModel(0.5)
        plain = mse_loss()(model(case), case)["total"]
        weighted_loss = mse_loss(case_weight_keys={"field": "w"})
        values, details = weighted_loss.evaluate(model(case), dict(case, w=torch.tensor([2.5])))
        torch.testing.assert_close(values["total"], 2.5 * plain)
        torch.testing.assert_close(details["field"]["case_weight"], torch.tensor(2.5))
        with self.assertRaisesRegex(ValueError, "one case per batch"):
            weighted_loss(model(case), dict(case, w=torch.tensor([1.0, 2.0])))
        two_cases = {
            key: torch.stack([tensor, tensor]) for key, tensor in case.items()
        }
        with self.assertRaisesRegex(ValueError, "rho holds 2 cases"):
            weighted_loss(model(two_cases), dict(two_cases, w=torch.tensor([2.5])))
        with self.assertRaisesRegex(KeyError, "case weight"):
            weighted_loss(model(case), case)
        with self.assertRaisesRegex(ValueError, "unknown terms"):
            mse_loss(case_weight_keys={"other": "w"})
        with self.assertRaisesRegex(ValueError, "one case per batch"):
            case_loss_statistics(
                model, mse_loss(), [{key: torch.stack([v, v]) for key, v in case.items()}],
                terms=["field"],
            )


class TestCaseWeightedExample(unittest.TestCase):
    def test_example_trains_and_resumes_with_recorded_weights(self):
        import json, subprocess, sys, tempfile
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        script = root / "examples" / "lj_nvt" / "train_case_weighted.py"
        env = dict(__import__("os").environ, PYTHONPATH=str(root / "src"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fit"
            command = [sys.executable, str(script), "--device", "cpu", "--noise-samples", "2",
                       "--output", str(output)]
            subprocess.run(command + ["--epochs", "1"], check=True, env=env,
                           capture_output=True, cwd=root)
            record = (output / "case_weights.json").read_text()
            weights = json.loads(record)
            self.assertEqual(len(weights["train"]["by_case"]), 3)
            self.assertEqual(len(weights["valid"]["by_case"]), 2)
            mean = sum(row["weight"] for row in weights["train"]["by_case"].values()) / 3
            self.assertAlmostEqual(mean, 1.0)
            resumed = subprocess.run(command + ["--epochs", "2"], check=True, env=env,
                                     capture_output=True, text=True, cwd=root)
            self.assertIn("Resumed from epoch 1", resumed.stdout)
            self.assertEqual((output / "case_weights.json").read_text(), record)


if __name__ == "__main__":
    unittest.main()
