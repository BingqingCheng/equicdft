"""Independent additive CACE pipelines under one physical functional."""

import copy
import re
import tempfile
import unittest
from pathlib import Path

import torch

from equicdft import (
    BChiMessage, CACEBranch, CartesianAFeatures, CartesianBFeatures,
    FixedDipoleIdeal, GridCACEModel, LDAReadout, LocalReadout,
    LongRangeReadout, ReciprocalFeatures, load_model, save_model,
)
from tests.representation import test_polarization_symmetry as symmetry


class TestCACEBranches(unittest.TestCase):
    def setUp(self):
        self.dtype, self.rng = torch.get_default_dtype(), torch.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(981)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)
        torch.set_rng_state(self.rng)

    @staticmethod
    def branch(polarized, backend="fft", messages=0, n_types=1):
        a = CartesianAFeatures(
            max_power=1, mean_density=.5, n_types=n_types, cutoff_grid=1,
            include_polarization=polarized, separate_center=True,
            radial_basis="gaussian", radial_exponents=(.2,),
            trainable_radial_exponents=True, convolution_backend=backend,
        )
        b = CartesianBFeatures(1, 2, include_polarization=polarized, separate_center=True)
        return CACEBranch(a, b, LocalReadout(n_types=n_types, hidden_sizes=(5,)),
                          message_layers=[BChiMessage(
                              b.n_features, a.n_radial_channels, a.n_output_channels,
                              hidden_sizes=(4,), include_polarization=polarized,
                              convolution_backend=backend,
                          ) for _ in range(messages)])

    @staticmethod
    def model(readouts, **kwargs):
        return GridCACEModel(
            readout=readouts, ideal=FixedDipoleIdeal(1.), grid_spacing=.7,
            mean_density=.5, polarization_scale=.3, mean_temperature=1.5,
            boltzmann_constant=1., compute_polarization_derivative=True, **kwargs,
        )

    @staticmethod
    def data():
        data = symmetry.TestFullPolarizationSymmetry.data()
        data["dipole_density"] *= .2
        return data

    @staticmethod
    def shorthand(branch, mode="beta"):
        branch = copy.deepcopy(branch)
        return GridCACEModel(
            branch.a_features, branch.b_features, [branch.readout],
            message_layers=list(branch.message_layers),
            ideal=FixedDipoleIdeal(1.), grid_spacing=.7, mean_temperature=1.5,
            boltzmann_constant=1., polarization_scale=.3,
            compute_polarization_derivative=True, free_energy_mode=mode,
        )

    @staticmethod
    def responses(model, data):
        inputs = copy.deepcopy(data)
        outputs = model(inputs, compute_ideal=False)
        result = {key: value.detach() for key, value in outputs.items()}
        for key in ("density_derivative_excess", "polarization_derivative_excess"):
            derivative = outputs[key]
            probe = torch.linspace(.2, 1., derivative.numel()).reshape_as(derivative)
            hessian = torch.autograd.grad(
                (derivative * probe).sum(),
                (inputs["rho"], inputs["dipole_density"]),
                allow_unused=True, retain_graph=True,
            )
            for field, value in zip(("rho", "dipole_density"), hessian):
                result[key + "/" + field] = (
                    torch.zeros_like(inputs[field]) if value is None else value.detach()
                )
        return result

    def assert_responses(self, actual, expected, atol=2e-12):
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            with self.subTest(output=key):
                torch.testing.assert_close(actual[key], expected[key], atol=atol, rtol=1e-10)

    def test_sum_and_all_hessian_blocks_match_independent_single_branches(self):
        for messages in (0, 1):
            for mode in ("beta", "physical"):
                with self.subTest(messages=messages, mode=mode):
                    branches = [self.branch(False, messages=messages), self.branch(True, messages=messages)]
                    data = self.data()
                    actual = self.responses(self.model(branches, free_energy_mode=mode), data)
                    parts = [self.responses(self.shorthand(branch, mode), data) for branch in branches]
                    self.assert_responses(actual, {key: parts[0][key] + parts[1][key] for key in actual})
                    self.assertEqual(torch.count_nonzero(parts[0]["polarization_derivative_excess"]), 0)
                    self.assertEqual(torch.count_nonzero(parts[0]["density_derivative_excess/dipole_density"]), 0)

    def test_zeroing_either_branch_recovers_other_and_rho_branch_ignores_P(self):
        model, data = self.model([self.branch(False), self.branch(True)]), self.data()
        for index in (0, 1):
            candidate = copy.deepcopy(model)
            with torch.no_grad():
                for parameter in candidate.readout[index].readout.parameters():
                    parameter.zero_()
            self.assert_responses(self.responses(candidate, data),
                                  self.responses(self.shorthand(model.readout[1-index]), data))
        rho_model = self.model([copy.deepcopy(model.readout[0])])
        expected = self.responses(rho_model, data)
        data["dipole_density"] *= 10
        self.assert_responses(self.responses(rho_model, data), expected, atol=0.)

    def test_both_branches_receive_gradients_from_field_losses(self):
        model = self.model([self.branch(False, messages=1), self.branch(True, messages=1)])
        outputs = model(self.data(), compute_ideal=False)
        loss = (outputs["c1"].square().mean()
                + outputs["polarization_derivative_excess"].square().mean())
        loss.backward()
        for branch in model.readout:
            for module in (branch.a_features, branch.message_layers, branch.readout):
                gradients = [parameter.grad for parameter in module.parameters()]
                self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))
                self.assertGreater(sum(value.abs().sum().item() for value in gradients), 0.)

    def test_global_terms_and_ideal_are_counted_once(self):
        branches = [self.branch(False), self.branch(True)]
        globals_ = [
            LDAReadout(include_polarization=True, hidden_sizes=(4,), zero_init=False),
            LongRangeReadout(charges=(.4,), include_polarization=True, coulomb_amplitude=1.8,
                             features=ReciprocalFeatures(radial_exponents=(.15,), kernel="coulomb")),
        ]
        data = self.data()
        full = self.model(copy.deepcopy(branches + globals_))(copy.deepcopy(data))
        local = self.model(branches)(copy.deepcopy(data), compute_ideal=False)
        global_output = self.model(globals_)(copy.deepcopy(data))
        for key in ("beta_F_exc", "density_derivative_excess", "polarization_derivative_excess"):
            torch.testing.assert_close(full[key], local[key] + global_output[key])
        torch.testing.assert_close(full["beta_F_id"], global_output["beta_F_id"], atol=0, rtol=0)
        torch.testing.assert_close(full["beta_F"], local["beta_F_exc"] + global_output["beta_F"])
        for key in ("density_derivative", "polarization_derivative"):
            torch.testing.assert_close(full[key], local[key + "_excess"] + global_output[key])

    def test_serialization_config_and_existing_shorthand_can_coexist(self):
        rho_branch, joint_branch = self.branch(False), self.branch(True, messages=1)
        model = self.shorthand(rho_branch)
        model.readout.append(joint_branch)
        state = model.state_dict()
        # Build through the constructor to check both paths and their metadata.
        model = GridCACEModel.from_config(model.to_config())
        model.load_state_dict(state)
        data = self.data()
        reference = self.responses(model, data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            restored = load_model(path).train()
        self.assert_responses(self.responses(restored, data), reference, atol=0.)
        self.assertEqual(restored.to_config(), model.to_config())
        self.assertEqual([key for key in model.state_dict() if "polarization_scale" in key],
                         ["polarization_scale"])
        # Pure branch syntax also roundtrips without a top-level feature pair.
        branch_only = self.model([copy.deepcopy(rho_branch), copy.deepcopy(joint_branch)])
        restored = GridCACEModel.from_config(branch_only.to_config())
        restored.load_state_dict(branch_only.state_dict())
        self.assert_responses(self.responses(restored, data), self.responses(branch_only, data), atol=0.)

    def test_fft_gather_batch_and_multiple_species(self):
        for messages in (0, 1):
            fft = self.model([self.branch(False, messages=messages, n_types=2),
                              self.branch(True, messages=messages, n_types=2)])
            gather = self.model([self.branch(False, "gather", messages, 2),
                                 self.branch(True, "gather", messages, 2)])
            gather.load_state_dict(fft.state_dict())
            self.assertFalse(fft.requires_local_density_index)
            self.assertTrue(gather.requires_local_density_index)
            self.assertEqual(fft.grid_info["cutoff_grid"], 1)
            self.assertEqual(fft.n_types, 2)
            data = self.data()
            data["rho"] = data["rho"].expand(-1, 2).clone()
            data["dipole_density"] = data["dipole_density"].expand(-1, 2, -1).clone()
            data = {key: torch.stack((value, value)) for key, value in data.items()}
            expected = self.responses(gather, data)
            data.pop("local_density_index")
            self.assert_responses(self.responses(fft, data), expected)

    def test_documented_p2v3_builder_trains_without_warmup(self):
        document = Path(__file__).resolve().parents[2] / "docs/polarization.md"
        source = next(block for block in re.findall(
            r"```python\n(.*?)```", document.read_text(), re.S,
        ) if "def build_additive_model(" in block)
        namespace = {}
        exec(source, namespace)
        model = namespace["build_additive_model"](
            density_reference=.5, polarization_scale=.3, dipole_magnitude=1.,
            grid_spacing=.7, temperature_reference=1.5, boltzmann_constant=1.,
            cutoff_grid=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        output = model(self.data())
        loss = (output["density_derivative"].square().mean()
                + output["polarization_derivative"].square().mean())
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))

    def test_reject_inconsistent_geometry_scales_species_and_double_counting(self):
        for attribute, value, message in (("cutoff_grid", 2, "cutoff_grid"),
                                          ("mean_density", torch.tensor(.7), "mean_density"),
                                          ("n_types", 2, "n_types")):
            branches = [self.branch(False), self.branch(True)]
            setattr(branches[1].a_features, attribute, value)
            with self.assertRaisesRegex(ValueError, message):
                self.model(branches)
        branch = self.branch(True)
        with self.assertRaisesRegex(ValueError, "count.*twice"):
            self.model([branch, branch.readout])
        with self.assertRaisesRegex(ValueError, "same module twice"):
            self.model([branch, branch])
        reused = CACEBranch(copy.deepcopy(branch.a_features),
                            copy.deepcopy(branch.b_features), branch.readout)
        with self.assertRaisesRegex(ValueError, "same inner readout"):
            self.model([branch, reused])
        with self.assertRaisesRegex(ValueError, "polarization_scale"):
            GridCACEModel(readout=[branch], grid_spacing=1.)
        with self.assertRaisesRegex(TypeError, "LocalReadout"):
            CACEBranch(branch.a_features, branch.b_features, LDAReadout())


if __name__ == "__main__":
    unittest.main()
