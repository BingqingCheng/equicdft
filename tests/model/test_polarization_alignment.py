"""Reduced-polarization learned inputs and their physical chain rule."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from equicdft import (
    CACEBranch,
    CartesianAFeatures,
    CartesianBFeatures,
    FixedDipoleIdeal,
    GridCACEModel,
    LDAReadout,
    LocalReadout,
    load_model,
    regularized_reduced_polarization,
    save_model,
)
from equicdft.functional.base import ReadoutContext
from equicdft.geometry.stencil import get_neighbor_indices


class TestPolarizationAlignment(unittest.TestCase):
    def setUp(self):
        self.dtype, self.rng = torch.get_default_dtype(), torch.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(712)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)
        torch.set_rng_state(self.rng)

    @staticmethod
    def data():
        positions = np.indices((3, 3, 3)).reshape(3, -1).T
        neighbors, _ = get_neighbor_indices(positions, cutoff_grid=1)
        rho = .2 + .3 * torch.rand(27, 1)
        return {
            "rho": rho,
            "dipole_density": .08 * rho[..., None] * torch.randn(27, 1, 3),
            "temperature": torch.tensor(1.2),
            "grid_size": torch.tensor([3, 3, 3]),
            "grid_positions": torch.tensor(positions),
            "local_density_index": torch.tensor(neighbors),
        }

    @staticmethod
    def branch(polarization_input, backend="gather"):
        a = CartesianAFeatures(
            1, .5, cutoff_grid=1, include_polarization=True,
            polarization_input=polarization_input,
            radial_basis="gaussian", radial_exponents=(.2,),
            convolution_backend=backend,
        )
        b = CartesianBFeatures(1, 2, include_polarization=True)
        return CACEBranch(a, b, LocalReadout(hidden_sizes=(5,)))

    @classmethod
    def model(cls, lda="raw", branches=("raw",), backend="gather"):
        uses_alignment = lda == "alignment" or "alignment" in branches
        return GridCACEModel(
            readout=[
                *(cls.branch(value, backend) for value in branches),
                LDAReadout(
                    include_polarization=True,
                    polarization_input=lda,
                    hidden_sizes=(5,),
                    zero_init=False,
                ),
            ],
            ideal=FixedDipoleIdeal(.7),
            grid_spacing=.5,
            mean_density=.5,
            polarization_scale=.3,
            alignment_scale=.2 if uses_alignment else None,
            alignment_density_floor=.01 if uses_alignment else None,
            boltzmann_constant=1.,
            compute_c1=True,
            compute_polarization_derivative=True,
        )

    def test_exact_transform_vacuum_and_finite_second_derivatives(self):
        rho = torch.tensor([[0.], [.01], [.4]], requires_grad=True)
        dipole = torch.tensor(
            [[[.2, -.1, .3]], [[.1, .2, -.2]], [[.3, .1, .2]]],
            requires_grad=True,
        )
        actual = regularized_reduced_polarization(rho, dipole, .7, .01)
        expected = rho[..., None] * dipole / (
            .7 * (rho.square() + .01**2)
        )[..., None]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(actual[0], torch.zeros_like(actual[0])))
        gradient = torch.autograd.grad(
            actual.square().sum(), (rho, dipole), create_graph=True,
        )
        hessian = torch.autograd.grad(
            sum(value.square().sum() for value in gradient),
            (rho, dipole),
        )
        self.assertTrue(all(torch.isfinite(value).all() for value in (*gradient, *hessian)))

    def test_lda_uses_separately_standardized_alignment(self):
        data = self.data()
        context = ReadoutContext(
            rho=data["rho"],
            dipole_density=data["dipole_density"],
            normalized_temperature=data["temperature"] / 1.5,
            voxel_volume=.125,
            mean_density=.5,
            polarization_scale=.3,
            alignment_scale=.2,
            alignment_density_floor=.01,
            dipole_magnitude=.7,
        )
        readout = LDAReadout(
            include_polarization=True,
            polarization_input="alignment",
            hidden_sizes=(),
        )
        with torch.no_grad():
            readout.mlp[0].weight.copy_(torch.tensor([[1., 2., 3.]]))
            readout.mlp[0].bias.fill_(.4)
        alignment = regularized_reduced_polarization(
            data["rho"], data["dipole_density"], .7, .01,
        ) / .2
        per_particle = (
            data["rho"][..., 0] / .5
            + 2 * alignment.square().sum(-1)[..., 0]
            + 3 * data["temperature"] / 1.5
            + .4
        )
        expected = .125 * (data["rho"][..., 0] * per_particle).sum()
        torch.testing.assert_close(readout.energy(context), expected)

    def test_alignment_a_and_message_carrier_are_exact(self):
        rho = torch.tensor([[.1, .2], [.3, .4]])
        dipole = torch.tensor([
            [[.01, .02, .03], [.04, .05, .06]],
            [[.07, .08, .09], [.10, .11, .12]],
        ])
        a_features = CartesianAFeatures(
            max_power=1,
            mean_density=.5,
            n_types=2,
            cutoff_grid=1,
            include_polarization=True,
            polarization_input="alignment",
        )
        fields, scales = a_features.polarization_fields(
            {"rho": rho, "dipole_density": dipole},
            alignment_scale=.25,
            dipole_magnitude=(.6, .8),
            alignment_density_floor=.05,
        )
        alignment = regularized_reduced_polarization(
            rho, dipole, (.6, .8), .05,
        )
        expected = torch.cat(
            (rho / .5, (alignment / .25).transpose(-1, -2).flatten(-2)),
            dim=-1,
        )
        # fields/scales are reused verbatim as normalized message carriers.
        torch.testing.assert_close(fields / scales, expected, rtol=0, atol=0)

    def test_raw_default_is_exact_and_alignment_requirements_are_explicit(self):
        implicit = self.model(lda="raw", branches=("raw",))
        explicit = copy.deepcopy(implicit)
        explicit.readout[0].a_features.polarization_input = "raw"
        explicit.readout[1].polarization_input = "raw"
        data = self.data()
        self.assertIsNone(implicit.alignment_scale)
        self.assertIsNone(implicit.alignment_density_floor)
        self.assertFalse(any(key.startswith("alignment_") for key in implicit.state_dict()))
        for key, value in explicit(copy.deepcopy(data)).items():
            torch.testing.assert_close(implicit(copy.deepcopy(data))[key], value, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "alignment_scale"):
            GridCACEModel(
                readout=[self.branch("alignment")],
                ideal=FixedDipoleIdeal(.7), grid_spacing=.5, mean_density=.5,
            )
        with self.assertRaisesRegex(ValueError, "dipole_magnitude"):
            GridCACEModel(
                readout=[self.branch("alignment")], grid_spacing=.5,
                mean_density=.5, alignment_scale=.2,
                alignment_density_floor=.01,
            )
        for value in ("other", 3):
            with self.assertRaises((TypeError, ValueError)):
                CartesianAFeatures(
                    1, .5, include_polarization=True,
                    polarization_input=value,
                )

    def test_custom_raw_a_keeps_the_original_forward_signature(self):
        class RawOnlyA(CartesianAFeatures):
            def forward(self, data, *, polarization_scale=None):
                return super().forward(
                    data,
                    polarization_scale=polarization_scale,
                )

        a_features = RawOnlyA(
            1,
            .5,
            cutoff_grid=1,
            include_polarization=True,
        )
        b_features = CartesianBFeatures(1, 2, include_polarization=True)
        model = GridCACEModel(
            readout=[CACEBranch(
                a_features,
                b_features,
                LocalReadout(hidden_sizes=(5,)),
            )],
            ideal=FixedDipoleIdeal(.7),
            grid_spacing=.5,
            mean_density=.5,
            polarization_scale=.3,
            boltzmann_constant=1.,
            compute_c1=True,
            compute_polarization_derivative=True,
        )
        output = model(self.data())
        self.assertTrue(torch.isfinite(output["beta_F"]))

    def test_fft_gather_additive_chain_rule_and_serialization(self):
        gather = self.model(lda="alignment", branches=("raw", "alignment"))
        fft = self.model(
            lda="alignment", branches=("raw", "alignment"), backend="fft",
        )
        fft.load_state_dict(gather.state_dict())
        data = self.data()
        expected = gather(copy.deepcopy(data), compute_ideal=False)
        fft_data = copy.deepcopy(data)
        fft_data.pop("local_density_index")
        actual = fft(fft_data, compute_ideal=False)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=2e-12, rtol=1e-10)

        live = copy.deepcopy(data)
        output = gather(live, compute_ideal=False)
        left = torch.autograd.grad(
            output["c1"][5, 0], live["dipole_density"], retain_graph=True,
        )[0]
        right = torch.autograd.grad(
            output["polarization_derivative_excess"][5, 0, 1],
            live["rho"], retain_graph=True,
        )[0]
        torch.testing.assert_close(left[5, 0, 1], -right[5, 0], atol=2e-12, rtol=1e-10)
        self.assertGreater(abs(left[5, 0, 1].item()), 1e-9)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alignment.pt"
            save_model(gather, path)
            restored = load_model(path)
        # load_model returns an evaluating model while a freshly constructed
        # one is training; that difference alone changes create_graph in the
        # derivative path and moves c1 by one ulp, so match the mode before
        # comparing bit for bit.
        restored.train(gather.training)
        self.assertEqual(restored.to_config(), gather.to_config())
        for key, value in gather(copy.deepcopy(data)).items():
            torch.testing.assert_close(restored(copy.deepcopy(data))[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
