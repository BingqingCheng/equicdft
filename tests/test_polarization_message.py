"""Joint field-gated messages: packing, physical derivatives and compatibility."""

import copy
import io
import unittest

import numpy as np
import torch

from equicdft import BChiMessage, CartesianAFeatures, CartesianBFeatures, GridCACEModel, LocalReadout
from equicdft.stencil import get_neighbor_indices


class TestPolarizationMessage(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.get_default_dtype()
        self.rng = torch.random.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(71)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)
        torch.random.set_rng_state(self.rng)

    @staticmethod
    def data(size=5, cutoff=1, n_types=1, batched=False):
        positions = np.indices((size,)*3).reshape(3, -1).T
        indices, _ = get_neighbor_indices(positions, cutoff_grid=cutoff)
        shape = (2, size**3, n_types) if batched else (size**3, n_types)
        data = dict(
            rho=.4 + .3*torch.rand(shape), dipole_density=.2*torch.randn(*shape, 3),
            grid_positions=torch.tensor(positions), grid_size=torch.tensor([size]*3),
            local_density_index=torch.tensor(indices),
            temperature=torch.full((2,) if batched else (), 1.2),
        )
        if batched:
            for key in ("grid_positions", "grid_size", "local_density_index"):
                data[key] = data[key].unsqueeze(0).expand(2, *data[key].shape).clone()
        return data

    @staticmethod
    def model(backend="gather", layers=1, center=True, radial="shared"):
        a = CartesianAFeatures(
            1, .7, cutoff_grid=2, include_polarization=True, dipole_density_scale=.3,
            radial_basis="gaussian", radial_exponents=(.2, .6),
            trainable_radial_exponents=True, coordinate_scaling="cutoff",
            separate_center=center, convolution_backend="fft" if backend=="fft" else "gather",
        )
        b = CartesianBFeatures(1, 2, include_polarization=True, separate_center=center)
        radial_args = {}
        if radial == "gaussian":
            radial_args = dict(radial_exponents=(.3, .8), trainable_radial_exponents=True)
        elif radial == "bessel":
            radial_args = dict(radial_basis="bessel", n_radial_functions=3)
        messages = [BChiMessage(
            b.n_features, a.n_radial_channels, a.n_output_channels,
            hidden_sizes=(5,), include_polarization=True,
            convolution_backend=backend, **radial_args,
        ) for _ in range(layers)]
        return GridCACEModel(
            a, b, [LocalReadout(hidden_sizes=(5,))], grid_spacing=.7,
            message_layers=messages, compute_polarization_derivative=True,
        )

    def test_direct_contraction_with_mixing_batches_and_center(self):
        weights = torch.tensor([[.8, .1, -.2], [.2, .5, 1.1]])
        for center in (False, True):
            for batched in (False, True):
                with self.subTest(center=center, batched=batched):
                    data = self.data(size=3, n_types=3, batched=batched)
                    original = {k:v.clone() for k,v in data.items()}
                    a = CartesianAFeatures(
                        1, .7, cutoff_grid=1, include_polarization=True,
                        dipole_density_scale=.3, separate_center=center,
                        n_types=3, density_transform=weights,
                        radial_basis="gaussian", radial_exponents=(.2, .6),
                    )
                    b = CartesianBFeatures(1, 2, include_polarization=True, separate_center=center)
                    B = b(a(data))
                    message = BChiMessage(b.n_features, 2, 2, hidden_sizes=(5,), include_polarization=True)
                    # Independent species mixing, scaling and field-major packing.
                    rho = torch.einsum("...gt,ct->...gc", data["rho"], weights) / .7
                    polar = torch.einsum("...gtv,ct->...gvc", data["dipole_density"], weights) / .3
                    fields = torch.cat((rho, polar.flatten(-2)), dim=-1)
                    raw, scales = a._polarization_fields(data)
                    torch.testing.assert_close(raw/scales, fields, atol=1e-14, rtol=1e-14)
                    flat = B.flatten(-3)
                    gates = message.mlp(flat) - message.mlp(torch.zeros_like(flat))
                    gates = gates.reshape(*B.shape[:-3], 2, 2, 2)
                    blocks = []
                    basis = a.stencil_basis()
                    for field in range(4):
                        gate = gates[..., 0 if field==0 else 1, :]
                        weighted = gate * fields[..., 2*field:2*field+2].unsqueeze(-2)
                        if batched:
                            neighbors = torch.stack([w[index] for w, index in
                                                     zip(weighted, data["local_density_index"])])
                        else:
                            neighbors = weighted[data["local_density_index"]]
                        block = torch.einsum("...gjnc,jnk->...gnkc", neighbors, basis)
                        if center:
                            block = torch.cat((block, weighted.unsqueeze(-2)), dim=-2)
                        blocks.append(block)
                    expected = torch.cat(blocks, dim=-2)
                    for backend in ("gather", "conv3d", "fft"):
                        message.convolution_backend = backend
                        actual = message(
                            B, data["local_density_index"] if backend=="gather" else None, basis,
                            normalized_fields=raw/scales, separate_center=center,
                            grid_positions=data["grid_positions"], grid_size=data["grid_size"],
                            stencil_positions=a.local_density_positions,
                        )
                        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
                    for key in data:
                        torch.testing.assert_close(data[key], original[key], atol=0, rtol=0)

    def test_model_backends_radial_bases_and_two_layers(self):
        data = self.data(cutoff=2)
        for center in (False, True):
            for radial in ("shared", "gaussian", "bessel"):
                with self.subTest(center=center, radial=radial):
                    reference = self.model(center=center, layers=2, radial=radial)
                    expected = reference(data)
                    expected_features = reference._local_invariant_features(data)
                    self.assertEqual(expected_features.shape[-1], 3*2*reference.b_features.n_features)
                    loss = expected["c1"].square().mean() + expected["polarization_derivative"].square().mean()
                    loss.backward()
                    for backend in ("conv3d", "fft"):
                        actual_model = self.model(backend, center=center, layers=2, radial=radial)
                        actual_model.load_state_dict(reference.state_dict(), strict=True)
                        inputs = dict(data)
                        if backend=="fft":
                            inputs.pop("local_density_index")
                            self.assertFalse(actual_model.requires_local_density_index)
                        actual = actual_model(inputs)
                        for key in expected:
                            torch.testing.assert_close(actual[key], expected[key], atol=2e-11, rtol=2e-11)
                        (actual["c1"].square().mean() + actual["polarization_derivative"].square().mean()).backward()
                        for p, q in zip(reference.parameters(), actual_model.parameters()):
                            self.assertIsNotNone(p.grad)
                            self.assertTrue(torch.isfinite(p.grad).all())
                            torch.testing.assert_close(q.grad, p.grad, atol=2e-11, rtol=2e-11)

    def test_energy_derivatives_and_mixed_hessian(self):
        model = self.model("fft")
        data = self.data(cutoff=2)
        out = model(data)
        drho = torch.randn_like(data["rho"])
        dp = torch.randn_like(data["dipole_density"])
        volume = .7**3
        analytic = volume*((-out["c1"]*drho).sum() + (out["polarization_derivative"]*dp).sum())
        eps = 1e-5
        energies = []
        for sign in (-1, 1):
            shifted = dict(data, rho=data["rho"].detach()+sign*eps*drho,
                           dipole_density=data["dipole_density"].detach()+sign*eps*dp)
            energies.append(model(shifted, compute_c1=False, compute_polarization_derivative=False)["beta_F_exc"])
        torch.testing.assert_close((energies[1]-energies[0])/(2*eps), analytic, atol=2e-8, rtol=2e-7)
        rho_then_p = torch.autograd.grad((-out["c1"]*drho).sum(), data["dipole_density"], retain_graph=True)[0]
        p_then_rho = torch.autograd.grad((out["polarization_derivative"]*dp).sum(), data["rho"])[0]
        torch.testing.assert_close((rho_then_p*dp).sum(), (p_then_rho*drho).sum(), atol=1e-11, rtol=1e-10)

    def test_message_has_two_hop_polarization_dependence(self):
        data = self.data(size=7)
        data["dipole_density"].requires_grad_(True)
        a = CartesianAFeatures(1, .7, cutoff_grid=1, include_polarization=True, dipole_density_scale=.3)
        b = CartesianBFeatures(1, 2, include_polarization=True)
        model = GridCACEModel(a, b, [LocalReadout()], grid_spacing=1., message_layers=[
            BChiMessage(b.n_features, 1, 1, include_polarization=True),
        ])
        # Only the local descriptor is scored: LR cannot create this dependency.
        B1 = model._local_invariant_features(data)[0, b.n_features:]
        gradient = torch.autograd.grad(B1.sum(), data["dipole_density"])[0]
        two_hops = np.ravel_multi_index((2, 0, 0), (7,)*3)
        outside = np.ravel_multi_index((3, 0, 0), (7,)*3)
        self.assertGreater(gradient[two_hops].abs().max().item(), 1e-10)
        torch.testing.assert_close(gradient[outside], torch.zeros_like(gradient[outside]), atol=0, rtol=0)

    def test_empty_fields_zero_gates_and_checkpoint_roundtrip(self):
        data = self.data(cutoff=2)
        data["rho"][:5] = 0
        data["dipole_density"].zero_()
        model = self.model(layers=2)
        outputs = model(data)
        (outputs["c1"].square().mean() + outputs["polarization_derivative"].square().mean()).backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        for key, expected in outputs.items():
            torch.testing.assert_close(restored(data)[key], expected, atol=0, rtol=0)
        zero_message = copy.deepcopy(model)
        for message in zero_message.message_layers:
            for p in message.mlp.parameters():
                p.data.zero_()
        features = zero_message._local_invariant_features(data)
        initial_width = 2*model.b_features.n_features
        torch.testing.assert_close(features[..., initial_width:], torch.zeros_like(features[..., initial_width:]), atol=0, rtol=0)

    def test_configuration_and_carrier_validation(self):
        a = CartesianAFeatures(1, .7, cutoff_grid=1)
        b = CartesianBFeatures(1, 2)
        joint = BChiMessage(b.n_features, 1, 1, include_polarization=True)
        with self.assertRaisesRegex(ValueError, "include_polarization must match"):
            GridCACEModel(a, b, [LocalReadout()], grid_spacing=1., message_layers=[joint])
        data = self.data(size=3)
        B = b(a(data))
        for fields in (None, torch.ones(27, 3), torch.ones(27, 4, dtype=torch.float32)):
            with self.assertRaisesRegex(ValueError, "normalized_fields"):
                joint(B, data["local_density_index"], a.stencil_basis(), normalized_fields=fields)
        with self.assertRaisesRegex(ValueError, "require include_polarization"):
            BChiMessage(b.n_features, 1, 1)(B, data["local_density_index"], a.stencil_basis(), normalized_fields=torch.ones(27,4))


if __name__ == "__main__":
    unittest.main()
