"""Exact bulk Hessian symbol and the bulk response loss."""

import math
import unittest

import numpy as np
import torch
from torch import nn

from equicdft import (
    BulkResponseLoss, CACEBranch, CartesianAFeatures, CartesianBFeatures,
    FixedDipoleIdeal, GridCACEModel, LDAReadout, LocalReadout, LongRangeReadout,
    ReciprocalFeatures,
)
from equicdft.geometry.stencil import get_neighbor_indices
from equicdft.response.bulk import (
    bulk_response_symbol, curvature_to_spectra, reduce_bulk_symbol, spectra_to_curvature,
)
from equicdft.response.curvature import _energy_only_model

GRID = 6
SPACING = 1.5
RHO = 0.5
MOMENT = 0.7


def positions():
    return torch.tensor(np.indices((GRID,) * 3).reshape(3, -1).T)


def bulk(n_types=1, density=RHO):
    neighbors, _ = get_neighbor_indices(positions().numpy(), cutoff_grid=1)
    return {
        "rho": torch.full((1, GRID ** 3, n_types), density),
        "dipole_density": torch.zeros((1, GRID ** 3, n_types, 3)),
        "temperature": torch.ones(1),
        "grid_size": torch.tensor([[GRID] * 3]),
        "grid_spacing": torch.full((1, 3), SPACING),
        "grid_positions": positions()[None],
        "local_density_index": torch.tensor(neighbors)[None],
    }


def cace_model():
    """CACE with polarization, LDA, exact Coulomb on P and a dielectric kernel."""

    torch.manual_seed(5)
    a = CartesianAFeatures(
        max_power=1, mean_density=RHO, cutoff_grid=1, include_polarization=True,
        separate_center=True, radial_basis="gaussian", radial_exponents=(0.4,),
        trainable_radial_exponents=True, convolution_backend="fft",
    )
    b = CartesianBFeatures(1, 2, include_polarization=True, separate_center=True)
    coulomb = LongRangeReadout(
        n_kernels=1, n_types=1, charges=(0.0,), include_polarization=True, coulomb_amplitude=0.3,
        features=ReciprocalFeatures(radial_exponents=(0.0,), kernel="coulomb", n_types=1),
    )
    kernel = LongRangeReadout(
        n_types=1, hidden_sizes=(), zero_init=True,
        features=ReciprocalFeatures(radial_exponents=(1.0, 3.0), kernel="gaussian", n_types=1,
                                    variable="dipole_density", include_divergence=True),
        coefficient_scale=(1.0,), initial_coefficients=(-0.4, 0.2, 0.3, -0.1),
    )
    model = GridCACEModel(
        readout=[CACEBranch(a, b, LocalReadout(hidden_sizes=(6,))), LDAReadout(), coulomb, kernel],
        ideal=FixedDipoleIdeal(MOMENT), grid_spacing=SPACING, mean_density=RHO,
        polarization_scale=0.1, mean_temperature=1.0, boltzmann_constant=1.0,
        rho_min=1.0e-3, compute_c1=False, compute_polarization_derivative=False,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    return model.eval()


class LocalCoupled(nn.Module):
    """``beta f_exc = a rho^2 + b |P|^2 + lam rho div P`` per voxel, central differences."""

    def __init__(self, n_types=1, a=0.4, b=-0.3, lam=0.2):
        super().__init__()
        self.ideal = FixedDipoleIdeal([MOMENT] * n_types)
        self.a, self.b, self.lam = a, b, lam

    def forward(self, batch, **kwargs):
        rho, pol = batch["rho"], batch["dipole_density"]
        shape = (*rho.shape[:-2], GRID, GRID, GRID, rho.shape[-1])
        field = pol.reshape(*shape, 3)
        divergence = sum(
            (torch.roll(field[..., d], -1, dims=-4 + d) - torch.roll(field[..., d], 1, dims=-4 + d)) / (2 * SPACING)
            for d in range(3)
        ).reshape(rho.shape)
        species = torch.arange(rho.shape[-1], dtype=rho.dtype) + 1.0
        local = (self.a * species * rho**2 + self.b * pol.square().sum(-1)
                 + self.lam * species * rho.flip(-1) * divergence)
        volume = torch.prod(batch["grid_spacing"].to(rho), dim=-1)
        return {"beta_F_exc": volume * local.sum(dim=(-2, -1))}


VECTORS = torch.tensor([[[1, 0, 0], [0, 2, 1], [1, 1, 1], [2, -1, 0], [3, 0, 0]]])


class TestBulkResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls._dtype)

    def test_symbol_matches_energy_second_differences(self):
        # Independent of autograd: for a real wave Re(w exp(i k.r)) the
        # second difference of the excess energy is (N/2) eps^2 Re(w^H H w).
        model, data = cace_model(), bulk()
        symbol = bulk_response_symbol(model, data, VECTORS, create_graph=False)[0]
        volume = SPACING ** 3
        scale = torch.tensor([math.sqrt(RHO / volume)] + [math.sqrt(RHO / volume) * MOMENT / math.sqrt(3)] * 3)
        excess = (symbol - torch.eye(4)) / (scale[:, None] * scale[None, :])
        x = positions().double()
        energy = lambda z: _energy_only_model(model, dict(data, rho=data["rho"] + z[..., :1], dipole_density=data["dipole_density"] + z[..., None, 1:]))["beta_F_exc"].item()
        epsilon = torch.tensor([1e-4 * RHO] + [1e-4 * MOMENT * RHO] * 3)
        for v, vector in enumerate(VECTORS[0]):
            phase = 2 * math.pi * (x * vector.double() / GRID).sum(-1)
            wave = torch.polar(torch.ones_like(phase), phase)
            for i, j in ((0, 0), (1, 1), (2, 3), (0, 1), (0, 2), (1, 3)):
                for coefficient in ((1.0, 0.0), (1.0, 1.0), (1.0, 1.0j)):
                    if i == j and coefficient != (1.0, 0.0):
                        continue
                    w = torch.zeros(4, dtype=torch.complex128)
                    w[i] += coefficient[0]
                    w[j] += coefficient[1]
                    w = w * epsilon.to(w.dtype)
                    field = (w[None, :] * wave[:, None]).real[None]
                    second = energy(field) + energy(-field) - 2 * energy(torch.zeros_like(field))
                    expected = 0.5 * GRID ** 3 * (w.conj() @ excess[v] @ w).real.item()
                    with self.subTest(vector=vector.tolist(), pair=(i, j), coefficient=coefficient):
                        self.assertAlmostEqual(second, expected, delta=2e-6 * max(1.0, abs(expected)) + 1e-12)

    def test_symbol_is_identity_for_the_ideal_gas_and_hermitian(self):
        model = LocalCoupled(a=0.0, b=0.0, lam=0.0)
        symbol = bulk_response_symbol(model, bulk(), VECTORS, create_graph=False)
        torch.testing.assert_close(symbol, torch.eye(4, dtype=symbol.dtype).expand_as(symbol), rtol=0, atol=1e-12)
        symbol = bulk_response_symbol(cace_model(), bulk(), VECTORS, create_graph=False)
        torch.testing.assert_close(symbol, symbol.conj().transpose(-1, -2), rtol=0, atol=1e-10)

    def test_local_model_has_the_analytic_blocks(self):
        # a rho^2 -> 2 a rho/dV * dV... in normalized units: K_rr = 1 + 2 a rho;
        # b |P|^2 -> K_PP = 1 + 2 b m^2 rho / 3; lam rho div P -> Im K_rL.
        model = LocalCoupled(a=0.4, b=-0.3, lam=0.2)
        symbol = bulk_response_symbol(model, bulk(), VECTORS, create_graph=False)
        lengths = torch.full((1, 3), GRID * SPACING)
        blocks = reduce_bulk_symbol(symbol, VECTORS, lengths)
        k = 2 * math.pi * VECTORS[0].double() / (GRID * SPACING)
        khat = k / k.norm(dim=-1, keepdim=True)
        sine = torch.sin(k * SPACING) / SPACING                       # central-difference derivative
        expected_cross = 0.2 * RHO * MOMENT / math.sqrt(3) * (sine * khat).sum(-1)
        torch.testing.assert_close(blocks["density"][0, :, 0, 0], torch.full((5,), 1 + 2 * 0.4 * RHO))
        torch.testing.assert_close(blocks["transverse"][0, :, 0, 0], torch.full((5,), 1 + 2 * -0.3 * MOMENT**2 * RHO / 3))
        torch.testing.assert_close(blocks["longitudinal"][0, :, 0, 0], torch.full((5,), 1 + 2 * -0.3 * MOMENT**2 * RHO / 3))
        torch.testing.assert_close(blocks["cross"][0, :, 0, 0].abs(), expected_cross.abs())
        # The central difference is not isotropic off the axes: it couples
        # density to transverse polarization there, which the reduction reports.
        on_axis = [0, 4]
        self.assertLess(blocks["anisotropy"][0, on_axis].max().item(), 1e-12)
        self.assertGreater(blocks["anisotropy"][0, 1].item(), 1e-4)
        self.assertLess(blocks["anisotropy"].max().item(), 0.05)

    def batch_with_targets(self, model, data, perturb=0.0, members=1):
        """Targets equal to the model's own bin-averaged curvature, optionally perturbed."""

        indices = VECTORS.reshape(1, 5, 1, 3)
        if members == 2:
            indices = torch.stack((VECTORS[0], -VECTORS[0]), dim=1)[None]    # inversion partners
        symbol = bulk_response_symbol(model, data, indices.reshape(1, -1, 3), create_graph=False)
        lengths = torch.full((1, 3), GRID * SPACING)
        blocks = reduce_bulk_symbol(symbol, indices.reshape(1, -1, 3), lengths)
        mean = {name: blocks[name].reshape(1, 5, members, 1, 1).mean(2) for name in ("density", "longitudinal", "transverse", "cross")}
        spectra = curvature_to_spectra(mean) * (1.0 + perturb)
        return dict(
            data,
            response_wavevector_indices=indices,
            response_member_mask=torch.ones(indices.shape[:-1], dtype=torch.bool),
            response_spectra=spectra,
            response_spectra_sem=0.01 * spectra.abs() + 1e-6,
        )

    def test_loss_vanishes_on_own_spectra_and_trains(self):
        model, data = cace_model(), bulk()
        for members in (1, 2):
            with self.subTest(members=members):
                term = BulkResponseLoss()
                exact = self.batch_with_targets(model, data, members=members)
                self.assertLess(term(model(exact), exact, model=model).item(), 1e-16)
        perturbed = self.batch_with_targets(model, data, perturb=0.1)
        term = BulkResponseLoss()
        model.train()
        loss = term(model(perturbed), perturbed, model=model)
        self.assertGreater(loss.item(), 1e-3)
        loss.backward()
        gradients = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients and max(gradients) > 0.0 and all(math.isfinite(g) for g in gradients))
        # Round trip of the conversion.
        spectra = perturbed["response_spectra"]
        torch.testing.assert_close(curvature_to_spectra(spectra_to_curvature(spectra)), spectra)

    def test_sem_propagation_and_zero_wavevector(self):
        spectra = torch.zeros(1, 1, 4, 1, 1)
        spectra[..., 0, 0, 0], spectra[..., 1, 0, 0], spectra[..., 2, 0, 0], spectra[..., 3, 0, 0] = 0.06, 0.055, 3.6, -0.003
        sem = torch.zeros_like(spectra)
        sem[..., 2, 0, 0] = 0.04
        model = LocalCoupled(a=0.0, b=0.0, lam=0.0)
        data = dict(
            bulk(), response_wavevector_indices=torch.tensor([[[[1, 0, 0]]]]),
            response_member_mask=torch.ones(1, 1, 1, dtype=torch.bool),
            response_spectra=spectra, response_spectra_sem=sem,
            response_gk=torch.tensor([[[3.75]]]), response_gk_sem=torch.tensor([[[0.09]]]),
        )
        term = BulkResponseLoss(zero_wavevector_key="response_gk", zero_wavevector_sem_key="response_gk_sem",
                                relative_floor=1e-12)
        result = term.evaluate(model(data), data, model=model)
        # Order: K_rr, K_LL, K_TT, Im K_rL; only G_T carries SEM here.
        self.assertAlmostEqual(result["scale"][0, 0, 2].item(), 0.04 / 3.6**2, places=12)
        self.assertLess(result["scale"][0, 0, [0, 1, 3]].max().item(), 1e-9)
        self.assertAlmostEqual(result["target"][0, 0, 2].item(), 1 / 3.6, places=12)
        self.assertAlmostEqual(result["zero_target"].item(), 1 / 3.75, places=12)
        self.assertAlmostEqual(result["zero_scale"].item(), 0.09 / 3.75**2, places=12)
        self.assertAlmostEqual(result["zero_prediction"].item(), 1.0, places=12)   # ideal gas at k = 0
        self.assertTrue(math.isfinite(result["loss"].item()))

    def test_unstable_model_gives_a_finite_loss(self):
        model = LocalCoupled(a=-3.0, b=-2.0, lam=0.0)    # K_rr = 1 - 3 < 0, K_PP < 0
        data = self.batch_with_targets(LocalCoupled(a=0.0, b=0.0, lam=0.0), bulk())
        result = BulkResponseLoss().evaluate(model(data), data, model=model)
        self.assertLess(result["prediction"][..., 0].max().item(), 0.0)
        self.assertTrue(math.isfinite(result["loss"].item()) and result["loss"].item() > 1.0)

    def test_two_species(self):
        model = LocalCoupled(n_types=2, a=0.3, b=-0.2, lam=0.25)
        data = bulk(n_types=2)
        indices = VECTORS.reshape(1, 5, 1, 3)
        symbol = bulk_response_symbol(model, data, VECTORS, create_graph=False)
        self.assertEqual(symbol.shape, (1, 5, 8, 8))
        blocks = reduce_bulk_symbol(symbol, VECTORS, torch.full((1, 3), GRID * SPACING))
        spectra = curvature_to_spectra(blocks)
        data.update(response_wavevector_indices=indices,
                    response_member_mask=torch.ones(1, 5, 1, dtype=torch.bool), response_spectra=spectra)
        data.pop("response_spectra_sem", None)
        result = BulkResponseLoss().evaluate(model(data), data, model=model)
        self.assertEqual(result["prediction"].shape, (1, 5, 3 * 3 + 4))
        self.assertLess(result["loss"].item(), 1e-16)
        # The cross block is not symmetric here (species flip in the coupling).
        self.assertGreater((blocks["cross"] - blocks["cross"].transpose(-1, -2)).abs().max().item(), 1e-3)

    def test_padding_members_do_not_poison_the_loss(self):
        # Regression: a masked (0, 0, 0) padding member has no wavevector
        # frame and used to turn the loss into NaN (0 * NaN).
        model, data = cace_model(), bulk()
        exact = self.batch_with_targets(model, data, perturb=0.05)
        padded = dict(exact)
        padded["response_wavevector_indices"] = torch.cat(
            (exact["response_wavevector_indices"], torch.zeros(1, 5, 1, 3, dtype=torch.long)), dim=2,
        )
        padded["response_member_mask"] = torch.cat(
            (exact["response_member_mask"], torch.zeros(1, 5, 1, dtype=torch.bool)), dim=2,
        )
        for mode in ("covariance", "curvature"):
            with self.subTest(mode=mode):
                term = BulkResponseLoss(bin_average=mode)
                reference = term.evaluate(model(exact), exact, model=model)["loss"]
                value = term.evaluate(model(padded), padded, model=model)["loss"]
                self.assertTrue(math.isfinite(value.item()))
                torch.testing.assert_close(value, reference)
        unmasked_zero = dict(padded, response_member_mask=torch.ones(1, 5, 2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            BulkResponseLoss().evaluate(model(unmasked_zero), unmasked_zero, model=model)

    def test_empty_bin_is_skipped(self):
        model, data = cace_model(), bulk()
        exact = self.batch_with_targets(model, data, perturb=0.05)
        padded = dict(exact)
        for key in ("response_wavevector_indices", "response_member_mask"):
            padded[key] = torch.cat((exact[key], torch.zeros_like(exact[key][:, :1])), dim=1)
        for key in ("response_spectra", "response_spectra_sem"):
            padded[key] = torch.cat((exact[key], exact[key][:, :1]), dim=1)

        for mode in ("covariance", "curvature"):
            with self.subTest(mode=mode):
                term = BulkResponseLoss(bin_average=mode)
                reference = term.evaluate(model(exact), exact, model=model)
                result = term.evaluate(model(padded), padded, model=model)
                torch.testing.assert_close(result["loss"], reference["loss"])
                self.assertEqual(result["element_weight"][:, -1].sum().item(), 0.0)
                self.assertEqual(result["prediction"][:, -1].abs().max().item(), 0.0)
                self.assertFalse(bool(result["unstable_bins"][:, -1].any()))

    def test_bins_average_the_reference_statistic(self):
        # Regression: a bin of two different wavevectors whose reference is the
        # mean of the model's own member spectra (as simulation averages a
        # bin). Covariance averaging reproduces it exactly; arithmetic
        # curvature averaging does not, unless members are symmetry related.
        model, data = cace_model(), bulk()
        lengths = torch.full((1, 3), GRID * SPACING)

        def own_spectra(members):
            indices = torch.tensor([members])                                  # [1, m, 3]
            symbol = bulk_response_symbol(model, data, indices, create_graph=False)
            blocks = reduce_bulk_symbol(symbol, indices, lengths)
            spectra = curvature_to_spectra({k: v for k, v in blocks.items() if k != "anisotropy"})
            return spectra.mean(1, keepdim=True)                               # [1, 1, 4, 1, 1]

        def batch(members):
            return dict(
                data, response_wavevector_indices=torch.tensor([[members]]),
                response_member_mask=torch.ones(1, 1, len(members), dtype=torch.bool),
                response_spectra=own_spectra(members),
            )

        mixed = batch([[1, 0, 0], [2, 1, 0]])
        covariance = BulkResponseLoss(bin_average="covariance").evaluate(model(mixed), mixed, model=model)
        curvature = BulkResponseLoss(bin_average="curvature").evaluate(model(mixed), mixed, model=model)
        self.assertLess(covariance["loss"].item(), 1e-16)
        self.assertGreater(curvature["loss"].item(), 1e-4)
        self.assertFalse(bool(covariance["unstable_bins"].any()))
        # A cubic orbit: both averages coincide.
        orbit = batch([[1, 2, 0], [0, 1, 2], [2, 0, 1], [-1, 0, 2], [0, -2, 1]])
        for mode in ("covariance", "curvature"):
            with self.subTest(mode=mode):
                result = BulkResponseLoss(bin_average=mode).evaluate(model(orbit), orbit, model=model)
                self.assertLess(result["loss"].item(), 1e-12)
        with self.assertRaisesRegex(ValueError, "bin_average"):
            BulkResponseLoss(bin_average="mean")

    def test_unstable_bins_are_reported(self):
        unstable = LocalCoupled(a=-3.0, b=0.0, lam=0.0)
        data = self.batch_with_targets(LocalCoupled(a=0.0, b=0.0, lam=0.0), bulk())
        result = BulkResponseLoss().evaluate(unstable(data), data, model=unstable)
        self.assertTrue(bool(result["unstable_bins"].all()))
        self.assertTrue(math.isfinite(result["loss"].item()) and result["loss"].item() > 1.0)

    def test_ideal_only_and_density_only_models(self):
        # Regression: an excess independent of rho and/or P used to raise
        # "differentiated tensors not used" instead of giving the identity.
        ideal_only = GridCACEModel(
            readout=[], n_types=1, ideal=FixedDipoleIdeal(MOMENT), grid_spacing=SPACING, mean_density=RHO,
            polarization_scale=0.1, mean_temperature=1.0, boltzmann_constant=1.0, rho_min=1.0e-3,
            compute_c1=False, compute_polarization_derivative=False,
        ).eval()
        density_only = GridCACEModel(
            readout=[LDAReadout()], ideal=FixedDipoleIdeal(MOMENT), grid_spacing=SPACING, mean_density=RHO,
            polarization_scale=0.1, mean_temperature=1.0, boltzmann_constant=1.0, rho_min=1.0e-3,
            compute_c1=False, compute_polarization_derivative=False,
        ).eval()
        with torch.no_grad():
            for parameter in density_only.parameters():
                parameter.add_(0.1 * torch.randn_like(parameter))
        identity = torch.eye(4, dtype=torch.complex128).expand(1, 5, 4, 4)
        for create_graph in (False, True):
            with self.subTest(create_graph=create_graph):
                symbol = bulk_response_symbol(ideal_only, bulk(), VECTORS, create_graph=create_graph)
                torch.testing.assert_close(symbol, identity, rtol=0, atol=0)
                symbol = bulk_response_symbol(density_only, bulk(), VECTORS, create_graph=create_graph)
                torch.testing.assert_close(symbol[..., 1:, :], identity[..., 1:, :], rtol=0, atol=0)
                self.assertGreater((symbol[..., 0, 0] - 1).abs().max().item(), 1e-6)
        data = self.batch_with_targets(LocalCoupled(a=0.0, b=0.0, lam=0.0), bulk())
        result = BulkResponseLoss().evaluate(ideal_only(data), data, model=ideal_only)
        self.assertLess(result["loss"].item(), 1e-16)

    def test_validation(self):
        model = LocalCoupled()
        data = bulk()
        uneven = dict(data, rho=data["rho"].clone())
        uneven["rho"][0, 3, 0] *= 1.1
        with self.assertRaisesRegex(ValueError, "uniform"):
            bulk_response_symbol(model, uneven, VECTORS)
        polarized = dict(data, dipole_density=data["dipole_density"] + 1e-3)
        with self.assertRaisesRegex(ValueError, "unpolarized"):
            bulk_response_symbol(model, polarized, VECTORS)
        with self.assertRaisesRegex(ValueError, "integer"):
            bulk_response_symbol(model, data, VECTORS.double())
        with self.assertRaisesRegex(TypeError, "FixedDipoleIdeal"):
            bad = LocalCoupled()
            bad.ideal = None
            bulk_response_symbol(bad, data, VECTORS)
        with self.assertRaisesRegex(ValueError, "zero_wavevector_key"):
            BulkResponseLoss(zero_wavevector_sem_key="x")
        with self.assertRaisesRegex(ValueError, "relative_floor"):
            BulkResponseLoss(relative_floor=0.0)


if __name__ == "__main__":
    unittest.main()
