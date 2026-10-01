"""The fixed MSA / Percus-Yevick reference of the restricted primitive model."""

import math
import os
import tempfile
import unittest

import numpy as np
import torch

from equicdft import (
    FourierResponse,
    GridCACEModel,
    LongRangeReadout,
    MetalWall,
    MSAReferenceReadout,
    ReciprocalFeatures,
    load_model,
    save_model,
)
from equicdft._config import build, registered_types
from equicdft.functional.base import ReadoutContext
from equicdft.functional.msa import (
    msa_electrostatic_direct_correlation,
    msa_screening_parameter,
    percus_yevick_direct_correlation,
    scaled_particle_free_energy,
)


T_STAR = 0.0660678488938004
AMPLITUDE = 1.0 / T_STAR
RHO_SPECIES = 0.2
ALPHA = 0.20
SPACING = 0.25
GRID = 32

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy < 2.0 has only trapz


def _homogeneous_batch(n_fields=1, grid=GRID, spacing=SPACING, dtype=torch.float64):
    positions = torch.tensor(np.indices((grid, grid, grid)).reshape(3, -1).T)
    return {
        "rho": torch.full((n_fields, grid ** 3, 2), RHO_SPECIES, dtype=dtype),
        "temperature": torch.ones(n_fields, dtype=dtype),
        "grid_spacing": torch.full((n_fields, 3), spacing, dtype=dtype),
        "grid_size": torch.tensor([[grid, grid, grid]] * n_fields),
        "grid_positions": torch.stack([positions] * n_fields),
    }


def _long_range(alpha, amplitude=AMPLITUDE, charges=(1.0, -1.0)):
    return LongRangeReadout(
        n_kernels=1, n_types=2, charges=charges, coulomb_amplitude=amplitude,
        features=ReciprocalFeatures(kernel="coulomb", n_types=2, radial_exponents=(alpha,)),
    )


def _msa(**overrides):
    arguments = dict(
        charges=(1.0, -1.0), diameter=1.0, coulomb_amplitude=AMPLITUDE,
        long_range_alpha=ALPHA,
    )
    arguments.update(overrides)
    return MSAReferenceReadout(2, **arguments)


def _wall(liquid):
    sites = {
        "metal_positions": torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 1.5]], dtype=torch.float64),
        "metal_site_groups": torch.tensor([0, 0]),
        "metal_group_ids": torch.tensor([0]),
        "metal_total_charge": torch.tensor([0.0], dtype=torch.float64),
        "metal_charge_units": "e",
    }
    return MetalWall(liquid, metal_sites=sites, metal_sigma=0.35).double()


def _model(readouts, spacing=SPACING, **kwargs):
    arguments = dict(
        a_features=None, b_features=None, readout=readouts, grid_spacing=spacing,
        mean_temperature=1.0, boltzmann_constant=1.0, compute_c1=False,
        mean_density=2 * RHO_SPECIES,
    )
    arguments.update(kwargs)
    return GridCACEModel(**arguments).to(dtype=torch.float64)


def _analytic_msa(k_values, rho_species=RHO_SPECIES):
    """Independent quadrature of the full MSA / PY inverse structure factors."""

    rho = 2 * rho_species
    kappa = math.sqrt(4 * math.pi * AMPLITUDE * rho)
    B = (kappa + 1 - math.sqrt(1 + 2 * kappa)) / kappa
    eta = math.pi * rho / 6
    r = np.linspace(1e-9, 1.0, 100001)
    c_el = -AMPLITUDE * (2 * B - B * B * r)
    c_hs = -((1 + 2 * eta) ** 2 - 6 * eta * (1 + eta / 2) ** 2 * r
             + 0.5 * eta * (1 + 2 * eta) ** 2 * r ** 3) / (1 - eta) ** 4
    charge, number = [], []
    for k in k_values:
        weight = 4 * np.pi * r ** 2 * np.sinc(k * r / np.pi)
        c_el_hat = _trapezoid(weight * c_el, r) - 4 * np.pi * AMPLITUDE * np.cos(k) / k ** 2
        charge.append(1 - rho * c_el_hat)
        number.append(1 - rho * _trapezoid(weight * c_hs, r))
    return np.array(charge), np.array(number)


class TestClosedForms(unittest.TestCase):
    def test_screening_parameter_limits_and_stability(self):
        self.assertEqual(msa_screening_parameter(0.0, 1.0), 0.0)
        self.assertAlmostEqual(msa_screening_parameter(8.7215, 1.0), 0.6223, places=3)
        self.assertLess(msa_screening_parameter(1e6, 1.0), 1.0)
        # The dilute limit B -> x/2 survives in float32 and at x = 1e-8 in float64.
        self.assertAlmostEqual(msa_screening_parameter(1e-8, 1.0) / 0.5e-8, 1.0, places=6)
        small = msa_screening_parameter(torch.tensor([1e-4, 1e-2], dtype=torch.float32), 1.0)
        torch.testing.assert_close(small / torch.tensor([0.5e-4, 0.5e-2]), torch.ones(2), rtol=1e-2, atol=0)
        self.assertEqual(msa_screening_parameter(torch.zeros(1, dtype=torch.float64), 1.0).item(), 0.0)

    def test_direct_correlations_are_bare_outside_the_core(self):
        r = torch.tensor([0.0, 0.5, 0.999, 1.001, 2.0], dtype=torch.float64)
        c_el = msa_electrostatic_direct_correlation(r, 1.0, AMPLITUDE, 0.6)
        torch.testing.assert_close(c_el[3:], -AMPLITUDE / r[3:])
        self.assertAlmostEqual(c_el[0].item(), -AMPLITUDE * 1.2)
        c_hs = percus_yevick_direct_correlation(r, 1.0, 0.2)
        self.assertTrue(torch.all(c_hs[3:] == 0))
        self.assertAlmostEqual(c_hs[0].item(), -(1.4 ** 2) / 0.8 ** 4)


class TestMSAReferenceReadout(unittest.TestCase):
    K_INDICES = ((1, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0), (8, 0, 0), (12, 0, 0), (16, 0, 0))

    def _curvature(self, model, directions, batch=None):
        batch = _homogeneous_batch() if batch is None else batch
        indices = torch.tensor([list(self.K_INDICES)] * batch["rho"].shape[0])
        curvature, valid = FourierResponse(relative_amplitude=0.01)(model, batch, indices, directions)
        # The cosine phase is valid everywhere; the Nyquist sine is identically zero.
        self.assertTrue(bool(valid[:, :, 0, :].all()))
        return curvature[:, :, 0, :]

    def _wavenumbers(self):
        return np.array([2 * math.pi * i[0] / (GRID * SPACING) for i in self.K_INDICES])

    def test_long_range_partner_alone_is_smeared_debye_hueckel(self):
        curvature = self._curvature(_model([_long_range(ALPHA)]), torch.tensor([[1.0, -1.0]]))
        kappa_squared = 4 * math.pi * AMPLITUDE * 2 * RHO_SPECIES
        expected = 1 + kappa_squared * np.exp(-ALPHA * self._wavenumbers() ** 2) / self._wavenumbers() ** 2
        np.testing.assert_allclose(curvature[0, :, 0].numpy(), expected, rtol=2e-4)

    def test_charge_response_is_the_analytic_msa_for_any_smearing(self):
        expected_charge, _ = _analytic_msa(self._wavenumbers())
        for alpha in (ALPHA, 0.05, 0.0):
            for owned in (True, False):
                with self.subTest(alpha=alpha, owned=owned):
                    if owned:
                        readouts = [_msa(long_range_alpha=alpha)]
                    else:
                        readouts = [_long_range(alpha), _msa(long_range_alpha=alpha, include_long_range=False)]
                    curvature = self._curvature(_model(readouts), torch.tensor([[1.0, -1.0]]))
                    np.testing.assert_allclose(curvature[0, :, 0].numpy(), expected_charge, rtol=1e-3)
        self.assertGreater(expected_charge.min(), 0.5)  # MSA never produces a pole here

    def test_number_response_is_percus_yevick(self):
        _, expected_number = _analytic_msa(self._wavenumbers())
        curvature = self._curvature(_model([_msa()]), torch.tensor([[1.0, 1.0]]))
        np.testing.assert_allclose(curvature[0, :, 0].numpy(), expected_number, rtol=1e-3)
        eta = math.pi * 2 * RHO_SPECIES / 6
        state = (RHO_SPECIES, RHO_SPECIES)
        number_zero, charge_zero = _msa().reciprocal_kernels(torch.zeros(1), state)
        self.assertAlmostEqual(1 + 2 * RHO_SPECIES * number_zero.item(), (1 + 2 * eta) ** 2 / (1 - eta) ** 4, places=6)
        self.assertTrue(math.isfinite(charge_zero.item()))
        # With an exact Coulomb partner only the inside-core polynomial remains.
        _, exact_partner = _msa(long_range_alpha=0.0).real_space_kernels(torch.tensor([1.5, 3.0]), state)
        torch.testing.assert_close(exact_partner, torch.zeros(2, dtype=torch.float64))
        _, smeared_partner = _msa().real_space_kernels(torch.tensor([1.5, 3.0]), state)
        r = torch.tensor([1.5, 3.0], dtype=torch.float64)
        torch.testing.assert_close(smeared_partner, AMPLITUDE * torch.erfc(r / (2 * math.sqrt(ALPHA))) / r)

    def test_reference_state_follows_each_field_by_default(self):
        low, high = 0.15, 0.25
        batch = _homogeneous_batch(n_fields=2)
        batch["rho"][0] = low
        batch["rho"][1] = high
        curvature = self._curvature(_model([_msa()]), torch.tensor([[1.0, -1.0], [1.0, 1.0]]), batch)
        for field, rho_species in ((0, low), (1, high)):
            expected_charge, expected_number = _analytic_msa(self._wavenumbers(), rho_species)
            np.testing.assert_allclose(curvature[field, :, 0].numpy(), expected_charge, rtol=1e-3)
            np.testing.assert_allclose(curvature[field, :, 1].numpy(), expected_number, rtol=1e-3)
        # A fixed reference density gives every field the same kernels instead.
        fixed = self._curvature(
            _model([_msa(reference_density=(RHO_SPECIES, RHO_SPECIES))]), torch.tensor([[1.0, 1.0]]), batch,
        )
        self.assertFalse(np.allclose(fixed[0, :, 0].numpy(), curvature[0, :, 1].numpy(), rtol=1e-3))
        # The reference state is read from the accessible region when a mask is present.
        readout = _msa()
        rho = torch.full((1, GRID ** 3, 2), RHO_SPECIES, dtype=torch.float64)
        excluded = torch.zeros(1, GRID ** 3, dtype=torch.bool)
        excluded[0, : GRID ** 3 // 2] = True
        rho[0, : GRID ** 3 // 2] = 0.0
        torch.testing.assert_close(readout.field_densities(rho, excluded), torch.full((1, 2), RHO_SPECIES, dtype=torch.float64))
        torch.testing.assert_close(readout.field_densities(rho, None), torch.full((1, 2), RHO_SPECIES / 2, dtype=torch.float64))

    def test_uniform_curvature_is_the_msa_py_compressibility(self):
        """The k = 0 Hessian of the variational reference is -c(0), as at k -> 0."""

        model = _model([_msa()])
        batch = _homogeneous_batch()
        eta = math.pi * 2 * RHO_SPECIES / 6
        expected = (1 + 2 * eta) ** 2 / (1 - eta) ** 4  # 1/S_NN(0), PY compressibility route

        def slope(scale):
            rho = (batch["rho"] * scale).requires_grad_(True)
            energy = model({**batch, "rho": rho}, compute_c1=False)["beta_F_exc"].sum()
            gradient, = torch.autograd.grad(energy, rho)
            return (gradient * batch["rho"]).sum().item()  # dE/d(scale)

        step = 1e-4
        curvature = (slope(1 + step) - slope(1 - step)) / (2 * step)  # d2E/dscale2
        volume = GRID ** 3 * SPACING ** 3
        rho_total = 2 * RHO_SPECIES
        self.assertAlmostEqual(1 + curvature / (volume * rho_total), expected, places=4)
        # ... and it is continuous with the smallest nonzero wavevector of the kernel.
        number_zero, _ = _msa().reciprocal_kernels(torch.zeros(1), (RHO_SPECIES, RHO_SPECIES))
        self.assertAlmostEqual(1 + rho_total * number_zero.item(), expected, places=6)
        # The bulk free energy is the compressibility-route partner of the kernel:
        # its second density derivative is K_N(0) at every density.
        for rho in (0.2, 0.4, 0.6):
            h = 1e-4
            second = (scaled_particle_free_energy(rho + h, 1.0) - 2 * scaled_particle_free_energy(rho, 1.0)
                      + scaled_particle_free_energy(rho - h, 1.0)) / h ** 2
            kernel_zero, _ = _msa().reciprocal_kernels(torch.zeros(1), (rho / 2, rho / 2))
            self.assertAlmostEqual(second / kernel_zero.item(), 1.0, places=5)

    def test_neutral_bulk_hessian_matches_both_msa_channels(self):
        readout = _msa(include_long_range=False)
        densities = torch.tensor(
            [RHO_SPECIES, RHO_SPECIES], dtype=torch.float64, requires_grad=True,
        )
        hessian = torch.autograd.functional.hessian(
            lambda state: readout.bulk_free_energy_density(state[None])[0],
            densities,
        )
        number_zero, charge_zero = readout.zero_wavevector_kernels(
            densities.detach()[None],
        )
        charges = torch.tensor(readout.charges, dtype=torch.float64)
        expected = (
            number_zero[0] * torch.ones((2, 2), dtype=torch.float64)
            + charge_zero[0] * charges[:, None] * charges[None, :]
        )
        torch.testing.assert_close(hessian, expected, rtol=1.0e-7, atol=5.0e-7)

    def test_reference_state_is_variational(self):
        """Autograd along a uniform density change equals the finite-difference slope."""

        model = _model([_msa()])
        batch = _homogeneous_batch()
        rho = batch["rho"].clone().requires_grad_(True)
        energy = model({**batch, "rho": rho}, compute_c1=False)["beta_F_exc"].sum()
        gradient, = torch.autograd.grad(energy, rho)
        direction = torch.ones_like(rho)
        autograd_slope = (gradient * direction).sum().item()
        step = 1e-4
        plus = model({**batch, "rho": batch["rho"] * (1 + step)}, compute_c1=False)["beta_F_exc"].sum().item()
        minus = model({**batch, "rho": batch["rho"] * (1 - step)}, compute_c1=False)["beta_F_exc"].sum().item()
        finite_difference = (plus - minus) / (2 * step * RHO_SPECIES)
        self.assertAlmostEqual(autograd_slope / finite_difference, 1.0, places=6)
        # A fixed reference state has no such term and is a plain quadratic form.
        outputs = _model([_msa(reference_density=(RHO_SPECIES, RHO_SPECIES))])(batch, compute_c1=True)
        number_zero, _ = _msa().reciprocal_kernels(torch.zeros(1), (RHO_SPECIES, RHO_SPECIES))
        torch.testing.assert_close(outputs["c1"], torch.full_like(outputs["c1"], -2 * RHO_SPECIES * number_zero.item()))

    def test_permuted_grid_rows_give_the_canonical_energy(self):
        torch.manual_seed(2)
        model = _model([_msa()])
        batch = _homogeneous_batch(grid=8)
        batch["rho"] = batch["rho"] * (
            1 + 0.4 * torch.rand(*batch["rho"].shape[:-1], 1, dtype=batch["rho"].dtype)
        )
        canonical = model(batch, compute_c1=True)
        order = torch.randperm(8 ** 3)
        permuted = dict(batch, rho=batch["rho"][:, order], grid_positions=batch["grid_positions"][:, order])
        result = model(permuted, compute_c1=True)
        torch.testing.assert_close(result["beta_F_exc"], canonical["beta_F_exc"])
        torch.testing.assert_close(result["c1"], canonical["c1"][:, order])

    def test_switches(self):
        curvature = self._curvature(_model([_msa(include_hard_sphere=False)]), torch.tensor([[1.0, 1.0]]))
        np.testing.assert_allclose(curvature[0, :, 0].numpy(), np.ones(len(self.K_INDICES)), atol=1e-4)  # ideal-gas finite-difference error ~ eps^2/8

    def test_batches_and_arbitrary_fields_are_supported(self):
        torch.manual_seed(0)
        model = _model([_msa()])
        batch = _homogeneous_batch(n_fields=2)
        batch["rho"] = batch["rho"] * (
            1 + 0.3 * torch.rand(*batch["rho"].shape[:-1], 1, dtype=batch["rho"].dtype)
        )
        batch["rho"].requires_grad_(True)
        energy = model(batch, compute_c1=False)["beta_F_exc"]
        self.assertEqual(tuple(energy.shape), (2,))
        separate = [model({key: value[[field]] for key, value in batch.items()}, compute_c1=False)["beta_F_exc"] for field in range(2)]
        torch.testing.assert_close(energy, torch.cat(separate))
        gradient, = torch.autograd.grad(energy.sum(), batch["rho"])
        self.assertTrue(torch.isfinite(gradient).all())

    def test_coulomb_partner_is_enforced_by_the_model(self):
        with self.assertRaisesRegex(ValueError, "owns the damped Coulomb branch"):
            _model([_long_range(ALPHA), _msa()])
        with self.assertRaisesRegex(ValueError, "one MSAReferenceReadout"):
            _model([_msa(), _msa()])
        unowned = dict(include_long_range=False)
        with self.assertRaisesRegex(ValueError, "exactly one charged Coulomb"):
            _model([_msa(**unowned)])
        with self.assertRaisesRegex(ValueError, "exactly one charged Coulomb"):
            _model([_long_range(0.1), _msa(**unowned)])
        with self.assertRaisesRegex(ValueError, "exactly one charged Coulomb"):
            _model([_long_range(ALPHA, amplitude=0.5 * AMPLITUDE), _msa(**unowned)])
        with self.assertRaisesRegex(ValueError, "exactly one charged Coulomb"):
            _model([_long_range(ALPHA, charges=(2.0, -2.0)), _msa(**unowned)])
        with self.assertRaisesRegex(ValueError, "exactly one charged Coulomb"):
            _model([_long_range(ALPHA), _long_range(ALPHA), _msa(**unowned)])
        learned_amplitude = LongRangeReadout(
            n_kernels=1, n_types=2, charges=(1.0, -1.0),
            features=ReciprocalFeatures(kernel="coulomb", n_types=2, radial_exponents=(ALPHA,)),
        )
        with self.assertRaisesRegex(ValueError, "exactly one charged Coulomb"):
            _model([learned_amplitude, _msa(**unowned)])
        # A gaussian long-range readout is not a Coulomb branch and is allowed alongside.
        gaussian = LongRangeReadout(
            n_kernels=1, n_types=2,
            features=ReciprocalFeatures(kernel="gaussian", n_types=2, radial_exponents=(0.5,)),
        )
        _model([gaussian, _msa()])
        _model([gaussian, _long_range(ALPHA), _msa(**unowned)])
        # The owned branch is reported as wrapped, so it cannot be listed again.
        owned = _msa()
        self.assertEqual(owned.wrapped_readouts(), (owned.long_range,))
        self.assertEqual(_msa(**unowned).wrapped_readouts(), ())
        with self.assertRaisesRegex(ValueError, "already evaluates its wrapped"):
            _model([owned, owned.long_range])

    def test_coulomb_partner_is_found_through_wrappers(self):
        unowned = dict(include_long_range=False)
        # An electrode wall wrapping the matching liquid Coulomb readout is the
        # correct composition for EDL calculations.
        wall = _wall(_long_range(ALPHA))
        model = _model([wall, _msa(**unowned)])
        self.assertIs(model.readout[0], wall)
        # The same wall beside an owning reference counts the liquid Coulomb energy twice.
        with self.assertRaisesRegex(ValueError, "top level or wrapped"):
            _model([_wall(_long_range(ALPHA)), _msa()])
        # A wrapped mismatch is rejected like a top-level one.
        with self.assertRaisesRegex(ValueError, "exactly one charged Coulomb"):
            _model([_wall(_long_range(0.1)), _msa(**unowned)])
        with self.assertRaisesRegex(ValueError, "exactly one charged Coulomb"):
            _model([_wall(_long_range(ALPHA)), _long_range(ALPHA), _msa(**unowned)])

    def test_unit_domain_is_enforced(self):
        batch = _homogeneous_batch()
        with self.assertRaisesRegex(ValueError, "free_energy_mode='beta'"):
            _model([_msa()], free_energy_mode="physical")(batch, compute_c1=False)
        with self.assertRaisesRegex(ValueError, "reference temperature"):
            _model([_msa()])({**batch, "temperature": torch.full((1,), 1.5, dtype=torch.float64)}, compute_c1=False)
        dense = dict(batch, rho=torch.full_like(batch["rho"], 1.0))
        with self.assertRaisesRegex(ValueError, "packing fraction"):
            _model([_msa()])(dense, compute_c1=False)

    def test_zero_screening_has_finite_exact_first_derivatives(self):
        vacuum = _homogeneous_batch(grid=8)
        vacuum["rho"].zero_()
        output = _model([_msa()])(vacuum, compute_c1=True, compute_ideal=False)
        torch.testing.assert_close(output["beta_F_exc"], torch.zeros_like(output["beta_F_exc"]))
        torch.testing.assert_close(output["c1"], torch.zeros_like(output["c1"]))

        no_coulomb = _model([
            _msa(coulomb_amplitude=0.0, include_hard_sphere=False),
        ])
        output = no_coulomb(
            _homogeneous_batch(grid=8), compute_c1=True, compute_ideal=False,
        )
        torch.testing.assert_close(output["beta_F_exc"], torch.zeros_like(output["beta_F_exc"]))
        torch.testing.assert_close(output["c1"], torch.zeros_like(output["c1"]))

    def test_reference_state_requires_global_not_local_neutrality(self):
        batch = _homogeneous_batch(grid=8)
        modulation = torch.linspace(
            -0.05, 0.05, batch["rho"].shape[-2], dtype=torch.float64,
        )
        batch["rho"][0, :, 0] += modulation
        batch["rho"][0, :, 1] -= modulation
        output = _model([_msa()])(batch, compute_c1=True, compute_ideal=False)
        self.assertTrue(torch.isfinite(output["beta_F_exc"]).all())
        self.assertTrue(torch.isfinite(output["c1"]).all())

        # Small float32 accumulation errors are not physical net charge.
        msa = _msa()
        msa.bulk_state(torch.tensor([0.2000001, 0.2], dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "electroneutral"):
            msa.bulk_state(torch.tensor([0.200001, 0.2], dtype=torch.float32))

        charged = _homogeneous_batch(grid=8)
        charged["rho"][..., 0] += 0.01
        with self.assertRaisesRegex(ValueError, "electroneutral"):
            _model([_msa()])(charged, compute_c1=False, compute_ideal=False)
        with self.assertRaisesRegex(ValueError, "electroneutral"):
            _msa(reference_density=(0.21, 0.20))

        # A neutral fixed bulk state remains usable for a finite liquid whose
        # net charge is compensated elsewhere, for example by electrodes.
        fixed = _model([_msa(reference_density=(RHO_SPECIES, RHO_SPECIES))])
        output = fixed(charged, compute_c1=True, compute_ideal=False)
        self.assertTrue(torch.isfinite(output["beta_F_exc"]).all())
        self.assertTrue(torch.isfinite(output["c1"]).all())

    def test_dynamic_state_inputs_are_validated_without_an_ideal_reference(self):
        for value in (-0.1, float("nan")):
            batch = _homogeneous_batch(grid=8)
            batch["rho"][0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, "finite nonnegative"):
                _model([_msa()])(batch, compute_c1=False, compute_ideal=False)

        batch = _homogeneous_batch(grid=8)
        batch["excluded_mask"] = torch.zeros(
            batch["rho"].shape[:-1], dtype=torch.int64,
        )
        with self.assertRaisesRegex(ValueError, "excluded_mask must be Boolean"):
            _model([_msa()])(batch, compute_c1=False, compute_ideal=False)

    def test_configuration_round_trip(self):
        self.assertIs(registered_types()["MSAReferenceReadout"], MSAReferenceReadout)
        readout = _msa(radial_resolution=500, reference_density=(RHO_SPECIES, RHO_SPECIES))
        rebuilt = build(readout.to_config())
        self.assertEqual(rebuilt.to_config(), readout.to_config())
        self.assertNotIn("long_range", readout.to_config())
        model = _model([readout])
        batch = _homogeneous_batch()
        batch["rho"] = batch["rho"] * (1 + 0.1 * torch.linspace(0, 1, GRID ** 3)[None, :, None])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            save_model(model, path)
            restored = load_model(path)
        torch.testing.assert_close(
            restored(batch, compute_c1=False)["beta_F_exc"], model(batch, compute_c1=False)["beta_F_exc"],
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 0)

    def test_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "n_types entries"):
            _msa(charges=(1.0,))
        with self.assertRaisesRegex(ValueError, "n_types entries"):
            _msa(reference_density=(0.2,))
        with self.assertRaisesRegex(ValueError, "packing fraction"):
            _msa(reference_density=(1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "long_range_alpha"):
            _msa(long_range_alpha=-0.1)
        with self.assertRaisesRegex(ValueError, "diameter"):
            _msa(diameter=0.0)
        with self.assertRaisesRegex(ValueError, "radial_resolution"):
            _msa(radial_resolution=10)
        with self.assertRaisesRegex(ValueError, "n_types"):
            _model([_msa()])({**_homogeneous_batch(), "rho": torch.full((1, GRID ** 3, 3), RHO_SPECIES)}, compute_c1=False)


if __name__ == "__main__":
    unittest.main()
