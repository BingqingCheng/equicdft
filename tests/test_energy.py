import unittest

import torch

from equicdft.energy import EnergyReadout, ReadoutContext
from equicdft.thermodynamics import (
    beta_external_potential,
    density_weighted_integral,
    equilibrium_log_density,
    ideal_free_energy,
    local_chemical_potential,
    log_dimensionless_density,
)


class TestDensityWeightedIntegral(unittest.TestCase):
    def test_local_multicomponent_integral(self):
        rho = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float64,
        )
        per_particle = torch.tensor(
            [[0.5, -1.0], [2.0, 0.25]],
            dtype=torch.float64,
        )

        energy = density_weighted_integral(
            rho,
            per_particle,
            torch.tensor(0.125),
        )

        self.assertEqual(energy.dtype, torch.float64)
        self.assertAlmostEqual(energy.item(), 0.6875)

    def test_spatially_constant_values_and_batched_volumes(self):
        rho = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 1.0], [4.0, 3.0]],
            ]
        )
        per_particle = torch.tensor(
            [[[2.0, -1.0]], [[0.5, 1.0]]]
        )

        energy = density_weighted_integral(
            rho,
            per_particle,
            torch.tensor([0.5, 0.25]),
        )

        self.assertTrue(torch.equal(energy, torch.tensor([1.0, 1.75])))

    def test_autograd_reaches_density_and_per_particle_values(self):
        rho = torch.tensor(
            [[1.0], [2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        per_particle = torch.tensor(
            [[3.0], [4.0]],
            dtype=torch.float64,
            requires_grad=True,
        )

        energy = density_weighted_integral(
            rho,
            per_particle,
            torch.tensor(0.5),
        )
        density_gradient, value_gradient = torch.autograd.grad(
            energy,
            (rho, per_particle),
        )

        self.assertTrue(
            torch.equal(
                density_gradient,
                torch.tensor([[1.5], [2.0]], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.equal(
                value_gradient,
                torch.tensor([[0.5], [1.0]], dtype=torch.float64),
            )
        )

    def test_zero_density_is_exactly_zero(self):
        rho = torch.zeros(2, 3, 1)
        per_particle = torch.randn_like(rho)

        energy = density_weighted_integral(rho, per_particle, 0.125)

        self.assertTrue(torch.equal(energy, torch.zeros(2)))

    def test_invalid_shapes_and_volume_are_rejected(self):
        rho = torch.ones(2, 3, 1)
        with self.assertRaisesRegex(ValueError, "rank"):
            density_weighted_integral(rho, torch.ones(3, 1), 1.0)
        with self.assertRaisesRegex(ValueError, "one or n_grid"):
            density_weighted_integral(rho, torch.ones(2, 2, 1), 1.0)
        with self.assertRaisesRegex(ValueError, "leading field shape"):
            density_weighted_integral(
                rho,
                torch.ones_like(rho),
                torch.ones(3),
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            density_weighted_integral(rho, torch.ones_like(rho), 0.0)


class TestIdealGasThermodynamics(unittest.TestCase):
    def test_log_density_and_vacuum_convention(self):
        rho = torch.tensor([[0.0, 0.5], [2.0, 1.0]], dtype=torch.float64)
        wavelength = torch.tensor([2.0, 3.0], dtype=torch.float64)

        value = log_dimensionless_density(rho, wavelength)

        self.assertTrue(torch.all(torch.isfinite(value)))
        self.assertEqual(value[1, 0], rho.new_tensor(16.0).log())
        self.assertEqual(value[1, 1], rho.new_tensor(27.0).log())

    def test_positive_subnormal_density_is_floored(self):
        tiny = torch.finfo(torch.float64).tiny
        rho = torch.tensor([[0.5 * tiny]], dtype=torch.float64)
        wavelength = torch.ones(1, dtype=torch.float64)

        self.assertEqual(
            log_dimensionless_density(rho, wavelength).item(),
            rho.new_tensor(tiny).log().item(),
        )

    def test_thermal_wavelength_contributes_expected_linear_term(self):
        rho = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float64)
        wavelength = torch.tensor([2.0, 3.0], dtype=torch.float64)
        volume = torch.tensor(0.25, dtype=torch.float64)

        full = ideal_free_energy(rho, wavelength, volume)
        unit_wavelength = ideal_free_energy(
            rho,
            torch.ones_like(wavelength),
            volume,
        )
        expected_offset = volume * torch.sum(
            rho * (3.0 * torch.log(wavelength))[None, :]
        )

        self.assertTrue(
            torch.allclose(full - unit_wavelength, expected_offset)
        )

    def test_vacuum_has_exactly_zero_ideal_free_energy(self):
        rho = torch.zeros((2, 3, 2), dtype=torch.float64)
        wavelength = torch.tensor([2.0, 3.0], dtype=torch.float64)

        value = ideal_free_energy(rho, wavelength, 0.125)

        self.assertTrue(torch.equal(value, torch.zeros(2, dtype=rho.dtype)))


class TestReadoutContext(unittest.TestCase):
    @staticmethod
    def _context(**overrides):
        arguments = dict(
            rho=torch.tensor([[[0.2, 0.4], [0.6, 0.8]], [[1.0, 1.0], [0.5, 1.5]]]),
            normalized_temperature=torch.tensor([1.5, 0.5]),
            voxel_volume=0.125,
            mean_density=2.0,
        )
        arguments.update(overrides)
        return ReadoutContext(**arguments)

    def test_primitives_are_validated_and_coerced(self):
        context = self._context()
        self.assertEqual(context.voxel_volume.dtype, context.rho.dtype)
        self.assertEqual(context.mean_density.shape, ())
        with self.assertRaisesRegex(ValueError, "normalized_temperature"):
            self._context(normalized_temperature=torch.tensor(1.0))
        with self.assertRaisesRegex(ValueError, "rho"):
            self._context(rho=torch.ones(3))
        with self.assertRaisesRegex(ValueError, "free_energy_mode"):
            self._context(free_energy_mode="reduced")
        with self.assertRaisesRegex(ValueError, "mean_density"):
            self._context(mean_density=torch.ones(2))

    def test_require_names_the_missing_primitive(self):
        context = self._context()
        with self.assertRaisesRegex(KeyError, "grid_size"):
            context.require("grid_size")
        with self.assertRaisesRegex(KeyError, "invariants"):
            context.local_features()
        without_scale = self._context(mean_density=None)
        with self.assertRaisesRegex(KeyError, "mean_density"):
            without_scale.normalized_rho()
        with self.assertRaisesRegex(KeyError, "mean_density"):
            without_scale.state_features()
        context.grid_size = torch.tensor([1, 1, 2])
        self.assertIs(context.require("grid_size"), context.grid_size)

    def test_composers_follow_the_documented_layouts_and_are_cached(self):
        rho = torch.tensor([[[0.2, 0.4], [0.6, 0.8]], [[1.0, 1.0], [0.5, 1.5]]])
        invariants = torch.arange(12.0).reshape(2, 2, 3)
        center = torch.tensor([[[0.1], [0.3]], [[0.5], [0.25]]])
        context = self._context(invariants=invariants, center_density=center)

        torch.testing.assert_close(context.normalized_rho(), rho / 2.0)
        temperature = context.temperature_feature()
        self.assertEqual(temperature.shape, (2, 2, 1))
        torch.testing.assert_close(temperature[..., 0], torch.tensor([[1.5, 1.5], [0.5, 0.5]]))
        torch.testing.assert_close(
            context.pointwise_state(), torch.cat((rho / 2.0, temperature), dim=-1),
        )
        torch.testing.assert_close(
            context.local_features(), torch.cat((center, invariants, temperature), dim=-1),
        )
        torch.testing.assert_close(
            context.state_features(),
            torch.cat((torch.tensor([[1.5], [0.5]]), rho.mean(-2) / 2.0), dim=-1),
        )
        for compose in (
            context.normalized_rho, context.temperature_feature, context.pointwise_state,
            context.local_features, context.state_features,
        ):
            self.assertIs(compose(), compose())
        # Without a separated center the local vector starts with the invariants.
        plain = self._context(invariants=invariants)
        torch.testing.assert_close(
            plain.local_features(), torch.cat((invariants, temperature), dim=-1),
        )

    def test_state_features_stay_connected_to_rho(self):
        rho = torch.rand(2, 4, 1, dtype=torch.float64, requires_grad=True)
        context = self._context(rho=rho, mean_density=0.5)
        gradient = torch.autograd.grad(context.state_features().sum(), rho)[0]
        torch.testing.assert_close(gradient, torch.full_like(rho, 0.5))

    def test_from_data_applies_the_unit_contract(self):
        data = {
            "rho": torch.rand(2, 8, 1, dtype=torch.float64),
            "temperature": torch.tensor([1.2, 2.4], dtype=torch.float64),
            "grid_size": torch.tensor([2, 2, 2]),
            "dipole_density": torch.zeros(2, 8, 1, 3, dtype=torch.float64),
            "excluded_mask": torch.zeros(2, 8, dtype=torch.bool),
        }
        context = ReadoutContext.from_data(
            data,
            grid_spacing=torch.tensor([0.5, 0.5, 1.0]),
            mean_temperature=1.2,
            boltzmann_constant=2.0,
            mean_density=0.3,
            free_energy_mode="physical",
        )
        torch.testing.assert_close(
            context.normalized_temperature, torch.tensor([1.0, 2.0], dtype=torch.float64),
        )
        torch.testing.assert_close(context.voxel_volume, torch.tensor(0.25, dtype=torch.float64))
        torch.testing.assert_close(
            context.beta, 1.0 / (2.0 * data["temperature"]),
        )
        torch.testing.assert_close(context.reference_energy, torch.tensor(2.4, dtype=torch.float64))
        self.assertEqual(context.free_energy_mode, "physical")
        self.assertEqual(context.grid_spacing.dtype, torch.float64)
        self.assertIs(context.grid_size, data["grid_size"])
        self.assertIs(context.dipole_density, data["dipole_density"])
        self.assertIs(context.excluded_mask, data["excluded_mask"])
        self.assertIsNone(context.grid_center)
        self.assertIsNone(context.invariants)
        with self.assertRaisesRegex(ValueError, "temperature"):
            ReadoutContext.from_data(
                dict(data, temperature=torch.tensor(1.0)), grid_spacing=0.5,
            )

    def test_readout_base_declares_its_requirements(self):
        self.assertFalse(EnergyReadout.requires_local_features)
        self.assertFalse(EnergyReadout.requires_local_density_index)
        self.assertFalse(EnergyReadout.requires_dipole_density)
        self.assertIsNone(EnergyReadout.n_types)
        with self.assertRaises(NotImplementedError):
            EnergyReadout().energy(self._context())


class TestEulerLagrangeRelations(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(3)
        self.rho = torch.rand(2, 5, 2, generator=generator, dtype=torch.float64) + 0.1
        self.V_ext = torch.randn(2, 5, 2, generator=generator, dtype=torch.float64)
        self.c1 = torch.randn(2, 5, 2, generator=generator, dtype=torch.float64)
        self.wavelength = torch.tensor([1.5, 0.5], dtype=torch.float64)
        self.beta = torch.tensor([0.8, 1.25], dtype=torch.float64)

    def test_local_chemical_potential_matches_formula(self):
        expected = (
            torch.log(self.rho * self.wavelength ** 3)
            + self.beta[:, None, None] * self.V_ext
            - self.c1
        )
        value = local_chemical_potential(
            self.rho, self.V_ext, self.beta, self.c1, self.wavelength,
        )
        torch.testing.assert_close(value, expected)
        target = local_chemical_potential(
            self.rho, self.V_ext, self.beta, None, self.wavelength,
        )
        torch.testing.assert_close(target, expected + self.c1)

    def test_scalar_beta_and_masks_zero_excluded_points(self):
        rho = self.rho.clone()
        rho[0, 2] = 0.0
        value = local_chemical_potential(
            rho, self.V_ext, 0.7, self.c1, self.wavelength,
        )
        self.assertTrue(torch.all(value[0, 2] == 0.0))
        self.assertTrue(torch.all(torch.isfinite(value)))
        mask = torch.ones(2, 5, dtype=torch.bool)
        mask[1, 4] = False
        masked = local_chemical_potential(
            rho, self.V_ext, 0.7, self.c1, self.wavelength, mask=mask[..., None],
        )
        self.assertTrue(torch.all(masked[1, 4] == 0.0))
        self.assertFalse(torch.all(masked[0, 2] == 0.0))
        with self.assertRaisesRegex(ValueError, "leading field shape"):
            local_chemical_potential(
                rho, self.V_ext, torch.ones(3), self.c1, self.wavelength,
            )
        with self.assertRaisesRegex(ValueError, "same shape"):
            local_chemical_potential(
                rho, self.V_ext[0], 0.7, self.c1, self.wavelength,
            )

    def test_equilibrium_log_density_solves_the_euler_lagrange_relation(self):
        mu = torch.tensor([0.3, -0.2], dtype=torch.float64)
        log_rho = equilibrium_log_density(
            self.V_ext, self.beta, mu, self.c1, self.wavelength,
        )
        rho = torch.exp(log_rho)
        # At that density the local chemical potential equals beta*mu.
        torch.testing.assert_close(
            local_chemical_potential(rho, self.V_ext, self.beta, self.c1, self.wavelength),
            (self.beta[:, None] * mu[None, :])[:, None, :].expand_as(rho),
        )
        with self.assertRaisesRegex(ValueError, "same shape"):
            equilibrium_log_density(self.V_ext, self.beta, mu, self.c1[0], self.wavelength)

    def test_beta_external_potential_inverts_local_chemical_potential(self):
        beta_mu = torch.tensor([[0.3, -0.2], [1.0, 0.4]], dtype=torch.float64)
        mu_local = local_chemical_potential(
            self.rho, self.V_ext, self.beta, self.c1, self.wavelength,
        )
        # At equilibrium beta*mu_loc equals beta*mu, so the inverse relation
        # must return the potential that produced the density.
        equilibrium_V_ext = self.V_ext + (
            beta_mu[:, None, :] - mu_local
        ) / self.beta[:, None, None]
        recovered = beta_external_potential(
            self.rho, self.c1, self.wavelength, beta_mu=beta_mu,
        )
        torch.testing.assert_close(
            recovered, self.beta[:, None, None] * equilibrium_V_ext,
        )
        per_type = beta_external_potential(
            self.rho, self.c1, self.wavelength, beta_mu=beta_mu[0],
        )
        torch.testing.assert_close(per_type[0], recovered[0])
        with self.assertRaisesRegex(ValueError, "beta_mu"):
            beta_external_potential(
                self.rho, self.c1, self.wavelength, beta_mu=torch.ones(3),
            )


if __name__ == "__main__":
    unittest.main()
