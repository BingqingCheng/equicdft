"""Independent Fourier and variational checks for dipolar liquids and metals."""

import itertools
import math
import unittest

import torch

from equicdft import GridCACEModel, LongRangeReadout, MetalWall, ReciprocalFeatures


def _liquid(charges, alpha=.23, amplitude=1.7, polar=True):
    return LongRangeReadout(
        n_kernels=1, n_types=len(charges), charges=charges,
        coulomb_amplitude=amplitude, include_polarization=polar,
        features=ReciprocalFeatures(
            kernel="coulomb", radial_exponents=(alpha,), n_types=len(charges),
        ),
    )


def _case(shape=(3, 5, 3), neutral_species=False, spacing=(.7, .9, 1.1)):
    spacing = torch.tensor(spacing)
    origin = torch.tensor([.13, -.21, .17])
    grid = torch.tensor(list(itertools.product(*(range(n) for n in shape)))) * spacing
    n_species = 1 if neutral_species else 3
    fluctuation = .025 * torch.randn(len(grid), n_species)
    fluctuation -= fluctuation.mean(0)
    # 1.3*.35 - .7*.65 = 0; all perturbations have zero species means.
    density = torch.tensor([.4] if neutral_species else [.35, .65, .3])
    data = dict(
        rho=density + fluctuation,
        dipole_density=.018 * torch.randn(len(grid), n_species, 3),
        grid_size=torch.tensor(shape), grid_spacing=spacing,
        grid_center=grid + origin, temperature=torch.tensor(1.),
        beta=torch.tensor(1.),
    )
    fractions = torch.tensor([
        [.08, .13, .17], [.27, .67, .74], [.62, .31, .43], [.88, .82, .91],
    ])
    sites = dict(
        metal_positions=origin + fractions * torch.tensor(shape) * spacing,
        metal_site_groups=torch.tensor([2, 2, 7, 7]),
        metal_group_ids=torch.tensor([2, 7]),
        metal_total_charge=torch.zeros(2), metal_charge_units="e",
    )
    return data, sites, (0.,) if neutral_species else (1.3, -.7, 0.)


def _copy(data):
    return {key: value.detach().clone() for key, value in data.items()}


def _oracle(data, sites, charges, alpha=.23, amplitude=1.7, sigma=.35):
    """Direct positive-phase Fourier sum and dense KKT solve, without FFTs.

    Even-axis endpoints have half weights at both Nyquist signs. The real
    spectral derivative is zero in each axis's Nyquist mode. Integrated voxel
    monopoles are rho*z*dV and dipoles are P*dV, with no extra moment factor.
    """
    shape = tuple(data["grid_size"].tolist())
    spacing = data["grid_spacing"]
    cell = torch.tensor(shape) * spacing
    modes = list(itertools.product(*(
        range(-(n // 2), n // 2 + 1) for n in shape
    )))
    integers = torch.tensor(modes)
    weights = torch.ones(len(modes))
    derivative_modes = integers.clone()
    for axis, n in enumerate(shape):
        if n % 2 == 0:
            endpoint = integers[:, axis].abs() == n // 2
            weights[endpoint] *= .5
            derivative_modes[endpoint, axis] = 0
    wavevectors = 2 * math.pi * integers / cell
    derivative_wavevectors = 2 * math.pi * derivative_modes / cell
    k2 = wavevectors.square().sum(-1)
    keep = k2 > 0
    k, kd, k2, weights = (
        item[keep] for item in (wavevectors, derivative_wavevectors, k2, weights)
    )
    grid, positions = data["grid_center"], sites["metal_positions"]
    charge_density = data["rho"] @ torch.tensor(charges)
    source = spacing.prod() * (
        (charge_density[:, None] + 1j * (data["dipole_density"].sum(-2) @ kd.T))
        * torch.exp(1j * (grid @ k.T))
    ).sum(0)
    coefficient = amplitude * 4 * math.pi * weights / (cell.prod() * k2)
    energy = .5 * (coefficient * torch.exp(-alpha * k2) * source.abs().square()).sum()
    potential = (
        torch.exp(-1j * (positions @ k.T)) * source
        * coefficient * torch.exp(-.5 * sigma**2 * k2)
    ).real.sum(-1)
    displacement = positions[:, None, :] - positions[None, :, :]
    a = (torch.cos(displacement @ k.T)
         * coefficient * torch.exp(-sigma**2 * k2)).sum(-1)
    c = (sites["metal_site_groups"][:, None]
         == sites["metal_group_ids"][None, :]).to(data["rho"])
    m, g = c.shape
    kkt = torch.cat((torch.cat((a, c), 1),
                     torch.cat((c.T, a.new_zeros(g, g)), 1)), 0)
    solution = torch.linalg.solve(kkt, torch.cat((-potential, sites["metal_total_charge"])))
    q = solution[:m]
    cross, metal = q @ potential, .5 * q @ a @ q
    return dict(
        energy=energy, potential=potential, metal_site_q=q,
        metal_potential=-solution[m:], coulomb_cross_energy=cross,
        coulomb_metal_energy=metal, total_energy=energy + cross + metal,
    )


class TestMetalWallPolarization(unittest.TestCase):
    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(314)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)

    def test_mixed_liquid_matches_independent_off_grid_fourier_and_kkt(self):
        for shape in ((3, 5, 3), (4, 3, 2)):
            with self.subTest(shape=shape):
                data, sites, charges = _case(shape)
                liquid = _liquid(charges)
                reference = _oracle(data, sites, charges)
                state = liquid.coulomb_at(data, sites["metal_positions"], target_sigma=.35)
                wall = MetalWall(liquid, sites, metal_sigma=.35)
                energy, result = wall.energy_and_outputs(data)
                torch.testing.assert_close(state.energy, reference["energy"], atol=2e-13, rtol=2e-12)
                torch.testing.assert_close(state.potential, reference["potential"], atol=2e-13, rtol=2e-12)
                torch.testing.assert_close(energy, reference["total_energy"], atol=2e-13, rtol=2e-12)
                for key in ("metal_site_q", "metal_potential", "coulomb_cross_energy", "coulomb_metal_energy"):
                    torch.testing.assert_close(result[key], reference[key], atol=2e-13, rtol=2e-12)
                self.assertLess(result["charge_residual"].item(), 1e-12)
                self.assertLess(result["potential_residual"].item(), 1e-12)
                self.assertLess(state.total_charge.abs().item(), 1e-12)

    def test_neutral_dipoles_induce_charges_and_reversing_p_reverses_response(self):
        data, sites, charges = _case(neutral_species=True)
        liquid = _liquid(charges)
        wall = MetalWall(liquid, sites, metal_sigma=.35)
        self.assertTrue(wall.requires_dipole_density)
        self.assertFalse(wall.requires_state_features)
        positive = wall(data)
        self.assertGreater(positive["metal_site_q"].abs().max().item(), 1e-4)
        self.assertEqual(liquid.coulomb_at(data, sites["metal_positions"]).total_charge.item(), 0.)
        negative = wall(dict(data, dipole_density=-data["dipole_density"]))
        torch.testing.assert_close(negative["metal_site_q"], -positive["metal_site_q"])
        torch.testing.assert_close(negative["electrode_coulomb_energy"], positive["electrode_coulomb_energy"])
        zero = wall(dict(data, dipole_density=torch.zeros_like(data["dipole_density"])))
        torch.testing.assert_close(zero["metal_site_q"], torch.zeros_like(zero["metal_site_q"]), atol=0., rtol=0.)
        # The neutral species' number density cannot create Coulomb energy.
        doubled_rho = dict(data, rho=2 * data["rho"])
        torch.testing.assert_close(wall.energy(doubled_rho), wall.energy(data), atol=0., rtol=0.)

    def test_complement_changes_site_potential_only_not_liquid_lr_energy(self):
        data, sites, charges = _case()
        states = []
        for alpha in (.08, .6):
            liquid = _liquid(charges, alpha=alpha)
            state = liquid.coulomb_at(data, sites["metal_positions"], target_sigma=.35)
            reference = _oracle(data, sites, charges, alpha=alpha)
            torch.testing.assert_close(state.energy, liquid.energy(data), atol=2e-13, rtol=2e-12)
            torch.testing.assert_close(state.energy, reference["energy"], atol=2e-13, rtol=2e-12)
            torch.testing.assert_close(state.potential, reference["potential"], atol=2e-13, rtol=2e-12)
            states.append(state)
        torch.testing.assert_close(states[0].potential, states[1].potential, atol=0., rtol=0.)
        self.assertGreater((states[0].energy - states[1].energy).abs().item(), 1e-3)

    def test_zero_p_preserves_charge_only_wall_energy_and_response(self):
        data, sites, charges = _case((4, 3, 2))
        data["dipole_density"].zero_()
        scalar = MetalWall(_liquid(charges, polar=False), sites, metal_sigma=.35)
        polar = MetalWall(_liquid(charges), sites, metal_sigma=.35)
        old_energy, old_output = scalar.energy_and_outputs(data)
        energy, output = polar.energy_and_outputs(data)
        torch.testing.assert_close(energy, old_energy, atol=1e-13, rtol=1e-13)
        for key in output:
            torch.testing.assert_close(output[key], old_output[key], atol=1e-13, rtol=1e-13)

    def test_batched_polarization_and_grid_origins_match_single_fields(self):
        first, sites, charges = _case(neutral_species=True)
        second = _copy(first)
        second["dipole_density"] *= -.6
        second["grid_center"] += torch.tensor([.11, .07, -.03])
        batch = dict(first)
        for key in ("rho", "dipole_density", "grid_center", "temperature", "beta"):
            batch[key] = torch.stack((first[key], second[key]))
        wall = MetalWall(_liquid(charges), sites, metal_sigma=.35)
        energy, output = wall.energy_and_outputs(batch)
        for index, data in enumerate((first, second)):
            single_energy, single_output = wall.energy_and_outputs(data)
            torch.testing.assert_close(energy[index], single_energy, atol=2e-13, rtol=2e-12)
            for key in output:
                torch.testing.assert_close(output[key][index], single_output[key], atol=2e-13, rtol=2e-12)

    def test_excluded_liquid_region_cannot_keep_a_dipolar_source(self):
        data, sites, charges = _case(neutral_species=True)
        excluded = torch.zeros(len(data["rho"]), dtype=torch.bool)
        excluded[3] = True
        data["excluded_mask"] = excluded
        data["rho"][excluded] = 0.
        wall = MetalWall(_liquid(charges), sites, metal_sigma=.35)
        with self.assertRaisesRegex(ValueError, "dipole_density.*zero.*excluded"):
            wall.energy(data)
        data["dipole_density"][excluded] = 0.
        self.assertTrue(torch.isfinite(wall.energy(data)).item())

    def test_charge_dipole_cross_sign_cancels_at_sites_and_in_energy(self):
        data, sites, _ = _case((7, 3, 3), neutral_species=True)
        k = 2 * math.pi / (7 * data["grid_spacing"][0])
        x = data["grid_center"][:, 0]
        rho_amplitude, valence = .04, 1.3
        data["rho"][:, 0] = .4 + rho_amplitude * torch.cos(k * x)
        data["dipole_density"].zero_()
        # -div(P) = -q*a*cos(k*x), cancelling the ionic density fluctuation.
        data["dipole_density"][:, 0, 0] = valence * rho_amplitude / k * torch.sin(k * x)
        liquid = _liquid((valence,))
        cancelled = liquid.coulomb_at(data, sites["metal_positions"], target_sigma=.35)
        self.assertLess(cancelled.energy.abs().item(), 1e-27)
        self.assertLess(cancelled.potential.abs().max().item(), 1e-13)
        ionic = liquid.coulomb_at(dict(data, dipole_density=torch.zeros_like(data["dipole_density"])), sites["metal_positions"], target_sigma=.35)
        added = liquid.coulomb_at(dict(data, dipole_density=-data["dipole_density"]), sites["metal_positions"], target_sigma=.35)
        torch.testing.assert_close(added.energy, 4 * ionic.energy, atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(added.potential, 2 * ionic.potential, atol=1e-13, rtol=1e-12)

    def test_origin_translation_and_nyquist_polarization_convention(self):
        data, sites, charges = _case((4, 3, 2), neutral_species=True)
        liquid = _liquid(charges)
        wall = MetalWall(liquid, sites, metal_sigma=.35)
        shift = torch.tensor([.31, -.47, .19])
        shifted = MetalWall(_liquid(charges), dict(sites, metal_positions=sites["metal_positions"] + shift), metal_sigma=.35)
        energy, output = wall.energy_and_outputs(data)
        moved_energy, moved_output = shifted.energy_and_outputs(dict(data, grid_center=data["grid_center"] + shift))
        torch.testing.assert_close(moved_energy, energy, atol=2e-13, rtol=2e-12)
        for key in output:
            torch.testing.assert_close(moved_output[key], output[key], atol=2e-13, rtol=2e-12)
        data["dipole_density"].zero_()
        checker = (-1.) ** torch.arange(4)
        data["dipole_density"][:, 0, 0] = .02 * checker[:, None, None].expand(4, 3, 2).reshape(-1)
        state = liquid.coulomb_at(data, sites["metal_positions"], target_sigma=.35)
        self.assertLess(state.energy.abs().item(), 1e-27)
        self.assertLess(state.potential.abs().max().item(), 1e-13)

    def test_variational_derivatives_and_model_propagation(self):
        data, sites, charges = _case((3, 3, 3), spacing=(.8, .8, .8))
        data["rho"].requires_grad_()
        data["dipole_density"].requires_grad_()
        wall = MetalWall(_liquid(charges), sites, metal_sigma=.35)
        energy = wall.energy(data)
        expected = _oracle(data, sites, charges)["total_energy"]
        fields = (data["rho"], data["dipole_density"])
        actual_gradients = torch.autograd.grad(energy, fields, create_graph=True)
        reference_gradients = torch.autograd.grad(expected, fields)
        for actual, reference in zip(actual_gradients, reference_gradients):
            torch.testing.assert_close(actual, reference, atol=3e-13, rtol=3e-12)
        # Mixed reciprocity includes electrode charge redistribution.
        left = torch.autograd.grad(actual_gradients[0][4, 1], fields[1], retain_graph=True)[0][5, 2, 0]
        right = torch.autograd.grad(actual_gradients[1][5, 2, 0], fields[0], retain_graph=True)[0][4, 1]
        self.assertGreater(left.abs().item(), 1e-8)
        torch.testing.assert_close(left, right, atol=2e-13, rtol=2e-12)
        model = GridCACEModel(None, None, [wall], grid_spacing=data["grid_spacing"], compute_polarization_derivative=True).eval()
        result = model(_copy(data))
        torch.testing.assert_close(result["beta_F_exc"], energy, atol=2e-13, rtol=2e-12)
        torch.testing.assert_close(result["c1"], -actual_gradients[0] / data["grid_spacing"].prod(), atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(result["polarization_derivative"], actual_gradients[1] / data["grid_spacing"].prod(), atol=2e-12, rtol=2e-12)
        for field, index in (("rho", (4, 1)), ("dipole_density", (5, 2, 0))):
            direction = torch.zeros_like(data[field])
            direction[index] = 1.
            if field == "rho":
                direction[7, 1] = -1.  # Remain inside fixed-charge neutrality.
            plus, minus = _copy(data), _copy(data)
            plus[field] += 1e-6 * direction
            minus[field] -= 1e-6 * direction
            difference = (wall.energy(plus) - wall.energy(minus)) / 2e-6
            gradient = actual_gradients[0 if field == "rho" else 1]
            torch.testing.assert_close(difference, (gradient * direction).sum(), atol=2e-10, rtol=2e-7)


if __name__ == "__main__":
    unittest.main()
