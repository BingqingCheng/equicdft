"""Nonnegatively averaged inputs for the learned terms of a model."""

import copy
import itertools
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from equicdft import (
    CACEBranch, CartesianAFeatures, CartesianBFeatures, EnergyReadout, FixedDipoleIdeal,
    GGAReadout, GridCACEModel, GridSolver, IdealGas, LDAReadout, LocalReadout,
    LongRangeReadout, PairwiseReadout, ReciprocalFeatures, load_model, save_model,
)
from equicdft.geometry.spectral import nonnegative_average
from equicdft.response.bulk import bulk_response_symbol
from equicdft.solver import Constraints, ConvergenceOptions
from tests.representation import test_polarization_symmetry as symmetry

N, SPACING, SIGMA = 6, 0.7, 0.5
KEYS = ("beta_F_exc", "c1", "polarization_derivative_excess")


def learned_readouts(n_types=1, polarization=None, cutoff=2):
    """CACE branch, LDA and pairwise readouts; ``polarization`` is None, 'raw' or 'alignment'."""

    polarized = polarization is not None
    mode = polarization or "raw"
    a = CartesianAFeatures(
        max_power=1, mean_density=.5, n_types=n_types, cutoff_grid=cutoff,
        include_polarization=polarized, polarization_input=mode, separate_center=True,
        radial_basis="gaussian", radial_exponents=(.2, .6), convolution_backend="fft",
    )
    b = CartesianBFeatures(1, 2, include_polarization=polarized, separate_center=True)
    return [
        CACEBranch(a, b, LocalReadout(n_types=n_types, hidden_sizes=(5,))),
        LDAReadout(n_types=n_types, hidden_sizes=(5,), zero_init=False,
                   include_polarization=polarized, polarization_input=mode),
        PairwiseReadout(cutoff_grid=2, n_types=n_types, hidden_sizes=(4,), zero_init=False),
    ]


def fixed_readouts(n_types=1, polarized=False):
    charges = (.7,) if n_types == 1 else (-.8, .4)
    return [LongRangeReadout(
        n_types=n_types, n_kernels=1, charges=charges, include_polarization=polarized,
        coulomb_amplitude=1.8,
        features=ReciprocalFeatures(radial_exponents=(.15,), kernel="coulomb", n_types=n_types),
    )]


def build(readouts, sigma=None, n_types=1, polarization=None, ideal=None):
    polarized = polarization is not None or isinstance(ideal, FixedDipoleIdeal)
    if ideal is None:
        ideal = FixedDipoleIdeal(1.) if polarized else IdealGas()
    return GridCACEModel(
        readout=readouts, ideal=ideal, n_types=n_types, grid_spacing=SPACING,
        mean_density=.5, mean_temperature=1.5, boltzmann_constant=1., rho_min=.3,
        polarization_scale=.3 if polarization == "raw" else None,
        alignment_scale=.2 if polarization == "alignment" else None,
        alignment_density_floor=.01 if polarization == "alignment" else None,
        compute_polarization_derivative=polarized, learned_input_smoothing=sigma,
    ).eval()


def grid(n=N, n_types=1, polarized=False, uniform=False, fields=None):
    positions = torch.tensor(np.indices((n,) * 3).reshape(3, -1).T)
    leading = () if fields is None else (fields,)
    shape = (*leading, n ** 3, n_types)
    rho = torch.full(shape, .5) if uniform else .1 + .6 * torch.rand(shape)
    data = dict(
        rho=rho, temperature=torch.full(leading, 1.2) if leading else torch.tensor(1.2),
        beta=torch.full(leading, 1 / 1.2) if leading else torch.tensor(1 / 1.2),
        grid_size=torch.tensor([n] * 3) if not leading else torch.tensor([[n] * 3] * fields),
        grid_positions=positions if not leading else positions.expand(fields, -1, -1),
        grid_spacing=torch.full((3,), SPACING) if not leading else torch.full((fields, 3), SPACING),
    )
    if polarized:
        data["dipole_density"] = (torch.zeros(*shape, 3) if uniform
                                  else .1 * torch.randn(*shape, 3) * rho[..., None])
    return data


def smoothed(data, sigma=SIGMA):
    out = dict(data)
    out["rho"] = nonnegative_average(data["rho"], data["grid_size"], SPACING, sigma,
                                     grid_positions=data["grid_positions"])
    if "dipole_density" in data:
        dipole = data["dipole_density"]
        out["dipole_density"] = nonnegative_average(
            dipole.flatten(-2), data["grid_size"], SPACING, sigma,
            grid_positions=data["grid_positions"],
        ).unflatten(-1, dipole.shape[-2:])
    return out


def transfer(k, sigma=SIGMA, spacing=SPACING):
    """Documented transfer of the average on a cubic grid: steps of h^2 / 6, then the rest."""

    widest = spacing ** 2 / 6
    steps = max(1, math.ceil(sigma ** 2 / widest - 1e-12))
    moments = [widest] * (steps - 1) + [sigma ** 2 - (steps - 1) * widest]
    total = torch.sin(k * spacing / 2).square().sum(-1) / spacing ** 2
    return math.prod(1 - 2 * moment * total for moment in moments)


class AlignmentProbe(EnergyReadout):
    """A learned-input readout that reports the alignment field it is given."""

    reads_smoothed_fields = True
    requires_dipole_density = True
    requires_alignment_scale = True

    def energy_and_outputs(self, context):
        alignment = context.reduced_polarization()
        energy = context.integrate_local(alignment.square().sum(-1))
        return energy, {"probe_rho": context.rho, "probe_alignment": alignment}


def evaluate(model, data, **kwargs):
    inputs = {key: value.detach().clone() if torch.is_tensor(value) else value
              for key, value in data.items()}
    outputs = model(inputs, **kwargs)
    return {key: value.detach() for key, value in outputs.items() if torch.is_tensor(value)}


class TestLearnedInputSmoothing(unittest.TestCase):
    def setUp(self):
        self.dtype, self.rng = torch.get_default_dtype(), torch.get_rng_state()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(2026)

    def tearDown(self):
        torch.set_default_dtype(self.dtype)
        torch.set_rng_state(self.rng)

    def assert_decomposition(self, learned, fixed, polarization=None, n_types=1, **kwargs):
        """F[rho, P] = F_learned[G rho, G P] + F_fixed[rho, P], derivatives included."""

        polarized = polarization is not None
        full = build(learned + fixed, SIGMA, n_types, polarization)
        learned_only = build(copy.deepcopy(learned), None, n_types, polarization)
        fixed_only = build(copy.deepcopy(fixed), None, n_types, polarization) if fixed else None
        data = grid(n_types=n_types, polarized=polarized, **kwargs)
        actual = evaluate(full, data, compute_ideal=False)
        part = evaluate(learned_only, smoothed(data), compute_ideal=False)
        rest = (evaluate(fixed_only, data, compute_ideal=False) if fixed_only is not None
                else {key: torch.zeros_like(value) for key, value in part.items()})
        size, positions = data["grid_size"], data["grid_positions"]
        for key in KEYS:
            if key not in actual:
                continue
            expected = part[key]
            if key != "beta_F_exc":
                # The average is its own adjoint: derivatives come back through G.
                flat = expected.flatten(-2) if expected.ndim == data["rho"].ndim + 1 else expected
                flat = nonnegative_average(flat, size, SPACING, SIGMA, grid_positions=positions)
                expected = flat.reshape(expected.shape)
            torch.testing.assert_close(actual[key], expected + rest[key], rtol=1e-10, atol=1e-11,
                                       msg=lambda message: key + ": " + message)
        return full, data

    def test_learned_terms_see_averaged_fields_and_the_rest_the_voxel_fields(self):
        for n_types, polarization in itertools.product((1, 2), (None, "raw", "alignment")):
            with self.subTest(n_types=n_types, polarization=polarization):
                self.assert_decomposition(
                    learned_readouts(n_types, polarization),
                    fixed_readouts(n_types, polarization is not None),
                    polarization, n_types,
                )

    def test_long_range_sources_are_learned_inputs_only_when_learned(self):
        sources = LongRangeReadout(
            n_types=1, hidden_sizes=(4,), zero_init=False, density_channels=2,
            density_hidden_sizes=(4,),
            features=ReciprocalFeatures(radial_exponents=(.3, .8), kernel="gaussian", n_types=1),
        )
        kernel = LongRangeReadout(
            n_types=1, hidden_sizes=(4,), zero_init=False,
            features=ReciprocalFeatures(radial_exponents=(.3,), kernel="gaussian", n_types=1),
        )
        self.assertTrue(sources.reads_smoothed_fields)
        self.assertFalse(kernel.reads_smoothed_fields)
        self.assert_decomposition([sources], fixed_readouts() + [kernel])

    def test_batched_fields_and_the_ideal_reference(self):
        full, data = self.assert_decomposition(
            learned_readouts(1, "raw"), fixed_readouts(1, True), "raw", fields=2,
        )
        reference = build(copy.deepcopy(list(full.readout)), None, polarization="raw")
        reference.load_state_dict(full.state_dict())
        actual, plain = evaluate(full, data), evaluate(reference, data)
        for key in ("beta_F_id", "ideal_density_derivative", "ideal_polarization_derivative"):
            torch.testing.assert_close(actual[key], plain[key], rtol=0, atol=0)
        torch.testing.assert_close(actual["beta_F"], actual["beta_F_id"] + actual["beta_F_exc"])
        torch.testing.assert_close(
            actual["density_derivative"], actual["ideal_density_derivative"] - actual["c1"],
        )

    def test_shared_features_with_local_and_gradient_readouts(self):
        def model(sigma):
            a = CartesianAFeatures(
                max_power=2, mean_density=.5, cutoff_grid=2, separate_center=True,
                radial_basis="gaussian", radial_exponents=(.3,), convolution_backend="fft",
            )
            b = CartesianBFeatures(2, 2, separate_center=True)
            return GridCACEModel(
                a, b, [LocalReadout(hidden_sizes=(5,)), GGAReadout(hidden_sizes=(4,))],
                grid_spacing=SPACING, mean_temperature=1.5, boltzmann_constant=1.,
                learned_input_smoothing=sigma,
            ).eval()

        full = model(SIGMA)
        reference = model(None)
        reference.load_state_dict(full.state_dict())
        data = grid()
        actual = evaluate(full, data, compute_ideal=False)
        expected = evaluate(reference, smoothed(data), compute_ideal=False)
        torch.testing.assert_close(actual["beta_F_exc"], expected["beta_F_exc"], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            actual["c1"],
            nonnegative_average(expected["c1"], data["grid_size"], SPACING, SIGMA),
            rtol=1e-10, atol=1e-11,
        )

    def test_learned_curvature_is_multiplied_by_the_squared_transfer(self):
        indices = torch.tensor([[[1, 0, 0], [2, 1, 0], [3, 0, 0], [3, 3, 3], [-1, 3, 2]]])
        k = 2 * math.pi * indices[0].double() / (N * SPACING)
        attenuation = transfer(k).square()[:, None, None]
        self.assertLess(float(attenuation[3]), 1e-28)                # the zone corner
        self.assertGreater(float(attenuation[0]), 0.5)
        for polarization in (None, "raw", "alignment"):
            with self.subTest(polarization=polarization):
                learned = learned_readouts(1, polarization)
                fixed = fixed_readouts(1, True)
                ideal = FixedDipoleIdeal(1.)
                full = build(learned + fixed, SIGMA, polarization=polarization, ideal=ideal)
                plain = build(copy.deepcopy(learned) + copy.deepcopy(fixed), None,
                              polarization=polarization, ideal=FixedDipoleIdeal(1.))
                plain.load_state_dict(full.state_dict())
                fixed_only = build(copy.deepcopy(fixed), None, ideal=FixedDipoleIdeal(1.))
                batch = grid(polarized=True, uniform=True, fields=1)
                symbol = {name: bulk_response_symbol(model, batch, indices)[0]
                          for name, model in (("full", full), ("plain", plain), ("fixed", fixed_only))}
                learned_part = symbol["plain"] - symbol["fixed"]
                self.assertGreater(float(learned_part.detach().abs().max()), 1e-2)
                torch.testing.assert_close(
                    symbol["full"] - symbol["fixed"], attenuation * learned_part,
                    rtol=1e-9, atol=1e-11,
                )

    def test_alignment_inputs_stay_physical_across_a_dense_dilute_interface(self):
        # The review's interface: 0.09 and 0.001 A^-3 halves of a 16-voxel,
        # 2 A periodic box with alignment 0.8 along +x and -x, m = 0.48937 e A,
        # floor 1e-5 A^-3, sigma = 0.6 A. Averaged by the zone-restricted
        # Gaussian, the learned inputs reached alignment 14.4 two voxels into
        # the dilute half (tests/geometry/test_spectral.py keeps that case).
        m, floor, h, shape = .48937, 1e-5, 2., (16, 3, 3)
        a = CartesianAFeatures(
            max_power=1, mean_density=.03, cutoff_grid=1, include_polarization=True,
            polarization_input="alignment", separate_center=True,
            radial_basis="gaussian", radial_exponents=(.2, .6), convolution_backend="fft",
        )
        b = CartesianBFeatures(1, 2, include_polarization=True, separate_center=True)
        model = GridCACEModel(
            readout=[
                CACEBranch(a, b, LocalReadout(hidden_sizes=(5,))),
                LDAReadout(hidden_sizes=(5,), zero_init=False, include_polarization=True,
                           polarization_input="alignment"),
                AlignmentProbe(),
            ],
            ideal=FixedDipoleIdeal(m), grid_spacing=h, mean_density=.03, mean_temperature=1.5,
            boltzmann_constant=1., alignment_scale=.5, alignment_density_floor=floor,
            compute_polarization_derivative=True, learned_input_smoothing=.6,
        ).eval()
        positions = torch.tensor(np.indices(shape).reshape(3, -1).T)
        x = positions[:, 0]
        dense = x < 8
        rho = torch.where(dense, .09, .001)[:, None]
        dipole = torch.zeros(len(x), 1, 3)
        dipole[:, 0, 0] = .8 * m * rho[:, 0] * torch.where(dense, 1., -1.)
        data = dict(
            rho=rho, dipole_density=dipole, temperature=torch.tensor(1.2),
            beta=torch.tensor(1 / 1.2), grid_size=torch.tensor(shape),
            grid_positions=positions, grid_spacing=torch.full((3,), h),
        )
        output = evaluate(model, data)
        for key in ("beta_F_exc", "c1", "polarization_derivative_excess"):
            self.assertTrue(bool(torch.isfinite(output[key]).all()), key)
        # The learned readouts see densities inside the input range and
        # alignments inside 0.8; two voxels into the dilute half the averaged
        # fluid is the dilute fluid itself.
        self.assertGreaterEqual(float(output["probe_rho"].min()), .001 * (1 - 1e-14))
        self.assertLessEqual(float(output["probe_rho"].max()), .09 * (1 + 1e-14))
        alignment = output["probe_alignment"].norm(dim=-1)
        self.assertLessEqual(float(alignment.max()), .8 * (1 + 1e-12))
        torch.testing.assert_close(output["probe_rho"][x == 9], torch.full((9, 1), .001))
        torch.testing.assert_close(alignment[x == 9], torch.full((9, 1), .8 / (1 + 1e-4)))

        # Functional derivatives at the interface agree with the energy.
        def energy(name, index, step):
            changed = {key: value.clone() if torch.is_tensor(value) else value
                       for key, value in data.items()}
            changed[name][index] += step
            return float(model(changed, compute_c1=False, compute_ideal=False,
                               compute_polarization_derivative=False)["beta_F_exc"].detach())

        for voxel in (6, 7, 8, 9):
            index = 9 * voxel                                     # (voxel, 0, 0)
            for name, component, response, sign in (
                ("rho", (index, 0), "c1", -1),
                ("dipole_density", (index, 0, 0), "polarization_derivative_excess", 1),
            ):
                step = 1e-4 * float(rho[index, 0])
                difference = (energy(name, component, step) - energy(name, component, -step)) / (2 * step)
                expected = sign * h ** 3 * float(output[response][component])
                self.assertAlmostEqual(difference, expected, delta=1e-7 * (1 + abs(expected)),
                                       msg="{} {} at x = {}".format(name, component, voxel))

    def test_fitting_selection_and_diagnostics_use_the_voxel_fields(self):
        learned, fixed = learned_readouts(), fixed_readouts()
        full = build(learned + fixed, SIGMA)
        plain = build(copy.deepcopy(learned) + copy.deepcopy(fixed), None)
        plain.load_state_dict(full.state_dict())
        data = dict(grid(), V_ext=.3 * torch.randn(N ** 3, 1))
        self.assertTrue(bool((data["rho"] < full.rho_min).any()))
        masked = evaluate(full, data, use_voxel_mask=True, compute_local_mu=True)
        reference = evaluate(plain, data, use_voxel_mask=True, compute_local_mu=True)
        for key in ("voxel_weights", "local_energy_mask"):
            torch.testing.assert_close(masked[key], reference[key], rtol=0, atol=0)
        active = masked["voxel_weights"] > 0
        beta = 1.0 / float(data["temperature"])
        torch.testing.assert_close(
            masked["beta_mu_local"],
            torch.where(active, masked["density_derivative"] + beta * data["V_ext"], 0.0),
        )
        # Selected learned centres integrate the averaged density, with the
        # selection itself taken from the voxel density.
        learned_only = build(copy.deepcopy(learned), None)
        fixed_only = build(copy.deepcopy(fixed), None)
        selected = dict(smoothed(data), voxel_weights=masked["voxel_weights"])
        expected = (evaluate(learned_only, selected, use_voxel_mask=True, compute_ideal=False)["beta_F_exc"]
                    + evaluate(fixed_only, data, compute_ideal=False)["beta_F_exc"])
        torch.testing.assert_close(masked["beta_F_exc"], expected, rtol=1e-12, atol=1e-12)

    def test_cubic_symmetry_with_polarization(self):
        data = symmetry.TestFullPolarizationSymmetry.data()
        data.pop("local_density_index")
        positions = data["grid_positions"].numpy()
        actions = [
            np.eye(3, dtype=int)[list(permutation)] * np.array(signs)[:, None]
            for permutation in itertools.permutations(range(3))
            for signs in itertools.product((1, -1), repeat=3)
        ]
        for alignment in (False, True):
            with self.subTest(alignment=alignment):
                base = symmetry.TestFullPolarizationSymmetry.model("fft", .7, alignment=alignment)
                config = base.to_config()
                config["learned_input_smoothing"] = .4
                model = GridCACEModel.from_config(config).eval()
                model.load_state_dict(base.state_dict())
                reference = evaluate(model, data)
                unsmoothed = evaluate(base, data)
                self.assertGreater(
                    float((reference["beta_F_exc"] - unsmoothed["beta_F_exc"]).abs()), 1e-6,
                )
                for rotation in actions:
                    centers = ((2 * positions + 1) @ rotation.T) % 8
                    destination = torch.tensor(np.ravel_multi_index(((centers - 1) // 2).T, (4, 4, 4)))
                    matrix = torch.tensor(rotation, dtype=torch.float64)
                    transformed = dict(data)
                    transformed["rho"] = torch.empty_like(data["rho"])
                    transformed["rho"][destination] = data["rho"]
                    transformed["dipole_density"] = torch.empty_like(data["dipole_density"])
                    transformed["dipole_density"][destination] = data["dipole_density"] @ matrix.T
                    actual = evaluate(model, transformed)
                    for key in ("beta_F", "c1", "polarization_derivative"):
                        value, expected = actual[key], reference[key]
                        if key != "beta_F":
                            value = value[destination]
                        if key == "polarization_derivative":
                            expected = expected @ matrix.T
                        torch.testing.assert_close(value, expected, atol=1e-11, rtol=1e-10)

    def test_configuration_files_and_validation(self):
        model = build(learned_readouts() + fixed_readouts(), SIGMA)
        config = model.to_config()
        self.assertEqual(config["learned_input_smoothing"], SIGMA)
        self.assertEqual(GridCACEModel.from_config(config).learned_input_smoothing, SIGMA)
        config.pop("learned_input_smoothing")          # written before the option existed
        self.assertIsNone(GridCACEModel.from_config(config).learned_input_smoothing)
        self.assertIsNone(build(learned_readouts()).to_config()["learned_input_smoothing"])
        self.assertFalse(any("smoothing" in key for key in model.state_dict()))
        data = grid()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_model(model, path)
            restored = load_model(path)
        self.assertEqual(restored.learned_input_smoothing, SIGMA)
        torch.testing.assert_close(evaluate(restored, data)["c1"], evaluate(model, data)["c1"],
                                   rtol=0, atol=0)

        for sigma in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "learned_input_smoothing"):
                build(learned_readouts(), sigma)
        for sigma in (True, "0.5", torch.tensor(.5)):
            with self.assertRaises(TypeError):
                build(learned_readouts(), sigma)
        with self.assertRaisesRegex(ValueError, "learned local readout"):
            build(fixed_readouts(), SIGMA)
        missing = dict(data)
        missing.pop("grid_size")
        with self.assertRaisesRegex(ValueError, "grid_size"):
            build(learned_readouts()[1:2], SIGMA)(missing)

    def test_solver_reaches_the_euler_lagrange_point(self):
        model = build([LDAReadout(hidden_sizes=(4,), zero_init=False)], SIGMA)
        n = 4
        x = (torch.arange(n, dtype=torch.float64) + .5) * SPACING
        potential = .2 * torch.cos(2 * math.pi * x / (n * SPACING))[:, None, None].expand(n, n, n)
        data = grid(n=n, uniform=True)
        data["V_ext"] = potential.reshape(-1, 1).clone()
        total = float(data["rho"].sum() * SPACING ** 3)
        result = GridSolver(model).solve(
            data, convergence=ConvergenceOptions(max_iter=300),
            constraints=Constraints(particle_numbers=[total]),
        )
        self.assertTrue(result["converged"])
        check = evaluate(model, dict(data, rho=result["rho"]), compute_local_mu=True)
        mu = check["beta_mu_local"]
        self.assertLess(float((mu - mu.mean()).abs().max()), 1e-2)


if __name__ == "__main__":
    unittest.main()
