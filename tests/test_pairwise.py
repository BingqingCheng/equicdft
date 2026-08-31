import unittest

import torch

from equicdft import GridCACEModel, PairwiseReadout
from equicdft.pairwise import _smooth_cutoff_envelope


def _context(rho, shape, temperature=1.1, voxel_volume=0.7):
    leading_shape = rho.shape[:-2]
    normalized_temperature = torch.as_tensor(
        temperature,
        dtype=rho.dtype,
        device=rho.device,
    ).expand(leading_shape)
    return {
        "rho": rho,
        "normalized_temperature": normalized_temperature,
        "voxel_volume": torch.as_tensor(
            voxel_volume,
            dtype=rho.dtype,
            device=rho.device,
        ),
        "grid_size": torch.tensor(shape, device=rho.device),
    }


def _set_linear_kernel(readout):
    with torch.no_grad():
        readout.mlp[0].weight.copy_(
            torch.tensor(
                [[0.2, 0.1], [0.3, -0.1], [0.4, 0.2]],
                dtype=readout.mlp[0].weight.dtype,
            )
        )
        readout.mlp[0].bias.copy_(
            torch.tensor(
                [0.05, -0.02, 0.03],
                dtype=readout.mlp[0].bias.dtype,
            )
        )


def _brute_force_energy(readout, context):
    rho = context["rho"]
    nx, ny, nz = context["grid_size"].tolist()
    rho_grid = rho.reshape(nx, ny, nz, readout.n_types)
    shell_values = readout.shell_kernel_values(
        context["normalized_temperature"]
    )
    offset_values = shell_values.index_select(
        -2,
        readout.offset_shell_index,
    )
    energy = rho.new_zeros(())
    for offset_index, offset in enumerate(readout.offsets.tolist()):
        shifted = torch.roll(
            rho_grid,
            shifts=tuple(-value for value in offset),
            dims=(0, 1, 2),
        )
        for pair_index, (first, second) in enumerate(readout.type_pairs):
            contraction = torch.sum(
                rho_grid[..., first] * shifted[..., second]
            )
            if first != second:
                contraction = contraction + torch.sum(
                    rho_grid[..., second] * shifted[..., first]
                )
            energy = (
                energy
                + offset_values[offset_index, pair_index] * contraction
            )
    return 0.5 * context["voxel_volume"].square() * energy


class TestSmoothCutoffEnvelope(unittest.TestCase):
    def test_value_and_two_derivatives_vanish_at_cutoff(self):
        x = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

        value = _smooth_cutoff_envelope(x)
        first = torch.autograd.grad(value, x, create_graph=True)[0]
        second = torch.autograd.grad(first, x)[0]

        self.assertEqual(value.item(), 0.0)
        self.assertEqual(first.item(), 0.0)
        self.assertEqual(second.item(), 0.0)

    def test_envelope_is_positive_inside_and_zero_outside(self):
        x = torch.tensor([-0.1, 0.0, 0.5, 1.0, 1.1])
        values = _smooth_cutoff_envelope(x)

        self.assertEqual(values[0].item(), 0.0)
        self.assertEqual(values[1].item(), 1.0)
        self.assertGreater(values[2].item(), 0.0)
        self.assertEqual(values[3].item(), 0.0)
        self.assertEqual(values[4].item(), 0.0)


class TestPairwiseReadout(unittest.TestCase):
    def test_constructor_and_arguments(self):
        readout = PairwiseReadout(
            cutoff_grid=2,
            n_types=2,
            hidden_sizes=(),
        )

        self.assertEqual(readout.n_type_pairs, 3)
        self.assertEqual(readout.type_pairs, ((0, 0), (0, 1), (1, 1)))
        self.assertEqual(readout.n_offsets, 26)
        self.assertEqual(readout.n_shells, 3)
        self.assertFalse(torch.any(torch.all(readout.offsets == 0, dim=1)))
        self.assertTrue(
            torch.all(
                readout.offsets.square().sum(dim=1)
                < readout.cutoff_grid**2
            )
        )

        with self.assertRaises(ValueError):
            PairwiseReadout(cutoff_grid=1)
        with self.assertRaises(TypeError):
            PairwiseReadout(cutoff_grid=True)
        with self.assertRaises(TypeError):
            PairwiseReadout(cutoff_grid=2, zero_init=1)

    def test_zero_initialization_is_exact_and_has_no_self_kernel(self):
        readout = PairwiseReadout(cutoff_grid=2, n_types=2)
        rho = torch.rand(5**3, 2, requires_grad=True)
        context = _context(rho, (5, 5, 5))

        kernel = readout.real_space_kernel(
            context["normalized_temperature"],
            context["grid_size"],
        )
        energy = readout.energy(context)
        gradient = torch.autograd.grad(
            energy,
            rho,
            create_graph=True,
        )[0]

        self.assertTrue(torch.equal(kernel, torch.zeros_like(kernel)))
        self.assertTrue(torch.equal(kernel[:, 0, 0, 0], torch.zeros(3)))
        self.assertEqual(energy.item(), 0.0)
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_fft_matches_brute_force_energy_and_gradient(self):
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            readout = PairwiseReadout(
                cutoff_grid=2,
                n_types=2,
                hidden_sizes=(),
                zero_init=False,
            )
            _set_linear_kernel(readout)
            rho = torch.rand(5**3, 2, requires_grad=True)
            context = _context(rho, (5, 5, 5))

            fft_energy = readout.energy(context)
            brute_energy = _brute_force_energy(readout, context)
            fft_gradient = torch.autograd.grad(
                fft_energy,
                rho,
                retain_graph=True,
            )[0]
            brute_gradient = torch.autograd.grad(brute_energy, rho)[0]

            self.assertTrue(
                torch.allclose(
                    fft_energy,
                    brute_energy,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )
            self.assertTrue(
                torch.allclose(
                    fft_gradient,
                    brute_gradient,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )
        finally:
            torch.set_default_dtype(original_dtype)

    def test_uniform_density_retains_nonzero_mode(self):
        readout = PairwiseReadout(
            cutoff_grid=2,
            n_types=2,
            hidden_sizes=(),
            zero_init=False,
        )
        _set_linear_kernel(readout)
        rho = torch.ones(5**3, 2)
        context = _context(rho, (5, 5, 5))

        energy = readout.energy(context)

        self.assertNotEqual(energy.item(), 0.0)
        self.assertTrue(torch.isfinite(energy))

    def test_translation_invariance(self):
        readout = PairwiseReadout(
            cutoff_grid=2,
            n_types=2,
            hidden_sizes=(),
            zero_init=False,
        )
        _set_linear_kernel(readout)
        rho = torch.rand(5, 5, 5, 2)
        shifted = torch.roll(rho, shifts=(1, -2, 2), dims=(0, 1, 2))

        reference = readout.energy(
            _context(rho.reshape(-1, 2), (5, 5, 5))
        )
        translated = readout.energy(
            _context(shifted.reshape(-1, 2), (5, 5, 5))
        )

        self.assertTrue(torch.allclose(reference, translated))

    def test_batched_fields_match_separate_evaluation(self):
        readout = PairwiseReadout(
            cutoff_grid=2,
            n_types=2,
            hidden_sizes=(),
            zero_init=False,
        )
        _set_linear_kernel(readout)
        rho = torch.rand(2, 5**3, 2)
        temperatures = torch.tensor([0.9, 1.2])
        batch_context = _context(
            rho,
            (5, 5, 5),
            temperature=temperatures,
        )

        batched = readout.energy(batch_context)
        separate = torch.stack(
            [
                readout.energy(
                    _context(
                        rho[index],
                        (5, 5, 5),
                        temperature=temperatures[index],
                    )
                )
                for index in range(2)
            ]
        )

        self.assertTrue(torch.allclose(batched, separate))

    def test_grid_and_temperature_are_validated(self):
        readout = PairwiseReadout(cutoff_grid=3, n_types=2)
        rho = torch.rand(5**3, 2)
        context = _context(rho, (5, 5, 5))
        with self.assertRaisesRegex(ValueError, "half"):
            readout.energy(context)

        readout = PairwiseReadout(cutoff_grid=2, n_types=2)
        context = _context(rho, (5, 5, 5))
        context["normalized_temperature"] = torch.tensor([1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "temperature"):
            readout.energy(context)

        context = _context(rho, (5, 5, 5))
        del context["grid_size"]
        with self.assertRaises(KeyError):
            readout.energy(context)

    def test_model_c2_is_negative_kernel_and_zero_at_self(self):
        readout = PairwiseReadout(
            cutoff_grid=2,
            n_types=2,
            hidden_sizes=(),
            zero_init=False,
        )
        _set_linear_kernel(readout)
        model = GridCACEModel(
            a_features=None,
            b_features=None,
            readout=[readout],
            grid_spacing=0.7 ** (1.0 / 3.0),
            mean_temperature=1.0,
            boltzmann_constant=1.0,
            compute_c1=True,
            compute_c2=True,
        )
        rho = torch.rand(5**3, 2)
        data = {
            "rho": rho,
            "temperature": torch.tensor(1.1),
            "grid_size": torch.tensor([5, 5, 5]),
        }

        output = model(data, c2_reference=(0, 0))
        kernel = readout.real_space_kernel(
            torch.tensor(1.1),
            data["grid_size"],
        )
        neighbor = 5**2

        self.assertAlmostEqual(output["c2"][0, 0].item(), 0.0, delta=1.0e-7)
        self.assertTrue(
            torch.allclose(output["c2"][neighbor, 0], -kernel[0, 1, 0, 0])
        )
        self.assertTrue(
            torch.allclose(output["c2"][neighbor, 1], -kernel[1, 1, 0, 0])
        )

    def test_kernel_parameters_receive_finite_gradients(self):
        readout = PairwiseReadout(
            cutoff_grid=2,
            n_types=2,
            hidden_sizes=(4,),
            zero_init=False,
        )
        rho = torch.rand(5**3, 2, requires_grad=True)

        readout.energy(_context(rho, (5, 5, 5))).backward()

        for parameter in readout.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.all(torch.isfinite(parameter.grad)))


if __name__ == "__main__":
    unittest.main()
