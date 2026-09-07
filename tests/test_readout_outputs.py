"""Optional readout observables preserve the scalar-energy interface."""

import unittest

import torch

from equicdft import GridCACEModel
from equicdft.energy import EnergyReadout


class _QuadraticReadout(EnergyReadout):
    n_types = 1

    def energy(self, context):
        return context["rho"].square().sum(dim=(-2, -1))


class _ObservedReadout(_QuadraticReadout):
    def __init__(self, key="density_observable"):
        super().__init__()
        self.key = key

    def energy_and_outputs(self, context):
        return self.energy(context), {self.key: context["rho"] * 3.0}


class TestReadoutOutputs(unittest.TestCase):
    @staticmethod
    def _model(readouts):
        return GridCACEModel(None, None, readouts, grid_spacing=1.0).double()

    @staticmethod
    def _data():
        return {
            "rho": torch.tensor([[0.2], [0.3]], dtype=torch.float64),
            "temperature": torch.tensor(1.0, dtype=torch.float64),
        }

    def test_scalar_only_readout_retains_default_interface(self):
        readout = _QuadraticReadout()
        data = self._data()
        energy, extra = readout.energy_and_outputs(data)
        torch.testing.assert_close(energy, readout.energy(data))
        self.assertEqual(extra, {})
        result = self._model([readout])(data)
        self.assertEqual(set(result), {"beta_F_exc", "c1"})

    def test_observable_does_not_change_energy_or_derivatives(self):
        data = self._data()
        expected = self._model([_QuadraticReadout()])(data, compute_c2=True)
        actual = self._model([_ObservedReadout()])(data, compute_c2=True)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        torch.testing.assert_close(actual["density_observable"], 3 * data["rho"])
        gradient = torch.autograd.grad(actual["density_observable"].sum(), data["rho"])[0]
        torch.testing.assert_close(gradient, torch.full_like(data["rho"], 3))

    def test_duplicate_readout_output_names_raise(self):
        model = self._model([_ObservedReadout(), _ObservedReadout()])
        with self.assertRaisesRegex(ValueError, "unique.*density_observable"):
            model(self._data())

    def test_reserved_names_raise_even_when_response_is_disabled(self):
        for key in ("beta_F_exc", "F_exc", "c1", "c2", "local_chemical_potential",
                    "average_chemical_potential", "chemical_potential_weights"):
            with self.subTest(key=key):
                model = self._model([_ObservedReadout(key)])
                with self.assertRaisesRegex(ValueError, "reserved"):
                    model(self._data(), compute_c1=False)


if __name__ == "__main__":
    unittest.main()
