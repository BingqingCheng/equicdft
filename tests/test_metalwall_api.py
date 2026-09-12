"""Public metalwall API, independent of the oracle-case test adapter."""

import io
import unittest

import torch

from equicdft import GridCACEModel, LDAReadout, LongRangeReadout, MetalWall
from test_metalwall_sites import _data, _reference


def _model(*charges):
    readouts = [LDAReadout(mean_density=1., n_types=2, hidden_sizes=(), zero_init=True)]
    readouts += [LongRangeReadout(n_kernels=1, n_types=2, charges=value,
                                 coulomb_amplitude=None) for value in charges]
    return GridCACEModel(None, None, readouts, grid_spacing=1.).double()


def _wall(**options):
    parameters = dict(liquid_charges=[1., -1.], metal_sigma=.35, liquid_sigma=.12,
                      coulomb_amplitude=1.7, boundary="periodic")
    parameters.update(options)
    return MetalWall(**parameters).double()


class TestMetalWallAPI(unittest.TestCase):
    def test_explicit_constructor_copies_readout_charges(self):
        model = _model([2., -1.])
        wall = _wall(liquid_charges=model.readout[1].charges)
        self.assertEqual(wall.liquid_charges.tolist(), [2., -1.])
        self.assertEqual(wall.coulomb_amplitude, 1.7)
        self.assertNotIn(model, list(wall.modules()))
        self.assertFalse(hasattr(MetalWall, "from_model"))
        with torch.no_grad():
            model.readout[1].charges[0] = 3.
        self.assertEqual(wall.liquid_charges.tolist(), [2., -1.])

    def test_explicit_charges_are_general_not_model_inferred(self):
        self.assertIsNone(_model(None).readout[1].charges)
        self.assertEqual(_wall(liquid_charges=[2., -1., 0.]).liquid_charges.tolist(), [2., -1., 0.])

    def test_old_constructor_names_and_model_factory_are_not_supported(self):
        for kwargs in (dict(charges=[1., -1.]), dict(sigma=.35), dict(model=_model([1., -1.]))):
            with self.assertRaises(TypeError):
                _wall(**kwargs)

    def test_module_field_matches_independent_physical_energy_and_response(self):
        data = _data()
        field = torch.tensor([.12, -.05, .08], dtype=torch.float64)
        origin = torch.tensor([.1, -.2, .3], dtype=torch.float64)
        reference = _reference(dict(data, metal_external_field=field, metal_field_origin=origin))
        rho = data["rho"].clone().requires_grad_()
        result = _wall(external_field=field, field_origin=origin)(dict(data, rho=rho))
        for key in reference.keys() - {"B", "S"}:
            torch.testing.assert_close(result[key], reference[key], atol=3e-11, rtol=3e-11)
        energy = result["electrode_coulomb_energy"] + result["metal_external_energy"]
        gradient = torch.autograd.grad(energy, rho, create_graph=True)[0]
        volume = data["grid_spacing"].prod()
        charges = torch.tensor([1., -1.], dtype=torch.float64)
        expected = volume * (reference["B"].T @ reference["metal_site_q"])[:, None] * charges
        torch.testing.assert_close(gradient, expected, atol=3e-11, rtol=3e-11)
        direction = torch.sin(torch.arange(rho.numel(), dtype=rho.dtype)).reshape_as(rho)
        hv = torch.autograd.grad((gradient * direction).sum(), rho)[0]
        kernel = -reference["B"].T @ reference["S"] @ reference["B"]
        expected_hv = volume.square() * (kernel @ (direction @ charges))[:, None] * charges
        torch.testing.assert_close(hv, expected_hv, atol=3e-11, rtol=3e-11)

    def test_data_fields_are_rejected_instead_of_ignored_or_overriding(self):
        from test_metal_model import TestMetalModel

        data = _data()
        for key in ("metal_external_field", "metal_field_origin"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "on MetalWall, not in data"):
                _wall(external_field=[0., 0., .1])(dict(data, **{key: [0., 0., .2]}))
            model = TestMetalModel._model()
            with self.subTest(model_key=key), self.assertRaisesRegex(ValueError, "on MetalWall, not in data"):
                model(dict(TestMetalModel._data(), **{key: [0., 0., .2]}))

    def test_serialization_and_dtype_movement_retain_field_configuration(self):
        field, origin = [0.12, -.05, .08], [.1, -.2, .3]
        wall = _wall(external_field=field, field_origin=origin)
        expected = wall(_data())
        stream = io.BytesIO()
        torch.save(wall, stream)
        stream.seek(0)
        restored = torch.load(stream)
        for key, value in expected.items():
            torch.testing.assert_close(restored(_data())[key], value)
        self.assertIn("external_field", restored.state_dict())
        self.assertIn("field_origin", restored.state_dict())
        restored.float()
        self.assertEqual(restored.external_field.dtype, torch.float32)
        self.assertEqual(restored.field_origin.dtype, torch.float32)

    def test_shared_and_batched_constructor_fields(self):
        data = _data()
        fields = torch.tensor([[.1, 0., 0.], [-.1, .02, 0.]], dtype=torch.float64)
        wall = _wall(external_field=fields, field_origin=[.1, .2, .3])
        batch = dict(data, rho=data["rho"].expand(2, -1, -1))
        result = wall(batch)
        for index in range(2):
            expected = _wall(external_field=fields[index], field_origin=[.1, .2, .3])(data)
            for key, value in expected.items():
                torch.testing.assert_close(result[key][index], value)
        with self.assertRaisesRegex(ValueError, "batch shape"):
            wall(data)

    def test_constructor_field_validation_and_zero_default(self):
        for name in ("external_field", "field_origin"):
            for value in (0., [1., 2.], [float("nan"), 0., 0.], [True]*3,
                          [1j]*3, torch.zeros(3, requires_grad=True)):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    _wall(**{name: value})
        result = _wall()(_data())
        self.assertEqual(result["metal_external_energy"].item(), 0.)


if __name__ == "__main__":
    unittest.main()
