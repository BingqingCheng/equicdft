"""Public API checks for the Coulomb-readout/electrode composition."""

import io
import unittest

import torch

from equicdft import LongRangeReadout, MetalWall, ReciprocalFeatures
from metal_helpers import _run, liquid_coulomb, metal_sites
from test_metalwall_sites import _data, _reference


def _one_electrode(data):
    data = dict(data)
    data["metal_site_groups"] = torch.zeros_like(data["metal_site_groups"])
    data["metal_group_ids"] = torch.tensor([0])
    data["metal_total_charge"] = data["metal_total_charge"].sum().reshape(1)
    return data


def _wall(sites=None, **options):
    sites = _data() if sites is None else sites
    parameters = dict(
        liquid_coulomb=liquid_coulomb(),
        metal_sites=metal_sites(sites),
        metal_sigma=.35,
    )
    parameters.update(options)
    return MetalWall(**parameters).double()


class TestMetalWallAPI(unittest.TestCase):
    def test_wall_owns_the_supplied_liquid_coulomb_readout(self):
        liquid = liquid_coulomb(charges=(2., -1.), amplitude=1.7)
        wall = _wall(liquid_coulomb=liquid)
        self.assertIs(wall.liquid_coulomb, liquid)
        self.assertEqual(wall.liquid_coulomb.charges.tolist(), [2., -1.])
        self.assertEqual(wall.n_types, 2)

    def test_only_one_charge_factorized_coulomb_kernel_is_accepted(self):
        invalid = (
            object(),
            LongRangeReadout(
                n_kernels=1, n_types=2,
                features=ReciprocalFeatures(
                    kernel="coulomb", radial_exponents=(.2,), n_types=2,
                ),
            ),
            LongRangeReadout(
                n_kernels=1, n_types=2, charges=(1., -1.),
                features=ReciprocalFeatures(
                    kernel="gaussian", radial_exponents=(.2,), n_types=2,
                ),
            ),
        )
        for liquid in invalid:
            with self.subTest(liquid=type(liquid).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    _wall(liquid_coulomb=liquid)

    def test_obsolete_independent_liquid_electrostatics_arguments_are_absent(self):
        for name, value in (
            ("liquid_charges", [1., -1.]),
            ("coulomb_amplitude", 1.7),
            ("boundary", "periodic"),
            ("field_origin", [0., 0., 0.]),
        ):
            with self.subTest(name=name), self.assertRaises(TypeError):
                _wall(**{name: value})

    def test_external_field_matches_independent_electrode_response(self):
        data = _one_electrode(_data())
        field = torch.tensor([.12, -.05, .08], dtype=torch.float64)
        reference = _reference(dict(data, metal_external_field=field))
        result = _run(
            _wall(data, external_field=field),
            dict(data, metal_external_field=field),
        )
        for key in (
            "metal_positions", "metal_site_groups",
            "metal_group_ids", "metal_total_charge",
        ):
            self.assertNotIn(key, result)
        for key in reference.keys() - {
            "B", "S", "q_liquid", "liquid_charge_density",
        }:
            torch.testing.assert_close(
                result[key], reference[key], atol=3e-7, rtol=3e-7,
            )

    def test_geometry_and_field_configuration_are_serialized(self):
        data = _one_electrode(_data())
        field = [0.12, -.05, .08]
        wall = _wall(data, external_field=field)
        expected = _run(wall, data)
        stream = io.BytesIO()
        torch.save(wall, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        for key, value in expected.items():
            torch.testing.assert_close(_run(restored, data)[key], value)
        for name in (
            "metal_positions", "metal_site_groups", "metal_group_ids",
            "metal_total_charge", "external_field",
        ):
            self.assertIn(name, restored.state_dict())
        self.assertIn("liquid_coulomb.charges", restored.state_dict())

    def test_shared_and_batched_constructor_fields(self):
        data = _one_electrode(_data())
        fields = torch.tensor(
            [[.1, 0., 0.], [-.1, .02, 0.]], dtype=torch.float64,
        )
        wall = _wall(data, external_field=fields)
        batch = dict(data, rho=data["rho"].expand(2, -1, -1))
        result = _run(wall, batch)
        for index in range(2):
            expected = _run(_wall(data, external_field=fields[index]), data)
            for key, value in expected.items():
                torch.testing.assert_close(result[key][index], value)

    def test_constructor_field_validation_and_zero_default(self):
        for value in (
            0., [1., 2.], [float("nan"), 0., 0.], [True] * 3,
            [1j] * 3, torch.zeros(3, requires_grad=True),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "external_field"):
                    _wall(external_field=value)
        with self.assertRaisesRegex(ValueError, "exactly one electrode group"):
            _wall(external_field=[.1, 0., 0.])
        _wall(external_field=[0., 0., 0.])
        result = _run(_wall(), _data())
        self.assertEqual(result["metal_external_energy"].item(), 0.)

    def test_liquid_sigma_is_not_a_metalwall_option(self):
        with self.assertRaises(TypeError):
            _wall(liquid_sigma=.4)


if __name__ == "__main__":
    unittest.main()
