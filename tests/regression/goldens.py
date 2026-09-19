"""Golden outputs that pin the package's numbers across restructurings.

``python -m tests.regression.goldens record`` evaluates every case below and
writes ``tests/regression/goldens.pt``; ``tests/regression/test_goldens.py``
re-evaluates the cases and compares. The cases use only the public
``equicdft`` API and the shared test fixtures, so they survive module moves
unchanged. Each case sets its own seeds and default dtype and returns a flat
mapping of names to CPU tensors.

Comparison is exact (``torch.equal``) unless a case declares a tolerance in
``TOLERANCES``; a documented numerical change must update the recorded file
in the same commit (``record`` again) and say why in the commit message.
"""

import copy
import itertools
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from equicdft import (
    CartesianAFeatures,
    CartesianBFeatures,
    GridCACEModel,
    GridSolver,
    LDAReadout,
    LocalReadout,
    LongRangeReadout,
    Loss,
    MetalWall,
    Metrics,
    ReciprocalFeatures,
    TensorLoss,
    Trainer,
)

from tests.metal_helpers import liquid_coulomb, metal_sites
from tests.model_fixtures import (
    bessel_message_model,
    example_model,
    gaussian_message_model,
    grid_data,
    long_range_model,
)


GOLDEN_PATH = Path(__file__).with_name("goldens.pt")

# name -> (rtol, atol) for cases whose numbers may legitimately move at the
# floating-point level; every other case must match exactly.
TOLERANCES: Dict[str, Tuple[float, float]] = {}


@contextmanager
def default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _flatten(prefix: str, value, into: Dict[str, torch.Tensor]) -> None:
    if torch.is_tensor(value):
        into[prefix] = value.detach().cpu().clone()
    elif isinstance(value, dict):
        for key in sorted(value):
            _flatten("{}/{}".format(prefix, key), value[key], into)
    elif isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (bool, int, float)) for item in value):
            into[prefix] = torch.tensor(value, dtype=torch.float64)
        else:
            for index, item in enumerate(value):
                _flatten("{}[{}]".format(prefix, index), item, into)
    elif isinstance(value, (bool, int, float)):
        into[prefix] = torch.tensor(value)
    # Strings and None are recorded implicitly through the key set only.


def _outputs(model: nn.Module, data, **kwargs) -> Dict[str, torch.Tensor]:
    model.eval()
    result = {}
    _flatten("out", model(copy.deepcopy(data), **kwargs), result)
    return result


# --------------------------------------------------------------------- models


def _example_lda_local():
    torch.manual_seed(11)
    return _outputs(example_model(), grid_data(cutoff_grid=1))


def _example_joint_center():
    torch.manual_seed(12)
    return _outputs(example_model(separate_center=False), grid_data(cutoff_grid=1))


def _example_two_types_physical():
    torch.manual_seed(13)
    model = example_model(n_types=2, free_energy_mode="physical", compute_c2=True)
    return _outputs(model, grid_data(cutoff_grid=1, n_types=2, seed=3))


def _gaussian_message(radial_basis, backend):
    def case():
        torch.manual_seed(14)
        model = gaussian_message_model(radial_basis=radial_basis, backend=backend)
        return _outputs(model, grid_data(cutoff_grid=1, n_types=2, seed=4, grid_spacing=0.5))

    return case


def _bessel_message():
    torch.manual_seed(15)
    return _outputs(bessel_message_model(), grid_data(cutoff_grid=2, seed=5))


def _long_range(charges, amplitude):
    def case():
        torch.manual_seed(16)
        model = long_range_model(charges=charges, coulomb_amplitude=amplitude)
        data = grid_data(shape=(4, 4, 4), cutoff_grid=2, n_types=2, seed=6, grid_spacing=0.5)
        return _outputs(model, data)

    return case


def _metal_data():
    mask = torch.full((16,), -1, dtype=torch.long)
    mask[:2], mask[-2:] = 0, 3
    rho = torch.zeros((16, 2), dtype=torch.float64)
    modulation = torch.linspace(-0.03, 0.03, 12, dtype=torch.float64)
    rho[2:14, 0] = 0.2 + modulation
    rho[2:14, 1] = 0.2 - modulation
    positions = torch.tensor(
        list(itertools.product(range(4), range(2), range(2))), dtype=torch.float64,
    )
    return {
        "rho": rho,
        "grid_size": torch.tensor([4, 2, 2]),
        "grid_spacing": torch.full((3,), 0.75, dtype=torch.float64),
        "temperature": torch.tensor(2.0, dtype=torch.float64),
        "beta": torch.tensor(1.0 / 3.0, dtype=torch.float64),
        "excluded_mask": mask >= 0,
        "metal_positions": positions[mask >= 0] * 0.75,
        "metal_site_groups": mask[mask >= 0],
        "metal_group_ids": torch.tensor([0, 3]),
        "metal_total_charge": torch.zeros(2, dtype=torch.float64),
        "metal_charge_units": "e",
    }


def _metal(mode):
    def case():
        torch.manual_seed(17)
        data = _metal_data()
        with default_dtype(torch.float64):
            wall = MetalWall(
                metal_sites=metal_sites(data),
                liquid_coulomb=liquid_coulomb(amplitude=0.2 / 3.0),
                metal_sigma=0.6,
            )
            model = GridCACEModel(
                a_features=None,
                b_features=None,
                readout=[LDAReadout(n_types=2, hidden_sizes=(3,)), wall],
                grid_spacing=0.75,
                mean_temperature=2.0,
                boltzmann_constant=1.5,
                free_energy_mode=mode,
                mean_density=1.0,
                compute_c2=True,
            )
        liquid = {
            key: value for key, value in data.items()
            if not key.startswith("metal_")
        }
        return _outputs(model, liquid)

    return case


def _polarization_data(batched=False):
    positions = np.indices((3, 3, 3)).reshape(3, -1).T
    from equicdft.stencil import get_neighbor_indices  # noqa: E402 (moves in step 5)

    indices, _ = get_neighbor_indices(positions, cutoff_grid=1)
    shape = (2, 27, 1) if batched else (27, 1)
    return {
        "rho": 0.4 + 0.2 * torch.rand(shape),
        "dipole_density": 0.05 * torch.randn(*shape, 3),
        "temperature": torch.tensor([1.2, 1.7] if batched else 1.2),
        "grid_spacing": torch.full((3,), 0.5),
        "grid_size": torch.tensor([3, 3, 3]),
        "local_density_index": (
            torch.tensor(indices).expand(2, -1, -1).clone()
            if batched else torch.tensor(indices)
        ),
    }


def _polarization_model(mode="beta", with_scalar=False):
    a_features = CartesianAFeatures(
        mean_density=0.5, dipole_density_scale=0.3, include_polarization=True,
        cutoff_grid=1, max_power=2, radial_basis="gaussian",
        radial_exponents=(0.125, 0.5), trainable_radial_exponents=True,
    )
    b_features = CartesianBFeatures(2, 3, include_polarization=True)
    readouts = [LocalReadout(hidden_sizes=(5,))]
    if with_scalar:
        readouts.append(LDAReadout(hidden_sizes=(4,)))
    return GridCACEModel(
        a_features=a_features,
        b_features=b_features,
        readout=readouts,
        grid_spacing=0.5,
        mean_temperature=1.5,
        free_energy_mode=mode,
        compute_polarization_derivative=True,
    )


def _polarization(mode, with_scalar, batched):
    def case():
        with default_dtype(torch.float64):
            torch.manual_seed(18)
            model = _polarization_model(mode, with_scalar)
            data = _polarization_data(batched)
            return _outputs(model, data)

    return case


def _polar_coulomb():
    """Dipole-density Coulomb response through ReciprocalFeatures."""

    def case():
        with default_dtype(torch.float64):
            torch.manual_seed(19)
            response = LongRangeReadout(
                features=ReciprocalFeatures(
                    (0.1, 0.8), variable="dipole_density", include_divergence=True,
                ),
                hidden_sizes=(3,),
            )
            model = GridCACEModel(
                None, None, [LDAReadout(hidden_sizes=(2,)), response],
                grid_spacing=0.5, mean_density=0.4,
                compute_polarization_derivative=True,
            )
            data = _polarization_data()
            return _outputs(model, data)

    return case


# --------------------------------------------------------------------- solver


class _Quadratic(nn.Module):
    """beta*F_exc = a*dV*sum(rho^2) with c1 = -2*a*rho: a convex test functional."""

    def __init__(self, a=0.5):
        super().__init__()
        self.register_buffer("a", torch.tensor(float(a)))

    def forward(self, data, compute_c1=True):
        rho = data["rho"]
        volume = torch.prod(data["grid_spacing"].to(rho))
        outputs = {"beta_F_exc": self.a.to(rho) * volume * torch.sum(rho ** 2)}
        if compute_c1:
            outputs["c1"] = -2.0 * self.a.to(rho) * rho
        return outputs


def _quadratic_data(n_types=1):
    generator = torch.Generator().manual_seed(21)
    shape = (4, 3, 3)
    n_grid = int(np.prod(shape))
    V_ext = 0.5 * torch.rand(n_grid, n_types, generator=generator, dtype=torch.float64)
    positions = np.indices(shape, dtype=int).reshape(3, -1).T
    return {
        "V_ext": V_ext,
        "grid_size": torch.tensor(shape),
        "grid_positions": torch.tensor(positions, dtype=torch.long),
        "grid_spacing": torch.full((3,), 0.5, dtype=torch.float64),
        "temperature": torch.tensor(1.25, dtype=torch.float64),
        "beta": torch.tensor(0.8, dtype=torch.float64),
        "thermal_wavelength": torch.full((n_types,), 0.9, dtype=torch.float64),
    }


def _solver_result(result) -> Dict[str, torch.Tensor]:
    flat = {}
    _flatten("solve", result, flat)
    return flat


def _quadratic_solve(**kwargs):
    def case():
        with default_dtype(torch.float64):
            solver = GridSolver(_Quadratic())
            data = _quadratic_data()
            if "mu" in kwargs:
                data["mu"] = torch.tensor([kwargs.pop("mu")], dtype=torch.float64)
            return _solver_result(solver.solve(data, **kwargs))

    return case


def _quadratic_homogeneous():
    def case():
        with default_dtype(torch.float64):
            solver = GridSolver(_Quadratic())
            data = _quadratic_data()
            # Homogeneity along x and y requires the field to share it.
            V_ext = data["V_ext"].reshape(4, 3, 3, 1).mean(dim=(0, 1), keepdim=True)
            data["V_ext"] = V_ext.expand(4, 3, 3, 1).reshape(-1, 1).clone()
            return _solver_result(solver.solve(
                data, particle_numbers=[3.0], method="minimize",
                homogeneous_axes=("x", "y"), tolerance_residual=1e-9,
            ))

    return case


def _lda_local_solve():
    """A learned (untrained) GridCACEModel drives the Euler solver at fixed N."""

    def case():
        with default_dtype(torch.float64):
            torch.manual_seed(22)
            model = example_model()
            data = grid_data(shape=(4, 4, 4), cutoff_grid=1, seed=7)
            data = {
                key: value.double() if value.is_floating_point() else value
                for key, value in data.items()
            }
            data["V_ext"] = 0.2 * data["V_ext"]
            data.pop("rho")
            solver = GridSolver(model)
            return _solver_result(solver.solve(
                data, particle_numbers=[20.0], max_iter=60,
                tolerance_residual=1e-6, mixing=0.1,
            ))

    return case


def _ideal_dipoles_case(amplitude=1.0, spacing=0.5):
    xyz = torch.cartesian_prod(torch.arange(4), torch.arange(3), torch.arange(2))
    phase = 2 * torch.pi * (xyz + 0.5) / torch.tensor([4, 3, 2])
    potential = torch.stack((0.7 * phase[:, 0].cos(), -0.4 * phase[:, 1].sin()), dim=-1)
    field = amplitude * (phase.cos() + torch.tensor([0.1, 0.3, -0.2]))
    return {
        "V_ext": potential,
        "E_ext": torch.stack((field, -0.6 * field), dim=1),
        "beta": torch.tensor(1.4),
        "grid_spacing": torch.full((3,), spacing),
        "thermal_wavelength": torch.tensor([0.8, 1.3]),
    }


def _ideal_dipoles(method, **kwargs):
    def case():
        with default_dtype(torch.float64):
            torch.manual_seed(23)
            solver = GridSolver(None, dipole_magnitude=[0.7, 1.9])
            return _solver_result(solver.solve(
                _ideal_dipoles_case(), particle_numbers=[3.0, 4.0],
                method=method, max_iter=400, tolerance_residual=1e-6, **kwargs,
            ))

    return case


def _model_dipoles(method, fixed_field=None):
    def case():
        with default_dtype(torch.float64):
            torch.manual_seed(24)
            model = _polarization_model().eval()
            data = _polarization_data()
            solver = GridSolver(model, dipole_magnitude=0.7)
            external = {
                key: data[key]
                for key in ("grid_size", "grid_spacing", "temperature", "local_density_index")
            }
            external.update(
                beta=torch.tensor(1.0 / 1.2),
                V_ext=torch.zeros_like(data["rho"]),
                E_ext=0.05 * torch.randn_like(data["dipole_density"]),
            )
            kwargs = {}
            if fixed_field is not None:
                kwargs = dict(
                    fixed_field=fixed_field,
                    initial_rho=data["rho"],
                    initial_polarization=data["dipole_density"],
                )
            return _solver_result(solver.solve(
                external, particle_numbers=data["rho"].sum(0) * 0.5 ** 3, method=method,
                max_iter=200, tolerance_residual=1e-5, **kwargs,
            ))

    return case


# -------------------------------------------------------------------- trainer


def _trainer_epoch():
    def case():
        torch.manual_seed(25)
        frames = []
        for seed in range(4):
            frame = grid_data(shape=(4, 4, 4), cutoff_grid=1, seed=30 + seed)
            frame["V_ext"] = 0.3 * frame["V_ext"]
            frames.append(frame)
        model = example_model()
        loss = Loss([
            TensorLoss(
                name="mu",
                prediction_key="local_chemical_potential",
                target_key="average_chemical_potential",
                weights_key="chemical_potential_weights",
            ),
        ])
        metrics = [Metrics(
            target_key="average_chemical_potential",
            prediction_key="local_chemical_potential",
            name="mu",
            selection_mask_key="chemical_potential_weights",
        )]
        trainer = Trainer(
            model=model, loss=loss, metrics=metrics,
            optimizer_args={"lr": 1.0e-3},
            scheduler_cls=torch.optim.lr_scheduler.StepLR,
            scheduler_args={"step_size": 1, "gamma": 0.5},
        )
        train = DataLoader(
            frames[:3], batch_size=1, shuffle=True,
            generator=torch.Generator().manual_seed(31),
        )
        valid = DataLoader(frames[3:], batch_size=1)
        history = trainer.fit(train, valid, epochs=2, verbose=False)
        result = {}
        _flatten("history", history, result)
        for key, value in model.state_dict().items():
            result["state/" + key] = value.detach().cpu().clone()
        return result

    return case


CASES: Dict[str, Callable[[], Dict[str, torch.Tensor]]] = {
    "model/example_lda_local": _example_lda_local,
    "model/example_joint_center": _example_joint_center,
    "model/example_two_types_physical_c2": _example_two_types_physical,
    "model/message_gaussian_gather": _gaussian_message("gaussian", "gather"),
    "model/message_gaussian_fft": _gaussian_message("gaussian", "fft"),
    "model/message_shared_gather": _gaussian_message("shared", "gather"),
    "model/message_bessel_gga": _bessel_message,
    "model/long_range_learned": _long_range(None, None),
    "model/long_range_coulomb": _long_range((1.0, -1.0), 0.5),
    "model/metal_beta": _metal("beta"),
    "model/metal_physical": _metal("physical"),
    "model/polarization_beta": _polarization("beta", False, False),
    "model/polarization_physical_scalar_batched": _polarization("physical", True, True),
    "model/polarization_coulomb_response": _polar_coulomb(),
    "solver/quadratic_euler_mu": _quadratic_solve(mu=-0.3, max_iter=300, tolerance_residual=1e-6),
    "solver/quadratic_euler_N_anderson_cap": _quadratic_solve(
        particle_numbers=[4.0], anderson=True, maximum_density=[1.2],
        max_iter=300, tolerance_residual=1e-6,
    ),
    "solver/quadratic_minimize_N": _quadratic_solve(
        particle_numbers=[6.0], method="minimize", max_iter=100, tolerance_residual=1e-8,
    ),
    "solver/quadratic_minimize_homogeneous": _quadratic_homogeneous(),
    "solver/lda_local_euler_N": _lda_local_solve(),
    "solver/ideal_dipoles_minimize": _ideal_dipoles("minimize"),
    "solver/ideal_dipoles_euler": _ideal_dipoles("euler"),
    "solver/model_dipoles_minimize": _model_dipoles("minimize"),
    "solver/model_dipoles_euler": _model_dipoles("euler"),
    "solver/model_dipoles_fixed_rho": _model_dipoles("minimize", fixed_field="rho"),
    "solver/model_dipoles_fixed_dipoles": _model_dipoles("minimize", fixed_field="dipole_density"),
    "trainer/two_epochs_lda_local": _trainer_epoch(),
}


def evaluate(names=None) -> Dict[str, Dict[str, torch.Tensor]]:
    return {name: CASES[name]() for name in (names or CASES)}


def record(path: Path = GOLDEN_PATH) -> None:
    results = evaluate()
    torch.save(
        {"torch_version": str(torch.__version__), "cases": results}, str(path),
    )
    total = sum(len(values) for values in results.values())
    print("recorded {} cases ({} tensors) to {}".format(len(results), total, path))


def compare(
    recorded: Dict[str, torch.Tensor],
    actual: Dict[str, torch.Tensor],
    tolerance: Optional[Tuple[float, float]] = None,
) -> None:
    """Raise AssertionError on the first difference between two case results."""

    if set(recorded) != set(actual):
        raise AssertionError(
            "key sets differ: missing {} unexpected {}".format(
                sorted(set(recorded) - set(actual)),
                sorted(set(actual) - set(recorded)),
            )
        )
    for key in sorted(recorded):
        expected, value = recorded[key], actual[key]
        if expected.dtype != value.dtype or expected.shape != value.shape:
            raise AssertionError(
                "{}: dtype/shape {} {} vs {} {}".format(
                    key, expected.dtype, tuple(expected.shape), value.dtype, tuple(value.shape),
                )
            )
        if tolerance is None:
            if not torch.equal(expected, value):
                difference = (expected.double() - value.double()).abs().max().item()
                raise AssertionError(
                    "{}: not identical (max abs difference {:.3e})".format(key, difference)
                )
        else:
            torch.testing.assert_close(value, expected, rtol=tolerance[0], atol=tolerance[1], msg=key)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "record":
        record()
    else:
        print("usage: python -m tests.regression.goldens record")
        sys.exit(2)
