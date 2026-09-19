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
from typing import Callable, Dict, Tuple

import numpy as np
from tests.ideal_helpers import with_ideal, IdealOnly
from equicdft.solver import (
    AndersonOptions,
    Constraints,
    ConvergenceOptions,
    EulerOptions,
    Initialization,
)
from equicdft import FixedDipoleIdeal
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
# floating-point level; every other case must match exactly when the running
# torch version is the one that recorded the file.
TOLERANCES: Dict[str, Tuple[float, float]] = {}

# Under another torch release kernels differ at rounding level (FFT, matmul
# accumulation order), so every case is compared with this tolerance instead.
CROSS_VERSION_TOLERANCE: Dict[torch.dtype, Tuple[float, float]] = {
    torch.float32: (1.0e-5, 1.0e-6),
    torch.float64: (1.0e-6, 1.0e-9),
}


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
        # Deprecated alias keys duplicate canonical entries; record the
        # canonical ones only (and read without triggering their warnings).
        skip = getattr(value, "_aliases", {})
        for key, item in sorted(value.items()):
            if key in skip:
                continue
            _flatten("{}/{}".format(prefix, key), item, into)
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
    from equicdft.geometry.stencil import get_neighbor_indices

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
        boltzmann_constant=1.0,
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
                boltzmann_constant=1.0,
            )
            data = _polarization_data()
            return _outputs(model, data)

    return case


# --------------------------------------------------------------------- solver


class _Quadratic(IdealOnly):
    """beta*F_exc = a*dV*sum(rho^2) with c1 = -2*a*rho: a convex test functional."""

    def __init__(self, a=0.5):
        super().__init__()
        self.register_buffer("a", torch.tensor(float(a)))

    def forward(self, data, compute_c1=True, compute_c2=False,
                compute_polarization_derivative=False):
        rho = data["rho"]
        volume = torch.prod(data["grid_spacing"].to(rho))
        outputs = super().forward(data, compute_c1=compute_c1)
        outputs["beta_F_exc"] = self.a.to(rho) * volume * torch.sum(rho ** 2)
        outputs["beta_F"] = outputs["beta_F_id"] + outputs["beta_F_exc"]
        if compute_c1:
            outputs["c1"] = -2.0 * self.a.to(rho) * rho
            outputs["density_derivative"] = outputs["ideal_density_derivative"] - outputs["c1"]
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
                data,
                method="minimize",
                convergence=ConvergenceOptions(tolerance_residual=1e-9),
                constraints=Constraints(particle_numbers=[3.0], homogeneous_axes=("x", "y")),
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
                data,
                convergence=ConvergenceOptions(max_iter=60, tolerance_residual=1e-6),
                euler=EulerOptions(mixing=0.1),
                constraints=Constraints(particle_numbers=[20.0]),
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
            solver = GridSolver(with_ideal(None, FixedDipoleIdeal([0.7, 1.9])))
            return _solver_result(solver.solve(
                _ideal_dipoles_case(),
                method=method,
                **kwargs,
                convergence=ConvergenceOptions(max_iter=400, tolerance_residual=1e-6),
                constraints=Constraints(particle_numbers=[3.0, 4.0]),
            ))

    return case


def _model_dipoles(method, fixed_field=None):
    def case():
        with default_dtype(torch.float64):
            torch.manual_seed(24)
            model = _polarization_model().eval()
            data = _polarization_data()
            solver = GridSolver(with_ideal(model, FixedDipoleIdeal(0.7)))
            external = {
                key: data[key]
                for key in ("grid_size", "grid_spacing", "temperature", "local_density_index")
            }
            external.update(
                beta=torch.tensor(1.0 / 1.2),
                V_ext=torch.zeros_like(data["rho"]),
                E_ext=0.05 * torch.randn_like(data["dipole_density"]),
            )
            initial = None if fixed_field is None else Initialization(
                rho=data["rho"], polarization=data["dipole_density"],
            )
            return _solver_result(solver.solve(
                external,
                method=method,
                initial=initial,
                convergence=ConvergenceOptions(max_iter=200, tolerance_residual=1e-5),
                constraints=Constraints(
                    particle_numbers=data["rho"].sum(0) * 0.5 ** 3,
                    fixed_field=fixed_field,
                ),
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
        # Frozen historical trainer regression only: reproduce retired output
        # arithmetic in the fixture, not in the production model. New fitting
        # uses total derivatives and loss-side gauge projection instead.
        def legacy_observables(module, arguments, outputs):
            batch = arguments[0]
            local = (outputs["ideal_density_derivative"].detach()
                     + batch["beta"][..., None, None] * batch["V_ext"]
                     - outputs["c1"])
            weights = (batch["rho"].detach() > 1.e-3).to(local)
            outputs.update(beta_mu_local=local, beta_mu_weights=weights,
                           beta_mu_average=(weights * local).sum(-2) / weights.sum(-2))
        model.register_forward_hook(legacy_observables)
        loss = Loss([
            TensorLoss(
                name="mu",
                prediction_key="beta_mu_local",
                target_key="beta_mu_average",
                weights_key="beta_mu_weights",
            ),
        ])
        metrics = [Metrics(
            target_key="beta_mu_average",
            prediction_key="beta_mu_local",
            name="mu",
            selection_mask_key="beta_mu_weights",
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
    "solver/quadratic_euler_mu": _quadratic_solve(
        mu=-0.3, convergence=ConvergenceOptions(max_iter=300, tolerance_residual=1e-6),
    ),
    "solver/quadratic_euler_N_anderson_cap": _quadratic_solve(
        constraints=Constraints(particle_numbers=[4.0], maximum_density=[1.2]),
        euler=EulerOptions(anderson=AndersonOptions()),
        convergence=ConvergenceOptions(max_iter=300, tolerance_residual=1e-6),
    ),
    "solver/quadratic_minimize_N": _quadratic_solve(
        constraints=Constraints(particle_numbers=[6.0]), method="minimize",
        convergence=ConvergenceOptions(max_iter=100, tolerance_residual=1e-8),
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


def _rename_recorded(
    recorded: Dict[str, torch.Tensor],
    renames: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """Return ``recorded`` under current output names, dropping duplicates.

    A record that already carries the new name kept the two as aliases of
    one tensor, so the retired entry is discarded; otherwise it is renamed
    and its recorded value is still compared.
    """

    result = {}
    for key, value in recorded.items():
        prefix, _, name = key.rpartition("/")
        new_name = renames.get(name)
        if new_name is None:
            result[key] = value
            continue
        renamed = "{}/{}".format(prefix, new_name) if prefix else new_name
        if renamed in recorded:
            continue
        result[renamed] = value
    return result


def compare(
    recorded: Dict[str, torch.Tensor],
    actual: Dict[str, torch.Tensor],
    tolerance=None,
) -> None:
    """Raise AssertionError on the first difference between two case results.

    ``tolerance`` is ``None`` for an exact comparison, an ``(rtol, atol)``
    pair, or a mapping from dtype or output key to such pairs (a key takes
    precedence; integer and boolean tensors are always compared exactly).
    """

    actual = dict(actual)
    # Historical unsuffixed P responses were excess-only. Rename the reference
    # keys, not their tensors, so the same physical quantities are compared.
    # (``_rename_recorded`` below does the same for the retired residual
    # spellings; both keep the original recorded values in play.)
    recorded = {
        (key.rsplit("/", 1)[0] + "/polarization_derivative_excess"
         if key.endswith("/polarization_derivative") else key): value
        for key, value in recorded.items()
    }
    # The Euler-Lagrange spellings of the residuals, and the coupled
    # solver's ``iterations``, were retired in favour of one name each. Where a record carries both --
    # the fixed-dipole results always did -- they held the same tensor, so
    # the duplicate is dropped rather than renamed, and the recorded values
    # are still compared against what the solver produces today.
    recorded = _rename_recorded(recorded, {
        "euler_lagrange_residual": "density_residual",
        "max_euler_lagrange_residual": "max_density_residual",
        "rms_euler_lagrange_residual": "rms_density_residual",
        "full_euler_lagrange_residual": "full_density_residual",
        "full_max_euler_lagrange_residual": "full_max_density_residual",
        "full_rms_euler_lagrange_residual": "full_rms_density_residual",
        "polarization_euler_lagrange_residual": "polarization_residual",
        "iterations": "n_iter",
    })
    # Model training-policy outputs were retired; preserve all energy,
    # derivative, solver and historical trainer comparisons.
    retired = {"out/beta_mu_local", "out/beta_mu_average", "out/beta_mu_weights",
               "solve/beta_mu_average", "solve/beta_mu_weights"}
    recorded = {k: v for k, v in recorded.items()
                if k in actual or k not in retired}
    if "state/thermal_wavelength" in recorded and "state/ideal.thermal_wavelength" in actual:
        actual["state/thermal_wavelength"] = actual.pop("state/ideal.thermal_wavelength")
    added_outputs = {"beta_F_id", "beta_F", "ideal_derivative_mask",
                     "ideal_density_derivative", "ideal_polarization_derivative",
                     "density_derivative", "density_derivative_excess",
                     "polarization_derivative", "beta_mu_local"}
    for key in list(actual):
        if key not in recorded and key.rsplit("/", 1)[-1] in added_outputs:
            del actual[key]
    # Unified solver reporting adds aliases/metadata, not new physical values.
    # Their identities are tested in test_evaluation; keep every recorded
    # solver tensor and the original golden file/tolerances unchanged.
    added_solver_outputs = {
        "beta_A", "beta_external", "beta_V_ext", "beta_P_ext",
        "density_residual", "max_density_residual", "rms_density_residual",
        "grid_spacing", "thermal_wavelength", "grid_center", "particle_numbers",
    }
    for key in list(actual):
        if (key.startswith("solve/") and key not in recorded
                and key.rsplit("/", 1)[-1] in added_solver_outputs):
            del actual[key]
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
        pair = tolerance
        if isinstance(tolerance, dict):
            pair = tolerance.get(key, tolerance.get(expected.dtype))
        if pair is None or not expected.is_floating_point():
            if not torch.equal(expected, value):
                difference = (expected.double() - value.double()).abs().max().item()
                raise AssertionError(
                    "{}: not identical (max abs difference {:.3e})".format(key, difference)
                )
        else:
            torch.testing.assert_close(value, expected, rtol=pair[0], atol=pair[1], msg=key)


def compare_trainer_cross_version(recorded, actual) -> None:
    """Allow measured roundoff in the canonical trainer's unfixed gauge only.

    Float32 cancellation leaves final-bias gradients of order 1e-8, which
    Adam amplifies because its epsilon is also 1e-8. Across torch versions
    the two unconstrained biases differ by 4.60e-4, whereas centered c1
    differs by at most 1.13e-7 and the loss by 2.98e-8. Float64 reduces the
    bias drift to 1.97e-12. Other fitted parameters/losses stay tightly tested.
    """

    recorded, actual = dict(recorded), dict(actual)
    tolerance = dict(CROSS_VERSION_TOLERANCE)
    for key in recorded:
        if key in ("state/readout.0.mlp.2.bias", "state/readout.1.mlp.2.bias"):
            tolerance[key] = (0.0, 5.0e-4)
        elif "/train_metrics/mu/" in key and key.endswith(("/r2", "/rmse_percent")):
            # The inferred target changes gauge between optimizer steps, so
            # these ratios vary more than the unchanged canonical residual.
            tolerance[key] = (3.0e-2, 1.0e-6)
        elif "/valid_metrics/mu/" in key and key.endswith(("/r2", "/rmse_percent")):
            # The single validation field has constant target, hence sigma=0.
            # Roundoff can give either an enormous ratio or infinity. Compare
            # its well-conditioned inverse, without accepting a wrong sign.
            for values in (recorded, actual):
                value = values[key]
                if key.endswith("/r2"):
                    if torch.any(value >= 0):
                        raise AssertionError(key + ": expected negative R2 for this constant target")
                    values[key] = 1.0 / (1.0 - value)
                else:
                    if torch.any(value <= 0):
                        raise AssertionError(key + ": expected positive RMSE/sigma")
                    values[key] = 100.0 / value
            tolerance[key] = (0.0, 1.0e-7)
    compare(recorded, actual, tolerance)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "record":
        record()
    else:
        print("usage: python -m tests.regression.goldens record")
        sys.exit(2)
