"""Conversion of whole-object ``torch.save(model)`` files.

Before :mod:`equicdft.serialization` existed, trained models were written as
pickled Python objects. Such a file pins the import path of every class and
the instance-attribute layout of the package version that wrote it, so each
later attribute needed a runtime fallback. This module confines all of that
history to one place:

* :class:`_RemappingUnpickler` translates historical module paths and class
  names to their current locations while the object graph is rebuilt;
* :func:`upgrade_legacy_model` fills in attributes and deterministic buffers
  that later package versions added, using the defaults those versions
  assumed, so the unpickled object satisfies the current class contract;
* :func:`legacy_model_to_current` describes the upgraded object with
  ``to_config()``, rebuilds a fresh model from that configuration, transfers
  the fitted state, and checks that both evaluate identically;
* :func:`convert_legacy_model` writes the result with
  :func:`equicdft.serialization.save_model`.

The runtime classes themselves carry no legacy fallbacks.
"""

import inspect
import pickle
import types
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from ._config import Configurable, build, mlp_hidden_sizes
from .data import GridData
from .features import CartesianAFeatures, _DensityMixing
from .interaction import BChiMessage
from .model import GridCACEModel
from .pairwise import PairwiseReadout
from .readout import BulkReadout, LocalReadout, LongRangeReadout
from .semilocal import GGAReadout, LDAReadout
from .serialization import default_dtype, model_dtype, save_model


PathLike = Union[str, Path]

# Historical import paths of pickled classes -> current import paths. Extend
# these tables when a module or class moves; whole-object files then keep
# loading without shims in the runtime modules.
_MODULE_ALIASES: Dict[str, str] = {}
_CLASS_ALIASES: Dict[Tuple[str, str], Tuple[str, str]] = {}


class _RemappingUnpickler(pickle.Unpickler):
    """Unpickler that resolves historical equicdft class references."""

    def find_class(self, module: str, name: str) -> Any:  # noqa: D401
        module, name = _CLASS_ALIASES.get((module, name), (module, name))
        module = _MODULE_ALIASES.get(module, module)
        return super().find_class(module, name)


def _remapping_pickle_module() -> types.ModuleType:
    """Return the ``pickle_module`` object ``torch.load`` expects."""

    module = types.ModuleType("equicdft.legacy._pickle")
    module.Unpickler = _RemappingUnpickler

    def load(file: Any, **arguments: Any) -> Any:
        return _RemappingUnpickler(file, **arguments).load()

    module.load = load
    return module


def load_legacy_model(
    path: PathLike,
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> nn.Module:
    """Unpickle a whole-object model file and upgrade it in place.

    The file is trusted: whole-object pickles execute arbitrary code while
    loading, which is one of the reasons the current format replaced them.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(
            "legacy model file does not exist: {}".format(source)
        )
    pickle_module = _remapping_pickle_module()
    try:
        model = torch.load(
            str(source),
            map_location=map_location,
            pickle_module=pickle_module,
            weights_only=False,
        )
    except TypeError:
        # PyTorch releases before the weights_only keyword.
        model = torch.load(
            str(source),
            map_location=map_location,
            pickle_module=pickle_module,
        )
    if not isinstance(model, nn.Module):
        raise TypeError(
            "{} does not contain a pickled model object".format(source)
        )
    return upgrade_legacy_model(model)


def _defines(module: nn.Module, name: str) -> bool:
    """Whether ``name`` exists as attribute, buffer, parameter, or child.

    ``nn.Module`` stores a ``None`` child or buffer differently from a real
    one (plain attribute versus registry entry), so presence must be checked
    in every place the constructor may have put it.
    """

    return (
        name in module.__dict__
        or name in module._modules
        or name in module._buffers
        or name in module._parameters
    )


def _set_default(module: nn.Module, name: str, value: Any) -> None:
    """Assign ``value`` through ``setattr`` unless the object defines it.

    ``setattr`` reproduces the constructor's own storage choice: modules
    register children, ``None`` becomes a plain attribute.
    """

    if not _defines(module, name):
        setattr(module, name, value)


def _has_buffer(module: nn.Module, name: str) -> bool:
    return module._buffers.get(name) is not None


def _constructor_default(cls: type, parameter: str) -> Any:
    return inspect.signature(cls.__init__).parameters[parameter].default


def _require(module: nn.Module, names: Iterable[str]) -> None:
    missing = [name for name in names if not _defines(module, name)]
    if missing:
        raise ValueError(
            "legacy {} object lacks {}; it predates every supported "
            "format".format(type(module).__name__, sorted(missing))
        )


def _upgrade_cartesian_a_features(module: CartesianAFeatures) -> None:
    _require(
        module,
        (
            "cutoff_grid",
            "max_power",
            "n_types",
            "mean_density",
            "squared_distances",
            "monomial_values",
            "powers",
            "local_density_positions",
        ),
    )
    _set_default(module, "convolution_backend", "gather")
    _set_default(module, "coordinate_scaling", "none")
    _set_default(module, "trainable_radial_exponents", False)
    _set_default(module, "trainable_radial_centers", False)
    _set_default(module, "separate_center", False)
    if "radial_basis" not in module.__dict__:
        fixed = module._buffers.get("fixed_radial_exponents")
        damped = fixed is not None and bool(torch.any(fixed != 0.0))
        gaussian = damped or "log_radial_exponents" in module._parameters
        module.radial_basis = "gaussian" if gaussian else "none"

    if not _has_buffer(module, "neighbor_mask"):
        center = module.squared_distances == 0
        module.register_buffer(
            "neighbor_mask",
            ~center if module.separate_center else torch.ones_like(center),
            persistent=False,
        )
    if (
        module.radial_basis == "none"
        and not _has_buffer(module, "fixed_radial_exponents")
    ):
        module.register_buffer(
            "fixed_radial_exponents",
            module.squared_distances.new_zeros(1),
        )
    if (
        module.radial_basis != "bessel"
        and not module.trainable_radial_centers
        and not _has_buffer(module, "fixed_radial_centers")
    ):
        # Zero-centered Gaussians (and the undamped basis) predate centers.
        module.register_buffer(
            "fixed_radial_centers",
            torch.zeros_like(module.radial_exponents),
            persistent=False,
        )
    _set_default(module, "radial_transform", None)

    if not _defines(module, "density_transform"):
        # density_transform was called channel_mixing in early versions.
        mixing = module._modules.pop("channel_mixing", None)
        if mixing is None:
            mixing = module.__dict__.pop("channel_mixing", None)
        if mixing is not None and not isinstance(mixing, _DensityMixing):
            raise TypeError(
                "legacy channel_mixing must be a density-mixing module"
            )
        module.density_transform = mixing
    transform = module.density_transform
    _set_default(
        module,
        "n_channels",
        None if transform is None else int(transform.n_channels),
    )
    _set_default(
        module,
        "trainable_density_transform",
        True if transform is None else bool(transform.weight.requires_grad),
    )
    _set_default(
        module,
        "n_output_channels",
        module.n_types if module.n_channels is None else module.n_channels,
    )
    if module.radial_basis == "bessel":
        _require(module, ("n_radial_functions", "fixed_bessel_stencil_basis"))
    else:
        _set_default(
            module,
            "n_radial_functions",
            int(module.radial_exponents.numel()),
        )
    _set_default(
        module,
        "n_radial_channels",
        (
            module.n_radial_functions
            if module.radial_transform is None
            else int(module.radial_transform.n_radial_channels)
        ),
    )


def _upgrade_message(module: BChiMessage) -> None:
    _require(
        module,
        ("n_invariant_features", "n_radial_channels", "n_channels", "mlp"),
    )
    _set_default(module, "convolution_backend", "gather")
    _set_default(module, "trainable_radial_exponents", False)
    _set_default(module, "trainable_radial_centers", False)
    if "radial_basis" not in module.__dict__:
        # Early message layers recorded only whether they owned a Gaussian
        # basis; Bessel layers always carried radial_basis explicitly.
        independent = module.__dict__.pop("independent_radial_basis", False)
        module.radial_basis = "gaussian" if independent else "shared"
    else:
        module.__dict__.pop("independent_radial_basis", None)
    if (
        module.radial_basis == "gaussian"
        and not module.trainable_radial_centers
        and not _has_buffer(module, "fixed_radial_centers")
    ):
        module.register_buffer(
            "fixed_radial_centers",
            torch.zeros_like(module.radial_exponents),
            persistent=False,
        )
    # Bessel binding slots exist on every current layer; unbound is None.
    _set_default(module, "radial_transform", None)
    for name in ("fixed_bessel_stencil_basis", "bessel_gram_eigenvalues"):
        if not _defines(module, name):
            module.register_buffer(name, None)
    _set_default(module, "_bessel_geometry_signature", None)
    _set_default(module, "hidden_sizes", tuple(mlp_hidden_sizes(module.mlp)))


def _upgrade_model(module: GridCACEModel) -> None:
    _require(
        module,
        ("readout", "grid_spacing", "mean_temperature", "boltzmann_constant"),
    )
    if not _defines(module, "message_layers"):
        module.message_layers = nn.ModuleList()
    _set_default(module, "a_features", None)
    _set_default(module, "b_features", None)
    _set_default(module, "free_energy_mode", "beta")
    _set_default(module, "compute_c1", True)
    _set_default(module, "compute_c2", False)
    _set_default(module, "compute_local_mu", False)
    _set_default(module, "rho_min", 0.0)
    if not _has_buffer(module, "thermal_wavelength"):
        module.register_buffer(
            "thermal_wavelength",
            torch.ones(
                module.n_types,
                dtype=module.grid_spacing.dtype,
                device=module.grid_spacing.device,
            ),
        )


def _upgrade_mlp_readout(module: nn.Module) -> None:
    mlp = module._modules.get("mlp")
    if mlp is not None:
        _set_default(module, "hidden_sizes", tuple(mlp_hidden_sizes(mlp)))
    else:
        _set_default(module, "hidden_sizes", ())
    if "zero_init" in inspect.signature(type(module).__init__).parameters:
        _set_default(
            module,
            "zero_init",
            _constructor_default(type(module), "zero_init"),
        )
    if isinstance(module, (LocalReadout, GGAReadout)):
        _set_default(module, "n_features", None)
    if isinstance(module, LongRangeReadout):
        for name in ("charges", "pair_charge_products", "coulomb_amplitude"):
            if not _defines(module, name):
                module.register_buffer(name, None)
        _set_default(module, "features", None)


_UPGRADES = (
    (CartesianAFeatures, _upgrade_cartesian_a_features),
    (BChiMessage, _upgrade_message),
    (GridCACEModel, _upgrade_model),
    (
        (
            LocalReadout,
            BulkReadout,
            LongRangeReadout,
            LDAReadout,
            GGAReadout,
            PairwiseReadout,
        ),
        _upgrade_mlp_readout,
    ),
)


def upgrade_legacy_model(model: nn.Module) -> nn.Module:
    """Fill attributes that later package versions added, in place.

    Every default reproduces what the package assumed for objects that
    lacked the attribute, so an upgraded object evaluates exactly as it did
    when it was saved. The object is returned for convenience.
    """

    # Upgrades may register submodules, so do not iterate the live tree.
    for module in list(model.modules()):
        for classes, upgrade in _UPGRADES:
            if isinstance(module, classes):
                upgrade(module)
    return model


def _attribute_by_path(root: nn.Module, path: str) -> Optional[torch.Tensor]:
    """Return the tensor stored at a dotted ``state_dict`` key, if any."""

    module: Any = root
    *parents, leaf = path.split(".")
    for name in parents:
        module = module._modules.get(name) if isinstance(module, nn.Module) else None
        if module is None:
            return None
    value = module._buffers.get(leaf)
    if value is None:
        value = module._parameters.get(leaf)
    return value


def legacy_model_to_current(
    legacy: nn.Module,
    verify: bool = True,
) -> nn.Module:
    """Rebuild an upgraded legacy object as a fresh current-format model.

    The fitted state is transferred strictly, except that a legacy file may
    carry buffers the current constructors recompute deterministically (for
    example the center mask); those are accepted only when their stored
    values equal the recomputed ones. With ``verify=True`` both models are
    evaluated on one synthetic field and must agree exactly.
    """

    if not isinstance(legacy, Configurable):
        raise TypeError(
            "{} cannot be converted: it provides no configuration".format(
                type(legacy).__name__
            )
        )
    config = legacy.to_config()
    with default_dtype(model_dtype(legacy)):
        model = build(config)
    if not isinstance(model, nn.Module):
        raise TypeError("configuration did not rebuild a torch module")

    legacy_state = {
        key: value.detach().clone()
        for key, value in legacy.state_dict().items()
    }
    result = model.load_state_dict(legacy_state, strict=False)
    if result.missing_keys:
        raise ValueError(
            "legacy state lacks required entries {}".format(
                sorted(result.missing_keys)
            )
        )
    for key in result.unexpected_keys:
        recomputed = _attribute_by_path(model, key)
        stored = legacy_state[key]
        if recomputed is None or recomputed.shape != stored.shape or not (
            torch.equal(recomputed.to(stored), stored)
        ):
            raise ValueError(
                "legacy state entry '{}' has no deterministic counterpart in "
                "the rebuilt model".format(key)
            )
    model.train(legacy.training)
    if verify:
        verify_equivalent(legacy, model)
    return model


def _verification_field(model: nn.Module) -> Dict[str, Any]:
    """Build one small positive density field compatible with ``model``."""

    if not isinstance(model, GridCACEModel):
        raise TypeError(
            "forward verification requires a GridCACEModel, not {}".format(
                type(model).__name__
            )
        )
    grid_info = model.grid_info
    cutoff = int(grid_info["cutoff_grid"])
    pair_cutoffs = [
        int(item.cutoff_grid)
        for item in model.readout
        if isinstance(item, PairwiseReadout)
    ]
    size = max([4, 2 * cutoff + 1] + [2 * value for value in pair_cutoffs])
    field = GridData.from_dict(
        {
            "grid_size": (size, size, size),
            "temperature": float(model.mean_temperature),
        },
        grid_info=grid_info,
        include_local_density_index=model.requires_local_density_index,
    )
    generator = torch.Generator().manual_seed(0)
    n_grid = int(field["index"].numel())
    n_types = int(grid_info["n_types"])
    dtype = model.grid_spacing.dtype
    field["rho"] = (
        float(model.mean_density)
        * (0.5 + torch.rand(n_grid, n_types, generator=generator, dtype=dtype))
    )
    field["V_ext"] = torch.rand(n_grid, n_types, generator=generator, dtype=dtype)
    device = model.grid_spacing.device
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in field.items()
    }


def verify_equivalent(reference: nn.Module, candidate: nn.Module) -> None:
    """Raise unless both models produce identical outputs on one field."""

    field = _verification_field(candidate)
    expected = reference(dict(field), compute_c1=True)
    actual = candidate(dict(field), compute_c1=True)
    for key in ("beta_F_exc", "c1"):
        if not torch.equal(expected[key].detach(), actual[key].detach()):
            raise ValueError(
                "converted model output '{}' differs from the legacy "
                "object".format(key)
            )


def convert_legacy_model(
    source: PathLike,
    destination: Optional[PathLike] = None,
    verify: bool = True,
    overwrite: bool = False,
) -> Path:
    """Convert a whole-object model file into the current format.

    ``destination`` defaults to the source path, replacing the legacy file.
    An existing distinct destination is only replaced with ``overwrite``.
    """

    source_path = Path(source).expanduser()
    destination_path = (
        source_path if destination is None else Path(destination).expanduser()
    )
    if (
        destination_path.exists()
        and destination_path.resolve() != source_path.resolve()
        and not overwrite
    ):
        raise FileExistsError(
            "{} exists; pass overwrite=True to replace it".format(
                destination_path
            )
        )
    legacy = load_legacy_model(source_path, map_location="cpu")
    model = legacy_model_to_current(legacy, verify=verify)
    return save_model(model, destination_path)
