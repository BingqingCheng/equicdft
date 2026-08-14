"""Shared helpers for the LJ-paper-v1 production regressions."""

import hashlib
import json
import math
from pathlib import Path

import torch

from equicdft import GridData, GridSolver


DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = DIRECTORY / "model.pt"
FIELD_PATH = DIRECTORY / "nvt_T1p5_general_frame120.extxyz"
PROVENANCE_PATH = DIRECTORY / "provenance.json"
EXPECTED_PATH = DIRECTORY / "expected_metrics.json"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def verify_fixtures():
    provenance = load_json(PROVENANCE_PATH)
    artifacts = provenance["artifacts"]
    for name, path in (("model", MODEL_PATH), ("field", FIELD_PATH)):
        actual = _sha256(path)
        expected = artifacts[name]["sha256"]
        if actual != expected:
            raise RuntimeError(
                "{} fixture hash mismatch: {} != {}".format(
                    name, actual, expected
                )
            )


def load_case():
    """Load the frozen CPU model and single held-out NVT field."""

    verify_fixtures()
    torch.set_num_threads(2)
    device = torch.device("cpu")
    try:
        # PyTorch 2.6 defaults to weights-only loading, but this fixture is a
        # complete serialized model and intentionally checks that contract.
        model = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=False,
        ).eval()
    except TypeError:
        # The weights_only keyword is unavailable in older supported PyTorch
        # releases.
        model = torch.load(MODEL_PATH, map_location=device).eval()
    frames = GridData.from_xyz(FIELD_PATH, grid_info=model.grid_info)
    if len(frames) != 1:
        raise RuntimeError("expected exactly one regression field")
    frame = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in frames[0].items()
    }
    return model, GridSolver(model, device=device), frame


def rmse(target, prediction):
    return math.sqrt(float(torch.mean((prediction - target).square())))


def r2(target, prediction):
    residual = float(torch.sum((prediction - target).square()))
    centered = float(torch.sum((target - torch.mean(target)).square()))
    return math.nan if centered <= 0.0 else 1.0 - residual / centered


def check_thresholds(section, metrics):
    """Raise AssertionError when a scientific compatibility gate fails."""

    thresholds = load_json(EXPECTED_PATH)[section]["acceptance"]
    for name, maximum in thresholds.get("maximum", {}).items():
        value = metrics[name]
        if not math.isfinite(value) or value > maximum:
            raise AssertionError(
                "{}={} exceeds maximum {}".format(name, value, maximum)
            )
    for name, minimum in thresholds.get("minimum", {}).items():
        value = metrics[name]
        if not math.isfinite(value) or value < minimum:
            raise AssertionError(
                "{}={} is below minimum {}".format(name, value, minimum)
            )
    for name, required in thresholds.get("equal", {}).items():
        if metrics[name] != required:
            raise AssertionError(
                "{}={!r} does not equal {!r}".format(
                    name, metrics[name], required
                )
            )


def print_metrics(metrics):
    print(json.dumps(metrics, indent=2, sort_keys=True))
