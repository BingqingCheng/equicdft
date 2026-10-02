#!/usr/bin/env python3
"""Train the Lennard--Jones example with noise-aware per-case loss weights.

Same model and data as ``train.py``, plus the per-case weighting of
``docs/training.md`` ("Per-case loss weights from signal and noise"):

1. build the model and loss as usual;
2. on the freshly initialised model, measure every training and validation
   case's signal ``S`` (its loss now) and noise ``N`` (the loss of the
   model's linear response to the density's sampling error);
3. turn them into frozen weights ``w ∝ (N + eta S)^-power``, assigned by case
   identity, and record them with the run;
4. train with ``Loss(case_weight_keys=...)``.

Run the unweighted control with the identical model, split and seed by
passing ``--weighting none``; compare the per-case table each run prints.
Run it from the repository root:

    python examples/lj_nvt/train_case_weighted.py
    python examples/lj_nvt/train_case_weighted.py --weighting none --output examples/lj_nvt/fit-control

The example frames carry no density standard error. Unless ``--sem-property``
names one in the EXTXYZ file, the script assumes counting statistics,
``rho_std = sqrt(rho / (dV n_eff))`` with ``n_eff = --effective-samples``
independent frames per field. That is a placeholder for the SEM a real
dataset provides (block averages of the trajectory), and it sets the noise
floor the weights see.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from equicdft import (
    CartesianAFeatures,
    CartesianBFeatures,
    GridCACEModel,
    GridData,
    LDAReadout,
    LocalReadout,
    Loss,
    Metrics,
    TensorLoss,
    Trainer,
    case_loss_statistics,
    case_loss_weights,
    make_dataloaders,
    save_model,
)

HERE = Path(__file__).resolve().parent
TERM = "beta_mu_local"           # the one field-loss term of this example
WEIGHT_KEY = "beta_mu_local_case_weight"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=HERE / "mini_lj_nvt.extxyz")
    parser.add_argument("--output", type=Path, default=HERE / "fit-case-weighted")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--valid-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cutoff-grid", type=int, default=3)
    parser.add_argument("--max-power", type=int, default=3)
    parser.add_argument("--max-product-order", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--rho-min", type=float, default=1.0e-3)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore an existing last.pt checkpoint and restart")

    # Case weighting. "none" is the unweighted control.
    parser.add_argument("--weighting", choices=("noise-aware", "none"), default="noise-aware")
    parser.add_argument("--relative-model-error", type=float, default=0.1,
                        help="eta: fraction of a case's signal a good fit still misses")
    parser.add_argument("--power", type=float, default=0.5,
                        help="tempering of the weight spread, in (0, 1]")
    parser.add_argument("--noise-samples", type=int, default=8)
    parser.add_argument("--noise-step", type=float, default=0.1)
    parser.add_argument("--sem-property", default=None,
                        help="EXTXYZ per-voxel density SEM property, if the data has one")
    parser.add_argument("--effective-samples", type=float, default=1.0e4,
                        help="independent samples per field for the counting-SEM placeholder")
    return parser.parse_args()


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def load_frames(args):
    """Return one dict per field, each with a ``case_id`` and ``rho_std``."""

    data_key = {"rho_std": args.sem_property} if args.sem_property else None
    dataset = GridData.from_xyz(
        args.data.expanduser().resolve(),
        cutoff_grid=args.cutoff_grid,
        boltzmann_constant=1.0,
        thermal_wavelength=1.0,
        **({"data_key": data_key} if data_key else {}),
    )
    frames = []
    for index in range(len(dataset)):
        frame = dict(dataset[index])
        # A stable identity: weights are assigned by it, never by position.
        frame["case_id"] = "frame{:02d}-T{:.2f}".format(index, float(frame["temperature"]))
        if "rho_std" not in frame:
            voxel_volume = float(torch.prod(frame["grid_spacing"]))
            frame["rho_std"] = torch.sqrt(
                frame["rho"].clamp_min(0.0) / (voxel_volume * args.effective_samples)
            )
        excluded = frame.get("excluded_mask")
        if excluded is not None:
            frame["rho_std"] = frame["rho_std"].masked_fill(excluded.bool()[:, None], 0.0)
        frames.append(frame)
    return frames


def build_model(args, frames, mean_density, mean_temperature):
    n_types = int(frames[0]["n_types"])
    a_features = CartesianAFeatures(
        mean_density=mean_density, cutoff_grid=args.cutoff_grid,
        max_power=args.max_power, radial_basis="none", n_radial_channels=1,
        trainable_radial_exponents=False, coordinate_scaling="none",
        separate_center=True, n_types=n_types,
    )
    b_features = CartesianBFeatures(
        max_power=args.max_power, max_product_order=args.max_product_order,
    )
    return GridCACEModel(
        a_features=a_features, b_features=b_features,
        readout=[LDAReadout(n_types=n_types, hidden_sizes=(32, 16)),
                 LocalReadout(n_types=n_types, hidden_sizes=(32, 16))],
        grid_spacing=frames[0]["grid_spacing"], mean_temperature=mean_temperature,
        boltzmann_constant=1.0, thermal_wavelength=1.0,
        compute_c1=True, compute_local_mu=True, rho_min=args.rho_min,
        free_energy_mode="beta",
    )


def make_loss(weighted):
    term = TensorLoss(TERM, "beta_mu_local", "beta_mu_average", weights_key="voxel_weights")
    return Loss([term], case_weight_keys={TERM: WEIGHT_KEY} if weighted else None)


def case_weights(args, model, loss, train_frames, valid_frames, record_path):
    """Measure, or reload, the frozen per-case weights of both splits."""

    expected = {"train": [f["case_id"] for f in train_frames],
                "valid": [f["case_id"] for f in valid_frames]}
    if record_path.is_file():
        # Computed once, before the first epoch; a resumed run must reuse it.
        record = json.loads(record_path.read_text())
        for split, ids in expected.items():
            if sorted(record[split]["by_case"]) != sorted(ids):
                raise ValueError("recorded case weights belong to a different split")
        return record

    def statistics(frames):
        return case_loss_statistics(
            model, loss, DataLoader(frames, batch_size=1, shuffle=False),
            terms=[TERM], case_key="case_id",
            noise_samples=args.noise_samples, noise_step=args.noise_step,
            seed=args.seed,
        )

    train_statistics = statistics(train_frames)
    train = case_loss_weights(
        train_statistics, relative_model_error=args.relative_model_error,
        power=args.power,
    )
    # Validation uses the training normalization and noise scale, so the
    # selection objective is on the training scale. Test cases never enter.
    valid_statistics = statistics(valid_frames)
    valid = case_loss_weights(
        valid_statistics, relative_model_error=args.relative_model_error,
        power=args.power, noise_scale=train["noise_scale"],
        normalization=train["normalization"],
    )

    def split_record(stats, weights):
        rows = {}
        for index, case in enumerate(stats["cases"]):
            rows[case] = {
                "signal": float(stats["signal"][index, 0]),
                "noise": float(stats["noise"][index, 0]),
                "noise_sem": float(stats["noise_sem"][index, 0]),
                "selection_changes": int(stats["selection_changes"][index]),
                "selection_activations": int(stats["selection_activations"][index]),
                "step_reductions": int(stats["step_reductions"][index]),
                "weight": weights["by_case"][case][TERM],
            }
        return {"by_case": rows}

    record = {
        "method": "w = c (noise_scale N + eta S)^-power, frozen at the fresh model",
        "relative_model_error": args.relative_model_error,
        "power": args.power,
        "noise_scale": train["noise_scale"].tolist(),
        "normalization": train["normalization"].tolist(),
        "noise_samples": args.noise_samples,
        "noise_step": args.noise_step,
        "sem_source": args.sem_property or (
            "counting placeholder sqrt(rho / (dV n_eff)), n_eff = {:g}".format(args.effective_samples)
        ),
        "train": split_record(train_statistics, train),
        "valid": split_record(valid_statistics, valid),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def attach(frames, record):
    """Copy the frames with their weight under ``WEIGHT_KEY``, by case identity."""

    return [
        dict(frame, **{WEIGHT_KEY: torch.tensor(record["by_case"][frame["case_id"]]["weight"])})
        for frame in frames
    ]


@torch.no_grad()
def per_case_table(model, frames, device, record, weighted):
    """Final per-case loss next to each case's signal, noise and weight."""

    model.eval()
    unweighted = make_loss(weighted=False)
    print("{:>16} {:>11} {:>11} {:>11} {:>9} {:>8}".format(
        "case", "loss", "signal", "noise", "loss/sig", "weight"))
    for frame in frames:
        batch = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v])
                 for k, v in frame.items()}
        value = float(unweighted(model(batch, use_voxel_mask=True), batch)["total"])
        row = record["by_case"][frame["case_id"]]
        signal = row["signal"]
        print("{:>16} {:11.4e} {:11.4e} {:11.4e} {:9.3f} {:8.3f}".format(
            frame["case_id"], value, signal, row["noise"],
            value / signal if signal else float("nan"), row["weight"] if weighted else 1.0))


def main():
    args = parse_arguments()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    torch.set_default_dtype(torch.float64)   # the noise differences need float64
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    frames = load_frames(args)
    # The library's reproducible split and train+validation statistics.
    split = make_dataloaders(
        train_dataset=frames, valid_fraction=args.valid_fraction, batch_size=1,
        seed=args.seed, compute_mean_density=True, compute_mean_temperature=True,
    )
    train_frames = [frames[i] for i in split["train"].dataset.indices]
    valid_frames = [frames[i] for i in split["valid"].dataset.indices]
    model = build_model(args, frames, split["mean_density"], split["mean_temperature"]).to(device)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = output / "checkpoints"
    weighted = args.weighting == "noise-aware"
    # The statistics need the fresh model; a resumed run reloads them. The
    # control measures them too (no parameter or RNG state changes), so both
    # runs print the same signal and noise columns.
    record = case_weights(args, model, make_loss(weighted=True), train_frames,
                          valid_frames, output / "case_weights.json")
    print("Per-case signal, noise and weight (train){}:".format(
        "" if weighted else " -- not applied in the control"))
    for case, row in record["train"]["by_case"].items():
        print("  {:>16}  S {:.3e}  N {:.3e}  w {:.3f}".format(case, row["signal"], row["noise"], row["weight"]))
    if weighted:
        train_frames = attach(train_frames, record["train"])
        valid_frames = attach(valid_frames, record["valid"])

    # Case weights need one case per batch; shuffling is safe because the
    # weight travels inside each frame.
    loaders = {
        "train": DataLoader(train_frames, batch_size=1, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed)),
        "valid": DataLoader(valid_frames, batch_size=1, shuffle=False),
    }
    loss = make_loss(weighted)
    trainer = Trainer(
        model=model, loss=loss,
        metrics=[Metrics(name="local beta mu", prediction_key="beta_mu_local",
                         target_key="beta_mu_average", metric_keys=("mae", "rmse", "r2"),
                         selection_mask_key="voxel_weights")],
        optimizer_cls=torch.optim.Adam, optimizer_args={"lr": args.learning_rate},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={"factor": 0.5, "patience": 3, "min_lr": 1.0e-6},
        device=device, checkpoint_dir=checkpoint_directory, checkpoint_interval=5,
        save_best=True, log_dir=output / "logs",
    )
    completed = 0
    last = checkpoint_directory / "last.pt"
    if last.is_file() and not args.fresh:
        completed = trainer.load_checkpoint(last, train_loader=loaders["train"])
        print("Resumed from epoch {}.".format(completed))
    if args.epochs > completed:
        trainer.fit(loaders["train"], loaders["valid"], epochs=args.epochs - completed)

    best = torch.load(str(checkpoint_directory / "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    save_model(model, output / "model.pt")
    (output / "run_config.json").write_text(json.dumps({
        "weighting": args.weighting,
        "case_weights": "case_weights.json (applied)" if weighted else "case_weights.json (measured, not applied)",
        "train_cases": [f["case_id"] for f in train_frames],
        "valid_cases": [f["case_id"] for f in valid_frames],
        "seed": args.seed, "epochs": args.epochs, "learning_rate": args.learning_rate,
        "best_epoch": int(best["record"]["epoch"]),
        "best_valid_loss": float(best["record"]["valid_losses"]["total"]),
    }, indent=2) + "\n")

    # Unweighted per-case losses: the same numbers for both runs, so the
    # weighted run and its control compare directly.
    for name, split_frames, split_record in (
        ("train", train_frames, record["train"]),
        ("valid", valid_frames, record["valid"]),
    ):
        print("\nFinal unweighted per-case loss ({}):".format(name))
        per_case_table(model, split_frames, device, split_record, weighted)
    print("\nSaved model: {}".format(output / "model.pt"))


if __name__ == "__main__":
    main()
