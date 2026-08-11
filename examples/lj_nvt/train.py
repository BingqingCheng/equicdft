#!/usr/bin/env python3
"""Train a compact LDA + Cartesian-invariant cDFT model.

This is a pedagogical version of the current Lennard--Jones training protocol.
It deliberately keeps model construction in one file so every important
physical and numerical choice is visible. Run it from the repository root:

    python examples/lj_nvt/train.py
"""

import argparse
import json
from pathlib import Path

import torch

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
    make_dataloaders,
)


HERE = Path(__file__).resolve().parent


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=HERE / "mini_lj_nvt.extxyz",
        help="EXTXYZ file containing complete equilibrium density fields",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "fit",
        help="directory for checkpoints, logs, and the final model",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    # Two validation fields keep R2 meaningful for this five-frame example;
    # larger scientific datasets normally use a smaller validation fraction.
    parser.add_argument("--valid-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto selects CUDA when it is available",
    )

    # Representation controls. cutoff_grid is an integer radius in grid
    # cells: offsets q satisfy q_x^2 + q_y^2 + q_z^2 <= cutoff_grid^2.
    parser.add_argument("--cutoff-grid", type=int, default=3)
    parser.add_argument(
        "--max-power",
        type=int,
        default=3,
        help="maximum Cartesian monomial degree p",
    )
    parser.add_argument(
        "--max-product-order",
        type=int,
        default=2,
        help="maximum invariant product/correlation order nu",
    )

    # Optimization controls. rho_min removes essentially empty cells from the
    # logarithm-based local-chemical-potential loss.
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--rho-min", type=float, default=1.0e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore an existing last.pt checkpoint and restart training",
    )
    return parser.parse_args()


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def main():
    args = parse_arguments()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    # In Lennard--Jones reduced units, k_B = Lambda = 1. Each item returned by
    # GridData is one complete 3D field, including all overlapping local
    # environments required by the functional derivative.
    dataset = GridData.from_xyz(
        args.data.expanduser().resolve(),
        cutoff_grid=args.cutoff_grid,
        boltzmann_constant=1.0,
        thermal_wavelength=1.0,
    )
    loaders = make_dataloaders(
        train_dataset=dataset,
        valid_fraction=args.valid_fraction,
        batch_size=args.batch_size,
        seed=args.seed,
        compute_mean_density=True,
        compute_mean_temperature=True,
    )
    n_types = int(dataset[0]["n_types"].item())

    # Equal-weight Cartesian moments are used without a radial basis. The
    # normalized center density is kept as its own channel; only noncentral
    # neighbors enter the moment sums.
    a_features = CartesianAFeatures(
        mean_density=loaders["mean_density"],
        cutoff_grid=args.cutoff_grid,
        max_power=args.max_power,
        radial_basis="none",
        n_radial_channels=1,
        trainable_radial_exponents=False,
        coordinate_scaling="none",
        separate_center=True,
        n_types=n_types,
    )
    b_features = CartesianBFeatures(
        max_power=args.max_power,
        max_product_order=args.max_product_order,
    )

    # The scalar functional is the sum of two extensive contributions:
    # a density-only LDA baseline and a local invariant many-neighbor
    # correction. Both MLPs use smooth SiLU activations internally.
    readouts = [
        LDAReadout(
            mean_density=loaders["mean_density"],
            n_types=n_types,
            hidden_sizes=(32, 16),
        ),
        LocalReadout(n_types=n_types, hidden_sizes=(32, 16)),
    ]
    model = GridCACEModel(
        a_features=a_features,
        b_features=b_features,
        readout=readouts,
        grid_spacing=dataset[0]["grid_spacing"],
        mean_temperature=loaders["mean_temperature"],
        boltzmann_constant=1.0,
        thermal_wavelength=1.0,
        compute_c1=True,
        compute_local_mu=True,
        rho_min=args.rho_min,
        # The selected protocol learns beta*F_exc directly. The model also
        # supports free_energy_mode="physical" for F_exc/(k_B*T_reference).
        free_energy_mode="beta",
    ).to(device)

    # NVT fields have no known reservoir mu. At equilibrium beta*mu_local is
    # constant, so its masked spatial mean is used as a per-field latent
    # target. Replacing target_key by "beta_mu" gives supervised GCMC fitting.
    loss = Loss(
        terms=[
            TensorLoss(
                name="local_chemical_potential",
                prediction_key="local_chemical_potential",
                target_key="average_chemical_potential",
                weights_key="chemical_potential_weights",
            )
        ]
    )
    metrics = [
        Metrics(
            name="local chemical potential",
            prediction_key="local_chemical_potential",
            target_key="average_chemical_potential",
            metric_keys=("mae", "rmse", "rmse_percent"),
            mask_key="chemical_potential_weights",
        )
    ]

    output = args.output.expanduser().resolve()
    checkpoint_directory = output / "checkpoints"
    log_directory = output / "logs"
    output.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": args.learning_rate},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={"factor": 0.5, "patience": 3, "min_lr": 1.0e-6},
        device=device,
        checkpoint_dir=checkpoint_directory,
        checkpoint_interval=5,
        save_best=True,
        early_stopping_patience=args.early_stopping_patience,
        log_dir=log_directory,
    )

    completed_epochs = 0
    last_checkpoint = checkpoint_directory / "last.pt"
    if last_checkpoint.is_file() and not args.fresh:
        completed_epochs = trainer.load_checkpoint(
            last_checkpoint,
            train_loader=loaders["train"],
        )
        print("Resumed from epoch {}.".format(completed_epochs))

    remaining_epochs = max(0, args.epochs - completed_epochs)
    if remaining_epochs:
        trainer.fit(
            train_loader=loaders["train"],
            valid_loader=loaders["valid"],
            epochs=remaining_epochs,
            print_interval=1,
        )
    elif not (checkpoint_directory / "best.pt").is_file():
        raise RuntimeError("no epochs requested and no best checkpoint exists")

    # Restore the best validation checkpoint before saving the self-contained
    # model. torch.load(model.pt) is sufficient for later inference.
    best = torch.load(
        str(checkpoint_directory / "best.pt"),
        map_location=device,
    )
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    torch.save(model, str(output / "model.pt"))

    run_config = {
        "data": str(args.data.expanduser().resolve()),
        "n_frames": len(dataset),
        "train_frames": len(loaders["train"].dataset),
        "valid_frames": len(loaders["valid"].dataset),
        "seed": args.seed,
        "epoch_budget": args.epochs,
        "best_epoch": int(best["record"]["epoch"]),
        "best_valid_loss": float(best["record"]["valid_losses"]["total"]),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "cutoff_grid": args.cutoff_grid,
        "max_power": args.max_power,
        "max_product_order": args.max_product_order,
        "radial_basis": "none",
        "separate_center": True,
        "readouts": ["LDAReadout", "LocalReadout"],
        "free_energy_mode": "beta",
        "rho_min": args.rho_min,
        "mean_density": float(loaders["mean_density"]),
        "mean_temperature": float(loaders["mean_temperature"]),
        "device": str(device),
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )
    print("Best epoch: {}".format(run_config["best_epoch"]))
    print("Best validation loss: {:.6e}".format(run_config["best_valid_loss"]))
    print("Saved model: {}".format(output / "model.pt"))


if __name__ == "__main__":
    main()
