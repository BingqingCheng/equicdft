"""Split complete density fields and construct PyTorch data loaders."""

from typing import Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset


LoaderTriple = Tuple[DataLoader, DataLoader, Optional[DataLoader]]
LoaderResult = Union[
    LoaderTriple,
    Tuple[DataLoader, DataLoader, Optional[DataLoader], float],
]


def make_dataloaders(
    train_dataset: Dataset,
    valid_dataset: Optional[Dataset] = None,
    valid_fraction: Optional[float] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 2,
    seed: int = 1,
    compute_mean_density: bool = False,
    num_workers: int = 0,
) -> LoaderResult:
    """Build train, validation, and optional test loaders.

    Each dataset item is one complete density field. Validation data must be
    supplied either as a separate dataset or as a fraction of
    ``train_dataset``. A fractional split is a reproducible random split; it
    does not inspect temperature, chemical potential, or any other metadata.

    Parameters
    ----------
    train_dataset
        Training dataset, or the combined train-plus-validation pool when
        ``valid_fraction`` is used.
    valid_dataset
        Explicit validation dataset. Mutually exclusive with
        ``valid_fraction``.
    valid_fraction
        Fraction of ``train_dataset`` assigned to validation. Must lie
        strictly between zero and one.
    test_dataset
        Optional separately constructed test dataset.
    batch_size
        Number of complete fields in each batch.
    seed
        Seed used for both the fractional split and training-loader shuffle.
    compute_mean_density
        If true, append the scalar mean density over train and validation
        frames to the returned loaders. Test frames are always excluded.
    num_workers
        Number of worker processes used by each data loader.

    Returns
    -------
    tuple
        ``(train_loader, valid_loader, test_loader)``. When
        ``compute_mean_density=True``, the scalar mean density is appended as
        a fourth item. ``test_loader`` is ``None`` when no test dataset is
        supplied.

    Notes
    -----
    PyTorch's default collation is used. Fields within each dataset must
    therefore have matching tensor shapes and dictionary keys.
    """

    if (valid_dataset is None) == (valid_fraction is None):
        raise ValueError(
            "provide exactly one of valid_dataset and valid_fraction"
        )
    _validate_loader_settings(batch_size, num_workers)
    _require_nonempty(train_dataset, "train_dataset")

    density_datasets = [train_dataset]
    if valid_fraction is not None:
        train_data, valid_data = _random_split(
            train_dataset,
            valid_fraction,
            seed,
        )
    else:
        _require_nonempty(valid_dataset, "valid_dataset")
        train_data = train_dataset
        valid_data = valid_dataset
        density_datasets.append(valid_dataset)

    if test_dataset is not None:
        _require_nonempty(test_dataset, "test_dataset")

    # Keep splitting and epoch shuffling reproducible but independent: drawing
    # a split must not advance the training sampler's random-number stream.
    shuffle_generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        generator=shuffle_generator,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

    loaders = (train_loader, valid_loader, test_loader)
    if not compute_mean_density:
        return loaders
    return loaders + (_mean_density(density_datasets),)


def _random_split(
    dataset: Dataset,
    valid_fraction: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    """Return deterministic, disjoint train and validation subsets."""

    try:
        fraction = float(valid_fraction)
    except (TypeError, ValueError):
        raise ValueError("valid_fraction must be a number between zero and one")
    if not 0.0 < fraction < 1.0:
        raise ValueError("valid_fraction must lie strictly between zero and one")

    n_frames = len(dataset)
    if n_frames < 2:
        raise ValueError(
            "train_dataset must contain at least two frames for a fractional split"
        )
    n_valid = int(round(fraction * n_frames))
    n_valid = max(1, min(n_frames - 1, n_valid))

    split_generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(
        n_frames,
        generator=split_generator,
    ).tolist()
    valid_indices = permutation[:n_valid]
    train_indices = permutation[n_valid:]
    return Subset(dataset, train_indices), Subset(dataset, valid_indices)


def _mean_density(datasets: Sequence[Dataset]) -> float:
    """Average the spatial-and-component mean of each selected frame."""

    frame_means = []
    for dataset in datasets:
        for frame_index in range(len(dataset)):
            frame = dataset[frame_index]
            if "rho" not in frame:
                raise KeyError("dataset frame is missing required field 'rho'")
            rho = torch.as_tensor(frame["rho"])
            if rho.numel() == 0:
                raise ValueError("rho tensors must not be empty")
            frame_mean = rho.detach().to(dtype=torch.float64).mean()
            if not torch.isfinite(frame_mean).item():
                raise ValueError("rho tensors must contain finite values")
            frame_means.append(frame_mean)
    return torch.stack(frame_means).mean().item()


def _require_nonempty(dataset: Optional[Dataset], name: str) -> None:
    """Reject absent or empty datasets with a field-specific message."""

    if dataset is None or len(dataset) == 0:
        raise ValueError("{} must contain at least one frame".format(name))


def _validate_loader_settings(batch_size: int, num_workers: int) -> None:
    """Validate the two integer DataLoader settings exposed by this module."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError("batch_size must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise ValueError("num_workers must be a nonnegative integer")
    if num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
