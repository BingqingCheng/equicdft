"""Split complete density fields and construct shape-compatible data loaders."""

import math
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


LoaderResult = Dict[str, Union[DataLoader, float, None]]


class GridSizeBatchSampler(Sampler):
    """Yield batches whose complete fields share one regular-grid shape.

    Dense tensors with different numbers of grid points cannot be stacked by a
    PyTorch ``DataLoader``. Bucketing by ``grid_size`` permits one dataset to
    contain arbitrary rectangular boxes without padding them to a global size.
    Component count is included in the bucket key to give malformed mixed-type
    datasets a clear boundary as well.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        seed: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(int(seed))
        self.drop_last = False

        buckets = defaultdict(list)
        for index in range(len(dataset)):
            buckets[_grid_bucket_key(dataset[index])].append(index)
        self.buckets = list(buckets.values())

    def __iter__(self) -> Iterator[List[int]]:
        batches = []
        for bucket in self.buckets:
            indices = list(bucket)
            if self.shuffle and len(indices) > 1:
                order = torch.randperm(
                    len(indices),
                    generator=self.generator,
                ).tolist()
                indices = [indices[position] for position in order]
            batches.extend(
                indices[start : start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            )
        if self.shuffle and len(batches) > 1:
            order = torch.randperm(
                len(batches),
                generator=self.generator,
            ).tolist()
            batches = [batches[position] for position in order]
        yield from batches

    def __len__(self) -> int:
        return sum(
            math.ceil(len(bucket) / self.batch_size)
            for bucket in self.buckets
        )


def make_dataloaders(
    train_dataset: Dataset,
    valid_dataset: Optional[Dataset] = None,
    valid_fraction: Optional[float] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 2,
    seed: int = 1,
    compute_mean_density: bool = False,
    compute_mean_temperature: bool = False,
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
    compute_mean_temperature
        If true, append the scalar mean temperature over train and validation
        frames to the returned loaders. Test frames are always excluded.
    num_workers
        Number of worker processes used by each data loader.

    Returns
    -------
    dict
        Always contains the ``train``, ``valid``, and ``test`` loaders, where
        ``test`` is ``None`` when no test dataset is supplied. Requested
        statistics are added under ``mean_density`` and
        ``mean_temperature``.

    Fields may use different ``grid_size`` values. The loaders automatically
    group equally shaped fields into dense batches; an uncommon shape simply
    produces a smaller batch. Dictionary keys must still agree within each
    dataset.
    """

    if (valid_dataset is None) == (valid_fraction is None):
        raise ValueError(
            "provide exactly one of valid_dataset and valid_fraction"
        )
    _validate_loader_settings(batch_size, num_workers)
    _require_nonempty(train_dataset, "train_dataset")

    statistics_datasets = [train_dataset]
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
        statistics_datasets.append(valid_dataset)

    if test_dataset is not None:
        _require_nonempty(test_dataset, "test_dataset")

    train_loader = DataLoader(
        train_data,
        batch_sampler=GridSizeBatchSampler(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        ),
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_sampler=GridSizeBatchSampler(
            valid_data,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        ),
        num_workers=num_workers,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=GridSizeBatchSampler(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
            ),
            num_workers=num_workers,
        )

    result: LoaderResult = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }
    if compute_mean_density:
        result["mean_density"] = _mean_density(statistics_datasets)
    if compute_mean_temperature:
        result["mean_temperature"] = _mean_temperature(statistics_datasets)
    return result


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


def _grid_bucket_key(frame: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int]:
    """Return ``(nx, ny, nz, n_types)`` for shape-compatible collation."""

    if "grid_size" not in frame:
        raise KeyError("dataset frame is missing required field 'grid_size'")
    raw_size = torch.as_tensor(frame["grid_size"]).detach().reshape(-1)
    if raw_size.numel() != 3:
        raise ValueError("grid_size must contain three values")
    grid_size = torch.round(raw_size).to(dtype=torch.long)
    if not torch.allclose(
        raw_size.to(dtype=torch.float64),
        grid_size.to(dtype=torch.float64),
    ) or torch.any(grid_size <= 0).item():
        raise ValueError("grid_size values must be positive integers")

    if "n_types" in frame:
        raw_n_types = torch.as_tensor(frame["n_types"]).detach().reshape(-1)
        if raw_n_types.numel() != 1:
            raise ValueError("n_types must be a positive integer")
        n_types = int(torch.round(raw_n_types[0]).item())
        if not torch.isclose(
            raw_n_types[0].to(dtype=torch.float64),
            torch.tensor(float(n_types), dtype=torch.float64),
        ).item() or n_types < 1:
            raise ValueError("n_types must be a positive integer")
    else:
        field = frame.get("rho", frame.get("V_ext"))
        if field is None or torch.as_tensor(field).ndim != 2:
            raise KeyError(
                "dataset frame needs n_types or a rank-two rho/V_ext field"
            )
        n_types = int(torch.as_tensor(field).shape[-1])

    return (*[int(value) for value in grid_size.cpu().tolist()], n_types)


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


def _mean_temperature(datasets: Sequence[Dataset]) -> float:
    """Average the scalar temperature of each selected frame."""

    temperatures = []
    for dataset in datasets:
        for frame_index in range(len(dataset)):
            frame = dataset[frame_index]
            if "temperature" not in frame:
                raise KeyError(
                    "dataset frame is missing required field 'temperature'"
                )
            temperature = torch.as_tensor(frame["temperature"])
            if temperature.numel() != 1:
                raise ValueError(
                    "temperature tensors must contain exactly one value"
                )
            temperature = temperature.detach().to(dtype=torch.float64).reshape(())
            if not torch.isfinite(temperature).item():
                raise ValueError("temperature tensors must contain finite values")
            if temperature.item() <= 0.0:
                raise ValueError("temperature values must be positive")
            temperatures.append(temperature)
    return torch.stack(temperatures).mean().item()


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
