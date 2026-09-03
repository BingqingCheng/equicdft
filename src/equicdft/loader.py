"""Split complete density fields and construct PyTorch data loaders."""

import weakref

from typing import (
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from ._argument_checks import (
    boolean,
    finite_scalar,
    nonnegative_integer,
    positive_integer,
)


LoaderResult = Dict[str, Union[DataLoader, float, None]]


class _GridGeometryBatch(dict):
    """Batched fields with identity-proven reusable geometry tensors."""

    def __init__(
        self,
        values: Mapping[str, torch.Tensor],
        shared_geometry: Mapping[str, torch.Tensor],
    ) -> None:
        super().__init__(values)
        self._shared_geometry = dict(shared_geometry)


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
    reuse_grid_geometry: bool = False,
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
    reuse_grid_geometry
        If true, reuse an identity-shared ``local_density_index`` tensor
        across batches instead of copying it for every field. This opt-in
        path currently requires ``num_workers=0``.

    Returns
    -------
    dict
        Always contains the ``train``, ``valid``, and ``test`` loaders, where
        ``test`` is ``None`` when no test dataset is supplied. Requested
        statistics are added under ``mean_density`` and
        ``mean_temperature``.

    Notes
    -----
    PyTorch's default collation is used unless ``reuse_grid_geometry=True``.
    The optimized collator changes only an identity-shared
    ``local_density_index``; all other values retain default collation.
    Fields within each dataset must therefore have matching tensor shapes and
    dictionary keys.
    """

    if (valid_dataset is None) == (valid_fraction is None):
        raise ValueError(
            "provide exactly one of valid_dataset and valid_fraction"
        )
    batch_size = positive_integer(batch_size, "batch_size")
    num_workers = nonnegative_integer(num_workers, "num_workers")
    reuse_grid_geometry = boolean(
        reuse_grid_geometry,
        "reuse_grid_geometry",
    )
    if reuse_grid_geometry and num_workers != 0:
        raise ValueError(
            "reuse_grid_geometry currently requires num_workers=0"
        )
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

    # Keep splitting and epoch shuffling reproducible but independent: drawing
    # a split must not advance the training sampler's random-number stream.
    shuffle_generator = torch.Generator().manual_seed(int(seed))
    train_collate = _grid_collator(train_data) if reuse_grid_geometry else None
    valid_collate = _grid_collator(valid_data) if reuse_grid_geometry else None
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        generator=shuffle_generator,
        collate_fn=train_collate,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=valid_collate,
    )
    test_loader = None
    if test_dataset is not None:
        test_collate = (
            _grid_collator(test_dataset) if reuse_grid_geometry else None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=test_collate,
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


def _grid_collator(
    dataset: Dataset,
) -> Callable[[Sequence[Mapping[str, torch.Tensor]]], dict]:
    """Return a collator that reuses repeated neighborhood-table objects."""

    shared_indices = _repeated_tensor_objects(
        dataset,
        "local_density_index",
    )

    def collate(frames: Sequence[Mapping[str, torch.Tensor]]) -> dict:
        source = _common_shared_tensor(
            frames,
            "local_density_index",
            shared_indices,
        )
        if source is None:
            return default_collate(frames)

        values = default_collate(
            [
                {
                    key: value
                    for key, value in frame.items()
                    if key != "local_density_index"
                }
                for frame in frames
            ]
        )
        values["local_density_index"] = source.unsqueeze(0).expand(
            len(frames),
            -1,
            -1,
        )
        return _GridGeometryBatch(
            values,
            {"local_density_index": source},
        )

    return collate


def _repeated_tensor_objects(
    dataset: Dataset,
    key: str,
) -> Dict[int, torch.Tensor]:
    """Return tensor objects used by more than one frame under ``key``."""

    tensors = weakref.WeakValueDictionary()
    counts: Dict[int, int] = {}
    for frame_index in range(len(dataset)):
        value = dataset[frame_index].get(key)
        if not torch.is_tensor(value):
            continue
        identity = id(value)
        known = tensors.get(identity)
        if known is not None and known is not value:
            raise RuntimeError("tensor object identity changed during scan")
        tensors[identity] = value
        counts[identity] = counts.get(identity, 0) + 1 if known is value else 1
    return {
        identity: tensor
        for identity in counts
        if counts[identity] > 1
        and (tensor := tensors.get(identity)) is not None
    }


def _common_shared_tensor(
    frames: Sequence[Mapping[str, torch.Tensor]],
    key: str,
    shared_tensors: Mapping[int, torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return one repeated tensor object shared by all selected frames."""

    if not frames or key not in frames[0]:
        return None
    source = frames[0][key]
    known = shared_tensors.get(id(source))
    if known is not source:
        return None
    if any(frame.get(key) is not source for frame in frames[1:]):
        return None
    return source


def _random_split(
    dataset: Dataset,
    valid_fraction: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    """Return deterministic, disjoint train and validation subsets."""

    fraction = finite_scalar(valid_fraction, "valid_fraction")
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
    for frame in _frames(datasets):
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
    for frame in _frames(datasets):
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


def _frames(datasets: Sequence[Dataset]) -> Iterator[Dict[str, torch.Tensor]]:
    """Yield every frame in an ordered collection of datasets."""

    for dataset in datasets:
        for frame_index in range(len(dataset)):
            yield dataset[frame_index]


def _require_nonempty(dataset: Optional[Dataset], name: str) -> None:
    """Reject absent or empty datasets with a field-specific message."""

    if dataset is None or len(dataset) == 0:
        raise ValueError("{} must contain at least one frame".format(name))
