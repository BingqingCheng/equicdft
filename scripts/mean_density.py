#!/usr/bin/env python3
"""Print one mean density for each frame of an EXTXYZ dataset."""

import argparse
import json
from pathlib import Path
from typing import List, Union

import numpy as np
from ase.io import iread


def compute_mean_densities(
    path: Union[str, Path],
    density_key: str = "density",
    index: str = ":",
) -> List[float]:
    """Return the scalar mean density of every selected EXTXYZ frame.

    This function can be imported by a training script. Use ``index`` to
    restrict the calculation to the training frames, then average the returned
    values to obtain the scalar passed to ``CartesianAFeatures``.
    """

    mean_densities = []
    for frame_index, atoms in enumerate(
        iread(str(Path(path).expanduser()), index=index)
    ):
        if density_key not in atoms.arrays:
            raise ValueError(
                "frame {} has no '{}' array".format(frame_index, density_key)
            )

        rho = np.asarray(atoms.arrays[density_key], dtype=float)
        if rho.size == 0:
            raise ValueError("frame {} has an empty density array".format(frame_index))

        mean_density = float(np.mean(rho))
        if not np.isfinite(mean_density):
            raise ValueError(
                "frame {} has a non-finite mean density".format(frame_index)
            )
        mean_densities.append(mean_density)

    if not mean_densities:
        raise ValueError("no frames were selected from the dataset")
    return mean_densities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print framewise mean densities as a JSON list."
    )
    parser.add_argument("dataset", help="Input EXTXYZ dataset")
    parser.add_argument(
        "--density-key",
        default="density",
        help="EXTXYZ array containing density values (default: density)",
    )
    parser.add_argument(
        "--index",
        default=":",
        help="ASE frame selection (default: all frames)",
    )
    args = parser.parse_args()

    print(
        json.dumps(
            compute_mean_densities(
                args.dataset,
                density_key=args.density_key,
                index=args.index,
            )
        )
    )


if __name__ == "__main__":
    main()
