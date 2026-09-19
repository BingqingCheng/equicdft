"""Convert trusted old model files to the current format.

    python -m equicdft.conversion old_model.pt converted_model.pt

The source is never overwritten. Legacy training-checkpoint resume is not
supported; this command converts fitted model files only.
"""

import argparse
import sys
from typing import Optional, Sequence

from .models import convert_model
from ..functional.polarization_ideal import FixedDipoleIdeal
from ..serialization import read_model_config

__all__ = [
    "build_parser",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m equicdft.conversion",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", help="trusted whole-object or versioned model file")
    parser.add_argument(
        "destination",
        help="separate output path (source is never replaced)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the forward-equivalence check between old and new model",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing destination file",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="print the converted model configuration as JSON",
    )
    parser.add_argument("--dipole-magnitude", type=float, nargs="+",
                        help="original molecular moment(s), required for old polarized models")
    parser.add_argument("--thermal-wavelength", type=float, nargs="+", default=[1.0],
                        help="ideal-reference wavelength(s) for polarization conversion")
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    destination = convert_model(
        args.source,
        args.destination,
        verify=not args.no_verify,
        overwrite=args.overwrite,
        ideal=(None if args.dipole_magnitude is None else FixedDipoleIdeal(
            args.dipole_magnitude, thermal_wavelength=args.thermal_wavelength)),
    )
    print("wrote {}".format(destination))
    if args.show_config:
        import json

        print(json.dumps(read_model_config(destination), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
