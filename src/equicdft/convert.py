"""Convert whole-object ``torch.save(model)`` files to the current format.

Run from an environment where ``equicdft`` is installed::

    python -m equicdft.convert legacy_model.pt [converted_model.pt]

Without a destination the legacy file is replaced in place. The converted
file loads with :func:`equicdft.load_model`.
"""

import argparse
import sys
from typing import Optional, Sequence

from .legacy import convert_legacy_model
from .serialization import read_model_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m equicdft.convert",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", help="whole-object torch.save(model) file")
    parser.add_argument(
        "destination",
        nargs="?",
        default=None,
        help="output path (default: replace the source file)",
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
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    destination = convert_legacy_model(
        args.source,
        args.destination,
        verify=not args.no_verify,
        overwrite=args.overwrite,
    )
    print("wrote {}".format(destination))
    if args.show_config:
        import json

        print(json.dumps(read_model_config(destination), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
