"""Private serialization and presentation helpers for :mod:`trainer`."""

import csv
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from ._argument_checks import boolean
from .metrics import format_metric_value, metric_label


def atomic_torch_save(value: object, path: Path) -> None:
    """Write a Torch object atomically through a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(".{}.tmp".format(path.name))
    try:
        torch.save(value, str(temporary_path))
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def append_log_message(
    message: str,
    path: Optional[Path],
    display: bool,
) -> None:
    """Display a message and optionally append it to a timestamped log."""

    if not isinstance(message, str):
        raise TypeError("message must be a string")
    display = boolean(display, "display")
    if display:
        print(message)
    if path is not None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as handle:
            handle.write("[{}]\n{}\n\n".format(timestamp, message))


def write_history_csv(history: List[Dict[str, Any]], path: Path) -> None:
    """Atomically write flattened epoch records to CSV."""

    if not history:
        return
    rows = [_flatten_history_record(record) for record in history]
    fieldnames = ["epoch", "learning_rate"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    temporary_path = path.with_name(".{}.tmp".format(path.name))
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _flatten_history_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one nested epoch record into stable CSV column names."""

    row = {
        "epoch": record["epoch"],
        "learning_rate": record["learning_rate"],
    }
    if "feature_learning_rate" in record:
        row["feature_learning_rate"] = record["feature_learning_rate"]
    loss_names = _ordered_union(
        record["train_losses"],
        record["valid_losses"],
    )
    for loss_name in loss_names:
        column_name = _column_name(loss_name)
        for subset in ("train", "valid"):
            values = record["{}_losses".format(subset)]
            if loss_name in values:
                row["{}_loss_{}".format(subset, column_name)] = values[
                    loss_name
                ]

    collection_names = _ordered_union(
        record["train_metrics"],
        record["valid_metrics"],
    )
    for collection_name in collection_names:
        collection_column = _column_name(collection_name)
        metric_names = _ordered_union(
            record["train_metrics"].get(collection_name, {}),
            record["valid_metrics"].get(collection_name, {}),
        )
        for metric_name in metric_names:
            metric_column = _column_name(metric_name)
            for subset in ("train", "valid"):
                values = record["{}_metrics".format(subset)].get(
                    collection_name,
                    {},
                )
                if metric_name in values:
                    row[
                        "{}_{}_{}".format(
                            subset,
                            collection_column,
                            metric_column,
                        )
                    ] = values[metric_name]
    return row


def _column_name(name: str) -> str:
    """Return a lowercase underscore-separated CSV column fragment."""

    characters = []
    previous_was_separator = False
    for character in name.strip().lower():
        if character.isalnum():
            characters.append(character)
            previous_was_separator = False
        elif not previous_was_separator:
            characters.append("_")
            previous_was_separator = True
    return "".join(characters).strip("_")


def _ordered_union(*collections: Iterable[str]) -> List[str]:
    """Return unique strings in first-occurrence order."""

    return list(
        dict.fromkeys(
            value for collection in collections for value in collection
        )
    )


def format_record(record: Dict[str, Any]) -> str:
    """Format one readable, terminal-safe multi-line epoch summary."""

    lines = [
        "Epoch {:4d} | learning rate {:.3e}".format(
            record["epoch"],
            record["learning_rate"],
        )
    ]
    if "feature_learning_rate" in record:
        lines[0] += " | feature learning rate {:.3e}".format(
            record["feature_learning_rate"]
        )

    available_losses = _ordered_union(
        record["train_losses"],
        record["valid_losses"],
    )
    loss_names = ["total"] + [
        name for name in available_losses if name != "total"
    ]
    loss_rows = []
    for subset in ("train", "valid"):
        values = record["{}_losses".format(subset)]
        if values:
            loss_rows.append(
                [subset]
                + [
                    (
                        "{:.6e}".format(values[name])
                        if name in values
                        else "-"
                    )
                    for name in loss_names
                ]
            )
    lines.extend(
        ["", _format_table("Losses", ["subset"] + loss_names, loss_rows)]
    )

    collection_names = _ordered_union(
        record["train_metrics"],
        record["valid_metrics"],
    )
    for collection_name in collection_names:
        metric_names = _ordered_union(
            record["train_metrics"].get(collection_name, {}),
            record["valid_metrics"].get(collection_name, {}),
        )
        metric_rows = []
        for subset in ("train", "valid"):
            values = record["{}_metrics".format(subset)].get(
                collection_name
            )
            if values is not None:
                metric_rows.append(
                    [subset]
                    + [
                        (
                            format_metric_value(name, values[name])
                            if name in values
                            else "-"
                        )
                        for name in metric_names
                    ]
                )
        lines.extend(
            [
                "",
                _format_table(
                    "{} metrics".format(collection_name),
                    ["subset"] + [metric_label(name) for name in metric_names],
                    metric_rows,
                ),
            ]
        )
    return "\n".join(lines)


def _format_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    """Format a small aligned ASCII table."""

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def format_row(row: Sequence[str]) -> str:
        cells = [row[0].ljust(widths[0])]
        cells.extend(
            value.rjust(width) for value, width in zip(row[1:], widths[1:])
        )
        return "  " + "  ".join(cells)

    separator = "  " + "  ".join("-" * width for width in widths)
    table_lines = [title, format_row(headers), separator]
    table_lines.extend(format_row(row) for row in rows)
    return "\n".join(table_lines)
