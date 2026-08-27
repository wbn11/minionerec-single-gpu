"""Statistics for processed MiniOneRec next-item datasets."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from minionerec.data.amazon18 import OFFICIAL_MAX_HISTORY
from minionerec.data.io import INTERACTION_HEADER


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _length_summary(lengths: Sequence[int], *, name: str) -> dict[str, int | float]:
    if not lengths:
        raise ValueError(f"Cannot summarize empty {name}")
    return {
        "min": min(lengths),
        "mean": round(sum(lengths) / len(lengths), 6),
        "max": max(lengths),
    }


def _user_sequence_lengths(path: Path) -> list[int]:
    histories = _load_json_object(path)
    lengths: list[int] = []
    for user_id, history in histories.items():
        if not isinstance(history, list) or not all(
            isinstance(item_id, int) for item_id in history
        ):
            raise TypeError(f"User {user_id} in {path} has an invalid item sequence")
        if not history:
            raise ValueError(f"User {user_id} in {path} has an empty item sequence")
        lengths.append(len(history))
    return lengths


def _split_history_lengths(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(path)

    lengths: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if header != INTERACTION_HEADER:
            raise ValueError(f"Unexpected interaction header in {path}")

        for line_number, line in enumerate(handle, start=2):
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 3:
                raise ValueError(f"Expected three columns at {path}:{line_number}")
            history_length = len(columns[1].split())
            if history_length < 1:
                raise ValueError(f"Empty history at {path}:{line_number}")
            lengths.append(history_length)
    return lengths


def build_dataset_statistics(data_dir: Path, dataset_name: str) -> dict[str, Any]:
    """Build concise history and truncation statistics from processed artifacts."""

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    user_lengths = _user_sequence_lengths(data_dir / f"{dataset_name}.inter.json")
    items = _load_json_object(data_dir / f"{dataset_name}.item.json")

    before_truncation = [
        history_length
        for user_length in user_lengths
        for history_length in range(1, user_length)
    ]
    expected_after_truncation = [
        min(length, OFFICIAL_MAX_HISTORY) for length in before_truncation
    ]

    split_lengths = {
        split: _split_history_lengths(data_dir / f"{dataset_name}.{split}.inter")
        for split in ("train", "valid", "test")
    }
    after_truncation = [
        length for split in ("train", "valid", "test") for length in split_lengths[split]
    ]

    if Counter(after_truncation) != Counter(expected_after_truncation):
        raise ValueError(
            "Actual split history lengths do not match the official recent-history "
            f"limit of {OFFICIAL_MAX_HISTORY}"
        )

    truncated_count = sum(
        length > OFFICIAL_MAX_HISTORY for length in before_truncation
    )
    sample_count = len(before_truncation)
    interaction_count = sum(user_lengths)

    return {
        "dataset": dataset_name,
        "processing": {
            "history_limit": OFFICIAL_MAX_HISTORY,
            "history_order": "chronological",
            "truncation": f"keep the most recent {OFFICIAL_MAX_HISTORY} interactions",
            "split": "stable global target-time 80/10/10",
        },
        "counts": {
            "users": len(user_lengths),
            "items": len(items),
            "interactions": interaction_count,
            "next_item_samples": sample_count,
            "train": len(split_lengths["train"]),
            "valid": len(split_lengths["valid"]),
            "test": len(split_lengths["test"]),
        },
        "user_sequence_length": _length_summary(
            user_lengths, name="user sequences"
        ),
        "sample_history_length": {
            "before_truncation": _length_summary(
                before_truncation, name="histories before truncation"
            ),
            "after_truncation": _length_summary(
                after_truncation, name="histories after truncation"
            ),
            "truncated_samples": truncated_count,
            "truncated_ratio": round(truncated_count / sample_count, 6),
        },
    }


def write_dataset_statistics(
    data_dir: Path,
    dataset_name: str,
    output_file: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write one statistics JSON without overwriting an existing file."""

    data_dir = Path(data_dir)
    output_file = Path(output_file) if output_file else data_dir / (
        f"{dataset_name}.data_stats.json"
    )
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_file}")

    statistics = build_dataset_statistics(data_dir, dataset_name)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(statistics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output_file, statistics
