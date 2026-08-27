"""Convert processed interactions to the CSV layout consumed by MiniOneRec."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OFFICIAL_FILE_SUFFIX = "5_2016-10-2018-11"
CSV_FIELDS = (
    "user_id",
    "history_item_title",
    "item_title",
    "history_item_id",
    "item_id",
    "history_item_sid",
    "item_sid",
)


@dataclass(frozen=True)
class ConversionOutputPaths:
    """Files emitted by the official train/valid/test/info layout."""

    train: Path
    valid: Path
    test: Path
    info: Path

    @property
    def files(self) -> tuple[Path, ...]:
        return (self.train, self.valid, self.test, self.info)


def output_paths(output_dir: Path, category: str) -> ConversionOutputPaths:
    filename = f"{category}_{OFFICIAL_FILE_SUFFIX}"
    return ConversionOutputPaths(
        train=output_dir / "train" / f"{filename}.csv",
        valid=output_dir / "valid" / f"{filename}.csv",
        test=output_dir / "test" / f"{filename}.csv",
        info=output_dir / "info" / f"{filename}.txt",
    )


def semantic_tokens_to_id(tokens: list[str]) -> str:
    """Concatenate the three bracketed RQ-VAE tokens without separators."""

    if len(tokens) != 3 or not all(isinstance(token, str) for token in tokens):
        raise ValueError(f"Expected three Semantic ID tokens, received: {tokens!r}")
    return "".join(tokens)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _read_interactions(path: Path) -> list[tuple[str, list[int], int]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    interactions: list[tuple[str, list[int], int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        expected_header = [
            "user_id:token",
            "item_id_list:token_seq",
            "item_id:token",
        ]
        if header != expected_header:
            raise ValueError(f"Unexpected interaction header in {path}: {header!r}")

        for line_number, row in enumerate(reader, start=2):
            if len(row) != 3:
                raise ValueError(f"Expected three columns at {path}:{line_number}")
            user_id, history_text, target_text = row
            try:
                history_item_ids = [int(value) for value in history_text.split()]
                target_item_id = int(target_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid item ID at {path}:{line_number}"
                ) from error
            interactions.append((user_id, history_item_ids, target_item_id))
    return interactions


def _item_title(items: dict[str, Any], item_id: int) -> str:
    item = items.get(str(item_id))
    if not isinstance(item, dict):
        raise KeyError(f"Missing item metadata for item {item_id}")
    title = item.get("title", f"Item_{item_id}")
    if not isinstance(title, str):
        raise TypeError(f"Item {item_id} has a non-string title")
    return title


def _item_sid(item_to_semantic: dict[str, Any], item_id: int) -> str:
    tokens = item_to_semantic.get(str(item_id))
    if not isinstance(tokens, list):
        raise KeyError(f"Missing Semantic ID for item {item_id}")
    return semantic_tokens_to_id(tokens)


def _convert_split(
    interaction_file: Path,
    output_file: Path,
    items: dict[str, Any],
    item_to_semantic: dict[str, Any],
) -> int:
    interactions = _read_interactions(interaction_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for user_id, history_item_ids, target_item_id in interactions:
            history_titles = [
                _item_title(items, item_id) for item_id in history_item_ids
            ]
            history_sids = [
                _item_sid(item_to_semantic, item_id) for item_id in history_item_ids
            ]
            writer.writerow(
                {
                    "user_id": f"A{user_id}",
                    "history_item_title": history_titles,
                    "item_title": _item_title(items, target_item_id),
                    "history_item_id": history_item_ids,
                    "item_id": target_item_id,
                    "history_item_sid": history_sids,
                    "item_sid": _item_sid(item_to_semantic, target_item_id),
                }
            )
    return len(interactions)


def _write_item_info(
    path: Path,
    items: dict[str, Any],
    item_to_semantic: dict[str, Any],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item_id_text in items:
            try:
                item_id = int(item_id_text)
            except ValueError as error:
                raise ValueError(f"Invalid item ID in item metadata: {item_id_text}") from error
            handle.write(
                f"{_item_sid(item_to_semantic, item_id)}\t"
                f"{_item_title(items, item_id)}\t{item_id}\n"
            )
    return len(items)


def convert_dataset(
    data_dir: Path,
    dataset_name: str,
    output_dir: Path,
    category: str | None = None,
) -> dict[str, Any]:
    """Create MiniOneRec CSV and item-info files without overwriting outputs."""

    category = category or dataset_name
    paths = output_paths(output_dir, category)
    existing = [path for path in paths.files if path.exists()]
    if existing:
        listed = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {listed}")

    items = _load_json_object(data_dir / f"{dataset_name}.item.json")
    item_to_semantic = _load_json_object(data_dir / f"{dataset_name}.index.json")

    if set(items) != set(item_to_semantic):
        missing_sids = sorted(set(items) - set(item_to_semantic))
        unknown_sids = sorted(set(item_to_semantic) - set(items))
        raise ValueError(
            "Item metadata and Semantic ID mappings differ: "
            f"missing_sids={missing_sids[:5]}, unknown_sids={unknown_sids[:5]}"
        )

    split_counts: dict[str, int] = {}
    for split_name, output_file in (
        ("train", paths.train),
        ("valid", paths.valid),
        ("test", paths.test),
    ):
        split_counts[split_name] = _convert_split(
            interaction_file=data_dir / f"{dataset_name}.{split_name}.inter",
            output_file=output_file,
            items=items,
            item_to_semantic=item_to_semantic,
        )

    item_count = _write_item_info(paths.info, items, item_to_semantic)
    return {
        "train_file": str(paths.train),
        "valid_file": str(paths.valid),
        "test_file": str(paths.test),
        "info_file": str(paths.info),
        "rows": split_counts,
        "items": item_count,
    }
