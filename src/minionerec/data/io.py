"""Writers for the eight artifacts emitted by Amazon18 preprocessing."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INTERACTION_HEADER = "user_id:token\titem_id_list:token_seq\titem_id:token\n"


@dataclass(frozen=True)
class DatasetArtifactPaths:
    """Exact output paths used by the official downstream pipeline."""

    directory: Path
    train: Path
    valid: Path
    test: Path
    interactions: Path
    items: Path
    reviews: Path
    user_mapping: Path
    item_mapping: Path

    @property
    def files(self) -> tuple[Path, ...]:
        return (
            self.train,
            self.valid,
            self.test,
            self.interactions,
            self.items,
            self.reviews,
            self.user_mapping,
            self.item_mapping,
        )


def artifact_paths(output_root: Path, dataset: str) -> DatasetArtifactPaths:
    directory = output_root / dataset
    return DatasetArtifactPaths(
        directory=directory,
        train=directory / f"{dataset}.train.inter",
        valid=directory / f"{dataset}.valid.inter",
        test=directory / f"{dataset}.test.inter",
        interactions=directory / f"{dataset}.inter.json",
        items=directory / f"{dataset}.item.json",
        reviews=directory / f"{dataset}.review.json",
        user_mapping=directory / f"{dataset}.user2id",
        item_mapping=directory / f"{dataset}.item2id",
    )


def create_new_artifact_directory(paths: DatasetArtifactPaths) -> None:
    """Create a new dataset directory without overwriting existing data."""

    if paths.directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset directory: {paths.directory}"
        )
    paths.directory.mkdir(parents=True)


def write_json_file(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2)


def write_remap_index(index_map: Mapping[str, int], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for original, mapped in index_map.items():
            handle.write(f"{original}\t{mapped}\n")


def write_interaction_file(
    rows: Iterable[tuple[int, Sequence[int], int]], path: Path
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(INTERACTION_HEADER)
        for user_id, history_item_ids, target_item_id in rows:
            history = " ".join(str(item_id) for item_id in history_item_ids)
            handle.write(f"{user_id}\t{history}\t{target_item_id}\n")
