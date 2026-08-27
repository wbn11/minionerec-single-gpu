# SPDX-License-Identifier: Apache-2.0
"""Amazon Review 2018 preprocessing for the MiniOneRec reproduction.

Adapted from ``data/amazon18_data_process.py`` in MiniOneRec commit
``0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed``. The data semantics are kept,
while path handling, input validation, overwrite protection, and testability
are local engineering changes.
"""

from __future__ import annotations

import collections
import datetime as dt
import html
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minionerec.data.io import (
    DatasetArtifactPaths,
    artifact_paths,
    create_new_artifact_directory,
    write_interaction_file,
    write_json_file,
    write_remap_index,
)
from minionerec.data.splits import TemporalSplits, global_target_time_split


OFFICIAL_MAX_HISTORY = 10
OFFICIAL_MINIMUM_ITEMS = 3000
OFFICIAL_EARLIEST_YEAR = 1996

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class Amazon18Config:
    """Inputs and official-compatible preprocessing parameters."""

    dataset: str
    metadata_file: Path
    reviews_file: Path
    output_root: Path
    k_core: int = 5
    start_year: int = 1996
    start_month: int = 10
    end_year: int = 2018
    end_month: int = 11
    expand_start_year: bool = True
    minimum_items: int = OFFICIAL_MINIMUM_ITEMS
    earliest_year: int = OFFICIAL_EARLIEST_YEAR

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata_file", Path(self.metadata_file))
        object.__setattr__(self, "reviews_file", Path(self.reviews_file))
        object.__setattr__(self, "output_root", Path(self.output_root))

        if not re.fullmatch(r"[A-Za-z0-9_]+", self.dataset):
            raise ValueError("dataset must contain only letters, digits, and underscores")
        if self.k_core < 1:
            raise ValueError("k_core must be at least 1")
        if self.minimum_items < 0:
            raise ValueError("minimum_items cannot be negative")
        if self.earliest_year > self.start_year:
            raise ValueError("earliest_year cannot be later than start_year")

        start = dt.datetime(self.start_year, self.start_month, 1)
        end = dt.datetime(self.end_year, self.end_month, 1)
        if start > end:
            raise ValueError("start month must not be later than end month")


@dataclass(frozen=True)
class MetadataCatalog:
    records: tuple[JsonObject, ...]
    titles: dict[str, str]
    removed_items: frozenset[str]


@dataclass(frozen=True)
class KCoreResult:
    reviews: tuple[JsonObject, ...]
    user_counts: dict[str, int]
    item_counts: dict[str, int]


@dataclass(frozen=True)
class IdMappings:
    user_histories: dict[int, list[int]]
    user_to_id: dict[str, int]
    item_to_id: dict[str, int]
    interactions: tuple[tuple[str, str, float, int], ...]


@dataclass(frozen=True)
class InteractionSample:
    user: str
    history_asins: tuple[str, ...]
    target_asin: str
    history_item_ids: tuple[int, ...]
    target_item_id: int
    history_titles: tuple[str, ...]
    target_title: str
    history_ratings: tuple[Any, ...]
    target_rating: Any
    history_timestamps: tuple[int, ...]
    target_timestamp: int


@dataclass(frozen=True)
class ProcessingSummary:
    dataset: str
    output_directory: Path
    start_year_used: int
    timezone: tuple[str, str]
    users: int
    items: int
    filtered_reviews: int
    review_feature_records: int
    sequences: int
    train_sequences: int
    valid_sequences: int
    test_sequences: int
    artifact_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "output_directory": str(self.output_directory),
            "start_year_used": self.start_year_used,
            "timezone": list(self.timezone),
            "users": self.users,
            "items": self.items,
            "filtered_reviews": self.filtered_reviews,
            "review_feature_records": self.review_feature_records,
            "sequences": self.sequences,
            "train_sequences": self.train_sequences,
            "valid_sequences": self.valid_sequences,
            "test_sequences": self.test_sequences,
            "artifact_paths": [str(path) for path in self.artifact_paths],
        }


def clean_text(text: Any) -> str:
    """Match the official review/description text cleanup."""

    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", "", str(text))
    cleaned = html.unescape(cleaned)
    cleaned = cleaned.replace("&quot;", '"').replace("&amp;", "&")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def month_start_timestamp(year: int, month: int) -> int:
    """Return the local-time Unix timestamp used by the official script."""

    return int(dt.datetime(year, month, 1).timestamp())


def _read_jsonl(path: Path, *, kind: str) -> list[JsonObject]:
    if not path.is_file():
        raise FileNotFoundError(f"{kind} JSONL file does not exist: {path}")

    records: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {kind} file {path} at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {kind} file {path} at line {line_number}"
                )
            records.append(record)
    return records


def load_metadata(path: Path) -> MetadataCatalog:
    """Load metadata and apply the official title eligibility rules."""

    metadata = _read_jsonl(Path(path), kind="metadata")
    titles: dict[str, str] = {}
    removed_items: set[str] = set()

    for line_number, record in enumerate(metadata, start=1):
        asin = record.get("asin")
        if not isinstance(asin, str) or not asin:
            raise ValueError(f"metadata line {line_number} has no valid asin")

        title = record.get("title")
        if not isinstance(title, str) or "<span id" in title:
            removed_items.add(asin)
            continue

        title = title.replace("&quot;", '"').replace("&amp;", "&")
        title = title.strip(" ").strip('"')
        record["title"] = title
        if len(title) > 1 and len(title.split(" ")) <= 20:
            titles[asin] = title
        else:
            removed_items.add(asin)

    return MetadataCatalog(tuple(metadata), titles, frozenset(removed_items))


def load_reviews(path: Path) -> list[JsonObject]:
    """Load uncompressed Amazon18 review JSONL without rating filtering."""

    reviews = _read_jsonl(Path(path), kind="reviews")
    required = ("reviewerID", "asin", "overall", "unixReviewTime")
    for line_number, review in enumerate(reviews, start=1):
        missing = [field for field in required if field not in review]
        if missing:
            raise ValueError(
                f"review line {line_number} is missing required fields: {', '.join(missing)}"
            )
        try:
            int(review["unixReviewTime"])
            float(review["overall"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"review line {line_number} has an invalid timestamp or rating"
            ) from error
    return reviews


def iterative_k_core(
    reviews: Sequence[JsonObject],
    titles: Mapping[str, str],
    *,
    k: int,
    start_timestamp: int,
    end_timestamp: int,
) -> KCoreResult:
    """Run the official iterative K-core over the inclusive time window."""

    if k < 1:
        raise ValueError("k must be at least 1")
    if start_timestamp > end_timestamp:
        raise ValueError("start_timestamp must not exceed end_timestamp")

    remove_users: set[str] = set()
    remove_items = {review["asin"] for review in reviews if review["asin"] not in titles}
    current_reviews = list(reviews)

    while True:
        new_reviews: list[JsonObject] = []
        user_counts: dict[str, int] = {}
        item_counts: dict[str, int] = {}

        for review in current_reviews:
            timestamp = int(review["unixReviewTime"])
            if timestamp < start_timestamp or timestamp > end_timestamp:
                continue
            user = review["reviewerID"]
            item = review["asin"]
            if user in remove_users or item in remove_items:
                continue
            user_counts[user] = user_counts.get(user, 0) + 1
            item_counts[item] = item_counts.get(item, 0) + 1
            new_reviews.append(review)

        low_users = {user for user, count in user_counts.items() if count < k}
        low_items = {item for item, count in item_counts.items() if count < k}
        if not low_users and not low_items:
            return KCoreResult(tuple(new_reviews), user_counts, item_counts)

        remove_users.update(low_users)
        remove_items.update(low_items)
        current_reviews = new_reviews


def build_id_mappings(reviews: Sequence[JsonObject]) -> IdMappings:
    """Create deterministic zero-based IDs with the official encounter order."""

    user_reviews: dict[str, list[JsonObject]] = {}
    for review in reviews:
        user_reviews.setdefault(review["reviewerID"], []).append(review)
    for records in user_reviews.values():
        records.sort(key=lambda record: int(record["unixReviewTime"]))

    user_to_id: dict[str, int] = {}
    item_to_id: dict[str, int] = {}
    user_histories: dict[int, list[int]] = {}
    interactions: list[tuple[str, str, float, int]] = []

    for user, records in user_reviews.items():
        user_id = len(user_to_id)
        user_to_id[user] = user_id
        item_ids: list[int] = []
        for review in records:
            item = review["asin"]
            if item not in item_to_id:
                item_to_id[item] = len(item_to_id)
            item_ids.append(item_to_id[item])
            interactions.append(
                (
                    user,
                    item,
                    float(review["overall"]),
                    int(review["unixReviewTime"]),
                )
            )
        user_histories[user_id] = item_ids

    return IdMappings(
        user_histories=user_histories,
        user_to_id=user_to_id,
        item_to_id=item_to_id,
        interactions=tuple(interactions),
    )


def build_interaction_samples(
    reviews: Sequence[JsonObject],
    item_to_id: Mapping[str, int],
    titles: Mapping[str, str],
) -> list[InteractionSample]:
    """Create one next-item sample per interaction after each user's first."""

    by_user: dict[str, list[JsonObject]] = {}
    for review in reviews:
        by_user.setdefault(review["reviewerID"], []).append(review)

    samples: list[InteractionSample] = []
    for user, records in by_user.items():
        records.sort(key=lambda record: int(record["unixReviewTime"]))
        for index in range(1, len(records)):
            start = max(index - OFFICIAL_MAX_HISTORY, 0)
            history = records[start:index]
            target = records[index]
            samples.append(
                InteractionSample(
                    user=user,
                    history_asins=tuple(record["asin"] for record in history),
                    target_asin=target["asin"],
                    history_item_ids=tuple(item_to_id[record["asin"]] for record in history),
                    target_item_id=item_to_id[target["asin"]],
                    history_titles=tuple(titles[record["asin"]] for record in history),
                    target_title=titles[target["asin"]],
                    history_ratings=tuple(record["overall"] for record in history),
                    target_rating=target["overall"],
                    history_timestamps=tuple(
                        int(record["unixReviewTime"]) for record in history
                    ),
                    target_timestamp=int(target["unixReviewTime"]),
                )
            )
    return samples


def create_item_features(
    metadata: Sequence[JsonObject],
    item_to_id: Mapping[str, int],
    titles: Mapping[str, str],
) -> dict[int, dict[str, str]]:
    """Create official-compatible title/description/brand/category features."""

    asin_to_metadata = {record["asin"]: record for record in metadata}
    features: dict[int, dict[str, str]] = {}

    for asin, item_id in item_to_id.items():
        record = asin_to_metadata[asin]
        title = titles.get(asin, clean_text(record.get("title", "")))
        description = clean_text(record.get("description", ""))
        brand = record.get("brand", "").replace("by\n", "").strip()

        categories = record.get("categories", [])
        if categories and len(categories) > 0:
            if isinstance(categories[0], list):
                categories = [item for group in categories for item in group]
            categories_text = ",".join(
                str(category).strip()
                for category in categories
                if "</span>" not in str(category)
            ).strip()
        else:
            categories_text = ""

        features[item_id] = {
            "title": title,
            "description": description,
            "brand": brand,
            "categories": categories_text,
        }
    return features


def create_review_features(
    reviews: Sequence[JsonObject],
    user_to_id: Mapping[str, int],
    item_to_id: Mapping[str, int],
) -> dict[str, dict[str, str]]:
    """Create review features keyed by stringified ``(uid, iid, timestamp)``."""

    features: dict[str, dict[str, str]] = {}
    for review in reviews:
        user = review["reviewerID"]
        item = review["asin"]
        if user not in user_to_id or item not in item_to_id:
            continue
        key = str(
            (user_to_id[user], item_to_id[item], review["unixReviewTime"])
        )
        features[key] = {
            "review": clean_text(review.get("reviewText", "")),
            "summary": clean_text(review.get("summary", "")),
        }
    return features


def _interaction_rows(
    samples: Sequence[InteractionSample], user_to_id: Mapping[str, int]
) -> list[tuple[int, Sequence[int], int]]:
    return [
        (user_to_id[sample.user], sample.history_item_ids[-50:], sample.target_item_id)
        for sample in samples
    ]


def _write_outputs(
    paths: DatasetArtifactPaths,
    mappings: IdMappings,
    splits: TemporalSplits[InteractionSample],
    item_features: Mapping[int, Mapping[str, str]],
    review_features: Mapping[str, Mapping[str, str]],
) -> None:
    create_new_artifact_directory(paths)
    write_interaction_file(_interaction_rows(splits.train, mappings.user_to_id), paths.train)
    write_interaction_file(_interaction_rows(splits.valid, mappings.user_to_id), paths.valid)
    write_interaction_file(_interaction_rows(splits.test, mappings.user_to_id), paths.test)
    write_json_file(mappings.user_histories, paths.interactions)
    write_json_file(item_features, paths.items)
    write_json_file(review_features, paths.reviews)
    write_remap_index(mappings.user_to_id, paths.user_mapping)
    write_remap_index(mappings.item_to_id, paths.item_mapping)


def process_amazon18(config: Amazon18Config) -> ProcessingSummary:
    """Run the complete official-compatible Amazon18 preprocessing pipeline.

    Inputs are uncompressed metadata and review JSONL files. Eight artifacts
    are written beneath ``output_root / dataset``. Existing dataset output
    directories are never overwritten.
    """

    catalog = load_metadata(config.metadata_file)
    if not catalog.records:
        raise ValueError("metadata file contains no records")
    reviews = load_reviews(config.reviews_file)
    if not reviews:
        raise ValueError("reviews file contains no records")

    end_timestamp = month_start_timestamp(config.end_year, config.end_month)
    start_year_used = config.start_year
    while True:
        start_timestamp = month_start_timestamp(start_year_used, config.start_month)
        filtered = iterative_k_core(
            reviews,
            catalog.titles,
            k=config.k_core,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
        should_expand = (
            config.expand_start_year
            and start_year_used > config.earliest_year
            and len(filtered.item_counts) < config.minimum_items
        )
        if not should_expand:
            break
        start_year_used -= 1

    mappings = build_id_mappings(filtered.reviews)
    samples = build_interaction_samples(
        filtered.reviews, mappings.item_to_id, catalog.titles
    )
    splits = global_target_time_split(
        samples, target_timestamp=lambda sample: sample.target_timestamp
    )
    item_features = create_item_features(
        catalog.records, mappings.item_to_id, catalog.titles
    )
    review_features = create_review_features(
        filtered.reviews, mappings.user_to_id, mappings.item_to_id
    )

    paths = artifact_paths(config.output_root, config.dataset)
    _write_outputs(paths, mappings, splits, item_features, review_features)
    return ProcessingSummary(
        dataset=config.dataset,
        output_directory=paths.directory,
        start_year_used=start_year_used,
        timezone=(time.tzname[0], time.tzname[1]),
        users=len(mappings.user_to_id),
        items=len(mappings.item_to_id),
        filtered_reviews=len(filtered.reviews),
        review_feature_records=len(review_features),
        sequences=splits.total,
        train_sequences=len(splits.train),
        valid_sequences=len(splits.valid),
        test_sequences=len(splits.test),
        artifact_paths=paths.files,
    )
