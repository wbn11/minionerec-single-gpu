"""GRPO datasets enabled by the fixed MiniOneRec reproduction commit."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


OFFICIAL_SEQUENCE_TITLE_SAMPLE = 10_000
OFFICIAL_SEQUENCE_TITLE_SAMPLE_SEED = 0


GRPORecord = dict[str, str]


@dataclass(frozen=True)
class GRPODatasetSizes:
    """Number of examples contributed by each official GRPO task."""

    sid_prediction: int
    title_to_sid: int
    description_to_sid: int
    sequence_title_to_sid: int
    train_total: int
    validation: int


@dataclass(frozen=True)
class GRPODatasets:
    """Training records and the fixed commit's reward lookup mappings."""

    train: tuple[GRPORecord, ...]
    validation: tuple[GRPORecord, ...]
    prompt_to_history: dict[str, str]
    history_to_target: dict[str, str]
    sizes: GRPODatasetSizes
    legal_sids: frozenset[str]


@dataclass(frozen=True)
class _TaskRecords:
    records: tuple[GRPORecord, ...]
    prompt_to_history: dict[str, str]
    history_to_target: dict[str, str]


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")
    return frame


def _load_json_object(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _parse_string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, str):
        raise ValueError(f"{field} must contain a Python-style list of strings")
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(
        isinstance(element, str) for element in parsed
    ):
        raise ValueError(f"{field} must contain a Python-style list of strings")
    return parsed


def _task_records(
    rows: list[tuple[str, str, str]],
) -> _TaskRecords:
    records: list[GRPORecord] = []
    prompt_to_history: dict[str, str] = {}
    history_to_target: dict[str, str] = {}
    for prompt, completion, history in rows:
        records.append({"prompt": prompt, "completion": completion})
        prompt_to_history[prompt] = history
        history_to_target[history] = completion
    return _TaskRecords(
        records=tuple(records),
        prompt_to_history=prompt_to_history,
        history_to_target=history_to_target,
    )


def _sid_prediction_records(frame: pd.DataFrame) -> _TaskRecords:
    rows: list[tuple[str, str, str]] = []
    for row in frame.to_dict(orient="records"):
        history_sids = _parse_string_list(
            row["history_item_sid"], "history_item_sid"
        )
        history_text = ", ".join(history_sids)
        history_key = "::".join(history_sids)
        target = str(row["item_sid"]) + "\n"
        prompt = (
            "### User Input:\n"
            f"The user has interacted with items {history_text} in "
            "chronological order. Can you predict the next possible item "
            "that the user may expect?\n"
            "### Response:\n"
        )
        rows.append((prompt, target, history_key))
    return _task_records(rows)


def _combined_sid(value: object, item_id: str) -> str:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"Item {item_id!r} must have at least three SID tokens")
    return "".join(str(token) for token in value[:3])


def _official_description(value: object, item_id: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Item {item_id!r} description must be a string")
    if value.startswith("['") and value.endswith("']"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list) and parsed:
            first_description = parsed[0]
            if isinstance(first_description, str):
                return first_description
    return value


def _item_feature_records(
    items: dict[str, Any], indices: dict[str, Any]
) -> tuple[_TaskRecords, int, int]:
    title_to_sid: dict[str, str] = {}
    description_to_sid: dict[str, str] = {}

    for raw_item_id, semantic_tokens in indices.items():
        item_id = str(raw_item_id)
        features = items.get(item_id)
        if not isinstance(features, dict):
            raise ValueError(f"Missing item features for item {item_id!r}")
        title = features.get("title")
        if not isinstance(title, str):
            raise TypeError(f"Item {item_id!r} title must be a string")
        description = _official_description(features.get("description"), item_id)
        semantic_id = _combined_sid(semantic_tokens, item_id)

        # Assignment intentionally preserves the fixed commit's last-value-wins
        # behavior for duplicate titles and descriptions.
        title_to_sid[title] = semantic_id
        description_to_sid[description] = semantic_id

    rows: list[tuple[str, str, str]] = []
    for title, semantic_id in title_to_sid.items():
        prompt = (
            "### User Input:\n"
            f"Which item has the title: {title}?\n"
            "### Response:\n"
        )
        rows.append((prompt, semantic_id + "\n", title))
    for description, semantic_id in description_to_sid.items():
        prompt = (
            "### User Input:\n"
            f'An item can be described as follows: "{description}". Which '
            "item is it describing?\n"
            "### Response:\n"
        )
        rows.append((prompt, semantic_id + "\n", description))

    return _task_records(rows), len(title_to_sid), len(description_to_sid)


def _sequence_title_records(
    frame: pd.DataFrame,
    sample_size: int,
) -> _TaskRecords:
    if sample_size <= 0:
        raise ValueError("sequence title sample size must be positive")
    if sample_size > len(frame):
        raise ValueError(
            f"Cannot sample {sample_size} rows from {len(frame)} training rows"
        )
    sampled = frame.sample(
        n=sample_size,
        random_state=OFFICIAL_SEQUENCE_TITLE_SAMPLE_SEED,
    )

    rows: list[tuple[str, str, str]] = []
    for row in sampled.to_dict(orient="records"):
        history_titles = _parse_string_list(
            row["history_item_title"], "history_item_title"
        )
        formatted_titles = ", ".join(
            f'"{title}"' for title in history_titles
        )
        history_key = "::".join(history_titles)
        question = (
            "Given the title sequence of user historical interactive items: "
            f"{formatted_titles}, can you recommend a suitable next item for "
            "the user?"
        )
        prompt = f"### User Input:\n{question}\n### Response:\n"
        rows.append((prompt, str(row["item_sid"]) + "\n", history_key))
    return _task_records(rows)


def _merge_mappings(
    datasets: list[_TaskRecords],
) -> tuple[dict[str, str], dict[str, str]]:
    prompt_to_history: dict[str, str] = {}
    history_to_target: dict[str, str] = {}
    for dataset in datasets:
        prompt_to_history.update(dataset.prompt_to_history)
        history_to_target.update(dataset.history_to_target)
    return prompt_to_history, history_to_target


def _validate_completions(
    records: tuple[GRPORecord, ...], legal_sids: frozenset[str]
) -> None:
    invalid = sorted(
        {
            record["completion"].removesuffix("\n")
            for record in records
            if record["completion"].removesuffix("\n") not in legal_sids
        }
    )
    if invalid:
        raise ValueError(f"GRPO completions contain unknown SIDs: {invalid[:5]}")


def build_grpo_datasets(
    *,
    train_file: Path,
    valid_file: Path,
    item_file: Path,
    index_file: Path,
    sequence_title_sample: int = OFFICIAL_SEQUENCE_TITLE_SAMPLE,
) -> GRPODatasets:
    """Build the three active official GRPO tasks and SID validation data."""

    train_frame = _read_csv(
        Path(train_file),
        {"history_item_sid", "history_item_title", "item_sid"},
    )
    valid_frame = _read_csv(
        Path(valid_file),
        {"history_item_sid", "item_sid"},
    )
    items = _load_json_object(Path(item_file))
    indices = _load_json_object(Path(index_file))

    legal_sids = frozenset(
        _combined_sid(tokens, str(item_id))
        for item_id, tokens in indices.items()
    )
    sid_prediction = _sid_prediction_records(train_frame)
    item_features, title_count, description_count = _item_feature_records(
        items, indices
    )
    sequence_titles = _sequence_title_records(
        train_frame,
        sequence_title_sample,
    )
    validation = _sid_prediction_records(valid_frame)

    train = (
        sid_prediction.records
        + item_features.records
        + sequence_titles.records
    )
    _validate_completions(train, legal_sids)
    _validate_completions(validation.records, legal_sids)

    # Preserve the fixed commit's update order: all training tasks first, then
    # validation. Later mappings overwrite earlier entries on duplicate keys.
    prompt_to_history, history_to_target = _merge_mappings(
        [sid_prediction, item_features, sequence_titles, validation]
    )
    sizes = GRPODatasetSizes(
        sid_prediction=len(sid_prediction.records),
        title_to_sid=title_count,
        description_to_sid=description_count,
        sequence_title_to_sid=len(sequence_titles.records),
        train_total=len(train),
        validation=len(validation.records),
    )
    return GRPODatasets(
        train=train,
        validation=validation.records,
        prompt_to_history=prompt_to_history,
        history_to_target=history_to_target,
        sizes=sizes,
        legal_sids=legal_sids,
    )
