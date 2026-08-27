"""SFT datasets enabled by the fixed MiniOneRec reproduction commit."""

from __future__ import annotations

import ast
import csv
import json
import random
from pathlib import Path
from typing import Any, Protocol

from torch.utils.data import Dataset


class TokenizerLike(Protocol):
    """Tokenizer operations used while constructing SFT examples."""

    bos_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str) -> list[int]: ...


EncodedExample = dict[str, list[int]]


_NEXT_SID_INSTRUCTION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Can you predict the next possible item that the user may expect?
"""

_ITEM_IDENTIFICATION_INSTRUCTION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Answer the question about item identification.
"""

_FUSION_INSTRUCTION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Can you recommend the next item for the user based on their interaction history?
"""


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _read_csv_rows(path: Path, sample: int, seed: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if sample > 0:
        if sample > len(rows):
            raise ValueError(f"Cannot sample {sample} rows from {len(rows)} rows")
        rows = random.Random(seed).sample(rows, sample)
    return rows


def _parse_string_list(value: str, field: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(
        isinstance(element, str) for element in parsed
    ):
        raise ValueError(f"{field} must contain a Python-style list of strings")
    return parsed


def _encode(
    tokenizer: TokenizerLike,
    text: str,
    *,
    bos: bool,
    eos: bool,
) -> list[int]:
    """Mirror the lightweight tokenizer wrapper used by MiniOneRec."""

    token_ids = list(tokenizer.encode(text))
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    while token_ids and bos_id is not None and token_ids[0] == bos_id:
        token_ids.pop(0)
    while token_ids and eos_id is not None and token_ids[-1] == eos_id:
        token_ids.pop()

    if bos and bos_id is not None:
        token_ids.insert(0, bos_id)
    if eos and eos_id is not None:
        token_ids.append(eos_id)
    return token_ids


def _encode_example(
    tokenizer: TokenizerLike,
    instruction: str,
    prompt: str,
    target: str,
    max_length: int,
    test: bool,
) -> EncodedExample:
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    input_ids = _encode(tokenizer, instruction, bos=True, eos=False)
    input_ids.extend(_encode(tokenizer, prompt, bos=False, eos=False))

    if test:
        return {
            "input_ids": input_ids[-max_length:],
            "attention_mask": [1] * min(len(input_ids), max_length),
        }

    prompt_length = len(input_ids)
    input_ids.extend(_encode(tokenizer, target, bos=False, eos=True))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    return {
        "input_ids": input_ids[-max_length:],
        "attention_mask": [1] * min(len(input_ids), max_length),
        "labels": labels[-max_length:],
    }


def _base_prompt(user_input: str) -> str:
    return f"### User Input:\n{user_input}\n### Response:\n"


class _EncodedDataset(Dataset[EncodedExample]):
    def __init__(self, inputs: list[EncodedExample]) -> None:
        self.inputs = inputs

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.inputs[index]


class SidSFTDataset(_EncodedDataset):
    """Predict the next Semantic ID from a chronological SID history."""

    def __init__(
        self,
        train_file: Path,
        tokenizer: TokenizerLike,
        max_length: int = 512,
        sample: int = -1,
        test: bool = False,
        seed: int = 0,
    ) -> None:
        rows = _read_csv_rows(Path(train_file), sample, seed)
        inputs: list[EncodedExample] = []
        for row in rows:
            history = _parse_string_list(
                row["history_item_sid"], "history_item_sid"
            )
            history_text = ", ".join(history)
            user_input = (
                f"The user has interacted with items {history_text} in "
                "chronological order. Can you predict the next possible item "
                "that the user may expect?"
            )
            inputs.append(
                _encode_example(
                    tokenizer,
                    _NEXT_SID_INSTRUCTION,
                    _base_prompt(user_input),
                    row["item_sid"] + "\n",
                    max_length,
                    test,
                )
            )
        super().__init__(inputs)


class SidItemFeatDataset(_EncodedDataset):
    """Teach both SID-to-title and title-to-SID item identification."""

    def __init__(
        self,
        item_file: Path,
        index_file: Path,
        tokenizer: TokenizerLike,
        max_length: int = 512,
        sample: int = -1,
        test: bool = False,
        seed: int = 0,
    ) -> None:
        items = _load_json_object(Path(item_file))
        indices = _load_json_object(Path(index_file))

        sid_to_title: dict[str, str] = {}
        title_to_sid: dict[str, str] = {}
        for item_id, semantic_tokens in indices.items():
            features = items.get(item_id)
            if not isinstance(features, dict):
                continue
            if not isinstance(semantic_tokens, list) or len(semantic_tokens) < 3:
                continue
            title = features.get("title")
            if not isinstance(title, str):
                continue
            sid = "".join(str(token) for token in semantic_tokens[:3])
            sid_to_title[sid] = title
            title_to_sid[title] = sid

        tasks: list[tuple[str, str, str]] = []
        tasks.extend(("sid2title", sid, title) for sid, title in sid_to_title.items())
        tasks.extend(("title2sid", title, sid) for title, sid in title_to_sid.items())
        if sample > 0 and sample < len(tasks):
            tasks = random.Random(seed).sample(tasks, sample)

        inputs: list[EncodedExample] = []
        for task, source, target in tasks:
            if task == "title2sid":
                question = f"Which item has the title: {source}?"
            else:
                question = f'What is the title of item "{source}"?'
            inputs.append(
                _encode_example(
                    tokenizer,
                    _ITEM_IDENTIFICATION_INSTRUCTION,
                    _base_prompt(question),
                    target + "\n",
                    max_length,
                    test,
                )
            )
        super().__init__(inputs)


class FusionSeqRecDataset(_EncodedDataset):
    """Predict the next item's title from a chronological SID history."""

    def __init__(
        self,
        train_file: Path,
        item_file: Path,
        index_file: Path,
        tokenizer: TokenizerLike,
        max_length: int = 512,
        sample: int = -1,
        test: bool = False,
        seed: int = 0,
    ) -> None:
        rows = _read_csv_rows(Path(train_file), sample, seed)
        items = _load_json_object(Path(item_file))
        indices = _load_json_object(Path(index_file))

        sid_to_title: dict[str, str] = {}
        for item_id, semantic_tokens in indices.items():
            features = items.get(item_id)
            if not isinstance(features, dict):
                continue
            if not isinstance(semantic_tokens, list) or len(semantic_tokens) < 3:
                continue
            title = features.get("title")
            if isinstance(title, str):
                sid_to_title["".join(str(token) for token in semantic_tokens[:3])] = title

        inputs: list[EncodedExample] = []
        for row in rows:
            history = _parse_string_list(
                row["history_item_sid"], "history_item_sid"
            )
            history_text = ", ".join(history)
            target_sid = row["item_sid"]
            target_title = sid_to_title.get(target_sid, target_sid)
            question = (
                f"The user has sequentially interacted with items {history_text}. "
                "Can you recommend the next item for him? Tell me the title of "
                "the item"
            )
            inputs.append(
                _encode_example(
                    tokenizer,
                    _FUSION_INSTRUCTION,
                    _base_prompt(question),
                    target_title + "\n",
                    max_length,
                    test,
                )
            )
        super().__init__(inputs)
