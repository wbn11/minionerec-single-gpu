"""Semantic-ID vocabulary helpers for MiniOneRec SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, Sequence


class TokenizerLike(Protocol):
    def __len__(self) -> int: ...

    def add_tokens(self, new_tokens: Sequence[str]) -> int: ...

    def encode(
        self, text: str, *, add_special_tokens: bool = False
    ) -> list[int]: ...


class ModelLike(Protocol):
    def resize_token_embeddings(self, new_num_tokens: int) -> Any: ...


def _load_index(index_file: Path) -> dict[str, list[str]]:
    if not index_file.is_file():
        raise FileNotFoundError(index_file)
    with index_file.open("r", encoding="utf-8") as handle:
        raw_index = json.load(handle)
    if not isinstance(raw_index, dict):
        raise TypeError(f"Expected a JSON object in {index_file}")

    index: dict[str, list[str]] = {}
    for item_id, semantic_tokens in raw_index.items():
        if not isinstance(item_id, str):
            raise TypeError("Semantic-ID index keys must be strings")
        if not isinstance(semantic_tokens, list) or len(semantic_tokens) != 3:
            raise ValueError(f"Item {item_id} does not have exactly three SID tokens")
        if not all(isinstance(token, str) for token in semantic_tokens):
            raise TypeError(f"Item {item_id} contains a non-string SID token")
        index[item_id] = semantic_tokens
    return index


def collect_sid_tokens(index_file: Path) -> list[str]:
    """Return the sorted, unique atomic SID tokens used by the dataset."""

    index = _load_index(Path(index_file))
    return sorted({token for semantic_tokens in index.values() for token in semantic_tokens})


def extend_tokenizer(tokenizer: TokenizerLike, sid_tokens: Sequence[str]) -> int:
    """Add dataset SID tokens as ordinary tokens and return the added count."""

    if len(set(sid_tokens)) != len(sid_tokens):
        raise ValueError("sid_tokens must not contain duplicates")
    return tokenizer.add_tokens(list(sid_tokens))


def resize_model_vocab(model: ModelLike, tokenizer: TokenizerLike) -> int:
    """Resize model input/output embeddings to the tokenizer vocabulary size."""

    vocabulary_size = len(tokenizer)
    model.resize_token_embeddings(vocabulary_size)
    return vocabulary_size


def validate_sid_tokens(
    tokenizer: TokenizerLike,
    index_file: Path,
) -> dict[str, int]:
    """Verify atomic SID tokens and three-token complete Semantic IDs."""

    index = _load_index(Path(index_file))
    sid_tokens = sorted(
        {token for semantic_tokens in index.values() for token in semantic_tokens}
    )

    non_atomic = [
        token
        for token in sid_tokens
        if len(tokenizer.encode(token, add_special_tokens=False)) != 1
    ]
    if non_atomic:
        preview = ", ".join(non_atomic[:5])
        raise ValueError(f"SID tokens are not atomic after extension: {preview}")

    invalid_items = [
        item_id
        for item_id, semantic_tokens in index.items()
        if len(
            tokenizer.encode("".join(semantic_tokens), add_special_tokens=False)
        )
        != 3
    ]
    if invalid_items:
        preview = ", ".join(invalid_items[:5])
        raise ValueError(f"Complete Semantic IDs do not encode to 3 tokens: {preview}")

    return {
        "item_count": len(index),
        "sid_token_count": len(sid_tokens),
        "tokens_per_item": 3,
    }
