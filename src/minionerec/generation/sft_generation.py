"""Official-compatible constrained SID generation for SFT evaluation."""

from __future__ import annotations

import ast
import csv
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from minionerec.evaluation.ranking import OFFICIAL_TOP_K, compute_ranking_metrics


OFFICIAL_NUM_BEAMS = 50
OFFICIAL_MAX_NEW_TOKENS = 256
OFFICIAL_LENGTH_PENALTY = 0.0

_EVALUATION_INSTRUCTION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Can you predict the next possible item that the user may expect?

"""


class TokenizerLike(Protocol):
    """Tokenizer operations required by the evaluation pipeline."""

    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class EvaluationExample:
    input_ids: tuple[int, ...]
    target_sid: str


@dataclass(frozen=True)
class SidCatalog:
    item_rows: int
    semantic_ids: tuple[str, ...]
    token_path_to_sid: dict[tuple[int, ...], str]

    @property
    def unique_sid_count(self) -> int:
        return len(self.semantic_ids)

    @property
    def collision_excess(self) -> int:
        return self.item_rows - self.unique_sid_count


class _TrieNode:
    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}


class SidTrie:
    """Token trie whose complete paths end with EOS."""

    def __init__(self, token_paths: Sequence[Sequence[int]]) -> None:
        if not token_paths:
            raise ValueError("SID catalog cannot be empty")
        self.root = _TrieNode()
        for path in token_paths:
            if not path:
                raise ValueError("SID token paths cannot be empty")
            node = self.root
            for token_id in path:
                node = node.children.setdefault(int(token_id), _TrieNode())

    def allowed_tokens(self, prefix: Sequence[int]) -> tuple[int, ...]:
        node = self.root
        for token_id in prefix:
            child = node.children.get(int(token_id))
            if child is None:
                return ()
            node = child
        return tuple(node.children)


class SidTrieLogitsProcessor(LogitsProcessor):
    """Mask every beam to token continuations present in the SID trie."""

    def __init__(self, trie: SidTrie, prompt_width: int, eos_token_id: int) -> None:
        self.trie = trie
        self.prompt_width = prompt_width
        self.eos_token_id = eos_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        masked_scores = cast(
            torch.FloatTensor, torch.full_like(scores, float("-inf"))
        )
        for row_index, sequence in enumerate(input_ids):
            generated_prefix = sequence[self.prompt_width :].tolist()
            allowed = self.trie.allowed_tokens(generated_prefix)
            if not allowed:
                allowed = (self.eos_token_id,)
            masked_scores[row_index, list(allowed)] = scores[row_index, list(allowed)]
        return masked_scores


def _encode_without_terminal_eos(tokenizer: TokenizerLike, text: str) -> list[int]:
    token_ids = list(tokenizer.encode(text))
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    while token_ids and bos_token_id is not None and token_ids[0] == bos_token_id:
        token_ids.pop(0)
    while token_ids and eos_token_id is not None and token_ids[-1] == eos_token_id:
        token_ids.pop()
    return token_ids


def build_official_evaluation_prompt(history_sids: Sequence[str]) -> str:
    """Build the fixed commit's EvalSidDataset prompt without the answer."""

    if not history_sids:
        raise ValueError("evaluation history cannot be empty")
    history = ", ".join(history_sids)
    user_input = (
        "Can you predict the next possible item the user may expect, given the "
        f"following chronological interaction history: {history}"
    )
    return (
        f"{_EVALUATION_INSTRUCTION}"
        f"### User Input:\n{user_input}\n"
        "### Response:\n"
    )


def _parse_sid_history(value: str) -> list[str]:
    history = ast.literal_eval(value)
    if not isinstance(history, list) or not all(
        isinstance(semantic_id, str) and semantic_id for semantic_id in history
    ):
        raise ValueError("history_item_sid must be a non-empty list of strings")
    if not history:
        raise ValueError("history_item_sid cannot be empty")
    return history


def load_evaluation_examples(
    test_file: Path,
    tokenizer: TokenizerLike,
    *,
    sample: int = -1,
    seed: int = 42,
) -> list[EvaluationExample]:
    """Load test CSV rows and encode official answer-free prompts."""

    test_file = Path(test_file)
    if not test_file.is_file():
        raise FileNotFoundError(test_file)
    with test_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if sample > 0:
        if sample > len(rows):
            raise ValueError(f"Cannot sample {sample} rows from {len(rows)} rows")
        rows = random.Random(seed).sample(rows, sample)

    examples: list[EvaluationExample] = []
    for row in rows:
        history = _parse_sid_history(row["history_item_sid"])
        prompt = build_official_evaluation_prompt(history)
        input_ids = _encode_without_terminal_eos(tokenizer, prompt)
        target_sid = row["item_sid"].strip()
        if not input_ids:
            raise ValueError("tokenized evaluation prompt cannot be empty")
        if not target_sid:
            raise ValueError("item_sid cannot be empty")
        examples.append(EvaluationExample(tuple(input_ids), target_sid))
    if not examples:
        raise ValueError("test dataset cannot be empty")
    return examples


def load_sid_catalog(info_file: Path, tokenizer: TokenizerLike) -> SidCatalog:
    """Load catalog SIDs and tokenize complete constrained generation paths."""

    info_file = Path(info_file)
    if not info_file.is_file():
        raise FileNotFoundError(info_file)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    item_rows = 0
    semantic_ids: list[str] = []
    seen_sids: set[str] = set()
    token_path_to_sid: dict[tuple[int, ...], str] = {}
    with info_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 3:
                raise ValueError(f"Expected SID, title, and item ID at {info_file}:{line_number}")
            semantic_id = columns[0].strip()
            if not semantic_id:
                raise ValueError(f"Empty SID at {info_file}:{line_number}")
            item_rows += 1
            if semantic_id in seen_sids:
                continue
            seen_sids.add(semantic_id)
            semantic_ids.append(semantic_id)
            generated_tokens = _encode_without_terminal_eos(
                tokenizer, semantic_id + "\n"
            )
            token_path = tuple([*generated_tokens, eos_token_id])
            existing = token_path_to_sid.get(token_path)
            if existing is not None and existing != semantic_id:
                raise ValueError(
                    f"Different SIDs tokenize to the same path: {existing!r}, {semantic_id!r}"
                )
            token_path_to_sid[token_path] = semantic_id

    if not semantic_ids:
        raise ValueError("SID catalog cannot be empty")
    return SidCatalog(item_rows, tuple(semantic_ids), token_path_to_sid)


def _pad_batch(
    examples: Sequence[EvaluationExample], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompt_width = max(len(example.input_ids) for example in examples)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    for example in examples:
        padding = prompt_width - len(example.input_ids)
        input_ids.append([pad_token_id] * padding + list(example.input_ids))
        attention_mask.append([0] * padding + [1] * len(example.input_ids))
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        prompt_width,
    )


def _generated_path(sequence: Sequence[int], eos_token_id: int) -> tuple[int, ...]:
    path: list[int] = []
    for token_id in sequence:
        path.append(int(token_id))
        if int(token_id) == eos_token_id:
            break
    return tuple(path)


def evaluate_sft_model(
    *,
    model_path: Path,
    test_file: Path,
    info_file: Path,
    device: str = "cuda:0",
    batch_size: int = 8,
    num_beams: int = OFFICIAL_NUM_BEAMS,
    max_new_tokens: int = OFFICIAL_MAX_NEW_TOKENS,
    length_penalty: float = OFFICIAL_LENGTH_PENALTY,
    sample: int = -1,
    seed: int = 42,
) -> dict[str, Any]:
    """Run deterministic constrained beam search and return one metrics object."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_beams < 1:
        raise ValueError("num_beams must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official BF16 evaluation requires an available CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16")

    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    examples = load_evaluation_examples(
        Path(test_file), tokenizer, sample=sample, seed=seed
    )
    catalog = load_sid_catalog(Path(info_file), tokenizer)
    catalog_sid_set = set(catalog.semantic_ids)
    missing_targets = sorted(
        {example.target_sid for example in examples} - catalog_sid_set
    )
    if missing_targets:
        raise ValueError(f"Test targets missing from SID catalog: {missing_targets[:5]}")

    model: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(torch_device)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    token_paths = tuple(catalog.token_path_to_sid)
    trie = SidTrie(token_paths)
    predictions: list[list[str]] = []
    invalid_candidates = 0
    total_candidates = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(torch_device)

    for batch_start in tqdm(
        range(0, len(examples), batch_size), desc="Evaluating SFT"
    ):
        batch = examples[batch_start : batch_start + batch_size]
        input_ids, attention_mask, prompt_width = _pad_batch(
            batch, tokenizer.pad_token_id
        )
        input_ids = input_ids.to(torch_device)
        attention_mask = attention_mask.to(torch_device)
        logits_processor = SidTrieLogitsProcessor(
            trie, prompt_width, tokenizer.eos_token_id
        )

        with torch.inference_mode():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([logits_processor]),
            )

        generated = sequences[:, prompt_width:].detach().cpu().tolist()
        expected_sequences = len(batch) * num_beams
        if len(generated) != expected_sequences:
            raise RuntimeError(
                f"Expected {expected_sequences} generated sequences, got {len(generated)}"
            )
        for row_start in range(0, len(generated), num_beams):
            candidates: list[str] = []
            for token_ids in generated[row_start : row_start + num_beams]:
                path = _generated_path(token_ids, tokenizer.eos_token_id)
                semantic_id = catalog.token_path_to_sid.get(path)
                total_candidates += 1
                if semantic_id is None:
                    invalid_candidates += 1
                    candidates.append("")
                else:
                    candidates.append(semantic_id)
            predictions.append(candidates)

    elapsed_seconds = time.perf_counter() - started
    targets = [example.target_sid for example in examples]
    valid_top_k = tuple(cutoff for cutoff in OFFICIAL_TOP_K if cutoff <= num_beams)
    metrics = compute_ranking_metrics(targets, predictions, valid_top_k)
    allocated_gib = torch.cuda.memory_allocated(torch_device) / (1024**3)
    peak_gib = torch.cuda.max_memory_allocated(torch_device) / (1024**3)

    return {
        "model_path": str(model_path),
        "test_file": str(test_file),
        "info_file": str(info_file),
        "cuda": {
            "logical_device": str(torch_device),
            "name": torch.cuda.get_device_name(torch_device),
            "total_memory_gib": round(
                torch.cuda.get_device_properties(torch_device).total_memory / (1024**3),
                2,
            ),
            "allocated_gib": round(allocated_gib, 2),
            "peak_gib": round(peak_gib, 2),
        },
        "catalog": {
            "item_rows": catalog.item_rows,
            "unique_sids": catalog.unique_sid_count,
            "sid_collision_excess": catalog.collision_excess,
        },
        "generation": {
            "sample_count": len(examples),
            "batch_size": batch_size,
            "num_beams": num_beams,
            "num_return_sequences": num_beams,
            "max_new_tokens": max_new_tokens,
            "length_penalty": length_penalty,
            "do_sample": False,
            "seed": seed,
            "candidate_count": total_candidates,
            "valid_candidates": total_candidates - invalid_candidates,
            "invalid_candidates": invalid_candidates,
            "valid_candidate_rate": (
                (total_candidates - invalid_candidates) / total_candidates
            ),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "samples_per_second": round(len(examples) / elapsed_seconds, 3),
        },
        "metrics": metrics,
    }
