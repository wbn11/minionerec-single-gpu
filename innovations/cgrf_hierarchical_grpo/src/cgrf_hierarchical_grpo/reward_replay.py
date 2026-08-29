"""Generate fixed SFT candidates and replay baseline and CGRF-H rewards."""

from __future__ import annotations

import ast
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from minionerec.generation.grpo_generation import GRPORolloutGenerator
from minionerec.generation.sft_generation import load_sid_catalog

from .reward_fusion import GroupRewardComponents, compute_group_reward_components
from .sasrec import SASRec, load_sasrec_checkpoint


COLLABORATIVE_TASKS = frozenset({"sid_prediction", "sequence_title_to_sid"})
OFFICIAL_SEQUENCE_TITLE_SAMPLE = 10_000
OFFICIAL_SEQUENCE_TITLE_SAMPLE_SEED = 0


@dataclass(frozen=True)
class ReplayRecord:
    """One GRPO prompt plus Item-ID metadata needed by the teacher."""

    source_index: int
    task: str
    prompt: str
    target_sid: str
    history_item_ids: tuple[int, ...]
    target_item_id: int | None


@dataclass(frozen=True)
class CandidateGroup:
    """One cached on-policy candidate group."""

    source_index: int
    task: str
    target_sid: str
    history_item_ids: tuple[int, ...]
    target_item_id: int | None
    candidate_sids: tuple[str, ...]


@dataclass(frozen=True)
class _AnalyzedGroup:
    group: CandidateGroup
    components: GroupRewardComponents


def _parse_item_history(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise ValueError(f"{field} must contain a Python-style integer list")
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not parsed or not all(
        isinstance(item_id, int) for item_id in parsed
    ):
        raise ValueError(f"{field} must contain a non-empty integer list")
    return tuple(parsed)


def _read_csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        return list(reader)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _parse_string_history(value: str, field: str) -> tuple[str, ...]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not parsed or not all(
        isinstance(element, str) for element in parsed
    ):
        raise ValueError(f"{field} must contain a non-empty string list")
    return tuple(parsed)


def _combined_sid(tokens: object, item_id: str) -> str:
    if not isinstance(tokens, list) or len(tokens) < 3:
        raise ValueError(f"Item {item_id!r} must have at least three SID tokens")
    return "".join(str(token) for token in tokens[:3])


def _official_description(value: object, item_id: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Item {item_id!r} description must be a string")
    if value.startswith("['") and value.endswith("']"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
            return parsed[0]
    return value


def _sid_prediction_record(index: int, row: dict[str, str]) -> ReplayRecord:
    history_sids = _parse_string_history(
        row["history_item_sid"], "history_item_sid"
    )
    history_text = ", ".join(history_sids)
    prompt = (
        "### User Input:\n"
        f"The user has interacted with items {history_text} in "
        "chronological order. Can you predict the next possible item "
        "that the user may expect?\n"
        "### Response:\n"
    )
    return ReplayRecord(
        source_index=index,
        task="sid_prediction",
        prompt=prompt,
        target_sid=row["item_sid"].strip(),
        history_item_ids=_parse_item_history(
            row["history_item_id"], "history_item_id"
        ),
        target_item_id=int(row["item_id"]),
    )


def build_replay_records(
    *,
    train_file: Path,
    valid_file: Path,
    item_file: Path,
    index_file: Path,
    sequence_title_sample: int = OFFICIAL_SEQUENCE_TITLE_SAMPLE,
) -> tuple[ReplayRecord, ...]:
    """Rebuild official prompts while retaining sequence Item IDs."""

    train_rows = _read_csv_rows(
        train_file,
        {
            "history_item_id",
            "item_id",
            "history_item_sid",
            "item_sid",
            "history_item_title",
        },
    )
    _read_csv_rows(valid_file, {"history_item_sid", "item_sid"})
    items = _load_json_object(item_file)
    indices = _load_json_object(index_file)
    legal_sids = {
        _combined_sid(tokens, str(item_id))
        for item_id, tokens in indices.items()
    }

    replay_records = [
        _sid_prediction_record(index, row)
        for index, row in enumerate(train_rows)
    ]
    title_to_sid: dict[str, str] = {}
    description_to_sid: dict[str, str] = {}
    for raw_item_id, tokens in indices.items():
        item_id = str(raw_item_id)
        features = items.get(item_id)
        if not isinstance(features, dict):
            raise ValueError(f"Missing item features for item {item_id!r}")
        title = features.get("title")
        if not isinstance(title, str):
            raise TypeError(f"Item {item_id!r} title must be a string")
        description = _official_description(features.get("description"), item_id)
        semantic_id = _combined_sid(tokens, item_id)
        title_to_sid[title] = semantic_id
        description_to_sid[description] = semantic_id

    for title, semantic_id in title_to_sid.items():
        prompt = (
            "### User Input:\n"
            f"Which item has the title: {title}?\n"
            "### Response:\n"
        )
        replay_records.append(
            ReplayRecord(
                source_index=len(replay_records),
                task="title_to_sid",
                prompt=prompt,
                target_sid=semantic_id,
                history_item_ids=(),
                target_item_id=None,
            )
        )
    for description, semantic_id in description_to_sid.items():
        prompt = (
            "### User Input:\n"
            f'An item can be described as follows: "{description}". Which '
            "item is it describing?\n"
            "### Response:\n"
        )
        replay_records.append(
            ReplayRecord(
                source_index=len(replay_records),
                task="description_to_sid",
                prompt=prompt,
                target_sid=semantic_id,
                history_item_ids=(),
                target_item_id=None,
            )
        )

    if sequence_title_sample <= 0 or sequence_title_sample > len(train_rows):
        raise ValueError(
            f"Cannot sample {sequence_title_sample} sequence-title rows from "
            f"{len(train_rows)} training rows"
        )
    sampled_indices = np.random.RandomState(
        OFFICIAL_SEQUENCE_TITLE_SAMPLE_SEED
    ).choice(len(train_rows), size=sequence_title_sample, replace=False)
    for sampled_index in sampled_indices.tolist():
        row = train_rows[int(sampled_index)]
        history_titles = _parse_string_history(
            row["history_item_title"], "history_item_title"
        )
        formatted_titles = ", ".join(
            f'"{title}"' for title in history_titles
        )
        question = (
            "Given the title sequence of user historical interactive items: "
            f"{formatted_titles}, can you recommend a suitable next item for "
            "the user?"
        )
        replay_records.append(
            ReplayRecord(
                source_index=len(replay_records),
                task="sequence_title_to_sid",
                prompt=f"### User Input:\n{question}\n### Response:\n",
                target_sid=row["item_sid"].strip(),
                history_item_ids=_parse_item_history(
                    row["history_item_id"], "history_item_id"
                ),
                target_item_id=int(row["item_id"]),
            )
        )

    unknown_targets = sorted(
        {record.target_sid for record in replay_records} - legal_sids
    )
    if unknown_targets:
        raise ValueError(f"Replay targets contain unknown SIDs: {unknown_targets[:5]}")
    return tuple(replay_records)


def build_prompt_metadata(
    *,
    train_file: Path,
    valid_file: Path,
    item_file: Path,
    index_file: Path,
    sequence_title_sample: int = OFFICIAL_SEQUENCE_TITLE_SAMPLE,
) -> dict[str, ReplayRecord]:
    """Map official train and validation prompts to reward metadata.

    Assignment order matches the baseline GRPO mappings: validation sequence
    records are applied last when an identical prompt occurs more than once.
    """

    records = list(
        build_replay_records(
            train_file=train_file,
            valid_file=valid_file,
            item_file=item_file,
            index_file=index_file,
            sequence_title_sample=sequence_title_sample,
        )
    )
    validation_rows = _read_csv_rows(
        valid_file,
        {
            "history_item_id",
            "item_id",
            "history_item_sid",
            "item_sid",
        },
    )
    records.extend(
        _sid_prediction_record(len(records) + index, row)
        for index, row in enumerate(validation_rows)
    )
    metadata: dict[str, ReplayRecord] = {}
    for record in records:
        metadata[record.prompt] = record
    return metadata


def sample_replay_records(
    records: Sequence[ReplayRecord], *, sample: int, seed: int
) -> tuple[ReplayRecord, ...]:
    """Take one deterministic sample without changing task contents."""

    if sample <= 0:
        raise ValueError("sample must be positive")
    if sample > len(records):
        raise ValueError(f"Cannot sample {sample} records from {len(records)}")
    indices = random.Random(seed).sample(range(len(records)), sample)
    return tuple(records[index] for index in indices)


def _validate_cuda_device(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Candidate generation requires an available CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Candidate generation requires BF16-capable CUDA")


def generate_candidate_groups(
    *,
    records: Sequence[ReplayRecord],
    model_path: Path,
    info_file: Path,
    output_file: Path,
    device: str,
    prompt_batch_size: int,
    num_generations: int,
    max_prompt_length: int,
    max_completion_length: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    """Generate and cache legal candidates without optimizer state."""

    if not records:
        raise ValueError("records cannot be empty")
    if prompt_batch_size <= 0:
        raise ValueError("prompt_batch_size must be positive")
    if output_file.exists():
        raise FileExistsError(output_file)
    torch_device = torch.device(device)
    _validate_cuda_device(torch_device)
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats(torch_device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(torch_device)
    model.eval()
    catalog = load_sid_catalog(info_file, tokenizer)
    legal_sids = frozenset(catalog.semantic_ids)
    generator = GRPORolloutGenerator(
        tokenizer=tokenizer,
        catalog=catalog,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        temperature=temperature,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    candidate_count = 0
    valid_candidate_count = 0
    started_at = time.perf_counter()
    with output_file.open("x", encoding="utf-8", newline="\n") as handle:
        for start in tqdm(
            range(0, len(records), prompt_batch_size),
            desc="Generating replay candidates",
        ):
            batch = records[start : start + prompt_batch_size]
            repeated_prompts = [
                record.prompt
                for record in batch
                for _ in range(num_generations)
            ]
            rollout = generator.generate(
                model=model,
                repeated_prompts=repeated_prompts,
                device=torch_device,
            )
            candidate_count += rollout.candidate_count
            valid_candidate_count += rollout.valid_candidate_count
            for batch_index, record in enumerate(batch):
                group_start = batch_index * num_generations
                candidates = tuple(
                    completion.strip('\n" ')
                    for completion in rollout.completions[
                        group_start : group_start + num_generations
                    ]
                )
                unknown = sorted(set(candidates) - legal_sids)
                if unknown:
                    raise ValueError(f"Generated unknown SIDs: {unknown[:5]}")
                cached = CandidateGroup(
                    source_index=record.source_index,
                    task=record.task,
                    target_sid=record.target_sid,
                    history_item_ids=record.history_item_ids,
                    target_item_id=record.target_item_id,
                    candidate_sids=candidates,
                )
                handle.write(json.dumps(asdict(cached), ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - started_at
    generation_statistics = {
        "sample_count": len(records),
        "candidate_count": candidate_count,
        "valid_candidate_count": valid_candidate_count,
        "valid_candidate_rate": valid_candidate_count / candidate_count,
        "elapsed_seconds": round(elapsed, 3),
        "samples_per_second": round(len(records) / elapsed, 3),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(torch_device) / 1024**3,
            3,
        ),
    }
    del model
    torch.cuda.empty_cache()
    return generation_statistics


def load_candidate_groups(candidate_file: Path) -> tuple[CandidateGroup, ...]:
    """Load and validate the compact JSONL candidate cache."""

    if not candidate_file.is_file():
        raise FileNotFoundError(candidate_file)
    groups: list[CandidateGroup] = []
    with candidate_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            try:
                group = CandidateGroup(
                    source_index=int(value["source_index"]),
                    task=str(value["task"]),
                    target_sid=str(value["target_sid"]),
                    history_item_ids=tuple(
                        int(item_id) for item_id in value["history_item_ids"]
                    ),
                    target_item_id=(
                        None
                        if value["target_item_id"] is None
                        else int(value["target_item_id"])
                    ),
                    candidate_sids=tuple(
                        str(semantic_id) for semantic_id in value["candidate_sids"]
                    ),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid candidate group at {candidate_file}:{line_number}"
                ) from error
            if group.task in COLLABORATIVE_TASKS and (
                not group.history_item_ids or group.target_item_id is None
            ):
                raise ValueError(
                    f"Collaborative task lacks Item IDs at line {line_number}"
                )
            if not group.candidate_sids:
                raise ValueError(f"Empty candidate group at line {line_number}")
            groups.append(group)
    if not groups:
        raise ValueError(f"No candidate groups found in {candidate_file}")
    candidate_counts = {len(group.candidate_sids) for group in groups}
    if len(candidate_counts) != 1:
        raise ValueError("All cached candidate groups must have equal size")
    return tuple(groups)


def load_sid_to_item_ids(index_file: Path) -> dict[str, tuple[int, ...]]:
    """Load the complete SID-to-Item-ID relation used by the teacher."""

    with index_file.open("r", encoding="utf-8") as handle:
        indices = json.load(handle)
    if not isinstance(indices, dict):
        raise TypeError("index_file must contain a JSON object")
    mapping: dict[str, list[int]] = {}
    for raw_item_id, tokens in indices.items():
        if not isinstance(tokens, list) or len(tokens) < 3:
            raise ValueError(f"Invalid SID tokens for item {raw_item_id!r}")
        semantic_id = "".join(str(token) for token in tokens[:3])
        mapping.setdefault(semantic_id, []).append(int(raw_item_id))
    return {
        semantic_id: tuple(item_ids)
        for semantic_id, item_ids in mapping.items()
    }


class CollaborativeScorer:
    """Score SID candidates with one frozen Item-ID SASRec teacher."""

    def __init__(
        self,
        *,
        model: SASRec,
        sid_to_item_ids: dict[str, tuple[int, ...]],
        device: torch.device,
    ) -> None:
        self.model = model
        self.sid_to_item_ids = sid_to_item_ids
        self.device = device
        maximum_item_id = max(
            item_id
            for item_ids in sid_to_item_ids.values()
            for item_id in item_ids
        )
        if maximum_item_id >= model.config.num_items:
            raise ValueError("SID catalog exceeds SASRec item vocabulary")

    def _sid_score(self, logits: torch.Tensor, semantic_id: str) -> float:
        try:
            item_ids = self.sid_to_item_ids[semantic_id]
        except KeyError as error:
            raise KeyError(f"SID missing from Item mapping: {semantic_id}") from error
        selected = logits[
            torch.tensor(item_ids, dtype=torch.long, device=self.device)
        ]
        return float(
            (torch.logsumexp(selected, dim=0) - math.log(len(item_ids))).item()
        )

    @torch.inference_mode()
    def score_group(
        self, group: CandidateGroup
    ) -> tuple[tuple[float, ...], float] | None:
        if group.task not in COLLABORATIVE_TASKS:
            return None
        history = group.history_item_ids[-self.model.config.max_sequence_length :]
        encoded = torch.zeros(
            (1, self.model.config.max_sequence_length),
            dtype=torch.long,
            device=self.device,
        )
        encoded[0, : len(history)] = (
            torch.tensor(history, dtype=torch.long, device=self.device) + 1
        )
        logits = self.model(encoded)[0]
        candidate_scores = tuple(
            self._sid_score(logits, semantic_id)
            for semantic_id in group.candidate_sids
        )
        target_score = self._sid_score(logits, group.target_sid)
        return candidate_scores, target_score


def _sample_standard_deviation(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _reward_summary(
    analyzed: Sequence[_AnalyzedGroup],
    *,
    dense_weight: float | None,
) -> dict[str, Any]:
    tolerance = 1e-12
    reward_groups = [
        (
            row.components.official
            if dense_weight is None
            else row.components.fused(dense_weight)
        )
        for row in analyzed
    ]
    standard_deviations = [
        _sample_standard_deviation(rewards) for rewards in reward_groups
    ]
    target_groups = [
        (row, rewards)
        for row, rewards in zip(analyzed, reward_groups)
        if row.group.target_sid in row.group.candidate_sids
    ]
    target_top_count = 0
    for row, rewards in target_groups:
        target_reward = max(
            reward
            for semantic_id, reward in zip(row.group.candidate_sids, rewards)
            if semantic_id == row.group.target_sid
        )
        if target_reward >= max(rewards) - tolerance:
            target_top_count += 1
    flattened = [reward for rewards in reward_groups for reward in rewards]
    return {
        "group_count": len(reward_groups),
        "finite": all(math.isfinite(value) for value in flattened),
        "zero_reward_group_count": sum(
            all(abs(value) <= tolerance for value in rewards)
            for rewards in reward_groups
        ),
        "zero_reward_group_rate": sum(
            all(abs(value) <= tolerance for value in rewards)
            for rewards in reward_groups
        )
        / len(reward_groups),
        "zero_advantage_group_count": sum(
            deviation <= tolerance for deviation in standard_deviations
        ),
        "zero_advantage_group_rate": sum(
            deviation <= tolerance for deviation in standard_deviations
        )
        / len(reward_groups),
        "mean_reward_std": statistics.fmean(standard_deviations),
        "reward_min": min(flattened),
        "reward_mean": statistics.fmean(flattened),
        "reward_max": max(flattened),
        "target_present_group_count": len(target_groups),
        "exact_target_top_reward_rate": (
            target_top_count / len(target_groups) if target_groups else None
        ),
    }


def _gate_summary(analyzed: Sequence[_AnalyzedGroup]) -> dict[str, Any]:
    eligible = [
        row for row in analyzed if row.group.task in COLLABORATIVE_TASKS
    ]
    gates = [row.components.gate for row in eligible]
    ranks = [
        row.components.target_collaborative_rank
        for row in eligible
        if row.components.target_collaborative_rank is not None
    ]
    if not gates:
        return {"eligible_group_count": 0}
    sorted_gates = sorted(gates)

    def percentile(fraction: float) -> float:
        index = round((len(sorted_gates) - 1) * fraction)
        return sorted_gates[index]

    return {
        "eligible_group_count": len(eligible),
        "gate_min": min(gates),
        "gate_mean": statistics.fmean(gates),
        "gate_median": statistics.median(gates),
        "gate_p25": percentile(0.25),
        "gate_p75": percentile(0.75),
        "gate_max": max(gates),
        "gate_zero_rate": sum(gate <= 1e-12 for gate in gates) / len(gates),
        "gate_high_rate": sum(gate >= 0.5 for gate in gates) / len(gates),
        "target_rank_mean": statistics.fmean(int(rank) for rank in ranks),
        "target_rank_median": statistics.median(int(rank) for rank in ranks),
    }


def _analyze_subset(
    analyzed: Sequence[_AnalyzedGroup], lambdas: Sequence[float]
) -> dict[str, Any]:
    return {
        "group_count": len(analyzed),
        "target_in_candidates_rate": sum(
            row.group.target_sid in row.group.candidate_sids for row in analyzed
        )
        / len(analyzed),
        "mean_unique_candidates": statistics.fmean(
            len(set(row.group.candidate_sids)) for row in analyzed
        ),
        "baseline": _reward_summary(analyzed, dense_weight=None),
        "cgrf_h": {
            str(dense_weight): _reward_summary(
                analyzed, dense_weight=dense_weight
            )
            for dense_weight in lambdas
        },
        "teacher_gate": _gate_summary(analyzed),
    }


def analyze_candidate_groups(
    *,
    groups: Sequence[CandidateGroup],
    sasrec_checkpoint: Path,
    index_file: Path,
    lambdas: Sequence[float],
    device: str,
) -> dict[str, Any]:
    """Replay all rewards over one immutable set of candidate groups."""

    if not groups:
        raise ValueError("groups cannot be empty")
    if not lambdas or any(value < 0.0 for value in lambdas):
        raise ValueError("lambdas must contain non-negative values")
    torch_device = torch.device(device)
    model, checkpoint = load_sasrec_checkpoint(
        sasrec_checkpoint,
        device=torch_device,
    )
    sid_to_item_ids = load_sid_to_item_ids(index_file)
    known_sids = frozenset(sid_to_item_ids)
    for group in groups:
        unknown_sids = (
            {group.target_sid, *group.candidate_sids} - known_sids
        )
        if unknown_sids:
            raise ValueError(
                f"Candidate cache contains unknown SIDs: {sorted(unknown_sids)[:5]}"
            )
        if group.target_item_id is not None and group.target_item_id not in (
            sid_to_item_ids[group.target_sid]
        ):
            raise ValueError(
                f"Target Item {group.target_item_id} does not belong to "
                f"SID {group.target_sid}"
            )
    scorer = CollaborativeScorer(
        model=model,
        sid_to_item_ids=sid_to_item_ids,
        device=torch_device,
    )

    analyzed: list[_AnalyzedGroup] = []
    for group in tqdm(groups, desc="Replaying rewards"):
        collaborative = scorer.score_group(group)
        components = compute_group_reward_components(
            candidate_sids=group.candidate_sids,
            target_sid=group.target_sid,
            collaborative_scores=(
                None if collaborative is None else collaborative[0]
            ),
            target_collaborative_score=(
                None if collaborative is None else collaborative[1]
            ),
        )
        analyzed.append(_AnalyzedGroup(group, components))

    by_task: defaultdict[str, list[_AnalyzedGroup]] = defaultdict(list)
    for row in analyzed:
        by_task[row.group.task].append(row)
    candidate_count = sum(len(group.candidate_sids) for group in groups)
    return {
        "candidate_groups": len(groups),
        "candidate_count": candidate_count,
        "valid_candidate_rate": 1.0,
        "lambdas": list(lambdas),
        "sasrec": {
            "checkpoint": str(sasrec_checkpoint),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_validation": checkpoint.get("best_validation"),
            "num_items": model.config.num_items,
        },
        "catalog": {
            "item_count": sum(len(items) for items in sid_to_item_ids.values()),
            "unique_sid_count": len(sid_to_item_ids),
        },
        "overall": _analyze_subset(analyzed, lambdas),
        "tasks": {
            task: _analyze_subset(rows, lambdas)
            for task, rows in sorted(by_task.items())
        },
    }


def write_analysis(value: dict[str, Any], output_file: Path) -> None:
    """Write one reviewable JSON result without overwriting an experiment."""

    if output_file.exists():
        raise FileExistsError(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
