"""Constrained stochastic beam rollouts for official MiniOneRec GRPO."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
)

from minionerec.generation.sft_generation import (
    SidCatalog,
    SidTrie,
    SidTrieLogitsProcessor,
)


OFFICIAL_NUM_GENERATIONS = 16
OFFICIAL_MAX_PROMPT_LENGTH = 512
OFFICIAL_MAX_COMPLETION_LENGTH = 128
OFFICIAL_TEMPERATURE = 1.0
OFFICIAL_LENGTH_PENALTY = 0.0


@dataclass(frozen=True)
class GRPORolloutBatch:
    """Generated candidates and masks consumed by the GRPO loss."""

    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    completions: tuple[str, ...]
    unique_prompts: tuple[str, ...]
    valid_candidate_count: int

    @property
    def candidate_count(self) -> int:
        return len(self.completions)

    @property
    def valid_candidate_rate(self) -> float:
        return self.valid_candidate_count / self.candidate_count


def _unique_prompt_groups(
    repeated_prompts: Sequence[str], num_generations: int
) -> tuple[str, ...]:
    if num_generations < 2:
        raise ValueError("num_generations must be at least 2")
    if not repeated_prompts:
        raise ValueError("repeated_prompts cannot be empty")
    if len(repeated_prompts) % num_generations != 0:
        raise ValueError(
            f"{len(repeated_prompts)} prompts cannot be divided into groups "
            f"of {num_generations}"
        )

    unique_prompts: list[str] = []
    for group_start in range(0, len(repeated_prompts), num_generations):
        group = repeated_prompts[
            group_start : group_start + num_generations
        ]
        if not all(isinstance(prompt, str) for prompt in group):
            raise TypeError("all prompts must be strings")
        first_prompt = group[0]
        if any(prompt != first_prompt for prompt in group[1:]):
            raise ValueError(
                "each GRPO group must contain repetitions of one prompt"
            )
        unique_prompts.append(first_prompt)
    return tuple(unique_prompts)


def _completion_mask(
    completion_ids: torch.Tensor, eos_token_id: int
) -> torch.Tensor:
    if completion_ids.ndim != 2:
        raise ValueError("completion_ids must be a two-dimensional tensor")
    is_eos = completion_ids == eos_token_id
    eos_indices = torch.full(
        (is_eos.size(0),),
        is_eos.size(1),
        dtype=torch.long,
        device=completion_ids.device,
    )
    rows_with_eos = is_eos.any(dim=1)
    eos_indices[rows_with_eos] = is_eos.to(torch.int64).argmax(dim=1)[
        rows_with_eos
    ]
    sequence_indices = torch.arange(
        is_eos.size(1), device=completion_ids.device
    ).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_indices.unsqueeze(1)).to(torch.int64)


def _path_through_eos(
    token_ids: Sequence[int], eos_token_id: int
) -> tuple[int, ...]:
    path: list[int] = []
    for token_id in token_ids:
        path.append(int(token_id))
        if int(token_id) == eos_token_id:
            break
    return tuple(path)


class GRPORolloutGenerator:
    """Generate G legal SID candidates for every repeated prompt group."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        catalog: SidCatalog,
        num_generations: int = OFFICIAL_NUM_GENERATIONS,
        max_prompt_length: int = OFFICIAL_MAX_PROMPT_LENGTH,
        max_completion_length: int = OFFICIAL_MAX_COMPLETION_LENGTH,
        temperature: float = OFFICIAL_TEMPERATURE,
        length_penalty: float = OFFICIAL_LENGTH_PENALTY,
    ) -> None:
        if num_generations < 2:
            raise ValueError("num_generations must be at least 2")
        if max_prompt_length < 1:
            raise ValueError("max_prompt_length must be positive")
        if max_completion_length < 1:
            raise ValueError("max_completion_length must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")

        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.tokenizer = tokenizer
        self.catalog = catalog
        self.num_generations = num_generations
        self.max_prompt_length = max_prompt_length
        self.temperature = temperature
        self.trie = SidTrie(tuple(catalog.token_path_to_sid))
        self.generation_config = GenerationConfig(
            max_new_tokens=max_completion_length,
            length_penalty=length_penalty,
            num_beams=num_generations,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_k=None,
            top_p=None,
            temperature=temperature,
            repetition_penalty=1.0,
            do_sample=True,
        )

    def generate(
        self,
        *,
        model: Any,
        repeated_prompts: Sequence[str],
        device: torch.device,
    ) -> GRPORolloutBatch:
        """Generate and validate one rollout batch without optimizer updates."""

        unique_prompts = _unique_prompt_groups(
            repeated_prompts,
            self.num_generations,
        )
        prompt_inputs = self.tokenizer(
            list(unique_prompts),
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        input_ids = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
        attention_mask = prompt_inputs["attention_mask"][
            :, -self.max_prompt_length :
        ]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        prompt_width = input_ids.size(1)

        trie_processor = SidTrieLogitsProcessor(
            self.trie,
            prompt_width,
            self.tokenizer.eos_token_id,
        )
        logits_processors = LogitsProcessorList([trie_processor])
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                logits_processor=logits_processors,
                use_model_defaults=False,
            )

        expected_candidates = len(unique_prompts) * self.num_generations
        if sequences.size(0) != expected_candidates:
            raise RuntimeError(
                f"Expected {expected_candidates} candidates, got "
                f"{sequences.size(0)}"
            )

        prompt_ids = sequences[:, :prompt_width]
        completion_ids = sequences[:, prompt_width:]
        prompt_mask = attention_mask.repeat_interleave(
            self.num_generations,
            dim=0,
        )
        completion_mask = _completion_mask(
            completion_ids,
            self.tokenizer.eos_token_id,
        )
        completions = tuple(
            self.tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )
        )

        legal_paths = self.catalog.token_path_to_sid
        candidate_paths = (
            _path_through_eos(token_ids, self.tokenizer.eos_token_id)
            for token_ids in completion_ids.detach().cpu().tolist()
        )
        valid_candidate_count = sum(
            path in legal_paths for path in candidate_paths
        )
        return GRPORolloutBatch(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completions=completions,
            unique_prompts=unique_prompts,
            valid_candidate_count=valid_candidate_count,
        )
