"""Official exact and ranking rewards for MiniOneRec GRPO."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


OFFICIAL_NUM_GENERATIONS = 16
OFFICIAL_ADVANTAGE_EPSILON = 1e-4


@dataclass(frozen=True)
class RankingGRPORewards:
    """Reward components and group-normalized advantages."""

    exact: tuple[float, ...]
    ranking: tuple[float, ...]
    total: tuple[float, ...]
    advantages: tuple[float, ...]
    group_means: tuple[float, ...]
    group_sample_stds: tuple[float, ...]


def _validate_text_inputs(
    completions: Sequence[str], targets: Sequence[str]
) -> None:
    if len(completions) != len(targets):
        raise ValueError(
            "completions and targets must contain the same number of values"
        )
    if not completions:
        raise ValueError("completions and targets cannot be empty")
    if not all(isinstance(value, str) for value in completions):
        raise TypeError("all completions must be strings")
    if not all(isinstance(value, str) for value in targets):
        raise TypeError("all targets must be strings")


def _validate_groups(total_values: int, num_generations: int) -> None:
    if num_generations < 2:
        raise ValueError("num_generations must be at least 2")
    if total_values % num_generations != 0:
        raise ValueError(
            f"{total_values} values cannot be divided into groups of "
            f"{num_generations}"
        )


def ranking_discount_penalties(
    num_generations: int = OFFICIAL_NUM_GENERATIONS,
) -> tuple[float, ...]:
    """Return the fixed commit's negative normalized rank discounts."""

    _validate_groups(num_generations, num_generations)
    discounts = [1.0 / math.log2(index + 2) for index in range(num_generations)]
    normalizer = sum(discounts)
    return tuple(-discount / normalizer for discount in discounts)


def exact_match_rewards(
    completions: Sequence[str], targets: Sequence[str]
) -> tuple[float, ...]:
    """Reward exact SID matches after the official whitespace cleanup."""

    _validate_text_inputs(completions, targets)
    return tuple(
        1.0
        if completion.strip('\n" ') == target.strip('\n" ')
        else 0.0
        for completion, target in zip(completions, targets)
    )


def ranking_position_rewards(
    completions: Sequence[str],
    targets: Sequence[str],
    num_generations: int = OFFICIAL_NUM_GENERATIONS,
) -> tuple[float, ...]:
    """Apply rank penalties only to groups containing at least one hit."""

    _validate_text_inputs(completions, targets)
    _validate_groups(len(completions), num_generations)
    penalties = ranking_discount_penalties(num_generations)
    rewards: list[float] = []

    for group_start in range(0, len(completions), num_generations):
        group_completions = completions[
            group_start : group_start + num_generations
        ]
        group_targets = targets[group_start : group_start + num_generations]
        matches = [
            completion.strip('\n"') == target.strip('\n"')
            for completion, target in zip(group_completions, group_targets)
        ]
        if any(matches):
            rewards.extend(
                0.0 if matched else penalties[position]
                for position, matched in enumerate(matches)
            )
        else:
            rewards.extend([0.0] * num_generations)

    return tuple(rewards)


def group_normalized_advantages(
    rewards: torch.Tensor,
    num_generations: int = OFFICIAL_NUM_GENERATIONS,
    epsilon: float = OFFICIAL_ADVANTAGE_EPSILON,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize rewards using each group's sample standard deviation."""

    if rewards.ndim != 1:
        raise ValueError("rewards must be a one-dimensional tensor")
    _validate_groups(rewards.numel(), num_generations)
    if not torch.is_floating_point(rewards):
        raise TypeError("rewards must use a floating-point dtype")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    grouped_rewards = rewards.reshape(-1, num_generations)
    group_means = grouped_rewards.mean(dim=1)
    group_sample_stds = grouped_rewards.std(dim=1, correction=1)
    repeated_means = group_means.repeat_interleave(num_generations)
    repeated_stds = group_sample_stds.repeat_interleave(num_generations)
    advantages = (rewards - repeated_means) / (repeated_stds + epsilon)
    return advantages, group_means, group_sample_stds


def compute_ranking_grpo_rewards(
    completions: Sequence[str],
    targets: Sequence[str],
    num_generations: int = OFFICIAL_NUM_GENERATIONS,
) -> RankingGRPORewards:
    """Compute both official reward functions and group advantages."""

    exact = exact_match_rewards(completions, targets)
    ranking = ranking_position_rewards(
        completions,
        targets,
        num_generations=num_generations,
    )
    total = tuple(
        exact_reward + ranking_reward
        for exact_reward, ranking_reward in zip(exact, ranking)
    )
    reward_tensor = torch.tensor(total, dtype=torch.float32)
    advantages, group_means, group_sample_stds = group_normalized_advantages(
        reward_tensor,
        num_generations=num_generations,
    )
    return RankingGRPORewards(
        exact=exact,
        ranking=ranking,
        total=total,
        advantages=tuple(float(value) for value in advantages.tolist()),
        group_means=tuple(float(value) for value in group_means.tolist()),
        group_sample_stds=tuple(
            float(value) for value in group_sample_stds.tolist()
        ),
    )
