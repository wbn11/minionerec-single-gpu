"""Reward components for confidence-gated collaborative GRPO."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from minionerec.rewards.ranking_grpo import compute_ranking_grpo_rewards


_SID_TOKEN_PATTERN = re.compile(r"<[^<>]+>")


@dataclass(frozen=True)
class GroupRewardComponents:
    """Official and dense reward values for one candidate group."""

    exact: tuple[float, ...]
    ranking: tuple[float, ...]
    hierarchical: tuple[float, ...]
    collaborative: tuple[float, ...]
    gate: float
    target_collaborative_rank: int | None

    @property
    def official(self) -> tuple[float, ...]:
        """Return the unchanged fixed-commit reward."""

        return tuple(
            exact + ranking
            for exact, ranking in zip(self.exact, self.ranking)
        )

    def has_informative_official_reward(
        self, tolerance: float = 1e-12
    ) -> bool:
        """Return whether the official reward distinguishes group candidates."""

        official = self.official
        return max(official) - min(official) > tolerance

    def fused(
        self,
        dense_weight: float,
        informative_dense_weight: float | None = None,
    ) -> tuple[float, ...]:
        """Fuse dense supervision with optional official-aware attenuation."""

        if dense_weight < 0.0:
            raise ValueError("dense_weight cannot be negative")
        if (
            informative_dense_weight is not None
            and not 0.0 <= informative_dense_weight <= dense_weight
        ):
            raise ValueError(
                "informative_dense_weight must be between zero and "
                "dense_weight"
            )
        effective_dense_weight = dense_weight
        if (
            informative_dense_weight is not None
            and self.has_informative_official_reward()
        ):
            effective_dense_weight = informative_dense_weight
        return tuple(
            official
            + effective_dense_weight
            * (
                self.gate * collaborative
                + (1.0 - self.gate) * hierarchical
            )
            for official, hierarchical, collaborative in zip(
                self.official,
                self.hierarchical,
                self.collaborative,
            )
        )


def split_sid_tokens(semantic_id: str) -> tuple[str, ...]:
    """Split one SID while rejecting malformed or incomplete values."""

    normalized = semantic_id.strip('\n" ')
    tokens = tuple(_SID_TOKEN_PATTERN.findall(normalized))
    if len(tokens) != 3 or "".join(tokens) != normalized:
        raise ValueError(f"Expected a three-token SID, got {semantic_id!r}")
    return tokens


def hierarchical_reward(candidate_sid: str, target_sid: str) -> float:
    """Return the HEPO-style three-level prefix reward."""

    candidate_tokens = split_sid_tokens(candidate_sid)
    target_tokens = split_sid_tokens(target_sid)
    common_prefix = 0
    for candidate_token, target_token in zip(candidate_tokens, target_tokens):
        if candidate_token != target_token:
            break
        common_prefix += 1
    return (0.0, 0.2, 0.5, 1.0)[common_prefix]


def rank_percentile_rewards(scores: Sequence[float]) -> tuple[float, ...]:
    """Convert arbitrary logits to deterministic scale-free group ranks."""

    if len(scores) < 2:
        raise ValueError("at least two collaborative scores are required")
    if not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("collaborative scores must be finite")
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), index),
    )
    rewards = [0.0] * len(scores)
    denominator = len(scores) - 1
    for rank, index in enumerate(ranked_indices):
        rewards[index] = 1.0 - rank / denominator
    return tuple(rewards)


def target_rank_gate(
    *,
    candidate_sids: Sequence[str],
    candidate_scores: Sequence[float],
    target_sid: str,
    target_score: float,
) -> tuple[float, int]:
    """Measure teacher confidence from target rank in the on-policy group."""

    if len(candidate_sids) != len(candidate_scores):
        raise ValueError("candidate SIDs and scores must have equal lengths")
    if not math.isfinite(target_score):
        raise ValueError("target collaborative score must be finite")

    normalized_target = target_sid.strip('\n" ')
    comparison_scores = [float(target_score)]
    seen = {normalized_target}
    for semantic_id, score in zip(candidate_sids, candidate_scores):
        normalized_sid = semantic_id.strip('\n" ')
        if normalized_sid in seen:
            continue
        seen.add(normalized_sid)
        comparison_scores.append(float(score))

    target_rank = 1 + sum(
        score > float(target_score) for score in comparison_scores[1:]
    )
    comparison_count = len(comparison_scores)
    if comparison_count == 1:
        return 1.0, target_rank

    reciprocal_discount = 1.0 / math.log2(target_rank + 1.0)
    minimum_discount = 1.0 / math.log2(comparison_count + 1.0)
    gate = (reciprocal_discount - minimum_discount) / (
        1.0 - minimum_discount
    )
    return min(1.0, max(0.0, gate)), target_rank


def compute_group_reward_components(
    *,
    candidate_sids: Sequence[str],
    target_sid: str,
    collaborative_scores: Sequence[float] | None = None,
    target_collaborative_score: float | None = None,
) -> GroupRewardComponents:
    """Compute all components without choosing the dense reward weight."""

    if len(candidate_sids) < 2:
        raise ValueError("a reward group must contain at least two candidates")
    normalized_candidates = tuple(
        semantic_id.strip('\n" ') for semantic_id in candidate_sids
    )
    normalized_target = target_sid.strip('\n" ')
    targets = (normalized_target,) * len(normalized_candidates)
    official = compute_ranking_grpo_rewards(
        normalized_candidates,
        targets,
        num_generations=len(normalized_candidates),
    )
    hierarchical = tuple(
        hierarchical_reward(candidate, normalized_target)
        for candidate in normalized_candidates
    )

    if collaborative_scores is None:
        if target_collaborative_score is not None:
            raise ValueError("target collaborative score requires candidate scores")
        collaborative = (0.0,) * len(normalized_candidates)
        gate = 0.0
        target_rank = None
    else:
        if target_collaborative_score is None:
            raise ValueError("candidate collaborative scores require a target score")
        if len(collaborative_scores) != len(normalized_candidates):
            raise ValueError("candidate and collaborative score counts must match")
        collaborative = rank_percentile_rewards(collaborative_scores)
        gate, target_rank = target_rank_gate(
            candidate_sids=normalized_candidates,
            candidate_scores=collaborative_scores,
            target_sid=normalized_target,
            target_score=target_collaborative_score,
        )

    return GroupRewardComponents(
        exact=official.exact,
        ranking=official.ranking,
        hierarchical=hierarchical,
        collaborative=collaborative,
        gate=gate,
        target_collaborative_rank=target_rank,
    )
