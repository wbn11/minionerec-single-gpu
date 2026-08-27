"""Top-K ranking metrics used by the fixed MiniOneRec evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence


OFFICIAL_TOP_K = (1, 3, 5, 10, 20, 50)


def compute_ranking_metrics(
    targets: Sequence[str],
    predictions: Sequence[Sequence[str]],
    top_k: Sequence[int] = OFFICIAL_TOP_K,
) -> dict[str, float]:
    """Compute official single-target HR@K and NDCG@K."""

    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    if not targets:
        raise ValueError("cannot evaluate an empty dataset")

    cutoffs = tuple(int(value) for value in top_k)
    if not cutoffs or any(value <= 0 for value in cutoffs):
        raise ValueError("top_k must contain positive integers")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("top_k values must be unique")

    hit_sums = {cutoff: 0.0 for cutoff in cutoffs}
    ndcg_sums = {cutoff: 0.0 for cutoff in cutoffs}

    for target, candidates in zip(targets, predictions, strict=True):
        try:
            zero_based_rank = candidates.index(target)
        except ValueError:
            continue

        one_based_rank = zero_based_rank + 1
        discount = 1.0 / math.log2(one_based_rank + 1)
        for cutoff in cutoffs:
            if one_based_rank <= cutoff:
                hit_sums[cutoff] += 1.0
                ndcg_sums[cutoff] += discount

    sample_count = len(targets)
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        metrics[f"HR@{cutoff}"] = hit_sums[cutoff] / sample_count
        metrics[f"NDCG@{cutoff}"] = ndcg_sums[cutoff] / sample_count
    return metrics
