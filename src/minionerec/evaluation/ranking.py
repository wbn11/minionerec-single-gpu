"""Top-K ranking metrics used by the fixed MiniOneRec evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


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


def _validated_cutoffs(top_k: Sequence[int]) -> tuple[int, ...]:
    cutoffs = tuple(int(value) for value in top_k)
    if not cutoffs or any(value <= 0 for value in cutoffs):
        raise ValueError("top_k must contain positive integers")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("top_k values must be unique")
    return cutoffs


def _normalized_sid_groups(
    sid_to_item_ids: Mapping[str, Sequence[str]],
) -> tuple[dict[str, tuple[str, ...]], set[str]]:
    groups: dict[str, tuple[str, ...]] = {}
    item_to_sid: dict[str, str] = {}
    for semantic_id, item_ids in sid_to_item_ids.items():
        group = tuple(str(item_id) for item_id in item_ids)
        if not semantic_id or not group or any(not item_id for item_id in group):
            raise ValueError(
                "sid_to_item_ids must contain non-empty SIDs and item IDs"
            )
        if len(set(group)) != len(group):
            raise ValueError(f"SID {semantic_id!r} contains duplicate item IDs")
        groups[semantic_id] = group
        for item_id in group:
            existing_sid = item_to_sid.get(item_id)
            if existing_sid is not None and existing_sid != semantic_id:
                raise ValueError(
                    f"Item ID {item_id!r} belongs to multiple SIDs: "
                    f"{existing_sid!r}, {semantic_id!r}"
                )
            item_to_sid[item_id] = semantic_id
    return groups, set(item_to_sid)


def _target_group_location(
    target_item_id: str,
    predicted_sids: Sequence[str],
    sid_groups: Mapping[str, Sequence[str]],
) -> tuple[int, int] | None:
    expanded_position = 1
    for semantic_id in predicted_sids:
        group = sid_groups.get(semantic_id)
        if group is None:
            # An invalid generated SID occupies one failed recommendation slot.
            expanded_position += 1
            continue
        if target_item_id in group:
            return expanded_position, len(group)
        expanded_position += len(group)
    return None


def _collision_corrected_credit(
    expanded_position: int, group_size: int, cutoff: int
) -> tuple[float, float]:
    items_inside_cutoff = min(
        group_size, max(0, cutoff - expanded_position + 1)
    )
    if items_inside_cutoff == 0:
        return 0.0, 0.0
    hit_credit = items_inside_cutoff / group_size
    ndcg_credit = sum(
        1.0 / math.log2(expanded_position + offset + 1)
        for offset in range(items_inside_cutoff)
    ) / group_size
    return hit_credit, ndcg_credit


def compute_collision_corrected_metrics(
    target_item_ids: Sequence[str],
    predicted_sids: Sequence[Sequence[str]],
    sid_to_item_ids: Mapping[str, Sequence[str]],
    top_k: Sequence[int] = OFFICIAL_TOP_K,
) -> dict[str, float]:
    """Compute CCE ItemHit@K and ItemNDCG@K from ranked SID sequences."""

    if len(target_item_ids) != len(predicted_sids):
        raise ValueError("target_item_ids and predicted_sids must have the same length")
    if not target_item_ids:
        raise ValueError("cannot evaluate an empty dataset")

    cutoffs = _validated_cutoffs(top_k)
    sid_groups, catalog_item_ids = _normalized_sid_groups(sid_to_item_ids)
    missing_targets = sorted(set(target_item_ids) - catalog_item_ids)
    if missing_targets:
        raise ValueError(
            f"Target item IDs missing from SID catalog: {missing_targets[:5]}"
        )

    metric_sums = {cutoff: [0.0, 0.0] for cutoff in cutoffs}

    for target_item_id, candidates in zip(
        target_item_ids, predicted_sids, strict=True
    ):
        location = _target_group_location(target_item_id, candidates, sid_groups)
        if location is None:
            continue
        for cutoff in cutoffs:
            contribution = _collision_corrected_credit(*location, cutoff)
            metric_sums[cutoff][0] += contribution[0]
            metric_sums[cutoff][1] += contribution[1]

    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        metrics[f"ItemHit@{cutoff}"] = metric_sums[cutoff][0] / len(
            target_item_ids
        )
        metrics[f"ItemNDCG@{cutoff}"] = metric_sums[cutoff][1] / len(
            target_item_ids
        )
    return metrics
