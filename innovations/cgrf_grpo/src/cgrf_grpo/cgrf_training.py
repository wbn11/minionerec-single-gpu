"""Independent CGRF GRPO training built on the baseline trainer."""

from __future__ import annotations

# The experiment deliberately reuses the baseline's construction helpers so
# model loading, TRL arguments, generation, KL, and optimizer settings remain
# identical. Only reward construction is replaced in this module.
# pyright: reportPrivateUsage=false

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from trl.models.utils import unwrap_model_for_generation

from minionerec.data.grpo_datasets import (
    GRPODatasetSizes,
    GRPORecord,
    build_grpo_datasets,
)
from minionerec.generation.grpo_generation import (
    GRPORolloutBatch,
    GRPORolloutGenerator,
)
from minionerec.generation.sft_generation import load_sid_catalog
from minionerec.rewards.ranking_grpo import (
    RankingGRPORewards,
    group_normalized_advantages,
)
from minionerec.training import grpo_training as baseline_grpo
from minionerec.training.grpo_loss import completion_token_log_probs

from .reward_fusion import GroupRewardComponents, compute_group_reward_components
from .reward_replay import (
    COLLABORATIVE_TASKS,
    CandidateGroup,
    CollaborativeScorer,
    ReplayRecord,
    build_prompt_metadata,
    load_sid_to_item_ids,
)
from .sasrec import load_sasrec_checkpoint


@dataclass(frozen=True)
class CGRFTrainingConfig:
    """Baseline GRPO configuration plus CGRF-only inputs."""

    grpo: baseline_grpo.GRPOTrainingConfig
    sasrec_checkpoint: Path
    dense_weight: float = 0.1


@dataclass(frozen=True)
class CGRFComponents:
    """Objects prepared for CGRF training without starting optimization."""

    model: Any
    reference_model: Any
    tokenizer: Any
    trainer: baseline_grpo.SingleGPUReReTrainer
    datasets: GRPODatasetSizes
    catalog: dict[str, int]
    teacher: dict[str, Any]
    model_load_count: int


class CGRFTrainer(baseline_grpo.SingleGPUReReTrainer):
    """Baseline single-GPU trainer with confidence-gated dense rewards."""

    def __init__(
        self,
        *,
        model: nn.Module,
        reference_model: nn.Module,
        tokenizer: Any,
        rollout_generator: GRPORolloutGenerator,
        dataset_bundle: Any,
        args: Any,
        prompt_metadata: dict[str, ReplayRecord],
        collaborative_scorer: CollaborativeScorer,
        dense_weight: float,
    ) -> None:
        if dense_weight < 0.0:
            raise ValueError("dense_weight cannot be negative")
        self.prompt_metadata = prompt_metadata
        self.collaborative_scorer = collaborative_scorer
        self.dense_weight = dense_weight
        super().__init__(
            model=model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            rollout_generator=rollout_generator,
            dataset_bundle=dataset_bundle,
            args=args,
        )

    def _components_for_rollout(
        self,
        *,
        rollout: GRPORolloutBatch,
        repeated_targets: list[str],
    ) -> list[GroupRewardComponents]:
        components: list[GroupRewardComponents] = []
        for group_index, prompt in enumerate(rollout.unique_prompts):
            group_start = group_index * self.num_generations
            group_end = group_start + self.num_generations
            group_targets = repeated_targets[group_start:group_end]
            normalized_target = group_targets[0].strip('\n" ')
            if any(
                target.strip('\n" ') != normalized_target
                for target in group_targets[1:]
            ):
                raise ValueError("one GRPO group contains different targets")
            try:
                metadata = self.prompt_metadata[prompt]
            except KeyError as error:
                raise KeyError("GRPO prompt is missing CGRF metadata") from error

            group = CandidateGroup(
                source_index=metadata.source_index,
                task=metadata.task,
                target_sid=normalized_target,
                history_item_ids=metadata.history_item_ids,
                target_item_id=metadata.target_item_id,
                candidate_sids=tuple(
                    completion.strip('\n" ')
                    for completion in rollout.completions[group_start:group_end]
                ),
            )
            collaborative = self.collaborative_scorer.score_group(group)
            components.append(
                compute_group_reward_components(
                    candidate_sids=group.candidate_sids,
                    target_sid=normalized_target,
                    collaborative_scores=(
                        None if collaborative is None else collaborative[0]
                    ),
                    target_collaborative_score=(
                        None if collaborative is None else collaborative[1]
                    ),
                )
            )
        return components

    def _record_cgrf_metrics(
        self,
        *,
        components: list[GroupRewardComponents],
        group_sample_stds: tuple[float, ...],
    ) -> None:
        flattened_hierarchical = [
            value for group in components for value in group.hierarchical
        ]
        flattened_collaborative = [
            value for group in components for value in group.collaborative
        ]
        flattened_dense = [
            fused - official
            for group in components
            for fused, official in zip(
                group.fused(self.dense_weight),
                group.official,
            )
        ]
        official = [
            value for group in components for value in group.official
        ]
        gates = [
            group.gate
            for group in components
            if group.target_collaborative_rank is not None
        ]
        target_ranks = [
            float(group.target_collaborative_rank)
            for group in components
            if group.target_collaborative_rank is not None
        ]

        self._grpo_metrics["reward/official"].append(
            sum(official) / len(official)
        )
        self._grpo_metrics["reward/dense"].append(
            sum(flattened_dense) / len(flattened_dense)
        )
        self._grpo_metrics["rewards/hierarchical"].append(
            sum(flattened_hierarchical) / len(flattened_hierarchical)
        )
        self._grpo_metrics["rewards/collaborative"].append(
            sum(flattened_collaborative) / len(flattened_collaborative)
        )
        self._grpo_metrics["zero_advantage_group_rate"].append(
            sum(value <= 1e-12 for value in group_sample_stds)
            / len(group_sample_stds)
        )
        if gates:
            self._grpo_metrics["teacher_gate"].append(sum(gates) / len(gates))
            self._grpo_metrics["target_collaborative_rank"].append(
                sum(target_ranks) / len(target_ranks)
            )

    def _prepare_inputs(self, inputs):
        if not isinstance(inputs, list) or not inputs:
            raise TypeError("GRPO batches must be non-empty lists of records")
        records = cast(list[GRPORecord], inputs)
        prompts = [record["prompt"] for record in records]
        targets = self._targets_for_prompts(prompts)
        device = self.accelerator.device

        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
        ) as unwrapped_model:
            rollout = self.rollout_generator.generate(
                model=unwrapped_model,
                repeated_prompts=prompts,
                device=device,
            )

        input_ids = torch.cat(
            [rollout.prompt_ids, rollout.completion_ids],
            dim=1,
        )
        attention_mask = torch.cat(
            [rollout.prompt_mask, rollout.completion_mask],
            dim=1,
        )
        with torch.inference_mode():
            reference_log_probs = completion_token_log_probs(
                model=self.reference_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                completion_length=rollout.completion_ids.size(1),
            )

        components = self._components_for_rollout(
            rollout=rollout,
            repeated_targets=targets,
        )
        exact = tuple(value for group in components for value in group.exact)
        ranking = tuple(value for group in components for value in group.ranking)
        total = tuple(
            value
            for group in components
            for value in group.fused(self.dense_weight)
        )
        reward_tensor = torch.tensor(total, dtype=torch.float32)
        advantages, group_means, group_sample_stds = group_normalized_advantages(
            reward_tensor,
            num_generations=self.num_generations,
        )
        rewards = RankingGRPORewards(
            exact=exact,
            ranking=ranking,
            total=total,
            advantages=tuple(float(value) for value in advantages.tolist()),
            group_means=tuple(float(value) for value in group_means.tolist()),
            group_sample_stds=tuple(
                float(value) for value in group_sample_stds.tolist()
            ),
        )
        self._record_rollout_metrics(rollout=rollout, rewards=rewards)
        self._record_cgrf_metrics(
            components=components,
            group_sample_stds=rewards.group_sample_stds,
        )
        return {
            "prompt_ids": rollout.prompt_ids,
            "prompt_mask": rollout.prompt_mask,
            "completion_ids": rollout.completion_ids,
            "completion_mask": rollout.completion_mask,
            "reference_log_probs": reference_log_probs,
            "advantages": advantages.to(device=device),
        }


def _validate_experiment_config(config: CGRFTrainingConfig) -> None:
    baseline_grpo._validate_config(config.grpo)
    if not config.sasrec_checkpoint.is_file():
        raise FileNotFoundError(config.sasrec_checkpoint)
    if config.dense_weight < 0.0:
        raise ValueError("dense_weight cannot be negative")


def build_cgrf_components(config: CGRFTrainingConfig) -> CGRFComponents:
    """Build CGRF while preserving every non-reward baseline component."""

    _validate_experiment_config(config)
    base = config.grpo
    baseline_grpo._set_reproducible_seed(base.seed)
    datasets = build_grpo_datasets(
        train_file=base.train_file,
        valid_file=base.valid_file,
        item_file=base.item_file,
        index_file=base.index_file,
        sequence_title_sample=base.sequence_title_sample,
    )
    prompt_metadata = build_prompt_metadata(
        train_file=base.train_file,
        valid_file=base.valid_file,
        item_file=base.item_file,
        index_file=base.index_file,
        sequence_title_sample=base.sequence_title_sample,
    )
    missing_metadata = sorted(
        {record["prompt"] for record in (*datasets.train, *datasets.validation)}
        - prompt_metadata.keys()
    )
    if missing_metadata:
        raise ValueError("CGRF metadata does not cover every GRPO prompt")
    collaborative_target_mismatches = [
        prompt
        for prompt, metadata in prompt_metadata.items()
        if prompt in datasets.prompt_to_history
        and metadata.task in COLLABORATIVE_TASKS
        and metadata.target_sid
        != datasets.history_to_target[
            datasets.prompt_to_history[prompt]
        ].strip()
    ]
    if collaborative_target_mismatches:
        raise ValueError(
            "CGRF collaborative metadata does not match baseline targets"
        )

    policy_model, reference_model, tokenizer = (
        baseline_grpo._load_models_and_tokenizer(base)
    )
    baseline_grpo._validate_model_vocabulary(policy_model, tokenizer)
    baseline_grpo._validate_model_vocabulary(reference_model, tokenizer)
    catalog = load_sid_catalog(base.info_file, tokenizer)
    rollout_generator = GRPORolloutGenerator(
        tokenizer=tokenizer,
        catalog=catalog,
        num_generations=base.num_generations,
        max_prompt_length=base.max_prompt_length,
        max_completion_length=base.max_completion_length,
        temperature=base.temperature,
        length_penalty=base.length_penalty,
    )

    device = torch.device("cuda:0")
    teacher_model, teacher_checkpoint = load_sasrec_checkpoint(
        config.sasrec_checkpoint,
        device=device,
    )
    sid_to_item_ids = load_sid_to_item_ids(base.index_file)
    collaborative_scorer = CollaborativeScorer(
        model=teacher_model,
        sid_to_item_ids=sid_to_item_ids,
        device=device,
    )
    trainer = CGRFTrainer(
        model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        rollout_generator=rollout_generator,
        dataset_bundle=datasets,
        args=baseline_grpo._training_arguments(base),
        prompt_metadata=prompt_metadata,
        collaborative_scorer=collaborative_scorer,
        dense_weight=config.dense_weight,
    )
    return CGRFComponents(
        model=trainer.model,
        reference_model=trainer.reference_model,
        tokenizer=tokenizer,
        trainer=trainer,
        datasets=datasets.sizes,
        catalog={
            "item_rows": catalog.item_rows,
            "unique_sids": catalog.unique_sid_count,
            "sid_collision_excess": catalog.collision_excess,
        },
        teacher={
            "checkpoint": str(config.sasrec_checkpoint),
            "best_epoch": teacher_checkpoint.get("best_epoch"),
            "best_validation": teacher_checkpoint.get("best_validation"),
            "num_items": teacher_model.config.num_items,
            "sid_aggregation": "logmeanexp",
        },
        model_load_count=3,
    )
