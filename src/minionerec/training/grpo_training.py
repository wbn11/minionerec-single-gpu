"""Build the official MiniOneRec GRPO pipeline for one CUDA GPU."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import Dataset, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, set_seed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig

from minionerec.data.grpo_datasets import (
    GRPODatasets,
    GRPODatasetSizes,
    GRPORecord,
    build_grpo_datasets,
)
from minionerec.generation.grpo_generation import (
    GRPORolloutBatch,
    GRPORolloutGenerator,
    OFFICIAL_LENGTH_PENALTY,
    OFFICIAL_MAX_COMPLETION_LENGTH,
    OFFICIAL_MAX_PROMPT_LENGTH,
    OFFICIAL_NUM_GENERATIONS,
    OFFICIAL_TEMPERATURE,
)
from minionerec.generation.sft_generation import load_sid_catalog
from minionerec.rewards.ranking_grpo import (
    RankingGRPORewards,
    compute_ranking_grpo_rewards,
)
from minionerec.training.grpo_loss import (
    OFFICIAL_BETA,
    completion_token_log_probs,
    official_grpo_loss,
)


OFFICIAL_SEQUENCE_TITLE_SAMPLE = 10_000
OFFICIAL_NUM_EPOCHS = 2
OFFICIAL_LEARNING_RATE = 1e-5
OFFICIAL_WARMUP_RATIO = 0.03
OFFICIAL_MAX_GRAD_NORM = 0.3
OFFICIAL_REF_MODEL_MIXUP_ALPHA = 0.6
OFFICIAL_REF_MODEL_SYNC_STEPS = 512


@dataclass(frozen=True)
class GRPOTrainingConfig:
    """Inputs and fixed-commit defaults for single-GPU ranking GRPO."""

    model_path: Path
    train_file: Path
    valid_file: Path
    item_file: Path
    index_file: Path
    info_file: Path
    output_dir: Path
    seed: int = 42
    micro_batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 64
    num_epochs: int = OFFICIAL_NUM_EPOCHS
    learning_rate: float = OFFICIAL_LEARNING_RATE
    num_generations: int = OFFICIAL_NUM_GENERATIONS
    max_prompt_length: int = OFFICIAL_MAX_PROMPT_LENGTH
    max_completion_length: int = OFFICIAL_MAX_COMPLETION_LENGTH
    temperature: float = OFFICIAL_TEMPERATURE
    length_penalty: float = OFFICIAL_LENGTH_PENALTY
    beta: float = OFFICIAL_BETA
    sequence_title_sample: int = OFFICIAL_SEQUENCE_TITLE_SAMPLE
    warmup_ratio: float = OFFICIAL_WARMUP_RATIO
    max_grad_norm: float = OFFICIAL_MAX_GRAD_NORM
    ref_model_mixup_alpha: float = OFFICIAL_REF_MODEL_MIXUP_ALPHA
    ref_model_sync_steps: int = OFFICIAL_REF_MODEL_SYNC_STEPS
    logging_steps: int = 1
    eval_steps: float = 0.0999
    save_steps: float = 0.1
    save_total_limit: int = 1
    gradient_checkpointing: bool = True
    run_evaluation: bool = True
    save_checkpoints: bool = True
    max_steps: int = -1

    @property
    def effective_batch_size(self) -> int:
        """Repeated candidate rows contributing to one optimizer update."""

        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def unique_prompts_per_update(self) -> int:
        """Independent prompt groups contributing to one optimizer update."""

        return self.effective_batch_size // self.num_generations


@dataclass(frozen=True)
class GRPOComponents:
    """Objects prepared for GRPO; constructing them does not call train()."""

    model: Any
    reference_model: Any
    tokenizer: Any
    trainer: Trainer
    datasets: GRPODatasetSizes
    catalog: dict[str, int]
    model_load_count: int


class _RecordDataset(Dataset[GRPORecord]):
    def __init__(self, records: Sequence[GRPORecord]) -> None:
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GRPORecord:
        return self.records[index]


class RepeatRandomSampler(Sampler[int]):
    """Randomly order records and repeat each index G consecutive times."""

    def __init__(
        self,
        data_source: Sized,
        repeat_count: int,
        seed: int,
    ) -> None:
        if repeat_count < 2:
            raise ValueError("repeat_count must be at least 2")
        self.num_samples = len(data_source)
        self.repeat_count = repeat_count
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self) -> Iterator[int]:
        indices = (
            index
            for index in torch.randperm(
                self.num_samples,
                generator=self.generator,
            ).tolist()
            for _ in range(self.repeat_count)
        )
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples * self.repeat_count


def _identity_collator(features: list[GRPORecord]) -> list[GRPORecord]:
    return features


class SingleGPUReReTrainer(Trainer):
    """Single-device form of the fixed commit's custom ReReTrainer."""

    def __init__(
        self,
        *,
        model: nn.Module,
        reference_model: nn.Module,
        tokenizer: Any,
        rollout_generator: GRPORolloutGenerator,
        dataset_bundle: GRPODatasets,
        args: GRPOConfig,
    ) -> None:
        self.reference_model = reference_model
        self._tokenizer = tokenizer
        self.rollout_generator = rollout_generator
        self.dataset_bundle = dataset_bundle
        self.num_generations = rollout_generator.num_generations
        self.beta = args.beta
        self._grpo_metrics: defaultdict[str, list[float]] = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            train_dataset=_RecordDataset(dataset_bundle.train),
            eval_dataset=_RecordDataset(dataset_bundle.validation),
            data_collator=cast(Any, _identity_collator),
            processing_class=tokenizer,
        )
        if self.accelerator.num_processes != 1:
            raise RuntimeError("SingleGPUReReTrainer requires exactly one process")

        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)
        self.reference_model.eval()
        self.reference_model = self.accelerator.prepare_model(
            self.reference_model,
            evaluation_mode=True,
        )
        if args.sync_ref_model:
            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.reference_model,
                    accelerator=self.accelerator,
                )
            )

        # The loss is computed locally, so Trainer must scale it normally when
        # gradient accumulation is enabled.
        self.model_accepts_loss_kwargs = False

    def _get_train_sampler(self, train_dataset=None) -> Sampler[int]:
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if dataset is None:
            raise ValueError("train_dataset cannot be None")
        return RepeatRandomSampler(
            cast(Sized, dataset),
            self.num_generations,
            self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler[int]:
        if eval_dataset is None:
            raise ValueError("eval_dataset cannot be None")
        return RepeatRandomSampler(
            cast(Sized, eval_dataset),
            self.num_generations,
            self.args.seed,
        )

    def _targets_for_prompts(self, prompts: Sequence[str]) -> list[str]:
        targets: list[str] = []
        for prompt in prompts:
            try:
                history = self.dataset_bundle.prompt_to_history[prompt]
                target = self.dataset_bundle.history_to_target[history]
            except KeyError as error:
                raise KeyError(
                    "GRPO prompt or history is missing from reward mappings"
                ) from error
            targets.append(target)
        return targets

    def _record_rollout_metrics(
        self,
        *,
        rollout: GRPORolloutBatch,
        rewards: RankingGRPORewards,
    ) -> None:
        exact = torch.tensor(rewards.exact, dtype=torch.float32)
        ranking = torch.tensor(rewards.ranking, dtype=torch.float32)
        total = torch.tensor(rewards.total, dtype=torch.float32)
        self._grpo_metrics["rewards/exact"].append(exact.mean().item())
        self._grpo_metrics["rewards/ranking"].append(ranking.mean().item())
        self._grpo_metrics["reward"].append(total.mean().item())
        self._grpo_metrics["reward_std"].append(
            sum(rewards.group_sample_stds) / len(rewards.group_sample_stds)
        )
        self._grpo_metrics["valid_candidate_rate"].append(
            rollout.valid_candidate_rate
        )

        group_diversities = []
        for start in range(0, rollout.candidate_count, self.num_generations):
            group = rollout.completions[start : start + self.num_generations]
            group_diversities.append(len(set(group)) / self.num_generations)
        self._grpo_metrics["categorical_diversity"].append(
            sum(group_diversities) / len(group_diversities)
        )

        non_padding_ids = rollout.completion_ids[
            rollout.completion_ids != self._tokenizer.pad_token_id
        ]
        token_diversity = 0.0
        if non_padding_ids.numel() > 0:
            token_diversity = float(
                torch.unique(non_padding_ids).numel() / non_padding_ids.numel()
            )
        self._grpo_metrics["token_diversity"].append(token_diversity)

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

        rewards = compute_ranking_grpo_rewards(
            rollout.completions,
            targets,
            num_generations=self.num_generations,
        )
        advantages = torch.tensor(
            rewards.advantages,
            dtype=torch.float32,
            device=device,
        )
        self._record_rollout_metrics(rollout=rollout, rewards=rewards)
        return {
            "prompt_ids": rollout.prompt_ids,
            "prompt_mask": rollout.prompt_mask,
            "completion_ids": rollout.completion_ids,
            "completion_mask": rollout.completion_mask,
            "reference_log_probs": reference_log_probs,
            "advantages": advantages,
        }

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        if return_outputs:
            raise ValueError("GRPO does not support returning model outputs")

        input_ids = torch.cat(
            [inputs["prompt_ids"], inputs["completion_ids"]],
            dim=1,
        )
        attention_mask = torch.cat(
            [inputs["prompt_mask"], inputs["completion_mask"]],
            dim=1,
        )
        policy_log_probs = completion_token_log_probs(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_length=inputs["completion_ids"].size(1),
        )
        loss_output = official_grpo_loss(
            policy_log_probs=policy_log_probs,
            reference_log_probs=inputs["reference_log_probs"],
            advantages=inputs["advantages"],
            completion_mask=inputs["completion_mask"],
            beta=self.beta,
        )
        self._grpo_metrics["completion_length"].append(
            loss_output.mean_completion_length.item()
        )
        self._grpo_metrics["kl"].append(loss_output.mean_kl.item())
        return loss_output.loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        del prediction_loss_only, ignore_keys
        prepared_inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, prepared_inputs)
        return loss.mean().detach(), None, None

    def log(self, logs: dict[str, float], start_time=None) -> None:
        metrics = {
            name: sum(values) / len(values)
            for name, values in self._grpo_metrics.items()
            if values
        }
        if logs and next(iter(logs)).startswith("eval_"):
            metrics = {f"eval_{name}": value for name, value in metrics.items()}
        super().log({**logs, **metrics}, start_time)
        self._grpo_metrics.clear()


def _validate_config(config: GRPOTrainingConfig) -> None:
    if not config.model_path.is_dir():
        raise FileNotFoundError(config.model_path)
    for path in (
        config.train_file,
        config.valid_file,
        config.item_file,
        config.index_file,
        config.info_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if config.micro_batch_size < config.num_generations:
        raise ValueError("micro_batch_size must be at least num_generations")
    if config.micro_batch_size % config.num_generations != 0:
        raise ValueError("micro_batch_size must be divisible by num_generations")
    if config.eval_batch_size % config.num_generations != 0:
        raise ValueError("eval_batch_size must be divisible by num_generations")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.effective_batch_size % config.num_generations != 0:
        raise ValueError("effective batch size must be divisible by generations")
    if config.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.beta < 0:
        raise ValueError("beta cannot be negative")
    if not 0 < config.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be between 0 and 1")
    if config.run_evaluation and not 0 < config.eval_steps < 1:
        raise ValueError("eval_steps must be a ratio between 0 and 1")
    if config.save_checkpoints and not 0 < config.save_steps < 1:
        raise ValueError("save_steps must be a ratio between 0 and 1")
    if config.ref_model_sync_steps < 1:
        raise ValueError("ref_model_sync_steps must be positive")
    if not 0 <= config.ref_model_mixup_alpha <= 1:
        raise ValueError("ref_model_mixup_alpha must be between 0 and 1")


def _set_reproducible_seed(seed: int) -> None:
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_models_and_tokenizer(config: GRPOTrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("The Qwen tokenizer must define an EOS token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "dtype": torch.bfloat16,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    policy_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        **model_kwargs,
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        **model_kwargs,
    )
    if config.gradient_checkpointing:
        policy_model.config.use_cache = False
    return policy_model, reference_model, tokenizer


def _validate_model_vocabulary(model: Any, tokenizer: Any) -> None:
    model_vocabulary_size = model.get_input_embeddings().num_embeddings
    if model_vocabulary_size != len(tokenizer):
        raise ValueError(
            f"Model vocabulary {model_vocabulary_size} does not match "
            f"tokenizer size {len(tokenizer)}"
        )


def _training_arguments(config: GRPOTrainingConfig) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.micro_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        eval_strategy="steps" if config.run_evaluation else "no",
        eval_steps=config.eval_steps,
        save_strategy="steps" if config.save_checkpoints else "no",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
        max_steps=config.max_steps,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        beta=config.beta,
        sync_ref_model=True,
        ref_model_mixup_alpha=config.ref_model_mixup_alpha,
        ref_model_sync_steps=config.ref_model_sync_steps,
        loss_type="grpo",
        use_vllm=False,
    )


def build_grpo_components(config: GRPOTrainingConfig) -> GRPOComponents:
    """Load local artifacts and construct GRPO components without training."""

    _validate_config(config)
    _set_reproducible_seed(config.seed)
    datasets = build_grpo_datasets(
        train_file=config.train_file,
        valid_file=config.valid_file,
        item_file=config.item_file,
        index_file=config.index_file,
        sequence_title_sample=config.sequence_title_sample,
    )
    policy_model, reference_model, tokenizer = _load_models_and_tokenizer(config)
    _validate_model_vocabulary(policy_model, tokenizer)
    _validate_model_vocabulary(reference_model, tokenizer)
    catalog = load_sid_catalog(config.info_file, tokenizer)
    rollout_generator = GRPORolloutGenerator(
        tokenizer=tokenizer,
        catalog=catalog,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        length_penalty=config.length_penalty,
    )
    trainer = SingleGPUReReTrainer(
        model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        rollout_generator=rollout_generator,
        dataset_bundle=datasets,
        args=_training_arguments(config),
    )
    return GRPOComponents(
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
        model_load_count=2,
    )
