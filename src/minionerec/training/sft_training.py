"""Build the official MiniOneRec SFT pipeline for one GPU."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from minionerec.data.sft_datasets import (
    FusionSeqRecDataset,
    SidItemFeatDataset,
    SidSFTDataset,
)
from minionerec.training.sft_vocab import (
    collect_sid_tokens,
    extend_tokenizer,
    resize_model_vocab,
    validate_sid_tokens,
)


@dataclass(frozen=True)
class SFTTrainingConfig:
    """Inputs and fixed-commit defaults for full-parameter SFT."""

    model_path: Path
    train_file: Path
    valid_file: Path
    item_file: Path
    index_file: Path
    output_dir: Path
    sample: int = -1
    seed: int = 42
    batch_size: int = 128
    micro_batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 3e-4
    max_length: int = 512
    warmup_steps: int = 20
    logging_steps: int = 1
    eval_steps: float = 0.05
    save_steps: float = 0.05
    save_total_limit: int = 1
    early_stopping_patience: int = 3
    group_by_length: bool = False
    gradient_checkpointing: bool = False

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.batch_size // self.micro_batch_size


@dataclass(frozen=True)
class SFTDatasetSizes:
    sid_prediction: int
    item_identification: int
    fusion_title_prediction: int
    train_total: int
    validation: int


@dataclass(frozen=True)
class SFTComponents:
    """Objects prepared for training; constructing them does not call train()."""

    model: Any
    tokenizer: Any
    trainer: Trainer
    vocabulary: dict[str, int]
    datasets: SFTDatasetSizes


def _validate_config(config: SFTTrainingConfig) -> None:
    if not config.model_path.is_dir():
        raise FileNotFoundError(config.model_path)
    for path in (
        config.train_file,
        config.valid_file,
        config.item_file,
        config.index_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if config.micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if config.batch_size < config.micro_batch_size:
        raise ValueError("batch_size must be at least micro_batch_size")
    if config.batch_size % config.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    if config.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.max_length <= 0:
        raise ValueError("max_length must be positive")
    if not 0 < config.eval_steps < 1:
        raise ValueError("eval_steps must be a ratio between 0 and 1")
    if not 0 < config.save_steps < 1:
        raise ValueError("save_steps must be a ratio between 0 and 1")


def _set_reproducible_seed(seed: int) -> None:
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_model_and_tokenizer(config: SFTTrainingConfig) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("The Qwen tokenizer must define an EOS token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    return model, tokenizer


def _extend_vocabulary(
    model: Any,
    tokenizer: Any,
    index_file: Path,
) -> dict[str, int]:
    original_vocabulary_size = len(tokenizer)
    sid_tokens = collect_sid_tokens(index_file)
    added_token_count = extend_tokenizer(tokenizer, sid_tokens)
    resized_vocabulary_size = resize_model_vocab(model, tokenizer)
    validation = validate_sid_tokens(tokenizer, index_file)
    return {
        "original_vocabulary_size": original_vocabulary_size,
        "sid_token_count": len(sid_tokens),
        "added_token_count": added_token_count,
        "resized_vocabulary_size": resized_vocabulary_size,
        "item_count": validation["item_count"],
        "tokens_per_item": validation["tokens_per_item"],
    }


def _build_datasets(
    config: SFTTrainingConfig,
    tokenizer: Any,
) -> tuple[ConcatDataset, SidSFTDataset, SFTDatasetSizes]:
    sid_prediction = SidSFTDataset(
        train_file=config.train_file,
        tokenizer=tokenizer,
        max_length=config.max_length,
        sample=config.sample,
        seed=config.seed,
    )
    item_identification = SidItemFeatDataset(
        item_file=config.item_file,
        index_file=config.index_file,
        tokenizer=tokenizer,
        max_length=config.max_length,
        sample=config.sample,
        seed=config.seed,
    )
    fusion_title_prediction = FusionSeqRecDataset(
        train_file=config.train_file,
        item_file=config.item_file,
        index_file=config.index_file,
        tokenizer=tokenizer,
        max_length=config.max_length,
        sample=config.sample,
        seed=config.seed,
    )
    validation = SidSFTDataset(
        train_file=config.valid_file,
        tokenizer=tokenizer,
        max_length=config.max_length,
        sample=config.sample,
        seed=config.seed,
    )

    train_dataset = ConcatDataset(
        [sid_prediction, item_identification, fusion_title_prediction]
    )
    sizes = SFTDatasetSizes(
        sid_prediction=len(sid_prediction),
        item_identification=len(item_identification),
        fusion_title_prediction=len(fusion_title_prediction),
        train_total=len(train_dataset),
        validation=len(validation),
    )
    return train_dataset, validation, sizes


def build_sft_components(config: SFTTrainingConfig) -> SFTComponents:
    """Load local artifacts and construct a Trainer without starting training."""

    _validate_config(config)
    _set_reproducible_seed(config.seed)
    model, tokenizer = _load_model_and_tokenizer(config)
    vocabulary = _extend_vocabulary(model, tokenizer, config.index_file)
    train_dataset, valid_dataset, dataset_sizes = _build_datasets(
        config, tokenizer
    )

    model.config.use_cache = False
    training_arguments = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.micro_batch_size,
        per_device_eval_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        bf16=True,
        logging_steps=config.logging_steps,
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=config.group_by_length,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ],
    )
    return SFTComponents(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        vocabulary=vocabulary,
        datasets=dataset_sizes,
    )
