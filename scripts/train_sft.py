#!/usr/bin/env python3
"""Train the official MiniOneRec SFT tasks on one local CUDA GPU."""

# pylint: disable=wrong-import-position,import-error

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from minionerec.training.sft_training import (  # noqa: E402
    SFTTrainingConfig,
    build_sft_components,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Full-parameter BF16 SFT for the three tasks enabled by the fixed "
            "MiniOneRec commit. All model and data inputs must already exist "
            "locally."
        )
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--valid-file", required=True, type=Path)
    parser.add_argument("--item-file", required=True, type=Path)
    parser.add_argument("--index-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Reduce activation memory at the cost of additional computation",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Existing Trainer checkpoint directory to resume",
    )
    return parser


def _prepare_output_paths(
    output_dir: Path,
    resume_from_checkpoint: Path | None,
) -> tuple[Path, Path]:
    final_model_dir = output_dir / "final_model"
    statistics_file = output_dir / "training_stats.json"

    if final_model_dir.exists():
        raise FileExistsError(final_model_dir)
    if statistics_file.exists():
        raise FileExistsError(statistics_file)
    if resume_from_checkpoint is not None and not resume_from_checkpoint.is_dir():
        raise FileNotFoundError(resume_from_checkpoint)

    if output_dir.exists() and resume_from_checkpoint is None:
        existing_entries = list(output_dir.iterdir())
        if existing_entries:
            raise FileExistsError(
                f"Output directory is not empty; use --resume-from-checkpoint: "
                f"{output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    return final_model_dir, statistics_file


def _cuda_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA GPU does not support BF16 training")

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "logical_device": f"cuda:{device_index}",
        "name": torch.cuda.get_device_name(device_index),
        "total_memory_gib": round(properties.total_memory / 1024**3, 2),
    }


def _training_summary(
    config: SFTTrainingConfig,
    components: Any,
    train_result: Any,
    final_model_dir: Path,
    statistics_file: Path,
    resume_from_checkpoint: Path | None,
    cuda: dict[str, Any],
) -> dict[str, Any]:
    metrics = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in train_result.metrics.items()
    }
    return {
        "final_model_dir": str(final_model_dir),
        "statistics_file": str(statistics_file),
        "resume_from_checkpoint": (
            str(resume_from_checkpoint)
            if resume_from_checkpoint is not None
            else None
        ),
        "cuda": cuda,
        "datasets": asdict(components.datasets),
        "vocabulary": components.vocabulary,
        "training_parameters": {
            "batch_size": config.batch_size,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": (
                config.gradient_accumulation_steps
            ),
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "max_length": config.max_length,
            "seed": config.seed,
            "bf16": True,
            "gradient_checkpointing": config.gradient_checkpointing,
        },
        "result": {
            "global_step": components.trainer.state.global_step,
            "epoch": components.trainer.state.epoch,
            "training_loss": train_result.training_loss,
            "best_checkpoint": components.trainer.state.best_model_checkpoint,
            "best_eval_loss": components.trainer.state.best_metric,
            "metrics": metrics,
        },
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()

    try:
        final_model_dir, statistics_file = _prepare_output_paths(
            arguments.output_dir,
            arguments.resume_from_checkpoint,
        )
        cuda = _cuda_summary()
        config = SFTTrainingConfig(
            model_path=arguments.model_path,
            train_file=arguments.train_file,
            valid_file=arguments.valid_file,
            item_file=arguments.item_file,
            index_file=arguments.index_file,
            output_dir=arguments.output_dir,
            sample=arguments.sample,
            seed=arguments.seed,
            batch_size=arguments.batch_size,
            micro_batch_size=arguments.micro_batch_size,
            num_epochs=arguments.num_epochs,
            learning_rate=arguments.learning_rate,
            max_length=arguments.max_length,
            gradient_checkpointing=arguments.gradient_checkpointing,
        )
        components = build_sft_components(config)
        preflight = {
            "cuda": cuda,
            "datasets": asdict(components.datasets),
            "vocabulary": components.vocabulary,
            "effective_batch_size": config.batch_size,
            "gradient_accumulation_steps": (
                config.gradient_accumulation_steps
            ),
        }
        print(json.dumps(preflight, indent=2, ensure_ascii=False))

        train_result = components.trainer.train(
            resume_from_checkpoint=(
                str(arguments.resume_from_checkpoint)
                if arguments.resume_from_checkpoint is not None
                else None
            )
        )
        components.trainer.save_model(str(final_model_dir))
        components.tokenizer.save_pretrained(final_model_dir)

        summary = _training_summary(
            config=config,
            components=components,
            train_result=train_result,
            final_model_dir=final_model_dir,
            statistics_file=statistics_file,
            resume_from_checkpoint=arguments.resume_from_checkpoint,
            cuda=cuda,
        )
        _write_json(statistics_file, summary)
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
