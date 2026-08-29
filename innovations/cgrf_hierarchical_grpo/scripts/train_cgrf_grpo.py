#!/usr/bin/env python3
"""Train CGRF-H GRPO on one local CUDA GPU."""

# pylint: disable=wrong-import-position,import-error,no-name-in-module

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
EXPERIMENT_SOURCE_ROOT = EXPERIMENT_ROOT / "src"
BASELINE_SOURCE_ROOT = REPOSITORY_ROOT / "src"
for source_root in (EXPERIMENT_SOURCE_ROOT, BASELINE_SOURCE_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from cgrf_hierarchical_grpo.cgrf_training import (  # noqa: E402
    CGRFHTrainingConfig,
    build_cgrf_h_components,
)
from minionerec.training.grpo_training import (  # noqa: E402
    GRPOTrainingConfig,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the independent CGRF-H training command parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Full-parameter BF16 CGRF-H GRPO. Baseline generation, KL, "
            "optimizer, and data settings remain unchanged."
        )
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--sasrec-checkpoint", required=True, type=Path)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--valid-file", required=True, type=Path)
    parser.add_argument("--item-file", required=True, type=Path)
    parser.add_argument("--index-file", required=True, type=Path)
    parser.add_argument("--info-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dense-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=64,
    )
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enabled by default to reduce full-parameter GRPO memory",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Existing Trainer checkpoint directory to resume",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run two optimizer steps without evaluation or checkpoints",
    )
    return parser


def _prepare_output_paths(
    output_dir: Path,
    resume_from_checkpoint: Path | None,
    smoke_test: bool,
) -> tuple[Path, Path]:
    final_model_dir = output_dir / "final_model"
    statistics_file = output_dir / "training_stats.json"
    if final_model_dir.exists():
        raise FileExistsError(final_model_dir)
    if statistics_file.exists():
        raise FileExistsError(statistics_file)
    if smoke_test and resume_from_checkpoint is not None:
        raise ValueError("smoke tests cannot resume from a checkpoint")
    if resume_from_checkpoint is not None and not resume_from_checkpoint.is_dir():
        raise FileNotFoundError(resume_from_checkpoint)
    if output_dir.exists() and resume_from_checkpoint is None:
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return final_model_dir, statistics_file


def _cuda_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU, for example with CUDA_VISIBLE_DEVICES=0"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA GPU does not support BF16")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    return {
        "logical_device": str(device),
        "name": torch.cuda.get_device_name(device),
        "total_memory_gib": round(properties.total_memory / 1024**3, 2),
    }


def _memory_summary() -> dict[str, float]:
    device = torch.device("cuda:0")
    return {
        "allocated_gib": round(torch.cuda.memory_allocated(device) / 1024**3, 2),
        "reserved_gib": round(torch.cuda.memory_reserved(device) / 1024**3, 2),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device) / 1024**3,
            2,
        ),
        "peak_reserved_gib": round(
            torch.cuda.max_memory_reserved(device) / 1024**3,
            2,
        ),
    }


def _build_config(arguments: argparse.Namespace) -> CGRFHTrainingConfig:
    grpo = GRPOTrainingConfig(
        model_path=arguments.model_path,
        train_file=arguments.train_file,
        valid_file=arguments.valid_file,
        item_file=arguments.item_file,
        index_file=arguments.index_file,
        info_file=arguments.info_file,
        output_dir=arguments.output_dir,
        seed=arguments.seed,
        micro_batch_size=arguments.micro_batch_size,
        eval_batch_size=arguments.eval_batch_size,
        gradient_accumulation_steps=(
            1 if arguments.smoke_test else arguments.gradient_accumulation_steps
        ),
        num_epochs=arguments.num_epochs,
        learning_rate=arguments.learning_rate,
        num_generations=arguments.num_generations,
        max_prompt_length=arguments.max_prompt_length,
        max_completion_length=arguments.max_completion_length,
        temperature=arguments.temperature,
        beta=arguments.beta,
        gradient_checkpointing=arguments.gradient_checkpointing,
        run_evaluation=not arguments.smoke_test,
        save_checkpoints=not arguments.smoke_test,
        max_steps=2 if arguments.smoke_test else -1,
    )
    return CGRFHTrainingConfig(
        grpo=grpo,
        sasrec_checkpoint=arguments.sasrec_checkpoint,
        dense_weight=arguments.dense_weight,
    )


def _training_parameters(
    config: CGRFHTrainingConfig,
    smoke_test: bool,
) -> dict[str, Any]:
    grpo = config.grpo
    return {
        "mode": "smoke" if smoke_test else "formal",
        "reward_type": "cgrf_h",
        "dense_weight": config.dense_weight,
        "formula": (
            "official + lambda * "
            "(gate * collaborative_rank + (1 - gate) * hierarchical)"
        ),
        "micro_batch_size": grpo.micro_batch_size,
        "eval_batch_size": grpo.eval_batch_size,
        "gradient_accumulation_steps": grpo.gradient_accumulation_steps,
        "effective_batch_size": grpo.effective_batch_size,
        "unique_prompts_per_update": grpo.unique_prompts_per_update,
        "num_generations": grpo.num_generations,
        "num_epochs": grpo.num_epochs,
        "max_steps": grpo.max_steps,
        "learning_rate": grpo.learning_rate,
        "warmup_ratio": grpo.warmup_ratio,
        "max_grad_norm": grpo.max_grad_norm,
        "beta": grpo.beta,
        "max_prompt_length": grpo.max_prompt_length,
        "max_completion_length": grpo.max_completion_length,
        "temperature": grpo.temperature,
        "length_penalty": grpo.length_penalty,
        "bf16": True,
        "gradient_checkpointing": grpo.gradient_checkpointing,
        "optimizer": "paged_adamw_32bit",
        "scheduler": "cosine",
        "sync_ref_model": True,
        "ref_model_mixup_alpha": grpo.ref_model_mixup_alpha,
        "ref_model_sync_steps": grpo.ref_model_sync_steps,
        "run_evaluation": grpo.run_evaluation,
        "save_checkpoints": grpo.save_checkpoints,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _checkpoint_names(output_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    """Run CGRF-H GRPO and save its model and statistics independently."""

    parser = build_parser()
    arguments = parser.parse_args()
    try:
        final_model_dir, statistics_file = _prepare_output_paths(
            arguments.output_dir,
            arguments.resume_from_checkpoint,
            arguments.smoke_test,
        )
        cuda = _cuda_summary()
        torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        config = _build_config(arguments)
        started = time.perf_counter()
        components = build_cgrf_h_components(config)
        parameters = _training_parameters(config, arguments.smoke_test)
        preflight = {
            "mode": "smoke" if arguments.smoke_test else "formal",
            "cuda": cuda,
            "datasets": asdict(components.datasets),
            "catalog": components.catalog,
            "teacher": components.teacher,
            "model_load_count": components.model_load_count,
            "training_parameters": parameters,
        }
        print(json.dumps(_json_safe(preflight), indent=2, ensure_ascii=False))

        train_result = components.trainer.train(
            resume_from_checkpoint=(
                str(arguments.resume_from_checkpoint)
                if arguments.resume_from_checkpoint is not None
                else None
            )
        )
        components.trainer.save_model(str(final_model_dir))
        components.tokenizer.save_pretrained(final_model_dir)
        summary = {
            "final_model_dir": str(final_model_dir),
            "statistics_file": str(statistics_file),
            "resume_from_checkpoint": (
                str(arguments.resume_from_checkpoint)
                if arguments.resume_from_checkpoint is not None
                else None
            ),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "cuda": {**cuda, **_memory_summary()},
            "datasets": asdict(components.datasets),
            "catalog": components.catalog,
            "teacher": components.teacher,
            "model_load_count": components.model_load_count,
            "training_parameters": parameters,
            "result": {
                "global_step": components.trainer.state.global_step,
                "epoch": components.trainer.state.epoch,
                "training_loss": train_result.training_loss,
                "metrics": train_result.metrics,
                "checkpoints": _checkpoint_names(arguments.output_dir),
                "log_history": components.trainer.state.log_history,
            },
        }
        _write_json(statistics_file, summary)
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))

    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
