#!/usr/bin/env python3
"""Create one compact, reproducible summary of baseline and CGRF-H results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _metric_comparison(
    baseline: dict[str, float],
    experiment: dict[str, float],
    sample_count: int,
) -> dict[str, dict[str, float | int]]:
    if baseline.keys() != experiment.keys():
        raise ValueError("Metric sets do not match")
    comparison: dict[str, dict[str, float | int]] = {}
    for name, baseline_value in baseline.items():
        experiment_value = experiment[name]
        difference = experiment_value - baseline_value
        values: dict[str, float | int] = {
            "absolute": difference,
            "absolute_percentage_points": difference * 100.0,
            "relative_percent": difference / baseline_value * 100.0,
        }
        if name.startswith("HR@"):
            values["baseline_hit_count"] = round(
                baseline_value * sample_count
            )
            values["experiment_hit_count"] = round(
                experiment_value * sample_count
            )
        comparison[name] = values
    return comparison


def _last_log_with_key(
    training: dict[str, Any], key: str
) -> dict[str, Any]:
    history = training["result"].get("log_history", [])
    matches = [row for row in history if key in row]
    if not matches:
        raise ValueError(f"Training statistics do not contain {key!r}")
    return matches[-1]


def _finite_metrics(metrics: dict[str, float]) -> None:
    invalid = [name for name, value in metrics.items() if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"Non-finite metrics: {invalid}")


def build_summary(arguments: argparse.Namespace) -> dict[str, Any]:
    """Read raw local artifacts and return a compact experiment summary."""

    data_statistics = _load_json(arguments.data_statistics)
    sid_statistics = _load_json(arguments.sid_statistics)
    rqvae_training = _load_json(arguments.rqvae_training)
    sft_metrics_file = _load_json(arguments.sft_metrics)
    grpo_metrics_file = _load_json(arguments.grpo_metrics)
    cgrf_metrics_file = _load_json(arguments.cgrf_metrics)
    sft_training = _load_json(arguments.sft_training)
    grpo_training = _load_json(arguments.grpo_training)
    cgrf_training = _load_json(arguments.cgrf_training)
    sasrec_training = _load_json(arguments.sasrec_training)
    reward_replay_file = _load_json(arguments.reward_replay)

    sft_metrics = sft_metrics_file["metrics"]
    grpo_metrics = grpo_metrics_file["metrics"]
    cgrf_metrics = cgrf_metrics_file["metrics"]
    for metrics in (sft_metrics, grpo_metrics, cgrf_metrics):
        _finite_metrics(metrics)

    sample_count = int(cgrf_metrics_file["generation"]["sample_count"])
    replay = reward_replay_file["analysis"]["overall"]
    replay_baseline = replay["baseline"]
    replay_cgrf = replay["cgrf_h"]["0.1"]
    baseline_zero_advantage = replay_baseline["zero_advantage_group_rate"]
    cgrf_zero_advantage = replay_cgrf["zero_advantage_group_rate"]

    cgrf_train_logs = [
        row
        for row in cgrf_training["result"]["log_history"]
        if "reward/dense" in row
    ]
    cgrf_final_eval = _last_log_with_key(cgrf_training, "eval_loss")
    grpo_final_eval = _last_log_with_key(grpo_training, "eval_loss")
    selected_rqvae_record = next(
        row
        for row in rqvae_training["records"]
        if row["epoch"] == rqvae_training["best_collision"]["epoch"]
    )
    grpo_runtime = float(grpo_training["elapsed_seconds"])
    cgrf_runtime = float(cgrf_training["elapsed_seconds"])
    grpo_kl = float(grpo_final_eval["eval_kl"])
    cgrf_kl = float(cgrf_final_eval["eval_kl"])

    return {
        "experiment": {
            "name": "MiniOneRec single-A6000 reproduction with CGRF-H",
            "dataset": "Amazon18 Industrial_and_Scientific",
            "upstream_commit": "0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed",
            "embedding_model": "Qwen3-Embedding-4B",
            "recommendation_model": "Qwen2.5-1.5B-Instruct",
            "gpu": cgrf_metrics_file["cuda"]["name"],
            "gpu_memory_gib": cgrf_metrics_file["cuda"]["total_memory_gib"],
            "users": data_statistics["counts"]["users"],
            "items": data_statistics["counts"]["items"],
            "unique_sids": sid_statistics["after"]["unique_sid_count"],
            "train_samples": data_statistics["counts"]["train"],
            "validation_samples": data_statistics["counts"]["valid"],
            "test_samples": data_statistics["counts"]["test"],
            "primary_evaluation": "SID-level constrained Beam-50 HR/NDCG",
        },
        "data_processing": {
            "processing": data_statistics["processing"],
            "counts": data_statistics["counts"],
            "user_sequence_length": data_statistics["user_sequence_length"],
            "sample_history_length": data_statistics[
                "sample_history_length"
            ],
        },
        "semantic_id": {
            "embedding_shape": rqvae_training["embedding_shape"],
            "embedding_dtype": rqvae_training["embedding_dtype"],
            "rqvae_parameters": rqvae_training["config"],
            "rqvae_best_loss": rqvae_training["best_loss"],
            "rqvae_best_collision": rqvae_training["best_collision"],
            "selected_rqvae_checkpoint": {
                "epoch": selected_rqvae_record["epoch"],
                "total_loss": selected_rqvae_record["total_loss"],
                "reconstruction_loss": selected_rqvae_record[
                    "reconstruction_loss"
                ],
                "quantization_loss": selected_rqvae_record[
                    "quantization_loss"
                ],
                "collision_count": selected_rqvae_record[
                    "collision_count"
                ],
                "collision_rate": selected_rqvae_record[
                    "collision_rate"
                ],
                "codebook_layers": selected_rqvae_record[
                    "codebook_layers"
                ],
            },
            "checkpoint_epoch": sid_statistics["checkpoint_epoch"],
            "sinkhorn_epsilon": sid_statistics["sinkhorn_epsilon"],
            "sinkhorn_iterations": sid_statistics["sinkhorn_iterations"],
            "rounds_executed": sid_statistics["rounds_executed"],
            "before_collision_resolution": sid_statistics["before"],
            "after_collision_resolution": sid_statistics["after"],
        },
        "sasrec_teacher": {
            "model": sasrec_training["model"],
            "best_epoch": sasrec_training["best"]["epoch"],
            "epochs_executed": sasrec_training["epochs_executed"],
            "elapsed_seconds": sasrec_training["elapsed_seconds"],
            "peak_allocated_gib": sasrec_training["cuda"][
                "peak_allocated_gib"
            ],
            "parameters": sasrec_training["training_parameters"],
            "best_validation": sasrec_training["best"]["validation"],
        },
        "reward_replay": {
            "candidate_groups": reward_replay_file["analysis"][
                "candidate_groups"
            ],
            "candidates_per_group": 16,
            "target_in_candidates_rate": replay[
                "target_in_candidates_rate"
            ],
            "baseline_zero_advantage_rate": baseline_zero_advantage,
            "cgrf_h_zero_advantage_rate": cgrf_zero_advantage,
            "absolute_reduction_percentage_points": (
                baseline_zero_advantage - cgrf_zero_advantage
            )
            * 100.0,
            "relative_reduction_percent": (
                baseline_zero_advantage - cgrf_zero_advantage
            )
            / baseline_zero_advantage
            * 100.0,
            "exact_target_top_reward_rate": replay_cgrf[
                "exact_target_top_reward_rate"
            ],
            "teacher_gate": replay["teacher_gate"],
            "selected_dense_weight": 0.1,
        },
        "training": {
            "sft": {
                "global_step": sft_training["result"]["global_step"],
                "epoch": sft_training["result"]["epoch"],
                "training_loss": sft_training["result"]["training_loss"],
                "best_eval_loss": sft_training["result"]["best_eval_loss"],
                "runtime_seconds": sft_training["result"]["metrics"][
                    "train_runtime"
                ],
                "parameters": sft_training["training_parameters"],
            },
            "baseline_grpo": {
                "global_step": grpo_training["result"]["global_step"],
                "epoch": grpo_training["result"]["epoch"],
                "training_loss": grpo_training["result"]["training_loss"],
                "runtime_seconds": grpo_runtime,
                "peak_allocated_gib": grpo_training["cuda"][
                    "peak_allocated_gib"
                ],
                "final_validation": grpo_final_eval,
                "parameters": grpo_training["training_parameters"],
            },
            "cgrf_h": {
                "global_step": cgrf_training["result"]["global_step"],
                "epoch": cgrf_training["result"]["epoch"],
                "training_loss": cgrf_training["result"]["training_loss"],
                "runtime_seconds": cgrf_runtime,
                "peak_allocated_gib": cgrf_training["cuda"][
                    "peak_allocated_gib"
                ],
                "mean_training_zero_advantage_group_rate": statistics.fmean(
                    row["zero_advantage_group_rate"] for row in cgrf_train_logs
                ),
                "final_validation": cgrf_final_eval,
                "teacher": cgrf_training["teacher"],
                "parameters": cgrf_training["training_parameters"],
            },
            "cgrf_h_vs_baseline_grpo": {
                "runtime_overhead_seconds": cgrf_runtime - grpo_runtime,
                "runtime_overhead_percent": (
                    (cgrf_runtime - grpo_runtime) / grpo_runtime * 100.0
                ),
                "peak_allocated_gib_difference": cgrf_training["cuda"][
                    "peak_allocated_gib"
                ]
                - grpo_training["cuda"]["peak_allocated_gib"],
                "final_validation_kl_ratio": cgrf_kl / grpo_kl,
            },
        },
        "sid_metrics": {
            "sft": sft_metrics,
            "baseline_grpo": grpo_metrics,
            "cgrf_h": cgrf_metrics,
            "cgrf_h_vs_baseline_grpo": _metric_comparison(
                grpo_metrics,
                cgrf_metrics,
                sample_count,
            ),
            "cgrf_h_vs_sft": _metric_comparison(
                sft_metrics,
                cgrf_metrics,
                sample_count,
            ),
        },
        "supplementary_item_metrics": cgrf_metrics_file.get("item_metrics"),
        "scope": {
            "completed_datasets": ["Industrial_and_Scientific"],
            "independent_training_runs_per_method": 1,
            "statistical_significance_tested": False,
            "semantic_code_revival_included": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build paths for raw inputs and the compact output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-statistics",
        type=Path,
        default=REPOSITORY_ROOT
        / (
            "artifacts/data/processed/amazon18/Industrial_and_Scientific/"
            "Industrial_and_Scientific.data_stats.json"
        ),
    )
    parser.add_argument(
        "--sid-statistics",
        type=Path,
        default=REPOSITORY_ROOT
        / (
            "artifacts/data/processed/amazon18/Industrial_and_Scientific/"
            "Industrial_and_Scientific.index.stats.json"
        ),
    )
    parser.add_argument(
        "--rqvae-training",
        type=Path,
        default=REPOSITORY_ROOT
        / "results/rqvae/Industrial_and_Scientific/rqvae_training_stats.json",
    )
    parser.add_argument(
        "--sft-metrics",
        type=Path,
        default=REPOSITORY_ROOT
        / "results/evaluation/Industrial_and_Scientific/sft_metrics.json",
    )
    parser.add_argument(
        "--grpo-metrics",
        type=Path,
        default=REPOSITORY_ROOT
        / "results/evaluation/Industrial_and_Scientific/grpo_metrics.json",
    )
    parser.add_argument(
        "--cgrf-metrics",
        type=Path,
        default=EXPERIMENT_ROOT
        / "results/evaluation/cgrf_h_lambda_01/sid_metrics.json",
    )
    parser.add_argument(
        "--sft-training",
        type=Path,
        default=REPOSITORY_ROOT
        / "results/sft/Industrial_and_Scientific/training_stats.json",
    )
    parser.add_argument(
        "--grpo-training",
        type=Path,
        default=REPOSITORY_ROOT
        / "results/grpo/Industrial_and_Scientific/training_stats.json",
    )
    parser.add_argument(
        "--cgrf-training",
        type=Path,
        default=EXPERIMENT_ROOT
        / "results/grpo/cgrf_h_lambda_01/training_stats.json",
    )
    parser.add_argument(
        "--sasrec-training",
        type=Path,
        default=EXPERIMENT_ROOT / "results/sasrec/training_stats.json",
    )
    parser.add_argument(
        "--reward-replay",
        type=Path,
        default=EXPERIMENT_ROOT / "results/reward_replay/reward_analysis.json",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=EXPERIMENT_ROOT / "experiment_summary.json",
    )
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        summary = build_summary(arguments)
        arguments.output_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = arguments.output_file.with_suffix(".json.tmp")
        with temporary_file.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary_file.replace(arguments.output_file)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(arguments.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
