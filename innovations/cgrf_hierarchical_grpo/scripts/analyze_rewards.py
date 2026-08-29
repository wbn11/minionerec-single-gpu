#!/usr/bin/env python3
"""Generate immutable candidates and compare baseline with CGRF-H rewards."""

# pylint: disable=wrong-import-position,import-error

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
EXPERIMENT_SOURCE = EXPERIMENT_ROOT / "src"
BASELINE_SOURCE = REPOSITORY_ROOT / "src"
for source_root in (EXPERIMENT_SOURCE, BASELINE_SOURCE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from cgrf_hierarchical_grpo.reward_replay import (  # noqa: E402
    analyze_candidate_groups,
    build_replay_records,
    generate_candidate_groups,
    load_candidate_groups,
    sample_replay_records,
    write_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the generation-and-replay command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Cache frozen-SFT candidate groups and replay official and "
            "confidence-gated collaborative rewards without GRPO updates."
        )
    )
    parser.add_argument("--sasrec-checkpoint", required=True, type=Path)
    parser.add_argument("--index-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--valid-file", type=Path)
    parser.add_argument("--item-file", type=Path)
    parser.add_argument("--info-file", type=Path)
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--sequence-title-sample", type=int, default=10_000)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--prompt-batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _required_generation_paths(arguments: argparse.Namespace) -> None:
    missing = [
        name
        for name in (
            "model_path",
            "train_file",
            "valid_file",
            "item_file",
            "info_file",
        )
        if getattr(arguments, name) is None
    ]
    if missing:
        formatted = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"Candidate generation requires: {formatted}")


def main() -> None:
    """Create or reuse candidates, replay rewards, and write one analysis."""

    arguments = build_parser().parse_args()
    output_dir = arguments.output_dir
    analysis_file = output_dir / "reward_analysis.json"
    generation_statistics = None

    if arguments.candidate_file is None:
        _required_generation_paths(arguments)
        candidate_file = output_dir / "candidate_groups.jsonl"
        records = build_replay_records(
            train_file=arguments.train_file,
            valid_file=arguments.valid_file,
            item_file=arguments.item_file,
            index_file=arguments.index_file,
            sequence_title_sample=arguments.sequence_title_sample,
        )
        sampled_records = sample_replay_records(
            records,
            sample=arguments.sample,
            seed=arguments.seed,
        )
        generation_statistics = generate_candidate_groups(
            records=sampled_records,
            model_path=arguments.model_path,
            info_file=arguments.info_file,
            output_file=candidate_file,
            device=arguments.device,
            prompt_batch_size=arguments.prompt_batch_size,
            num_generations=arguments.num_generations,
            max_prompt_length=arguments.max_prompt_length,
            max_completion_length=arguments.max_completion_length,
            temperature=arguments.temperature,
            seed=arguments.seed,
        )
    else:
        candidate_file = arguments.candidate_file

    groups = load_candidate_groups(candidate_file)
    analysis = analyze_candidate_groups(
        groups=groups,
        sasrec_checkpoint=arguments.sasrec_checkpoint,
        index_file=arguments.index_file,
        lambdas=arguments.lambdas,
        device=arguments.device,
    )
    result = {
        "candidate_file": str(candidate_file),
        "analysis_file": str(analysis_file),
        "generation": generation_statistics,
        "analysis": analysis,
    }
    write_analysis(result, analysis_file)
    print(
        json.dumps(
            {
                "candidate_file": str(candidate_file),
                "analysis_file": str(analysis_file),
                "overall": analysis["overall"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

