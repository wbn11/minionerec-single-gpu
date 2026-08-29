#!/usr/bin/env python3
"""Train the frozen Item-ID SASRec teacher for CGRF-H."""

# pylint: disable=wrong-import-position,import-error

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = EXPERIMENT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cgrf_hierarchical_grpo.sasrec_training import (  # noqa: E402
    SASRecTrainingConfig,
    train_sasrec,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone teacher-training command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Train an Item-ID SASRec teacher using only MiniOneRec final "
            "train and validation CSV files."
        )
    )
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--valid-file", required=True, type=Path)
    parser.add_argument("--num-items", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-sequence-length", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--sample",
        type=int,
        help="Optional deterministic per-split sample for a smoke test",
    )
    return parser


def main() -> None:
    """Parse arguments, train the teacher, and print its compact summary."""

    arguments = build_parser().parse_args()
    statistics = train_sasrec(
        SASRecTrainingConfig(
            train_file=arguments.train_file,
            valid_file=arguments.valid_file,
            output_dir=arguments.output_dir,
            num_items=arguments.num_items,
            max_sequence_length=arguments.max_sequence_length,
            hidden_size=arguments.hidden_size,
            num_layers=arguments.num_layers,
            num_heads=arguments.num_heads,
            dropout=arguments.dropout,
            batch_size=arguments.batch_size,
            eval_batch_size=arguments.eval_batch_size,
            num_epochs=arguments.num_epochs,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            patience=arguments.patience,
            seed=arguments.seed,
            num_workers=arguments.num_workers,
            device=arguments.device,
            sample=arguments.sample,
        )
    )
    print(
        json.dumps(
            {
                "model_file": statistics["model_file"],
                "statistics_file": statistics["statistics_file"],
                "best": statistics["best"],
                "epochs_executed": statistics["epochs_executed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

