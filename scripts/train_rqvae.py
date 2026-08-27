#!/usr/bin/env python3
"""Train the official MiniOneRec RQ-VAE on one GPU."""

# pylint: disable=wrong-import-position,import-error

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from minionerec.semantic_id.rqvae_training import (  # noqa: E402
    RQVAETrainingConfig,
    train_rqvae,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the fixed MiniOneRec three-level RQ-VAE and keep one model "
            "file plus one JSON statistics file."
        )
    )
    parser.add_argument("--embedding-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=20_480)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-step", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--expected-items",
        type=int,
        default=3686,
        help="Expected Industrial_and_Scientific item count",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from rqvae_model.pth and rqvae_training_stats.json",
    )
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    config = RQVAETrainingConfig(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        eval_step=arguments.eval_step,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        warmup_epochs=arguments.warmup_epochs,
        device=arguments.device,
        seed=arguments.seed,
    )
    try:
        summary = train_rqvae(
            embedding_file=arguments.embedding_file,
            output_dir=arguments.output_dir,
            config=config,
            expected_items=arguments.expected_items,
            resume=arguments.resume,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        FloatingPointError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
