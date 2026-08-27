#!/usr/bin/env python3
"""Generate MiniOneRec item Semantic IDs from the best RQ-VAE checkpoint."""

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

from minionerec.semantic_id.semantic_ids import (  # noqa: E402
    OFFICIAL_COLLISION_EPSILON,
    OFFICIAL_MAX_COLLISION_ROUNDS,
    generate_semantic_ids,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load best_collision from rqvae_model.pth, generate three-token "
            "item SIDs, and resolve collisions with third-level Sinkhorn."
        )
    )
    parser.add_argument("--embedding-file", required=True, type=Path)
    parser.add_argument("--checkpoint-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument(
        "--statistics-file",
        required=True,
        type=Path,
        help="JSON metrics file, normally stored under results/",
    )
    parser.add_argument("--expected-items", type=int, default=3686)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--sinkhorn-epsilon",
        type=float,
        default=OFFICIAL_COLLISION_EPSILON,
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=OFFICIAL_MAX_COLLISION_ROUNDS,
    )
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        summary = generate_semantic_ids(
            embedding_file=arguments.embedding_file,
            checkpoint_file=arguments.checkpoint_file,
            output_file=arguments.output_file,
            statistics_file=arguments.statistics_file,
            expected_items=arguments.expected_items,
            batch_size=arguments.batch_size,
            device=arguments.device,
            epsilon=arguments.sinkhorn_epsilon,
            max_rounds=arguments.max_rounds,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        FloatingPointError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
