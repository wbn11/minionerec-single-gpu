#!/usr/bin/env python3
"""Evaluate a MiniOneRec SFT model with constrained beam search."""

# pylint: disable=wrong-import-position,import-error,no-name-in-module

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from minionerec.generation.sft_generation import (  # noqa: E402
    OFFICIAL_LENGTH_PENALTY,
    OFFICIAL_MAX_NEW_TOKENS,
    OFFICIAL_NUM_BEAMS,
    evaluate_sft_model,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the single-GPU SFT evaluation command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Single-GPU BF16 constrained-beam evaluation using the fixed "
            "MiniOneRec commit's prompt and HR/NDCG definitions."
        )
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--info-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=OFFICIAL_NUM_BEAMS)
    parser.add_argument(
        "--max-new-tokens", type=int, default=OFFICIAL_MAX_NEW_TOKENS
    )
    parser.add_argument(
        "--length-penalty", type=float, default=OFFICIAL_LENGTH_PENALTY
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    """Run evaluation and save one metrics JSON file."""

    parser = build_parser()
    arguments = parser.parse_args()
    output_file = arguments.output_file
    if output_file.exists():
        parser.error(f"Refusing to overwrite existing output: {output_file}")

    try:
        result = evaluate_sft_model(
            model_path=arguments.model_path,
            test_file=arguments.test_file,
            info_file=arguments.info_file,
            device=arguments.device,
            batch_size=arguments.batch_size,
            num_beams=arguments.num_beams,
            max_new_tokens=arguments.max_new_tokens,
            length_penalty=arguments.length_penalty,
            sample=arguments.sample,
            seed=arguments.seed,
        )
    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"output_file": str(output_file), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
