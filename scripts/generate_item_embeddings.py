#!/usr/bin/env python3
"""Generate MiniOneRec title-plus-description item embeddings."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from minionerec.semantic_id.embeddings import (  # noqa: E402
    OFFICIAL_MAX_LENGTH,
    generate_embeddings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Encode Amazon item title+description text with Qwen and masked "
            "mean pooling in contiguous item-ID order."
        )
    )
    parser.add_argument("--item-file", required=True, type=Path)
    parser.add_argument("--item-mapping-file", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="New .npy file; existing outputs are never overwritten",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=OFFICIAL_MAX_LENGTH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float16",
    )
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        manifest = generate_embeddings(
            item_file=arguments.item_file,
            item_mapping_file=arguments.item_mapping_file,
            model_path=arguments.model_path,
            output_file=arguments.output_file,
            batch_size=arguments.batch_size,
            max_length=arguments.max_length,
            device=arguments.device,
            torch_dtype=arguments.torch_dtype,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
