#!/usr/bin/env python3
"""Convert processed MiniOneRec interactions to official SFT/RL CSV files."""

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

from minionerec.data.conversion import convert_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert .inter, .item.json, and .index.json files to the "
            "official MiniOneRec train/valid/test/info layout."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing the processed files for one dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Industrial_and_Scientific",
        help="Common filename prefix of the processed dataset files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New root directory under which train/valid/test/info are created.",
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        default=None,
        help=(
            "Optional Semantic ID index path; defaults to "
            "<data-dir>/<dataset-name>.index.json."
        ),
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Complete filename stem shared by train, valid, test and info.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        summary = convert_dataset(
            data_dir=arguments.data_dir,
            dataset_name=arguments.dataset_name,
            output_dir=arguments.output_dir,
            output_name=arguments.output_name,
            index_file=arguments.index_file,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
