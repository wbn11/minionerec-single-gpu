#!/usr/bin/env python3
"""Generate concise statistics for a processed MiniOneRec dataset."""

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

from minionerec.data.dataset_statistics import write_dataset_statistics  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize full user sequences, actual split histories, and official "
            "recent-10 truncation without reading raw Amazon JSONL files."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing .inter.json, .item.json, and split .inter files",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Common filename prefix of the processed dataset files",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Defaults to DATA_DIR/DATASET_NAME.data_stats.json",
    )
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        output_file, statistics = write_dataset_statistics(
            data_dir=arguments.data_dir,
            dataset_name=arguments.dataset_name,
            output_file=arguments.output_file,
        )
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {"statistics_file": str(output_file), **statistics},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
