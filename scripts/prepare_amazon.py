#!/usr/bin/env python3
"""Prepare an Amazon18 category with the official MiniOneRec protocol."""

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

from minionerec.data.amazon18 import Amazon18Config, process_amazon18  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert uncompressed Amazon Review 2018 metadata/review JSONL "
            "into MiniOneRec's eight processed artifacts."
        )
    )
    parser.add_argument("--dataset", required=True, help="Output dataset/category name")
    parser.add_argument(
        "--metadata-file", required=True, type=Path, help="Uncompressed metadata JSONL"
    )
    parser.add_argument(
        "--reviews-file", required=True, type=Path, help="Uncompressed reviews JSONL"
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New dataset directory is created below this path",
    )
    parser.add_argument("--k-core", type=int, default=5)
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--start-month", type=int, default=10)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument("--end-month", type=int, default=11)
    parser.add_argument("--minimum-items", type=int, default=3000)
    parser.add_argument("--earliest-year", type=int, default=1996)
    parser.add_argument(
        "--no-start-year-expansion",
        action="store_false",
        dest="expand_start_year",
        help="Disable the official <3000-item recursive start-year expansion",
    )
    parser.set_defaults(expand_start_year=True)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    config = Amazon18Config(
        dataset=arguments.dataset,
        metadata_file=arguments.metadata_file,
        reviews_file=arguments.reviews_file,
        output_root=arguments.output_root,
        k_core=arguments.k_core,
        start_year=arguments.start_year,
        start_month=arguments.start_month,
        end_year=arguments.end_year,
        end_month=arguments.end_month,
        expand_start_year=arguments.expand_start_year,
        minimum_items=arguments.minimum_items,
        earliest_year=arguments.earliest_year,
    )
    try:
        summary = process_amazon18(config)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
