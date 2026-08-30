#!/usr/bin/env python3
"""Generate dependency-free SVG figures for the project README."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    REPOSITORY_ROOT
    / "innovations/cgrf_hierarchical_grpo/experiment_summary.json"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "assets/figures"

COLORS = {
    "ink": "#263238",
    "muted": "#66757F",
    "grid": "#D9E1E6",
    "blue": "#4C78A8",
    "blue_light": "#A9C7E5",
    "orange": "#E6A04B",
    "red": "#D76666",
    "green": "#55A678",
    "stage_blue": "#EEF5FB",
    "stage_orange": "#FFF4E8",
    "stage_red": "#FDEEEE",
    "stage_green": "#EDF7F1",
    "white": "#FFFFFF",
}


class SVG:
    """Small SVG builder with consistent scientific-figure styling."""

    def __init__(self, width: int, height: int, title: str, description: str):
        self.width = width
        self.height = height
        self.title = title
        self.description = description
        self.elements: list[str] = []

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 1,
        radius: float = 0,
        dash: str | None = None,
    ) -> None:
        dash_value = f' stroke-dasharray="{dash}"' if dash else ""
        self.elements.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
            f'height="{height:.1f}" rx="{radius:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.1f}"{dash_value}/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str,
        width: float = 1,
        dash: str | None = None,
        arrow: bool = False,
    ) -> None:
        dash_value = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.elements.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
            f'y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width:.1f}"'
            f'{dash_value}{marker}/>'
        )

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        *,
        stroke: str,
        width: float = 1,
        dash: str | None = None,
        arrow: bool = False,
    ) -> None:
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash_value = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.elements.append(
            f'<polyline points="{point_text}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width:.1f}"{dash_value}{marker}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: int = 16,
        weight: int = 400,
        fill: str | None = None,
        anchor: str = "start",
        family: str = "Arial, Helvetica, sans-serif",
        rotate: float | None = None,
    ) -> None:
        transform = (
            f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"'
            if rotate is not None
            else ""
        )
        self.elements.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" '
            f'fill="{fill or COLORS["ink"]}" text-anchor="{anchor}"'
            f'{transform}>{escape(value)}</text>'
        )

    def multiline_text(
        self,
        x: float,
        y: float,
        lines: Sequence[str],
        *,
        size: int = 16,
        weight: int = 400,
        fill: str | None = None,
        anchor: str = "middle",
        line_gap: float = 1.25,
    ) -> None:
        spans = []
        for index, line in enumerate(lines):
            dy = 0 if index == 0 else size * line_gap
            spans.append(
                f'<tspan x="{x:.1f}" dy="{dy:.1f}">{escape(line)}</tspan>'
            )
        self.elements.append(
            f'<text x="{x:.1f}" y="{y:.1f}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill or COLORS["ink"]}" '
            f'text-anchor="{anchor}">{"".join(spans)}</text>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        *,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 1,
    ) -> None:
        self.elements.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>'
        )

    def render(self) -> str:
        return "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
                f'height="{self.height}" viewBox="0 0 {self.width} {self.height}" '
                'role="img">',
                f"<title>{escape(self.title)}</title>",
                f"<desc>{escape(self.description)}</desc>",
                "<defs>",
                '<marker id="arrow" markerWidth="10" markerHeight="10" '
                'refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
                f'<path d="M0,0 L0,6 L9,3 z" fill="{COLORS["muted"]}"/>',
                "</marker>",
                '<filter id="soft-shadow" x="-10%" y="-10%" width="120%" height="130%">',
                '<feDropShadow dx="0" dy="2" stdDeviation="2" flood-opacity="0.12"/>',
                "</filter>",
                "</defs>",
                f'<rect width="100%" height="100%" fill="{COLORS["white"]}"/>',
                *self.elements,
                "</svg>",
                "",
            ]
        )


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise TypeError("summary must contain a JSON object")
    return summary


def _write_svg(canvas: SVG, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canvas.render(), encoding="utf-8", newline="\n")


def _figure_header(canvas: SVG, title: str, subtitle: str) -> None:
    canvas.text(54, 54, title, size=30, weight=700)
    canvas.text(54, 84, subtitle, size=15, fill=COLORS["muted"])
    canvas.line(54, 101, canvas.width - 54, 101, stroke=COLORS["grid"], width=1.2)


def _legend(
    canvas: SVG,
    entries: Sequence[tuple[str, str]],
    *,
    x: float,
    y: float,
    item_width: float,
) -> None:
    for index, (label, color) in enumerate(entries):
        offset = x + index * item_width
        canvas.rect(offset, y - 13, 18, 13, fill=color, radius=2)
        canvas.text(offset + 26, y - 1, label, size=14)


def _draw_axes(
    canvas: SVG,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
    ticks: Sequence[float],
    y_format: str,
) -> None:
    for tick in ticks:
        py = y + height - (tick - y_min) / (y_max - y_min) * height
        canvas.line(x, py, x + width, py, stroke=COLORS["grid"], width=1)
        canvas.text(
            x - 12,
            py + 5,
            format(tick, y_format),
            size=12,
            fill=COLORS["muted"],
            anchor="end",
        )
    canvas.line(x, y, x, y + height, stroke=COLORS["ink"], width=1.3)
    canvas.line(
        x,
        y + height,
        x + width,
        y + height,
        stroke=COLORS["ink"],
        width=1.3,
    )


def build_performance_figure(summary: dict) -> SVG:
    canvas = SVG(
        1440,
        720,
        "Recommendation performance comparison",
        "Grouped bars compare SFT, MiniOneRec GRPO, and CGRF-H across HR and NDCG cutoffs.",
    )
    _figure_header(
        canvas,
        "Recommendation Performance",
        "Industrial_and_Scientific · 4,533 test cases · constrained beam search · width 50",
    )
    series = [
        ("SFT", COLORS["blue_light"], summary["sid_metrics"]["sft"]),
        ("MiniOneRec GRPO", COLORS["orange"], summary["sid_metrics"]["baseline_grpo"]),
        ("CGRF-H", COLORS["red"], summary["sid_metrics"]["cgrf_h"]),
    ]
    _legend(
        canvas,
        [(label, color) for label, color, _ in series],
        x=845,
        y=76,
        item_width=170,
    )
    cutoffs = [1, 3, 5, 10, 20, 50]
    panels = [
        ("Hit Rate", "HR", 0.27, [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]),
        ("NDCG", "NDCG", 0.13, [0.00, 0.025, 0.05, 0.075, 0.10, 0.125]),
    ]
    panel_positions = [(92, 160), (770, 160)]
    plot_width = 575
    plot_height = 440
    for (panel_title, prefix, y_max, ticks), (plot_x, plot_y) in zip(
        panels, panel_positions
    ):
        canvas.text(plot_x, 132, panel_title, size=21, weight=700)
        _draw_axes(
            canvas,
            x=plot_x,
            y=plot_y,
            width=plot_width,
            height=plot_height,
            y_min=0,
            y_max=y_max,
            ticks=ticks,
            y_format=".3f" if prefix == "NDCG" else ".2f",
        )
        group_width = plot_width / len(cutoffs)
        bar_width = 21
        for group_index, cutoff in enumerate(cutoffs):
            center = plot_x + (group_index + 0.5) * group_width
            for series_index, (_, color, values) in enumerate(series):
                value = float(values[f"{prefix}@{cutoff}"])
                bar_height = value / y_max * plot_height
                bar_x = center + (series_index - 1) * (bar_width + 4) - bar_width / 2
                bar_y = plot_y + plot_height - bar_height
                canvas.rect(
                    bar_x,
                    bar_y,
                    bar_width,
                    bar_height,
                    fill=color,
                    radius=2,
                )
                if series_index == 2:
                    canvas.text(
                        bar_x + bar_width / 2,
                        bar_y - 8,
                        f"{value:.3f}",
                        size=11,
                        weight=700,
                        fill=COLORS["red"],
                        anchor="middle",
                    )
            canvas.text(
                center,
                plot_y + plot_height + 27,
                str(cutoff),
                size=13,
                anchor="middle",
            )
        canvas.text(
            plot_x + plot_width / 2,
            plot_y + plot_height + 58,
            "Cutoff K",
            size=14,
            weight=700,
            anchor="middle",
        )
        canvas.text(
            plot_x - 62,
            plot_y + plot_height / 2,
            "Score",
            size=14,
            weight=700,
            anchor="middle",
            rotate=-90,
        )
    canvas.text(
        720,
        690,
        "CGRF-H improves the candidate range at K ≥ 5 while slightly trading off HR/NDCG at K = 1 and 3.",
        size=14,
        fill=COLORS["muted"],
        anchor="middle",
    )
    return canvas


def build_relative_gain_figure(summary: dict) -> SVG:
    canvas = SVG(
        1320,
        660,
        "CGRF-H relative gain over MiniOneRec GRPO",
        "Diverging bars show relative percentage changes for HR and NDCG at six cutoffs.",
    )
    _figure_header(
        canvas,
        "CGRF-H Relative Change vs. MiniOneRec GRPO",
        "Positive values indicate improvement; negative values indicate a trade-off",
    )
    comparison = summary["sid_metrics"]["cgrf_h_vs_baseline_grpo"]
    cutoffs = [1, 3, 5, 10, 20, 50]
    entries = [("HR", COLORS["blue"]), ("NDCG", COLORS["red"])]
    _legend(canvas, entries, x=995, y=76, item_width=125)
    plot_x, plot_y, plot_width, plot_height = 105, 155, 1130, 400
    y_min, y_max = -2.0, 4.5
    ticks = [-2, -1, 0, 1, 2, 3, 4]
    _draw_axes(
        canvas,
        x=plot_x,
        y=plot_y,
        width=plot_width,
        height=plot_height,
        y_min=y_min,
        y_max=y_max,
        ticks=ticks,
        y_format=".0f",
    )
    zero_y = plot_y + plot_height - (0 - y_min) / (y_max - y_min) * plot_height
    canvas.line(plot_x, zero_y, plot_x + plot_width, zero_y, stroke=COLORS["ink"], width=1.7)
    group_width = plot_width / len(cutoffs)
    bar_width = 38
    for group_index, cutoff in enumerate(cutoffs):
        center = plot_x + (group_index + 0.5) * group_width
        for series_index, (metric, color) in enumerate(entries):
            value = float(comparison[f"{metric}@{cutoff}"]["relative_percent"])
            value_y = plot_y + plot_height - (value - y_min) / (y_max - y_min) * plot_height
            top = min(value_y, zero_y)
            height = abs(value_y - zero_y)
            x = center + (series_index - 0.5) * (bar_width + 9) - bar_width / 2
            canvas.rect(x, top, bar_width, height, fill=color, radius=2)
            label_y = top - 9 if value >= 0 else top + height + 18
            canvas.text(
                x + bar_width / 2,
                label_y,
                f"{value:+.2f}%",
                size=12,
                weight=700,
                fill=color,
                anchor="middle",
            )
        canvas.text(
            center,
            plot_y + plot_height + 30,
            f"K={cutoff}",
            size=13,
            anchor="middle",
        )
    canvas.text(
        plot_x - 68,
        plot_y + plot_height / 2,
        "Relative change (%)",
        size=14,
        weight=700,
        anchor="middle",
        rotate=-90,
    )
    canvas.text(
        660,
        625,
        "Largest gain: HR@50 +3.95% (43 additional hits among 4,533 test cases)",
        size=15,
        fill=COLORS["muted"],
        anchor="middle",
    )
    return canvas


def _node(
    canvas: SVG,
    x: float,
    y: float,
    width: float,
    height: float,
    lines: Sequence[str],
    *,
    fill: str,
    stroke: str,
    title_size: int = 16,
    dashed: bool = False,
) -> None:
    canvas.rect(
        x,
        y,
        width,
        height,
        fill=fill,
        stroke=stroke,
        stroke_width=1.5,
        radius=10,
        dash="7 5" if dashed else None,
    )
    total_height = (len(lines) - 1) * title_size * 1.25
    canvas.multiline_text(
        x + width / 2,
        y + height / 2 - total_height / 2 + title_size * 0.35,
        lines,
        size=title_size,
        weight=700,
    )


def _stage(
    canvas: SVG,
    y: float,
    height: float,
    label: str,
    fill: str,
    number: str,
) -> None:
    canvas.rect(42, y, 1516, height, fill=fill, radius=16)
    canvas.circle(76, y + 35, 19, fill=COLORS["ink"])
    canvas.text(76, y + 41, number, size=16, weight=700, fill=COLORS["white"], anchor="middle")
    canvas.text(108, y + 42, label, size=20, weight=700)


def build_system_figure() -> SVG:
    canvas = SVG(
        1600,
        930,
        "MiniOneRec reproduction and CGRF-H system",
        "Three-stage architecture covers semantic ID construction, supervised alignment, and MiniOneRec or CGRF-H reinforcement learning.",
    )
    _figure_header(
        canvas,
        "Single-A6000 MiniOneRec Reproduction with CGRF-H",
        "Paper-inspired end-to-end view · shared Semantic IDs and SFT initialization · training-only collaborative teacher",
    )

    _stage(canvas, 122, 205, "Semantic ID Construction", COLORS["stage_blue"], "1")
    _node(canvas, 100, 190, 180, 78, ["Item title", "+ description"], fill=COLORS["white"], stroke=COLORS["blue"])
    _node(canvas, 340, 190, 220, 78, ["Qwen3-Embedding-4B", "frozen text encoder"], fill=COLORS["white"], stroke=COLORS["blue"], title_size=15)
    _node(canvas, 620, 190, 170, 78, ["2,560-d", "item vector"], fill=COLORS["white"], stroke=COLORS["blue"])
    _node(canvas, 850, 177, 260, 104, ["RQ-VAE", "32-d latent", "3 × 256 codebooks"], fill="#FFE9DA", stroke=COLORS["orange"])
    _node(canvas, 1180, 177, 300, 104, ["Semantic ID", "<aᵢ> <bⱼ> <cₖ>", "3,673 unique SIDs"], fill=COLORS["white"], stroke=COLORS["blue"])
    for x1, x2 in [(280, 340), (560, 620), (790, 850), (1110, 1180)]:
        canvas.line(x1, 229, x2 - 10, 229, stroke=COLORS["muted"], width=2, arrow=True)

    _stage(canvas, 346, 165, "Supervised Full-Process Alignment", COLORS["stage_green"], "2")
    _node(canvas, 105, 404, 270, 72, ["User SID history", "+ item text tasks"], fill=COLORS["white"], stroke=COLORS["green"])
    _node(canvas, 490, 390, 390, 98, ["Qwen2.5-1.5B-Instruct", "full-parameter BF16 SFT"], fill="#FFF0C9", stroke=COLORS["orange"], title_size=19)
    _node(canvas, 1010, 404, 300, 72, ["SFT policy", "best validation checkpoint"], fill=COLORS["white"], stroke=COLORS["green"])
    canvas.line(375, 440, 480, 440, stroke=COLORS["muted"], width=2, arrow=True)
    canvas.line(880, 440, 1000, 440, stroke=COLORS["muted"], width=2, arrow=True)
    canvas.text(1395, 422, "SID vocabulary", size=14, weight=700, fill=COLORS["green"], anchor="middle")
    canvas.text(1395, 447, "540 added tokens", size=14, fill=COLORS["muted"], anchor="middle")

    _stage(canvas, 530, 324, "Recommendation-Oriented Reinforcement Learning", COLORS["stage_orange"], "3")
    _node(canvas, 78, 645, 160, 82, ["SFT policy", "Qwen"], fill=COLORS["white"], stroke=COLORS["orange"])
    _node(canvas, 278, 632, 190, 108, ["Constrained", "generation", "G = 16 valid SIDs"], fill=COLORS["white"], stroke=COLORS["orange"], title_size=15)
    _node(canvas, 510, 632, 170, 108, ["Candidate", "SID group", "distinct + valid"], fill=COLORS["white"], stroke=COLORS["orange"], title_size=15)
    canvas.line(238, 686, 268, 686, stroke=COLORS["muted"], width=2, arrow=True)
    canvas.line(468, 686, 500, 686, stroke=COLORS["muted"], width=2, arrow=True)

    _node(canvas, 735, 545, 220, 70, ["Official reward", "exact + ranking"], fill="#FFF7E8", stroke=COLORS["orange"], title_size=15)
    canvas.rect(
        710,
        625,
        565,
        190,
        fill="none",
        stroke=COLORS["red"],
        stroke_width=2,
        radius=14,
        dash="9 6",
    )
    canvas.rect(1000, 612, 164, 26, fill=COLORS["stage_orange"], radius=4)
    canvas.text(1010, 631, "CGRF-H (Ours)", size=15, weight=700, fill=COLORS["red"])

    _node(canvas, 735, 651, 220, 70, ["Hierarchical reward", "SID prefix: 0/0.2/0.5/1"], fill="#F2F6FC", stroke=COLORS["blue"], title_size=14)
    _node(canvas, 735, 735, 220, 70, ["SASRec teacher", "Item-ID collaborative rank"], fill=COLORS["stage_green"], stroke=COLORS["green"], title_size=14)
    canvas.polyline([(680, 670), (705, 670), (705, 580), (725, 580)], stroke=COLORS["muted"], width=1.7, arrow=True)
    canvas.line(680, 686, 725, 686, stroke=COLORS["muted"], width=1.7, arrow=True)
    canvas.polyline([(680, 703), (705, 703), (705, 770), (725, 770)], stroke=COLORS["muted"], width=1.7, arrow=True)

    _node(canvas, 1015, 651, 235, 70, ["Confidence gate g", "R = Rofficial + λ·Rdense", "λ = 0.1"], fill="#FDE7E7", stroke=COLORS["red"], title_size=14)
    canvas.line(955, 686, 1005, 686, stroke=COLORS["muted"], width=1.7, arrow=True)
    canvas.polyline([(955, 770), (985, 770), (985, 704), (1005, 704)], stroke=COLORS["muted"], width=1.7, arrow=True)
    canvas.polyline([(955, 580), (985, 580), (985, 668), (1005, 668)], stroke=COLORS["muted"], width=1.7, arrow=True)
    _node(canvas, 1300, 642, 115, 92, ["GRPO", "update"], fill="#F7DADA", stroke=COLORS["red"], title_size=17)
    canvas.polyline(
        [(955, 580), (1357.5, 580), (1357.5, 632)],
        stroke=COLORS["orange"],
        width=1.8,
        arrow=True,
    )
    canvas.text(1170, 569, "Official MiniOneRec path", size=12, fill=COLORS["orange"], anchor="middle")
    canvas.line(1250, 686, 1290, 686, stroke=COLORS["red"], width=2, arrow=True)
    _node(canvas, 1460, 623, 95, 126, ["Final", "Qwen", "policy"], fill=COLORS["white"], stroke=COLORS["red"], title_size=16)
    canvas.line(1415, 686, 1450, 686, stroke=COLORS["muted"], width=2, arrow=True)

    canvas.text(845, 837, "SASRec is used for GRPO training only", size=12, fill=COLORS["green"], anchor="middle")
    canvas.rect(82, 872, 1435, 34, fill="#F0F2F4", stroke=COLORS["grid"], radius=8)
    canvas.text(
        800,
        895,
        "Full-process SID alignment  •  shared data and SFT initialization  •  final inference uses Qwen only  •  Beam-50 HR/NDCG",
        size=15,
        weight=700,
        fill=COLORS["muted"],
        anchor="middle",
    )
    return canvas


def generate_figures(summary: dict, output_dir: Path) -> list[Path]:
    figures = {
        "minionerec-system.svg": build_system_figure(),
        "performance-comparison.svg": build_performance_figure(summary),
        "cgrf-relative-gain.svg": build_relative_gain_figure(summary),
    }
    paths = []
    for filename, canvas in figures.items():
        path = output_dir / filename
        _write_svg(canvas, path)
        paths.append(path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    summary = _load_summary(arguments.summary_file)
    for path in generate_figures(summary, arguments.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
