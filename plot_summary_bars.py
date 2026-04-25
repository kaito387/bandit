"""Plot algorithm comparison bar charts from a summary.json file.

This script reads the algorithms_summary section and renders a 2x2 bar chart
comparison for:
- avgCost
- bestPathRate
- finalAccumRegret
- finalAvgRegret
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _require_wandb() -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is required for this workflow. Install it with: pip install wandb"
        ) from exc
    return wandb


METRICS: Sequence[tuple[str, str, str]] = (
    ("avgCost", "avgCost", "avgCost"),
    ("bestPathRate", "bestPathRate", "bestPathRate"),
    ("finalAccumRegret", "finalAccumRegret", "finalAccumRegret"),
    ("finalAvgRegret", "finalAvgRegret", "finalAvgRegret"),
)


def _load_summary(summary_path: Path) -> tuple[str, Dict[str, Dict[str, Any]]]:
    if not summary_path.exists():
        raise FileNotFoundError(f"summary file not found: {summary_path}")

    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("summary.json must be a JSON object")

    algorithms_summary = raw.get("algorithms_summary")
    if not isinstance(algorithms_summary, dict) or not algorithms_summary:
        raise ValueError("summary.json must contain a non-empty algorithms_summary object")

    env_name = str(raw.get("env_name", summary_path.parent.name))
    return env_name, algorithms_summary


def _extract_metric(
    algorithms_summary: Dict[str, Dict[str, Any]],
    metric: str,
) -> tuple[List[str], np.ndarray]:
    algorithms = list(algorithms_summary.keys())
    values: List[float] = []

    for algo in algorithms:
        summary = algorithms_summary[algo]
        if not isinstance(summary, dict):
            raise ValueError(f"algorithms_summary[{algo}] must be an object")
        if metric not in summary:
            raise ValueError(f"algorithms_summary[{algo}] missing field: {metric}")
        try:
            values.append(float(summary[metric]))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"algorithms_summary[{algo}].{metric} must be numeric") from exc

    return algorithms, np.asarray(values, dtype=np.float64)


def _annotate_bars(ax: plt.Axes, bars: Any, values: np.ndarray) -> None:
    y_min, y_max = ax.get_ylim()
    span = max(y_max - y_min, 1e-12)
    offset = span * 0.02

    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        ax.text(
            x,
            y + offset,
            f"{value:.4g}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )


def plot_summary_bars(summary_path: Path) -> None:
    wandb = _require_wandb()
    env_name, algorithms_summary = _load_summary(summary_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.ravel()
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    for idx, (metric, title, ylabel) in enumerate(METRICS):
        ax = axes_flat[idx]
        algorithms, values = _extract_metric(algorithms_summary, metric)
        x = np.arange(len(algorithms))
        bars = ax.bar(x, values, color=colors[idx % len(colors)], width=0.68)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if metric == "bestPathRate":
            upper = max(1.0, float(np.max(values)) * 1.1 if values.size else 1.0)
            ax.set_ylim(0.0, upper)
        else:
            lower = min(0.0, float(np.min(values)) * 1.05 if values.size else 0.0)
            upper = float(np.max(values)) * 1.08 if values.size else 1.0
            if upper <= lower:
                upper = lower + 1.0
            ax.set_ylim(lower, upper)

        _annotate_bars(ax, bars, values)

    fig.suptitle(f"Algorithm Summary Comparison ({env_name})", fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    wandb.log({"charts/summary_bars": wandb.Image(fig)})
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot algorithm comparison bar charts from a summary.json file",
    )
    parser.add_argument("--input", required=True, help="path to summary.json")
    parser.add_argument("--wandb-project", default="bandit", help="WandB project name")
    parser.add_argument("--wandb-entity", default="kaito15-sun-yat-sen-university", help="WandB entity/team")
    parser.add_argument("--wandb-group", default=None, help="optional WandB group")
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="WandB mode",
    )
    return parser.parse_args()


def main() -> None:
    wandb = _require_wandb()
    args = parse_args()
    summary_path = Path(args.input)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type="summary_bars",
        name=f"{summary_path.parent.name}-summary-bars",
        mode=args.wandb_mode,
        reinit=True,
    )
    try:
        plot_summary_bars(summary_path)
    finally:
        run.finish()

    print(f"logged summary bars from {summary_path} to WandB")


if __name__ == "__main__":
    main()