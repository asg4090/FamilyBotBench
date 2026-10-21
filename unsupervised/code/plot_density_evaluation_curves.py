#!/usr/bin/env python3
"""Plot label-revealed evaluation curves for density-based account discovery.

The discovery stage is label-free. This script only uses labels after ranked
suspicious clusters have already been produced, so the curves are evaluation
figures rather than training signals.
"""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Density experiment output directory containing density_suspicious_cluster_scores.csv.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=31,
        help="Maximum number of ranked suspicious clusters to merge.",
    )
    return parser.parse_args()


def load_operating_points(output_dir: Path, top_n: int) -> tuple[pd.DataFrame, float]:
    scores_path = output_dir / "density_suspicious_cluster_scores.csv"
    metadata_path = output_dir / "density_label_revealed_account_metadata.csv"
    scores = pd.read_csv(scores_path)
    metadata = pd.read_csv(metadata_path)

    label_by_account = dict(zip(metadata["account_id"].astype(str), metadata["eval_label"].astype(int)))
    total_bots = int(metadata["eval_label"].sum())
    total_accounts = int(len(metadata))
    base_bot_rate = float(total_bots / total_accounts) if total_accounts else 0.0

    selected: set[str] = set()
    rows: list[dict[str, float | int]] = []
    ranked = scores.sort_values("suspicious_rank").head(top_n)
    for row in ranked.itertuples(index=False):
        members_raw = getattr(row, "member_account_ids", "")
        members = [item for item in str(members_raw).split(";") if item]
        selected.update(members)
        selected_labels = [label_by_account.get(account_id, 0) for account_id in selected]
        selected_count = len(selected_labels)
        bot_count = int(sum(selected_labels))
        human_count = selected_count - bot_count
        precision = float(bot_count / selected_count) if selected_count else 0.0
        recall = float(bot_count / total_bots) if total_bots else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        false_positive_rate = float(human_count / (total_accounts - total_bots)) if total_accounts > total_bots else 0.0
        rows.append(
            {
                "top_k_clusters": int(getattr(row, "suspicious_rank")),
                "selected_account_count": int(selected_count),
                "bot_count": int(bot_count),
                "human_count": int(human_count),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": false_positive_rate,
                "lift_over_base_rate": float(precision / base_bot_rate) if base_bot_rate else 0.0,
                "base_bot_rate": base_bot_rate,
            }
        )
    return pd.DataFrame(rows), base_bot_rate


def plot_f1_curve(points: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(points["top_k_clusters"], points["f1"], marker="o", linewidth=2.0, color="#7B61FF", label="F1")
    best = points.sort_values(["f1", "top_k_clusters"], ascending=[False, True]).iloc[0]
    ax.scatter([best["top_k_clusters"]], [best["f1"]], s=95, color="#EB5757", zorder=3, label="best F1")
    ax.annotate(
        f"best Top-{int(best['top_k_clusters'])}\nF1={best['f1']:.3f}",
        xy=(best["top_k_clusters"], best["f1"]),
        xytext=(8, 16),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
    )
    ax.set_title("F1 curve after revealing labels")
    ax.set_xlabel("Top-K suspicious clusters merged")
    ax.set_ylabel("F1")
    ax.set_ylim(0, max(0.62, float(points["f1"].max()) + 0.1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    path = figures_dir / "density_f1_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_precision_recall(points: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    x_max = min(1.0, max(0.45, float(points["recall"].max()) + 0.05))
    scatter = ax.scatter(
        points["recall"],
        points["precision"],
        c=points["top_k_clusters"],
        s=55 + np.sqrt(points["selected_account_count"]) * 7,
        cmap="viridis",
        edgecolors="#222222",
        linewidths=0.35,
    )
    ax.plot(points["recall"], points["precision"], color="#333333", alpha=0.5, linewidth=1.2)

    for f1 in [0.2, 0.3, 0.4, 0.5, 0.6]:
        recall_grid = np.linspace(0.01, x_max, 300)
        precision_grid = (f1 * recall_grid) / (2 * recall_grid - f1)
        valid = (precision_grid >= 0) & (precision_grid <= 1) & (2 * recall_grid > f1)
        if valid.any():
            ax.plot(recall_grid[valid], precision_grid[valid], color="#999999", linestyle="--", linewidth=0.6)
            ax.text(
                recall_grid[valid][-1] - 0.01,
                precision_grid[valid][-1],
                f"F1={f1:.1f}",
                fontsize=7,
                color="#777777",
                ha="right",
                va="center",
            )

    best = points.sort_values(["f1", "top_k_clusters"], ascending=[False, True]).iloc[0]
    ax.scatter([best["recall"]], [best["precision"]], s=120, facecolors="none", edgecolors="#EB5757", linewidths=1.8)
    ax.annotate(
        f"best Top-{int(best['top_k_clusters'])}",
        xy=(best["recall"], best["precision"]),
        xytext=(-58, -24),
        textcoords="offset points",
        fontsize=9,
        ha="right",
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
    )

    ax.set_title("Precision-Recall trade-off")
    ax.set_xlabel("Bot recall")
    ax.set_ylabel("Bot precision")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.22)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Top-K suspicious clusters")
    fig.tight_layout()
    path = figures_dir / "density_precision_recall_tradeoff.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_unsupervised_lift(points: pd.DataFrame, base_bot_rate: float, figures_dir: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(9.2, 5.4))
    ax1.plot(
        points["top_k_clusters"],
        points["precision"],
        marker="o",
        linewidth=2.0,
        color="#EB5757",
        label="revealed bot precision among selected accounts",
    )
    ax1.axhline(base_bot_rate, color="#555555", linestyle=":", linewidth=1.2, label=f"dataset bot base rate = {base_bot_rate:.3f}")
    ax1.set_xlabel("Top-K suspicious clusters merged by label-free score")
    ax1.set_ylabel("Bot concentration / precision")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(
        points["top_k_clusters"],
        points["lift_over_base_rate"],
        marker="s",
        linewidth=1.6,
        color="#2F80ED",
        label="lift over base rate",
    )
    ax2.axhline(1.0, color="#2F80ED", linestyle=":", linewidth=0.9, alpha=0.8)
    ax2.set_ylabel("Lift over random selection")
    ax2.set_ylim(0, max(3.0, float(points["lift_over_base_rate"].max()) + 0.25))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)
    ax1.set_title("Label-free suspicious ranking effect after label reveal")
    fig.tight_layout()
    path = figures_dir / "density_unsupervised_effect_lift.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    points, base_bot_rate = load_operating_points(output_dir, args.top_n)
    points_path = output_dir / "density_unsupervised_operating_points.csv"
    points.to_csv(points_path, index=False)

    paths = {
        "operating_points": points_path,
        "f1": plot_f1_curve(points, figures_dir),
        "precision_recall": plot_precision_recall(points, figures_dir),
        "unsupervised_lift": plot_unsupervised_lift(points, base_bot_rate, figures_dir),
    }
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
