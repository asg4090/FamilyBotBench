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
import pandas as pd


def short_family(name: str) -> str:
    return str(name).split("_", 1)[0]


def plot_protocol_means(metrics: pd.DataFrame, figures_dir: Path) -> None:
    order = [
        "original_only_zero_shot",
        "single_family_transfer",
        "leave_one_family_out",
        "incremental_curriculum",
    ]
    pivot = metrics.groupby(["protocol", "model"])["recall"].mean().unstack().reindex(order).fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pivot.plot(kind="bar", ax=ax, color={"tfidf": "#4C78A8", "bge": "#F58518"})
    ax.set_title("Mean Bot Recall by Protocol")
    ax.set_ylabel("Recall")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(title="Detector")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticklabels([label.replace("_", "\n") for label in pivot.index], rotation=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "protocol_mean_recall.png", dpi=180)
    plt.close(fig)


def plot_family_protocol(metrics: pd.DataFrame, figures_dir: Path) -> None:
    subset = metrics[metrics["protocol"].isin(["original_only_zero_shot", "leave_one_family_out"])]
    for model in sorted(subset["model"].unique()):
        data = subset[subset["model"].eq(model)].pivot_table(
            index="eval_family",
            columns="protocol",
            values="recall",
            aggfunc="mean",
        )
        data = data.rename(index=short_family)
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        im = ax.imshow(data.fillna(0.0).to_numpy(), vmin=0, vmax=1, cmap="YlOrRd")
        ax.set_title(f"{model.upper()} Recall: Zero-shot vs LOFO")
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels([c.replace("_", "\n") for c in data.columns], rotation=0)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{model}_zero_vs_lofo_heatmap.png", dpi=180)
        plt.close(fig)


def plot_single_transfer(metrics: pd.DataFrame, figures_dir: Path) -> None:
    subset = metrics[metrics["protocol"].eq("single_family_transfer")]
    for model in sorted(subset["model"].unique()):
        data = subset[subset["model"].eq(model)].pivot_table(
            index="train_family",
            columns="eval_family",
            values="recall",
            aggfunc="mean",
        )
        data = data.rename(index=short_family, columns=short_family)
        fig, ax = plt.subplots(figsize=(7, 5.8))
        im = ax.imshow(data.fillna(0.0).to_numpy(), vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"{model.upper()} Single-Family Transfer Recall")
        ax.set_xlabel("Eval family")
        ax.set_ylabel("Train family")
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{model}_single_family_transfer_heatmap.png", dpi=180)
        plt.close(fig)


def append_figure_section(report_path: Path) -> None:
    marker = "\n## 6. 图表\n"
    text = report_path.read_text(encoding="utf-8")
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    text += marker
    text += "\n".join(
        [
            "",
            "![Protocol Mean Recall](figures/protocol_mean_recall.png)",
            "",
            "![TF-IDF Zero vs LOFO](figures/tfidf_zero_vs_lofo_heatmap.png)",
            "",
            "![BGE Zero vs LOFO](figures/bge_zero_vs_lofo_heatmap.png)",
            "",
            "![TF-IDF Single Family Transfer](figures/tfidf_single_family_transfer_heatmap.png)",
            "",
            "![BGE Single Family Transfer](figures/bge_single_family_transfer_heatmap.png)",
            "",
        ]
    )
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metrics_path = output_dir / "static_language_family_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_protocol_means(metrics, figures_dir)
    plot_family_protocol(metrics, figures_dir)
    plot_single_transfer(metrics, figures_dir)
    append_figure_section(output_dir / "STATIC_LANGUAGE_FAMILY_REPORT_CN.md")
    print(figures_dir)


if __name__ == "__main__":
    main()
