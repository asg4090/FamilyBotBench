#!/usr/bin/env python3
"""Build graph algorithm recall/drop matrix figures for presentation."""

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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "GraphStats_LR",
    "GraphStats_EBM",
    "GraphStats_RF",
    "GraphSAGE_Inductive",
    "RGCN_Transductive",
]

MODEL_LABELS = {
    "GraphStats_LR": "GraphStats LR",
    "GraphStats_EBM": "GraphStats EBM",
    "GraphStats_RF": "GraphStats RF",
    "GraphSAGE_Inductive": "GraphSAGE",
    "RGCN_Transductive": "RGCN",
}

FAMILY_LABELS = {
    "G1_population_calibrated": "LOFO G1",
    "G2_anchor_behavior": "LOFO G2",
    "G3_contextual_interactive": "LOFO G3",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def ensure_fig_dir(result_dir: Path) -> Path:
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def all_metrics(result_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ["graphstats_metrics.csv", "graphsage_metrics.csv", "rgcn_metrics.csv"]:
        df = read_csv(result_dir / name)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["bot_recall"] = pd.to_numeric(df["bot_recall"], errors="coerce")
    return df


def pick_value(df: pd.DataFrame, model: str, split_name: str, family: str = "ALL") -> float | None:
    view = df[
        (df["model"] == model)
        & (df["split_name"] == split_name)
        & (df["eval_group"] == "test")
        & (df["family"] == family)
    ]
    if view.empty:
        return None
    return float(view.iloc[0]["bot_recall"])


def lofo_mean(df: pd.DataFrame, model: str) -> float | None:
    values = []
    for split_name, family in [
        ("lofo_g1", "G1_population_calibrated"),
        ("lofo_g2", "G2_anchor_behavior"),
        ("lofo_g3", "G3_contextual_interactive"),
    ]:
        value = pick_value(df, model, split_name, family)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(np.mean(values))


def build_compact_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "Zero-shot OOD": pick_value(df, model, "original_only_zero_shot"),
                "LOFO mean": lofo_mean(df, model),
                "1% labels": pick_value(df, model, "label_scarcity_001pct"),
                "10% labels": pick_value(df, model, "label_scarcity_010pct"),
                "50% labels": pick_value(df, model, "label_scarcity_050pct"),
            }
        )
    return pd.DataFrame(rows)


def build_detailed_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        row = {"model": MODEL_LABELS[model], "Zero-shot OOD": pick_value(df, model, "original_only_zero_shot")}
        for family, label in FAMILY_LABELS.items():
            split_name = f"lofo_{label.split()[-1].lower()}"
            row[label] = pick_value(df, model, split_name, family)
        row["1% labels"] = pick_value(df, model, "label_scarcity_001pct")
        row["10% labels"] = pick_value(df, model, "label_scarcity_010pct")
        row["50% labels"] = pick_value(df, model, "label_scarcity_050pct")
        rows.append(row)
    return pd.DataFrame(rows)


def build_drop_matrix(compact: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [col for col in compact.columns if col != "model"]
    for _, row in compact.iterrows():
        values = [float(row[col]) for col in metric_cols if pd.notna(row[col])]
        ref = max(values) if values else 0.0
        out = {"model": row["model"], "reference_recall": ref}
        for col in metric_cols:
            value = row[col]
            if pd.isna(value) or ref <= 0:
                out[col] = np.nan
            else:
                out[col] = ((float(value) - ref) / ref) * 100.0
        rows.append(out)
    return pd.DataFrame(rows)


def plot_recall_heatmap(detailed: pd.DataFrame, fig_dir: Path) -> Path:
    value_cols = [col for col in detailed.columns if col != "model"]
    data = detailed[value_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(value_cols)), labels=value_cols, rotation=18, ha="right")
    ax.set_yticks(range(len(detailed)), labels=detailed["model"].tolist())
    ax.set_title("Graph algorithm absolute recall across transfer settings")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Bot recall")
    fig.tight_layout()
    out = fig_dir / "graph_algorithm_recall_matrix.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def drop_color(drop_pct: float) -> tuple[float, float, float, float]:
    if not np.isfinite(drop_pct):
        return (0.85, 0.85, 0.85, 1.0)
    if drop_pct <= -95:
        return mpl.colors.to_rgba("#111111")
    if drop_pct <= -70:
        return mpl.colors.to_rgba("#6A040F")
    if drop_pct <= -40:
        return mpl.colors.to_rgba("#D62828")
    if drop_pct <= -20:
        return mpl.colors.to_rgba("#F77F00")
    if drop_pct <= -5:
        return mpl.colors.to_rgba("#F9C74F")
    return mpl.colors.to_rgba("#90BE6D")


def drop_label(drop_pct: float) -> str:
    if not np.isfinite(drop_pct):
        return "NA"
    if drop_pct <= -95:
        return f"{drop_pct:.0f}% (collapse)"
    return f"{drop_pct:.0f}%"


def plot_drop_matrix(compact_drop: pd.DataFrame, fig_dir: Path) -> Path:
    metric_cols = [col for col in compact_drop.columns if col not in {"model", "reference_recall"}]
    n_rows = len(compact_drop)
    n_cols = len(metric_cols)
    fig_w = 2.6 + 2.1 * n_cols
    fig_h = 1.6 + 1.05 * n_rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.9)
    ax.axis("off")

    for col_idx, col in enumerate(metric_cols):
        ax.text(col_idx, -0.55, col, ha="center", va="center", fontsize=13, fontweight="bold")

    for row_idx, (_, row) in enumerate(compact_drop.iterrows()):
        ax.text(-0.78, row_idx, row["model"], ha="right", va="center", fontsize=13, fontweight="bold")
        for col_idx, col in enumerate(metric_cols):
            value = float(row[col]) if pd.notna(row[col]) else np.nan
            color = drop_color(value)
            square = plt.Rectangle((col_idx - 0.18, row_idx - 0.18), 0.24, 0.24, color=color, ec="#333333", lw=0.6)
            ax.add_patch(square)
            ax.text(col_idx + 0.12, row_idx, drop_label(value), ha="left", va="center", fontsize=12, color="#222222")

    ax.set_title("Graph algorithm relative recall drop by transfer setting", fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    out = fig_dir / "graph_algorithm_drop_matrix.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    fig_dir = ensure_fig_dir(result_dir)
    metrics = all_metrics(result_dir)
    if metrics.empty:
        raise FileNotFoundError(f"No metrics found in {result_dir}")

    compact = build_compact_matrix(metrics)
    detailed = build_detailed_matrix(metrics)
    compact_drop = build_drop_matrix(compact)

    compact.to_csv(result_dir / "graph_algorithm_compact_recall_matrix.csv", index=False, encoding="utf-8-sig")
    detailed.to_csv(result_dir / "graph_algorithm_detailed_recall_matrix.csv", index=False, encoding="utf-8-sig")
    compact_drop.to_csv(result_dir / "graph_algorithm_relative_drop_matrix.csv", index=False, encoding="utf-8-sig")

    recall_fig = plot_recall_heatmap(detailed, fig_dir)
    drop_fig = plot_drop_matrix(compact_drop, fig_dir)

    note = [
        "# Graph Algorithm Drop Matrix Note",
        "",
        "- `graph_algorithm_recall_matrix.png`: absolute recall across zero-shot, LOFO, and label-scarcity settings.",
        "- `graph_algorithm_drop_matrix.png`: relative recall drop, computed against each algorithm's best recall among the compact settings.",
        "- `collapse` means relative drop <= -95%.",
    ]
    (result_dir / "GRAPH_ALGORITHM_DROP_MATRIX_NOTE.md").write_text("\n".join(note) + "\n", encoding="utf-8")

    print(recall_fig)
    print(drop_fig)


if __name__ == "__main__":
    main()
