#!/usr/bin/env python3
"""Build human-alignment proof figures for G1-G3 behavior families."""

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FAMILIES = [
    "G1_population_calibrated",
    "G2_anchor_behavior",
    "G3_contextual_interactive",
]

FAMILY_LABELS = {
    "original_human": "Human",
    "G1_population_calibrated": "G1",
    "G2_anchor_behavior": "G2",
    "G3_contextual_interactive": "G3",
}

ALIGNMENT_FEATURES = [
    "total_degree",
    "total_out_degree",
    "total_in_degree",
    "direction_balance",
    "comment1_rate",
    "comment2_rate",
    "receive_rate",
    "unique_target_authors",
]

FEATURE_LABELS = {
    "total_degree": "total degree",
    "total_out_degree": "out degree",
    "total_in_degree": "in degree",
    "direction_balance": "direction balance",
    "comment1_rate": "comment1 rate",
    "comment2_rate": "comment2 rate",
    "receive_rate": "receive rate",
    "unique_target_authors": "unique targets",
    "reciprocal_share": "reciprocal share",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def family_label(name: str) -> str:
    return FAMILY_LABELS.get(str(name), str(name))


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(str(name), str(name))


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_percentile_coverage(percentiles: pd.DataFrame, fig_dir: Path) -> tuple[pd.DataFrame, Path]:
    rows = []
    feature_names = sorted(
        c[: -len("_human_percentile")]
        for c in percentiles.columns
        if c.endswith("_human_percentile")
    )
    for family, fam_df in percentiles.groupby("family"):
        row = {"family": family}
        for feature in feature_names:
            col = f"{feature}_human_percentile"
            values = pd.to_numeric(fam_df[col], errors="coerce").dropna()
            row[f"{feature}_within_5_95_share"] = float(((values >= 5) & (values <= 95)).mean()) if len(values) else np.nan
        rows.append(row)
    coverage = pd.DataFrame(rows)
    coverage.to_csv(fig_dir.parent / "human_alignment_coverage_by_family.csv", index=False, encoding="utf-8-sig")

    plot_df = coverage.set_index("family")
    plot_df.index = [family_label(x) for x in plot_df.index]
    plot_df = plot_df[[f"{feature}_within_5_95_share" for feature in feature_names]]
    plot_df.columns = [feature_label(c.replace("_within_5_95_share", "")) for c in plot_df.columns]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(plot_df.to_numpy(dtype=float), cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(plot_df.columns)), labels=plot_df.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(plot_df.index)), labels=plot_df.index)
    ax.set_title("Share of bots within human 5th-95th percentile range")
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = plot_df.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Coverage share")
    out = fig_dir / "human_alignment_coverage_heatmap.png"
    savefig(out)
    return coverage, out


def plot_median_percentile_distance(alignment: pd.DataFrame, fig_dir: Path) -> tuple[pd.DataFrame, Path]:
    rows = []
    for _, row in alignment.iterrows():
        vals = []
        for feature in ALIGNMENT_FEATURES:
            col = f"{feature}_median_percentile"
            if col in row and pd.notna(row[col]):
                vals.append(abs(float(row[col]) - 50.0))
        rows.append(
            {
                "family": row["family"],
                "mean_abs_percentile_distance_from_50": float(np.mean(vals)) if vals else np.nan,
            }
        )
    dist = pd.DataFrame(rows)
    dist.to_csv(fig_dir.parent / "human_alignment_distance_summary.csv", index=False, encoding="utf-8-sig")

    plot_df = dist.copy()
    plot_df["family_label"] = plot_df["family"].map(family_label)
    plot_df = plot_df.sort_values("mean_abs_percentile_distance_from_50")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(plot_df["family_label"], plot_df["mean_abs_percentile_distance_from_50"], color=["#4C78A8", "#72B7B2", "#F58518"])
    ax.set_ylabel("Mean |median percentile - 50|")
    ax.set_xlabel("Family")
    ax.set_title("Overall distance from human median behavior")
    ax.grid(axis="y", alpha=0.25)
    for x, y in zip(plot_df["family_label"], plot_df["mean_abs_percentile_distance_from_50"]):
        ax.text(x, y + 0.3, f"{y:.1f}", ha="center", va="bottom", fontsize=9)
    out = fig_dir / "human_alignment_distance_bar.png"
    savefig(out)
    return dist, out


def clipped_values(series: pd.Series, low: float, high: float) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    return np.clip(values, low, high)


def plot_feature_distributions(features: pd.DataFrame, fig_dir: Path) -> Path:
    plot_df = features[features["family"].isin(["original_human", *FAMILIES])].copy()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    feature_list = [
        "total_degree",
        "direction_balance",
        "comment1_rate",
        "comment2_rate",
        "receive_rate",
        "unique_target_authors",
    ]
    axes = axes.flatten()
    for ax, feature in zip(axes, feature_list):
        human = pd.to_numeric(plot_df.loc[plot_df["family"] == "original_human", feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if human.empty:
            continue
        low = float(human.quantile(0.01))
        high = float(human.quantile(0.99))
        groups = []
        labels = []
        for family in ["original_human", *FAMILIES]:
            series = plot_df.loc[plot_df["family"] == family, feature]
            values = clipped_values(series, low, high)
            groups.append(values)
            labels.append(family_label(family))
        ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_title(feature_label(feature))
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Human vs G1-G3 behavior distribution overlap (clipped to human 1%-99%)", y=1.02)
    out = fig_dir / "human_alignment_feature_boxplots.png"
    savefig(out)
    return out


def plot_pca_overlap(features: pd.DataFrame, fig_dir: Path, seed: int) -> Path:
    cols = [f for f in ALIGNMENT_FEATURES if f in features.columns]
    plot_df = features[features["family"].isin(["original_human", *FAMILIES])].copy()
    human = plot_df[plot_df["family"] == "original_human"].copy()
    bots = plot_df[plot_df["family"].isin(FAMILIES)].copy()
    human_sample = human.sample(min(400, len(human)), random_state=seed) if len(human) else human
    merged = pd.concat([human_sample, bots], ignore_index=True)
    x = merged[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    x = StandardScaler().fit_transform(x)
    coords = PCA(n_components=2, random_state=seed).fit_transform(x)
    merged["pca1"] = coords[:, 0]
    merged["pca2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(8.2, 6))
    human_view = merged[merged["family"] == "original_human"]
    ax.scatter(human_view["pca1"], human_view["pca2"], s=18, alpha=0.22, color="#9E9E9E", label="Human sample")
    colors = {
        "G1_population_calibrated": "#4C78A8",
        "G2_anchor_behavior": "#72B7B2",
        "G3_contextual_interactive": "#F58518",
    }
    for family in FAMILIES:
        sub = merged[merged["family"] == family]
        ax.scatter(sub["pca1"], sub["pca2"], s=34, alpha=0.85, color=colors[family], label=family_label(family))
    ax.set_title("PCA overlap of human and generated behavior features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    ax.legend()
    out = fig_dir / "human_alignment_pca_overlap.png"
    savefig(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    fig_dir = ensure_dir(result_dir / "figures")
    percentiles = read_csv(result_dir / "human_percentiles_by_bot.csv")
    alignment = read_csv(result_dir / "human_alignment_by_family.csv")
    features = read_csv(result_dir / "graph_feature_by_bot.csv")
    if percentiles.empty or alignment.empty or features.empty:
        raise FileNotFoundError("Missing one of human_percentiles_by_bot.csv, human_alignment_by_family.csv, graph_feature_by_bot.csv")

    coverage, coverage_fig = plot_percentile_coverage(percentiles, fig_dir)
    distance, distance_fig = plot_median_percentile_distance(alignment, fig_dir)
    box_fig = plot_feature_distributions(features, fig_dir)
    pca_fig = plot_pca_overlap(features, fig_dir, args.seed)

    lines = [
        "# Human Alignment Visual Note",
        "",
        "- `human_percentile_by_family.png`: each family's median location inside the human percentile distribution; values near 50 are better aligned.",
        "- `human_alignment_coverage_heatmap.png`: share of bots whose feature values fall inside the human 5th-95th percentile range.",
        "- `human_alignment_distance_bar.png`: overall distance from the human median percentile; lower is better.",
        "- `human_alignment_feature_boxplots.png`: direct distribution overlap for key behavior features.",
        "- `human_alignment_pca_overlap.png`: whether G1-G3 occupy the same broad feature space as humans.",
        "",
        "Important: these figures support statistical alignment with human behavior distributions, not indistinguishability.",
    ]
    (result_dir / "HUMAN_ALIGNMENT_VISUAL_NOTE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(coverage_fig)
    print(distance_fig)
    print(box_fig)
    print(pca_fig)
    print((result_dir / "human_alignment_coverage_by_family.csv").resolve())
    print((result_dir / "human_alignment_distance_summary.csv").resolve())


if __name__ == "__main__":
    main()
