#!/usr/bin/env python3
from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(
    "dynamic_drift/continuous_generalization_formal_v1"
)
OUT_DIR = ROOT / "figures_pretty"

FAMILY_ORDER = ["A_language_drift", "C_graph_drift", "E_mixed_emergent_drift"]
FAMILY_SHORT = {
    "A_language_drift": "A",
    "C_graph_drift": "C",
    "E_mixed_emergent_drift": "E",
}
CASE_ORDER = [f"{fam}|{gen}" for fam in FAMILY_ORDER for gen in [1, 2, 3]]
TARGET_LABELS = [f"{FAMILY_SHORT[fam]}{gen}" for fam in FAMILY_ORDER for gen in [1, 2, 3]]

TEXT_MODELS = ["TFIDF_LR", "BGE_LR"]
TEXT_MODEL_LABELS = {
    "TFIDF_LR": "TF-IDF + LR",
    "BGE_LR": "BGE + LR",
}

GRAPH_MODELS = ["GraphStats_LR", "GraphStats_RF", "GraphStats_RGCN", "Constant_RGCN"]
GRAPH_MODEL_LABELS = {
    "GraphStats_LR": "GraphStats + LR",
    "GraphStats_RF": "GraphStats + RF",
    "GraphStats_RGCN": "GraphStats + RGCN",
    "Constant_RGCN": "Constant + RGCN",
}

PROTOCOL_TITLES = {
    "incremental_generation": (
        "Incremental-generation generalization",
        "Train bots: all earlier generations from A/C/E",
    ),
    "heldout_family_same_generation": (
        "Held-out-family transfer within the same generation",
        "Train bots: other families from the same generation",
    ),
}


def _parse_case(case: str) -> tuple[str, int]:
    family, generation = case.split("|")
    return family, int(generation)


def _source_label(protocol: str, family: str, generation: int) -> str:
    if protocol == "incremental_generation":
        if generation == 1:
            source = "none"
        elif generation == 2:
            source = "all g1"
        elif generation == 3:
            source = "all g1-2"
        else:
            source = "earlier gens"
        return source

    other_families = [fam for fam in FAMILY_ORDER if fam != family]
    return "+".join(f"{FAMILY_SHORT[fam]}{generation}" for fam in other_families)


def _column_labels(protocol: str) -> list[str]:
    labels = []
    for case in CASE_ORDER:
        family, generation = _parse_case(case)
        labels.append(_source_label(protocol, family, generation))
    return labels


def _pivot(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    work = df.copy()
    work["case"] = work["target_family"].astype(str) + "|" + work["target_generation"].astype(int).astype(str)
    pivot = work.pivot_table(index="model", columns="case", values="target_bot_recall", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=CASE_ORDER)
    return pivot


def _draw_matrix(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    row_labels: dict[str, str],
    title: str,
    subtitle: str,
    protocol: str,
) -> plt.AxesImage:
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(CASE_ORDER)))
    ax.set_xticklabels(TARGET_LABELS, fontsize=10)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([row_labels.get(idx, idx) for idx in pivot.index], fontsize=11)
    ax.set_xlabel("Target case", fontsize=12)
    ax.set_ylabel("Detector", fontsize=12)
    # Thin white cell borders.
    ax.set_xticks(np.arange(-0.5, len(CASE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Family separators.
    for xpos in [2.5, 5.5]:
        ax.axvline(x=xpos, color="#d9d9d9", linewidth=2.2)

    # Top axis shows train source per target case.
    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(np.arange(len(CASE_ORDER)), labels=_column_labels(protocol))
    top_ax.tick_params(axis="x", length=0, pad=4, labelsize=9, colors="#444444")
    top_ax.set_xlabel("Train source", fontsize=11, labelpad=8, color="#444444")
    for label in top_ax.get_xticklabels():
        label.set_fontweight("semibold")

    ax.text(
        0.5,
        1.145,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="semibold",
        color="#222222",
    )
    ax.text(
        0.5,
        1.095,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#444444",
    )

    # Numeric annotations.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                continue
            color = "white" if value < 0.55 else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10.5, color=color)

    return im


def _render_panel(
    df: pd.DataFrame,
    models: list[str],
    row_labels: dict[str, str],
    metric_kind: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(19.0, 8.2))
    last_im = None
    for ax, protocol in zip(axes, ["incremental_generation", "heldout_family_same_generation"]):
        sub = df[df["protocol"] == protocol].copy()
        pivot = _pivot(sub, models)
        title, subtitle = PROTOCOL_TITLES[protocol]
        last_im = _draw_matrix(ax, pivot, row_labels, title, subtitle, protocol)

    fig.subplots_adjust(left=0.09, right=0.955, top=0.72, bottom=0.14, wspace=0.18)
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Target-bot recall", fontsize=12)
    fig.suptitle(
        metric_kind,
        fontsize=18,
        fontweight="semibold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.055,
        "Column blocks are ordered as A1-3 | C1-3 | E1-3. Top axis = train source. Bottom axis = target case.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _render_single_protocol_panel(
    df: pd.DataFrame,
    models: list[str],
    row_labels: dict[str, str],
    metric_kind: str,
    protocol: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11.6, 8.0))
    sub = df[df["protocol"] == protocol].copy()
    pivot = _pivot(sub, models)
    title, subtitle = PROTOCOL_TITLES[protocol]
    im = _draw_matrix(ax, pivot, row_labels, title, subtitle, protocol)

    fig.subplots_adjust(left=0.16, right=0.90, top=0.72, bottom=0.14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.03)
    cbar.set_label("Target-bot recall", fontsize=12)
    fig.suptitle(
        metric_kind,
        fontsize=18,
        fontweight="semibold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.055,
        "Column blocks are ordered as A1-3 | C1-3 | E1-3. Top axis = train source. Bottom axis = target case.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _compute_gap(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    keys = ["model", "target_family", "target_generation"]
    inc = (
        df[df["protocol"] == "incremental_generation"][keys + ["target_bot_recall"]]
        .rename(columns={"target_bot_recall": "incremental_recall"})
    )
    hold = (
        df[df["protocol"] == "heldout_family_same_generation"][keys + ["target_bot_recall"]]
        .rename(columns={"target_bot_recall": "heldout_recall"})
    )
    merged = inc.merge(hold, on=keys, how="inner")
    merged["gap"] = merged["incremental_recall"] - merged["heldout_recall"]
    merged["case"] = merged["target_family"].astype(str) + "|" + merged["target_generation"].astype(int).astype(str)
    pivot = merged.pivot_table(index="model", columns="case", values="gap", aggfunc="mean")
    return pivot.reindex(index=models, columns=CASE_ORDER)


def _draw_gap(ax: plt.Axes, pivot: pd.DataFrame, row_labels: dict[str, str], panel_label: str) -> plt.AxesImage:
    data = pivot.to_numpy(dtype=float)
    vmax = max(0.25, float(np.nanmax(np.abs(data))))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(CASE_ORDER)))
    ax.set_xticklabels(TARGET_LABELS, fontsize=10)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([row_labels.get(idx, idx) for idx in pivot.index], fontsize=11)
    ax.set_xlabel("Target case", fontsize=12)
    ax.set_ylabel("Detector", fontsize=12)
    ax.text(
        0.0,
        1.035,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="semibold",
        color="#222222",
    )

    ax.set_xticks(np.arange(-0.5, len(CASE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for xpos in [2.5, 5.5]:
        ax.axvline(x=xpos, color="#d9d9d9", linewidth=2.2)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                continue
            color = "white" if abs(value) > vmax * 0.45 else "#222222"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=10.5, color=color)
    return im


def _render_gap_figure(text_df: pd.DataFrame, graph_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15.6, 10.4))
    text_gap = _compute_gap(text_df, TEXT_MODELS)
    graph_gap = _compute_gap(graph_df, GRAPH_MODELS)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.84, bottom=0.10, hspace=0.34)
    im = _draw_gap(
        axes[0],
        text_gap,
        TEXT_MODEL_LABELS,
        "Text detectors",
    )
    _draw_gap(
        axes[1],
        graph_gap,
        GRAPH_MODEL_LABELS,
        "Graph detectors",
    )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Recall gap", fontsize=12)
    fig.suptitle(
        "Generalization gap = incremental recall - held-out-family recall",
        fontsize=17.5,
        fontweight="semibold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.935,
        "Positive values: detector benefits more from earlier-generation exposure. Negative values: same-generation cross-family transfer is stronger.",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#444444",
    )
    fig.text(
        0.5,
        0.045,
        "Column blocks are ordered as A1-3 | C1-3 | E1-3. Each cell compares two source regimes for the same target case.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _render_two_layer_axis_demo(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 4.0))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    xs = [2.7, 6.6, 10.5]
    top_labels = ["none", "C2+E2", "all g1-2"]
    bottom_labels = ["A1", "A2", "A3"]
    top_colors = ["#dfeaf7", "#dfeaf7", "#dfeaf7"]
    bottom_colors = ["#f8ead6", "#f8ead6", "#f8ead6"]

    ax.text(
        6.6,
        3.88,
        "Two-layer x-axis: source -> target",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="semibold",
    )
    ax.text(
        1.0,
        2.85,
        "Top layer\n(train source)",
        ha="right",
        va="center",
        fontsize=12,
        color="#444444",
    )
    ax.text(
        1.0,
        1.15,
        "Bottom layer\n(target case)",
        ha="right",
        va="center",
        fontsize=12,
        color="#444444",
    )

    for x, top_label, bottom_label, top_color, bottom_color in zip(
        xs, top_labels, bottom_labels, top_colors, bottom_colors
    ):
        top_rect = Rectangle((x - 1.2, 2.35), 2.4, 0.85, facecolor=top_color, edgecolor="#8aa8c7", linewidth=1.6)
        bottom_rect = Rectangle((x - 1.2, 0.70), 2.4, 0.85, facecolor=bottom_color, edgecolor="#c8a57a", linewidth=1.6)
        ax.add_patch(top_rect)
        ax.add_patch(bottom_rect)
        ax.text(x, 2.775, top_label, ha="center", va="center", fontsize=13, fontweight="semibold")
        ax.text(x, 1.125, bottom_label, ha="center", va="center", fontsize=13, fontweight="semibold")
        ax.annotate(
            "",
            xy=(x, 1.55),
            xytext=(x, 2.35),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#666666"),
        )
        ax.text(x, 1.95, "one transfer\ntask", ha="center", va="center", fontsize=10, color="#555555")

    ax.text(
        6.6,
        0.20,
        "Example: top 'C2+E2' above bottom 'A2' means train on C2 and E2, then test on A2.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#444444",
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text_df = pd.read_csv(ROOT / "text_generalization_metrics_all.csv")
    graph_df = pd.read_csv(ROOT / "graph_generalization_metrics_all.csv")

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "axes.edgecolor": "#cfcfcf",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    _render_panel(
        text_df,
        TEXT_MODELS,
        TEXT_MODEL_LABELS,
        metric_kind="Text detectors",
        out_path=OUT_DIR / "text_detector_transfer_matrices.png",
    )
    _render_panel(
        graph_df,
        GRAPH_MODELS,
        GRAPH_MODEL_LABELS,
        metric_kind="Graph detectors",
        out_path=OUT_DIR / "graph_detector_transfer_matrices.png",
    )
    _render_single_protocol_panel(
        graph_df,
        GRAPH_MODELS,
        GRAPH_MODEL_LABELS,
        metric_kind="Graph detectors",
        protocol="incremental_generation",
        out_path=OUT_DIR / "graph_detector_incremental_generation.png",
    )
    _render_single_protocol_panel(
        graph_df,
        GRAPH_MODELS,
        GRAPH_MODEL_LABELS,
        metric_kind="Graph detectors",
        protocol="heldout_family_same_generation",
        out_path=OUT_DIR / "graph_detector_heldout_family_same_generation.png",
    )
    _render_gap_figure(
        text_df,
        graph_df,
        out_path=OUT_DIR / "generalization_gap_matrices.png",
    )
    _render_two_layer_axis_demo(
        out_path=OUT_DIR / "two_layer_axis_demo.png",
    )

    summary_lines = [
        "# Pretty Generalization Heatmaps",
        "",
        "- Transfer figures use a two-level x-axis: top = `train source`, bottom = `target case`.",
        "- Example: top label `C2+E2` above bottom label `A2` means `train on C2 and E2 -> test on A2`.",
        "- Example: top label `all g1-2` above bottom label `A3` means `train on all earlier generations 1-2 from A/C/E -> test on A3`.",
        "- `two_layer_axis_demo.png` is a schematic that shows how one column represents one explicit `source -> target` transfer task.",
        "- Rows: detector algorithms",
        "- Left panel: `incremental_generation`",
        "- Right panel: `heldout_family_same_generation`",
        "- Graph transfer is also exported as two separate figures: `graph_detector_incremental_generation.png` and `graph_detector_heldout_family_same_generation.png`.",
        "- Cell value: target-bot recall",
        "- Gap figure uses `incremental recall - held-out-family recall`; positive values mean the detector benefits more from earlier-generation exposure than from same-generation cross-family transfer.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
