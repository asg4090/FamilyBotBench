#!/usr/bin/env python3
"""Run Static Text OOD detection experiments.

All OOD metrics in this script use scores where larger values indicate more
OOD-like samples. The supervised probability baselines are explicitly marked as
proxy baselines, not pure OOD methods.
"""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, pairwise_distances, roc_auc_score

from common_static_text import (
    FAMILIES,
    SHORT_NAME,
    FIGURES_DIR,
    RESULTS_DIR,
    family_train_indices,
    get_family_indices,
    get_original_train_val_test_indices,
    load_combined_data,
    predict_bge_lr,
    predict_tfidf_lr,
    save_figure,
    save_table,
    short_family_list,
    train_bge_lr,
    train_tfidf_lr,
)


OUTPUT_DIR = RESULTS_DIR / "static_text_ood"
SCORES_PATH = OUTPUT_DIR / "ood_scores_by_sample.csv"
METRICS_PATH = OUTPUT_DIR / "ood_metrics_by_family.csv"
SUMMARY_PATH = OUTPUT_DIR / "ood_method_summary.csv"
SCORE_DISTRIBUTION_PATH = FIGURES_DIR / "ood_score_distribution_by_family.png"
AUROC_FIGURE_PATH = FIGURES_DIR / "ood_auroc_by_method_family.png"
CENTROID_SCATTER_PATH = FIGURES_DIR / "bge_centroid_margin_scatter.png"

SCENARIOS = ["original_only_known", "lofo_known"]
KNN_K_VALUES = [5, 10, 20]
FAMILY_SHORT_ORDER = [SHORT_NAME[family] for family in FAMILIES]


@dataclass(frozen=True)
class MethodInfo:
    method: str
    score_column: str
    method_category: str
    feature_space: str
    score_direction: str
    is_supervised_proxy: bool
    notes: str


METHODS: dict[str, MethodInfo] = {
    "bge_centroid_min_distance": MethodInfo(
        method="bge_centroid_min_distance",
        score_column="centroid_min_distance",
        method_category="semi_supervised_ood",
        feature_space="L2-normalized BGE",
        score_direction="larger score = farther from nearest known human/bot centroid = more OOD",
        is_supervised_proxy=False,
        notes="Uses only known-train class centroids and labels for centroid grouping.",
    ),
    "bge_centroid_bot_margin": MethodInfo(
        method="bge_centroid_bot_margin",
        score_column="centroid_bot_minus_human_margin",
        method_category="semi_supervised_ood",
        feature_space="L2-normalized BGE",
        score_direction="larger score = farther from known bot centroid than human centroid = more bot-family OOD-like",
        is_supervised_proxy=False,
        notes="Margin is useful for unseen bot-family drift but can score ID humans high.",
    ),
    "bge_knn_k5_distance": MethodInfo(
        method="bge_knn_k5_distance",
        score_column="knn_k5_mean_distance",
        method_category="unsupervised_ood",
        feature_space="L2-normalized BGE",
        score_direction="larger score = larger mean distance to 5 nearest known-train samples = more OOD",
        is_supervised_proxy=False,
        notes="Nearest-neighbor distances are fit only on known train.",
    ),
    "bge_knn_k10_distance": MethodInfo(
        method="bge_knn_k10_distance",
        score_column="knn_k10_mean_distance",
        method_category="unsupervised_ood",
        feature_space="L2-normalized BGE",
        score_direction="larger score = larger mean distance to 10 nearest known-train samples = more OOD",
        is_supervised_proxy=False,
        notes="Nearest-neighbor distances are fit only on known train.",
    ),
    "bge_knn_k20_distance": MethodInfo(
        method="bge_knn_k20_distance",
        score_column="knn_k20_mean_distance",
        method_category="unsupervised_ood",
        feature_space="L2-normalized BGE",
        score_direction="larger score = larger mean distance to 20 nearest known-train samples = more OOD",
        is_supervised_proxy=False,
        notes="Nearest-neighbor distances are fit only on known train.",
    ),
    "bge_isolation_forest": MethodInfo(
        method="bge_isolation_forest",
        score_column="isolation_forest_ood_score",
        method_category="unsupervised_ood",
        feature_space="L2-normalized BGE",
        score_direction="larger score = lower IsolationForest normality score = more OOD",
        is_supervised_proxy=False,
        notes="IsolationForest is fit only on known train.",
    ),
    "tfidf_lr_bot_probability_proxy": MethodInfo(
        method="tfidf_lr_bot_probability_proxy",
        score_column="tfidf_lr_bot_probability",
        method_category="supervised_proxy",
        feature_space="TF-IDF text",
        score_direction="larger score = higher supervised bot probability used as OOD proxy; not pure OOD",
        is_supervised_proxy=True,
        notes="Included only as a supervised bot-detection proxy baseline.",
    ),
    "bge_lr_bot_probability_proxy": MethodInfo(
        method="bge_lr_bot_probability_proxy",
        score_column="bge_lr_bot_probability",
        method_category="supervised_proxy",
        feature_space="BGE",
        score_direction="larger score = higher supervised bot probability used as OOD proxy; not pure OOD",
        is_supervised_proxy=True,
        notes="Included only as a supervised bot-detection proxy baseline.",
    ),
}

METHOD_ORDER = list(METHODS)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return embeddings / safe_norms


def _family_short(value: object) -> str:
    family = "" if pd.isna(value) else str(value)
    if family in SHORT_NAME:
        return SHORT_NAME[family]
    if family == "original_human":
        return "OrigH"
    if family == "original_bot":
        return "OrigB"
    return family or "unknown"


def _scenario_indices(
    scenario: str,
    heldout_family: str,
    original_train_idx: np.ndarray,
    original_test_idx: np.ndarray,
    family_indices: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if scenario == "original_only_known":
        train_families: list[str] = []
        known_train_idx = original_train_idx
        id_idx = original_test_idx
    elif scenario == "lofo_known":
        train_families = [family for family in FAMILIES if family != heldout_family]
        known_train_idx = family_train_indices(original_train_idx, family_indices, train_families)
        id_idx = np.concatenate([original_test_idx] + [family_indices[family] for family in train_families])
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    ood_idx = family_indices[heldout_family]
    if set(known_train_idx).intersection(set(ood_idx)):
        raise AssertionError(f"held-out sample leakage detected: {scenario} {heldout_family}")
    train_family_values = set(train_families)
    if heldout_family in train_family_values:
        raise AssertionError(f"held-out family leaked into train_families: {scenario} {heldout_family}")
    return known_train_idx, id_idx, ood_idx, train_families


def _compute_centroid_components(
    embeddings_norm: np.ndarray,
    labels: np.ndarray,
    known_train_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    human_idx = known_train_idx[labels[known_train_idx] == 0]
    bot_idx = known_train_idx[labels[known_train_idx] == 1]
    if len(human_idx) == 0 or len(bot_idx) == 0:
        raise AssertionError("known train must contain both human and bot samples")

    human_centroid = embeddings_norm[human_idx].mean(axis=0)
    bot_centroid = embeddings_norm[bot_idx].mean(axis=0)
    d_human = np.linalg.norm(embeddings_norm - human_centroid, axis=1)
    d_bot = np.linalg.norm(embeddings_norm - bot_centroid, axis=1)
    return {
        "d_to_human_centroid": d_human,
        "d_to_known_bot_centroid": d_bot,
        "centroid_min_distance": np.minimum(d_human, d_bot),
        "centroid_bot_minus_human_margin": d_bot - d_human,
    }


def _compute_knn_scores(
    embeddings_norm: np.ndarray,
    known_train_idx: np.ndarray,
    query_idx: np.ndarray,
    k_values: list[int],
) -> dict[int, np.ndarray]:
    max_k = max(k_values)
    if len(known_train_idx) <= max_k:
        raise AssertionError(f"known train is too small for k={max_k}")

    distances = pairwise_distances(
        embeddings_norm[query_idx],
        embeddings_norm[known_train_idx],
        metric="euclidean",
        n_jobs=-1,
    )
    train_position = {int(sample_idx): pos for pos, sample_idx in enumerate(known_train_idx)}
    for row_pos, sample_idx in enumerate(query_idx):
        train_pos = train_position.get(int(sample_idx))
        if train_pos is not None:
            distances[row_pos, train_pos] = np.inf

    nearest = np.partition(distances, kth=max_k - 1, axis=1)[:, :max_k]
    nearest.sort(axis=1)
    return {k: nearest[:, :k].mean(axis=1) for k in k_values}


def _compute_scores_for_setting(
    docs: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    embeddings_norm: np.ndarray,
    known_train_idx: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_samples = len(labels)
    all_idx = np.arange(n_samples)
    score_arrays: dict[str, np.ndarray] = {}
    component_arrays = _compute_centroid_components(embeddings_norm, labels, known_train_idx)

    score_arrays["bge_centroid_min_distance"] = component_arrays["centroid_min_distance"]
    score_arrays["bge_centroid_bot_margin"] = component_arrays["centroid_bot_minus_human_margin"]

    knn_by_k = _compute_knn_scores(embeddings_norm, known_train_idx, all_idx, KNN_K_VALUES)
    for k, scores in knn_by_k.items():
        key = f"bge_knn_k{k}_distance"
        score_arrays[key] = scores
        component_arrays[f"knn_k{k}_mean_distance"] = scores

    isolation_forest = IsolationForest(
        n_estimators=300,
        random_state=20260602,
        contamination="auto",
        n_jobs=-1,
    )
    isolation_forest.fit(embeddings_norm[known_train_idx])
    iforest_score = -isolation_forest.score_samples(embeddings_norm)
    score_arrays["bge_isolation_forest"] = iforest_score
    component_arrays["isolation_forest_ood_score"] = iforest_score

    tfidf_model = train_tfidf_lr(docs, labels, known_train_idx)
    _tfidf_pred, tfidf_prob = predict_tfidf_lr(tfidf_model, docs, all_idx)
    score_arrays["tfidf_lr_bot_probability_proxy"] = tfidf_prob
    component_arrays["tfidf_lr_bot_probability"] = tfidf_prob

    bge_model = train_bge_lr(embeddings, labels, known_train_idx)
    _bge_pred, bge_prob = predict_bge_lr(bge_model, embeddings, all_idx)
    score_arrays["bge_lr_bot_probability_proxy"] = bge_prob
    component_arrays["bge_lr_bot_probability"] = bge_prob

    return score_arrays, component_arrays


def _rank_family_means(
    score: np.ndarray,
    family_indices: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, int], str]:
    means = {family: float(np.mean(score[family_indices[family]])) for family in FAMILIES}
    ordered = sorted(means, key=lambda family: (-means[family], SHORT_NAME[family]))
    ranks: dict[str, int] = {}
    previous_value: float | None = None
    current_rank = 0
    for position, family in enumerate(ordered, start=1):
        value = means[family]
        if previous_value is None or not np.isclose(value, previous_value):
            current_rank = position
            previous_value = value
        ranks[family] = current_rank
    order_string = ">".join(f"{SHORT_NAME[family]}:{means[family]:.4f}" for family in ordered)
    return means, ranks, order_string


def _sample_rows_for_method(
    combined: pd.DataFrame,
    labels: np.ndarray,
    scenario: str,
    heldout_family: str,
    train_families: list[str],
    known_train_idx: np.ndarray,
    id_idx: np.ndarray,
    ood_idx: np.ndarray,
    method: str,
    score: np.ndarray,
    component_arrays: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    method_info = METHODS[method]
    known_train_set = set(int(i) for i in known_train_idx)
    train_families_short = short_family_list(train_families) if train_families else "none"
    rows: list[dict[str, object]] = []
    for eval_group, idx_values in [("id", id_idx), ("ood", ood_idx)]:
        ood_label = 1 if eval_group == "ood" else 0
        for sample_idx in idx_values:
            sample = combined.loc[int(sample_idx)]
            sample_family = str(sample["family"])
            if eval_group == "ood":
                sample_role = "heldout_family_ood"
            elif sample_family in FAMILIES:
                sample_role = "seen_family_id"
            else:
                sample_role = "original_test_id"
            rows.append(
                {
                    "scenario": scenario,
                    "heldout_family": heldout_family,
                    "heldout_family_short": SHORT_NAME[heldout_family],
                    "known_train_families": train_families_short,
                    "method": method,
                    "method_category": method_info.method_category,
                    "is_supervised_proxy": method_info.is_supervised_proxy,
                    "score_direction": method_info.score_direction,
                    "eval_group": eval_group,
                    "ood_label": ood_label,
                    "sample_role": sample_role,
                    "sample_index": int(sample_idx),
                    "row": int(sample["row"]) if not pd.isna(sample["row"]) else int(sample_idx),
                    "source": sample["source"],
                    "family": sample_family,
                    "family_short": _family_short(sample_family),
                    "label": int(labels[int(sample_idx)]),
                    "is_known_train_sample": int(int(sample_idx) in known_train_set),
                    "score": float(score[int(sample_idx)]),
                    "d_to_human_centroid": float(component_arrays["d_to_human_centroid"][int(sample_idx)]),
                    "d_to_known_bot_centroid": float(component_arrays["d_to_known_bot_centroid"][int(sample_idx)]),
                    "centroid_min_distance": float(component_arrays["centroid_min_distance"][int(sample_idx)]),
                    "centroid_bot_minus_human_margin": float(
                        component_arrays["centroid_bot_minus_human_margin"][int(sample_idx)]
                    ),
                    "knn_k5_mean_distance": float(component_arrays["knn_k5_mean_distance"][int(sample_idx)]),
                    "knn_k10_mean_distance": float(component_arrays["knn_k10_mean_distance"][int(sample_idx)]),
                    "knn_k20_mean_distance": float(component_arrays["knn_k20_mean_distance"][int(sample_idx)]),
                    "isolation_forest_ood_score": float(
                        component_arrays["isolation_forest_ood_score"][int(sample_idx)]
                    ),
                    "tfidf_lr_bot_probability": float(component_arrays["tfidf_lr_bot_probability"][int(sample_idx)]),
                    "bge_lr_bot_probability": float(component_arrays["bge_lr_bot_probability"][int(sample_idx)]),
                }
            )
    return rows


def _metric_row_for_method(
    scenario: str,
    heldout_family: str,
    train_families: list[str],
    known_train_idx: np.ndarray,
    id_idx: np.ndarray,
    ood_idx: np.ndarray,
    labels: np.ndarray,
    method: str,
    score: np.ndarray,
    family_indices: dict[str, np.ndarray],
) -> dict[str, object]:
    method_info = METHODS[method]
    eval_idx = np.concatenate([id_idx, ood_idx])
    y_ood = np.concatenate([np.zeros(len(id_idx), dtype=int), np.ones(len(ood_idx), dtype=int)])
    eval_score = score[eval_idx]
    family_means, family_ranks, family_rank_order = _rank_family_means(score, family_indices)
    id_score = score[id_idx]
    ood_score = score[ood_idx]
    known_train_labels = labels[known_train_idx]
    train_families_short = short_family_list(train_families) if train_families else "none"
    return {
        "scenario": scenario,
        "heldout_family": heldout_family,
        "heldout_family_short": SHORT_NAME[heldout_family],
        "known_train_families": train_families_short,
        "method": method,
        "method_category": method_info.method_category,
        "is_supervised_proxy": method_info.is_supervised_proxy,
        "score_direction": method_info.score_direction,
        "auroc_ood_vs_id": float(roc_auc_score(y_ood, eval_score)),
        "auprc_ood": float(average_precision_score(y_ood, eval_score)),
        "mean_ood_score_id": float(np.mean(id_score)),
        "mean_ood_score_heldout": float(np.mean(ood_score)),
        "median_ood_score_id": float(np.median(id_score)),
        "median_ood_score_heldout": float(np.median(ood_score)),
        "std_ood_score_id": float(np.std(id_score, ddof=0)),
        "std_ood_score_heldout": float(np.std(ood_score, ddof=0)),
        "score_gap_heldout_minus_id": float(np.mean(ood_score) - np.mean(id_score)),
        "heldout_family_rank": int(family_ranks[heldout_family]),
        "heldout_family_mean_among_all_families": float(family_means[heldout_family]),
        "family_rank_order_desc": family_rank_order,
        "top_scored_family_short": SHORT_NAME[max(family_means, key=family_means.get)],
        "id_sample_count": int(len(id_idx)),
        "ood_sample_count": int(len(ood_idx)),
        "known_train_sample_count": int(len(known_train_idx)),
        "known_train_human_count": int(np.sum(known_train_labels == 0)),
        "known_train_bot_count": int(np.sum(known_train_labels == 1)),
    }


def _build_method_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        method_info = METHODS[method]
        method_metrics = metrics_df[metrics_df["method"] == method]
        best = method_metrics.loc[method_metrics["auroc_ood_vs_id"].idxmax()]
        worst = method_metrics.loc[method_metrics["auroc_ood_vs_id"].idxmin()]
        rows.append(
            {
                "method": method,
                "score_column": method_info.score_column,
                "method_category": method_info.method_category,
                "feature_space": method_info.feature_space,
                "score_direction": method_info.score_direction,
                "is_supervised_proxy": method_info.is_supervised_proxy,
                "notes": method_info.notes,
                "n_metric_rows": int(len(method_metrics)),
                "mean_auroc_ood_vs_id": float(method_metrics["auroc_ood_vs_id"].mean()),
                "median_auroc_ood_vs_id": float(method_metrics["auroc_ood_vs_id"].median()),
                "mean_auprc_ood": float(method_metrics["auprc_ood"].mean()),
                "mean_score_gap_heldout_minus_id": float(method_metrics["score_gap_heldout_minus_id"].mean()),
                "mean_heldout_family_rank": float(method_metrics["heldout_family_rank"].mean()),
                "best_case": f"{best['scenario']}:{best['heldout_family_short']}={best['auroc_ood_vs_id']:.3f}",
                "worst_case": f"{worst['scenario']}:{worst['heldout_family_short']}={worst['auroc_ood_vs_id']:.3f}",
            }
        )
    return pd.DataFrame(rows)


def _plot_score_distribution(scores_df: pd.DataFrame) -> None:
    plot_methods = [
        "bge_centroid_min_distance",
        "bge_knn_k10_distance",
        "bge_isolation_forest",
    ]
    plot_df = scores_df[
        (scores_df["scenario"] == "original_only_known")
        & (scores_df["method"].isin(plot_methods))
        & (scores_df["sample_role"].isin(["heldout_family_ood", "original_test_id"]))
    ].copy()
    plot_df.loc[plot_df["sample_role"] == "original_test_id", "plot_family"] = "OrigTest"
    plot_df.loc[plot_df["sample_role"] == "heldout_family_ood", "plot_family"] = plot_df["family_short"]
    family_order = ["OrigTest"] + FAMILY_SHORT_ORDER

    fig, axes = plt.subplots(1, len(plot_methods), figsize=(13.8, 4.2), sharey=False)
    colors = ["#b9c7d9"] + ["#e3b174", "#c97c6d", "#8fb984", "#9f90c8", "#74a7bf"]
    for ax, method in zip(axes, plot_methods, strict=True):
        method_df = plot_df[plot_df["method"] == method]
        data = [
            method_df.loc[method_df["plot_family"] == family_short, "score"].to_numpy()
            for family_short in family_order
        ]
        ax.boxplot(
            data,
            tick_labels=family_order,
            patch_artist=True,
            showfliers=False,
            boxprops={"edgecolor": "#46515c", "linewidth": 1.0},
            medianprops={"color": "#2d2d2d", "linewidth": 1.3},
            whiskerprops={"color": "#46515c"},
            capprops={"color": "#46515c"},
        )
        for patch, color in zip(ax.patches, colors, strict=True):
            patch.set_facecolor(color)
        ax.set_title(method.replace("bge_", "").replace("_", " "))
        ax.set_xlabel("Family")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", color="#d4d4d4", linewidth=0.7, alpha=0.75)
    axes[0].set_ylabel("OOD score, larger = more OOD")
    fig.suptitle("Original-only known: OOD score distributions by family", y=1.02)
    fig.tight_layout()
    save_figure(fig, SCORE_DISTRIBUTION_PATH)


def _plot_auroc(metrics_df: pd.DataFrame) -> None:
    method_labels = {
        "bge_centroid_min_distance": "Centroid min",
        "bge_centroid_bot_margin": "Centroid margin",
        "bge_knn_k5_distance": "kNN-5",
        "bge_knn_k10_distance": "kNN-10",
        "bge_knn_k20_distance": "kNN-20",
        "bge_isolation_forest": "IForest",
        "tfidf_lr_bot_probability_proxy": "TF-IDF proxy",
        "bge_lr_bot_probability_proxy": "BGE proxy",
    }
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(14.2, 6.0), sharey=True, constrained_layout=True)
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        pivot = metrics_df[metrics_df["scenario"] == scenario].pivot(
            index="method",
            columns="heldout_family_short",
            values="auroc_ood_vs_id",
        )
        pivot = pivot.reindex(index=METHOD_ORDER, columns=FAMILY_SHORT_ORDER)
        image = ax.imshow(pivot.to_numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(FAMILY_SHORT_ORDER)), labels=FAMILY_SHORT_ORDER)
        ax.set_yticks(np.arange(len(METHOD_ORDER)), labels=[method_labels[m] for m in METHOD_ORDER])
        ax.set_title(scenario)
        ax.set_xlabel("Held-out family")
        for row_pos, method in enumerate(METHOD_ORDER):
            for col_pos, family_short in enumerate(FAMILY_SHORT_ORDER):
                value = pivot.loc[method, family_short]
                ax.text(
                    col_pos,
                    row_pos,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value < 0.35 or value > 0.72 else "black",
                )
    axes[0].set_ylabel("Method")
    cbar = fig.colorbar(image, ax=axes, fraction=0.026, pad=0.025)
    cbar.set_label("AUROC, OOD label = 1")
    fig.suptitle("OOD vs ID AUROC by method and held-out family", y=1.01)
    save_figure(fig, AUROC_FIGURE_PATH)


def _plot_centroid_scatter(scores_df: pd.DataFrame) -> None:
    scatter_df = scores_df[
        (scores_df["scenario"] == "original_only_known")
        & (scores_df["method"] == "bge_centroid_min_distance")
        & (scores_df["sample_role"].isin(["original_test_id", "heldout_family_ood"]))
    ].copy()
    scatter_df.loc[scatter_df["sample_role"] == "original_test_id", "plot_family"] = "OrigTest"
    scatter_df.loc[scatter_df["sample_role"] == "heldout_family_ood", "plot_family"] = scatter_df["family_short"]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    color_map = {
        "OrigTest": "#7f8896",
        "F1": "#e3a249",
        "F2": "#ca6e61",
        "F3": "#6ba36f",
        "F4": "#8d7bc1",
        "F5": "#5d9db7",
    }
    alpha_map = {"OrigTest": 0.22, "F1": 0.85, "F2": 0.85, "F3": 0.85, "F4": 0.85, "F5": 0.85}
    for family_short in ["OrigTest"] + FAMILY_SHORT_ORDER:
        family_df = scatter_df[scatter_df["plot_family"] == family_short]
        ax.scatter(
            family_df["d_to_human_centroid"],
            family_df["d_to_known_bot_centroid"],
            s=22 if family_short == "OrigTest" else 34,
            alpha=alpha_map[family_short],
            color=color_map[family_short],
            label=family_short,
            edgecolor="none",
        )
    low = float(min(scatter_df["d_to_human_centroid"].min(), scatter_df["d_to_known_bot_centroid"].min()))
    high = float(max(scatter_df["d_to_human_centroid"].max(), scatter_df["d_to_known_bot_centroid"].max()))
    ax.plot([low, high], [low, high], color="#404040", linestyle="--", linewidth=1.0, label="equal distance")
    ax.set_xlabel("Distance to known human centroid")
    ax.set_ylabel("Distance to known bot centroid")
    ax.set_title("BGE centroid distance and bot-minus-human margin")
    ax.grid(color="#d5d5d5", linewidth=0.7, alpha=0.75)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, CENTROID_SCATTER_PATH)


def run_static_text_ood() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined, docs, labels, embeddings = load_combined_data()
    embeddings_norm = _normalize_embeddings(embeddings.astype(np.float64, copy=False))
    original_train_idx, _original_val_idx, original_test_idx = get_original_train_val_test_indices(combined, labels)
    family_indices = get_family_indices(combined)

    score_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        for heldout_family in FAMILIES:
            known_train_idx, id_idx, ood_idx, train_families = _scenario_indices(
                scenario=scenario,
                heldout_family=heldout_family,
                original_train_idx=original_train_idx,
                original_test_idx=original_test_idx,
                family_indices=family_indices,
            )
            score_arrays, component_arrays = _compute_scores_for_setting(
                docs=docs,
                labels=labels,
                embeddings=embeddings,
                embeddings_norm=embeddings_norm,
                known_train_idx=known_train_idx,
            )
            for method in METHOD_ORDER:
                score = score_arrays[method]
                metric_rows.append(
                    _metric_row_for_method(
                        scenario=scenario,
                        heldout_family=heldout_family,
                        train_families=train_families,
                        known_train_idx=known_train_idx,
                        id_idx=id_idx,
                        ood_idx=ood_idx,
                        labels=labels,
                        method=method,
                        score=score,
                        family_indices=family_indices,
                    )
                )
                score_rows.extend(
                    _sample_rows_for_method(
                        combined=combined,
                        labels=labels,
                        scenario=scenario,
                        heldout_family=heldout_family,
                        train_families=train_families,
                        known_train_idx=known_train_idx,
                        id_idx=id_idx,
                        ood_idx=ood_idx,
                        method=method,
                        score=score,
                        component_arrays=component_arrays,
                    )
                )

    scores_df = pd.DataFrame(score_rows)
    metrics_df = pd.DataFrame(metric_rows)
    summary_df = _build_method_summary(metrics_df)

    save_table(scores_df, SCORES_PATH)
    save_table(metrics_df, METRICS_PATH)
    save_table(summary_df, SUMMARY_PATH)
    _plot_score_distribution(scores_df)
    _plot_auroc(metrics_df)
    _plot_centroid_scatter(scores_df)

    return scores_df, metrics_df, summary_df


def validate_outputs(scores_df: pd.DataFrame, metrics_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    for method in METHOD_ORDER:
        if method not in set(scores_df["method"]):
            raise AssertionError(f"missing method in scores: {method}")
        if method not in set(metrics_df["method"]):
            raise AssertionError(f"missing method in metrics: {method}")
        if method not in set(summary_df["method"]):
            raise AssertionError(f"missing method in summary: {method}")

    expected_metric_rows = len(SCENARIOS) * len(FAMILIES) * len(METHOD_ORDER)
    if len(metrics_df) != expected_metric_rows:
        raise AssertionError(f"expected {expected_metric_rows} metric rows, got {len(metrics_df)}")

    if metrics_df["auroc_ood_vs_id"].isna().any() or metrics_df["auprc_ood"].isna().any():
        raise AssertionError("AUROC/AUPRC contains NaN")
    if not metrics_df["auroc_ood_vs_id"].between(0.0, 1.0).all():
        raise AssertionError("AUROC outside [0, 1]")
    if not metrics_df["auprc_ood"].between(0.0, 1.0).all():
        raise AssertionError("AUPRC outside [0, 1]")

    for row in metrics_df.itertuples(index=False):
        train_families = [] if row.known_train_families == "none" else str(row.known_train_families).split("+")
        if row.heldout_family_short in train_families:
            raise AssertionError(f"held-out family leaked into train list: {row}")

    for path in [
        SCORES_PATH,
        METRICS_PATH,
        SUMMARY_PATH,
        SCORE_DISTRIBUTION_PATH,
        AUROC_FIGURE_PATH,
        CENTROID_SCATTER_PATH,
    ]:
        if not Path(path).exists():
            raise AssertionError(f"missing output: {path}")


def main() -> None:
    research_cli(globals(), "text_evaluation")
    scores_df, metrics_df, summary_df = run_static_text_ood()
    validate_outputs(scores_df, metrics_df, summary_df)

    print("Static Text OOD detection completed.")
    print(f"scores: {SCORES_PATH}")
    print(f"metrics: {METRICS_PATH}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"score distribution figure: {SCORE_DISTRIBUTION_PATH}")
    print(f"AUROC figure: {AUROC_FIGURE_PATH}")
    print(f"centroid scatter figure: {CENTROID_SCATTER_PATH}")
    print()
    print("Mean AUROC by scenario/method:")
    auroc_summary = metrics_df.pivot_table(
        index="method",
        columns="scenario",
        values="auroc_ood_vs_id",
        aggfunc="mean",
    ).reindex(METHOD_ORDER)
    print(auroc_summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print()
    print("Method summary:")
    print(
        summary_df[
            [
                "method",
                "method_category",
                "is_supervised_proxy",
                "mean_auroc_ood_vs_id",
                "mean_auprc_ood",
                "mean_heldout_family_rank",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
