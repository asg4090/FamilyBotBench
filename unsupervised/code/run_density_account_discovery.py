#!/usr/bin/env python3
"""Density-based account discovery for original_human vs original_bot.

The discovery stage is label-free:
- word TF-IDF builds the primary account language space;
- HDBSCAN finds dense regions in an SVD-reduced TF-IDF space;
- dense component scan finds connected high-similarity components in the
  original cosine kNN graph;
- labels are revealed only after suspicious clusters are scored.
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
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_unsupervised_account_discovery import (  # noqa: E402
    BOT_FAMILY,
    HUMAN_FAMILY,
    TARGET_FAMILIES,
    FeatureView,
    build_eval_label,
    cluster_geometry,
    fit_tfidf,
    random_baseline,
    safe_z,
    select_rows,
    stable_sha1,
    style_features,
)


PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "inputs" / "static_family"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "density_original_human_vs_oldbot"


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        number = int(part)
        if number <= 1:
            raise ValueError(f"List values must be greater than 1, got {number}.")
        out.append(number)
    if not out:
        raise ValueError("No valid integer values were provided.")
    return sorted(set(out))


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing combined_sample_index.csv. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--sample-human", type=int, default=None)
    parser.add_argument("--sample-bot", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--svd-components", type=int, default=100)
    parser.add_argument(
        "--hdbscan-min-cluster-sizes",
        default="15,30,50,80",
        help="Comma-separated min_cluster_size values for HDBSCAN.",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=8,
        help="HDBSCAN min_samples. Smaller values find more clusters.",
    )
    parser.add_argument(
        "--component-knn-k",
        type=int,
        default=50,
        help="k used to build the cosine kNN graph for dense component scan.",
    )
    parser.add_argument(
        "--component-threshold-quantiles",
        default="0.90,0.95,0.975,0.99",
        help="Similarity quantiles used as automatic dense-component thresholds.",
    )
    parser.add_argument(
        "--component-min-size",
        type=int,
        default=10,
        help="Minimum connected-component size to keep.",
    )
    parser.add_argument(
        "--component-min-edge-density",
        type=float,
        default=0.025,
        help="Minimum internal threshold-graph edge density for dense components.",
    )
    parser.add_argument(
        "--max-cluster-fraction",
        type=float,
        default=0.25,
        help="Skip components/clusters larger than this fraction of all accounts.",
    )
    parser.add_argument("--null-repeats", type=int, default=300)
    parser.add_argument("--top-n-report", type=int, default=30)
    return parse_release_args(parser, globals(), "unsupervised")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_primary_tfidf(docs: list[str], max_features: int) -> FeatureView:
    return fit_tfidf(
        "word_tfidf",
        docs,
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_features,
    )


def svd_embedding(view: FeatureView, n_components: int, random_state: int) -> tuple[np.ndarray, int]:
    n_samples, n_features = view.matrix.shape
    usable_components = max(2, min(int(n_components), n_samples - 1, n_features - 1))
    reducer = TruncatedSVD(n_components=usable_components, random_state=random_state)
    values = reducer.fit_transform(view.matrix)
    values = StandardScaler().fit_transform(values)
    values = normalize(np.nan_to_num(values), norm="l2", copy=False)
    return values.astype(np.float32, copy=False), usable_components


def write_account_features(
    output_dir: Path,
    selected: pd.DataFrame,
    style_df: pd.DataFrame,
    tfidf_view: FeatureView,
    svd_components: int,
) -> Path:
    out = pd.DataFrame(
        {
            "account_index": np.arange(len(selected), dtype=int),
            "account_id": selected["account_id"],
            "account_id_raw": selected["account_id_raw"],
            "input_position": selected["_input_pos"].astype(int),
            "input_row": selected["row"] if "row" in selected.columns else selected["_input_pos"].astype(int),
            "doc_sha1": selected["doc"].map(stable_sha1),
            "view_word_tfidf_n_features": tfidf_view.n_features,
            "svd_components": svd_components,
        }
    )
    for optional_col in ("user_name", "provider", "model", "temperature"):
        if optional_col in selected.columns:
            out[f"meta_{optional_col}"] = selected[optional_col]
    out = pd.concat([out, style_df.reset_index(drop=True)], axis=1)
    path = output_dir / "density_account_features.csv"
    out.to_csv(path, index=False)
    return path


def write_label_revealed_account_metadata(output_dir: Path, selected: pd.DataFrame, eval_label: pd.Series) -> Path:
    out = pd.DataFrame(
        {
            "account_index": np.arange(len(selected), dtype=int),
            "account_id": selected["account_id"],
            "account_id_raw": selected["account_id_raw"],
            "input_position": selected["_input_pos"].astype(int),
            "eval_family": selected["family"].astype(str),
            "eval_label": eval_label.astype(int),
        }
    )
    for optional_col in ("source", "row_kind", "family_display"):
        if optional_col in selected.columns:
            out[f"eval_{optional_col}"] = selected[optional_col]
    path = output_dir / "density_label_revealed_account_metadata.csv"
    out.to_csv(path, index=False)
    return path


def run_hdbscan(
    embedding: np.ndarray,
    min_cluster_sizes: list[int],
    min_samples: int,
    account_ids: np.ndarray,
    max_cluster_fraction: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    assignment_rows: list[dict[str, Any]] = []
    clusters: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, Any]] = {}
    n_accounts = len(account_ids)
    max_size = max(2, int(math.floor(max_cluster_fraction * n_accounts)))

    for min_cluster_size in min_cluster_sizes:
        print(f"[info] HDBSCAN min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            n_jobs=-1,
            copy=True,
        )
        labels = model.fit_predict(embedding)
        probabilities = getattr(model, "probabilities_", np.ones(len(labels), dtype=float))
        run_id = f"hdbscan_mcs{min_cluster_size}_ms{min_samples}"
        for idx, label in enumerate(labels.astype(int)):
            assignment_rows.append(
                {
                    "account_index": idx,
                    "account_id": account_ids[idx],
                    "method": "hdbscan",
                    "run_id": run_id,
                    "cluster_id": int(label),
                    "membership_strength": float(probabilities[idx]) if label >= 0 else 0.0,
                }
            )
        for cluster_id in sorted(label for label in np.unique(labels.astype(int)) if label >= 0):
            members = np.flatnonzero(labels == cluster_id).astype(int)
            if len(members) < min_cluster_size or len(members) > max_size:
                continue
            key = f"{run_id}:c={int(cluster_id)}"
            clusters[key] = members
            meta[key] = {
                "method": "hdbscan",
                "run_id": run_id,
                "cluster_id": int(cluster_id),
                "cluster_size_parameter": int(min_cluster_size),
                "threshold": np.nan,
                "graph_edge_count": np.nan,
                "graph_edge_density": np.nan,
                "mean_membership_strength": float(np.nanmean(probabilities[members])),
            }
    return pd.DataFrame(assignment_rows), clusters, meta


def build_knn_edges(matrix: Any, account_ids: np.ndarray, k: int) -> pd.DataFrame:
    n = len(account_ids)
    n_neighbors = min(n, max(2, k + 1))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    model.fit(matrix)
    distances, indices = model.kneighbors(matrix, return_distance=True)

    pair_best: dict[tuple[int, int], float] = {}
    rows: list[dict[str, Any]] = []
    for source in range(n):
        rank = 0
        for distance, target in zip(distances[source], indices[source]):
            target = int(target)
            if target == source:
                continue
            rank += 1
            if rank > k:
                break
            similarity = float(np.clip(1.0 - distance, -1.0, 1.0))
            rows.append(
                {
                    "source_account_index": source,
                    "source_account_id": account_ids[source],
                    "target_account_index": target,
                    "target_account_id": account_ids[target],
                    "rank": rank,
                    "cosine_similarity": similarity,
                }
            )
            left, right = sorted((source, target))
            pair_best[(left, right)] = max(pair_best.get((left, right), -1.0), similarity)

    edge_rows = [
        {
            "u": left,
            "v": right,
            "u_account_id": account_ids[left],
            "v_account_id": account_ids[right],
            "cosine_similarity": similarity,
        }
        for (left, right), similarity in pair_best.items()
    ]
    return pd.DataFrame(edge_rows), pd.DataFrame(rows)


def automatic_thresholds(edge_sims: np.ndarray, quantiles: list[float]) -> list[float]:
    thresholds: list[float] = []
    finite = edge_sims[np.isfinite(edge_sims)]
    if len(finite) == 0:
        return thresholds
    for q in quantiles:
        q = float(q)
        if q > 1.0:
            q = q / 100.0
        q = min(max(q, 0.0), 1.0)
        thresholds.append(float(np.quantile(finite, q)))
    unique = sorted({round(value, 6) for value in thresholds})
    return unique


def run_dense_component_scan(
    edge_pairs: pd.DataFrame,
    thresholds: list[float],
    n_accounts: int,
    account_ids: np.ndarray,
    min_size: int,
    min_edge_density: float,
    max_cluster_fraction: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    clusters: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, Any]] = {}
    assignment_rows: list[dict[str, Any]] = []
    max_size = max(2, int(math.floor(max_cluster_fraction * n_accounts)))

    if edge_pairs.empty:
        return pd.DataFrame(), clusters, meta

    for threshold in thresholds:
        print(f"[info] Dense component scan threshold={threshold:.6f}")
        keep = edge_pairs["cosine_similarity"].to_numpy(dtype=float) >= threshold
        kept = edge_pairs.loc[keep].copy()
        run_id = f"dense_component_t{threshold:.6f}"
        if kept.empty:
            continue

        rows = np.concatenate([kept["u"].to_numpy(dtype=int), kept["v"].to_numpy(dtype=int)])
        cols = np.concatenate([kept["v"].to_numpy(dtype=int), kept["u"].to_numpy(dtype=int)])
        data = np.ones(len(rows), dtype=np.uint8)
        adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(n_accounts, n_accounts))
        _, labels = connected_components(adjacency, directed=False, return_labels=True)

        component_sizes = pd.Series(labels).value_counts().to_dict()
        valid_components = {
            int(component_id)
            for component_id, size in component_sizes.items()
            if int(size) >= min_size and int(size) <= max_size
        }
        if not valid_components:
            continue

        for idx, component_id in enumerate(labels.astype(int)):
            if int(component_id) in valid_components:
                assignment_rows.append(
                    {
                        "account_index": idx,
                        "account_id": account_ids[idx],
                        "method": "dense_component",
                        "run_id": run_id,
                        "cluster_id": int(component_id),
                        "membership_strength": 1.0,
                    }
                )

        edge_component = labels[kept["u"].to_numpy(dtype=int)]
        for component_id in sorted(valid_components):
            members = np.flatnonzero(labels == component_id).astype(int)
            internal_edges = int((edge_component == component_id).sum())
            possible_edges = len(members) * (len(members) - 1) / 2
            edge_density = float(internal_edges / possible_edges) if possible_edges else 0.0
            if edge_density < min_edge_density:
                continue
            key = f"{run_id}:c={component_id}"
            clusters[key] = members
            meta[key] = {
                "method": "dense_component",
                "run_id": run_id,
                "cluster_id": int(component_id),
                "cluster_size_parameter": int(min_size),
                "threshold": float(threshold),
                "graph_edge_count": internal_edges,
                "graph_edge_density": edge_density,
                "mean_membership_strength": 1.0,
            }

    return pd.DataFrame(assignment_rows), clusters, meta


def cluster_id_sort_key(cluster_key: str) -> tuple[str, str, int]:
    prefix, _, cluster = cluster_key.partition(":c=")
    try:
        cluster_num = int(cluster)
    except ValueError:
        cluster_num = -1
    method = "hdbscan" if prefix.startswith("hdbscan") else "dense_component"
    return method, prefix, cluster_num


def compute_scores(
    output_dir: Path,
    tfidf_view: FeatureView,
    clusters: dict[str, np.ndarray],
    meta: dict[str, dict[str, Any]],
    account_ids: np.ndarray,
    null_repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, Path]:
    rng = np.random.default_rng(random_state)
    baseline_cache: dict[int, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []

    for cluster_key in sorted(clusters, key=cluster_id_sort_key):
        members = clusters[cluster_key]
        size = int(len(members))
        mean_internal, mean_radius = cluster_geometry(tfidf_view.matrix, members)
        if size not in baseline_cache:
            baseline_cache[size] = random_baseline(tfidf_view.matrix, size, null_repeats, rng)
        baseline = baseline_cache[size]
        density_z = safe_z(mean_internal, baseline["null_density_mean"], baseline["null_density_std"])
        radius_z = safe_z(baseline["null_radius_mean"], mean_radius, baseline["null_radius_std"])
        info = meta[cluster_key]
        graph_edge_density = info.get("graph_edge_density", np.nan)
        membership_strength = info.get("mean_membership_strength", np.nan)
        graph_bonus = 0.0 if pd.isna(graph_edge_density) else 2.0 * float(graph_edge_density)
        membership_bonus = 0.0 if pd.isna(membership_strength) else float(membership_strength)
        suspicious_score = float(
            density_z
            + radius_z
            + 0.25 * math.log1p(size)
            + graph_bonus
            + 0.5 * membership_bonus
        )
        rows.append(
            {
                "cluster_key": cluster_key,
                "method": info["method"],
                "run_id": info["run_id"],
                "cluster_id": info["cluster_id"],
                "cluster_size_parameter": info["cluster_size_parameter"],
                "threshold": info["threshold"],
                "cluster_size": size,
                "mean_internal_similarity": float(mean_internal),
                "mean_radius": float(mean_radius),
                "density_z_vs_random": float(density_z),
                "radius_z_vs_random": float(radius_z),
                "graph_edge_count": info.get("graph_edge_count", np.nan),
                "graph_edge_density": info.get("graph_edge_density", np.nan),
                "mean_membership_strength": info.get("mean_membership_strength", np.nan),
                "suspicious_score": suspicious_score,
                "null_repeats": int(null_repeats),
                "null_density_mean": baseline["null_density_mean"],
                "null_density_std": baseline["null_density_std"],
                "null_radius_mean": baseline["null_radius_mean"],
                "null_radius_std": baseline["null_radius_std"],
                "member_account_ids": ";".join(str(account_ids[i]) for i in members),
            }
        )

    scores = pd.DataFrame(rows)
    if not scores.empty:
        scores = scores.sort_values(
            ["suspicious_score", "density_z_vs_random", "cluster_size"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        scores.insert(0, "suspicious_rank", np.arange(1, len(scores) + 1, dtype=int))
    path = output_dir / "density_suspicious_cluster_scores.csv"
    scores.to_csv(path, index=False)
    return scores, path


def write_cluster_assignments(output_dir: Path, frames: list[pd.DataFrame]) -> Path:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    assignments = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    path = output_dir / "density_cluster_assignments.csv"
    assignments.to_csv(path, index=False)
    return path


def write_label_revealed_evaluation(
    output_dir: Path,
    scores: pd.DataFrame,
    clusters: dict[str, np.ndarray],
    eval_label: pd.Series,
) -> tuple[pd.DataFrame, Path]:
    labels = eval_label.to_numpy(dtype=int)
    total_bots = int((labels == 1).sum())
    rows: list[dict[str, Any]] = []

    for score_row in scores.itertuples(index=False):
        members = clusters.get(str(score_row.cluster_key), np.array([], dtype=int))
        member_labels = labels[members]
        bot_count = int((member_labels == 1).sum())
        human_count = int((member_labels == 0).sum())
        size = int(len(members))
        rows.append(
            {
                "row_type": "cluster",
                "top_cluster_count": np.nan,
                "suspicious_rank": int(score_row.suspicious_rank),
                "cluster_key": score_row.cluster_key,
                "method": score_row.method,
                "run_id": score_row.run_id,
                "cluster_id": int(score_row.cluster_id),
                "selected_account_count": size,
                "human_count": human_count,
                "bot_count": bot_count,
                "bot_rate": float(bot_count / size) if size else 0.0,
                "bot_precision": float(bot_count / size) if size else 0.0,
                "bot_recall": float(bot_count / total_bots) if total_bots else 0.0,
                "suspicious_score": float(score_row.suspicious_score),
            }
        )

    cutoffs = [1, 3, 5, 10, 20, 50]
    cutoffs = [cutoff for cutoff in cutoffs if cutoff <= len(scores)]
    if len(scores) and len(scores) not in cutoffs:
        cutoffs.append(len(scores))
    for cutoff in cutoffs:
        selected_members: set[int] = set()
        for score_row in scores.head(cutoff).itertuples(index=False):
            selected_members.update(map(int, clusters.get(str(score_row.cluster_key), np.array([], dtype=int))))
        chosen = np.array(sorted(selected_members), dtype=int)
        chosen_labels = labels[chosen] if len(chosen) else np.array([], dtype=int)
        bot_count = int((chosen_labels == 1).sum())
        human_count = int((chosen_labels == 0).sum())
        selected_count = int(len(chosen))
        rows.append(
            {
                "row_type": "top_clusters_union",
                "top_cluster_count": int(cutoff),
                "suspicious_rank": np.nan,
                "cluster_key": "ALL",
                "method": "ALL",
                "run_id": "ALL",
                "cluster_id": np.nan,
                "selected_account_count": selected_count,
                "human_count": human_count,
                "bot_count": bot_count,
                "bot_rate": float(bot_count / selected_count) if selected_count else 0.0,
                "bot_precision": float(bot_count / selected_count) if selected_count else 0.0,
                "bot_recall": float(bot_count / total_bots) if total_bots else 0.0,
                "suspicious_score": np.nan,
            }
        )

    label_eval = pd.DataFrame(rows)
    path = output_dir / "density_label_revealed_evaluation.csv"
    label_eval.to_csv(path, index=False)
    return label_eval, path


def plot_knn_similarity_distribution(edge_pairs: pd.DataFrame, thresholds: list[float], figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    if not edge_pairs.empty:
        sims = edge_pairs["cosine_similarity"].to_numpy(dtype=float)
        ax.hist(sims, bins=80, color="#2F80ED", alpha=0.78, label="TF-IDF kNN edges")
        for idx, threshold in enumerate(thresholds):
            ax.axvline(
                threshold,
                color="#D64545",
                linestyle="--",
                linewidth=1.0,
                label="dense scan thresholds" if idx == 0 else None,
            )
    ax.set_title("TF-IDF kNN cosine similarity distribution")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("edge count")
    ax.grid(alpha=0.2)
    ax.legend(frameon=True)
    fig.tight_layout()
    path = figures_dir / "density_knn_similarity_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_top_cluster_purity(label_eval: pd.DataFrame, figures_dir: Path, top_n: int) -> Path:
    clusters = label_eval[label_eval["row_type"].eq("cluster")].head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    red_count = yellow_count = green_count = 0
    if not clusters.empty:
        labels = [f"{int(row.suspicious_rank)}\\n{row.method}" for row in clusters.itertuples(index=False)]
        red_count = int((clusters["bot_rate"] >= 0.8).sum())
        yellow_count = int(((clusters["bot_rate"] >= 0.5) & (clusters["bot_rate"] < 0.8)).sum())
        green_count = int((clusters["bot_rate"] < 0.5).sum())
        colors = [
            "#D64545" if value >= 0.8 else "#F2C94C" if value >= 0.5 else "#6FCF97"
            for value in clusters["bot_rate"]
        ]
        ax.bar(np.arange(len(clusters)), clusters["bot_rate"], color=colors, edgecolor="#222222", linewidth=0.25)
        ax.set_xticks(np.arange(len(clusters)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        if yellow_count == 0:
            ax.text(
                0.01,
                0.96,
                f"No top-{len(clusters)} cluster falls in 0.50-0.80 bot purity",
                transform=ax.transAxes,
                fontsize=8,
                color="#6A4F00",
                va="top",
            )
    ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Top {top_n} suspicious dense clusters: revealed bot purity")
    ax.set_ylabel("revealed bot purity")
    ax.grid(axis="y", alpha=0.2)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#D64545", label=f"bot purity >= 0.80 (n={red_count})"),
        plt.Rectangle((0, 0), 1, 1, color="#F2C94C", label=f"0.50 <= bot purity < 0.80 (n={yellow_count})"),
        plt.Rectangle((0, 0), 1, 1, color="#6FCF97", label=f"bot purity < 0.50 (n={green_count})"),
        plt.Line2D([0], [0], color="#555555", linestyle=":", label="0.50 reference"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)
    fig.tight_layout()
    path = figures_dir / "density_top_cluster_purity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_score_vs_purity(label_eval: pd.DataFrame, figures_dir: Path) -> Path:
    clusters = label_eval[label_eval["row_type"].eq("cluster")].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    if not clusters.empty:
        methods = sorted(clusters["method"].dropna().unique())
        palette = {"hdbscan": "#2F80ED", "dense_component": "#EB5757"}
        for method in methods:
            part = clusters[clusters["method"].eq(method)]
            ax.scatter(
                part["suspicious_score"],
                part["bot_rate"],
                s=np.clip(part["selected_account_count"], 20, 500) / 4,
                alpha=0.68,
                label=method,
                color=palette.get(method, None),
                edgecolors="white",
                linewidths=0.4,
            )
    ax.set_title("Suspicious score vs revealed bot purity")
    ax.set_xlabel("label-free suspicious score")
    ax.set_ylabel("revealed bot purity")
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.95, color="#D64545", linestyle="--", linewidth=0.9, label="0.95 purity")
    ax.axhline(0.50, color="#555555", linestyle=":", linewidth=0.9, label="0.50 purity")
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    size_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="larger dot = larger cluster", markerfacecolor="#9B9B9B", markersize=7),
    ]
    ax.legend(handles + size_handles, labels + ["larger dot = larger cluster"], frameon=True)
    fig.tight_layout()
    path = figures_dir / "density_score_vs_purity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def compute_2d_embedding(embedding: np.ndarray, random_state: int) -> tuple[np.ndarray, str]:
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, min(30, len(embedding) - 1)),
            min_dist=0.05,
            metric="euclidean",
            random_state=random_state,
        )
        return reducer.fit_transform(embedding), "UMAP"
    except Exception:
        return PCA(n_components=2, random_state=random_state).fit_transform(embedding), "PCA fallback"


def plot_embedding(
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    top_members: set[int],
    figures_dir: Path,
    random_state: int,
) -> tuple[Path, str]:
    coords, method = compute_2d_embedding(embedding, random_state)
    labels = metadata["eval_label"].to_numpy(dtype=int) if "eval_label" in metadata.columns else np.zeros(len(coords), dtype=int)
    top_mask = np.array([idx in top_members for idx in range(len(coords))], dtype=bool)

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    human_mask = labels == 0
    bot_mask = labels == 1
    ax.scatter(
        coords[human_mask, 0],
        coords[human_mask, 1],
        c="#2F80ED",
        s=20,
        alpha=0.45,
        linewidths=0,
        label="revealed human",
    )
    ax.scatter(
        coords[bot_mask, 0],
        coords[bot_mask, 1],
        c="#D64545",
        s=20,
        alpha=0.55,
        linewidths=0,
        label="revealed old bot",
    )
    if top_mask.any():
        ax.scatter(
            coords[top_mask, 0],
            coords[top_mask, 1],
            facecolors="none",
            edgecolors="#111111",
            s=64,
            linewidths=1.1,
            label="Top suspicious cluster",
        )
    ax.legend(frameon=True)
    ax.set_title(f"Account TF-IDF density embedding ({method})")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = figures_dir / "density_account_embedding_2d.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, method


def top_table(label_eval: pd.DataFrame, top_n: int) -> pd.DataFrame:
    cols = [
        "suspicious_rank",
        "method",
        "cluster_key",
        "selected_account_count",
        "human_count",
        "bot_count",
        "bot_rate",
        "bot_recall",
        "suspicious_score",
    ]
    clusters = label_eval[label_eval["row_type"].eq("cluster")].copy()
    return clusters[[col for col in cols if col in clusters.columns]].head(top_n)


def union_table(label_eval: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "top_cluster_count",
        "selected_account_count",
        "human_count",
        "bot_count",
        "bot_precision",
        "bot_recall",
    ]
    rows = label_eval[label_eval["row_type"].eq("top_clusters_union")].copy()
    return rows[[col for col in cols if col in rows.columns]]


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_无数据_"
    view = df.head(max_rows).copy()

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    headers = [str(col) for col in view.columns]
    rows = [[fmt(value) for value in row] for row in view.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    selected: pd.DataFrame,
    scores: pd.DataFrame,
    label_eval: pd.DataFrame,
    thresholds: list[float],
    figures: dict[str, str],
    args: argparse.Namespace,
) -> Path:
    cluster_rows = label_eval[label_eval["row_type"].eq("cluster")]
    union_rows = label_eval[label_eval["row_type"].eq("top_clusters_union")]
    top1 = union_rows.head(1)
    top1_text = "无"
    if not top1.empty:
        top1_text = (
            f"precision={float(top1['bot_precision'].iloc[0]):.4f}, "
            f"recall={float(top1['bot_recall'].iloc[0]):.4f}, "
            f"selected={int(top1['selected_account_count'].iloc[0])}"
        )

    by_method = []
    for method, group in cluster_rows.groupby("method"):
        high = group[group["bot_rate"] >= 0.95]
        by_method.append(
            {
                "method": method,
                "clusters": len(group),
                "high_purity_ge_095": len(high),
                "max_bot_rate": group["bot_rate"].max(),
                "max_bot_recall": group["bot_recall"].max(),
                "best_rank": group["suspicious_rank"].min(),
            }
        )
    by_method_df = pd.DataFrame(by_method)

    lines = [
        "# 密度式账号机制簇发现实验报告",
        "",
        "## 1. 实验目的",
        "",
        "本实验把上一版 KMeans 基线替换为更贴合问题的密度发现方案。目标不是强行把所有账号分成“人/机器人”两类，而是在账号级 TF-IDF 语言空间里寻找异常高密度的 suspicious dense clusters。",
        "",
        "流程是：",
        "",
        "```text",
        "账号聚合文本 -> word TF-IDF -> HDBSCAN + dense component scan -> suspicious dense clusters -> 最后揭示标签评估 bot purity / recall -> PCA/UMAP 2D 展示",
        "```",
        "",
        "标签不参与 TF-IDF、HDBSCAN、相似图、dense component scan 或 suspicious score，只在最后的 revealed evaluation 中使用。",
        "",
        "## 2. 数据口径",
        "",
        f"- 输入目录：`{Path(args.input_dir)}`",
        f"- 输出目录：`{output_dir}`",
        f"- 筛选 family：`{TARGET_FAMILIES}`",
        f"- 最小词数：`{args.min_words}`",
        f"- 进入实验账号数：`{len(selected)}`",
        f"- human 数：`{int((selected['family'].astype(str) == HUMAN_FAMILY).sum())}`",
        f"- old bot 数：`{int((selected['family'].astype(str) == BOT_FAMILY).sum())}`",
        "",
        "## 3. 方法实现",
        "",
        "### 3.1 主空间：word TF-IDF",
        "",
        f"- word 1-2gram TF-IDF，最大特征数 `{args.tfidf_max_features}`。",
        "- 使用 cosine 相似度作为账号文本分布接近程度。",
        "- dense component scan 在原始 TF-IDF cosine kNN 图上运行。",
        "- HDBSCAN 在 SVD 降维并标准化后的 TF-IDF 空间运行，避免直接在几万维稀疏空间里估计密度。",
        "",
        "### 3.2 HDBSCAN",
        "",
        f"- SVD 维度：`{args.svd_components}` 请求，实际维度见 `density_account_features.csv`。",
        f"- min_cluster_size：`{args.hdbscan_min_cluster_sizes}`",
        f"- min_samples：`{args.hdbscan_min_samples}`",
        "- 输出非噪声密集簇，过大的簇按 `max_cluster_fraction` 过滤，避免把半个数据集当作一个机制簇。",
        "",
        "### 3.3 Dense component scan",
        "",
        f"- kNN 图：每个账号连接 TF-IDF cosine 最近的 `{args.component_knn_k}` 个账号。",
        f"- 自动阈值 quantiles：`{args.component_threshold_quantiles}`。",
        f"- 实际 cosine thresholds：`{', '.join(f'{value:.6f}' for value in thresholds)}`。",
        f"- 最小 component size：`{args.component_min_size}`。",
        f"- 最小 component 内部边密度：`{args.component_min_edge_density}`。",
        f"- 最大 cluster 占比：`{args.max_cluster_fraction}`。",
        "- 每个阈值下保留高相似边，寻找 connected components。若一批账号在高相似阈值下仍连成组件，就说明它们在词汇使用分布上非常接近。",
        "",
        "### 3.4 Suspicious score",
        "",
        "每个候选簇都在原始 TF-IDF 空间里计算：",
        "",
        "- 簇内平均 cosine similarity；",
        "- 簇半径；",
        "- 与同规模随机账号组的 null baseline 对比；",
        "- dense component 的图边密度；",
        "- HDBSCAN 的平均 membership strength；",
        "- 簇规模校正。",
        "",
        "评分越高，表示这个簇越像“异常集中的账号语言机制团块”。",
        "",
        "## 4. 输出规模",
        "",
        f"- suspicious clusters：`{len(scores)}`",
        f"- HDBSCAN/dense assignments 文件：`density_cluster_assignments.csv`",
        f"- suspicious score 文件：`density_suspicious_cluster_scores.csv`",
        f"- 标签揭示评估文件：`density_label_revealed_evaluation.csv`",
        f"- Top-1 并集结果：`{top1_text}`",
        "",
        "## 5. Top suspicious dense clusters",
        "",
        markdown_table(top_table(label_eval, args.top_n_report), max_rows=args.top_n_report),
        "",
        "## 6. Top-N 并集评估",
        "",
        markdown_table(union_table(label_eval), max_rows=20),
        "",
        "## 7. 分方法表现",
        "",
        markdown_table(by_method_df, max_rows=20),
        "",
        "## 8. 图表",
        "",
        f"- kNN 相似度分布：`{figures.get('knn', '')}`",
        f"- Top cluster purity：`{figures.get('purity', '')}`",
        f"- Suspicious score vs purity：`{figures.get('score', '')}`",
        f"- 2D 展示图：`{figures.get('embedding', '')}`",
        f"- 2D 降维方法：`{figures.get('embedding_method', '')}`",
        "",
        "## 9. 结论口径",
        "",
        "- 这不是二分类实验，不假设世界只有 human cluster 和 bot cluster。",
        "- HDBSCAN 和 dense component scan 直接寻找高密度账号群，更贴合“某些账号集中在某个点上”的假设。",
        "- 标签揭示只用于评估这些高密度簇是否富含 old bot。",
        "- 2D 图只用于展示，不作为判定依据；真实判定来自原始 TF-IDF 空间的密度、半径和 null baseline。",
    ]
    path = output_dir / "DENSITY_ACCOUNT_DISCOVERY_CN.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    output_dir = ensure_dir(Path(args.output_dir))
    figures_dir = ensure_dir(output_dir / "figures")

    input_dir = Path(args.input_dir)
    combined_path = input_dir / "combined_sample_index.csv"
    if not combined_path.exists():
        raise FileNotFoundError(f"Missing input file: {combined_path}")

    print(f"[info] Loading {combined_path}")
    combined = pd.read_csv(combined_path)
    selected = select_rows(combined, args.min_words, args.sample_human, args.sample_bot, args.random_state)
    eval_label = build_eval_label(selected)
    docs = selected["doc"].fillna("").astype(str).tolist()
    account_ids = selected["account_id"].astype(str).to_numpy()
    print(
        "[info] Selected accounts: "
        f"n={len(selected)}, humans={(selected['family'].astype(str) == HUMAN_FAMILY).sum()}, "
        f"bots={(selected['family'].astype(str) == BOT_FAMILY).sum()}"
    )

    print("[info] Building primary word TF-IDF view")
    tfidf_view = build_primary_tfidf(docs, args.tfidf_max_features)
    style_df = style_features(docs, selected)
    embedding, actual_svd_components = svd_embedding(tfidf_view, args.svd_components, args.random_state)

    output_paths: list[Path] = []
    output_paths.append(write_account_features(output_dir, selected, style_df, tfidf_view, actual_svd_components))
    metadata_path = write_label_revealed_account_metadata(output_dir, selected, eval_label)
    output_paths.append(metadata_path)

    hdbscan_assignments, hdbscan_clusters, hdbscan_meta = run_hdbscan(
        embedding,
        parse_int_list(args.hdbscan_min_cluster_sizes),
        args.hdbscan_min_samples,
        account_ids,
        args.max_cluster_fraction,
    )

    print(f"[info] Building TF-IDF kNN graph for dense component scan, k={args.component_knn_k}")
    edge_pairs, directed_edges = build_knn_edges(tfidf_view.matrix, account_ids, args.component_knn_k)
    edge_pairs_path = output_dir / "density_similarity_edges.csv"
    edge_pairs.to_csv(edge_pairs_path, index=False)
    output_paths.append(edge_pairs_path)
    directed_edges_path = output_dir / "density_similarity_directed_knn_edges.csv"
    directed_edges.to_csv(directed_edges_path, index=False)
    output_paths.append(directed_edges_path)

    thresholds = automatic_thresholds(
        edge_pairs["cosine_similarity"].to_numpy(dtype=float) if not edge_pairs.empty else np.array([]),
        parse_float_list(args.component_threshold_quantiles),
    )
    dense_assignments, dense_clusters, dense_meta = run_dense_component_scan(
        edge_pairs,
        thresholds,
        len(selected),
        account_ids,
        args.component_min_size,
        args.component_min_edge_density,
        args.max_cluster_fraction,
    )

    clusters = {**hdbscan_clusters, **dense_clusters}
    meta = {**hdbscan_meta, **dense_meta}
    output_paths.append(write_cluster_assignments(output_dir, [hdbscan_assignments, dense_assignments]))
    scores, scores_path = compute_scores(
        output_dir,
        tfidf_view,
        clusters,
        meta,
        account_ids,
        args.null_repeats,
        args.random_state,
    )
    output_paths.append(scores_path)
    label_eval, label_eval_path = write_label_revealed_evaluation(output_dir, scores, clusters, eval_label)
    output_paths.append(label_eval_path)

    figures: dict[str, str] = {}
    figures["knn"] = str(plot_knn_similarity_distribution(edge_pairs, thresholds, figures_dir))
    figures["purity"] = str(plot_top_cluster_purity(label_eval, figures_dir, args.top_n_report))
    figures["score"] = str(plot_score_vs_purity(label_eval, figures_dir))
    top_members: set[int] = set()
    if not scores.empty:
        top_key = str(scores.iloc[0]["cluster_key"])
        top_members = set(map(int, clusters.get(top_key, np.array([], dtype=int))))
    embedding_path, embedding_method = plot_embedding(
        embedding,
        pd.read_csv(metadata_path),
        top_members,
        figures_dir,
        args.random_state,
    )
    figures["embedding"] = str(embedding_path)
    figures["embedding_method"] = embedding_method

    report_path = write_report(output_dir, selected, scores, label_eval, thresholds, figures, args)
    output_paths.append(report_path)

    print("[info] Wrote outputs:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
