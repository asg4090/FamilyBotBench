#!/usr/bin/env python3
"""Run anchor similarity evaluation for the static text benchmark."""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

from common_static_text import (
    FAMILIES,
    FIGURES_DIR,
    RESULTS_DIR,
    SHORT_NAME,
    STATIC_RUN,
    get_original_train_val_test_indices,
    load_combined_data,
    load_generated_tables,
    predict_bge_lr,
    predict_tfidf_lr,
    sample_text_stats,
    save_figure,
    save_table,
    train_bge_lr,
    train_tfidf_lr,
)


OUTPUT_DIR = RESULTS_DIR / "anchor_similarity"
AUDIT_PATH = OUTPUT_DIR / "anchor_linkage_audit.csv"
COMMENT_PATH = OUTPUT_DIR / "anchor_similarity_by_comment.csv"
FAMILY_PATH = OUTPUT_DIR / "anchor_similarity_by_family.csv"
DETECTION_PATH = OUTPUT_DIR / "anchor_similarity_vs_detection.csv"

ANCHOR_BOXPLOT_PATH = FIGURES_DIR / "anchor_similarity_by_family_boxplot.png"
SIMILARITY_PROB_PATH = FIGURES_DIR / "anchor_similarity_vs_bot_probability.png"
STYLE_BOXPLOT_PATH = FIGURES_DIR / "stylistic_distance_by_family.png"

PREDICTIONS_PATH = STATIC_RUN / "static_language_family_predictions.csv"

FAMILY_SHORT_ORDER = [SHORT_NAME[family] for family in FAMILIES]
STYLE_FEATURES = list(sample_text_stats("").keys())


def _nonempty_text_mask(df: pd.DataFrame) -> pd.Series:
    return df["doc"].fillna("").astype(str).str.strip().ne("") & (df["word_count"].fillna(0) > 0)


def _split_anchor_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return values / norms


def _safe_corr(x: Iterable[float], y: Iterable[float], method: str) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return float("nan")
    if np.nanstd(x_arr[mask]) == 0.0 or np.nanstd(y_arr[mask]) == 0.0:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(x_arr[mask], y_arr[mask]).statistic)
    if method == "spearman":
        return float(spearmanr(x_arr[mask], y_arr[mask]).statistic)
    raise ValueError(f"unknown correlation method: {method}")


def _make_anchor_lookup(anchors: pd.DataFrame) -> dict[str, list[str]]:
    return {
        str(row.bot_user_id): _split_anchor_ids(row.anchor_user_ids)
        for row in anchors.itertuples(index=False)
    }


def _build_linkage_audit(
    profiles: pd.DataFrame,
    comments: pd.DataFrame,
    anchors: pd.DataFrame,
    human_user_ids: set[str],
) -> pd.DataFrame:
    anchor_lookup = _make_anchor_lookup(anchors)
    unique_anchor_users = sorted(
        {
            anchor_user_id
            for anchor_ids in anchor_lookup.values()
            for anchor_user_id in anchor_ids
        }
    )
    resolved_anchor_users = {user_id for user_id in unique_anchor_users if user_id in human_user_ids}

    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        family_profiles = profiles[profiles["family"] == family]
        family_comments = comments[comments["family"] == family]
        generated_users = set(family_profiles["user_id"].astype(str))
        users_with_manifest = {user_id for user_id in generated_users if user_id in anchor_lookup}
        users_with_resolved_anchors = {
            user_id
            for user_id in users_with_manifest
            if anchor_lookup[user_id] and all(anchor_id in human_user_ids for anchor_id in anchor_lookup[user_id])
        }
        comments_with_manifest = family_comments["user_id"].astype(str).isin(users_with_manifest).sum()
        comments_with_resolved_anchors = family_comments["user_id"].astype(str).isin(users_with_resolved_anchors).sum()

        user_coverage = len(users_with_resolved_anchors) / len(generated_users) if generated_users else float("nan")
        comment_coverage = (
            comments_with_resolved_anchors / len(family_comments) if len(family_comments) else float("nan")
        )
        if len(users_with_resolved_anchors) == len(generated_users) and generated_users:
            mode = "assigned_anchor_plus_nearest_human_pool"
            note = "All generated users have resolved specified anchors; assigned-anchor and nearest-pool metrics are both valid."
        elif users_with_manifest:
            mode = "partial_assigned_anchor_with_nearest_human_pool_fallback"
            note = "Some specified anchors could not be resolved; unresolved rows fall back to nearest original human pool metrics."
        else:
            mode = "nearest_human_pool_fallback_no_manifest"
            note = "No specified anchor manifest rows for this family; only nearest original human pool metrics are applicable."

        rows.append(
            {
                "family": family,
                "family_short": SHORT_NAME[family],
                "generated_users": len(generated_users),
                "generated_comments": int(len(family_comments)),
                "users_with_anchor_manifest": len(users_with_manifest),
                "users_with_resolved_specified_anchors": len(users_with_resolved_anchors),
                "comments_with_anchor_manifest": int(comments_with_manifest),
                "comments_with_resolved_specified_anchors": int(comments_with_resolved_anchors),
                "user_anchor_linkage_coverage": float(user_coverage),
                "comment_anchor_linkage_coverage": float(comment_coverage),
                "analysis_mode": mode,
                "anchor_manifest_rows": int(len(anchors)),
                "anchor_manifest_unique_anchor_users": len(unique_anchor_users),
                "anchor_manifest_unique_anchor_users_resolved": len(resolved_anchor_users),
                "audit_note": note,
            }
        )
    return pd.DataFrame(rows)


def _fit_anchor_tfidf(human_docs: list[str], bot_comment_docs: list[str]):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(human_docs + bot_comment_docs)
    human_matrix = matrix[: len(human_docs)]
    bot_matrix = matrix[len(human_docs) :]
    return human_matrix, bot_matrix


def _matrix_row_max(matrix: sparse.spmatrix, row_positions: list[int]) -> tuple[float, float, int | None]:
    if not row_positions:
        return float("nan"), float("nan"), None
    sims = matrix[:, row_positions].toarray().ravel()
    if sims.size == 0:
        return float("nan"), float("nan"), None
    best_offset = int(np.argmax(sims))
    return float(sims[best_offset]), float(np.mean(sims)), row_positions[best_offset]


def _compute_tfidf_metrics(
    comment_df: pd.DataFrame,
    human_df: pd.DataFrame,
    anchor_lookup: dict[str, list[str]],
) -> pd.DataFrame:
    human_docs = human_df["doc"].fillna("").astype(str).tolist()
    bot_docs = comment_df["comment_body"].fillna("").astype(str).tolist()
    human_matrix, bot_matrix = _fit_anchor_tfidf(human_docs, bot_docs)
    all_sims = bot_matrix @ human_matrix.T

    human_user_ids = human_df["user_id"].astype(str).tolist()
    human_user_to_pos = {user_id: pos for pos, user_id in enumerate(human_user_ids)}
    nearest_pos = np.asarray(all_sims.argmax(axis=1)).ravel()
    nearest_scores_sparse = all_sims.max(axis=1)
    nearest_scores = np.asarray(nearest_scores_sparse.toarray()).ravel()

    assigned_best: list[float] = []
    assigned_mean: list[float] = []
    assigned_nearest_user: list[str | None] = []
    nearest_users: list[str] = []
    for row_idx, row in enumerate(comment_df.itertuples(index=False)):
        bot_user_id = str(row.bot_user_id)
        anchor_positions = [
            human_user_to_pos[anchor_id]
            for anchor_id in anchor_lookup.get(bot_user_id, [])
            if anchor_id in human_user_to_pos
        ]
        best, mean, assigned_pos = _matrix_row_max(all_sims[row_idx], anchor_positions)
        assigned_best.append(best)
        assigned_mean.append(mean)
        assigned_nearest_user.append(human_user_ids[assigned_pos] if assigned_pos is not None else None)
        nearest_users.append(human_user_ids[int(nearest_pos[row_idx])])

    out = pd.DataFrame(
        {
            "bot_to_assigned_anchor_tfidf_cosine": assigned_best,
            "bot_to_assigned_anchor_tfidf_mean_cosine": assigned_mean,
            "nearest_assigned_anchor_user_id_tfidf": assigned_nearest_user,
            "bot_to_nearest_anchor_tfidf_cosine": nearest_scores.astype(float),
            "nearest_anchor_pool_user_id_tfidf": nearest_users,
        }
    )
    return out


def _compute_user_bge_metrics(
    combined: pd.DataFrame,
    embeddings: np.ndarray,
    human_idx: np.ndarray,
    original_bot_idx: np.ndarray,
    anchor_lookup: dict[str, list[str]],
) -> pd.DataFrame:
    generated_idx = combined.index[combined["source"] == "static_language_bot"].to_numpy()
    user_to_idx = {str(row.user_id): int(idx) for idx, row in combined.iterrows()}
    human_user_ids = combined.loc[human_idx, "user_id"].astype(str).tolist()
    human_user_to_pos = {user_id: pos for pos, user_id in enumerate(human_user_ids)}

    human_embeddings = embeddings[human_idx]
    human_norm = _l2_normalize(human_embeddings)
    generated_embeddings = embeddings[generated_idx]
    generated_norm = _l2_normalize(generated_embeddings)
    sims = generated_norm @ human_norm.T
    nearest_pos = sims.argmax(axis=1)
    nearest_scores = sims.max(axis=1)

    human_centroid = human_embeddings.mean(axis=0)
    bot_centroid = embeddings[original_bot_idx].mean(axis=0)
    generated_to_human_centroid = np.linalg.norm(generated_embeddings - human_centroid, axis=1)
    generated_to_bot_centroid = np.linalg.norm(generated_embeddings - bot_centroid, axis=1)

    rows: list[dict[str, object]] = []
    for rel_pos, sample_idx in enumerate(generated_idx):
        sample = combined.loc[int(sample_idx)]
        bot_user_id = str(sample["user_id"])
        anchor_positions = [
            human_user_to_pos[anchor_id]
            for anchor_id in anchor_lookup.get(bot_user_id, [])
            if anchor_id in human_user_to_pos
        ]
        if anchor_positions:
            assigned_sims = sims[rel_pos, anchor_positions]
            assigned_best_offset = int(np.argmax(assigned_sims))
            assigned_best = float(assigned_sims[assigned_best_offset])
            assigned_mean = float(np.mean(assigned_sims))
            assigned_user = human_user_ids[anchor_positions[assigned_best_offset]]
        else:
            assigned_best = float("nan")
            assigned_mean = float("nan")
            assigned_user = None
        rows.append(
            {
                "user_id": bot_user_id,
                "combined_row": int(sample["row"]),
                "bot_to_nearest_human_bge_cosine": float(nearest_scores[rel_pos]),
                "nearest_original_human_user_id_bge": human_user_ids[int(nearest_pos[rel_pos])],
                "bot_to_assigned_anchor_bge_cosine": assigned_best,
                "bot_to_assigned_anchor_bge_mean_cosine": assigned_mean,
                "nearest_assigned_anchor_user_id_bge": assigned_user,
                "bot_to_human_centroid_distance": float(generated_to_human_centroid[rel_pos]),
                "bot_to_bot_centroid_distance": float(generated_to_bot_centroid[rel_pos]),
            }
        )
    return pd.DataFrame(rows)


def _style_frame(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([sample_text_stats(text) for text in texts.fillna("").astype(str)], columns=STYLE_FEATURES)


def _compute_style_metrics(
    comment_df: pd.DataFrame,
    human_df: pd.DataFrame,
    anchor_lookup: dict[str, list[str]],
) -> pd.DataFrame:
    human_stats = _style_frame(human_df["doc"])
    bot_stats = _style_frame(comment_df["comment_body"])
    human_mean = human_stats.mean(axis=0)
    human_std = human_stats.std(axis=0, ddof=0).replace(0.0, 1.0)

    human_z = (human_stats - human_mean) / human_std
    bot_z = (bot_stats - human_mean) / human_std
    human_user_to_z = {
        str(user_id): human_z.iloc[pos].to_numpy(dtype=float)
        for pos, user_id in enumerate(human_df["user_id"].astype(str).tolist())
    }
    human_centroid_z = np.zeros(len(STYLE_FEATURES), dtype=float)

    assigned_distance: list[float] = []
    human_centroid_distance: list[float] = []
    nearest_human_distance: list[float] = []
    nearest_human_user: list[str | None] = []
    human_z_matrix = human_z.to_numpy(dtype=float)

    for row_idx, row in enumerate(comment_df.itertuples(index=False)):
        bot_vec = bot_z.iloc[row_idx].to_numpy(dtype=float)
        human_centroid_distance.append(float(np.linalg.norm(bot_vec - human_centroid_z)))

        distances = np.linalg.norm(human_z_matrix - bot_vec, axis=1)
        nearest_pos = int(np.argmin(distances))
        nearest_human_distance.append(float(distances[nearest_pos]))
        nearest_human_user.append(str(human_df.iloc[nearest_pos]["user_id"]))

        anchor_vectors = [
            human_user_to_z[anchor_id]
            for anchor_id in anchor_lookup.get(str(row.bot_user_id), [])
            if anchor_id in human_user_to_z
        ]
        if anchor_vectors:
            anchor_centroid = np.vstack(anchor_vectors).mean(axis=0)
            assigned_distance.append(float(np.linalg.norm(bot_vec - anchor_centroid)))
        else:
            assigned_distance.append(float("nan"))

    out = bot_stats.add_prefix("style_")
    out["bot_to_assigned_anchor_style_z_l2"] = assigned_distance
    out["bot_to_human_centroid_style_z_l2"] = human_centroid_distance
    out["bot_to_nearest_human_style_z_l2"] = nearest_human_distance
    out["nearest_original_human_user_id_style"] = nearest_human_user
    return out


def _load_or_train_detection_probabilities(
    combined: pd.DataFrame,
    docs: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    generated = combined[combined["source"] == "static_language_bot"][["row", "user_id", "family"]].copy()
    generated["row"] = generated["row"].astype(int)

    if PREDICTIONS_PATH.exists():
        predictions = pd.read_csv(PREDICTIONS_PATH)
        subset = predictions[
            (predictions["protocol"] == "original_only_zero_shot")
            & (predictions["train_family"] == "original_only")
            & (predictions["model_x"].isin(["tfidf", "bge"]))
        ].copy()
        if not subset.empty:
            by_user_model = (
                subset.groupby(["user_id", "model_x"], as_index=False)["bot_probability"]
                .mean()
                .pivot(index="user_id", columns="model_x", values="bot_probability")
                .reset_index()
            )
            for model_name in ["tfidf", "bge"]:
                if model_name not in by_user_model.columns:
                    by_user_model[model_name] = np.nan
            out = generated.merge(by_user_model, on="user_id", how="left")
            if out[["tfidf", "bge"]].notna().all().all():
                return pd.DataFrame(
                    {
                        "user_id": out["user_id"],
                        "detector_tfidf_bot_probability": out["tfidf"].astype(float),
                        "detector_bge_bot_probability": out["bge"].astype(float),
                        "detector_probability_source": "static_language_family_predictions:original_only_zero_shot",
                    }
                )

    original_train_idx, _original_val_idx, _original_test_idx = get_original_train_val_test_indices(
        combined,
        labels,
    )
    generated_idx = generated.index.to_numpy()
    tfidf_model = train_tfidf_lr(docs, labels, original_train_idx)
    bge_model = train_bge_lr(embeddings, labels, original_train_idx)
    _tfidf_pred, tfidf_prob = predict_tfidf_lr(tfidf_model, docs, generated_idx)
    _bge_pred, bge_prob = predict_bge_lr(bge_model, embeddings, generated_idx)
    return pd.DataFrame(
        {
            "user_id": generated["user_id"].astype(str).to_numpy(),
            "detector_tfidf_bot_probability": tfidf_prob.astype(float),
            "detector_bge_bot_probability": bge_prob.astype(float),
            "detector_probability_source": "trained_fallback_original_train",
        }
    )


def _build_family_summary(comment_df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    audit_by_family = audit_df.set_index("family")
    for family in FAMILIES:
        family_df = comment_df[comment_df["family"] == family]
        audit = audit_by_family.loc[family]
        rows.append(
            {
                "family": family,
                "family_short": SHORT_NAME[family],
                "n_comments": int(len(family_df)),
                "n_users": int(family_df["bot_user_id"].nunique()),
                "comment_anchor_linkage_coverage": float(audit["comment_anchor_linkage_coverage"]),
                "analysis_mode": audit["analysis_mode"],
                "nearest_anchor_tfidf_cosine_mean": float(family_df["bot_to_nearest_anchor_tfidf_cosine"].mean()),
                "nearest_anchor_tfidf_cosine_median": float(family_df["bot_to_nearest_anchor_tfidf_cosine"].median()),
                "assigned_anchor_tfidf_cosine_mean": float(family_df["bot_to_assigned_anchor_tfidf_cosine"].mean()),
                "nearest_human_bge_cosine_mean": float(family_df["bot_to_nearest_human_bge_cosine"].mean()),
                "nearest_human_bge_cosine_median": float(family_df["bot_to_nearest_human_bge_cosine"].median()),
                "human_centroid_bge_distance_mean": float(family_df["bot_to_human_centroid_distance"].mean()),
                "original_bot_centroid_bge_distance_mean": float(family_df["bot_to_bot_centroid_distance"].mean()),
                "human_centroid_style_z_l2_mean": float(family_df["bot_to_human_centroid_style_z_l2"].mean()),
                "human_centroid_style_z_l2_median": float(family_df["bot_to_human_centroid_style_z_l2"].median()),
                "assigned_anchor_style_z_l2_mean": float(family_df["bot_to_assigned_anchor_style_z_l2"].mean()),
                "detector_tfidf_bot_probability_mean": float(family_df["detector_tfidf_bot_probability"].mean()),
                "detector_bge_bot_probability_mean": float(family_df["detector_bge_bot_probability"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _quadrant_counts(similarity: pd.Series, probability: pd.Series, higher_is_more_human: bool) -> dict[str, int]:
    mask = similarity.notna() & probability.notna()
    sim = similarity[mask]
    prob = probability[mask]
    if sim.empty:
        return {
            "high_similarity_low_probability_count": 0,
            "high_similarity_high_probability_count": 0,
            "low_similarity_high_probability_count": 0,
            "low_similarity_low_probability_count": 0,
        }
    threshold = float(sim.median())
    if higher_is_more_human:
        high_similarity = sim >= threshold
    else:
        high_similarity = sim <= threshold
    high_probability = prob >= 0.5
    return {
        "high_similarity_low_probability_count": int((high_similarity & ~high_probability).sum()),
        "high_similarity_high_probability_count": int((high_similarity & high_probability).sum()),
        "low_similarity_high_probability_count": int((~high_similarity & high_probability).sum()),
        "low_similarity_low_probability_count": int((~high_similarity & ~high_probability).sum()),
    }


def _build_detection_summary(comment_df: pd.DataFrame) -> pd.DataFrame:
    metric_specs = [
        ("bot_to_nearest_anchor_tfidf_cosine", "higher_more_human"),
        ("bot_to_nearest_human_bge_cosine", "higher_more_human"),
        ("bot_to_human_centroid_style_z_l2", "lower_more_human"),
    ]
    detector_specs = [
        ("TF-IDF detector", "detector_tfidf_bot_probability"),
        ("BGE detector", "detector_bge_bot_probability"),
    ]

    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        family_df = comment_df[comment_df["family"] == family]
        for detector_name, probability_col in detector_specs:
            for metric, direction in metric_specs:
                higher_is_more_human = direction == "higher_more_human"
                rows.append(
                    {
                        "family": family,
                        "family_short": SHORT_NAME[family],
                        "detector_model": detector_name,
                        "probability_column": probability_col,
                        "similarity_metric": metric,
                        "similarity_direction": direction,
                        "n_comments": int(family_df[[metric, probability_col]].dropna().shape[0]),
                        "mean_similarity": float(family_df[metric].mean()),
                        "median_similarity": float(family_df[metric].median()),
                        "mean_bot_probability": float(family_df[probability_col].mean()),
                        "pearson_similarity_vs_probability": _safe_corr(
                            family_df[metric],
                            family_df[probability_col],
                            "pearson",
                        ),
                        "spearman_similarity_vs_probability": _safe_corr(
                            family_df[metric],
                            family_df[probability_col],
                            "spearman",
                        ),
                        **_quadrant_counts(
                            family_df[metric],
                            family_df[probability_col],
                            higher_is_more_human,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _family_box_data(df: pd.DataFrame, value_col: str) -> list[np.ndarray]:
    return [
        df.loc[df["family_short"] == family_short, value_col].dropna().to_numpy(dtype=float)
        for family_short in FAMILY_SHORT_ORDER
    ]


def _plot_anchor_similarity(comment_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    specs = [
        ("bot_to_nearest_anchor_tfidf_cosine", "TF-IDF nearest original human cosine"),
        ("bot_to_nearest_human_bge_cosine", "BGE nearest original human cosine"),
    ]
    for ax, (metric, title) in zip(axes, specs, strict=True):
        ax.boxplot(
            _family_box_data(comment_df, metric),
            tick_labels=FAMILY_SHORT_ORDER,
            showmeans=True,
            patch_artist=True,
            boxprops={"facecolor": "#dce7f2", "edgecolor": "#45627a"},
            medianprops={"color": "#9a3d3d", "linewidth": 1.4},
            meanprops={"marker": "o", "markerfacecolor": "#1f7a55", "markeredgecolor": "#1f7a55"},
            whiskerprops={"color": "#45627a"},
            capprops={"color": "#45627a"},
        )
        ax.set_title(title)
        ax.set_xlabel("Family")
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axes[0].set_ylabel("Cosine similarity")
    fig.suptitle("Anchor similarity by family", y=1.03)
    fig.tight_layout()
    save_figure(fig, ANCHOR_BOXPLOT_PATH)


def _plot_similarity_vs_probability(comment_df: pd.DataFrame) -> None:
    colors = {
        "F1": "#4c78a8",
        "F2": "#f58518",
        "F3": "#54a24b",
        "F4": "#b279a2",
        "F5": "#e45756",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.2), sharey=True)
    specs = [
        (
            "bot_to_nearest_anchor_tfidf_cosine",
            "detector_tfidf_bot_probability",
            "TF-IDF similarity vs TF-IDF detector",
        ),
        (
            "bot_to_nearest_human_bge_cosine",
            "detector_bge_bot_probability",
            "BGE similarity vs BGE detector",
        ),
    ]
    for ax, (x_col, y_col, title) in zip(axes, specs, strict=True):
        for family_short in FAMILY_SHORT_ORDER:
            part = comment_df[comment_df["family_short"] == family_short]
            ax.scatter(
                part[x_col],
                part[y_col],
                s=22,
                alpha=0.62,
                color=colors[family_short],
                edgecolors="none",
                label=family_short,
            )
        ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Nearest human cosine")
        ax.set_title(title)
        ax.grid(color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axes[0].set_ylabel("Predicted bot probability")
    axes[1].legend(title="Family", loc="best", frameon=False)
    fig.tight_layout()
    save_figure(fig, SIMILARITY_PROB_PATH)


def _plot_style_distance(comment_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.boxplot(
        _family_box_data(comment_df, "bot_to_human_centroid_style_z_l2"),
        tick_labels=FAMILY_SHORT_ORDER,
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": "#efe1cf", "edgecolor": "#7a6045"},
        medianprops={"color": "#9a3d3d", "linewidth": 1.4},
        meanprops={"marker": "o", "markerfacecolor": "#1f7a55", "markeredgecolor": "#1f7a55"},
        whiskerprops={"color": "#7a6045"},
        capprops={"color": "#7a6045"},
    )
    ax.set_title("Stylistic distance to original-human centroid")
    ax.set_xlabel("Family")
    ax.set_ylabel("Standardized L2 distance")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    save_figure(fig, STYLE_BOXPLOT_PATH)


def run_anchor_similarity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined, docs, labels, embeddings = load_combined_data()
    profiles, comments, anchors = load_generated_tables()
    anchor_lookup = _make_anchor_lookup(anchors)

    comments = comments.copy()
    comments["bot_user_id"] = comments["user_id"].astype(str)
    comments["generated_comment_id"] = comments["comment_id"].astype(str)
    comments["family_short"] = comments["family"].map(SHORT_NAME)
    comments["comment_body"] = comments["comment_body"].fillna("").astype(str)

    human_mask = (combined["source"] == "original_human") & _nonempty_text_mask(combined)
    original_bot_mask = (combined["source"] == "original_bot") & _nonempty_text_mask(combined)
    human_df = combined.loc[human_mask].copy()
    human_idx = human_df.index.to_numpy()
    original_bot_idx = combined.index[original_bot_mask].to_numpy()
    human_user_ids = set(human_df["user_id"].astype(str))

    audit_df = _build_linkage_audit(profiles, comments, anchors, human_user_ids)
    save_table(audit_df, AUDIT_PATH)

    comment_base = comments[
        [
            "family",
            "family_short",
            "bot_user_id",
            "user_name",
            "generated_comment_id",
            "link_id",
            "parent_id",
            "subreddit",
            "created_utc",
            "comment_body",
            "f3_revision_mode",
            "f3_revised",
            "f3_revision_iters",
            "f3_detector_prob_before",
            "f3_detector_prob_after",
        ]
    ].copy()
    comment_base["anchor_user_ids"] = comment_base["bot_user_id"].map(
        lambda user_id: "|".join(anchor_lookup.get(str(user_id), []))
    )
    comment_base["assigned_anchor_count"] = comment_base["bot_user_id"].map(
        lambda user_id: len(anchor_lookup.get(str(user_id), []))
    )
    comment_base["assigned_anchor_available"] = comment_base["assigned_anchor_count"] > 0
    comment_base["nearest_pool_description"] = "original_human_nonempty_docs"
    comment_base["bge_metric_grain"] = "user_doc_embedding_repeated_for_comments"

    tfidf_df = _compute_tfidf_metrics(comment_base, human_df, anchor_lookup)
    bge_user_df = _compute_user_bge_metrics(combined, embeddings, human_idx, original_bot_idx, anchor_lookup)
    style_df = _compute_style_metrics(comment_base, human_df, anchor_lookup)
    detector_df = _load_or_train_detection_probabilities(combined, docs, labels, embeddings)

    comment_df = pd.concat([comment_base.reset_index(drop=True), tfidf_df, style_df], axis=1)
    comment_df = comment_df.merge(bge_user_df, left_on="bot_user_id", right_on="user_id", how="left")
    comment_df = comment_df.drop(columns=["user_id"])
    comment_df = comment_df.merge(detector_df, left_on="bot_user_id", right_on="user_id", how="left")
    comment_df = comment_df.drop(columns=["user_id"])

    family_df = _build_family_summary(comment_df, audit_df)
    detection_df = _build_detection_summary(comment_df)

    save_table(comment_df, COMMENT_PATH)
    save_table(family_df, FAMILY_PATH)
    save_table(detection_df, DETECTION_PATH)

    _plot_anchor_similarity(comment_df)
    _plot_similarity_vs_probability(comment_df)
    _plot_style_distance(comment_df)

    return audit_df, comment_df, family_df, detection_df


def main() -> None:
    research_cli(globals(), "text_evaluation")
    audit_df, comment_df, family_df, detection_df = run_anchor_similarity()
    print(f"Wrote {AUDIT_PATH} ({len(audit_df)} rows)")
    print(f"Wrote {COMMENT_PATH} ({len(comment_df)} rows)")
    print(f"Wrote {FAMILY_PATH} ({len(family_df)} rows)")
    print(f"Wrote {DETECTION_PATH} ({len(detection_df)} rows)")
    print(f"Wrote {ANCHOR_BOXPLOT_PATH}")
    print(f"Wrote {SIMILARITY_PROB_PATH}")
    print(f"Wrote {STYLE_BOXPLOT_PATH}")


if __name__ == "__main__":
    main()
