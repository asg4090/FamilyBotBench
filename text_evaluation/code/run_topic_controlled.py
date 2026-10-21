#!/usr/bin/env python3
"""Topic-controlled static text detection experiments."""

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
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from common_static_text import (
    FIGURES_DIR,
    FAMILIES,
    RESULTS_DIR,
    SHORT_NAME,
    compute_binary_metrics,
    get_family_indices,
    get_original_train_val_test_indices,
    load_combined_data,
    predict_bge_lr,
    predict_tfidf_lr,
    save_figure,
    save_table,
    train_bge_lr,
    train_tfidf_lr,
)


SEED = 20260531
K_VALUES = (8, 12, 16)
MAIN_K = 12
TOP_N_TERMS = 16
MIN_WITHIN_TOPIC_CLASS_COUNT = 10
MODEL_NAMES = ("tfidf_lr", "bge_lr")

TOPIC_RESULTS_DIR = RESULTS_DIR / "topic_controlled"


@dataclass(frozen=True)
class SplitBundle:
    original_train_idx: np.ndarray
    original_val_idx: np.ndarray
    original_test_idx: np.ndarray
    family_train_idx: np.ndarray
    family_test_idx: np.ndarray

    @property
    def train_idx(self) -> np.ndarray:
        return np.sort(np.concatenate([self.original_train_idx, self.family_train_idx]))

    @property
    def test_idx(self) -> np.ndarray:
        return np.sort(np.concatenate([self.original_test_idx, self.family_test_idx]))


def compact_text(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", "" if pd.isna(text) else str(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def make_splits(combined: pd.DataFrame, labels: np.ndarray) -> SplitBundle:
    original_train_idx, original_val_idx, original_test_idx = get_original_train_val_test_indices(
        combined,
        labels,
        seed=SEED,
    )
    family_indices = get_family_indices(combined)
    family_train_parts: list[np.ndarray] = []
    family_test_parts: list[np.ndarray] = []
    for family_pos, family in enumerate(FAMILIES):
        idx = np.sort(family_indices[family])
        rng = np.random.default_rng(SEED + 101 * (family_pos + 1))
        shuffled = rng.permutation(idx)
        n_train = int(round(len(shuffled) * 0.70))
        family_train_parts.append(np.sort(shuffled[:n_train]))
        family_test_parts.append(np.sort(shuffled[n_train:]))
    return SplitBundle(
        original_train_idx=np.sort(original_train_idx),
        original_val_idx=np.sort(original_val_idx),
        original_test_idx=np.sort(original_test_idx),
        family_train_idx=np.sort(np.concatenate(family_train_parts)),
        family_test_idx=np.sort(np.concatenate(family_test_parts)),
    )


def family_count_columns(combined: pd.DataFrame, idx: np.ndarray) -> dict[str, int]:
    family_values = combined.iloc[idx]["family"]
    counts = {
        f"family_count_{SHORT_NAME[family]}": int((family_values == family).sum())
        for family in FAMILIES
    }
    counts["original_bot_count"] = int((family_values == "original_bot").sum())
    return counts


def topic_count_record(
    combined: pd.DataFrame,
    labels: np.ndarray,
    topic_labels: np.ndarray,
    k: int,
    topic_id: int,
) -> dict[str, int]:
    idx = np.flatnonzero(topic_labels == topic_id)
    record: dict[str, int] = {
        "k": int(k),
        "topic_id": int(topic_id),
        "topic_total_count": int(len(idx)),
        "human_count": int((labels[idx] == 0).sum()),
        "bot_count": int((labels[idx] == 1).sum()),
    }
    record.update(family_count_columns(combined, idx))
    return record


def fit_topics(embeddings: np.ndarray) -> dict[int, np.ndarray]:
    topic_by_k: dict[int, np.ndarray] = {}
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=SEED, n_init=30)
        topic_by_k[k] = model.fit_predict(embeddings)
    return topic_by_k


def build_topic_assignments(combined: pd.DataFrame, topic_by_k: dict[int, np.ndarray]) -> pd.DataFrame:
    out = combined.copy()
    out["family_short"] = out["family"].map(SHORT_NAME).fillna(out["family"])
    for k, topic_labels in topic_by_k.items():
        out[f"topic_k{k}"] = topic_labels.astype(int)
    return out


def build_topic_keyword_summary(
    combined: pd.DataFrame,
    docs: list[str],
    labels: np.ndarray,
    topic_by_k: dict[int, np.ndarray],
) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )
    x_tfidf = vectorizer.fit_transform(docs)
    terms = np.asarray(vectorizer.get_feature_names_out())

    rows: list[dict[str, object]] = []
    for k, topic_labels in topic_by_k.items():
        for topic_id in range(k):
            idx = np.flatnonzero(topic_labels == topic_id)
            row: dict[str, object] = topic_count_record(combined, labels, topic_labels, k, topic_id)
            if len(idx) == 0:
                row.update(
                    {
                        "top_tfidf_terms": "",
                        "example_human_text": "",
                        "example_bot_text": "",
                    }
                )
                rows.append(row)
                continue
            weights = np.asarray(x_tfidf[idx].mean(axis=0)).ravel()
            top_pos = np.argsort(weights)[::-1][:TOP_N_TERMS]
            top_terms = [terms[pos] for pos in top_pos if weights[pos] > 0]
            if not top_terms:
                top_terms = ["no_nonempty_terms"]
            human_idx = [i for i in idx if labels[i] == 0 and docs[i].strip()]
            bot_idx = [i for i in idx if labels[i] == 1 and docs[i].strip()]
            row.update(
                {
                    "top_tfidf_terms": "; ".join(top_terms),
                    "example_human_text": compact_text(docs[human_idx[0]]) if human_idx else "",
                    "example_bot_text": compact_text(docs[bot_idx[0]]) if bot_idx else "",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def train_and_predict(
    model_name: str,
    docs: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, str]:
    if len(train_idx) == 0 or len(test_idx) == 0:
        return {}, np.array([], dtype=int), np.array([], dtype=float), "skipped_empty_split"
    if len(np.unique(labels[train_idx])) < 2:
        return {}, np.array([], dtype=int), np.array([], dtype=float), "skipped_one_class_train"
    try:
        if model_name == "tfidf_lr":
            model = train_tfidf_lr(docs, labels, train_idx)
            pred, prob = predict_tfidf_lr(model, docs, test_idx)
        elif model_name == "bge_lr":
            model = train_bge_lr(embeddings, labels, train_idx)
            pred, prob = predict_bge_lr(model, embeddings, test_idx)
        else:
            raise ValueError(f"unknown model: {model_name}")
    except ValueError as exc:
        return {}, np.array([], dtype=int), np.array([], dtype=float), f"skipped_model_error: {exc}"
    metrics = compute_binary_metrics(labels[test_idx], pred, prob)
    return metrics, pred, prob, "ok"


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def make_topic_matched_pool(topic_labels: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, pd.DataFrame]:
    selected_parts: list[np.ndarray] = []
    rows: list[dict[str, int]] = []
    for topic_id in range(k):
        idx = np.flatnonzero(topic_labels == topic_id)
        human_idx = idx[labels[idx] == 0]
        bot_idx = idx[labels[idx] == 1]
        keep = min(len(human_idx), len(bot_idx))
        if keep > 0:
            rng = np.random.default_rng(SEED + 1000 * k + topic_id)
            selected_parts.append(rng.choice(human_idx, keep, replace=False))
            selected_parts.append(rng.choice(bot_idx, keep, replace=False))
        rows.append(
            {
                "k": int(k),
                "topic_id": int(topic_id),
                "topic_human_count": int(len(human_idx)),
                "topic_bot_count": int(len(bot_idx)),
                "matched_keep_per_class": int(keep),
                "matched_keep_total": int(2 * keep),
            }
        )
    if selected_parts:
        selected = np.sort(np.concatenate(selected_parts))
    else:
        selected = np.array([], dtype=int)
    return selected, pd.DataFrame(rows)


def run_topic_matched_detection(
    docs: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    topic_by_k: dict[int, np.ndarray],
    splits: SplitBundle,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    global_train_idx = splits.train_idx
    global_test_idx = splits.test_idx
    for k, topic_labels in topic_by_k.items():
        matched_pool, match_details = make_topic_matched_pool(topic_labels, labels, k)
        matched_train_idx = np.intersect1d(global_train_idx, matched_pool, assume_unique=False)
        matched_test_idx = np.intersect1d(global_test_idx, matched_pool, assume_unique=False)
        matched_labels = labels[matched_pool]
        for model_name in MODEL_NAMES:
            baseline_metrics, _, _, baseline_status = train_and_predict(
                model_name,
                docs,
                labels,
                embeddings,
                global_train_idx,
                global_test_idx,
            )
            matched_uncontrolled_metrics, _, _, matched_uncontrolled_status = train_and_predict(
                model_name,
                docs,
                labels,
                embeddings,
                global_train_idx,
                matched_test_idx,
            )
            controlled_metrics, _, _, controlled_status = train_and_predict(
                model_name,
                docs,
                labels,
                embeddings,
                matched_train_idx,
                matched_test_idx,
            )
            row: dict[str, object] = {
                "k": int(k),
                "model": model_name,
                "status": controlled_status,
                "baseline_status": baseline_status,
                "matched_uncontrolled_status": matched_uncontrolled_status,
                "baseline_train_count": int(len(global_train_idx)),
                "baseline_test_count": int(len(global_test_idx)),
                "matched_total_retained": int(len(matched_pool)),
                "matched_human_count": int((matched_labels == 0).sum()) if len(matched_labels) else 0,
                "matched_bot_count": int((matched_labels == 1).sum()) if len(matched_labels) else 0,
                "matched_train_count": int(len(matched_train_idx)),
                "matched_test_count": int(len(matched_test_idx)),
                "matched_train_human_count": int((labels[matched_train_idx] == 0).sum()) if len(matched_train_idx) else 0,
                "matched_train_bot_count": int((labels[matched_train_idx] == 1).sum()) if len(matched_train_idx) else 0,
                "matched_test_human_count": int((labels[matched_test_idx] == 0).sum()) if len(matched_test_idx) else 0,
                "matched_test_bot_count": int((labels[matched_test_idx] == 1).sum()) if len(matched_test_idx) else 0,
                "matched_nonempty_topics": int((match_details["matched_keep_total"] > 0).sum()),
                "matched_empty_topics": int((match_details["matched_keep_total"] == 0).sum()),
            }
            row.update(prefixed("baseline_global_test", baseline_metrics))
            row.update(prefixed("uncontrolled_on_matched_test", matched_uncontrolled_metrics))
            row.update(prefixed("controlled_matched_test", controlled_metrics))
            baseline_recall = baseline_metrics.get("bot_recall", np.nan)
            uncontrolled_matched_recall = matched_uncontrolled_metrics.get("bot_recall", np.nan)
            controlled_recall = controlled_metrics.get("bot_recall", np.nan)
            row["bot_recall_drop_vs_global"] = float(baseline_recall - controlled_recall)
            row["bot_recall_drop_vs_uncontrolled_matched_test"] = float(
                uncontrolled_matched_recall - controlled_recall
            )
            rows.append(row)
    return pd.DataFrame(rows)


def split_within_topic(idx: np.ndarray, labels: np.ndarray, topic_id: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=SEED + topic_id,
        stratify=labels[idx],
    )
    return np.sort(train_idx), np.sort(test_idx)


def run_within_topic_detection(
    combined: pd.DataFrame,
    docs: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    topic_by_k: dict[int, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for k, topic_labels in topic_by_k.items():
        for topic_id in range(k):
            idx = np.flatnonzero(topic_labels == topic_id)
            counts = topic_count_record(combined, labels, topic_labels, k, topic_id)
            human_count = int(counts["human_count"])
            bot_count = int(counts["bot_count"])
            skip_reason = ""
            if min(human_count, bot_count) < MIN_WITHIN_TOPIC_CLASS_COUNT:
                skip_reason = (
                    "skipped_small_topic: "
                    f"min_class_count={min(human_count, bot_count)} "
                    f"< {MIN_WITHIN_TOPIC_CLASS_COUNT}"
                )
            if skip_reason:
                for model_name in MODEL_NAMES:
                    row: dict[str, object] = {
                        **counts,
                        "model": model_name,
                        "status": "skipped_small_topic",
                        "skip_reason": skip_reason,
                        "train_count": 0,
                        "test_count": 0,
                    }
                    rows.append(row)
                continue
            train_idx, test_idx = split_within_topic(idx, labels, topic_id + 100 * k)
            for model_name in MODEL_NAMES:
                metrics, _, _, status = train_and_predict(
                    model_name,
                    docs,
                    labels,
                    embeddings,
                    train_idx,
                    test_idx,
                )
                row = {
                    **counts,
                    "model": model_name,
                    "status": status,
                    "skip_reason": "" if status == "ok" else status,
                    "train_count": int(len(train_idx)),
                    "test_count": int(len(test_idx)),
                    "train_human_count": int((labels[train_idx] == 0).sum()),
                    "train_bot_count": int((labels[train_idx] == 1).sum()),
                    "test_human_count": int((labels[test_idx] == 0).sum()),
                    "test_bot_count": int((labels[test_idx] == 1).sum()),
                }
                row.update(metrics)
                rows.append(row)
    return pd.DataFrame(rows)


def run_leave_one_topic_out(
    combined: pd.DataFrame,
    docs: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    topic_by_k: dict[int, np.ndarray],
) -> pd.DataFrame:
    all_idx = np.arange(len(labels))
    rows: list[dict[str, object]] = []
    for k, topic_labels in topic_by_k.items():
        for topic_id in range(k):
            test_idx = np.flatnonzero(topic_labels == topic_id)
            train_idx = all_idx[topic_labels != topic_id]
            counts = topic_count_record(combined, labels, topic_labels, k, topic_id)
            if int(counts["bot_count"]) == 0:
                for model_name in MODEL_NAMES:
                    row = {
                        **counts,
                        "model": model_name,
                        "status": "skipped_small_topic",
                        "skip_reason": "skipped_small_topic: no bot samples in held-out topic",
                        "train_count": int(len(train_idx)),
                        "test_count": int(len(test_idx)),
                        "train_human_count": int((labels[train_idx] == 0).sum()),
                        "train_bot_count": int((labels[train_idx] == 1).sum()),
                        "test_human_count": int((labels[test_idx] == 0).sum()),
                        "test_bot_count": int((labels[test_idx] == 1).sum()),
                    }
                    rows.append(row)
                continue
            for model_name in MODEL_NAMES:
                metrics, _, _, status = train_and_predict(
                    model_name,
                    docs,
                    labels,
                    embeddings,
                    train_idx,
                    test_idx,
                )
                row: dict[str, object] = {
                    **counts,
                    "model": model_name,
                    "status": status,
                    "skip_reason": "" if status == "ok" else status,
                    "train_count": int(len(train_idx)),
                    "test_count": int(len(test_idx)),
                    "train_human_count": int((labels[train_idx] == 0).sum()),
                    "train_bot_count": int((labels[train_idx] == 1).sum()),
                    "test_human_count": int((labels[test_idx] == 0).sum()),
                    "test_bot_count": int((labels[test_idx] == 1).sum()),
                }
                row.update(metrics)
                rows.append(row)
    return pd.DataFrame(rows)


def label_family_group(combined: pd.DataFrame) -> pd.Series:
    family_short = combined["family"].map(SHORT_NAME)
    return np.where(
        combined["label"].to_numpy(dtype=int) == 0,
        "human",
        np.where(combined["family"].eq("original_bot"), "original_bot", family_short.fillna("bot")),
    )


def plot_topic_distribution(assignments: pd.DataFrame) -> None:
    topic_col = f"topic_k{MAIN_K}"
    plot_df = assignments.copy()
    plot_df["group"] = label_family_group(plot_df)
    order = ["human", "original_bot", "F1", "F2", "F3", "F4", "F5"]
    counts = pd.crosstab(plot_df[topic_col], plot_df["group"]).reindex(columns=order, fill_value=0)
    counts = counts.sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(counts))
    colors = plt.get_cmap("tab20").colors
    for pos, group in enumerate(order):
        values = counts[group].to_numpy()
        ax.bar(counts.index.astype(str), values, bottom=bottom, label=group, color=colors[pos])
        bottom += values
    ax.set_title(f"Topic distribution by label/family (k={MAIN_K})")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Sample count")
    ax.legend(ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "topic_distribution_by_label_family.png")


def plot_topic_controlled_drop(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(K_VALUES))
    width = 0.34
    for pos, model_name in enumerate(MODEL_NAMES):
        subset = metrics[metrics["model"] == model_name].set_index("k").reindex(K_VALUES)
        offset = (pos - 0.5) * width
        ax.bar(
            x + offset,
            subset["bot_recall_drop_vs_global"].to_numpy(dtype=float),
            width=width,
            label=model_name,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("KMeans topic count")
    ax.set_ylabel("Bot recall drop")
    ax.set_title("Recall drop after topic-matched downsampling")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "topic_controlled_recall_drop.png")


def plot_within_topic_recall(metrics: pd.DataFrame) -> None:
    subset = metrics[(metrics["k"] == MAIN_K) & (metrics["status"] == "ok")].copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    topics = list(range(MAIN_K))
    x = np.arange(len(topics))
    width = 0.34
    for pos, model_name in enumerate(MODEL_NAMES):
        model_df = subset[subset["model"] == model_name].set_index("topic_id")
        values = model_df.reindex(topics)["bot_recall"].to_numpy(dtype=float)
        ax.bar(x + (pos - 0.5) * width, values, width=width, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in topics])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Bot recall")
    ax.set_title(f"Within-topic detection recall (k={MAIN_K})")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "within_topic_recall.png")


def plot_leave_one_topic_out_recall(metrics: pd.DataFrame) -> None:
    subset = metrics[(metrics["k"] == MAIN_K) & (metrics["status"] == "ok")].copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    topics = list(range(MAIN_K))
    x = np.arange(len(topics))
    width = 0.34
    for pos, model_name in enumerate(MODEL_NAMES):
        model_df = subset[subset["model"] == model_name].set_index("topic_id")
        values = model_df.reindex(topics)["bot_recall"].to_numpy(dtype=float)
        ax.bar(x + (pos - 0.5) * width, values, width=width, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in topics])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Held-out topic")
    ax.set_ylabel("Bot recall")
    ax.set_title(f"Leave-one-topic-out recall (k={MAIN_K})")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "leave_one_topic_out_recall.png")


def main() -> None:
    research_cli(globals(), "text_evaluation")
    combined, docs, labels, embeddings = load_combined_data()
    topic_by_k = fit_topics(embeddings)
    assignments = build_topic_assignments(combined, topic_by_k)
    keyword_summary = build_topic_keyword_summary(combined, docs, labels, topic_by_k)
    splits = make_splits(combined, labels)

    topic_matched_metrics = run_topic_matched_detection(docs, labels, embeddings, topic_by_k, splits)
    within_topic_metrics = run_within_topic_detection(combined, docs, labels, embeddings, topic_by_k)
    leave_one_topic_out_metrics = run_leave_one_topic_out(combined, docs, labels, embeddings, topic_by_k)

    save_table(assignments, TOPIC_RESULTS_DIR / "topic_assignments.csv")
    save_table(keyword_summary, TOPIC_RESULTS_DIR / "topic_keyword_summary.csv")
    save_table(topic_matched_metrics, TOPIC_RESULTS_DIR / "topic_matched_detection_metrics.csv")
    save_table(within_topic_metrics, TOPIC_RESULTS_DIR / "within_topic_detection_metrics.csv")
    save_table(leave_one_topic_out_metrics, TOPIC_RESULTS_DIR / "leave_one_topic_out_metrics.csv")

    plot_topic_distribution(assignments)
    plot_topic_controlled_drop(topic_matched_metrics)
    plot_within_topic_recall(within_topic_metrics)
    plot_leave_one_topic_out_recall(leave_one_topic_out_metrics)

    print(f"topic assignments: {TOPIC_RESULTS_DIR / 'topic_assignments.csv'}")
    print(f"keyword summary: {TOPIC_RESULTS_DIR / 'topic_keyword_summary.csv'}")
    print(f"topic matched metrics: {TOPIC_RESULTS_DIR / 'topic_matched_detection_metrics.csv'}")
    print(f"within-topic metrics: {TOPIC_RESULTS_DIR / 'within_topic_detection_metrics.csv'}")
    print(f"leave-one-topic-out metrics: {TOPIC_RESULTS_DIR / 'leave_one_topic_out_metrics.csv'}")
    print(f"figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
