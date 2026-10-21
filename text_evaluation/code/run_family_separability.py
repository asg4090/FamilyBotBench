#!/usr/bin/env python3
"""Run static-text family separability experiments for F1-F5 bot texts."""

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
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from common_static_text import (
    FAMILIES,
    FIGURES_DIR,
    RESULTS_DIR,
    SHORT_NAME,
    load_combined_data,
    load_generated_tables,
    save_figure,
    save_table,
)


SEED = 20260602
RESULT_DIR = RESULTS_DIR / "family_separability"
FIGURE_PREFIX = FIGURES_DIR


@dataclass(frozen=True)
class FamilySplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_within_family_split(combined: pd.DataFrame, seed: int = SEED) -> FamilySplit:
    """Make a deterministic 70/15/15 split inside each family."""
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for family in FAMILIES:
        idx = combined.index[combined["family"] == family].to_numpy()
        if len(idx) < 4:
            raise ValueError(f"family {family} has too few samples for 70/15/15 split: {len(idx)}")
        train_idx, temp_idx = train_test_split(idx, train_size=0.70, random_state=seed, shuffle=True)
        val_idx, test_idx = train_test_split(temp_idx, train_size=0.50, random_state=seed, shuffle=True)
        train_parts.append(np.sort(train_idx))
        val_parts.append(np.sort(val_idx))
        test_parts.append(np.sort(test_idx))

    return FamilySplit(
        train_idx=np.concatenate(train_parts),
        val_idx=np.concatenate(val_parts),
        test_idx=np.concatenate(test_parts),
    )


def labels_for(combined: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    return combined.loc[idx, "family"].to_numpy(dtype=str)


def make_tfidf_matrix(docs: list[str], train_idx: np.ndarray, eval_idx: np.ndarray):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform([docs[i] for i in train_idx])
    x_eval = vectorizer.transform([docs[i] for i in eval_idx])
    return x_train, x_eval


def make_lr(mode: str):
    if mode == "multinomial":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )
    if mode == "ovr":
        return OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=SEED,
            )
        )
    raise ValueError(f"unknown LR mode: {mode}")


def fit_predict_model(
    feature: str,
    mode: str,
    docs: list[str],
    embeddings: np.ndarray,
    combined: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
):
    y_train = labels_for(combined, train_idx)

    if feature == "tfidf":
        x_train, x_eval = make_tfidf_matrix(docs, train_idx, eval_idx)
    elif feature == "bge":
        x_train = embeddings[train_idx]
        x_eval = embeddings[eval_idx]
    else:
        raise ValueError(f"unknown feature: {feature}")

    clf = make_lr(mode)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)
    prob = clf.predict_proba(x_eval)
    return clf, y_pred, prob


def metric_rows(
    protocol: str,
    feature: str,
    mode: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_size: int,
    eval_size: int,
    status: str = "ok",
) -> list[dict[str, object]]:
    model = f"{feature}_{mode}_lr"
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "protocol": protocol,
            "model": model,
            "feature": feature,
            "lr_mode": mode,
            "split": split_name,
            "status": status,
            "family": "ALL",
            "family_short": "ALL",
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=FAMILIES, average="macro", zero_division=0)),
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "support": int(len(y_true)),
            "train_size": int(train_size),
            "eval_size": int(eval_size),
            "note": "",
        }
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=FAMILIES,
        zero_division=0,
    )
    for family, p_val, r_val, f_val, s_val in zip(FAMILIES, precision, recall, f1, support):
        rows.append(
            {
                "protocol": protocol,
                "model": model,
                "feature": feature,
                "lr_mode": mode,
                "split": split_name,
                "status": status,
                "family": family,
                "family_short": SHORT_NAME[family],
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "precision": float(p_val),
                "recall": float(r_val),
                "f1": float(f_val),
                "support": int(s_val),
                "train_size": int(train_size),
                "eval_size": int(eval_size),
                "note": "",
            }
        )
    return rows


def prediction_rows(
    protocol: str,
    feature: str,
    mode: str,
    split_name: str,
    combined: pd.DataFrame,
    idx: np.ndarray,
    y_pred: np.ndarray,
    prob: np.ndarray,
    classes: np.ndarray,
) -> list[dict[str, object]]:
    prob_by_class = {class_name: prob[:, pos] for pos, class_name in enumerate(classes)}
    rows: list[dict[str, object]] = []
    for pos, sample_idx in enumerate(idx):
        true_family = str(combined.at[sample_idx, "family"])
        pred_family = str(y_pred[pos])
        row: dict[str, object] = {
            "protocol": protocol,
            "model": f"{feature}_{mode}_lr",
            "feature": feature,
            "lr_mode": mode,
            "split": split_name,
            "sample_index": int(sample_idx),
            "row": int(combined.at[sample_idx, "row"]),
            "user_id": combined.at[sample_idx, "user_id"],
            "family_true": true_family,
            "family_true_short": SHORT_NAME.get(true_family, true_family),
            "family_pred": pred_family,
            "family_pred_short": SHORT_NAME.get(pred_family, pred_family),
            "correct": bool(true_family == pred_family),
        }
        for family in FAMILIES:
            values = prob_by_class.get(family)
            row[f"prob_{SHORT_NAME[family]}"] = float(values[pos]) if values is not None else np.nan
        rows.append(row)
    return rows


def save_confusion_outputs(
    combined: pd.DataFrame,
    idx: np.ndarray,
    y_pred: np.ndarray,
    feature: str,
    mode: str,
) -> pd.DataFrame:
    y_true = labels_for(combined, idx)
    cm = confusion_matrix(y_true, y_pred, labels=FAMILIES)
    matrix = pd.DataFrame(cm, index=[SHORT_NAME[f] for f in FAMILIES], columns=[SHORT_NAME[f] for f in FAMILIES])
    matrix.index.name = "true_family"
    save_table(matrix.reset_index(), RESULT_DIR / f"family_confusion_matrix_{feature}.csv")

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{feature.upper()} {mode} LR family confusion matrix")
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.set_xticks(np.arange(len(FAMILIES)), labels=[SHORT_NAME[f] for f in FAMILIES])
    ax.set_yticks(np.arange(len(FAMILIES)), labels=[SHORT_NAME[f] for f in FAMILIES])
    max_value = int(cm.max()) if cm.size else 0
    threshold = max_value / 2 if max_value else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, FIGURE_PREFIX / f"family_confusion_matrix_{feature}.png")
    return matrix


def detect_round_split(combined: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Return cross-round train/test indices if reliable round metadata exists."""
    family_idx = combined.index[combined["family"].isin(FAMILIES)].to_numpy()
    candidates: list[tuple[str, pd.Series]] = []

    for col in combined.columns:
        lower = col.lower()
        if "family_round" in lower or "family_generation" in lower:
            values = combined.loc[family_idx, col]
            if values.notna().all():
                candidates.append((f"combined.{col}", values))

    profiles, _comments, _anchors = load_generated_tables()
    for col in ["family_round", "family_generation"]:
        if col not in profiles.columns:
            continue
        mapping = profiles.drop_duplicates("user_id").set_index("user_id")[col]
        values = combined.loc[family_idx, "user_id"].map(mapping)
        if values.notna().all():
            candidates.append((f"generated_profiles_all.{col}", values))

    if not candidates:
        return None, None, "round_metadata_missing: no complete family_round/family_generation mapping"

    skip_notes: list[str] = []
    for source_note, round_series in candidates:
        numeric_rounds = pd.to_numeric(round_series, errors="coerce")
        if numeric_rounds.isna().any():
            skip_notes.append(f"{source_note} has nonnumeric values")
            continue
        unique_rounds = sorted(numeric_rounds.unique().tolist())
        if len(unique_rounds) < 2:
            skip_notes.append(f"{source_note} has only one unique value {unique_rounds}")
            continue

        last_round = unique_rounds[-1]
        train_idx = family_idx[numeric_rounds.to_numpy() < last_round]
        test_idx = family_idx[numeric_rounds.to_numpy() == last_round]
        train_families = set(combined.loc[train_idx, "family"])
        test_families = set(combined.loc[test_idx, "family"])
        if train_families != set(FAMILIES) or test_families != set(FAMILIES):
            skip_notes.append(f"{source_note} does not cover all families in train/test")
            continue

        return train_idx, test_idx, f"round source: {source_note}; train rounds < {last_round}; test round = {last_round}"

    return None, None, "round_metadata_missing: " + "; ".join(skip_notes)


def add_round_metrics_or_skip(
    combined: pd.DataFrame,
    docs: list[str],
    embeddings: np.ndarray,
    metrics: list[dict[str, object]],
    predictions: list[dict[str, object]],
) -> None:
    train_idx, test_idx, note = detect_round_split(combined)
    if train_idx is None or test_idx is None:
        metrics.append(
            {
                "protocol": "D2_cross_round",
                "model": "ALL",
                "feature": "ALL",
                "lr_mode": "ALL",
                "split": "test",
                "status": "round_metadata_missing",
                "family": "ALL",
                "family_short": "ALL",
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "support": 0,
                "train_size": 0,
                "eval_size": 0,
                "note": note,
            }
        )
        return

    for feature in ["tfidf", "bge"]:
        for mode in ["multinomial", "ovr"]:
            clf, y_pred, prob = fit_predict_model(feature, mode, docs, embeddings, combined, train_idx, test_idx)
            y_true = labels_for(combined, test_idx)
            metrics.extend(
                metric_rows(
                    protocol="D2_cross_round",
                    feature=feature,
                    mode=mode,
                    split_name="test",
                    y_true=y_true,
                    y_pred=y_pred,
                    train_size=len(train_idx),
                    eval_size=len(test_idx),
                )
            )
            for row in metrics[-6:]:
                row["note"] = note
            predictions.extend(
                prediction_rows("D2_cross_round", feature, mode, "test", combined, test_idx, y_pred, prob, clf.classes_)
            )


def save_embedding_plot(combined: pd.DataFrame, embeddings: np.ndarray) -> Path:
    family_idx = combined.index[combined["family"].isin(FAMILIES)].to_numpy()
    x = StandardScaler().fit_transform(embeddings[family_idx])
    method = "umap"
    try:
        import umap  # type: ignore

        coords = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.12).fit_transform(x)
    except Exception:
        method = "pca"
        coords = PCA(n_components=2, random_state=SEED).fit_transform(x)

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    palette = {
        "F1": "#1f77b4",
        "F2": "#ff7f0e",
        "F3": "#2ca02c",
        "F4": "#d62728",
        "F5": "#9467bd",
    }
    families = combined.loc[family_idx, "family"].to_numpy(dtype=str)
    for family in FAMILIES:
        mask = families == family
        short = SHORT_NAME[family]
        ax.scatter(coords[mask, 0], coords[mask, 1], s=32, alpha=0.82, label=short, color=palette[short], edgecolors="none")
    ax.set_title(f"BGE embedding family separability ({method.upper()})")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend(title="Family", frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    ax.grid(alpha=0.20)
    fig.tight_layout()
    path = FIGURE_PREFIX / f"family_separability_embedding_{method}.png"
    save_figure(fig, path)
    return path


def main() -> None:
    research_cli(globals(), "text_evaluation")
    combined, docs, _binary_labels, embeddings = load_combined_data()
    family_mask = combined["family"].isin(FAMILIES)
    if int(family_mask.sum()) != 250:
        raise ValueError(f"expected 250 F1-F5 bot samples, got {int(family_mask.sum())}")

    split = make_within_family_split(combined)
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    primary_predictions: dict[str, np.ndarray] = {}

    for feature in ["tfidf", "bge"]:
        for mode in ["multinomial", "ovr"]:
            for split_name, eval_idx in [("val", split.val_idx), ("test", split.test_idx)]:
                clf, y_pred, prob = fit_predict_model(
                    feature=feature,
                    mode=mode,
                    docs=docs,
                    embeddings=embeddings,
                    combined=combined,
                    train_idx=split.train_idx,
                    eval_idx=eval_idx,
                )
                y_true = labels_for(combined, eval_idx)
                metrics.extend(
                    metric_rows(
                        protocol="D1_within_family_stratified",
                        feature=feature,
                        mode=mode,
                        split_name=split_name,
                        y_true=y_true,
                        y_pred=y_pred,
                        train_size=len(split.train_idx),
                        eval_size=len(eval_idx),
                    )
                )
                predictions.extend(
                    prediction_rows(
                        protocol="D1_within_family_stratified",
                        feature=feature,
                        mode=mode,
                        split_name=split_name,
                        combined=combined,
                        idx=eval_idx,
                        y_pred=y_pred,
                        prob=prob,
                        classes=clf.classes_,
                    )
                )
                if split_name == "test" and mode == "multinomial":
                    primary_predictions[feature] = y_pred

    for feature, y_pred in primary_predictions.items():
        save_confusion_outputs(combined, split.test_idx, y_pred, feature, "multinomial")

    add_round_metrics_or_skip(combined, docs, embeddings, metrics, predictions)
    embedding_path = save_embedding_plot(combined, embeddings)

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.DataFrame(predictions)
    save_table(metrics_df, RESULT_DIR / "family_separability_metrics.csv")
    save_table(predictions_df, RESULT_DIR / "family_separability_predictions.csv")

    print(f"saved metrics: {RESULT_DIR / 'family_separability_metrics.csv'}")
    print(f"saved predictions: {RESULT_DIR / 'family_separability_predictions.csv'}")
    print(f"saved embedding figure: {embedding_path}")


if __name__ == "__main__":
    main()
