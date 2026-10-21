#!/usr/bin/env python3
"""Run the formal Leave-One-Family-Out static text benchmark."""

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

from common_static_text import (
    FAMILIES,
    SHORT_NAME,
    FIGURES_DIR,
    RESULTS_DIR,
    compute_binary_metrics,
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


OUTPUT_DIR = RESULTS_DIR / "lofo_formal"
METRICS_PATH = OUTPUT_DIR / "lofo_formal_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "lofo_formal_predictions.csv"
MATRIX_PATH = OUTPUT_DIR / "lofo_formal_matrix.csv"
RECALL_HEATMAP_PATH = FIGURES_DIR / "lofo_formal_recall_heatmap.png"
PROBABILITY_BOXPLOT_PATH = FIGURES_DIR / "lofo_formal_probability_boxplot.png"

MODEL_ORDER = ["TF-IDF+LR", "BGE+LR"]
FAMILY_SHORT_ORDER = [SHORT_NAME[family] for family in FAMILIES]


def _make_prediction_rows(
    combined: pd.DataFrame,
    idx: np.ndarray,
    labels: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
    model_name: str,
    heldout_family: str,
    train_families: list[str],
    eval_set: str,
) -> list[dict[str, object]]:
    """Create one CSV row per evaluated sample."""
    train_families_short = short_family_list(train_families)
    rows: list[dict[str, object]] = []
    for sample_idx, y_pred, y_prob in zip(idx, pred, prob, strict=True):
        sample = combined.loc[int(sample_idx)]
        y_true = int(labels[int(sample_idx)])
        rows.append(
            {
                "model": model_name,
                "heldout_family": heldout_family,
                "heldout_family_short": SHORT_NAME[heldout_family],
                "train_families": train_families_short,
                "eval_set": eval_set,
                "sample_index": int(sample_idx),
                "row": int(sample["row"]) if not pd.isna(sample["row"]) else int(sample_idx),
                "source": sample["source"],
                "family": sample["family"],
                "label": y_true,
                "prediction": int(y_pred),
                "bot_probability": float(y_prob),
                "correct": int(y_true == int(y_pred)),
            }
        )
    return rows


def _fit_and_predict(
    model_name: str,
    docs: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    train_idx: np.ndarray,
    heldout_idx: np.ndarray,
    original_test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model_name == "TF-IDF+LR":
        model = train_tfidf_lr(docs, labels, train_idx)
        heldout_pred, heldout_prob = predict_tfidf_lr(model, docs, heldout_idx)
        original_pred, original_prob = predict_tfidf_lr(model, docs, original_test_idx)
    elif model_name == "BGE+LR":
        model = train_bge_lr(embeddings, labels, train_idx)
        heldout_pred, heldout_prob = predict_bge_lr(model, embeddings, heldout_idx)
        original_pred, original_prob = predict_bge_lr(model, embeddings, original_test_idx)
    else:
        raise ValueError(f"unknown model: {model_name}")
    return heldout_pred, heldout_prob, original_pred, original_prob


def _build_matrix(metrics_df: pd.DataFrame) -> pd.DataFrame:
    matrix_frames: list[pd.DataFrame] = []
    matrix_metrics = [
        "heldout_bot_recall",
        "heldout_mean_bot_probability",
        "original_test_accuracy",
        "original_test_bot_recall",
        "original_test_human_recall",
        "original_test_macro_f1",
    ]
    for metric in matrix_metrics:
        pivot = metrics_df.pivot(
            index="model",
            columns="heldout_family_short",
            values=metric,
        ).reindex(index=MODEL_ORDER, columns=FAMILY_SHORT_ORDER)
        pivot = pivot.reset_index()
        pivot.insert(0, "metric", metric)
        matrix_frames.append(pivot)
    return pd.concat(matrix_frames, ignore_index=True)


def _plot_recall_heatmap(metrics_df: pd.DataFrame) -> None:
    recall = metrics_df.pivot(
        index="model",
        columns="heldout_family_short",
        values="heldout_bot_recall",
    ).reindex(index=MODEL_ORDER, columns=FAMILY_SHORT_ORDER)

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    image = ax.imshow(recall.to_numpy(), vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(FAMILY_SHORT_ORDER)), labels=FAMILY_SHORT_ORDER)
    ax.set_yticks(np.arange(len(MODEL_ORDER)), labels=MODEL_ORDER)
    ax.set_xlabel("Held-out family")
    ax.set_title("Formal LOFO held-out bot recall")

    for row_idx, model_name in enumerate(MODEL_ORDER):
        for col_idx, family_short in enumerate(FAMILY_SHORT_ORDER):
            value = recall.loc[model_name, family_short]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.55 else "black",
                fontsize=10,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Recall")
    fig.tight_layout()
    save_figure(fig, RECALL_HEATMAP_PATH)


def _plot_probability_boxplot(predictions_df: pd.DataFrame) -> None:
    heldout_predictions = predictions_df[predictions_df["eval_set"] == "heldout_family"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, model_name in zip(axes, MODEL_ORDER, strict=True):
        model_predictions = heldout_predictions[heldout_predictions["model"] == model_name]
        data = [
            model_predictions.loc[
                model_predictions["heldout_family_short"] == family_short,
                "bot_probability",
            ].to_numpy()
            for family_short in FAMILY_SHORT_ORDER
        ]
        ax.boxplot(
            data,
            tick_labels=FAMILY_SHORT_ORDER,
            showmeans=True,
            patch_artist=True,
            boxprops={"facecolor": "#d9e8f5", "edgecolor": "#3f6178"},
            medianprops={"color": "#9a3d3d", "linewidth": 1.4},
            meanprops={"marker": "o", "markerfacecolor": "#1f7a55", "markeredgecolor": "#1f7a55"},
            whiskerprops={"color": "#3f6178"},
            capprops={"color": "#3f6178"},
        )
        ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_title(model_name)
        ax.set_xlabel("Held-out family")
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.7)

    axes[0].set_ylabel("Predicted bot probability")
    fig.suptitle("Formal LOFO held-out bot probability distributions", y=1.02)
    fig.tight_layout()
    save_figure(fig, PROBABILITY_BOXPLOT_PATH)


def run_lofo_formal() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined, docs, labels, embeddings = load_combined_data()
    original_train_idx, _original_val_idx, original_test_idx = get_original_train_val_test_indices(
        combined,
        labels,
    )
    family_indices = get_family_indices(combined)

    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for heldout_family in FAMILIES:
        train_families = [family for family in FAMILIES if family != heldout_family]
        train_idx = family_train_indices(original_train_idx, family_indices, train_families)
        heldout_idx = family_indices[heldout_family]

        if set(heldout_idx).intersection(set(train_idx)):
            raise AssertionError(f"held-out sample leakage detected for {heldout_family}")
        train_family_values = set(combined.loc[train_idx, "family"].dropna().astype(str))
        if heldout_family in train_family_values:
            raise AssertionError(f"held-out family appears in training family column: {heldout_family}")
        if not np.all(labels[heldout_idx] == 1):
            raise AssertionError(f"held-out family contains non-bot labels: {heldout_family}")

        for model_name in MODEL_ORDER:
            heldout_pred, heldout_prob, original_pred, original_prob = _fit_and_predict(
                model_name=model_name,
                docs=docs,
                labels=labels,
                embeddings=embeddings,
                train_idx=train_idx,
                heldout_idx=heldout_idx,
                original_test_idx=original_test_idx,
            )

            original_metrics = compute_binary_metrics(
                labels[original_test_idx],
                original_pred,
                original_prob,
            )
            heldout_bot_recall = float(np.mean(heldout_pred == 1))
            heldout_mean_prob = float(np.mean(heldout_prob))

            metrics_rows.append(
                {
                    "model": model_name,
                    "heldout_family": heldout_family,
                    "heldout_family_short": SHORT_NAME[heldout_family],
                    "train_families": short_family_list(train_families),
                    "heldout_bot_recall": heldout_bot_recall,
                    "heldout_mean_bot_probability": heldout_mean_prob,
                    "original_test_accuracy": original_metrics["accuracy"],
                    "original_test_bot_recall": original_metrics["bot_recall"],
                    "original_test_human_recall": original_metrics["human_recall"],
                    "original_test_macro_f1": original_metrics["macro_f1"],
                }
            )
            prediction_rows.extend(
                _make_prediction_rows(
                    combined=combined,
                    idx=heldout_idx,
                    labels=labels,
                    pred=heldout_pred,
                    prob=heldout_prob,
                    model_name=model_name,
                    heldout_family=heldout_family,
                    train_families=train_families,
                    eval_set="heldout_family",
                )
            )
            prediction_rows.extend(
                _make_prediction_rows(
                    combined=combined,
                    idx=original_test_idx,
                    labels=labels,
                    pred=original_pred,
                    prob=original_prob,
                    model_name=model_name,
                    heldout_family=heldout_family,
                    train_families=train_families,
                    eval_set="original_test",
                )
            )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    matrix_df = _build_matrix(metrics_df)

    save_table(metrics_df, METRICS_PATH)
    save_table(predictions_df, PREDICTIONS_PATH)
    save_table(matrix_df, MATRIX_PATH)
    _plot_recall_heatmap(metrics_df)
    _plot_probability_boxplot(predictions_df)

    return metrics_df, predictions_df, matrix_df


def validate_outputs(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
) -> None:
    expected_metric_columns = [
        "model",
        "heldout_family",
        "heldout_family_short",
        "train_families",
        "heldout_bot_recall",
        "heldout_mean_bot_probability",
        "original_test_accuracy",
        "original_test_bot_recall",
        "original_test_human_recall",
        "original_test_macro_f1",
    ]
    missing = [column for column in expected_metric_columns if column not in metrics_df.columns]
    if missing:
        raise AssertionError(f"missing metric columns: {missing}")

    for model_name in MODEL_ORDER:
        model_metrics = metrics_df[metrics_df["model"] == model_name]
        if len(model_metrics) != len(FAMILIES):
            raise AssertionError(f"{model_name} has {len(model_metrics)} held-out rows")
        if set(model_metrics["heldout_family"]) != set(FAMILIES):
            raise AssertionError(f"{model_name} does not cover all held-out families")

    for row in metrics_df.itertuples(index=False):
        heldout_short = SHORT_NAME[row.heldout_family]
        train_shorts = str(row.train_families).split("+")
        if heldout_short in train_shorts:
            raise AssertionError(f"held-out family leaked into train_families: {row}")

    expected_eval_sets = {"heldout_family", "original_test"}
    if set(predictions_df["eval_set"]) != expected_eval_sets:
        raise AssertionError("predictions do not contain both heldout_family and original_test eval sets")
    if matrix_df.empty:
        raise AssertionError("matrix output is empty")
    for path in [
        METRICS_PATH,
        PREDICTIONS_PATH,
        MATRIX_PATH,
        RECALL_HEATMAP_PATH,
        PROBABILITY_BOXPLOT_PATH,
    ]:
        if not Path(path).exists():
            raise AssertionError(f"missing output: {path}")


def main() -> None:
    research_cli(globals(), "text_evaluation")
    metrics_df, predictions_df, matrix_df = run_lofo_formal()
    validate_outputs(metrics_df, predictions_df, matrix_df)

    print("Formal LOFO completed.")
    print(f"metrics: {METRICS_PATH}")
    print(f"predictions: {PREDICTIONS_PATH}")
    print(f"matrix: {MATRIX_PATH}")
    print(f"recall heatmap: {RECALL_HEATMAP_PATH}")
    print(f"probability boxplot: {PROBABILITY_BOXPLOT_PATH}")
    print()
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
