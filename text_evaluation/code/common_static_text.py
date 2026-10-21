#!/usr/bin/env python3
"""Shared utilities for static text extended benchmark experiments."""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, recall_score


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
STATIC_RUN = PROJECT_ROOT / "static_language" / "outputs" / "full_5family_50bot_3round_gpt54_20260601"
ANALYSIS_DIR = PROJECT_ROOT / "baselines" / "text"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
apply_paths(globals(), "text_evaluation")

if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from comment_text_only_experiment import split_indices  # noqa: E402


FAMILIES = [
    "F1_original_architecture",
    "F2_tfidf_informed_language",
    "F3_human_anchor_local_tfidf",
    "F4_peba_trait_prompt",
    "F5_gemini_trait_prompt",
]

SHORT_NAME = {
    "F1_original_architecture": "F1",
    "F2_tfidf_informed_language": "F2",
    "F3_human_anchor_local_tfidf": "F3",
    "F4_peba_trait_prompt": "F4",
    "F5_gemini_trait_prompt": "F5",
}

FULL_NAME = {short: full for full, short in SHORT_NAME.items()}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_combined_data() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    combined = pd.read_csv(STATIC_RUN / "combined_sample_index.csv")
    combined["doc"] = combined["doc"].fillna("")
    docs = combined["doc"].astype(str).tolist()
    labels = combined["label"].to_numpy(dtype=int)
    embeddings = np.load(STATIC_RUN / "combined_bge_embeddings.npy")
    if len(combined) != len(embeddings):
        raise ValueError(f"row mismatch: combined={len(combined)} embeddings={len(embeddings)}")
    return combined, docs, labels, embeddings


def load_generated_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles = pd.read_csv(STATIC_RUN / "generated_profiles_all.csv")
    comments = pd.read_csv(STATIC_RUN / "generated_comments_all.csv")
    anchors = pd.read_csv(STATIC_RUN / "anchor_manifest.csv")
    return profiles, comments, anchors


def get_original_indices(combined: pd.DataFrame) -> np.ndarray:
    return combined.index[combined["source"].isin(["original_human", "original_bot"])].to_numpy()


def get_original_train_val_test_indices(combined: pd.DataFrame, labels: np.ndarray, seed: int = 20260531):
    original_idx = get_original_indices(combined)
    original_labels = labels[original_idx]
    train_rel, val_rel, test_rel = split_indices(original_labels, seed)
    return original_idx[train_rel], original_idx[val_rel], original_idx[test_rel]


def get_family_indices(combined: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        family: combined.index[combined["family"] == family].to_numpy()
        for family in FAMILIES
    }


def train_tfidf_lr(docs: list[str], labels: np.ndarray, train_idx: np.ndarray):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    x_train = vectorizer.fit_transform([docs[i] for i in train_idx])
    clf.fit(x_train, labels[train_idx])
    return vectorizer, clf


def predict_tfidf_lr(model, docs: list[str], idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectorizer, clf = model
    x = vectorizer.transform([docs[i] for i in idx])
    return clf.predict(x), clf.predict_proba(x)[:, 1]


def train_bge_lr(embeddings: np.ndarray, labels: np.ndarray, train_idx: np.ndarray):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(embeddings[train_idx], labels[train_idx])
    return clf


def predict_bge_lr(model, embeddings: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return model.predict(embeddings[idx]), model.predict_proba(embeddings[idx])[:, 1]


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "human_precision": float(precision[0]),
        "bot_precision": float(precision[1]),
        "human_recall": float(recall[0]),
        "bot_recall": float(recall[1]),
        "human_f1": float(f1[0]),
        "bot_f1": float(f1[1]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None:
        out["mean_bot_probability"] = float(np.mean(y_prob))
    return out


def bot_recall(labels: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> float:
    y = labels[idx]
    positives = int((y == 1).sum())
    if positives == 0:
        return float("nan")
    return float(((pred == 1) & (y == 1)).sum() / positives)


def family_train_indices(
    original_train_idx: np.ndarray,
    family_indices: dict[str, np.ndarray],
    include_families: Iterable[str],
) -> np.ndarray:
    include = list(include_families)
    pieces = [original_train_idx] + [family_indices[f] for f in include]
    return np.concatenate(pieces)


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def save_figure(fig: plt.Figure, path: Path, dpi: int = 220) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def short_family_list(families: Iterable[str]) -> str:
    return "+".join(SHORT_NAME[f] for f in families)


def sample_text_stats(text: str) -> dict[str, float]:
    text = "" if pd.isna(text) else str(text)
    words = text.split()
    lowered = text.lower()
    sentence_marks = sum(text.count(mark) for mark in [".", "?", "!"])
    contractions = sum(lowered.count(token) for token in ["n't", "'re", "'ll", "'ve", "'m", "'d", "'s"])
    first_person = sum(lowered.split().count(token) for token in ["i", "i'm", "ive", "i've", "me", "my", "mine"])
    second_person = sum(lowered.split().count(token) for token in ["you", "your", "you're", "u"])
    slang = sum(lowered.split().count(token) for token in ["lol", "lmao", "wtf", "fuck", "shit", "damn", "yeah", "nah"])
    hedge = sum(lowered.split().count(token) for token in ["maybe", "probably", "seems", "seem", "guess", "might", "could"])
    abstract = sum(lowered.split().count(token) for token in ["democracy", "accountability", "implications", "framework", "institutions", "society"])
    n_words = max(len(words), 1)
    return {
        "length_chars": float(len(text)),
        "length_words": float(len(words)),
        "sentence_count": float(sentence_marks),
        "avg_word_len": float(np.mean([len(w) for w in words]) if words else 0.0),
        "question_mark_rate": float(text.count("?") / n_words),
        "exclamation_rate": float(text.count("!") / n_words),
        "contraction_rate": float(contractions / n_words),
        "first_person_rate": float(first_person / n_words),
        "second_person_rate": float(second_person / n_words),
        "slang_or_profanity_rate": float(slang / n_words),
        "hedge_word_rate": float(hedge / n_words),
        "abstract_word_rate": float(abstract / n_words),
    }
