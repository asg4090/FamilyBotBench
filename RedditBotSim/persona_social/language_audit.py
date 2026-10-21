from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import re
from collections import Counter
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_']+")
HIGH_RISK_TERMS = {
    "political",
    "implications",
    "accountability",
    "ongoing",
    "landscape",
    "conflict",
    "justice",
    "legal",
    "truly",
    "crucial",
    "broader",
    "critical",
    "dialogue",
    "ramifications",
    "strategy",
}
HUMAN_SIDE_TERMS = {
    "said",
    "people",
    "israel",
    "good",
    "hamas",
    "vote",
    "did",
    "going",
    "trump",
    "fuck",
    "shit",
    "know",
    "yeah",
    "lol",
}


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(str(text or ""))]


def term_rate(tokens: list[str], terms: set[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token in terms) / len(tokens)


def doc_language_stats(doc: str) -> dict[str, float]:
    tokens = tokenize(doc)
    sentences = [part.strip() for part in re.split(r"[.!?]+", str(doc or "")) if part.strip()]
    return {
        "word_count": float(len(tokens)),
        "sentence_count": float(len(sentences)),
        "mean_sentence_words": float(len(tokens) / max(1, len(sentences))),
        "high_risk_term_rate": term_rate(tokens, HIGH_RISK_TERMS),
        "human_side_term_rate": term_rate(tokens, HUMAN_SIDE_TERMS),
        "question_mark_count": float(str(doc or "").count("?")),
        "exclamation_count": float(str(doc or "").count("!")),
        "first_person_count": float(sum(1 for token in tokens if token in {"i", "me", "my", "we", "our"})),
    }


def audit_documents(rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    out = []
    for row in rows:
        payload = dict(row)
        stats = doc_language_stats(str(payload.get("doc", "")))
        out.append({**payload, **stats})
    return pd.DataFrame(out)


def summarize_by_family(audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return audit_df
    numeric_cols = [
        "word_count",
        "sentence_count",
        "mean_sentence_words",
        "high_risk_term_rate",
        "human_side_term_rate",
        "question_mark_count",
        "exclamation_count",
        "first_person_count",
    ]
    return (
        audit_df.groupby("family", as_index=False)
        .agg(
            bot_count=("user_id", "nunique"),
            mean_word_count=("word_count", "mean"),
            median_word_count=("word_count", "median"),
            mean_sentence_words=("mean_sentence_words", "mean"),
            high_risk_term_rate=("high_risk_term_rate", "mean"),
            human_side_term_rate=("human_side_term_rate", "mean"),
            question_mark_count=("question_mark_count", "mean"),
            first_person_count=("first_person_count", "mean"),
        )
        .sort_values("family")
    )


def centroid_distances(embeddings: np.ndarray, families: list[str], human_centroid: np.ndarray, bot_centroid: np.ndarray) -> pd.DataFrame:
    rows = []
    fam_arr = np.asarray(families)
    for family in sorted(set(families)):
        idx = np.where(fam_arr == family)[0]
        if len(idx) == 0:
            continue
        centroid = embeddings[idx].mean(axis=0)
        rows.append(
            {
                "family": family,
                "count": int(len(idx)),
                "distance_to_human_centroid": float(np.linalg.norm(centroid - human_centroid)),
                "distance_to_original_bot_centroid": float(np.linalg.norm(centroid - bot_centroid)),
            }
        )
    return pd.DataFrame(rows)
