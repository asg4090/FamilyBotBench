#!/usr/bin/env python3
"""
Comment-only BotSim-24 text classification experiment.

Protocol:
- Use only comment_1/comment_2 comment_body from user_post_comment.json.
- Do not use profile, username, user_id, post text, metadata, graph edges.
- Split first, then fit TF-IDF only on training comments.
- Compare raw comments, cleaned comments, and cleaned+truncated comments.

The goal is to test whether comment text alone remains highly separable after
removing links/Markdown/Reddit-specific artifacts.
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
import html
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


URL_RE = re.compile(
    r"""
    (?:
        https?://\S+
        |www\.\S+
        |\b\S+\.(?:com|org|net|edu|gov|io|co|uk|cn|de|fr|jp|ru|info|me|ly|ai)\S*
        |/?r/[A-Za-z0-9_]+
        |/?u/[A-Za-z0-9_-]+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
REDDIT_PREFIX_RE = re.compile(r"\b(?:t1|t3)_[A-Za-z0-9]+\b")
CODE_BLOCK_RE = re.compile(r"`{1,3}[^`]*`{1,3}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
PLACEHOLDER_RE = re.compile(r"^\s*(?:\[?(?:deleted|removed)\]?|null|none|nan)\s*$", re.IGNORECASE)
MODERATION_TEMPLATE_RE = re.compile(
    r"""
    thank\s+you\s+for\s+participating\s+in\s+/?r/
    |has\s+been\s+removed\s+for\s+the\s+following\s+reason
    |message\s+the\s+moderators
    |all\s+submissions\s+to\s+/?r/politics
    |i\s+am\s+a\s+bot,\s+and\s+this\s+action\s+was\s+performed\s+automatically
    """,
    re.IGNORECASE | re.VERBOSE,
)
LLM_REFUSAL_RE = re.compile(
    r"""
    as\s+an\s+ai(?:\s+language\s+model)?
    |i\s+(?:can't|cannot|am\s+unable\s+to|won't)\s+generate
    |unable\s+to\s+generate\s+a\s+response
    |cannot\s+generate\s+a\s+response
    |specific\s+news\s+content
    """,
    re.IGNORECASE | re.VERBOSE,
)


def clean_comment(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub(" ", text)
    text = REDDIT_PREFIX_RE.sub(" ", text)
    text = CODE_BLOCK_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("&amp;", " ")
    text = re.sub(r"[*_~>#|]{1,}", " ", text)
    text = re.sub(r"\b(?:archive|excerpt)\s*:", " ", text, flags=re.IGNORECASE)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def strict_clean_comment(text: str) -> tuple[str, str | None]:
    """Clean a single comment and return (clean_text, drop_reason)."""
    raw = html.unescape(str(text or ""))
    if PLACEHOLDER_RE.match(raw):
        return "", "placeholder"
    if MODERATION_TEMPLATE_RE.search(raw):
        return "", "moderation_template"
    if LLM_REFUSAL_RE.search(raw):
        return "", "llm_refusal"

    cleaned = clean_comment(raw)
    if not cleaned or PLACEHOLDER_RE.match(cleaned):
        return "", "empty_after_clean"
    return cleaned, None


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def load_comment_docs(root: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    users = pd.read_csv(root / "BotSim-24-Dataset" / "Users.csv")
    with open(root / "BotSim-24-Dataset" / "user_post_comment.json", encoding="utf-8") as f:
        data = json.load(f)

    raw_docs: list[str] = []
    clean_docs: list[str] = []
    strict_docs: list[str] = []
    stats = []
    for i, uid in enumerate(users["user_id"].astype(str)):
        entry = data.get(uid, {})
        comments: list[str] = []
        for key in ("comment_1", "comment_2"):
            for comment in entry.get(key, []) or []:
                body = str(comment.get("comment_body", "") or "")
                comments.append(body)
        raw = " ".join(comments)
        cleaned_comments = [clean_comment(c) for c in comments]
        cleaned = " ".join(c for c in cleaned_comments if c)
        strict_cleaned_comments = []
        drop_counts: Counter[str] = Counter()
        for comment in comments:
            strict_cleaned, drop_reason = strict_clean_comment(comment)
            if drop_reason is None:
                strict_cleaned_comments.append(strict_cleaned)
            else:
                drop_counts[drop_reason] += 1
        strict_cleaned_doc = " ".join(strict_cleaned_comments)
        raw_docs.append(raw)
        clean_docs.append(cleaned)
        strict_docs.append(strict_cleaned_doc)
        stats.append(
            {
                "row": i,
                "user_id": uid,
                "label": 1 if i >= 1907 else 0,
                "comment_count": len(comments),
                "strict_kept_comment_count": len(strict_cleaned_comments),
                "dropped_placeholder_comments": drop_counts["placeholder"],
                "dropped_moderation_template_comments": drop_counts["moderation_template"],
                "dropped_llm_refusal_comments": drop_counts["llm_refusal"],
                "dropped_empty_after_clean_comments": drop_counts["empty_after_clean"],
                "raw_chars": len(raw),
                "raw_words": len(raw.split()),
                "clean_chars": len(cleaned),
                "clean_words": len(cleaned.split()),
                "strict_chars": len(strict_cleaned_doc),
                "strict_words": len(strict_cleaned_doc.split()),
                "raw_has_link": bool(URL_RE.search(raw) or MARKDOWN_LINK_RE.search(raw)),
                "clean_has_link": bool(URL_RE.search(cleaned) or MARKDOWN_LINK_RE.search(cleaned)),
                "strict_has_link": bool(URL_RE.search(strict_cleaned_doc) or MARKDOWN_LINK_RE.search(strict_cleaned_doc)),
                "empty_after_clean": len(cleaned.strip()) == 0,
                "empty_after_strict_clean": len(strict_cleaned_doc.strip()) == 0,
            }
        )
    return pd.DataFrame(stats), raw_docs, clean_docs, strict_docs


def split_indices(labels: np.ndarray, seed: int, candidate_idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(labels)) if candidate_idx is None else np.asarray(candidate_idx)
    train_idx, holdout_idx = train_test_split(idx, test_size=0.3, random_state=seed, stratify=labels[idx])
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=1 / 3,
        random_state=seed,
        stratify=labels[holdout_idx],
    )
    return train_idx, val_idx, test_idx


def train_eval_text_model(
    docs: list[str],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    max_features: int,
) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray, TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    X_train = vectorizer.fit_transform([docs[i] for i in train_idx])
    clf.fit(X_train, labels[train_idx])

    scores: dict[str, dict[str, float]] = {}
    preds_by_split = {}
    probs_by_split = {}
    for split_name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        X = vectorizer.transform([docs[i] for i in idx])
        pred = clf.predict(X)
        prob = clf.predict_proba(X)[:, 1]
        preds_by_split[split_name] = pred
        probs_by_split[split_name] = prob
        scores[split_name] = {
            "accuracy": accuracy_score(labels[idx], pred),
            "precision": precision_score(labels[idx], pred, zero_division=0),
            "recall": recall_score(labels[idx], pred, zero_division=0),
            "f1": f1_score(labels[idx], pred, zero_division=0),
        }
    return scores, preds_by_split["test"], probs_by_split["test"], vectorizer, clf


def top_terms(vectorizer: TfidfVectorizer, clf: LogisticRegression, n: int = 40) -> pd.DataFrame:
    terms = np.array(vectorizer.get_feature_names_out())
    coef = clf.coef_[0]
    bot_order = np.argsort(coef)[-n:][::-1]
    human_order = np.argsort(coef)[:n]
    return pd.DataFrame(
        {
            "rank": np.arange(1, n + 1),
            "bot_terms": terms[bot_order],
            "bot_coef": coef[bot_order],
            "human_terms": terms[human_order],
            "human_coef": coef[human_order],
        }
    )


def save_metric_plot(metrics: pd.DataFrame, path: Path) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    x = np.arange(len(test))
    width = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, test["accuracy"], width, label="Accuracy", color="#4C78A8")
    plt.bar(x + width / 2, test["f1"], width, label="F1", color="#E45756")
    plt.xticks(x, test["variant"], rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Comment-only text classification performance")
    plt.legend()
    for i, row in test.reset_index(drop=True).iterrows():
        plt.text(i - width / 2, row["accuracy"] + 0.012, f"{row['accuracy']:.3f}", ha="center", fontsize=8)
        plt.text(i + width / 2, row["f1"] + 0.012, f"{row['f1']:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_length_plot(stats: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8.5, 5))
    data = []
    labels = []
    for label_value, label_name in [(0, "human"), (1, "bot")]:
        subset = stats[stats["label"] == label_value]
        data.append(np.log1p(subset["raw_words"]))
        labels.append(f"{label_name} raw")
        data.append(np.log1p(subset["clean_words"]))
        labels.append(f"{label_name} clean")
        data.append(np.log1p(subset["strict_words"]))
        labels.append(f"{label_name} strict")
    plt.boxplot(data, tick_labels=labels, patch_artist=True)
    plt.ylabel("log(1 + words)")
    plt.title("Comment text length before and after cleaning")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_link_plot(stats: pd.DataFrame, path: Path) -> None:
    grouped = stats.groupby("label")[["raw_has_link", "clean_has_link", "strict_has_link"]].mean().reset_index()
    labels = ["human", "bot"]
    x = np.arange(2)
    width = 0.25
    plt.figure(figsize=(6.8, 4.5))
    plt.bar(x - width, grouped["raw_has_link"], width, label="raw", color="#4C78A8")
    plt.bar(x, grouped["clean_has_link"], width, label="clean", color="#54A24B")
    plt.bar(x + width, grouped["strict_has_link"], width, label="strict", color="#F58518")
    plt.xticks(x, labels)
    plt.ylabel("Share of users with link-like text")
    plt.title("Link-like artifacts before and after cleaning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_confusion(labels: np.ndarray, idx: np.ndarray, pred: np.ndarray, path: Path, title: str) -> None:
    cm = confusion_matrix(labels[idx], pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["human", "bot"])
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--truncate-words", type=int, default=600)
    args = parser.parse_args()

    root = _RELEASE_ROOT
    out_dir = root / "outputs" / "comment_text_baseline"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    stats, raw_docs, clean_docs, strict_docs = load_comment_docs(root)
    labels = stats["label"].to_numpy(dtype=int)
    train_idx, val_idx, test_idx = split_indices(labels, args.seed)
    nonempty_idx = stats.index[~stats["empty_after_strict_clean"]].to_numpy()
    nonempty_train_idx, nonempty_val_idx, nonempty_test_idx = split_indices(labels, args.seed, nonempty_idx)

    variants = {
        "raw_comments": raw_docs,
        "clean_comments": clean_docs,
        f"clean_comments_truncated_{args.truncate_words}w": [
            truncate_words(doc, args.truncate_words) for doc in clean_docs
        ],
        "strict_clean_comments": strict_docs,
        f"strict_clean_comments_truncated_{args.truncate_words}w": [
            truncate_words(doc, args.truncate_words) for doc in strict_docs
        ],
    }
    nonempty_variants = {
        "strict_clean_comments_nonempty_users": strict_docs,
        f"strict_clean_comments_nonempty_users_truncated_{args.truncate_words}w": [
            truncate_words(doc, args.truncate_words) for doc in strict_docs
        ],
    }

    metric_rows = []
    predictions = pd.DataFrame(
        {
            "row": np.arange(len(labels)),
            "user_id": stats["user_id"],
            "true_label": np.where(labels == 1, "bot", "human"),
            "split": "unused",
        }
    )
    predictions.loc[train_idx, "split"] = "train"
    predictions.loc[val_idx, "split"] = "validation"
    predictions.loc[test_idx, "split"] = "test"

    term_tables = {}
    best_variant = None
    best_f1 = -1.0
    best_pred = None
    strict_truncated_test_pred = None
    for variant, docs in variants.items():
        scores, test_pred, test_prob, vectorizer, clf = train_eval_text_model(
            docs, labels, train_idx, val_idx, test_idx, args.max_features
        )
        for split, vals in scores.items():
            metric_rows.append({"variant": variant, "split": split, **vals})
        predictions.loc[test_idx, f"{variant}_pred"] = np.where(test_pred == 1, "bot", "human")
        predictions.loc[test_idx, f"{variant}_bot_probability"] = test_prob
        if variant == f"strict_clean_comments_truncated_{args.truncate_words}w":
            strict_truncated_test_pred = test_pred
        term_tables[variant] = top_terms(vectorizer, clf, n=40)
        term_tables[variant].to_csv(out_dir / f"top_terms_{variant}.csv", index=False)
        if scores["test"]["f1"] > best_f1:
            best_f1 = scores["test"]["f1"]
            best_variant = variant
            best_pred = test_pred
    for variant, docs in nonempty_variants.items():
        scores, test_pred, test_prob, vectorizer, clf = train_eval_text_model(
            docs,
            labels,
            nonempty_train_idx,
            nonempty_val_idx,
            nonempty_test_idx,
            args.max_features,
        )
        for split, vals in scores.items():
            metric_rows.append({"variant": variant, "split": split, **vals})
        term_tables[variant] = top_terms(vectorizer, clf, n=40)
        term_tables[variant].to_csv(out_dir / f"top_terms_{variant}.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "comment_text_metrics.csv", index=False)
    stats.to_csv(out_dir / "comment_text_cleaning_stats.csv", index=False)
    predictions.to_csv(out_dir / "comment_text_predictions.csv", index=False)

    artifact_summary = (
        stats.groupby("label")
        .agg(
            users=("row", "count"),
            mean_comments=("comment_count", "mean"),
            median_comments=("comment_count", "median"),
            mean_raw_words=("raw_words", "mean"),
            median_raw_words=("raw_words", "median"),
            mean_clean_words=("clean_words", "mean"),
            median_clean_words=("clean_words", "median"),
            mean_strict_words=("strict_words", "mean"),
            median_strict_words=("strict_words", "median"),
            raw_link_share=("raw_has_link", "mean"),
            clean_link_share=("clean_has_link", "mean"),
            strict_link_share=("strict_has_link", "mean"),
            empty_after_clean_share=("empty_after_clean", "mean"),
            empty_after_strict_clean_share=("empty_after_strict_clean", "mean"),
            mean_strict_kept_comments=("strict_kept_comment_count", "mean"),
            median_strict_kept_comments=("strict_kept_comment_count", "median"),
            dropped_placeholder_comments=("dropped_placeholder_comments", "sum"),
            dropped_moderation_template_comments=("dropped_moderation_template_comments", "sum"),
            dropped_llm_refusal_comments=("dropped_llm_refusal_comments", "sum"),
            dropped_empty_after_clean_comments=("dropped_empty_after_clean_comments", "sum"),
        )
        .reset_index()
    )
    artifact_summary["label_name"] = artifact_summary["label"].map({0: "human", 1: "bot"})
    artifact_summary.to_csv(out_dir / "comment_artifact_summary.csv", index=False)

    split_summary = pd.DataFrame(
        [
            {"scenario": "all_users", "split": "train", "size": len(train_idx), "bot_rate": labels[train_idx].mean()},
            {"scenario": "all_users", "split": "validation", "size": len(val_idx), "bot_rate": labels[val_idx].mean()},
            {"scenario": "all_users", "split": "test", "size": len(test_idx), "bot_rate": labels[test_idx].mean()},
            {
                "scenario": "strict_nonempty_users",
                "split": "train",
                "size": len(nonempty_train_idx),
                "bot_rate": labels[nonempty_train_idx].mean(),
            },
            {
                "scenario": "strict_nonempty_users",
                "split": "validation",
                "size": len(nonempty_val_idx),
                "bot_rate": labels[nonempty_val_idx].mean(),
            },
            {
                "scenario": "strict_nonempty_users",
                "split": "test",
                "size": len(nonempty_test_idx),
                "bot_rate": labels[nonempty_test_idx].mean(),
            },
        ]
    )
    split_summary.to_csv(out_dir / "split_summary.csv", index=False)

    save_metric_plot(metrics, fig_dir / "comment_text_metric_comparison.png")
    save_length_plot(stats, fig_dir / "comment_length_cleaning_boxplot.png")
    save_link_plot(stats, fig_dir / "comment_link_artifact_share.png")
    if best_pred is not None and best_variant is not None:
        save_confusion(
            labels,
            test_idx,
            best_pred,
            fig_dir / "best_comment_model_confusion_matrix.png",
            f"Best comment-only model: {best_variant}",
        )
    if strict_truncated_test_pred is not None:
        save_confusion(
            labels,
            test_idx,
            strict_truncated_test_pred,
            fig_dir / f"strict_clean_truncated_{args.truncate_words}w_confusion_matrix.png",
            f"Strict cleaned comments, truncated to {args.truncate_words} words",
        )

    summary = {
        "seed": args.seed,
        "max_features": args.max_features,
        "truncate_words": args.truncate_words,
        "num_users": int(len(labels)),
        "num_humans": int((labels == 0).sum()),
        "num_bots": int((labels == 1).sum()),
        "best_variant": best_variant,
        "best_test_f1": float(best_f1),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(metrics[metrics["split"] == "test"].to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
