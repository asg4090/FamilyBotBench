#!/usr/bin/env python3
"""Thread-derived topic-controlled static text detection.

This experiment defines topic at the Reddit thread/post level, not by comment
embeddings. The topic classifier only reads post text and subreddit.
"""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


from collections import defaultdict
import json
from pathlib import Path
import re
import unicodedata

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common_static_text import (
    FIGURES_DIR,
    FAMILIES,
    PROJECT_ROOT,
    RESULTS_DIR,
    SHORT_NAME,
    STATIC_RUN,
    compute_binary_metrics,
    ensure_dir,
    predict_tfidf_lr,
    save_figure,
    save_table,
    train_tfidf_lr,
)


SEED = 20260603
MIN_TOPIC_CLASS_COUNT = 30
MAX_PER_CLASS_PER_TOPIC = 200
TRAIN_SIZE = 0.70
THREAD_RESULTS_DIR = RESULTS_DIR / "thread_topic_controlled"

POSTS_CSV = PROJECT_ROOT / "RedditBotSim" / "Data" / "RedditData" / "Posts.csv"
USER_POST_COMMENT_JSON = PROJECT_ROOT / "BotSim-24-Dataset" / "user_post_comment.json"
BOT_COMMENTS_CSV = STATIC_RUN / "generated_comments_all.csv"

TOPIC_RULES: list[tuple[str, list[str]]] = [
    (
        "israel_palestine_middle_east",
        ["israel", "palestine", "gaza", "hamas", "west bank", "rafah", "hostage", "idf", "netanyahu"],
    ),
    (
        "russia_ukraine_war",
        ["russia", "ukraine", "putin", "zelensky", "zelenskyy", "nato", "crimea", "kremlin", "moscow"],
    ),
    (
        "us_politics_elections",
        ["trump", "biden", "election", "congress", "senate", "republican", "democrat", "campaign", "vote"],
    ),
    (
        "brazil_latin_america",
        ["brazil", "bolsonaro", "lula", "vinicius", "paqueta", "copa", "latin america"],
    ),
    (
        "law_crime_rights",
        ["court", "prison", "police", "ban", "banned", "charged", "legal", "judge", "rights", "protest", "flag"],
    ),
    (
        "global_affairs_policy",
        ["china", "europe", "un", "refugee", "sanctions", "diplomacy", "government", "war", "military", "border"],
    ),
    (
        "economy_health_society",
        ["economy", "inflation", "market", "drug", "cannabis", "health", "addiction", "education", "labor", "labour"],
    ),
]
FALLBACK_TOPIC = "other_or_mixed"
TOPIC_ORDER = [name for name, _ in TOPIC_RULES] + [FALLBACK_TOPIC]


def normalize_text(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.lower()).strip()


def standard_link_id(submission_id: object) -> str:
    raw = "" if pd.isna(submission_id) else str(submission_id).strip()
    if raw.startswith("t3_"):
        return raw
    return f"t3_{raw}"


def classify_topic(post_text: object, subreddit: object) -> tuple[str, str]:
    haystack = f"{normalize_text(post_text)} {normalize_text(subreddit)}"
    for topic, keywords in TOPIC_RULES:
        hits = [kw for kw in keywords if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", haystack)]
        if hits:
            return topic, ";".join(hits)
    return FALLBACK_TOPIC, ""


def compact(text: object, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", "" if pd.isna(text) else str(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def clean_field(text: object, limit: int | None = None) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if limit is not None and len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def make_post_record(
    link_id: str,
    post_text: object,
    subreddit: object,
    source: str,
    author_name: object = "",
    created_utc: object = "",
) -> dict[str, object]:
    topic, matched_keywords = classify_topic(post_text, subreddit)
    return {
        "link_id": link_id,
        "submission_id": link_id.replace("t3_", "", 1),
        "post_text": clean_field(post_text, limit=1200),
        "subreddit": clean_field(subreddit, limit=120),
        "post_author_name": clean_field(author_name, limit=120),
        "post_created_utc": clean_field(created_utc, limit=120),
        "topic": topic,
        "matched_keywords": matched_keywords,
        "post_source": source,
    }


def build_post_topic_map() -> pd.DataFrame:
    records_by_link: dict[str, dict[str, object]] = {}

    user_data = json.loads(USER_POST_COMMENT_JSON.read_text(encoding="utf-8"))
    for user_blob in user_data.values():
        for post in user_blob.get("posts", []):
            link_id = standard_link_id(post.get("submission_id", ""))
            if link_id == "t3_":
                continue
            rec = make_post_record(
                link_id=link_id,
                post_text=post.get("posts", ""),
                subreddit=post.get("subreddit", ""),
                source="human_original_posts",
                author_name=post.get("author_name", ""),
                created_utc=post.get("created_utc", ""),
            )
            records_by_link.setdefault(link_id, rec)

    bot_posts = pd.read_csv(POSTS_CSV)
    for row in bot_posts.itertuples(index=False):
        link_id = standard_link_id(getattr(row, "submission_id"))
        rec = make_post_record(
            link_id=link_id,
            post_text=getattr(row, "posts"),
            subreddit=getattr(row, "subreddit"),
            source="redditbotsim_candidate_posts",
            author_name=getattr(row, "author_name", ""),
            created_utc=getattr(row, "created_utc", ""),
        )
        if link_id in records_by_link:
            records_by_link[link_id]["post_source"] = f"{records_by_link[link_id]['post_source']}+redditbotsim_candidate_posts"
        else:
            records_by_link[link_id] = rec

    post_map = pd.DataFrame(records_by_link.values())
    post_map = post_map.sort_values(["topic", "link_id"]).reset_index(drop=True)
    return post_map


def load_human_comments(post_map: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, int]]:
    post_lookup = post_map.set_index("link_id").to_dict("index")
    user_data = json.loads(USER_POST_COMMENT_JSON.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    stats = {"human_total_comments": 0, "human_mapped_comments": 0, "human_unmapped_comments": 0}
    serial = 0
    for user_id, user_blob in user_data.items():
        for level_key in ["comment_1", "comment_2"]:
            for comment in user_blob.get(level_key, []):
                stats["human_total_comments"] += 1
                link_id = str(comment.get("link_id", "")).strip()
                post = post_lookup.get(link_id)
                if post is None:
                    stats["human_unmapped_comments"] += 1
                    continue
                stats["human_mapped_comments"] += 1
                serial += 1
                rows.append(
                    {
                        "sample_id": f"human_{serial:06d}_{comment.get('comment_id', '')}",
                        "label": 0,
                        "source": "original_human_comment",
                        "family": "original_human",
                        "family_short": "human",
                        "user_id": user_id,
                        "comment_id": comment.get("comment_id", ""),
                        "link_id": link_id,
                        "parent_id": comment.get("parent_id", ""),
                        "level": int(float(comment.get("level", 0) or 0)),
                        "subreddit": comment.get("subreddit", post.get("subreddit", "")),
                        "post_text": post["post_text"],
                        "topic": post["topic"],
                        "comment_body": clean_field(comment.get("comment_body", ""), limit=10000),
                    }
                )
    return rows, stats


def load_bot_comments(post_map: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, int]]:
    post_lookup = post_map.set_index("link_id").to_dict("index")
    bot_comments = pd.read_csv(BOT_COMMENTS_CSV)
    rows: list[dict[str, object]] = []
    stats = {"bot_total_comments": int(len(bot_comments)), "bot_mapped_comments": 0, "bot_unmapped_comments": 0}
    serial = 0
    for row in bot_comments.itertuples(index=False):
        link_id = str(getattr(row, "link_id", "")).strip()
        post = post_lookup.get(link_id)
        if post is None:
            stats["bot_unmapped_comments"] += 1
            continue
        stats["bot_mapped_comments"] += 1
        serial += 1
        family = getattr(row, "family", "")
        rows.append(
            {
                "sample_id": f"bot_{serial:06d}_{getattr(row, 'comment_id', '')}",
                "label": 1,
                "source": "generated_static_family_comment",
                "family": family,
                "family_short": SHORT_NAME.get(family, family),
                "user_id": getattr(row, "user_id", ""),
                "comment_id": getattr(row, "comment_id", ""),
                "link_id": link_id,
                "parent_id": getattr(row, "parent_id", ""),
                "level": int(getattr(row, "level", 0) or 0),
                "subreddit": getattr(row, "subreddit", post.get("subreddit", "")),
                "post_text": post["post_text"],
                "topic": post["topic"],
                "comment_body": clean_field(getattr(row, "comment_body", ""), limit=10000),
            }
        )
    return rows, stats


def build_comment_topic_dataset(post_map: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    human_rows, human_stats = load_human_comments(post_map)
    bot_rows, bot_stats = load_bot_comments(post_map)
    dataset = pd.DataFrame(human_rows + bot_rows)
    dataset["comment_body"] = dataset["comment_body"].fillna("").astype(str)
    dataset = dataset[dataset["comment_body"].str.strip().ne("")].copy()
    dataset = dataset.sort_values(["label", "topic", "link_id", "sample_id"]).reset_index(drop=True)
    return dataset, {**human_stats, **bot_stats, "usable_mapped_comments": int(len(dataset))}


def build_resource_audit(dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for topic in TOPIC_ORDER:
        part = dataset[dataset["topic"].eq(topic)]
        human_count = int((part["label"] == 0).sum())
        bot_count = int((part["label"] == 1).sum())
        n_posts = int(part["link_id"].nunique())
        fam_counts = {
            f"family_count_{SHORT_NAME[family]}": int((part["family"] == family).sum())
            for family in FAMILIES
        }
        eligible_matched = human_count >= MIN_TOPIC_CLASS_COUNT and bot_count >= MIN_TOPIC_CLASS_COUNT
        eligible_loto = eligible_matched
        eligible_lpo = False
        lpo_reason = ""
        if n_posts < 4:
            lpo_reason = "n_posts_lt_4"
        else:
            labels_by_post = part.groupby("link_id")["label"].agg(lambda s: ",".join(map(str, sorted(set(s)))))
            posts_with_human = int(labels_by_post.str.contains("0").sum())
            posts_with_bot = int(labels_by_post.str.contains("1").sum())
            eligible_lpo = posts_with_human >= 2 and posts_with_bot >= 2 and eligible_matched
            if not eligible_lpo:
                lpo_reason = f"post_label_coverage_low_human_posts={posts_with_human}_bot_posts={posts_with_bot}"
        skip_reasons = []
        if not eligible_matched:
            skip_reasons.append(f"low_resource_human={human_count}_bot={bot_count}")
        if not eligible_lpo:
            skip_reasons.append(lpo_reason)
        row = {
            "topic": topic,
            "n_posts": n_posts,
            "human_comments": human_count,
            "bot_comments": bot_count,
            **fam_counts,
            "eligible_for_matched": bool(eligible_matched),
            "eligible_for_loto": bool(eligible_loto),
            "eligible_for_leave_posts_out": bool(eligible_lpo),
            "skip_reason": ";".join([r for r in skip_reasons if r]) if skip_reasons else "",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def make_balanced_pool(
    dataset: pd.DataFrame,
    group_col: str | None = None,
    eligible_topics: list[str] | None = None,
    max_per_class: int = MAX_PER_CLASS_PER_TOPIC,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    if group_col is None:
        groups = [("__global__", dataset)]
    else:
        groups = list(dataset.groupby(group_col, sort=True))
    eligible_set = set(eligible_topics) if eligible_topics is not None else None
    for group_name, part in groups:
        if eligible_set is not None and group_name not in eligible_set:
            continue
        human = part[part["label"].eq(0)]
        bot = part[part["label"].eq(1)]
        keep = min(len(human), len(bot), max_per_class)
        if keep <= 0:
            continue
        selected.append(human.sample(n=keep, random_state=int(rng.integers(0, 1_000_000_000))))
        selected.append(bot.sample(n=keep, random_state=int(rng.integers(0, 1_000_000_000))))
    if not selected:
        return dataset.iloc[[]].copy()
    out = pd.concat(selected, ignore_index=False).sort_index().reset_index(drop=True)
    return out


def docs_labels(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    return df["comment_body"].fillna("").astype(str).tolist(), df["label"].to_numpy(dtype=int)


def fit_predict_tfidf(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[dict[str, float], np.ndarray, np.ndarray, str]:
    if train_df.empty or test_df.empty:
        return {}, np.array([], dtype=int), np.array([], dtype=float), "skipped_empty_split"
    if train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
        return {}, np.array([], dtype=int), np.array([], dtype=float), "skipped_one_class_split"
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    docs, labels = docs_labels(all_df)
    train_idx = np.arange(len(train_df))
    test_idx = np.arange(len(train_df), len(all_df))
    try:
        model = train_tfidf_lr(docs, labels, train_idx)
        pred, prob = predict_tfidf_lr(model, docs, test_idx)
    except Exception as exc:
        return {}, np.array([], dtype=int), np.array([], dtype=float), f"skipped_model_error:{exc}"
    return compute_binary_metrics(labels[test_idx], pred, prob), pred, prob, "ok"


def safe_stratified_split(df: pd.DataFrame, stratify_cols: list[str], seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if df.empty:
        return df.copy(), df.copy(), "skipped_empty_pool"
    stratify = df[stratify_cols].astype(str).agg("__".join, axis=1)
    if stratify.value_counts().min() < 2:
        stratify = df["label"].astype(str)
    if stratify.value_counts().min() < 2:
        return df.copy(), df.iloc[[]].copy(), "skipped_too_few_per_class"
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        train_size=TRAIN_SIZE,
        random_state=seed,
        stratify=stratify,
    )
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True), "ok"


def prediction_records(
    protocol: str,
    model: str,
    topic: str,
    test_df: pd.DataFrame,
    pred: np.ndarray,
    prob: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, row in enumerate(test_df.itertuples(index=False)):
        rows.append(
            {
                "protocol": protocol,
                "model": model,
                "topic": topic,
                "sample_id": getattr(row, "sample_id"),
                "label": int(getattr(row, "label")),
                "family_short": getattr(row, "family_short"),
                "link_id": getattr(row, "link_id"),
                "y_pred": int(pred[i]),
                "bot_probability": float(prob[i]),
            }
        )
    return rows


def run_protocol_a(dataset: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
    eligible_topics = audit.loc[audit["eligible_for_matched"], "topic"].tolist()
    matched_pool = make_balanced_pool(dataset, group_col="topic", eligible_topics=eligible_topics, seed=SEED)
    baseline_pool = make_balanced_pool(dataset, group_col=None, seed=SEED + 11)

    baseline_train, baseline_test, baseline_split_status = safe_stratified_split(
        baseline_pool,
        ["label"],
        seed=SEED + 12,
    )
    matched_train, matched_test, matched_split_status = safe_stratified_split(
        matched_pool,
        ["label", "topic"],
        seed=SEED + 13,
    )

    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    baseline_metrics, baseline_pred, baseline_prob, baseline_model_status = fit_predict_tfidf(baseline_train, baseline_test)
    matched_metrics, matched_pred, matched_prob, matched_model_status = fit_predict_tfidf(matched_train, matched_test)
    if baseline_model_status == "ok":
        predictions.extend(prediction_records("global_balanced_baseline", "TF-IDF+LR", "__global__", baseline_test, baseline_pred, baseline_prob))
    if matched_model_status == "ok":
        predictions.extend(prediction_records("topic_matched_aggregate", "TF-IDF+LR", "__all_eligible_topics__", matched_test, matched_pred, matched_prob))

    rows.append(
        {
            "model": "TF-IDF+LR",
            "status": "ok" if baseline_model_status == "ok" and matched_model_status == "ok" else f"baseline={baseline_model_status};matched={matched_model_status}",
            "eligible_topics": ";".join(eligible_topics),
            "baseline_total_retained": int(len(baseline_pool)),
            "baseline_train_n": int(len(baseline_train)),
            "baseline_test_n": int(len(baseline_test)),
            "matched_total_retained": int(len(matched_pool)),
            "matched_train_n": int(len(matched_train)),
            "matched_test_n": int(len(matched_test)),
            "baseline_global_balanced_bot_recall": baseline_metrics.get("bot_recall", np.nan),
            "baseline_global_balanced_human_recall": baseline_metrics.get("human_recall", np.nan),
            "baseline_global_balanced_macro_f1": baseline_metrics.get("macro_f1", np.nan),
            "topic_matched_bot_recall": matched_metrics.get("bot_recall", np.nan),
            "topic_matched_human_recall": matched_metrics.get("human_recall", np.nan),
            "topic_matched_macro_f1": matched_metrics.get("macro_f1", np.nan),
            "bot_recall_drop_vs_global_balanced": baseline_metrics.get("bot_recall", np.nan) - matched_metrics.get("bot_recall", np.nan),
        }
    )
    rows.append(
        {
            "model": "BGE+LR",
            "status": "bge_skipped_no_comment_level_local_embeddings",
            "eligible_topics": ";".join(eligible_topics),
            "baseline_total_retained": int(len(baseline_pool)),
            "baseline_train_n": np.nan,
            "baseline_test_n": np.nan,
            "matched_total_retained": int(len(matched_pool)),
            "matched_train_n": np.nan,
            "matched_test_n": np.nan,
            "baseline_global_balanced_bot_recall": np.nan,
            "baseline_global_balanced_human_recall": np.nan,
            "baseline_global_balanced_macro_f1": np.nan,
            "topic_matched_bot_recall": np.nan,
            "topic_matched_human_recall": np.nan,
            "topic_matched_macro_f1": np.nan,
            "bot_recall_drop_vs_global_balanced": np.nan,
        }
    )
    return pd.DataFrame(rows), predictions, matched_pool


def run_protocol_b(dataset: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    eligible_topics = audit.loc[audit["eligible_for_loto"], "topic"].tolist()
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for heldout in eligible_topics:
        train_topics = [topic for topic in eligible_topics if topic != heldout]
        train_pool = make_balanced_pool(dataset, group_col="topic", eligible_topics=train_topics, seed=SEED + 101)
        test_pool = make_balanced_pool(dataset, group_col="topic", eligible_topics=[heldout], seed=SEED + 102)
        metrics, pred, prob, status = fit_predict_tfidf(train_pool, test_pool)
        if status == "ok":
            predictions.extend(prediction_records("leave_one_topic_out", "TF-IDF+LR", heldout, test_pool, pred, prob))
        rows.append(
            {
                "model": "TF-IDF+LR",
                "heldout_topic": heldout,
                "status": status,
                "train_topics": ";".join(train_topics),
                "train_n": int(len(train_pool)),
                "test_n": int(len(test_pool)),
                "test_human_n": int((test_pool["label"] == 0).sum()) if not test_pool.empty else 0,
                "test_bot_n": int((test_pool["label"] == 1).sum()) if not test_pool.empty else 0,
                **metrics,
            }
        )
    rows.append(
        {
            "model": "BGE+LR",
            "heldout_topic": "__all__",
            "status": "bge_skipped_no_comment_level_local_embeddings",
            "train_topics": "",
            "train_n": np.nan,
            "test_n": np.nan,
            "test_human_n": np.nan,
            "test_bot_n": np.nan,
        }
    )
    return pd.DataFrame(rows), predictions


def find_post_split(topic_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    links = np.array(sorted(topic_df["link_id"].unique()))
    if len(links) < 4:
        return topic_df.iloc[[]], topic_df.iloc[[]], "skipped_n_posts_lt_4"
    rng = np.random.default_rng(seed)
    for attempt in range(200):
        shuffled = rng.permutation(links)
        n_train_links = max(2, int(round(len(shuffled) * TRAIN_SIZE)))
        n_train_links = min(n_train_links, len(shuffled) - 2)
        train_links = set(shuffled[:n_train_links])
        test_links = set(shuffled[n_train_links:])
        train = topic_df[topic_df["link_id"].isin(train_links)]
        test = topic_df[topic_df["link_id"].isin(test_links)]
        if train["label"].nunique() == 2 and test["label"].nunique() == 2:
            train_bal = make_balanced_pool(train, group_col=None, seed=seed + attempt + 1)
            test_bal = make_balanced_pool(test, group_col=None, seed=seed + attempt + 1000)
            if len(train_bal) >= 20 and len(test_bal) >= 10:
                return train_bal, test_bal, "ok"
    return topic_df.iloc[[]], topic_df.iloc[[]], "skipped_no_disjoint_post_split_with_both_labels"


def run_protocol_c(dataset: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    candidate_topics = audit.loc[audit["eligible_for_matched"], "topic"].tolist()
    for pos, topic in enumerate(candidate_topics):
        topic_df = dataset[dataset["topic"].eq(topic)].copy()
        train_pool, test_pool, split_status = find_post_split(topic_df, seed=SEED + 200 + pos)
        metrics, pred, prob, model_status = fit_predict_tfidf(train_pool, test_pool) if split_status == "ok" else ({}, np.array([]), np.array([]), split_status)
        status = model_status if split_status == "ok" else split_status
        if status == "ok":
            predictions.extend(prediction_records("leave_posts_out_within_topic", "TF-IDF+LR", topic, test_pool, pred, prob))
        rows.append(
            {
                "model": "TF-IDF+LR",
                "topic": topic,
                "status": status,
                "train_n": int(len(train_pool)),
                "test_n": int(len(test_pool)),
                "train_posts": int(train_pool["link_id"].nunique()) if not train_pool.empty else 0,
                "test_posts": int(test_pool["link_id"].nunique()) if not test_pool.empty else 0,
                "train_human_n": int((train_pool["label"] == 0).sum()) if not train_pool.empty else 0,
                "train_bot_n": int((train_pool["label"] == 1).sum()) if not train_pool.empty else 0,
                "test_human_n": int((test_pool["label"] == 0).sum()) if not test_pool.empty else 0,
                "test_bot_n": int((test_pool["label"] == 1).sum()) if not test_pool.empty else 0,
                **metrics,
            }
        )
    rows.append(
        {
            "model": "BGE+LR",
            "topic": "__all__",
            "status": "bge_skipped_no_comment_level_local_embeddings",
            "train_n": np.nan,
            "test_n": np.nan,
            "train_posts": np.nan,
            "test_posts": np.nan,
            "train_human_n": np.nan,
            "train_bot_n": np.nan,
            "test_human_n": np.nan,
            "test_bot_n": np.nan,
        }
    )
    return pd.DataFrame(rows), predictions


def plot_distribution(dataset: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    label_counts = dataset.groupby(["topic", "label"]).size().unstack(fill_value=0).reindex(TOPIC_ORDER, fill_value=0)
    label_counts.rename(columns={0: "human", 1: "bot"}).plot(kind="bar", ax=axes[0])
    axes[0].set_title("Thread topic comment counts by label")
    axes[0].set_xlabel("topic")
    axes[0].set_ylabel("comments")
    axes[0].tick_params(axis="x", rotation=45)
    bot = dataset[dataset["label"].eq(1)]
    fam_counts = bot.groupby(["topic", "family_short"]).size().unstack(fill_value=0).reindex(TOPIC_ORDER, fill_value=0)
    fam_counts.plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_title("Bot comments by family inside thread topics")
    axes[1].set_xlabel("topic")
    axes[1].set_ylabel("bot comments")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_figure(fig, path)


def plot_resource_audit(audit: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(audit))
    ax.bar(x - 0.18, audit["human_comments"], width=0.36, label="human")
    ax.bar(x + 0.18, audit["bot_comments"], width=0.36, label="bot")
    ax.axhline(MIN_TOPIC_CLASS_COUNT, color="red", linestyle="--", linewidth=1, label="min per class")
    ax.set_xticks(x)
    ax.set_xticklabels(audit["topic"], rotation=45, ha="right")
    ax.set_ylabel("comments")
    ax.set_title("Thread topic resource audit")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, path)


def plot_protocol_a(metrics: pd.DataFrame, path: Path) -> None:
    tfidf = metrics[metrics["model"].eq("TF-IDF+LR")].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["global balanced", "thread-topic matched"]
    vals = [tfidf["baseline_global_balanced_bot_recall"], tfidf["topic_matched_bot_recall"]]
    ax.bar(names, vals, color=["#4C78A8", "#F58518"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("bot recall")
    ax.set_title("TF-IDF bot recall before/after thread-topic control")
    for i, v in enumerate(vals):
        ax.text(i, min(float(v) + 0.03, 1.02), f"{float(v):.3f}", ha="center")
    fig.tight_layout()
    save_figure(fig, path)


def plot_tfidf_recall_by_thread_topic(predictions: pd.DataFrame, dataset: pd.DataFrame, csv_path: Path, figure_path: Path) -> pd.DataFrame:
    pred = predictions[predictions["protocol"].eq("topic_matched_aggregate")].drop(columns=["topic"], errors="ignore")
    if pred.empty:
        out = pd.DataFrame(columns=["display_topic", "topic", "bot_recall", "bot_test_n"])
    else:
        topic_lookup = dataset[["sample_id", "topic"]]
        merged = pred.merge(topic_lookup, on="sample_id", how="left")
        bot = merged[merged["label"].eq(1)].copy()
        overall = float(bot["y_pred"].eq(1).mean()) if len(bot) else np.nan
        rows = [
            {
                "display_topic": "Overall",
                "topic": "__overall__",
                "bot_recall": overall,
                "bot_test_n": int(len(bot)),
            }
        ]
        labels = {
            "israel_palestine_middle_east": "Israel/Palestine",
            "russia_ukraine_war": "Russia/Ukraine",
            "us_politics_elections": "US politics",
            "law_crime_rights": "Law/rights",
            "other_or_mixed": "Other/mixed",
        }
        for topic in ["israel_palestine_middle_east", "russia_ukraine_war", "us_politics_elections", "law_crime_rights", "other_or_mixed"]:
            part = bot[bot["topic"].eq(topic)]
            rows.append(
                {
                    "display_topic": labels[topic],
                    "topic": topic,
                    "bot_recall": float(part["y_pred"].eq(1).mean()) if len(part) else np.nan,
                    "bot_test_n": int(len(part)),
                }
            )
        out = pd.DataFrame(rows)
    save_table(out, csv_path)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    if out.empty:
        ax.text(0.5, 0.5, "No topic-matched TF-IDF predictions", ha="center", va="center")
        ax.axis("off")
    else:
        overall = float(out.loc[out["topic"].eq("__overall__"), "bot_recall"].iloc[0])
        bars = ax.bar(out["display_topic"], out["bot_recall"], color=["#333333"] + ["#4C78A8"] * (len(out) - 1))
        ax.axhline(overall, color="#D62728", linestyle="--", linewidth=1.5, label=f"Overall = {overall:.3f}")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("TF-IDF bot recall")
        ax.set_xlabel("Thread-derived topic")
        ax.set_title("TF-IDF detection under thread-topic-matched control")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        for bar, rec, n in zip(bars, out["bot_recall"], out["bot_test_n"]):
            if pd.isna(rec):
                continue
            inside = float(rec) >= 0.82
            y = float(rec) - 0.12 if inside else float(rec) + 0.035
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{float(rec):.3f}\nn={int(n)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="white" if inside else "black",
                fontweight="semibold" if inside else "normal",
            )
        ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, figure_path)
    return out


def plot_loto(metrics: pd.DataFrame, path: Path) -> None:
    df = metrics[(metrics["model"].eq("TF-IDF+LR")) & (metrics["status"].eq("ok"))].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    if df.empty:
        ax.text(0.5, 0.5, "No eligible leave-one-topic-out runs", ha="center", va="center")
        ax.axis("off")
    else:
        ax.bar(df["heldout_topic"], df["bot_recall"], color="#54A24B")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("bot recall")
        ax.set_title("Leave-one-topic-out TF-IDF bot recall")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_figure(fig, path)


def plot_lpo(metrics: pd.DataFrame, path: Path) -> None:
    df = metrics[metrics["model"].eq("TF-IDF+LR")].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ok = df[df["status"].eq("ok")]
    if ok.empty:
        ax.text(0.5, 0.5, "No eligible leave-posts-out runs", ha="center", va="center")
        ax.axis("off")
    else:
        ax.bar(ok["topic"], ok["bot_recall"], color="#B279A2")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("bot recall")
        ax.set_title("Leave-posts-out within topic TF-IDF bot recall")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_figure(fig, path)


def fmt(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def md_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> list[str]:
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return ["空表"]
    return ["```text", df.to_string(index=False), "```"]


def write_report(
    path: Path,
    post_map: pd.DataFrame,
    dataset: pd.DataFrame,
    audit: pd.DataFrame,
    build_stats: dict[str, int],
    protocol_a: pd.DataFrame,
    protocol_b: pd.DataFrame,
    protocol_c: pd.DataFrame,
    topic_recall: pd.DataFrame,
) -> None:
    tfidf_a = protocol_a[protocol_a["model"].eq("TF-IDF+LR")].iloc[0]
    loto_ok = protocol_b[(protocol_b["model"].eq("TF-IDF+LR")) & (protocol_b["status"].eq("ok"))]
    lpo_ok = protocol_c[(protocol_c["model"].eq("TF-IDF+LR")) & (protocol_c["status"].eq("ok"))]
    lines = [
        "# Thread-Derived Topic-Controlled Detection 报告",
        "",
        "## 这个实验重做了什么",
        "",
        "旧的 `Topic-Controlled Detection` 使用 BGE embedding 对评论聚类。这个做法适合探索语义簇，但不能严格等同于 topic control，因为 BGE cluster 可能同时混入话题、语言风格、人机差异和 family 差异。",
        "",
        "本实验改为使用 Reddit 的帖子结构：先只根据帖子 `posts` 字段和 `subreddit` 给帖子归类，再把同一话题下多个帖子的 human/bot 评论聚合起来检测。topic 分类没有读取评论正文，也没有读取 label 或 family。",
        "",
        "## 数据构建审计",
        "",
        f"- 帖子 topic map：`{len(post_map)}` 个唯一 `link_id`。",
        f"- human 原始评论：总计 `{build_stats['human_total_comments']}`，可映射到帖子 topic `{build_stats['human_mapped_comments']}`，未映射 `{build_stats['human_unmapped_comments']}`。",
        f"- bot 生成评论：总计 `{build_stats['bot_total_comments']}`，可映射到帖子 topic `{build_stats['bot_mapped_comments']}`，未映射 `{build_stats['bot_unmapped_comments']}`。",
        f"- 最终可用 comment-topic 样本：`{len(dataset)}`。",
        "",
        "## Topic 资源表",
        "",
        *md_table(
            audit,
            [
                "topic",
                "n_posts",
                "human_comments",
                "bot_comments",
                "family_count_F1",
                "family_count_F2",
                "family_count_F3",
                "family_count_F4",
                "family_count_F5",
                "eligible_for_matched",
                "eligible_for_leave_posts_out",
                "skip_reason",
            ],
        ),
        "",
        "## Protocol A: Topic-Matched Aggregate Detection",
        "",
        "问题：在每个大话题内聚合多个帖子，并在每个 topic 内抽取等量 human/bot 后，TF-IDF 检测器还能否识别人机文本？",
        "",
        *md_table(
            protocol_a,
            [
                "model",
                "status",
                "baseline_total_retained",
                "matched_total_retained",
                "baseline_global_balanced_bot_recall",
                "topic_matched_bot_recall",
                "bot_recall_drop_vs_global_balanced",
                "topic_matched_human_recall",
                "topic_matched_macro_f1",
            ],
        ),
        "",
        f"- TF-IDF 全局随机平衡 baseline bot recall = `{fmt(tfidf_a['baseline_global_balanced_bot_recall'])}`。",
        f"- TF-IDF 帖子话题控制后 bot recall = `{fmt(tfidf_a['topic_matched_bot_recall'])}`。",
        f"- 二者差值 = `{fmt(tfidf_a['bot_recall_drop_vs_global_balanced'])}`。正值表示控制话题后检出下降，说明原检测可能利用了部分话题分布差异；接近 0 表示 topic control 后仍保留较强文本识别能力。",
        "",
        "### Topic 内检出率直观对比",
        "",
        "下表和图把 `Overall` 与各个可用 topic 的 TF-IDF bot recall 放在同一个横轴上。若每个 topic 都接近 Overall，说明粗粒度帖子话题不是主要影响源。",
        "",
        *md_table(topic_recall, ["display_topic", "bot_recall", "bot_test_n"]),
        "",
        "## Protocol B: Leave-One-Topic-Out",
        "",
        "问题：如果一个大话题完全不进入训练集，检测器能否泛化到这个未见 topic？",
        "",
        *md_table(
            protocol_b,
            ["model", "heldout_topic", "status", "train_n", "test_n", "test_human_n", "test_bot_n", "human_recall", "bot_recall", "macro_f1"],
        ),
        "",
        f"- 可运行 held-out topic 数：`{len(loto_ok)}`。",
        f"- 可运行 topic 的平均 bot recall：`{fmt(loto_ok['bot_recall'].mean() if not loto_ok.empty else np.nan)}`。",
        "",
        "## Protocol C: Leave-Posts-Out Within Topic",
        "",
        "问题：在同一大话题内，训练帖子和测试帖子完全不重叠时，检测器是否仍能识别？这个协议最严格，但也最容易受样本不足限制。",
        "",
        *md_table(
            protocol_c,
            ["model", "topic", "status", "train_posts", "test_posts", "train_n", "test_n", "human_recall", "bot_recall", "macro_f1"],
        ),
        "",
        f"- 可运行 leave-posts-out topic 数：`{len(lpo_ok)}`。",
        f"- 可运行 topic 的平均 bot recall：`{fmt(lpo_ok['bot_recall'].mean() if not lpo_ok.empty else np.nan)}`。",
        "",
        "## 图表",
        "",
        "![thread topic distribution](../figures/thread_topic_distribution_by_label_family.png)",
        "",
        "![thread topic resource audit](../figures/thread_topic_resource_audit.png)",
        "",
        "![topic matched recall drop](../figures/thread_topic_matched_recall_drop.png)",
        "",
        "![tfidf recall by topic](../figures/thread_topic_tfidf_recall_by_topic_overall.png)",
        "",
        "![leave one topic out recall](../figures/thread_leave_one_topic_out_recall.png)",
        "",
        "![leave posts out recall](../figures/thread_leave_posts_out_within_topic_recall.png)",
        "",
        "## 结论边界",
        "",
        "- 这个实验比 BGE 聚类版更适合写成正式 topic control，因为 topic 来自帖子而不是评论 embedding。",
        "- 由于 bot 生成评论集中在 36 个候选帖子上，严格同帖 human/bot 平衡不可行；本实验采用同一大话题下多帖聚合，这是当前数据条件下更稳妥的控制方式。",
        "- BGE comment-level 检测本轮没有强行运行，因为当前没有对应的 comment-level 本地 embedding；报告中保留 `bge_skipped_no_comment_level_local_embeddings` 状态，避免把不可比的 user-level BGE 向量混入评论级实验。",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_detailed_report() -> None:
    report_path = RESULTS_DIR / "STATIC_TEXT_EXTENDED_DETAILED_ANALYSIS_CN.md"
    if not report_path.exists():
        return
    text = report_path.read_text(encoding="utf-8")
    text = text.replace("实验二：Topic-Controlled Detection", "实验二：Semantic-Cluster-Controlled Detection")
    text = text.replace("- Topic-Controlled Detection", "- Semantic-Cluster-Controlled Detection")
    text = re.sub(
        r"\n## 新增实验：Thread-Derived (?:Semantic-Cluster-Controlled|Topic-Controlled) Detection\n"
        r".*?(?=\n## 新增实验：Thread-Derived |\Z)",
        "\n",
        text,
        flags=re.S,
    )
    addition = f"""
## 新增实验：Thread-Derived Topic-Controlled Detection

旧的 BGE 聚类实验只能说明“语义簇控制”下的结果，不能作为严格 topic control。新的正式实验已经改为帖子级 topic：只用帖子 `posts` 字段和 `subreddit` 做规则归类，再在同一 topic 下聚合多个帖子并抽取等量 human/bot 评论。

关键产物：

- `results/thread_topic_controlled/post_topic_map.csv`
- `results/thread_topic_controlled/comment_topic_dataset.csv`
- `results/thread_topic_controlled/topic_resource_audit.csv`
- `results/thread_topic_controlled/topic_matched_detection_metrics.csv`
- `results/thread_topic_controlled/leave_one_topic_out_metrics.csv`
- `results/thread_topic_controlled/leave_posts_out_within_topic_metrics.csv`
- `results/thread_topic_controlled/THREAD_TOPIC_CONTROLLED_REPORT_CN.md`

正式论文或汇报中的 topic-control 结论应优先引用这个新实验；原 BGE 聚类实验保留为探索性补充。
"""
    text = text.rstrip() + "\n" + addition
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    research_cli(globals(), "text_evaluation")
    ensure_dir(THREAD_RESULTS_DIR)
    ensure_dir(FIGURES_DIR)

    post_map = build_post_topic_map()
    dataset, build_stats = build_comment_topic_dataset(post_map)
    audit = build_resource_audit(dataset)

    protocol_a, pred_a, matched_pool = run_protocol_a(dataset, audit)
    protocol_b, pred_b = run_protocol_b(dataset, audit)
    protocol_c, pred_c = run_protocol_c(dataset, audit)
    predictions = pd.DataFrame(pred_a + pred_b + pred_c)

    save_table(post_map, THREAD_RESULTS_DIR / "post_topic_map.csv")
    save_table(dataset, THREAD_RESULTS_DIR / "comment_topic_dataset.csv")
    save_table(audit, THREAD_RESULTS_DIR / "topic_resource_audit.csv")
    save_table(protocol_a, THREAD_RESULTS_DIR / "topic_matched_detection_metrics.csv")
    save_table(protocol_b, THREAD_RESULTS_DIR / "leave_one_topic_out_metrics.csv")
    save_table(protocol_c, THREAD_RESULTS_DIR / "leave_posts_out_within_topic_metrics.csv")
    save_table(predictions, THREAD_RESULTS_DIR / "thread_topic_controlled_predictions.csv")

    plot_distribution(dataset, FIGURES_DIR / "thread_topic_distribution_by_label_family.png")
    plot_resource_audit(audit, FIGURES_DIR / "thread_topic_resource_audit.png")
    plot_protocol_a(protocol_a, FIGURES_DIR / "thread_topic_matched_recall_drop.png")
    plot_loto(protocol_b, FIGURES_DIR / "thread_leave_one_topic_out_recall.png")
    plot_lpo(protocol_c, FIGURES_DIR / "thread_leave_posts_out_within_topic_recall.png")
    topic_recall = plot_tfidf_recall_by_thread_topic(
        predictions,
        dataset,
        THREAD_RESULTS_DIR / "thread_topic_tfidf_recall_by_topic.csv",
        FIGURES_DIR / "thread_topic_tfidf_recall_by_topic_overall.png",
    )

    write_report(
        THREAD_RESULTS_DIR / "THREAD_TOPIC_CONTROLLED_REPORT_CN.md",
        post_map=post_map,
        dataset=dataset,
        audit=audit,
        build_stats=build_stats,
        protocol_a=protocol_a,
        protocol_b=protocol_b,
        protocol_c=protocol_c,
        topic_recall=topic_recall,
    )
    update_detailed_report()

    print("Thread-derived topic-controlled detection complete")
    print(f"post_map={len(post_map)} comments={len(dataset)} matched_pool={len(matched_pool)}")
    print(f"report={THREAD_RESULTS_DIR / 'THREAD_TOPIC_CONTROLLED_REPORT_CN.md'}")


if __name__ == "__main__":
    main()
