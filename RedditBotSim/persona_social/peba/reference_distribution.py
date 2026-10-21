from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .behavior_taxonomy import BEHAVIOR_CLASSES, classify_from_features
from .peba_metrics import normalize_distribution


BOT_ID_PREFIXES = ("psbot", "family_", "bot_", "ai_", "a_language", "c_graph", "e_mixed", "pevo_")


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _is_bot_id(user_id: Any) -> bool:
    text = str(user_id or "").strip().lower()
    return text.startswith(BOT_ID_PREFIXES) or "_g" in text or "family" in text


def load_reddit_tables(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    project_root = config_path.resolve().parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config.get("paths", {})
    posts = pd.read_csv(_resolve_path(project_root, paths["reddit_posts"]), dtype=str).fillna("")
    c1 = pd.read_csv(_resolve_path(project_root, paths["reddit_comments1"]), dtype=str).fillna("")
    c2 = pd.read_csv(_resolve_path(project_root, paths["reddit_comments2"]), dtype=str).fillna("")
    users = pd.read_csv(_resolve_path(project_root, paths["reddit_users"]), dtype=str).fillna("")
    return posts, c1, c2, users


def _topic_from_text(text: str) -> str:
    lowered = str(text or "").lower()
    topics = {
        "US politics": ["trump", "biden", "republican", "democrat", "election", "vote", "congress"],
        "Russia-Ukraine war": ["ukraine", "russia", "putin", "zelensky", "nato", "kyiv"],
        "Israel-Palestine": ["israel", "gaza", "hamas", "palestinian", "idf", "hostage"],
        "world news": ["global", "country", "government", "minister", "police", "protest", "world"],
    }
    scores = {topic: sum(1 for keyword in keywords if keyword in lowered) for topic, keywords in topics.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "world news"


def build_human_feature_rows(config_path: Path) -> pd.DataFrame:
    posts, c1, c2, users = load_reddit_tables(config_path)
    user_col = "author_id" if "author_id" in users.columns else users.columns[0]
    human_ids = {uid for uid in users[user_col].astype(str) if not _is_bot_id(uid)}
    posts = posts[posts["author_id"].isin(human_ids)].copy()
    c1 = c1[c1["comment_author_id"].isin(human_ids)].copy()
    c2 = c2[c2["comment_author_id"].isin(human_ids)].copy()

    post_author = dict(zip(posts["submission_id"], posts["author_id"]))
    c1_author_all = dict(zip(c1["comment_id"], c1["comment_author_id"]))
    post_subreddit = dict(zip(posts["submission_id"], posts.get("subreddit", "")))
    post_text = dict(zip(posts["submission_id"], posts.get("posts", "")))

    by_user: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "post_count": 0,
        "comment1_count": 0,
        "comment2_count": 0,
        "revisit_count": 0,
        "skip_count": 0,
        "action_events": 0,
        "subreddits": set(),
        "topics": set(),
        "target_authors": set(),
        "inbound_authors": set(),
        "comment_words": [],
    })

    for _, row in posts.iterrows():
        uid = str(row["author_id"])
        item = by_user[uid]
        item["post_count"] += 1
        item["action_events"] += 1
        item["subreddits"].add(str(row.get("subreddit", "")))
        item["topics"].add(_topic_from_text(row.get("posts", "")))

    for _, row in c1.iterrows():
        uid = str(row["comment_author_id"])
        post_id = str(row["link_id"]).replace("t3_", "")
        target_author = post_author.get(post_id, "")
        item = by_user[uid]
        item["comment1_count"] += 1
        item["action_events"] += 1
        item["subreddits"].add(str(row.get("subreddit", post_subreddit.get(post_id, ""))))
        item["topics"].add(_topic_from_text(f"{post_text.get(post_id, '')} {row.get('comment_body', '')}"))
        if target_author:
            item["target_authors"].add(target_author)
            if target_author in human_ids and target_author != uid:
                by_user[target_author]["inbound_authors"].add(uid)
        item["comment_words"].extend(str(row.get("comment_body", "")).split())
        if target_author == uid:
            item["revisit_count"] += 1

    for _, row in c2.iterrows():
        uid = str(row["comment_author_id"])
        parent_id = str(row["parent_id"]).replace("t1_", "")
        target_author = c1_author_all.get(parent_id, "")
        post_id = str(row["link_id"]).replace("t3_", "")
        item = by_user[uid]
        item["comment2_count"] += 1
        item["action_events"] += 1
        item["subreddits"].add(str(row.get("subreddit", post_subreddit.get(post_id, ""))))
        item["topics"].add(_topic_from_text(f"{post_text.get(post_id, '')} {row.get('comment_body', '')}"))
        if target_author:
            item["target_authors"].add(target_author)
            if target_author in human_ids and target_author != uid:
                by_user[target_author]["inbound_authors"].add(uid)
        item["comment_words"].extend(str(row.get("comment_body", "")).split())
        if target_author == uid:
            item["revisit_count"] += 1

    rows = []
    for uid in sorted(human_ids):
        item = by_user[uid]
        target_authors = set(item["target_authors"]) - {uid, ""}
        inbound_authors = set(item["inbound_authors"]) - {uid, ""}
        reciprocal = len(target_authors & inbound_authors)
        comment_count = item["comment1_count"] + item["comment2_count"]
        rows.append(
            {
                "user_id": uid,
                "post_count": item["post_count"],
                "comment1_count": item["comment1_count"],
                "comment2_count": item["comment2_count"],
                "comment_count": comment_count,
                "revisit_count": item["revisit_count"],
                "skip_count": 0,
                "action_events": item["action_events"],
                "unique_subreddits": len(set(item["subreddits"]) - {""}),
                "unique_topics": len(set(item["topics"]) - {""}),
                "unique_target_authors": len(target_authors),
                "unique_inbound_authors": len(inbound_authors),
                "reciprocal_neighbor_count_proxy": reciprocal,
                "mean_comment_words": len(item["comment_words"]) / max(1, comment_count),
                "explore_source_share": 0.0,
                "own_history_source_share": 0.0,
                "notification_source_share": 0.0,
            }
        )
    df = pd.DataFrame(rows)
    decisions = [classify_from_features(row._asdict() if hasattr(row, "_asdict") else row).label for row in df.to_dict(orient="records")]
    df["behavior_label"] = decisions
    return df


def _slice_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    comment_count = df["comment_count"].astype(float)
    post_count = df["post_count"].astype(float)
    c2 = df["comment2_count"].astype(float)
    revisit = df["revisit_count"].astype(float)
    reciprocal = df["reciprocal_neighbor_count_proxy"].astype(float)
    unique_topics = df["unique_topics"].astype(float)
    return {
        "global": pd.Series(True, index=df.index),
        "high_activity": comment_count >= comment_count.quantile(0.75),
        "low_activity": comment_count <= comment_count.quantile(0.35),
        "thread_heavy": c2 >= max(1.0, c2.quantile(0.75)),
        "poster_heavy": post_count >= max(1.0, post_count.quantile(0.75)),
        "cross_topic": unique_topics >= max(2.0, unique_topics.quantile(0.70)),
        "high_revisit": revisit >= max(1.0, revisit.quantile(0.80)),
        "low_reciprocity": reciprocal <= reciprocal.quantile(0.50),
    }


def build_human_reference(config_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = build_human_feature_rows(config_path)
    features.to_csv(out_dir / "human_behavior_features.csv", index=False)
    references: dict[str, Any] = {}
    for slice_name, mask in _slice_masks(features).items():
        sliced = features[mask].copy()
        if sliced.empty:
            sliced = features.copy()
        counts = Counter(sliced["behavior_label"])
        distribution = normalize_distribution(counts, BEHAVIOR_CLASSES, alpha=0.5)
        payload = {
            "slice": slice_name,
            "n_users": int(len(sliced)),
            "behavior_counts": {label: int(counts.get(label, 0)) for label in BEHAVIOR_CLASSES},
            "behavior_distribution": {label: round(float(distribution[label]), 8) for label in BEHAVIOR_CLASSES},
        }
        (out_dir / f"{slice_name}_behavior_distribution.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        references[slice_name] = payload
    (out_dir / "reference_index.json").write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")
    return references


def load_reference(reference_dir: Path, slice_name: str = "global") -> dict[str, Any]:
    path = reference_dir / f"{slice_name}_behavior_distribution.json"
    if not path.exists():
        path = reference_dir / "global_behavior_distribution.json"
    return json.loads(path.read_text(encoding="utf-8"))
