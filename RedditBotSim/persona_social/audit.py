from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def read_bot_outputs(run_dir: Path, bot_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    posts = pd.read_csv(run_dir / "data" / "Posts.csv", dtype=str).fillna("")
    c1 = pd.read_csv(run_dir / "data" / "Comment_1.csv", dtype=str).fillna("")
    c2 = pd.read_csv(run_dir / "data" / "Comment_2.csv", dtype=str).fillna("")
    return (
        posts[posts["author_id"].isin(bot_ids)].copy(),
        c1[c1["comment_author_id"].isin(bot_ids)].copy(),
        c2[c2["comment_author_id"].isin(bot_ids)].copy(),
    )


def export_generated_comments(run_dir: Path, out_path: Path, bot_ids: set[str]) -> pd.DataFrame:
    _, c1, c2 = read_bot_outputs(run_dir, bot_ids)
    rows: list[dict[str, Any]] = []
    for _, row in c1.iterrows():
        rows.append(
            {
                "level": 1,
                "user_id": row["comment_author_id"],
                "user_name": row["comment_author_name"],
                "comment_id": row["comment_id"],
                "link_id": row["link_id"],
                "parent_id": row["parent_id"],
                "subreddit": row["subreddit"],
                "created_utc": row["created_utc"],
                "comment_body": row["comment_body"],
            }
        )
    for _, row in c2.iterrows():
        rows.append(
            {
                "level": 2,
                "user_id": row["comment_author_id"],
                "user_name": row["comment_author_name"],
                "comment_id": row["comment_id"],
                "link_id": row["link_id"],
                "parent_id": row["parent_id"],
                "subreddit": row["subreddit"],
                "created_utc": row["created_utc"],
                "comment_body": row["comment_body"],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["user_id", "created_utc", "level", "comment_id"])
    df.to_csv(out_path, index=False)
    return df


def export_generated_posts(run_dir: Path, out_path: Path, bot_ids: set[str]) -> pd.DataFrame:
    posts, _, _ = read_bot_outputs(run_dir, bot_ids)
    rows: list[dict[str, Any]] = []
    for _, row in posts.iterrows():
        rows.append(
            {
                "user_id": row["author_id"],
                "user_name": row["author_name"],
                "post_id": row["submission_id"],
                "subreddit": row["subreddit"],
                "created_utc": row["created_utc"],
                "score": row["score"],
                "num_comments": row["num_comments"],
                "upvote_ratio": row["upvote_ratio"],
                "post_body": row["posts"],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["user_id", "created_utc", "post_id"])
    df.to_csv(out_path, index=False)
    return df


def build_behavior_audit(run_dir: Path, bot_ids: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    trace_path = run_dir / "decision_trace.jsonl"
    decision_rows = []
    if trace_path.exists():
        import json

        with trace_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") == "decision":
                    decision_rows.append(row)
    decisions = pd.DataFrame(decision_rows)

    bot_posts, bot_c1, bot_c2 = read_bot_outputs(run_dir, bot_ids)
    all_posts = pd.read_csv(run_dir / "data" / "Posts.csv", dtype=str).fillna("")
    all_c1 = pd.read_csv(run_dir / "data" / "Comment_1.csv", dtype=str).fillna("")
    all_c2 = pd.read_csv(run_dir / "data" / "Comment_2.csv", dtype=str).fillna("")
    post_author = dict(zip(all_posts["submission_id"], all_posts["author_id"]))
    c1_author = dict(zip(all_c1["comment_id"], all_c1["comment_author_id"]))

    action_counts = Counter(decisions["action"]) if not decisions.empty and "action" in decisions else Counter()
    source_counts = Counter(decisions["candidate_source"]) if not decisions.empty and "candidate_source" in decisions else Counter()
    notification_actions = int((decisions["action"] == "revisit").sum()) if not decisions.empty and "action" in decisions else 0
    browse_mean = float(decisions["browse_count"].astype(float).mean()) if not decisions.empty and "browse_count" in decisions else 0.0
    bot_to_bot_c1 = 0
    bot_to_human_c1 = 0
    for _, row in bot_c1.iterrows():
        target_author = post_author.get(row["link_id"].replace("t3_", ""), "")
        if target_author in bot_ids:
            bot_to_bot_c1 += 1
        else:
            bot_to_human_c1 += 1
    bot_to_bot_c2 = 0
    bot_to_human_c2 = 0
    for _, row in bot_c2.iterrows():
        target_author = c1_author.get(row["parent_id"].replace("t1_", ""), "")
        if target_author in bot_ids:
            bot_to_bot_c2 += 1
        else:
            bot_to_human_c2 += 1

    per_user_rows = []
    for bot_id in sorted(bot_ids):
        user_posts = bot_posts[bot_posts["author_id"] == bot_id]
        user_c1 = bot_c1[bot_c1["comment_author_id"] == bot_id]
        user_c2 = bot_c2[bot_c2["comment_author_id"] == bot_id]
        body_words = []
        for body in list(user_c1["comment_body"]) + list(user_c2["comment_body"]):
            body_words.extend(str(body).split())
        per_user_rows.append(
            {
                "user_id": bot_id,
                "post_count": int(len(user_posts)),
                "comment1_count": int(len(user_c1)),
                "comment2_count": int(len(user_c2)),
                "comment_count": int(len(user_c1) + len(user_c2)),
                "mean_comment_words": float(len(body_words) / max(1, len(user_c1) + len(user_c2))),
            }
        )
    per_user = pd.DataFrame(per_user_rows)

    summary = {
        "bot_count": len(bot_ids),
        "generated_post_count": int(len(bot_posts)),
        "generated_comment1_count": int(len(bot_c1)),
        "generated_comment2_count": int(len(bot_c2)),
        "generated_comment_count": int(len(bot_c1) + len(bot_c2)),
        "action_counts": dict(action_counts),
        "candidate_source_counts": dict(source_counts),
        "bot_to_bot_c1": bot_to_bot_c1,
        "bot_to_human_c1": bot_to_human_c1,
        "bot_to_bot_c2": bot_to_bot_c2,
        "bot_to_human_c2": bot_to_human_c2,
        "bot_to_bot_total": bot_to_bot_c1 + bot_to_bot_c2,
        "bot_to_human_total": bot_to_human_c1 + bot_to_human_c2,
        "notification_revisit_count": notification_actions,
        "mean_browse_count": browse_mean,
        "mean_posts_per_bot": float(per_user["post_count"].mean()) if not per_user.empty else 0.0,
        "mean_comments_per_bot": float(per_user["comment_count"].mean()) if not per_user.empty else 0.0,
        "mean_comment_words": float(per_user["mean_comment_words"].mean()) if not per_user.empty else 0.0,
    }
    return per_user, summary


def graph_proxy_audit(run_dir: Path, bot_ids: set[str]) -> pd.DataFrame:
    posts = pd.read_csv(run_dir / "data" / "Posts.csv", dtype=str).fillna("")
    c1 = pd.read_csv(run_dir / "data" / "Comment_1.csv", dtype=str).fillna("")
    c2 = pd.read_csv(run_dir / "data" / "Comment_2.csv", dtype=str).fillna("")
    post_author = dict(zip(posts["submission_id"], posts["author_id"]))
    c1_author = dict(zip(c1["comment_id"], c1["comment_author_id"]))

    rows = []
    for bot_id in sorted(bot_ids):
        bot_c1 = c1[c1["comment_author_id"] == bot_id]
        bot_c2 = c2[c2["comment_author_id"] == bot_id]
        c1_out = len(bot_c1)
        c2_out = len(bot_c2)
        c2_c1_in = int((c2["parent_id"].str.replace("t1_", "", regex=False).map(c1_author) == bot_id).sum())
        c1_p_in = int((c1["link_id"].str.replace("t3_", "", regex=False).map(post_author) == bot_id).sum())
        target_authors = set()
        for _, row in bot_c1.iterrows():
            target_authors.add(post_author.get(row["link_id"].replace("t3_", ""), ""))
        for _, row in bot_c2.iterrows():
            target_authors.add(c1_author.get(row["parent_id"].replace("t1_", ""), ""))
        inbound_authors = set(c1[c1["link_id"].str.replace("t3_", "", regex=False).map(post_author) == bot_id]["comment_author_id"])
        inbound_authors.update(c2[c2["parent_id"].str.replace("t1_", "", regex=False).map(c1_author) == bot_id]["comment_author_id"])
        reciprocal = len((target_authors & inbound_authors) - {bot_id})
        out_total = c1_out + c2_out
        in_total = c1_p_in + c2_c1_in
        rows.append(
            {
                "user_id": bot_id,
                "C1_P_out_weight_proxy": c1_out,
                "C2_C1_out_weight_proxy": c2_out,
                "C1_P_in_weight_proxy": c1_p_in,
                "C2_C1_in_weight_proxy": c2_c1_in,
                "total_out_weight_proxy": out_total,
                "total_in_weight_proxy": in_total,
                "direction_balance_proxy": out_total / max(1, out_total + in_total),
                "reciprocal_neighbor_count_proxy": reciprocal,
                "reciprocal_share_proxy": reciprocal / max(1, len(target_authors | inbound_authors)),
                "unique_target_authors": len(target_authors - {bot_id}),
                "unique_inbound_authors": len(inbound_authors - {bot_id}),
            }
        )
    return pd.DataFrame(rows)
