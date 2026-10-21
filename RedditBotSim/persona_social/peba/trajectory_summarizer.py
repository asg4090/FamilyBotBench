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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def summarize_trajectories(iteration_dir: Path, out_path: Path | None = None) -> pd.DataFrame:
    run_dir = iteration_dir / "run"
    family_manifest = _safe_read_csv(iteration_dir / "family_manifest.csv")
    generated_comments = _safe_read_csv(iteration_dir / "generated_comments.csv")
    generated_posts = _safe_read_csv(iteration_dir / "generated_posts.csv")
    graph_proxy = _safe_read_csv(iteration_dir / "graph_proxy_audit.csv")
    decision_rows = [row for row in _read_jsonl(run_dir / "decision_trace.jsonl") if row.get("event") == "decision"]
    browse_rows = [row for row in _read_jsonl(run_dir / "decision_trace.jsonl") if row.get("event") == "browse"]
    llm_rows = [row for row in _read_jsonl(run_dir / "llm_trace.jsonl") if row.get("event") == "llm_result"]

    users = set()
    for df, col in [
        (family_manifest, "user_id"),
        (generated_comments, "user_id"),
        (generated_posts, "user_id"),
        (graph_proxy, "user_id"),
    ]:
        if not df.empty and col in df.columns:
            users.update(df[col].astype(str))
    for row in decision_rows + browse_rows:
        if row.get("user_id"):
            users.add(str(row["user_id"]))

    manifest_by_user = family_manifest.set_index("user_id").to_dict(orient="index") if not family_manifest.empty and "user_id" in family_manifest else {}
    graph_by_user = graph_proxy.set_index("user_id").to_dict(orient="index") if not graph_proxy.empty and "user_id" in graph_proxy else {}
    comments_by_user = generated_comments.groupby("user_id") if not generated_comments.empty and "user_id" in generated_comments else {}
    posts_by_user = generated_posts.groupby("user_id") if not generated_posts.empty and "user_id" in generated_posts else {}
    decisions_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    browses_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    llm_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        decisions_by_user[str(row.get("user_id", ""))].append(row)
    for row in browse_rows:
        browses_by_user[str(row.get("user_id", ""))].append(row)
    for row in llm_rows:
        llm_by_user[str(row.get("user_id", ""))].append(row)

    rows = []
    for user_id in sorted(users):
        user_comments = comments_by_user.get_group(user_id) if hasattr(comments_by_user, "groups") and user_id in comments_by_user.groups else pd.DataFrame()
        user_posts = posts_by_user.get_group(user_id) if hasattr(posts_by_user, "groups") and user_id in posts_by_user.groups else pd.DataFrame()
        decisions = decisions_by_user.get(user_id, [])
        browses = browses_by_user.get(user_id, [])
        action_counts = Counter(str(row.get("action", "")) for row in decisions)
        source_counts = Counter(str(row.get("candidate_source", "")) for row in decisions)
        topics = [str(row.get("topic", "")) for row in decisions if row.get("topic")]
        subreddits = [str(row.get("subreddit", "")) for row in decisions if row.get("subreddit")]
        comment1_count = int((user_comments.get("level", pd.Series(dtype=str)).astype(str) == "1").sum()) if not user_comments.empty else 0
        comment2_count = int((user_comments.get("level", pd.Series(dtype=str)).astype(str) == "2").sum()) if not user_comments.empty else 0
        words = []
        if not user_comments.empty and "comment_body" in user_comments:
            for body in user_comments["comment_body"].astype(str):
                words.extend(body.split())
        fallback_count = sum(1 for row in llm_by_user.get(user_id, []) if row.get("fallback"))
        browse_item_count = sum(len(row.get("browse_items", []) or []) for row in browses)
        manifest = manifest_by_user.get(user_id, {})
        graph = graph_by_user.get(user_id, {})
        target_authors = int(float(graph.get("unique_target_authors", 0) or 0))
        row = {
            "user_id": user_id,
            "family": manifest.get("family", ""),
            "family_generation": manifest.get("family_generation", ""),
            "provider": manifest.get("provider", ""),
            "model": manifest.get("model", ""),
            "temperature": manifest.get("temperature", ""),
            "post_count": int(len(user_posts)),
            "comment1_count": comment1_count,
            "comment2_count": comment2_count,
            "comment_count": comment1_count + comment2_count,
            "revisit_count": int(action_counts.get("revisit", 0)),
            "skip_count": int(action_counts.get("skip", 0)),
            "action_events": int(sum(action_counts.values())),
            "browse_events": int(len(browses)),
            "browse_item_count": int(browse_item_count),
            "unique_subreddits": len(set(item for item in subreddits if item)),
            "unique_topics": len(set(item for item in topics if item)),
            "unique_target_authors": target_authors,
            "unique_inbound_authors": int(float(graph.get("unique_inbound_authors", 0) or 0)),
            "reciprocal_neighbor_count_proxy": int(float(graph.get("reciprocal_neighbor_count_proxy", 0) or 0)),
            "mean_comment_words": len(words) / max(1, comment1_count + comment2_count),
            "fallback_count": fallback_count,
            "fallback_share": fallback_count / max(1, len(llm_by_user.get(user_id, []))),
            "explore_source_share": source_counts.get("explore", 0) / max(1, sum(source_counts.values())),
            "own_history_source_share": source_counts.get("own_history", 0) / max(1, sum(source_counts.values())),
            "notification_source_share": source_counts.get("notification", 0) / max(1, sum(source_counts.values())),
            "action_counts_json": json.dumps(dict(action_counts), ensure_ascii=False, sort_keys=True),
            "source_counts_json": json.dumps(dict(source_counts), ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if out_path is not None:
        df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    return df
