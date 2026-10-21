#!/usr/bin/env python3
"""Replay a small number of human-like replies onto generated bot content.

The replay pass is intentionally conservative. It samples reply shapes and text
from original human comments, creates background replay-human users, and appends
their replies under generated bot posts/comments so generated bots do not have a
trivial zero-inbound graph signature.
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
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


USER_FIELDS = [
    "user_id",
    "user_name",
    "source",
    "label",
    "family",
    "is_generated",
    "is_replay_human",
    "created_by_run",
]

POST_FIELDS = [
    "post_id",
    "author_id",
    "author_source",
    "family",
    "subreddit",
    "topic",
    "created_utc",
    "title",
    "body",
    "score",
    "num_comments",
    "upvote_ratio",
    "is_generated",
    "is_replay_context",
]

COMMENT_FIELDS = [
    "comment_id",
    "author_id",
    "author_source",
    "family",
    "post_id",
    "parent_id",
    "parent_type",
    "level",
    "created_utc",
    "body",
    "score",
    "is_generated",
    "is_replay_human",
    "action_type",
    "target_author_id",
]

LABEL_FIELDS = [
    "node_id",
    "node_type",
    "label",
    "label_known",
    "source",
    "family",
    "split_group",
]

EDGE_FIELDS = [
    "edge_id",
    "src_id",
    "dst_id",
    "src_type",
    "dst_type",
    "edge_type",
    "time",
    "weight",
    "source_event_id",
]

MANIFEST_FIELDS = [
    "replay_comment_id",
    "replay_author_id",
    "target_type",
    "target_id",
    "target_author_id",
    "target_family",
    "post_id",
    "source_pattern_comment_id",
    "source_pattern_author_id",
    "source_pattern_level",
    "source_pattern_body_excerpt",
    "created_utc",
    "seed",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _parse_time(value: str, fallback: datetime) -> datetime:
    if not value:
        return fallback
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.replace(tzinfo=None)
        except ValueError:
            pass
    return fallback


def _format_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _excerpt(text: str, limit: int = 120) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _load_existing_manifest(path: Path) -> List[Dict[str, str]]:
    rows = _read_csv(path)
    if rows:
        return rows
    return []


def _sample_patterns(comments: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    patterns: List[Dict[str, str]] = []
    for row in comments:
        if _as_bool(row.get("is_generated")) or _as_bool(row.get("is_replay_human")):
            continue
        if row.get("author_source") not in {"original_human", "human", ""}:
            continue
        body = (row.get("body") or "").strip()
        if not body:
            continue
        patterns.append(row)
    return patterns


def _next_replay_user(
    rng: random.Random,
    users: List[Dict[str, str]],
    labels: List[Dict[str, str]],
    source_humans: Sequence[Dict[str, str]],
    run_name: str,
    index: int,
) -> Dict[str, str]:
    source = rng.choice(source_humans) if source_humans else {}
    user_id = f"rh_{run_name}_{index:04d}"
    user = {
        "user_id": user_id,
        "user_name": f"replay_{source.get('user_name') or source.get('user_id') or index}",
        "source": "replay_human",
        "label": "-1",
        "family": "human_replay",
        "is_generated": "false",
        "is_replay_human": "true",
        "created_by_run": run_name,
    }
    users.append(user)
    labels.append(
        {
            "node_id": user_id,
            "node_type": "user",
            "label": "-1",
            "label_known": "0",
            "source": "replay_human",
            "family": "human_replay",
            "split_group": "unlabeled_background",
        }
    )
    return user


def apply_human_reply_replay(
    output_dir: Path | str,
    *,
    seed: int = 20260603,
    max_replay_per_bot_post: int = 2,
    max_replay_per_bot_comment: int = 1,
    replay_probability: float = 0.35,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Append replay-human replies to generated bot posts/comments in output_dir."""

    output_path = Path(output_dir)
    rng = random.Random(seed)
    run_name = run_name or output_path.name

    users_path = output_path / "users.csv"
    posts_path = output_path / "posts.csv"
    comments_path = output_path / "comments.csv"
    labels_path = output_path / "labels.csv"
    manifest_path = output_path / "human_replay_manifest.csv"
    edges_path = output_path / "edges.csv"

    users = _read_csv(users_path)
    posts = _read_csv(posts_path)
    comments = _read_csv(comments_path)
    labels = _read_csv(labels_path)
    edges = _read_csv(edges_path)
    existing_manifest = _load_existing_manifest(manifest_path)

    patterns = _sample_patterns(comments)
    if not patterns:
        _write_csv(manifest_path, existing_manifest, MANIFEST_FIELDS)
        return {"created": 0, "reason": "no_original_human_reply_patterns"}

    source_humans = [row for row in users if row.get("source") == "original_human"]
    generated_posts = [row for row in posts if _as_bool(row.get("is_generated")) and row.get("author_source") == "generated_bot"]
    generated_comments = [
        row
        for row in comments
        if _as_bool(row.get("is_generated")) and row.get("author_source") == "generated_bot"
    ]

    post_by_id = {row.get("post_id"): row for row in posts}
    comment_by_id = {row.get("comment_id"): row for row in comments}
    created_rows: List[Dict[str, str]] = []
    jsonl_events: List[Dict[str, Any]] = []
    jsonl_actions: List[Dict[str, Any]] = []
    jsonl_memory: List[Dict[str, Any]] = []

    replay_index = len(existing_manifest) + 1
    edge_index = len(edges) + 1

    def add_edge(
        src_id: str,
        dst_id: str,
        src_type: str,
        dst_type: str,
        edge_type: str,
        created_utc: str,
        source_event_id: str,
    ) -> None:
        nonlocal edge_index
        if not src_id or not dst_id:
            return
        edges.append(
            {
                "edge_id": f"e_hr_{run_name}_{edge_index:08d}",
                "src_id": src_id,
                "dst_id": dst_id,
                "src_type": src_type,
                "dst_type": dst_type,
                "edge_type": edge_type,
                "time": created_utc,
                "weight": "1",
                "source_event_id": source_event_id,
            }
        )
        edge_index += 1

    def create_reply(target_type: str, target: Dict[str, str], per_target_index: int) -> None:
        nonlocal replay_index
        pattern = rng.choice(patterns)
        author = _next_replay_user(rng, users, labels, source_humans, run_name, replay_index)
        target_id = target.get("post_id") if target_type == "post" else target.get("comment_id")
        post_id = target.get("post_id", "")
        parent_type = "post" if target_type == "post" else "comment"
        parent_id = target_id or ""
        target_author_id = target.get("author_id", "")
        target_family = target.get("family", "")
        base_time = _parse_time(target.get("created_utc", ""), datetime(2024, 1, 1, 8, 0, 0))
        created_at = base_time + timedelta(minutes=5 + per_target_index * 11 + rng.randint(0, 9))
        comment_id = f"c_hr_{run_name}_{replay_index:06d}"
        level = "1" if target_type == "post" else "2"
        body = (
            f"[human_replay] In a replayed human reply pattern, this responds to "
            f"{target_family or 'generated'} {target_type} {target_id}: {_excerpt(pattern.get('body', ''), 180)}"
        )
        comment = {
            "comment_id": comment_id,
            "author_id": author["user_id"],
            "author_source": "replay_human",
            "family": "human_replay",
            "post_id": post_id,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "level": level,
            "created_utc": _format_time(created_at),
            "body": body,
            "score": pattern.get("score") or pattern.get("comment_score") or "1",
            "is_generated": "false",
            "is_replay_human": "true",
            "action_type": "human_replay",
            "target_author_id": target_author_id,
        }
        event_id = f"evt_hr_{run_name}_{replay_index:06d}"
        comments.append(comment)
        comment_by_id[comment_id] = comment
        if post_id in post_by_id:
            post_by_id[post_id]["num_comments"] = str(_as_int(post_by_id[post_id].get("num_comments"), 0) + 1)
        add_edge(author["user_id"], comment_id, "user", "comment", "user_writes_comment", comment["created_utc"], event_id)
        if target_type == "post":
            add_edge(comment_id, parent_id, "comment", "post", "comment_replies_post", comment["created_utc"], event_id)
            add_edge(author["user_id"], target_author_id, "user", "user", "user_comments_on_user_post", comment["created_utc"], event_id)
        else:
            add_edge(comment_id, parent_id, "comment", "comment", "comment_replies_comment", comment["created_utc"], event_id)
            add_edge(author["user_id"], target_author_id, "user", "user", "user_replies_to_user_comment", comment["created_utc"], event_id)
        add_edge(target_author_id, author["user_id"], "user", "user", "user_receives_reply", comment["created_utc"], event_id)

        manifest = {
            "replay_comment_id": comment_id,
            "replay_author_id": author["user_id"],
            "target_type": target_type,
            "target_id": target_id or "",
            "target_author_id": target_author_id,
            "target_family": target_family,
            "post_id": post_id,
            "source_pattern_comment_id": pattern.get("comment_id", ""),
            "source_pattern_author_id": pattern.get("author_id", ""),
            "source_pattern_level": pattern.get("level", ""),
            "source_pattern_body_excerpt": _excerpt(pattern.get("body", ""), 180),
            "created_utc": comment["created_utc"],
            "seed": str(seed),
        }
        existing_manifest.append(manifest)
        created_rows.append(manifest)
        jsonl_events.append(
            {
                "event_id": event_id,
                "event_type": "human_replay_reply",
                "time": comment["created_utc"],
                "actor_user_id": author["user_id"],
                "actor_family": "human_replay",
                "target_post_id": post_id,
                "target_comment_id": target_id if target_type == "comment" else "",
                "target_author_id": target_author_id,
                "content_id": comment_id,
                "policy_reason": "sampled from original human reply pattern",
                "llm_call_id": "",
            }
        )
        jsonl_events.append(
            {
                "event_id": f"{event_id}_notify",
                "event_type": "notification_created",
                "time": comment["created_utc"],
                "actor_user_id": target_author_id,
                "actor_family": target_family,
                "target_post_id": post_id,
                "target_comment_id": comment_id,
                "target_author_id": author["user_id"],
                "content_id": comment_id,
                "policy_reason": "replay human replied to generated bot content",
                "llm_call_id": "",
            }
        )
        jsonl_actions.append(
            {
                "event_id": event_id,
                "time": comment["created_utc"],
                "actor_user_id": author["user_id"],
                "actor_family": "human_replay",
                "action_type": "human_replay",
                "target_type": target_type,
                "target_id": target_id,
                "content_id": comment_id,
                "dry_run": True,
                "policy_reason": "human reply replay",
            }
        )
        jsonl_memory.append(
            {
                "event_id": event_id,
                "time": comment["created_utc"],
                "actor_user_id": author["user_id"],
                "actor_family": "human_replay",
                "memory_type": "replay_source",
                "source_pattern_comment_id": pattern.get("comment_id", ""),
                "target_type": target_type,
                "target_id": target_id,
            }
        )
        replay_index += 1

    for post in generated_posts:
        if rng.random() > replay_probability:
            continue
        for idx in range(rng.randint(1, max(1, max_replay_per_bot_post))):
            create_reply("post", post, idx)

    for comment in generated_comments:
        if rng.random() > replay_probability:
            continue
        for idx in range(rng.randint(1, max(1, max_replay_per_bot_comment))):
            create_reply("comment", comment, idx)

    _write_csv(users_path, users, USER_FIELDS)
    _write_csv(posts_path, posts, POST_FIELDS)
    _write_csv(comments_path, comments, COMMENT_FIELDS)
    _write_csv(labels_path, labels, LABEL_FIELDS)
    _write_csv(edges_path, edges, EDGE_FIELDS)
    _write_csv(manifest_path, existing_manifest, MANIFEST_FIELDS)
    _append_jsonl(output_path / "events.jsonl", jsonl_events)
    _append_jsonl(output_path / "action_trace.jsonl", jsonl_actions)
    _append_jsonl(output_path / "memory_trace.jsonl", jsonl_memory)

    return {"created": len(created_rows), "manifest": str(manifest_path)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay sampled human replies onto generated bot content.")
    parser.add_argument("--output-dir", required=True, help="Existing mixed runner output directory.")
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--max-replay-per-bot-post", type=int, default=2)
    parser.add_argument("--max-replay-per-bot-comment", type=int, default=1)
    parser.add_argument("--replay-probability", type=float, default=0.35)
    args = parser.parse_args(argv)

    result = apply_human_reply_replay(
        args.output_dir,
        seed=args.seed,
        max_replay_per_bot_post=args.max_replay_per_bot_post,
        max_replay_per_bot_comment=args.max_replay_per_bot_comment,
        replay_probability=args.replay_probability,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
