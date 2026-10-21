#!/usr/bin/env python3
"""Build a G4 slot-conditioned human-to-bot derived dataset.

G4 does not copy a human user's full trajectory. Each G4 bot is bound to a
moderately active original-human slot, browses that slot's local environment,
generates autonomous actions, and receives sampled original-human replies only
after it has produced a post/comment target.
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
import json
import math
import random
import shutil
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FAMILY = "G4_slot_conditioned"
FAMILY_LABEL = "G4"
DEFAULT_SEED = 20260606


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--g4-count", type=int, default=50)
    parser.add_argument("--time-windows", type=int, default=8)
    parser.add_argument("--browse-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--human-response-probability", type=float, default=0.55)
    parser.add_argument("--max-human-responses-per-action", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_clean_output(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("output-dir must be different from input-dir")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output-dir already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(input_dir, output_dir)
    for stale in [output_dir / "graph", output_dir / "splits"]:
        if stale.exists():
            shutil.rmtree(stale)
    stale_edges = output_dir / "edges.csv"
    if stale_edges.exists():
        stale_edges.unlink()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=None)
    except ValueError:
        return None


def fmt_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def text_excerpt(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def topic_from_post(post: pd.Series | dict[str, Any] | None) -> str:
    if post is None:
        return "general"
    get = post.get if isinstance(post, dict) else post.get
    return str(get("topic", "") or get("subreddit", "") or "general")


def compute_human_slot_metrics(users: pd.DataFrame, posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    human_ids = set(users.loc[users["family"].eq("original_human"), "user_id"])
    post_author = posts.set_index("post_id")["author_id"].to_dict() if not posts.empty else {}

    authored_posts = posts[posts["author_id"].isin(human_ids)].groupby("author_id").size()
    authored_comments = comments[comments["author_id"].isin(human_ids)].groupby("author_id").size()

    comments2 = comments.copy()
    comments2["post_author_id"] = comments2["post_id"].map(post_author).fillna("")
    outbound = comments2[
        comments2["author_id"].isin(human_ids)
        & comments2["target_author_id"].ne("")
        & comments2["author_id"].ne(comments2["target_author_id"])
    ].groupby("author_id").size()
    inbound_target = comments2[
        comments2["target_author_id"].isin(human_ids)
        & comments2["author_id"].ne(comments2["target_author_id"])
    ].groupby("target_author_id").size()
    inbound_post = comments2[
        comments2["post_author_id"].isin(human_ids)
        & comments2["author_id"].ne(comments2["post_author_id"])
    ].groupby("post_author_id").size()

    rows = []
    for user_id in sorted(human_ids):
        ap = int(authored_posts.get(user_id, 0))
        ac = int(authored_comments.get(user_id, 0))
        out = int(outbound.get(user_id, 0))
        inbound = int(inbound_target.get(user_id, 0)) + int(inbound_post.get(user_id, 0))
        rows.append(
            {
                "source_human_slot_id": user_id,
                "authored_posts": ap,
                "authored_comments": ac,
                "outbound_interactions": out,
                "inbound_opportunities": inbound,
                "total_activity": ap + ac,
                "slot_score": math.sqrt(max(1, ap + ac)) + math.sqrt(max(1, out)) + math.sqrt(max(1, inbound)),
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("No original_human users found for slot selection")

    qualified = metrics[
        (metrics["total_activity"] > 0)
        & (metrics["outbound_interactions"] > 0)
        & (metrics["inbound_opportunities"] > 0)
    ].copy()
    if len(qualified) < 10:
        qualified = metrics[(metrics["total_activity"] > 0) & (metrics["inbound_opportunities"] > 0)].copy()
    if len(qualified) < 10:
        qualified = metrics[metrics["total_activity"] > 0].copy()

    if len(qualified) > 20:
        low = qualified["total_activity"].quantile(0.20)
        high = qualified["total_activity"].quantile(0.90)
        trimmed = qualified[(qualified["total_activity"] >= low) & (qualified["total_activity"] <= high)].copy()
        if len(trimmed) >= 10:
            qualified = trimmed
    return qualified.sort_values(["slot_score", "total_activity"], ascending=[False, True]).reset_index(drop=True)


def build_response_bank(posts: pd.DataFrame, comments: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    slot_ids = set(slots["source_human_slot_id"])
    post_author = posts.set_index("post_id")["author_id"].to_dict() if not posts.empty else {}
    post_by_id = posts.set_index("post_id").to_dict("index") if not posts.empty else {}
    rows = []
    seen: set[str] = set()
    for row in comments.itertuples(index=False):
        comment = row._asdict()
        author_id = str(comment.get("author_id", ""))
        if str(comment.get("author_source", "")) != "original_human":
            continue
        if not str(comment.get("body", "")).strip():
            continue
        post_id = str(comment.get("post_id", ""))
        target_author_id = str(comment.get("target_author_id", ""))
        post_owner = str(post_author.get(post_id, ""))
        target_slots = set()
        if target_author_id in slot_ids and target_author_id != author_id:
            target_slots.add(target_author_id)
        if post_owner in slot_ids and post_owner != author_id:
            target_slots.add(post_owner)
        if not target_slots:
            continue
        for slot_id in target_slots:
            key = f"{slot_id}:{comment.get('comment_id', '')}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source_human_slot_id": slot_id,
                    "source_comment_id": str(comment.get("comment_id", "")),
                    "responder_user_id": author_id,
                    "source_post_id": post_id,
                    "source_parent_id": str(comment.get("parent_id", "")),
                    "source_parent_type": str(comment.get("parent_type", "")),
                    "source_level": str(comment.get("level", "")),
                    "source_created_utc": str(comment.get("created_utc", "")),
                    "source_topic": topic_from_post(post_by_id.get(post_id, {})),
                    "source_body": str(comment.get("body", "")),
                }
            )
    return pd.DataFrame(rows)


def sample_slots(slot_metrics: pd.DataFrame, count: int, rng: random.Random) -> list[dict[str, Any]]:
    shuffled = slot_metrics.copy()
    shuffled["_jitter"] = [rng.random() for _ in range(len(shuffled))]
    shuffled = shuffled.sort_values(["slot_score", "_jitter"], ascending=[False, True])
    if len(shuffled) >= count:
        selected = shuffled.head(count)
    else:
        selected = shuffled.sample(n=count, replace=True, random_state=rng.randint(1, 10**9))
    return selected.drop(columns=["_jitter"], errors="ignore").to_dict("records")


def next_base_time(posts: pd.DataFrame, comments: pd.DataFrame) -> datetime:
    times: list[datetime] = []
    for series in [posts.get("created_utc", pd.Series(dtype=str)), comments.get("created_utc", pd.Series(dtype=str))]:
        for value in series.tail(5000):
            parsed = parse_time(value)
            if parsed:
                times.append(parsed)
    if not times:
        return datetime(2024, 1, 6, 8, 0, 0)
    return max(times) + timedelta(hours=2)


def make_profile(bot_id: str, slot: dict[str, Any], rng: random.Random, index: int) -> dict[str, Any]:
    post_base = 0.16 + min(0.22, safe_int(slot.get("authored_posts")) / max(20.0, safe_int(slot.get("total_activity"), 1)))
    reply_base = 0.34 + min(0.24, safe_int(slot.get("outbound_interactions")) / max(60.0, safe_int(slot.get("total_activity"), 1)))
    revisit = 0.10 + min(0.22, safe_int(slot.get("inbound_opportunities")) / 80.0)
    provider = "openai" if index % 2 == 0 else "gemini"
    model = "gpt-5.4" if provider == "openai" else "gemini-3.5-flash"
    temp = 0.4 if index % 4 in {0, 1} else 0.8
    topics = ["politics", "worldnews", "news", "international", "society"]
    rng.shuffle(topics)
    return {
        "user_id": bot_id,
        "family": FAMILY,
        "profile_source": "slot_conditioned_policy",
        "anchor_user_ids": str(slot.get("source_human_slot_id", "")),
        "llm_provider": provider,
        "llm_model": model,
        "llm_temperature": f"{temp:.3f}",
        "llm_reasoning_effort": "medium",
        "post_tendency": f"{min(0.55, post_base):.3f}",
        "reply_tendency": f"{min(0.70, reply_base):.3f}",
        "comment1_tendency": f"{0.28 + rng.random() * 0.12:.3f}",
        "comment2_tendency": f"{0.18 + rng.random() * 0.16:.3f}",
        "revisit_tendency": f"{min(0.46, revisit):.3f}",
        "skip_tendency": f"{0.12 + rng.random() * 0.16:.3f}",
        "exploration_tendency": f"{0.25 + rng.random() * 0.30:.3f}",
        "notification_reactivity": f"{min(0.55, revisit + 0.08):.3f}",
        "topic_entropy_target": f"{0.45 + rng.random() * 0.30:.3f}",
        "active_rhythm_type": rng.choice(["steady", "bursty", "late_reply", "thread_returner"]),
        "preferred_topics": "|".join(topics[:2]),
        "context_sensitivity": f"{0.65 + rng.random() * 0.22:.3f}",
        "diversity_preference": f"{0.50 + rng.random() * 0.25:.3f}",
        "memory_decay": f"{0.45 + rng.random() * 0.30:.3f}",
        "language_style_summary": "slot-conditioned, concise social-media commentary with local thread context",
        "behavior_style_summary": "inherits slot visibility and response opportunities, but samples its own actions",
        "metadata_json": json.dumps(
            {
                "g4": True,
                "source_human_slot_id": slot.get("source_human_slot_id", ""),
                "slot_activity": safe_int(slot.get("total_activity")),
                "slot_inbound_opportunities": safe_int(slot.get("inbound_opportunities")),
                "live_llm": False,
            },
            ensure_ascii=False,
        ),
    }


def candidate_summary(candidates: list[dict[str, Any]], limit: int = 10) -> list[dict[str, str]]:
    return [
        {
            "target_type": str(c.get("target_type", "")),
            "target_id": str(c.get("target_id", "")),
            "author_id": str(c.get("author_id", "")),
            "family": str(c.get("family", "")),
            "candidate_type": str(c.get("candidate_type", "")),
        }
        for c in candidates[:limit]
    ]


def build_candidates(
    bot_id: str,
    source_slot: str,
    posts: pd.DataFrame,
    comments: pd.DataFrame,
    response_bank: pd.DataFrame,
    rng: random.Random,
    browse_count: int,
) -> list[dict[str, Any]]:
    post_by_id = posts.set_index("post_id").to_dict("index") if not posts.empty else {}
    slot_bank = response_bank[response_bank["source_human_slot_id"].eq(source_slot)] if not response_bank.empty else pd.DataFrame()
    slot_post_ids = set(slot_bank.get("source_post_id", pd.Series(dtype=str)).astype(str))
    slot_post_ids.update(posts.loc[posts["author_id"].eq(source_slot), "post_id"].astype(str).tolist())
    slot_post_ids.update(comments.loc[comments["author_id"].eq(source_slot), "post_id"].astype(str).tolist())
    neighbor_ids = set(slot_bank.get("responder_user_id", pd.Series(dtype=str)).astype(str))

    candidates: list[dict[str, Any]] = []
    local_posts = posts[posts["post_id"].isin(slot_post_ids) | posts["author_id"].isin(neighbor_ids)].copy()
    bot_posts = posts[posts["is_generated"].astype(str).str.lower().eq("true")].copy()
    exploration_posts = posts[posts["author_source"].eq("original_human")].sample(
        n=min(max(3, browse_count), len(posts[posts["author_source"].eq("original_human")])),
        random_state=rng.randint(1, 10**9),
    ) if not posts.empty and not posts[posts["author_source"].eq("original_human")].empty else pd.DataFrame()

    post_pool = pd.concat([local_posts, bot_posts.tail(300), exploration_posts], ignore_index=True).drop_duplicates("post_id")
    if not post_pool.empty:
        post_pool = post_pool.sample(n=min(len(post_pool), browse_count * 3), random_state=rng.randint(1, 10**9))
        for row in post_pool.itertuples(index=False):
            rec = row._asdict()
            if rec.get("author_id") == bot_id:
                continue
            candidates.append(
                {
                    "target_type": "post",
                    "target_id": rec.get("post_id", ""),
                    "post_id": rec.get("post_id", ""),
                    "comment_id": "",
                    "author_id": rec.get("author_id", ""),
                    "family": rec.get("family", ""),
                    "subreddit": rec.get("subreddit", ""),
                    "topic": rec.get("topic", ""),
                    "excerpt": text_excerpt(f"{rec.get('title', '')} {rec.get('body', '')}"),
                    "candidate_type": "slot_or_mixed_post",
                }
            )

    local_comments = comments[comments["post_id"].isin(slot_post_ids) | comments["author_id"].isin(neighbor_ids)].copy()
    bot_comments = comments[comments["is_generated"].astype(str).str.lower().eq("true")].copy()
    comment_pool = pd.concat([local_comments, bot_comments.tail(500)], ignore_index=True).drop_duplicates("comment_id")
    if not comment_pool.empty:
        comment_pool = comment_pool.sample(n=min(len(comment_pool), browse_count * 3), random_state=rng.randint(1, 10**9))
        for row in comment_pool.itertuples(index=False):
            rec = row._asdict()
            if rec.get("author_id") == bot_id:
                continue
            post = post_by_id.get(rec.get("post_id", ""), {})
            candidates.append(
                {
                    "target_type": "comment",
                    "target_id": rec.get("comment_id", ""),
                    "post_id": rec.get("post_id", ""),
                    "comment_id": rec.get("comment_id", ""),
                    "author_id": rec.get("author_id", ""),
                    "family": rec.get("family", ""),
                    "subreddit": post.get("subreddit", ""),
                    "topic": topic_from_post(post),
                    "excerpt": text_excerpt(rec.get("body", "")),
                    "candidate_type": "slot_or_mixed_comment",
                }
            )

    rng.shuffle(candidates)
    return candidates[:browse_count]


def sample_candidates(pool: list[dict[str, Any]], rng: random.Random, browse_count: int) -> list[dict[str, Any]]:
    if not pool:
        return []
    if len(pool) <= browse_count:
        sampled = list(pool)
    else:
        sampled = rng.sample(pool, browse_count)
    rng.shuffle(sampled)
    return sampled


def choose_action(bot_index: int, window: int, candidates: list[dict[str, Any]], rng: random.Random) -> str:
    if window == 0:
        return "post"
    if window == 1:
        return "comment1"
    if window == 2:
        return "comment2" if any(c["target_type"] == "comment" for c in candidates) else "comment1"
    if window == 3:
        return "skip" if bot_index % 5 == 0 else "revisit"
    choices = ["comment1", "comment2", "post", "revisit", "skip"]
    weights = [0.32, 0.22, 0.16, 0.18, 0.12]
    if not any(c["target_type"] == "comment" for c in candidates):
        weights[1] = 0.02
        weights[0] += 0.20
    return rng.choices(choices, weights=weights, k=1)[0]


def generate_post_text(bot_index: int, source_slot: str, candidates: list[dict[str, Any]], rng: random.Random) -> tuple[str, str, str]:
    topic = rng.choice([c.get("topic") or c.get("subreddit") or "news" for c in candidates] or ["news"])
    hooks = [
        "Can someone explain why this part keeps getting skipped?",
        "This is the bit I keep coming back to.",
        "Maybe I am missing something, but this framing feels too neat.",
        "The weird thing is not the headline, it is what happens next.",
    ]
    title = rng.choice(hooks)
    body = (
        f"I was reading a few {topic} threads and the part that feels off is how quickly people jump "
        f"from one detail to the whole story. I get the instinct, but the incentives and follow-up matter too."
    )
    if rng.random() < 0.35:
        body += " Curious how other people are reading it."
    return title, body, str(topic)


def generate_comment_text(action: str, candidate: dict[str, Any], rng: random.Random) -> str:
    excerpt = candidate.get("excerpt", "")
    templates = {
        "comment1": [
            "I can see that angle, but I do not think the main issue is just the headline. The follow-through is what matters.",
            "This seems less settled than people are making it sound. A lot depends on what happens after the first reaction.",
            "I mostly agree with the concern, though the incentives here feel messier than the thread is admitting.",
        ],
        "comment2": [
            "That is fair, but I think there is another layer here: people are reacting to the pattern, not only this one example.",
            "I read it a bit differently. The point is not that everyone involved is wrong, it is that the system rewards bad shortcuts.",
            "Maybe, but that still leaves the practical question unanswered. What would actually change the behavior?",
        ],
        "revisit": [
            "Coming back to this after reading the replies, I think the strongest point is the one about incentives rather than intentions.",
            "After sitting with this, I am less convinced by the simple explanation. The context around it matters more than I first thought.",
            "I still think the earlier concern holds up, but the replies made me more cautious about treating it as obvious.",
        ],
    }
    body = rng.choice(templates.get(action, templates["comment1"]))
    if excerpt and rng.random() < 0.35:
        body += f" The part about '{text_excerpt(excerpt, 80)}' is exactly why I hesitate."
    return body


def add_human_responses(
    bot_id: str,
    bot_family: str,
    parent_kind: str,
    parent_id: str,
    post_id: str,
    created_time: datetime,
    response_bank: pd.DataFrame,
    rng: random.Random,
    probability: float,
    max_per_action: int,
    counters: dict[str, int],
    run_slug: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if response_bank.empty or rng.random() > probability:
        return [], [], []
    take = rng.randint(1, max(1, max_per_action))
    sample = response_bank.sample(n=min(take, len(response_bank)), replace=len(response_bank) < take, random_state=rng.randint(1, 10**9))
    comments: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for _, src in sample.iterrows():
        counters["human_response"] += 1
        comment_id = f"c_g4hr_{run_slug}_{counters['human_response']:06d}"
        event_id = f"evt_g4hr_{run_slug}_{counters['human_response']:06d}"
        level = "1" if parent_kind == "post" else "2"
        body = str(src.get("source_body", "")).strip()
        if not body:
            continue
        response_time = created_time + timedelta(minutes=rng.randint(8, 180))
        comments.append(
            {
                "comment_id": comment_id,
                "author_id": str(src.get("responder_user_id", "")),
                "author_source": "original_human",
                "family": "original_human",
                "post_id": post_id,
                "parent_id": parent_id,
                "parent_type": parent_kind,
                "level": level,
                "created_utc": fmt_time(response_time),
                "body": body,
                "score": str(rng.randint(0, 12)),
                "is_generated": "false",
                "is_replay_human": "true",
                "action_type": "human_slot_response",
                "target_author_id": bot_id,
            }
        )
        events.append(
            {
                "event_id": event_id,
                "event_type": "notification_created",
                "actor_user_id": str(src.get("responder_user_id", "")),
                "actor_family": "original_human",
                "target_author_id": bot_id,
                "target_post_id": post_id,
                "target_comment_id": comment_id,
                "content_id": comment_id,
                "policy_reason": "slot_conditioned_human_response_opportunity",
                "time": fmt_time(response_time),
            }
        )
        manifest.append(
            {
                "g4_user_id": bot_id,
                "human_response_comment_id": comment_id,
                "responder_user_id": str(src.get("responder_user_id", "")),
                "source_comment_id": str(src.get("source_comment_id", "")),
                "parent_kind": parent_kind,
                "parent_id": parent_id,
                "post_id": post_id,
                "created_utc": fmt_time(response_time),
            }
        )
    return comments, events, manifest


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_clean_output(input_dir, output_dir, args.overwrite)
    run_slug = output_dir.name

    users = read_csv(output_dir / "users.csv")
    posts = read_csv(output_dir / "posts.csv")
    comments = read_csv(output_dir / "comments.csv")
    labels = read_csv(output_dir / "labels.csv")
    profiles = read_csv(output_dir / "persona_profiles.csv")
    family_manifest = read_csv(output_dir / "family_manifest.csv")

    required = {"users": users, "posts": posts, "comments": comments, "labels": labels}
    missing = [name for name, df in required.items() if df.empty]
    if missing:
        raise ValueError(f"Missing or empty required tables: {missing}")

    slot_metrics = compute_human_slot_metrics(users, posts, comments)
    selected_slots = sample_slots(slot_metrics, args.g4_count, rng)
    response_bank = build_response_bank(posts, comments, pd.DataFrame(selected_slots))

    base_time = next_base_time(posts, comments)
    post_by_id = posts.set_index("post_id").to_dict("index")
    comment_by_id = comments.set_index("comment_id").to_dict("index")

    new_users: list[dict[str, Any]] = []
    new_labels: list[dict[str, Any]] = []
    new_profiles: list[dict[str, Any]] = []
    new_posts: list[dict[str, Any]] = []
    new_comments: list[dict[str, Any]] = []
    new_actions: list[dict[str, Any]] = []
    new_events: list[dict[str, Any]] = []
    new_memory: list[dict[str, Any]] = []
    slot_manifest: list[dict[str, Any]] = []
    human_response_manifest: list[dict[str, Any]] = []
    counters = {"post": 0, "comment": 0, "event": 0, "human_response": 0}
    slot_candidate_pools: dict[str, list[dict[str, Any]]] = {}

    for bot_index, slot in enumerate(selected_slots):
        source_slot = str(slot["source_human_slot_id"])
        bot_id = f"gb_g4_{run_slug}_{bot_index:04d}"
        profile = make_profile(bot_id, slot, rng, bot_index)
        new_profiles.append(profile)
        new_users.append(
            {
                "user_id": bot_id,
                "user_name": f"g4_slot_{bot_index:04d}",
                "source": "generated_bot",
                "label": "1",
                "family": FAMILY,
                "is_generated": "true",
                "is_replay_human": "false",
                "created_by_run": run_slug,
            }
        )
        new_labels.append(
            {
                "node_id": bot_id,
                "node_type": "user",
                "label": "1",
                "label_known": "1",
                "source": "generated_bot",
                "family": FAMILY,
                "split_group": "generated_g4",
            }
        )
        slot_bank = response_bank[response_bank["source_human_slot_id"].eq(source_slot)].copy()
        slot_manifest.append(
            {
                "g4_user_id": bot_id,
                "source_human_slot_id": source_slot,
                "source_authored_posts": safe_int(slot.get("authored_posts")),
                "source_authored_comments": safe_int(slot.get("authored_comments")),
                "source_outbound_interactions": safe_int(slot.get("outbound_interactions")),
                "source_inbound_opportunities": safe_int(slot.get("inbound_opportunities")),
                "response_bank_size": len(slot_bank),
                "provider": profile["llm_provider"],
                "model": profile["llm_model"],
                "temperature": profile["llm_temperature"],
            }
        )

        own_posts: list[str] = []
        own_comments: list[dict[str, str]] = []
        for window in range(args.time_windows):
            now = base_time + timedelta(minutes=bot_index * 17 + window * 97 + rng.randint(0, 25))
            if source_slot not in slot_candidate_pools:
                slot_candidate_pools[source_slot] = build_candidates(
                    bot_id,
                    source_slot,
                    posts,
                    comments,
                    response_bank,
                    rng,
                    max(50, args.browse_count * 5),
                )
            candidates = sample_candidates(slot_candidate_pools[source_slot], rng, args.browse_count)
            counters["event"] += 1
            browse_event_id = f"evt_g4_{run_slug}_{counters['event']:08d}"
            new_events.append(
                {
                    "event_id": browse_event_id,
                    "event_type": "browse",
                    "actor_user_id": bot_id,
                    "actor_family": FAMILY,
                    "candidate_count": len(candidates),
                    "candidate_summary": candidate_summary(candidates, args.browse_count),
                    "policy_reason": "slot-conditioned browse: source slot threads, slot neighbors, generated content, exploration",
                    "target_author_id": "",
                    "target_post_id": "",
                    "target_comment_id": "",
                    "content_id": "",
                    "time": fmt_time(now),
                }
            )
            new_memory.append(
                {
                    "event_id": browse_event_id,
                    "memory_type": "browse_context",
                    "actor_user_id": bot_id,
                    "actor_family": FAMILY,
                    "source_human_slot_id": source_slot,
                    "candidates": candidates,
                    "notification_queue_size": len(slot_bank),
                    "own_history_size": len(own_posts) + len(own_comments),
                    "time": fmt_time(now),
                }
            )

            action = choose_action(bot_index, window, candidates, rng)
            content_id = ""
            target_post_id = ""
            target_comment_id = ""
            target_author_id = ""
            policy_reason = f"G4 slot-conditioned autonomous {action}"
            action_time = now + timedelta(minutes=rng.randint(1, 12))
            if action == "skip" or not candidates and action != "post":
                action = "skip"
            elif action == "post":
                counters["post"] += 1
                post_id = f"p_g4_{run_slug}_{counters['post']:06d}"
                title, body, topic = generate_post_text(bot_index, source_slot, candidates, rng)
                new_posts.append(
                    {
                        "post_id": post_id,
                        "author_id": bot_id,
                        "author_source": "generated_bot",
                        "family": FAMILY,
                        "subreddit": rng.choice(["politics", "worldnews", "news", "GlobalTalk"]),
                        "topic": topic,
                        "created_utc": fmt_time(action_time),
                        "title": title,
                        "body": body,
                        "score": str(rng.randint(0, 8)),
                        "num_comments": "0",
                        "upvote_ratio": f"{0.55 + rng.random() * 0.35:.2f}",
                        "is_generated": "true",
                        "is_replay_context": "false",
                    }
                )
                posts = pd.concat([posts, pd.DataFrame([new_posts[-1]])], ignore_index=True)
                post_by_id[post_id] = new_posts[-1]
                own_posts.append(post_id)
                content_id = post_id
                target_post_id = post_id
                hr_comments, hr_events, hr_manifest = add_human_responses(
                    bot_id, FAMILY, "post", post_id, post_id, action_time, slot_bank, rng,
                    args.human_response_probability, args.max_human_responses_per_action,
                    counters, run_slug,
                )
                new_comments.extend(hr_comments)
                new_events.extend(hr_events)
                human_response_manifest.extend(hr_manifest)
            else:
                if action == "revisit" and own_comments:
                    target = dict(rng.choice(own_comments))
                    target["target_type"] = "comment"
                elif action == "revisit" and own_posts:
                    post_id = rng.choice(own_posts)
                    target = {"target_type": "post", "target_id": post_id, "post_id": post_id, "author_id": bot_id, "family": FAMILY, "topic": post_by_id.get(post_id, {}).get("topic", ""), "excerpt": post_by_id.get(post_id, {}).get("title", "")}
                elif action == "comment2":
                    comment_candidates = [c for c in candidates if c["target_type"] == "comment"]
                    target = rng.choice(comment_candidates or candidates)
                    if target["target_type"] != "comment":
                        action = "comment1"
                else:
                    post_candidates = [c for c in candidates if c["target_type"] == "post"]
                    target = rng.choice(post_candidates or candidates)
                    if target["target_type"] != "post" and action == "comment1":
                        action = "comment2"

                counters["comment"] += 1
                comment_id = f"c_g4_{run_slug}_{counters['comment']:06d}"
                if target["target_type"] == "comment":
                    parent_id = str(target["target_id"])
                    parent_type = "comment"
                    parent_comment = comment_by_id.get(parent_id, {})
                    post_id = str(target.get("post_id") or parent_comment.get("post_id", ""))
                    level = "2"
                    target_author_id = str(target.get("author_id") or parent_comment.get("author_id", ""))
                else:
                    post_id = str(target["post_id"])
                    parent_id = post_id
                    parent_type = "post"
                    level = "1"
                    target_author_id = str(target.get("author_id") or post_by_id.get(post_id, {}).get("author_id", ""))
                body = generate_comment_text(action, target, rng)
                comment_row = {
                    "comment_id": comment_id,
                    "author_id": bot_id,
                    "author_source": "generated_bot",
                    "family": FAMILY,
                    "post_id": post_id,
                    "parent_id": parent_id,
                    "parent_type": parent_type,
                    "level": level,
                    "created_utc": fmt_time(action_time),
                    "body": body,
                    "score": str(rng.randint(0, 6)),
                    "is_generated": "true",
                    "is_replay_human": "false",
                    "action_type": action,
                    "target_author_id": target_author_id if target_author_id != bot_id else "",
                }
                new_comments.append(comment_row)
                comments = pd.concat([comments, pd.DataFrame([comment_row])], ignore_index=True)
                comment_by_id[comment_id] = comment_row
                own_comments.append({"target_type": "comment", "target_id": comment_id, "post_id": post_id, "author_id": bot_id, "family": FAMILY, "topic": target.get("topic", ""), "excerpt": body})
                content_id = comment_id
                target_post_id = post_id
                target_comment_id = parent_id if parent_type == "comment" else ""
                hr_comments, hr_events, hr_manifest = add_human_responses(
                    bot_id, FAMILY, "comment", comment_id, post_id, action_time, slot_bank, rng,
                    args.human_response_probability, args.max_human_responses_per_action,
                    counters, run_slug,
                )
                new_comments.extend(hr_comments)
                new_events.extend(hr_events)
                human_response_manifest.extend(hr_manifest)

            counters["event"] += 1
            event_id = f"evt_g4_{run_slug}_{counters['event']:08d}"
            new_actions.append(
                {
                    "action_type": action,
                    "actor_family": FAMILY,
                    "actor_user_id": bot_id,
                    "content_id": content_id,
                    "dry_run": False,
                    "event_id": event_id,
                    "llm_call_id": "",
                    "llm_status": "slot_policy_text",
                    "policy_reason": policy_reason,
                    "source_human_slot_id": source_slot,
                    "target_author_id": target_author_id,
                    "target_comment_id": target_comment_id,
                    "target_post_id": target_post_id,
                    "time": fmt_time(action_time),
                }
            )
            new_events.append(
                {
                    "event_id": event_id,
                    "event_type": action,
                    "actor_user_id": bot_id,
                    "actor_family": FAMILY,
                    "content_id": content_id,
                    "llm_call_id": "",
                    "llm_status": "slot_policy_text",
                    "policy_reason": policy_reason,
                    "source_human_slot_id": source_slot,
                    "target_author_id": target_author_id,
                    "target_comment_id": target_comment_id,
                    "target_post_id": target_post_id,
                    "time": fmt_time(action_time),
                }
            )
            new_memory.append(
                {
                    "event_id": event_id,
                    "memory_type": "post_action_state",
                    "action_type": action,
                    "actor_user_id": bot_id,
                    "actor_family": FAMILY,
                    "source_human_slot_id": source_slot,
                    "notification_queue_size": len(slot_bank),
                    "own_history_size": len(own_posts) + len(own_comments),
                    "time": fmt_time(action_time),
                }
            )

    users = pd.concat([users, pd.DataFrame(new_users)], ignore_index=True)
    labels = pd.concat([labels, pd.DataFrame(new_labels)], ignore_index=True)
    profiles = pd.concat([profiles, pd.DataFrame(new_profiles)], ignore_index=True)
    posts = pd.concat([read_csv(output_dir / "posts.csv"), pd.DataFrame(new_posts)], ignore_index=True)
    comments = pd.concat([read_csv(output_dir / "comments.csv"), pd.DataFrame(new_comments)], ignore_index=True)
    family_manifest = pd.concat(
        [
            family_manifest,
            pd.DataFrame(
                [
                    {
                        "family": FAMILY,
                        "family_label": FAMILY_LABEL,
                        "bot_count": str(args.g4_count),
                        "policy_source": "build_g4_slot_conditioned_dataset.py",
                        "dry_run": "false",
                        "live_llm": "false",
                        "description": "slot-conditioned human-to-bot family with sampled original-human response opportunities",
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("family", keep="last")

    write_csv(output_dir / "users.csv", users)
    write_csv(output_dir / "labels.csv", labels)
    write_csv(output_dir / "persona_profiles.csv", profiles)
    write_csv(output_dir / "posts.csv", posts)
    write_csv(output_dir / "comments.csv", comments)
    write_csv(output_dir / "family_manifest.csv", family_manifest)
    write_csv(output_dir / "g4_slot_manifest.csv", pd.DataFrame(slot_manifest))
    write_csv(output_dir / "g4_response_opportunity_bank.csv", response_bank)
    write_csv(output_dir / "g4_human_response_manifest.csv", pd.DataFrame(human_response_manifest))

    append_jsonl(output_dir / "action_trace.jsonl", new_actions)
    append_jsonl(output_dir / "events.jsonl", new_events)
    append_jsonl(output_dir / "memory_trace.jsonl", new_memory)
    append_jsonl(output_dir / "g4_action_trace.jsonl", new_actions)

    manifest = read_json(output_dir / "dataset_manifest.json")
    manifest.update(
        {
            "run_name": run_slug,
            "source_dataset": str(input_dir),
            "derived_family_added": FAMILY,
            "g4_bot_count": args.g4_count,
            "g4_time_windows": args.time_windows,
            "g4_human_response_count": len(human_response_manifest),
            "g4_generated_post_count": len(new_posts),
            "g4_generated_comment_count": sum(1 for row in new_comments if str(row.get("author_source")) == "generated_bot"),
            "g4_human_replay_comment_count": sum(1 for row in new_comments if str(row.get("author_source")) == "original_human"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "notes": "G4 appends slot-conditioned bots and sampled original-human replies; graph/splits must be rebuilt after this script.",
        }
    )
    write_json(output_dir / "dataset_manifest.json", manifest)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "g4_bot_count": args.g4_count,
        "g4_generated_posts": len(new_posts),
        "g4_generated_comments": sum(1 for row in new_comments if str(row.get("author_source")) == "generated_bot"),
        "g4_human_to_g4_replay_comments": sum(1 for row in new_comments if str(row.get("author_source")) == "original_human"),
        "selected_slot_count": len(slot_manifest),
        "response_bank_rows": len(response_bank),
        "action_counts": dict(Counter(row["action_type"] for row in new_actions)),
        "needs_postprocess": ["export_static_graph_dataset.py", "build_graph_splits.py", "check_dataset_integrity.py"],
    }
    write_json(output_dir / "g4_build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
