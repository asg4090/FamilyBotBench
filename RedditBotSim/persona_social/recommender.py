from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import math
import random
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Iterable

from .models import Candidate, MemoryState, Persona


TOPIC_KEYWORDS = {
    "US politics": ["trump", "biden", "republican", "democrat", "congress", "election", "vote", "court", "white house"],
    "Russia-Ukraine war": ["ukraine", "russia", "putin", "zelensky", "nato", "kyiv", "moscow", "war"],
    "Israel-Palestine": ["israel", "gaza", "hamas", "palestinian", "idf", "rafah", "hostage"],
    "world news": ["global", "country", "government", "minister", "police", "protest", "climate", "world"],
}


def _is_bot_like(author_id: str | None) -> bool:
    if not author_id:
        return False
    author = str(author_id).strip().lower()
    return author.startswith(("psbot", "family_", "bot_", "ai_")) or "_g" in author or "family" in author


def _persona_value(persona: Persona | Mapping[str, object], key: str, default: object = None) -> object:
    if isinstance(persona, Mapping):
        return persona.get(key, default)
    return getattr(persona, key, default)


def _persona_float(persona: Persona | Mapping[str, object], key: str, default: float) -> float:
    value = _persona_value(persona, key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_topic(text: str) -> str:
    lowered = str(text or "").lower()
    scores = {
        topic: sum(1 for keyword in keywords if keyword in lowered)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    best_topic = max(scores, key=scores.get)
    return best_topic if scores[best_topic] > 0 else "world news"


def parse_time(value: str) -> datetime:
    return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")


def as_time(value) -> str:
    return str(value)


class PersonaRecommender:
    def __init__(self, env, rng: random.Random) -> None:
        self.env = env
        self.rng = rng

    def build_candidates(
        self,
        persona: Persona,
        memory: MemoryState,
        round_time: datetime,
        round_id: int,
        browse_count: int,
    ) -> list[Candidate]:
        all_candidates: list[Candidate] = []
        all_candidates.extend(self._notification_candidates(persona, memory, round_time))
        all_candidates.extend(self._own_history_candidates(persona, memory, round_time))
        all_candidates.extend(self._post_candidates(persona, memory, round_time))
        all_candidates.extend(self._comment_candidates(persona, memory, round_time))
        all_candidates.extend(self._exploration_candidates(persona, memory, round_time))
        deduped = self._dedupe(all_candidates)
        for cand in deduped:
            cand.score = self._score_candidate(cand, persona, memory, round_time)
        deduped.sort(key=lambda c: (c.score, self.rng.random()), reverse=True)

        selected = self._mixed_browse(persona, memory, round_time, round_id, deduped, browse_count)
        selected = self._cap_bot_authored(selected, browse_count)
        memory.mark_browse(selected, round_id=round_id)
        selected.sort(key=lambda c: (c.source == "notification", c.source == "own_history", c.score), reverse=True)
        return selected[:browse_count]

    def _notification_candidates(self, persona: Persona, memory: MemoryState, round_time: datetime) -> list[Candidate]:
        out: list[Candidate] = []
        posts = self.env.get_env_reddit_data().posts
        c1_by_id = {c.comment1_id: c for c in self.env.get_env_reddit_data().comments1}
        for note in memory.notifications:
            note_id = note.get("notification_id")
            if not note_id or note_id in memory.revisited_notifications:
                continue
            post = posts.get(note.get("post_id"))
            if post is None:
                continue
            comment = c1_by_id.get(note.get("comment_id"))
            if comment is None:
                continue
            out.append(
                Candidate(
                    candidate_id=f"notification:{note_id}",
                    source="notification",
                    action_hint="browse_thread",
                    post_id=post.submission_id,
                    post_time=as_time(post.time),
                    post_author_id=post.author_id,
                    post_author_name=post.author_name,
                    subreddit=post.subreddit,
                    topic=infer_topic(f"{post.posts} {comment.comment1_body}"),
                    post_text=post.posts,
                    comment_id=comment.comment1_id,
                    comment_time=as_time(comment.comment1_time),
                    comment_author_id=comment.comment1_author_id,
                    comment_author_name=comment.comment1_author_name,
                    comment_text=comment.comment1_body,
                    post_score=post.score,
                    post_num_comments=post.num_comments,
                    post_upvote_ratio=post.upvote_ratio,
                    thread_comments=self._thread_comments(post.submission_id),
                    interaction_summary=self._thread_summary(post.submission_id, focus_comment_id=comment.comment1_id),
                    reasons=["someone replied to this bot's post/comment"],
                )
            )
        return out

    def _own_history_candidates(self, persona: Persona, memory: MemoryState, round_time: datetime) -> list[Candidate]:
        out: list[Candidate] = []
        posts = self.env.get_env_reddit_data().posts
        c1_by_id = {c.comment1_id: c for c in self.env.get_env_reddit_data().comments1}
        c2_by_id = {c.comment2_id: c for c in self.env.get_env_reddit_data().comments2}
        for post_id in list(memory.own_posts)[-6:]:
            post = posts.get(post_id)
            if post is None:
                continue
            out.append(
                Candidate(
                    candidate_id=f"ownpost:{post_id}",
                    source="own_history",
                    action_hint="browse_thread",
                    post_id=post.submission_id,
                    post_time=as_time(post.time),
                    post_author_id=post.author_id,
                    post_author_name=post.author_name,
                    subreddit=post.subreddit,
                    topic=infer_topic(post.posts),
                    post_text=post.posts,
                    post_score=post.score,
                    post_num_comments=post.num_comments,
                    post_upvote_ratio=post.upvote_ratio,
                    thread_comments=self._thread_comments(post.submission_id),
                    interaction_summary=self._thread_summary(post.submission_id),
                    reasons=["own post history"],
                )
            )
        for comment_id in list(memory.own_comment1)[-4:]:
            comment = c1_by_id.get(comment_id)
            if comment is None:
                continue
            post = posts.get(str(comment.link_id).replace("t3_", ""))
            if post is None:
                continue
            out.append(
                Candidate(
                    candidate_id=f"ownc1:{comment_id}",
                    source="own_history",
                    action_hint="browse_thread",
                    post_id=post.submission_id,
                    post_time=as_time(post.time),
                    post_author_id=post.author_id,
                    post_author_name=post.author_name,
                    subreddit=post.subreddit,
                    topic=infer_topic(f"{post.posts} {comment.comment1_body}"),
                    post_text=post.posts,
                    comment_id=comment.comment1_id,
                    comment_time=as_time(comment.comment1_time),
                    comment_author_id=comment.comment1_author_id,
                    comment_author_name=comment.comment1_author_name,
                    comment_text=comment.comment1_body,
                    post_score=post.score,
                    post_num_comments=post.num_comments,
                    post_upvote_ratio=post.upvote_ratio,
                    thread_comments=self._thread_comments(post.submission_id),
                    interaction_summary=self._thread_summary(post.submission_id, focus_comment_id=comment.comment1_id),
                    reasons=["own comment history"],
                )
            )
        for comment_id in list(memory.own_comment2)[-4:]:
            comment = c2_by_id.get(comment_id)
            if comment is None:
                continue
            post = posts.get(str(comment.link_id).replace("t3_", ""))
            if post is None:
                continue
            parent_comment = c1_by_id.get(str(comment.parent_id).replace("t1_", ""))
            out.append(
                Candidate(
                    candidate_id=f"ownc2:{comment_id}",
                    source="own_history",
                    action_hint="browse_thread",
                    post_id=post.submission_id,
                    post_time=as_time(post.time),
                    post_author_id=post.author_id,
                    post_author_name=post.author_name,
                    subreddit=post.subreddit,
                    topic=infer_topic(f"{post.posts} {(parent_comment.comment1_body if parent_comment else '')} {comment.comment2_body}"),
                    post_text=post.posts,
                    comment_id=parent_comment.comment1_id if parent_comment else str(comment.parent_id).replace("t1_", ""),
                    comment_time=as_time(parent_comment.comment1_time if parent_comment else comment.comment2_time),
                    comment_author_id=parent_comment.comment1_author_id if parent_comment else comment.comment2_author_id,
                    comment_author_name=parent_comment.comment1_author_name if parent_comment else comment.comment2_author_name,
                    comment_text=parent_comment.comment1_body if parent_comment else comment.comment2_body,
                    post_score=post.score,
                    post_num_comments=post.num_comments,
                    post_upvote_ratio=post.upvote_ratio,
                    thread_comments=self._thread_comments(post.submission_id),
                    interaction_summary=self._thread_summary(post.submission_id, focus_comment_id=(parent_comment.comment1_id if parent_comment else None)),
                    reasons=["own reply history"],
                )
            )
        return out

    def _post_candidates(self, persona: Persona, memory: MemoryState, round_time: datetime) -> list[Candidate]:
        posts = list(self.env.get_env_reddit_data().posts.values())
        start = round_time - timedelta(days=18)
        out: list[Candidate] = []
        for post in posts:
            try:
                post_time = parse_time(as_time(post.time))
            except Exception:
                continue
            if post_time > round_time or post_time < start:
                continue
            if post.subreddit not in persona.subreddits and self.rng.random() > persona.exploration_tendency:
                continue
            topic = infer_topic(post.posts)
            out.append(
                Candidate(
                    candidate_id=f"post:{post.submission_id}",
                    source="feed",
                    action_hint="browse_thread",
                    post_id=post.submission_id,
                    post_time=as_time(post.time),
                    post_author_id=post.author_id,
                    post_author_name=post.author_name,
                    subreddit=post.subreddit,
                    topic=topic,
                    post_text=post.posts,
                    post_score=post.score,
                    post_num_comments=post.num_comments,
                    post_upvote_ratio=post.upvote_ratio,
                    thread_comments=self._thread_comments(post.submission_id),
                    interaction_summary=self._thread_summary(post.submission_id),
                    reasons=["feed post"],
                )
            )
        return out

    def _comment_candidates(self, persona: Persona, memory: MemoryState, round_time: datetime) -> list[Candidate]:
        posts = self.env.get_env_reddit_data().posts
        start = round_time - timedelta(days=18)
        out: list[Candidate] = []
        for comment in self.env.get_env_reddit_data().comments1:
            post_id = str(comment.link_id).replace("t3_", "")
            post = posts.get(post_id)
            if post is None:
                continue
            try:
                comment_time = parse_time(as_time(comment.comment1_time))
            except Exception:
                continue
            if comment_time > round_time or comment_time < start:
                continue
            if post.subreddit not in persona.subreddits and self.rng.random() > persona.exploration_tendency:
                continue
            text = f"{post.posts} {comment.comment1_body}"
            out.append(
                Candidate(
                    candidate_id=f"comment:{comment.comment1_id}",
                    source="thread",
                    action_hint="comment2",
                    post_id=post.submission_id,
                    post_time=as_time(post.time),
                    post_author_id=post.author_id,
                    post_author_name=post.author_name,
                    subreddit=post.subreddit,
                    topic=infer_topic(text),
                    post_text=post.posts,
                    comment_id=comment.comment1_id,
                    comment_time=as_time(comment.comment1_time),
                    comment_author_id=comment.comment1_author_id,
                    comment_author_name=comment.comment1_author_name,
                    comment_text=comment.comment1_body,
                    reasons=["first-level comment in browsed thread"],
                )
            )
        return out

    def _exploration_candidates(self, persona: Persona, memory: MemoryState, round_time: datetime) -> list[Candidate]:
        posts = list(self.env.get_env_reddit_data().posts.values())
        self.rng.shuffle(posts)
        out: list[Candidate] = []
        subreddits = set(_persona_value(persona, "subreddits", []) or [])
        for post in posts[: max(12, len(posts) // 12 or 1)]:
            if post.subreddit in subreddits and self.rng.random() < 0.7:
                continue
            topic = infer_topic(post.posts)
            out.append(
                Candidate(
                    candidate_id=f"explore:{post.submission_id}",
                    source="explore",
                    action_hint="browse_thread",
                    post_id=post.submission_id,
                    post_time=as_time(post.time),
                    post_author_id=post.author_id,
                    post_author_name=post.author_name,
                    subreddit=post.subreddit,
                    topic=topic,
                    post_text=post.posts,
                    post_score=post.score,
                    post_num_comments=post.num_comments,
                    post_upvote_ratio=post.upvote_ratio,
                    thread_comments=self._thread_comments(post.submission_id),
                    interaction_summary=self._thread_summary(post.submission_id),
                    reasons=["exploration stream"],
                )
            )
        return out

    def _thread_comments(self, post_id: str, *, max_first_level: int = 7, max_second_level: int = 4) -> list[dict[str, object]]:
        comments1 = [
            c
            for c in self.env.get_env_reddit_data().comments1
            if str(c.link_id).replace("t3_", "") == str(post_id)
        ]
        comments2 = [
            c
            for c in self.env.get_env_reddit_data().comments2
            if str(c.link_id).replace("t3_", "") == str(post_id)
        ]
        comments1.sort(key=lambda c: (int(c.comment1_score or 0), str(c.comment1_time)), reverse=True)
        replies_by_parent: dict[str, list[object]] = {}
        for c2 in comments2:
            parent = str(c2.parent_id).replace("t1_", "")
            replies_by_parent.setdefault(parent, []).append(c2)
        out: list[dict[str, object]] = []
        for c1 in comments1[:max_first_level]:
            replies = replies_by_parent.get(c1.comment1_id, [])
            replies.sort(key=lambda c: (int(c.comment2_score or 0), str(c.comment2_time)), reverse=True)
            out.append(
                {
                    "comment_id": c1.comment1_id,
                    "author": c1.comment1_author_name,
                    "author_id": c1.comment1_author_id,
                    "time": as_time(c1.comment1_time),
                    "score": c1.comment1_score,
                    "text": c1.comment1_body,
                    "replies": [
                        {
                            "comment_id": c2.comment2_id,
                            "author": c2.comment2_author_name,
                            "author_id": c2.comment2_author_id,
                            "time": as_time(c2.comment2_time),
                            "score": c2.comment2_score,
                            "text": c2.comment2_body,
                        }
                        for c2 in replies[:max_second_level]
                    ],
                }
            )
        return out

    def _thread_summary(self, post_id: str, *, focus_comment_id: str | None = None) -> dict[str, object]:
        comments = self._thread_comments(post_id, max_first_level=20, max_second_level=10)
        reply_count = sum(len(item.get("replies", [])) for item in comments)
        authors = {str(item.get("author_id", "")) for item in comments if item.get("author_id")}
        for item in comments:
            for reply in item.get("replies", []) or []:
                if isinstance(reply, Mapping) and reply.get("author_id"):
                    authors.add(str(reply.get("author_id")))
        return {
            "visible_first_level_comments": len(comments),
            "visible_second_level_replies": reply_count,
            "visible_unique_comment_authors": len(authors),
            "focus_comment_id": focus_comment_id or "",
        }

    def _score_candidate(self, cand: Candidate, persona: Persona, memory: MemoryState, round_time: datetime) -> float:
        graph_affinity = _persona_float(persona, "graph_affinity", 0.5)
        notification_reactivity = _persona_float(persona, "notification_reactivity", 0.5)
        exploration_tendency = _persona_float(persona, "exploration_tendency", 0.0)
        interest_weights = _persona_value(persona, "interest_weights", {}) or {}
        if not isinstance(interest_weights, Mapping):
            interest_weights = {}
        subreddits = _persona_value(persona, "subreddits", []) or []
        if not isinstance(subreddits, list):
            subreddits = list(subreddits) if not isinstance(subreddits, str) else [subreddits]
        score = float(interest_weights.get(cand.topic, 0.05)) * 3.0
        if cand.subreddit in subreddits:
            score += 0.8
        if _is_bot_like(cand.post_author_id) and cand.post_author_id != persona.user_id:
            score += 0.25 + 0.15 * (graph_affinity - 0.5)
        if cand.comment_author_id and _is_bot_like(cand.comment_author_id) and cand.comment_author_id != persona.user_id:
            score += 0.2 + 0.1 * (graph_affinity - 0.5)
        if cand.post_id in memory.seen_posts:
            score -= 0.8
        if cand.post_author_id == persona.user_id:
            score += 1.0
        if cand.comment_author_id == persona.user_id:
            score += 0.7
        if cand.source == "notification":
            score += 2.0 * _persona_float(persona, "revisit_tendency", 0.0) + 0.6 + 0.6 * (notification_reactivity - 0.5)
        elif cand.source == "own_history":
            score += 0.7 + 0.5 * _persona_float(persona, "revisit_tendency", 0.0)
        elif cand.source == "explore":
            score += 0.35 + exploration_tendency
        if cand.comment_author_id in memory.interacted_authors:
            score += 0.35
        if cand.candidate_id in memory.seen_candidate_ids:
            score -= 0.9
        if memory.recent_actions:
            if memory.recent_actions[-1] == "post":
                score += 0.1
            if memory.recent_actions[-1] in {"comment1", "comment2"} and cand.source in {"own_history", "notification"}:
                score += 0.15
        try:
            age_hours = max(1.0, (round_time - parse_time(cand.comment_time or cand.post_time)).total_seconds() / 3600)
            score += 0.45 / math.log(age_hours + 2)
        except Exception:
            pass
        score += self.rng.uniform(-0.2, 0.2)
        return score

    @staticmethod
    def _dedupe(candidates: Iterable[Candidate]) -> list[Candidate]:
        seen = set()
        out = []
        for cand in candidates:
            key = cand.candidate_id
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
        return out

    @staticmethod
    def _is_bot_authored(cand: Candidate) -> bool:
        return (
            _is_bot_like(cand.post_author_id)
            or bool(cand.comment_author_id and _is_bot_like(cand.comment_author_id))
            or cand.source == "notification"
        )

    def _cap_bot_authored(self, candidates: list[Candidate], browse_count: int) -> list[Candidate]:
        max_bot_authored = max(1, min(2, browse_count // 5))
        bot_items = [c for c in candidates if self._is_bot_authored(c)]
        human_items = [c for c in candidates if not self._is_bot_authored(c)]
        notification_items = [c for c in bot_items if c.source == "notification"][:1]
        other_bot_items = [c for c in bot_items if c.source != "notification"]
        capped_bot = (notification_items + other_bot_items)[:max_bot_authored]
        selected = human_items[: max(0, browse_count - len(capped_bot))] + capped_bot
        if len(selected) < browse_count:
            selected.extend(human_items[len(selected) : browse_count])
        return selected

    def _mixed_browse(
        self,
        persona: Persona,
        memory: MemoryState,
        round_time: datetime,
        round_id: int,
        candidates: list[Candidate],
        browse_count: int,
    ) -> list[Candidate]:
        if not candidates:
            return []
        target = max(1, browse_count)
        buckets = {
            "notification": [c for c in candidates if c.source == "notification"],
            "own_history": [c for c in candidates if c.source == "own_history"],
            "feed": [c for c in candidates if c.source == "feed"],
            "thread": [c for c in candidates if c.source == "thread"],
            "explore": [c for c in candidates if c.source == "explore"],
        }
        quotas = self._browse_quotas(persona, target)
        selected: list[Candidate] = []
        used_ids: set[str] = set()

        def take_from(bucket_name: str, limit: int) -> None:
            if limit <= 0:
                return
            pool = [c for c in buckets.get(bucket_name, []) if c.candidate_id not in used_ids and not memory.recently_seen(c.candidate_id, round_id=round_id, cooldown=1)]
            if not pool:
                return
            pool = sorted(pool, key=lambda c: c.score, reverse=True)
            if bucket_name == "explore":
                self.rng.shuffle(pool)
            for cand in pool[:limit]:
                if len(selected) >= target:
                    break
                selected.append(cand)
                used_ids.add(cand.candidate_id)

        for name in ["notification", "own_history", "feed", "thread", "explore"]:
            take_from(name, quotas.get(name, 0))

        if len(selected) < target:
            remainder = [c for c in candidates if c.candidate_id not in used_ids and not memory.recently_seen(c.candidate_id, round_id=round_id, cooldown=1)]
            remainder.sort(key=lambda c: c.score, reverse=True)
            for cand in remainder:
                if len(selected) >= target:
                    break
                selected.append(cand)
                used_ids.add(cand.candidate_id)

        if len(selected) < target:
            spill = [c for c in candidates if c.candidate_id not in used_ids]
            spill.sort(key=lambda c: c.score, reverse=True)
            for cand in spill:
                if len(selected) >= target:
                    break
                selected.append(cand)
                used_ids.add(cand.candidate_id)
        return selected[:target]

    def _browse_quotas(self, persona: Persona, browse_count: int) -> dict[str, int]:
        notification_reactivity = _persona_float(persona, "notification_reactivity", 0.5)
        revisit_tendency = _persona_float(persona, "revisit_tendency", 0.0)
        exploration_tendency = _persona_float(persona, "exploration_tendency", 0.0)
        graph_affinity = _persona_float(persona, "graph_affinity", 0.5)
        quotas = {
            "notification": 1 if notification_reactivity > 0.35 else 0,
            "own_history": 1 if revisit_tendency > 0.15 else 0,
            "feed": max(2, browse_count // 3),
            "thread": max(1, browse_count // 4 + (1 if graph_affinity > 0.55 else 0)),
            "explore": max(1, browse_count // 4 + (1 if exploration_tendency > 0.45 else 0)),
        }
        total = sum(quotas.values())
        while total > browse_count:
            for name in ["explore", "thread", "feed", "own_history", "notification"]:
                if quotas[name] > 0 and total > browse_count:
                    quotas[name] -= 1
                    total -= 1
        while total < browse_count:
            quotas["feed"] += 1
            total += 1
        return quotas
