from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class Persona:
    user_id: str
    user_name: str
    age: int
    gender: str
    region: str
    education: str
    ideology: str
    description: str
    subreddits: list[str]
    interest_weights: dict[str, float]
    style: str
    stance: str
    debate_tendency: float
    post_tendency: float
    reply_tendency: float
    revisit_tendency: float
    exploration_tendency: float
    brevity: float
    family: str = "baseline"
    family_round: int = 0
    family_generation: int = 0
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = None
    llm_reasoning_effort: str | None = None
    graph_affinity: float = 0.5
    notification_reactivity: float = 0.5
    traits: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_interest(self) -> str:
        return max(self.interest_weights, key=self.interest_weights.get)

    def llm_profile(self, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
        defaults = defaults or {}
        return {
            "provider": self.llm_provider or str(defaults.get("provider", "openai")),
            "model": self.llm_model or str(defaults.get("model", "gpt-5.4")),
            "temperature": float(self.llm_temperature if self.llm_temperature is not None else defaults.get("temperature", 0.4)),
            "reasoning_effort": self.llm_reasoning_effort or str(defaults.get("reasoning_effort", "medium")),
        }


@dataclass
class MemoryState:
    user_id: str
    own_posts: list[str] = field(default_factory=list)
    own_comment1: list[str] = field(default_factory=list)
    own_comment2: list[str] = field(default_factory=list)
    seen_posts: set[str] = field(default_factory=set)
    seen_candidate_ids: set[str] = field(default_factory=set)
    candidate_last_seen_round: dict[str, int] = field(default_factory=dict)
    interacted_authors: dict[str, int] = field(default_factory=dict)
    recent_topics: list[str] = field(default_factory=list)
    recent_actions: list[str] = field(default_factory=list)
    recent_candidate_sources: list[str] = field(default_factory=list)
    notifications: list[dict[str, Any]] = field(default_factory=list)
    revisited_notifications: set[str] = field(default_factory=set)

    def remember_author(self, author_id: str) -> None:
        if not author_id:
            return
        self.interacted_authors[author_id] = self.interacted_authors.get(author_id, 0) + 1

    def mark_browse(self, candidates: Iterable["Candidate"], round_id: int | None = None) -> None:
        recent_sources: list[str] = []
        for cand in candidates:
            self.seen_candidate_ids.add(cand.candidate_id)
            if round_id is not None:
                self.candidate_last_seen_round[cand.candidate_id] = round_id
            recent_sources.append(cand.source)
            if cand.topic:
                self.recent_topics.append(cand.topic)
        self.recent_topics = self.recent_topics[-12:]
        self.recent_candidate_sources = recent_sources[-12:]

    def record_action(self, action: str, candidate: "Candidate" | None = None) -> None:
        self.recent_actions.append(action)
        self.recent_actions = self.recent_actions[-12:]
        if candidate is not None:
            self.seen_candidate_ids.add(candidate.candidate_id)
            if candidate.topic:
                self.recent_topics.append(candidate.topic)
                self.recent_topics = self.recent_topics[-12:]

    def pending_notifications(self) -> list[dict[str, Any]]:
        pending = []
        for note in self.notifications:
            note_id = str(note.get("notification_id", ""))
            if note_id and note_id in self.revisited_notifications:
                continue
            if note.get("resolved"):
                continue
            pending.append(note)
        return pending[-8:]

    def recently_seen(self, candidate_id: str, round_id: int | None = None, cooldown: int = 1) -> bool:
        last_round = self.candidate_last_seen_round.get(candidate_id)
        if last_round is None:
            return False
        if round_id is None:
            return True
        return round_id - last_round < cooldown


@dataclass
class Candidate:
    candidate_id: str
    source: str
    action_hint: str
    post_id: str
    post_time: str
    post_author_id: str
    post_author_name: str
    subreddit: str
    topic: str
    post_text: str
    comment_id: str | None = None
    comment_time: str | None = None
    comment_author_id: str | None = None
    comment_author_name: str | None = None
    comment_text: str | None = None
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    post_score: int | float | None = None
    post_num_comments: int | None = None
    post_upvote_ratio: float | None = None
    thread_comments: list[dict[str, Any]] = field(default_factory=list)
    interaction_summary: dict[str, Any] = field(default_factory=dict)

    def to_prompt_item(self, index: int) -> dict[str, Any]:
        item = {
            "index": index,
            "candidate_id": self.candidate_id,
            "source": self.source,
            "suggested_action": self.action_hint,
            "post_id": self.post_id,
            "post_time": self.post_time,
            "post_author": self.post_author_name,
            "subreddit": self.subreddit,
            "topic": self.topic,
            "post": self.post_text,
            "post_stats": {
                "score": self.post_score,
                "num_comments": self.post_num_comments,
                "upvote_ratio": self.post_upvote_ratio,
            },
            "interaction_summary": dict(self.interaction_summary),
            "thread_comments": list(self.thread_comments),
            "reasons": list(self.reasons),
        }
        if self.comment_id:
            item.update(
                {
                    "first_level_comment_id": self.comment_id,
                    "first_level_comment_time": self.comment_time,
                    "first_level_comment_author": self.comment_author_name,
                    "first_level_comment": self.comment_text,
                }
            )
        return item


@dataclass
class Decision:
    persona: Persona
    round_id: int
    action: str
    candidate: Candidate | None
    reason: str
    planned_time: str
    prompt: str | None = None
    raw_response: str | None = None
    content: str | None = None
    parsed: dict[str, Any] | None = None
    fallback: bool = False
    result: str | None = None
