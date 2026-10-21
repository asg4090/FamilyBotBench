from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import random
from collections.abc import Mapping
from datetime import datetime

from .models import Candidate, Decision, MemoryState, Persona


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


class DecisionEngine:
    def __init__(self, rng: random.Random, mode: str = "system_first") -> None:
        self.rng = rng
        self.mode = mode

    def choose(self, persona: Persona, memory: MemoryState, round_id: int, round_time: datetime, candidates: list[Candidate]) -> Decision:
        planned_time = round_time.strftime("%Y-%m-%d %H:%M:%S")
        if self.mode == "llm_first":
            return Decision(persona, round_id, "browse_decide", None, "llm_first_persona_browser", planned_time)
        notification_candidates = [c for c in candidates if c.source == "notification"]
        explore_candidates = [c for c in candidates if c.source == "explore" and c.action_hint == "comment1"]
        revisit_tendency = _persona_float(persona, "revisit_tendency", 0.0)
        debate_tendency = _persona_float(persona, "debate_tendency", 0.0)
        post_tendency = _persona_float(persona, "post_tendency", 0.0)
        notification_reactivity = _persona_float(persona, "notification_reactivity", 0.5)
        revisit_prob = min(0.88, revisit_tendency * (0.35 + 0.55 * notification_reactivity) + 0.08 * len(notification_candidates))
        post_prob = min(
            0.22,
            0.035
            + 0.45 * post_tendency
            + (0.045 if not memory.own_posts else 0.02 if len(memory.own_posts) < 3 else 0.0)
            + 0.025 * len(explore_candidates) / max(1, len(candidates)),
        )
        skip_prob = max(0.05, 0.16 - 0.1 * debate_tendency - 0.05 * len(notification_candidates))
        if not candidates:
            if self.rng.random() < post_prob:
                return Decision(persona, round_id, "post", None, "no_feed_persona_post_tendency", planned_time)
            return Decision(persona, round_id, "skip", None, "no_viable_candidate", planned_time)
        if notification_candidates and self.rng.random() < revisit_prob:
            cand = self._weighted_pick(notification_candidates)
            return Decision(persona, round_id, "revisit", cand, "notification_revisit", planned_time)
        if self.rng.random() < skip_prob:
            return Decision(persona, round_id, "skip", None, "browsed_but_no_action", planned_time)
        if self.rng.random() < post_prob:
            return Decision(persona, round_id, "post", None, "persona_post_tendency_after_browse", planned_time)
        return Decision(persona, round_id, "browse_decide", None, "llm_choose_from_thread_browser", planned_time)

    def _weighted_pick(self, candidates: list[Candidate]) -> Candidate:
        top = sorted(candidates, key=lambda c: c.score, reverse=True)[: min(5, len(candidates))]
        weights = [max(0.05, c.score) for c in top]
        return self.rng.choices(top, weights=weights, k=1)[0]
