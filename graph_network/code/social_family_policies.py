"""Behavior policies for the three static graph social families.

This module is intentionally independent from the old RedditBotSim runner.  It
only decides *what kind of action* a bot should take and *which target* to act
on.  Text generation, event writing, notification delivery, and graph export
belong to the mixed environment runner.
"""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # Optional convenience only; this module must still import without pandas.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - depends on local environment.
    pd = None  # type: ignore

try:  # Optional convenience only; random.Random is enough for core behavior.
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - depends on local environment.
    np = None  # type: ignore


FAMILY_G1 = "G1_population_calibrated"
FAMILY_G2 = "G2_anchor_behavior"
FAMILY_G3 = "G3_contextual_interactive"
SUPPORTED_FAMILIES = (FAMILY_G1, FAMILY_G2, FAMILY_G3)
VALID_ACTIONS = ("skip", "post", "comment1", "comment2", "revisit")


@dataclass
class CandidateView:
    """Compact view of a visible post/comment/thread for policy scoring."""

    target_id: str
    target_type: str = "post"  # post, comment, thread, topic
    post_id: str = ""
    comment_id: str = ""
    parent_id: str = ""
    parent_type: str = ""
    author_id: str = ""
    topic: str = ""
    subreddit: str = ""
    created_utc: Any = None
    score: float = 0.0
    num_comments: int = 0
    level: int = 0
    age_hours: Optional[float] = None
    is_notification: bool = False
    already_seen: bool = False
    participated: bool = False
    title: str = ""
    body: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: Any) -> "CandidateView":
        """Build a candidate from another CandidateView, dataclass, or mapping."""

        if isinstance(value, cls):
            return value
        if hasattr(value, "__dataclass_fields__"):
            value = asdict(value)
        if not isinstance(value, Mapping):
            raise TypeError(f"Unsupported candidate type: {type(value)!r}")

        data = dict(value)
        comment_id = _first_text(data, ("comment_id", "target_comment_id", "id"))
        post_id = _first_text(data, ("post_id", "target_post_id", "thread_id"))
        explicit_target_id = _first_text(data, ("target_id", "content_id", "id"))
        target_type = _first_text(data, ("target_type", "type", "parent_type")) or ""

        level = _coerce_int(_first_present(data, ("level", "reply_depth")), 0)
        if not target_type:
            target_type = "comment" if comment_id and level > 0 else "post"
        target_id = explicit_target_id or comment_id or post_id

        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {
                key: val
                for key, val in data.items()
                if key
                not in {
                    "target_id",
                    "content_id",
                    "id",
                    "target_type",
                    "type",
                    "post_id",
                    "target_post_id",
                    "thread_id",
                    "comment_id",
                    "target_comment_id",
                }
            }

        return cls(
            target_id=str(target_id or ""),
            target_type=str(target_type or "post"),
            post_id=str(post_id or ""),
            comment_id=str(comment_id or ""),
            parent_id=_first_text(data, ("parent_id", "target_parent_id")),
            parent_type=_first_text(data, ("parent_type",)),
            author_id=_first_text(data, ("author_id", "target_author_id")),
            topic=_first_text(data, ("topic", "subreddit", "community")),
            subreddit=_first_text(data, ("subreddit", "community")),
            created_utc=data.get("created_utc", data.get("created_at")),
            score=_coerce_float(data.get("score"), 0.0),
            num_comments=_coerce_int(data.get("num_comments"), 0),
            level=level,
            age_hours=_optional_float(data.get("age_hours")),
            is_notification=bool(
                data.get("is_notification")
                or data.get("from_notification")
                or data.get("notification_id")
            ),
            already_seen=bool(data.get("already_seen") or data.get("seen")),
            participated=bool(data.get("participated") or data.get("has_participated")),
            title=str(data.get("title", "")),
            body=str(data.get("body", "")),
            metadata=metadata,
        )


@dataclass
class FamilyPolicyConfig:
    """Tunable policy defaults for one social behavior family."""

    family_id: str
    anchors_per_bot: Tuple[int, int] = (1, 3)
    population_noise: float = 0.14
    anchor_noise: float = 0.10
    context_noise: float = 0.18
    min_action_weight: float = 0.02
    max_action_weight: float = 0.95
    notification_boost: float = 1.0
    revisit_boost: float = 0.4
    context_boost: float = 0.6

    @classmethod
    def for_family(cls, family_id: str) -> "FamilyPolicyConfig":
        _validate_family(family_id)
        if family_id == FAMILY_G1:
            return cls(
                family_id=family_id,
                population_noise=0.12,
                notification_boost=0.35,
                revisit_boost=0.25,
                context_boost=0.30,
            )
        if family_id == FAMILY_G2:
            return cls(
                family_id=family_id,
                population_noise=0.08,
                anchor_noise=0.08,
                notification_boost=0.60,
                revisit_boost=0.45,
                context_boost=0.40,
            )
        return cls(
            family_id=family_id,
            population_noise=0.16,
            notification_boost=1.15,
            revisit_boost=0.75,
            context_boost=0.90,
        )


@dataclass
class SocialPersona:
    """Auditable behavior profile used by the mixed environment runner."""

    user_id: str
    family: str
    seed: int
    profile_source: str
    anchor_user_ids: Tuple[str, ...] = ()
    provider_model: str = ""
    post_tendency: float = 0.20
    reply_tendency: float = 0.55
    comment1_tendency: float = 0.35
    comment2_tendency: float = 0.20
    revisit_tendency: float = 0.12
    exploration_tendency: float = 0.45
    notification_reactivity: float = 0.35
    skip_tendency: float = 0.25
    topic_entropy_target: float = 1.8
    active_rhythm_type: str = "mixed"
    preferred_topics: Tuple[str, ...] = ()
    language_style_summary: str = "runner_generates_text"
    behavior_style_summary: str = ""
    anchor_weights: Tuple[float, ...] = ()
    context_sensitivity: float = 0.35
    controversy_tolerance: float = 0.30
    diversity_preference: float = 0.45
    memory_decay: float = 0.50
    rate_scale: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        return persona_to_row(self)


@dataclass
class ActionDecision:
    """Policy-layer action choice.  No generated text is stored here."""

    action_type: str
    target_id: str = ""
    target_type: str = ""
    target_post_id: str = ""
    target_comment_id: str = ""
    target_author_id: str = ""
    reason: str = ""
    score: float = 0.0
    sampled_probabilities: Dict[str, float] = field(default_factory=dict)
    notification_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        return decision_to_row(self)


def build_personas(
    family_id: str,
    n: int,
    seed: int,
    human_reference: Any = None,
    anchor_pool: Any = None,
    provider_models: Any = None,
) -> List[SocialPersona]:
    """Create personas for one social behavior family.

    Parameters
    ----------
    family_id:
        One of G1_population_calibrated, G2_anchor_behavior, or
        G3_contextual_interactive.
    n:
        Number of generated bot personas.
    seed:
        Deterministic seed for this family.
    human_reference:
        Optional human behavior records.  Accepts pandas DataFrame, list of
        dicts, dict, CSV/JSON/JSONL path, or None.
    anchor_pool:
        Optional human anchor records for G2.  If omitted, human_reference is
        reused.  Text and exact targets are never copied by this module.
    provider_models:
        Optional list/dict used only as persona audit metadata.  It is mixed
        across families and should not be used as a detection feature.
    """

    _validate_family(family_id)
    if n < 0:
        raise ValueError("n must be non-negative")

    rng = random.Random(seed)
    config = FamilyPolicyConfig.for_family(family_id)
    reference_rows = _records_from_any(human_reference) or _default_human_rows()
    anchor_rows = _records_from_any(anchor_pool) or reference_rows

    personas: List[SocialPersona] = []
    for index in range(n):
        if family_id == FAMILY_G1:
            persona = _build_g1_persona(
                index, seed, rng, config, reference_rows, provider_models
            )
        elif family_id == FAMILY_G2:
            persona = _build_g2_persona(
                index, seed, rng, config, anchor_rows, provider_models
            )
        else:
            persona = _build_g3_persona(
                index, seed, rng, config, reference_rows, anchor_rows, provider_models
            )
        personas.append(persona)
    return personas


def choose_action(
    persona: SocialPersona,
    candidates: Iterable[Any],
    memory: Any,
    rng: Any,
) -> ActionDecision:
    """Choose skip/post/comment1/comment2/revisit and a target id.

    The policy uses persona tendencies, visible candidates, lightweight memory,
    notifications, and family-specific weighting.  It does not call an LLM and
    does not produce text.
    """

    if persona.family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported persona family: {persona.family}")

    rng = rng if rng is not None else random.Random(persona.seed)
    candidate_views = [CandidateView.from_any(item) for item in candidates or []]
    candidate_views.extend(_notification_candidates(memory))
    candidate_views = _dedupe_candidates(candidate_views)

    memory_seen = _text_set(_mem_get(memory, "seen_target_ids", ()))
    recent_targets = _text_set(_mem_get(memory, "recent_target_ids", ()))
    participated_posts = _text_set(_mem_get(memory, "participated_post_ids", ()))
    participated_comments = _text_set(
        _mem_get(memory, "participated_comment_ids", ())
    )
    notification_count = len(_as_list(_mem_get(memory, "notifications", ()))) + len(
        _as_list(_mem_get(memory, "notification_queue", ()))
    )

    for candidate in candidate_views:
        if candidate.target_id in memory_seen:
            candidate.already_seen = True
        if candidate.post_id in participated_posts or candidate.comment_id in participated_comments:
            candidate.participated = True

    comment1_targets = [
        c for c in candidate_views if _can_comment1(c, recent_targets)
    ]
    comment2_targets = [
        c for c in candidate_views if _can_comment2(c, recent_targets)
    ]
    revisit_targets = [
        c
        for c in candidate_views
        if _can_revisit(c, participated_posts, participated_comments, recent_targets)
    ]

    weights = _action_weights(
        persona,
        has_comment1=bool(comment1_targets),
        has_comment2=bool(comment2_targets),
        has_revisit=bool(revisit_targets),
        notification_count=notification_count,
        candidate_count=len(candidate_views),
    )
    action = _weighted_choice(list(weights), [weights[key] for key in weights], rng)

    if action == "skip":
        return ActionDecision(
            action_type="skip",
            reason=_skip_reason(persona, candidate_views, notification_count),
            sampled_probabilities=_normalize_weights(weights),
        )

    if action == "post":
        topic = _choose_post_topic(persona, candidate_views, rng)
        return ActionDecision(
            action_type="post",
            target_id=topic,
            target_type="topic",
            reason="self_initiated_post_from_persona_topic_mix",
            score=weights.get("post", 0.0),
            sampled_probabilities=_normalize_weights(weights),
            metadata={"topic": topic},
        )

    targets_by_action = {
        "comment1": comment1_targets,
        "comment2": comment2_targets,
        "revisit": revisit_targets,
    }
    selected = _select_candidate_for_action(
        persona, action, targets_by_action.get(action, []), rng
    )
    if selected is None:
        fallback = "post" if persona.post_tendency >= persona.skip_tendency else "skip"
        if fallback == "post":
            topic = _choose_post_topic(persona, candidate_views, rng)
            return ActionDecision(
                action_type="post",
                target_id=topic,
                target_type="topic",
                reason=f"fallback_from_{action}_without_valid_target",
                score=weights.get("post", 0.0),
                sampled_probabilities=_normalize_weights(weights),
                metadata={"topic": topic},
            )
        return ActionDecision(
            action_type="skip",
            reason=f"fallback_from_{action}_without_valid_target",
            sampled_probabilities=_normalize_weights(weights),
        )

    return _decision_from_candidate(
        action,
        selected,
        reason=_action_reason(persona, action, selected),
        score=_candidate_score(persona, selected, action),
        probabilities=_normalize_weights(weights),
    )


def persona_to_row(persona: SocialPersona) -> Dict[str, Any]:
    """Convert a persona into a stable persona_profiles.csv-style row."""

    return {
        "user_id": persona.user_id,
        "family": persona.family,
        "profile_source": persona.profile_source,
        "anchor_user_ids": "|".join(persona.anchor_user_ids),
        "post_tendency": round(persona.post_tendency, 6),
        "reply_tendency": round(persona.reply_tendency, 6),
        "revisit_tendency": round(persona.revisit_tendency, 6),
        "exploration_tendency": round(persona.exploration_tendency, 6),
        "notification_reactivity": round(persona.notification_reactivity, 6),
        "topic_entropy_target": round(persona.topic_entropy_target, 6),
        "active_rhythm_type": persona.active_rhythm_type,
        "language_style_summary": persona.language_style_summary,
        "behavior_style_summary": persona.behavior_style_summary,
        "provider_model": persona.provider_model,
        "comment1_tendency": round(persona.comment1_tendency, 6),
        "comment2_tendency": round(persona.comment2_tendency, 6),
        "skip_tendency": round(persona.skip_tendency, 6),
        "preferred_topics": "|".join(persona.preferred_topics),
        "anchor_weights": "|".join(f"{weight:.6f}" for weight in persona.anchor_weights),
        "context_sensitivity": round(persona.context_sensitivity, 6),
        "controversy_tolerance": round(persona.controversy_tolerance, 6),
        "diversity_preference": round(persona.diversity_preference, 6),
        "memory_decay": round(persona.memory_decay, 6),
        "rate_scale": round(persona.rate_scale, 6),
        "seed": persona.seed,
        "metadata_json": _json_dumps(persona.metadata),
    }


def decision_to_row(decision: ActionDecision) -> Dict[str, Any]:
    """Convert an action decision into an action_trace-friendly row."""

    return {
        "action_type": decision.action_type,
        "target_id": decision.target_id,
        "target_type": decision.target_type,
        "target_post_id": decision.target_post_id,
        "target_comment_id": decision.target_comment_id,
        "target_author_id": decision.target_author_id,
        "reason": decision.reason,
        "score": round(decision.score, 6),
        "notification_id": decision.notification_id,
        "sampled_probabilities_json": _json_dumps(decision.sampled_probabilities),
        "metadata_json": _json_dumps(decision.metadata),
    }


def _build_g1_persona(
    index: int,
    seed: int,
    rng: random.Random,
    config: FamilyPolicyConfig,
    reference_rows: Sequence[Mapping[str, Any]],
    provider_models: Any,
) -> SocialPersona:
    base = dict(rng.choice(list(reference_rows)))
    metrics = _profile_metrics(base)
    metrics = _jitter_profile(metrics, rng, config.population_noise)
    user_id = _generated_user_id(FAMILY_G1, seed, index)
    topics = _topics_from_row(base) or _sample_default_topics(rng)
    active_rhythm = _active_rhythm(base, rng)

    return SocialPersona(
        user_id=user_id,
        family=FAMILY_G1,
        seed=_child_seed(seed, index, "g1"),
        profile_source="human_population_sample",
        provider_model=_select_provider(provider_models, index, rng),
        preferred_topics=topics,
        active_rhythm_type=active_rhythm,
        behavior_style_summary=(
            "population-calibrated macro rates with per-person sampling noise"
        ),
        context_sensitivity=_bounded(0.20 + 0.35 * metrics["exploration"], 0.05, 0.75),
        diversity_preference=_bounded(0.35 + 0.35 * metrics["exploration"], 0.05, 0.90),
        controversy_tolerance=_bounded(0.20 + 0.25 * rng.random(), 0.05, 0.70),
        memory_decay=_bounded(0.45 + 0.25 * rng.random(), 0.10, 0.90),
        rate_scale=_bounded(_lognormal(rng, 0.0, 0.25), 0.55, 1.80),
        metadata={"reference_row_id": _row_identifier(base)},
        **_persona_metric_kwargs(metrics),
    )


def _build_g2_persona(
    index: int,
    seed: int,
    rng: random.Random,
    config: FamilyPolicyConfig,
    anchor_rows: Sequence[Mapping[str, Any]],
    provider_models: Any,
) -> SocialPersona:
    min_anchors, max_anchors = config.anchors_per_bot
    count = rng.randint(min_anchors, max_anchors)
    anchors = [dict(rng.choice(list(anchor_rows))) for _ in range(count)]
    raw_weights = [0.3 + rng.random() for _ in anchors]
    weight_sum = sum(raw_weights) or 1.0
    weights = [value / weight_sum for value in raw_weights]

    metrics = _weighted_anchor_metrics(anchors, weights)
    metrics = _jitter_profile(metrics, rng, config.anchor_noise)
    anchor_ids = tuple(_anchor_identifier(row, pos) for pos, row in enumerate(anchors))
    topics = _merge_anchor_topics(anchors, rng)

    return SocialPersona(
        user_id=_generated_user_id(FAMILY_G2, seed, index),
        family=FAMILY_G2,
        seed=_child_seed(seed, index, "g2"),
        profile_source="human_anchor_behavior_profile",
        anchor_user_ids=anchor_ids,
        provider_model=_select_provider(provider_models, index, rng),
        preferred_topics=topics,
        active_rhythm_type=_weighted_rhythm(anchors, weights, rng),
        behavior_style_summary=(
            "anchor-style action mix, rhythm, topic breadth, and revisit habits; "
            "no text or exact target copying"
        ),
        anchor_weights=tuple(weights),
        context_sensitivity=_bounded(0.18 + 0.25 * metrics["exploration"], 0.05, 0.70),
        controversy_tolerance=_bounded(0.15 + 0.30 * rng.random(), 0.05, 0.65),
        diversity_preference=_bounded(0.25 + 0.45 * metrics["exploration"], 0.05, 0.90),
        memory_decay=_bounded(0.35 + 0.30 * metrics["revisit"], 0.10, 0.90),
        rate_scale=_bounded(_lognormal(rng, 0.0, 0.18), 0.60, 1.65),
        metadata={"anchor_profile_count": len(anchors)},
        **_persona_metric_kwargs(metrics),
    )


def _build_g3_persona(
    index: int,
    seed: int,
    rng: random.Random,
    config: FamilyPolicyConfig,
    reference_rows: Sequence[Mapping[str, Any]],
    anchor_rows: Sequence[Mapping[str, Any]],
    provider_models: Any,
) -> SocialPersona:
    base = dict(rng.choice(list(reference_rows)))
    metrics = _profile_metrics(base)
    metrics = _jitter_profile(metrics, rng, config.context_noise)

    # G3 still respects human-like rates, but shifts the path: browse context,
    # notifications, and revisits have a larger effect on final action choice.
    metrics["post"] = _bounded(metrics["post"] + rng.uniform(-0.03, 0.08))
    metrics["revisit"] = _bounded(metrics["revisit"] + rng.uniform(0.08, 0.22))
    metrics["notification"] = _bounded(
        max(metrics["notification"], 0.35) + rng.uniform(0.05, 0.28)
    )
    metrics["exploration"] = _bounded(
        max(metrics["exploration"], 0.35) + rng.uniform(0.02, 0.22)
    )
    metrics = _rebalance_action_metrics(metrics)

    optional_anchor: Tuple[str, ...] = ()
    if anchor_rows and rng.random() < 0.35:
        anchor = dict(rng.choice(list(anchor_rows)))
        optional_anchor = (_anchor_identifier(anchor, 0),)
        topics = _topics_from_row(anchor) or _topics_from_row(base)
        profile_source = "contextual_interactive_with_light_anchor_prior"
    else:
        topics = _topics_from_row(base)
        profile_source = "contextual_interactive_population_prior"

    return SocialPersona(
        user_id=_generated_user_id(FAMILY_G3, seed, index),
        family=FAMILY_G3,
        seed=_child_seed(seed, index, "g3"),
        profile_source=profile_source,
        anchor_user_ids=optional_anchor,
        provider_model=_select_provider(provider_models, index, rng),
        preferred_topics=topics or _sample_default_topics(rng),
        active_rhythm_type=_contextual_rhythm(base, rng),
        behavior_style_summary=(
            "context, notification, thread revisit, and environment feedback "
            "drive target choice"
        ),
        anchor_weights=(1.0,) if optional_anchor else (),
        context_sensitivity=_bounded(0.52 + 0.35 * rng.random(), 0.20, 0.95),
        controversy_tolerance=_bounded(0.25 + 0.45 * rng.random(), 0.05, 0.85),
        diversity_preference=_bounded(0.40 + 0.35 * rng.random(), 0.10, 0.95),
        memory_decay=_bounded(0.25 + 0.35 * rng.random(), 0.05, 0.85),
        rate_scale=_bounded(_lognormal(rng, 0.0, 0.30), 0.55, 1.95),
        metadata={"environment_feedback": "strong"},
        **_persona_metric_kwargs(metrics),
    )


def _action_weights(
    persona: SocialPersona,
    has_comment1: bool,
    has_comment2: bool,
    has_revisit: bool,
    notification_count: int,
    candidate_count: int,
) -> Dict[str, float]:
    config = FamilyPolicyConfig.for_family(persona.family)
    candidate_factor = math.log1p(candidate_count) / math.log(12.0) if candidate_count else 0.0
    notification_factor = min(1.0, notification_count / 3.0)

    weights = {
        "skip": persona.skip_tendency,
        "post": persona.post_tendency * (1.0 + 0.20 * persona.exploration_tendency),
        "comment1": persona.comment1_tendency if has_comment1 else 0.0,
        "comment2": persona.comment2_tendency if has_comment2 else 0.0,
        "revisit": persona.revisit_tendency if has_revisit else 0.0,
    }

    if persona.family == FAMILY_G1:
        weights["skip"] *= 1.0 + 0.20 * max(0.0, 0.4 - candidate_factor)
        weights["comment1"] *= 1.0 + 0.20 * candidate_factor
        weights["comment2"] *= 1.0 + 0.10 * candidate_factor
        weights["revisit"] *= 1.0 + config.revisit_boost * notification_factor
    elif persona.family == FAMILY_G2:
        weights["comment1"] *= 0.95 + 0.25 * persona.diversity_preference
        weights["comment2"] *= 0.95 + 0.35 * persona.revisit_tendency
        weights["revisit"] *= 1.0 + config.revisit_boost + 0.35 * notification_factor
        weights["post"] *= 0.90 + 0.25 * persona.exploration_tendency
    else:
        weights["skip"] *= max(0.35, 1.0 - 0.45 * candidate_factor)
        weights["comment1"] *= 1.0 + config.context_boost * candidate_factor
        weights["comment2"] *= 1.0 + 0.45 * candidate_factor + 0.65 * notification_factor
        weights["revisit"] *= 1.0 + config.revisit_boost + config.notification_boost * notification_factor
        weights["post"] *= 1.0 + 0.25 * max(0.0, 0.5 - candidate_factor)

    for action, value in list(weights.items()):
        weights[action] = max(config.min_action_weight, float(value))
    return weights


def _select_candidate_for_action(
    persona: SocialPersona,
    action: str,
    candidates: Sequence[CandidateView],
    rng: Any,
) -> Optional[CandidateView]:
    if not candidates:
        return None
    scores = [_candidate_score(persona, candidate, action) for candidate in candidates]
    return _weighted_choice(list(candidates), scores, rng)


def _candidate_score(
    persona: SocialPersona,
    candidate: CandidateView,
    action: str,
) -> float:
    topic_match = 1.0 if candidate.topic and candidate.topic in persona.preferred_topics else 0.0
    novelty = 0.25 if candidate.already_seen else 1.0
    participated = 1.0 if candidate.participated else 0.0
    notification = 1.0 if candidate.is_notification else 0.0
    freshness = _freshness_score(candidate.age_hours)
    popularity = _popularity_score(candidate.score, candidate.num_comments)
    depth_fit = _depth_fit(persona, candidate, action)

    score = 0.20 + novelty
    score += persona.context_sensitivity * (0.60 * topic_match + 0.35 * popularity)
    score += persona.exploration_tendency * (0.35 * freshness)
    score += persona.notification_reactivity * (1.25 * notification)
    score += persona.revisit_tendency * (0.80 * participated)
    score += persona.diversity_preference * _author_diversity_bonus(candidate)
    score += depth_fit

    if action == "revisit":
        score += 0.75 * participated + 0.85 * notification
    elif action == "comment2":
        score += 0.45 * notification + 0.25 * participated
    elif action == "comment1":
        score += 0.25 * topic_match + 0.20 * freshness

    if persona.family == FAMILY_G1:
        score += 0.20 * popularity + 0.15 * freshness
    elif persona.family == FAMILY_G2:
        score += 0.35 * topic_match + 0.25 * participated
        score *= 0.90 + 0.20 * persona.rate_scale
    else:
        score += 0.45 * freshness + 0.45 * popularity + 0.65 * notification
        if candidate.num_comments >= 8:
            score += 0.25 * persona.controversy_tolerance

    return max(0.001, score)


def _decision_from_candidate(
    action: str,
    candidate: CandidateView,
    reason: str,
    score: float,
    probabilities: Dict[str, float],
) -> ActionDecision:
    notification_id = _first_text(candidate.metadata, ("notification_id",))
    return ActionDecision(
        action_type=action,
        target_id=candidate.target_id,
        target_type=candidate.target_type,
        target_post_id=candidate.post_id,
        target_comment_id=candidate.comment_id,
        target_author_id=candidate.author_id,
        reason=reason,
        score=score,
        sampled_probabilities=probabilities,
        notification_id=notification_id,
        metadata={
            "topic": candidate.topic,
            "level": candidate.level,
            "is_notification": candidate.is_notification,
            "participated": candidate.participated,
        },
    )


def _action_reason(persona: SocialPersona, action: str, candidate: CandidateView) -> str:
    if candidate.is_notification and action in {"comment2", "revisit"}:
        return f"{persona.family}: notification-reactive {action}"
    if candidate.participated and action == "revisit":
        return f"{persona.family}: revisit participated thread"
    if persona.family == FAMILY_G1:
        return f"{persona.family}: population-calibrated candidate fit"
    if persona.family == FAMILY_G2:
        return f"{persona.family}: anchor-style behavior fit without target copying"
    return f"{persona.family}: context and environment feedback fit"


def _skip_reason(
    persona: SocialPersona,
    candidates: Sequence[CandidateView],
    notification_count: int,
) -> str:
    if not candidates:
        return f"{persona.family}: no visible candidate and no self-post impulse"
    if notification_count and persona.notification_reactivity < 0.25:
        return f"{persona.family}: low notification reactivity this step"
    return f"{persona.family}: sampled silence/browsing-only behavior"


def _can_comment1(candidate: CandidateView, recent_targets: set) -> bool:
    if candidate.target_id in recent_targets:
        return False
    if candidate.target_type == "post":
        return bool(candidate.target_id or candidate.post_id)
    return bool(candidate.post_id and not candidate.comment_id and candidate.level <= 0)


def _can_comment2(candidate: CandidateView, recent_targets: set) -> bool:
    if candidate.target_id in recent_targets:
        return False
    if candidate.target_type != "comment" and not candidate.comment_id:
        return False
    return candidate.level <= 1


def _can_revisit(
    candidate: CandidateView,
    participated_posts: set,
    participated_comments: set,
    recent_targets: set,
) -> bool:
    if candidate.target_id in recent_targets:
        return False
    if candidate.is_notification or candidate.participated:
        return True
    return candidate.post_id in participated_posts or candidate.comment_id in participated_comments


def _choose_post_topic(
    persona: SocialPersona,
    candidates: Sequence[CandidateView],
    rng: Any,
) -> str:
    topic_weights: Dict[str, float] = {}
    for idx, topic in enumerate(persona.preferred_topics):
        topic_weights[topic] = topic_weights.get(topic, 0.0) + max(0.2, 1.0 - 0.15 * idx)
    if persona.family == FAMILY_G3:
        for candidate in candidates:
            if candidate.topic:
                topic_weights[candidate.topic] = topic_weights.get(candidate.topic, 0.0) + (
                    0.25 + 0.25 * persona.context_sensitivity
                )
    if not topic_weights:
        for topic in _DEFAULT_TOPICS:
            topic_weights[topic] = 1.0
    topics = list(topic_weights)
    return _weighted_choice(topics, [topic_weights[topic] for topic in topics], rng)


def _notification_candidates(memory: Any) -> List[CandidateView]:
    rows = _as_list(_mem_get(memory, "notifications", ())) + _as_list(
        _mem_get(memory, "notification_queue", ())
    )
    candidates: List[CandidateView] = []
    for row in rows:
        if isinstance(row, CandidateView):
            candidate = row
            candidate.is_notification = True
        elif isinstance(row, Mapping):
            data = dict(row)
            data["is_notification"] = True
            candidate = CandidateView.from_any(data)
        else:
            continue
        candidates.append(candidate)
    return candidates


def _dedupe_candidates(candidates: Sequence[CandidateView]) -> List[CandidateView]:
    seen = set()
    deduped: List[CandidateView] = []
    for candidate in candidates:
        key = (
            candidate.target_id,
            candidate.target_type,
            candidate.post_id,
            candidate.comment_id,
            candidate.is_notification,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _records_from_any(data: Any) -> List[Mapping[str, Any]]:
    if data is None:
        return []
    if pd is not None and isinstance(data, pd.DataFrame):  # type: ignore[attr-defined]
        return [dict(row) for row in data.to_dict(orient="records")]
    if isinstance(data, (str, Path)):
        return _records_from_path(Path(data))
    if isinstance(data, Mapping):
        if isinstance(data.get("records"), Sequence):
            return [dict(row) for row in data["records"] if isinstance(row, Mapping)]
        if data and all(isinstance(value, Sequence) and not isinstance(value, str) for value in data.values()):
            keys = list(data)
            length = min(len(data[key]) for key in keys)
            return [{key: data[key][idx] for key in keys} for idx in range(length)]
        return [dict(data)]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        records = []
        for row in data:
            if hasattr(row, "__dataclass_fields__"):
                row = asdict(row)
            if isinstance(row, Mapping):
                records.append(dict(row))
            else:
                records.append({"user_id": str(row)})
        return records
    return []


def _records_from_path(path: Path) -> List[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if pd is not None and suffix in {".csv", ".tsv", ".json", ".jsonl", ".ndjson"}:
        if suffix == ".csv":
            return pd.read_csv(path).to_dict(orient="records")  # type: ignore[union-attr]
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t").to_dict(orient="records")  # type: ignore[union-attr]
        if suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            return _records_from_any(value)
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    if suffix == ".tsv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh, delimiter="\t"))
    if suffix == ".json":
        return _records_from_any(json.loads(path.read_text(encoding="utf-8")))
    if suffix in {".jsonl", ".ndjson"}:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raise ValueError(f"Unsupported reference file suffix: {path.suffix}")


def _profile_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    post = _metric(
        row,
        ("post_tendency", "post_rate", "posts_per_window", "post_ratio", "posts"),
        0.18,
    )
    comment1 = _metric(
        row,
        (
            "comment1_tendency",
            "comment1_rate",
            "top_level_comment_rate",
            "c1_rate",
            "comment_rate",
        ),
        0.34,
    )
    comment2 = _metric(
        row,
        (
            "comment2_tendency",
            "comment2_rate",
            "second_level_reply_rate",
            "reply_rate",
            "c2_rate",
        ),
        0.21,
    )
    revisit = _metric(
        row,
        ("revisit_tendency", "revisit_rate", "return_rate", "thread_return_rate"),
        0.13,
    )
    exploration = _metric(
        row,
        (
            "exploration_tendency",
            "exploration_rate",
            "unique_author_rate",
            "interaction_diversity",
        ),
        0.45,
    )
    notification = _metric(
        row,
        ("notification_reactivity", "notification_reply_rate", "inbound_reply_rate"),
        0.28,
    )
    skip = _metric(row, ("skip_tendency", "silence_rate", "inactive_rate"), 0.24)
    topic_entropy = _metric(
        row,
        ("topic_entropy_target", "topic_entropy", "subreddit_entropy"),
        1.8,
        upper=5.0,
    )

    metrics = {
        "post": post,
        "comment1": comment1,
        "comment2": comment2,
        "revisit": revisit,
        "exploration": exploration,
        "notification": notification,
        "skip": skip,
        "topic_entropy": topic_entropy,
    }
    return _rebalance_action_metrics(metrics)


def _metric(
    row: Mapping[str, Any],
    names: Sequence[str],
    default: float,
    upper: float = 1.0,
) -> float:
    value = _first_present(row, names)
    numeric = _coerce_float(value, default)
    if upper <= 1.0:
        if numeric > 1.0:
            numeric = numeric / (numeric + 5.0)
        return _bounded(numeric)
    return _bounded(numeric, 0.0, upper)


def _jitter_profile(
    metrics: Mapping[str, float],
    rng: random.Random,
    noise: float,
) -> Dict[str, float]:
    jittered = dict(metrics)
    for key in ("post", "comment1", "comment2", "revisit", "skip"):
        jittered[key] = _bounded(jittered[key] * _lognormal(rng, 0.0, noise))
    for key in ("exploration", "notification"):
        jittered[key] = _bounded(jittered[key] + rng.uniform(-noise, noise))
    jittered["topic_entropy"] = _bounded(
        jittered["topic_entropy"] + rng.uniform(-0.35, 0.35),
        0.2,
        5.0,
    )
    return _rebalance_action_metrics(jittered)


def _rebalance_action_metrics(metrics: Mapping[str, float]) -> Dict[str, float]:
    result = dict(metrics)
    action_sum = sum(max(0.001, result[key]) for key in ("post", "comment1", "comment2", "revisit", "skip"))
    scale = min(1.20, max(0.65, action_sum))
    for key in ("post", "comment1", "comment2", "revisit", "skip"):
        result[key] = _bounded(max(0.001, result[key]) / action_sum * scale, 0.01, 0.90)
    result["reply"] = _bounded(result["comment1"] + result["comment2"], 0.01, 0.95)
    result["exploration"] = _bounded(result.get("exploration", 0.45), 0.01, 0.95)
    result["notification"] = _bounded(result.get("notification", 0.28), 0.01, 0.95)
    result["topic_entropy"] = _bounded(result.get("topic_entropy", 1.8), 0.2, 5.0)
    return result


def _weighted_anchor_metrics(
    anchors: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
) -> Dict[str, float]:
    profiles = [_profile_metrics(anchor) for anchor in anchors]
    keys = profiles[0].keys()
    return {
        key: sum(profile[key] * weight for profile, weight in zip(profiles, weights))
        for key in keys
    }


def _persona_metric_kwargs(metrics: Mapping[str, float]) -> Dict[str, float]:
    return {
        "post_tendency": metrics["post"],
        "reply_tendency": metrics["reply"],
        "comment1_tendency": metrics["comment1"],
        "comment2_tendency": metrics["comment2"],
        "revisit_tendency": metrics["revisit"],
        "exploration_tendency": metrics["exploration"],
        "notification_reactivity": metrics["notification"],
        "skip_tendency": metrics["skip"],
        "topic_entropy_target": metrics["topic_entropy"],
    }


_DEFAULT_TOPICS = (
    "askreddit",
    "technology",
    "science",
    "gaming",
    "movies",
    "books",
    "fitness",
    "news",
    "personalfinance",
    "learnpython",
)


def _default_human_rows() -> List[Mapping[str, Any]]:
    return [
        {
            "user_id": "default_human_quiet_reader",
            "post_tendency": 0.08,
            "comment1_tendency": 0.24,
            "comment2_tendency": 0.12,
            "revisit_tendency": 0.08,
            "skip_tendency": 0.48,
            "exploration_tendency": 0.35,
            "notification_reactivity": 0.22,
            "topic_entropy_target": 1.2,
            "active_rhythm_type": "bursty_evening",
            "preferred_topics": "books|movies|askreddit",
        },
        {
            "user_id": "default_human_commenter",
            "post_tendency": 0.12,
            "comment1_tendency": 0.44,
            "comment2_tendency": 0.20,
            "revisit_tendency": 0.10,
            "skip_tendency": 0.18,
            "exploration_tendency": 0.55,
            "notification_reactivity": 0.30,
            "topic_entropy_target": 2.1,
            "active_rhythm_type": "daily_regular",
            "preferred_topics": "technology|science|news",
        },
        {
            "user_id": "default_human_discussion_returner",
            "post_tendency": 0.10,
            "comment1_tendency": 0.28,
            "comment2_tendency": 0.32,
            "revisit_tendency": 0.20,
            "skip_tendency": 0.10,
            "exploration_tendency": 0.38,
            "notification_reactivity": 0.56,
            "topic_entropy_target": 1.6,
            "active_rhythm_type": "reply_bursty",
            "preferred_topics": "gaming|technology|askreddit",
        },
        {
            "user_id": "default_human_poster",
            "post_tendency": 0.30,
            "comment1_tendency": 0.24,
            "comment2_tendency": 0.14,
            "revisit_tendency": 0.14,
            "skip_tendency": 0.18,
            "exploration_tendency": 0.48,
            "notification_reactivity": 0.42,
            "topic_entropy_target": 2.4,
            "active_rhythm_type": "weekend_bursty",
            "preferred_topics": "fitness|personalfinance|learnpython",
        },
    ]


def _topics_from_row(row: Mapping[str, Any]) -> Tuple[str, ...]:
    value = _first_present(
        row,
        (
            "preferred_topics",
            "top_topics",
            "topics",
            "subreddits",
            "active_subreddits",
            "topic",
            "subreddit",
        ),
    )
    if value is None:
        return ()
    if isinstance(value, str):
        raw_parts = value.replace(",", "|").replace(";", "|").split("|")
    elif isinstance(value, Sequence):
        raw_parts = [str(item) for item in value]
    else:
        raw_parts = [str(value)]
    topics = []
    for part in raw_parts:
        topic = part.strip()
        if topic and topic not in topics:
            topics.append(topic)
    return tuple(topics[:6])


def _sample_default_topics(rng: random.Random) -> Tuple[str, ...]:
    count = rng.randint(2, 4)
    return tuple(rng.sample(list(_DEFAULT_TOPICS), count))


def _merge_anchor_topics(
    anchors: Sequence[Mapping[str, Any]],
    rng: random.Random,
) -> Tuple[str, ...]:
    topics: List[str] = []
    for anchor in anchors:
        for topic in _topics_from_row(anchor):
            if topic not in topics:
                topics.append(topic)
    if not topics:
        return _sample_default_topics(rng)
    rng.shuffle(topics)
    return tuple(topics[:6])


def _active_rhythm(row: Mapping[str, Any], rng: random.Random) -> str:
    value = _first_text(
        row,
        ("active_rhythm_type", "active_rhythm", "rhythm", "activity_pattern"),
    )
    return value or rng.choice(
        ["daily_regular", "bursty_evening", "weekend_bursty", "reply_bursty", "mixed"]
    )


def _weighted_rhythm(
    anchors: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    rng: random.Random,
) -> str:
    rhythms = [_active_rhythm(anchor, rng) for anchor in anchors]
    return _weighted_choice(rhythms, weights, rng)


def _contextual_rhythm(row: Mapping[str, Any], rng: random.Random) -> str:
    if rng.random() < 0.55:
        return rng.choice(["notification_bursty", "browse_then_return", "event_driven"])
    return _active_rhythm(row, rng)


def _select_provider(provider_models: Any, index: int, rng: random.Random) -> str:
    if not provider_models:
        return ""
    if isinstance(provider_models, Mapping):
        values: List[str] = []
        for provider, models in provider_models.items():
            if isinstance(models, str):
                values.append(f"{provider}:{models}")
            elif isinstance(models, Sequence):
                values.extend(f"{provider}:{model}" for model in models)
            else:
                values.append(f"{provider}:{models}")
    elif isinstance(provider_models, Sequence) and not isinstance(provider_models, str):
        values = [str(item) for item in provider_models]
    else:
        values = [str(provider_models)]
    if not values:
        return ""
    return values[(index + rng.randrange(len(values))) % len(values)]


def _generated_user_id(family_id: str, seed: int, index: int) -> str:
    digest = hashlib.sha256(f"{family_id}:{seed}:{index}".encode("utf-8")).hexdigest()
    return f"u_{digest[:16]}"


def _child_seed(seed: int, index: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{index}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _row_identifier(row: Mapping[str, Any]) -> str:
    return _first_text(row, ("user_id", "author_id", "anchor_user_id", "id")) or ""


def _anchor_identifier(row: Mapping[str, Any], position: int) -> str:
    identifier = _row_identifier(row)
    if identifier:
        return identifier
    digest = hashlib.sha256(_json_dumps(dict(row)).encode("utf-8")).hexdigest()
    return f"anchor_{position}_{digest[:10]}"


def _mem_get(memory: Any, key: str, default: Any = None) -> Any:
    if memory is None:
        return default
    if isinstance(memory, Mapping):
        return memory.get(key, default)
    return getattr(memory, key, default)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _text_set(value: Any) -> set:
    return {str(item) for item in _as_list(value) if str(item)}


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _first_text(row: Mapping[str, Any], names: Sequence[str]) -> str:
    value = _first_present(row, names)
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _coerce_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    numeric = _coerce_float(value, math.nan)
    if math.isnan(numeric):
        return None
    return numeric


def _coerce_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bounded(value: float, low: float = 0.01, high: float = 0.95) -> float:
    return min(high, max(low, float(value)))


def _lognormal(rng: random.Random, mu: float, sigma: float) -> float:
    return math.exp(rng.gauss(mu, sigma))


def _weighted_choice(items: Sequence[Any], weights: Sequence[float], rng: Any) -> Any:
    if not items:
        raise ValueError("weighted choice requires at least one item")
    clean_weights = [max(0.0, float(weight)) for weight in weights]
    total = sum(clean_weights)
    if total <= 0:
        clean_weights = [1.0 for _ in items]
        total = float(len(items))

    if np is not None and isinstance(rng, getattr(np.random, "Generator", ())):  # type: ignore[attr-defined]
        probabilities = [weight / total for weight in clean_weights]
        return items[int(rng.choice(len(items), p=probabilities))]

    draw = float(rng.random() if hasattr(rng, "random") else random.random()) * total
    cumulative = 0.0
    for item, weight in zip(items, clean_weights):
        cumulative += weight
        if draw <= cumulative:
            return item
    return items[-1]


def _normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values()) or 1.0
    return {key: round(max(0.0, value) / total, 6) for key, value in weights.items()}


def _freshness_score(age_hours: Optional[float]) -> float:
    if age_hours is None:
        return 0.45
    return 1.0 / (1.0 + max(0.0, age_hours) / 12.0)


def _popularity_score(score: float, num_comments: int) -> float:
    score_part = math.log1p(max(0.0, score)) / math.log(101.0)
    comments_part = math.log1p(max(0, num_comments)) / math.log(51.0)
    return _bounded(0.55 * score_part + 0.45 * comments_part, 0.0, 1.0)


def _depth_fit(persona: SocialPersona, candidate: CandidateView, action: str) -> float:
    if action == "comment2":
        return 0.35 * persona.comment2_tendency + 0.15 * max(0, 1 - candidate.level)
    if action == "comment1":
        return 0.25 * persona.comment1_tendency
    return 0.10 * persona.revisit_tendency


def _author_diversity_bonus(candidate: CandidateView) -> float:
    if not candidate.author_id:
        return 0.05
    if candidate.metadata.get("author_seen_before"):
        return -0.10
    return 0.18


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _validate_family(family_id: str) -> None:
    if family_id not in SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unsupported family_id {family_id!r}; expected one of {SUPPORTED_FAMILIES}"
        )


__all__ = [
    "ActionDecision",
    "CandidateView",
    "FamilyPolicyConfig",
    "SocialPersona",
    "SUPPORTED_FAMILIES",
    "VALID_ACTIONS",
    "build_personas",
    "choose_action",
    "decision_to_row",
    "persona_to_row",
]
