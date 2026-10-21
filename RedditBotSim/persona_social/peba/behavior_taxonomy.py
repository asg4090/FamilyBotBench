from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


REDDIT_BEHAVIOR_LABELS = ("post", "comment1", "comment2", "revisit", "browse", "skip")
UNKNOWN_BEHAVIOR = "unknown"


@dataclass(frozen=True)
class BehaviorSpec:
    label: str
    display_name: str
    description: str
    aliases: tuple[str, ...] = ()
    action_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleContribution:
    behavior: str
    feature: str
    value: float | str
    score: float
    reason: str


@dataclass(frozen=True)
class BehaviorClassification:
    label: str
    confidence: float
    scores: dict[str, float]
    contributions: tuple[RuleContribution, ...]

    @property
    def ranking(self) -> list[str]:
        return [label for label, _ in sorted(self.scores.items(), key=lambda item: (-item[1], item[0]))]

    @property
    def evidence(self) -> dict[str, Any]:
        return {
            "top_features": [
                {
                    "behavior": item.behavior,
                    "feature": item.feature,
                    "value": item.value,
                    "score": round(item.score, 6),
                    "reason": item.reason,
                }
                for item in sorted(self.contributions, key=lambda entry: -entry.score)[:8]
            ]
        }


BEHAVIOR_TAXONOMY: dict[str, BehaviorSpec] = {
    "post": BehaviorSpec(
        label="post",
        display_name="Submission authoring",
        description="Creates a new Reddit submission in a subreddit.",
        aliases=("submission", "submit", "create_post", "post_submitter", "poster"),
        action_names=("post", "submit"),
    ),
    "comment1": BehaviorSpec(
        label="comment1",
        display_name="Top-level commenting",
        description="Replies directly to a Reddit submission.",
        aliases=("top_level_comment", "comment_to_post", "reply_to_post", "c1", "comment_1"),
        action_names=("comment1", "comment_1", "reply_to_post"),
    ),
    "comment2": BehaviorSpec(
        label="comment2",
        display_name="Nested replying",
        description="Replies to an existing first-level comment.",
        aliases=("nested_comment", "comment_to_comment", "reply_to_comment", "c2", "comment_2"),
        action_names=("comment2", "comment_2", "reply_to_comment"),
    ),
    "revisit": BehaviorSpec(
        label="revisit",
        display_name="Notification revisit",
        description="Returns to a prior thread or notification and replies there.",
        aliases=("notification", "notification_reply", "thread_return", "returner"),
        action_names=("revisit", "notification_revisit"),
    ),
    "browse": BehaviorSpec(
        label="browse",
        display_name="Browsing",
        description="Reads candidate posts or comments without necessarily writing.",
        aliases=("read", "explore", "thread_browser", "browse_decide", "lurking"),
        action_names=("browse", "read", "explore", "browse_decide"),
    ),
    "skip": BehaviorSpec(
        label="skip",
        display_name="No-op skipping",
        description="Browses or reaches a planning step but takes no visible action.",
        aliases=("no_action", "noop", "passive", "inactive", "browsed_but_no_action"),
        action_names=("skip", "stop", "none"),
    ),
}


_ALIAS_TO_LABEL: dict[str, str] = {}
for _label, _spec in BEHAVIOR_TAXONOMY.items():
    for _name in (_label, *_spec.aliases, *_spec.action_names):
        _ALIAS_TO_LABEL[str(_name).strip().lower().replace("-", "_").replace(" ", "_")] = _label


BEHAVIOR_CLASSES = (
    "lurker_skipper",
    "top_level_commenter",
    "thread_replier",
    "op_returner",
    "poster_starter",
    "debate_participant",
    "cross_topic_explorer",
    "community_regular",
    "weak_tie_interactor",
)


BEHAVIOR_CLASS_DESCRIPTIONS = {
    "lurker_skipper": "Mostly reads or skips; low visible action relative to browsing.",
    "top_level_commenter": "Usually replies directly to posts rather than long comment chains.",
    "thread_replier": "Often joins existing comment chains and responds to other commenters.",
    "op_returner": "Returns to prior threads or notifications and clarifies or follows up.",
    "poster_starter": "Starts new submissions or discussion prompts.",
    "debate_participant": "Pushes back, corrects framing, and sustains argumentative exchanges.",
    "cross_topic_explorer": "Moves across subreddits or topics instead of staying in one narrow lane.",
    "community_regular": "Shows repeated local community habits and reciprocal/familiar interactions.",
    "weak_tie_interactor": "Interacts with many distinct authors without forming a tight reciprocal cluster.",
}


ACTION_TO_BEHAVIOR_CLASS = {
    "post": "poster_starter",
    "comment1": "top_level_commenter",
    "comment2": "thread_replier",
    "revisit": "op_returner",
    "browse": "lurker_skipper",
    "skip": "lurker_skipper",
}


BROAD_TO_ACTION_LABEL = {
    "poster_starter": "post",
    "top_level_commenter": "comment1",
    "thread_replier": "comment2",
    "op_returner": "revisit",
    "lurker_skipper": "skip",
}


BROAD_BEHAVIOR_ALIASES = {
    "lurker": "lurker_skipper",
    "skipper": "lurker_skipper",
    "passive_reader": "lurker_skipper",
    "top_level_comment": "top_level_commenter",
    "comment_to_post": "top_level_commenter",
    "first_level_commenter": "top_level_commenter",
    "thread_reply": "thread_replier",
    "comment_chain_replier": "thread_replier",
    "nested_replier": "thread_replier",
    "op_returner": "op_returner",
    "notification_returner": "op_returner",
    "returner": "op_returner",
    "poster": "poster_starter",
    "starter": "poster_starter",
    "submission_starter": "poster_starter",
    "debater": "debate_participant",
    "argumentative": "debate_participant",
    "cross_topic": "cross_topic_explorer",
    "topic_explorer": "cross_topic_explorer",
    "regular": "community_regular",
    "community_regular": "community_regular",
    "weak_tie": "weak_tie_interactor",
    "wide_author_mix": "weak_tie_interactor",
}

for _label in BEHAVIOR_CLASSES:
    _ALIAS_TO_LABEL[_label] = _label
for _alias, _label in BROAD_BEHAVIOR_ALIASES.items():
    _ALIAS_TO_LABEL[str(_alias).strip().lower().replace("-", "_").replace(" ", "_")] = _label


FEATURE_RULES: dict[str, tuple[tuple[str, float, str], ...]] = {
    "post": (
        ("post_count", 1.6, "count"),
        ("submission_count", 1.6, "count"),
        ("posts_per_user", 1.5, "count"),
        ("generated_post_count", 1.2, "count"),
        ("mean_posts_per_bot", 1.2, "count"),
        ("post_share", 3.5, "share"),
        ("post_rate", 3.0, "share"),
        ("post_tendency", 2.0, "score"),
    ),
    "comment1": (
        ("comment1_count", 1.5, "count"),
        ("comment_1_count", 1.5, "count"),
        ("c1_per_user", 1.4, "count"),
        ("top_level_comment_count", 1.5, "count"),
        ("comment1_share", 3.4, "share"),
        ("comment_1_share", 3.4, "share"),
        ("reply_tendency", 1.2, "score"),
    ),
    "comment2": (
        ("comment2_count", 1.5, "count"),
        ("comment_2_count", 1.5, "count"),
        ("c2_per_user", 1.4, "count"),
        ("nested_reply_count", 1.5, "count"),
        ("sent_reply_count", 1.2, "count"),
        ("comment2_share", 3.4, "share"),
        ("comment_2_share", 3.4, "share"),
        ("reply_depth_score", 2.0, "score"),
    ),
    "revisit": (
        ("revisit_count", 1.7, "count"),
        ("notification_revisit_count", 1.7, "count"),
        ("notification_actions", 1.6, "count"),
        ("revisit_share", 3.6, "share"),
        ("candidate_source_notification_share", 2.4, "share"),
        ("notification_reactivity", 1.8, "score"),
        ("revisit_tendency", 2.0, "score"),
    ),
    "browse": (
        ("browse_count", 0.7, "count"),
        ("mean_browse_count", 0.8, "count"),
        ("active_window_count", 0.8, "count"),
        ("unique_threads_joined", 0.9, "count"),
        ("browse_share", 2.4, "share"),
        ("explore_share", 2.3, "share"),
        ("exploration_tendency", 1.8, "score"),
    ),
    "skip": (
        ("skip_count", 1.7, "count"),
        ("no_action_count", 1.7, "count"),
        ("skip_share", 3.6, "share"),
        ("no_action_share", 3.6, "share"),
        ("skip_bias", 2.4, "score"),
    ),
}


BROAD_FEATURE_RULES: dict[str, tuple[tuple[str, float, str], ...]] = {
    "lurker_skipper": (
        ("skip_count", 1.9, "count"),
        ("skip_share", 3.8, "share"),
        ("browse_item_count", 0.35, "count"),
        ("browse_events", 0.45, "count"),
        ("skip_bias", 2.4, "score"),
    ),
    "top_level_commenter": (
        ("comment1_count", 1.8, "count"),
        ("comment_1_count", 1.8, "count"),
        ("comment1_share", 3.8, "share"),
        ("reply_tendency", 1.2, "score"),
    ),
    "thread_replier": (
        ("comment2_count", 1.8, "count"),
        ("comment_2_count", 1.8, "count"),
        ("comment2_share", 3.8, "share"),
        ("sent_reply_count", 1.0, "count"),
        ("mean_comment_words", 0.35, "count"),
    ),
    "op_returner": (
        ("revisit_count", 2.0, "count"),
        ("revisit_share", 4.0, "share"),
        ("notification_source_share", 3.2, "share"),
        ("notification_reactivity", 1.7, "score"),
        ("revisit_tendency", 2.0, "score"),
    ),
    "poster_starter": (
        ("post_count", 1.9, "count"),
        ("submission_count", 1.9, "count"),
        ("post_share", 4.0, "share"),
        ("post_tendency", 2.0, "score"),
    ),
    "debate_participant": (
        ("comment_count", 0.9, "count"),
        ("comment2_count", 1.0, "count"),
        ("mean_comment_words", 0.45, "count"),
        ("debate_tendency", 2.5, "score"),
        ("reciprocal_neighbor_count_proxy", 0.9, "count"),
    ),
    "cross_topic_explorer": (
        ("unique_topics", 1.4, "count"),
        ("unique_subreddits", 1.3, "count"),
        ("explore_source_share", 3.0, "share"),
        ("exploration_tendency", 2.0, "score"),
    ),
    "community_regular": (
        ("own_history_source_share", 2.8, "share"),
        ("reciprocal_neighbor_count_proxy", 1.4, "count"),
        ("unique_inbound_authors", 0.7, "count"),
        ("active_window_count", 0.7, "count"),
    ),
    "weak_tie_interactor": (
        ("unique_target_authors", 1.4, "count"),
        ("unique_action_target_authors", 1.4, "count"),
        ("bot_to_human_share_out", 2.2, "share"),
        ("low_reciprocity_proxy", 2.0, "score"),
    ),
}


DIRECT_LABEL_KEYS = (
    "behavior",
    "behavior_label",
    "dominant_behavior",
    "target_behavior",
    "action",
    "action_type",
    "planned_action",
    "llm_selected_action",
)


def normalize_behavior_label(value: Any, default: str = UNKNOWN_BEHAVIOR) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIAS_TO_LABEL.get(text, default)


def behavior_from_action(action: Any, candidate_source: Any = None, default: str = UNKNOWN_BEHAVIOR) -> str:
    label = normalize_behavior_label(action, default=default)
    if label == "comment2" and normalize_behavior_label(candidate_source) == "revisit":
        return "revisit"
    if label != default:
        return label
    source = str(candidate_source or "").strip().lower()
    if source == "notification":
        return "revisit"
    if source in {"explore", "feed", "recommendation"}:
        return "browse"
    return default


def classify_rule_features(features: Mapping[str, Any], min_confidence: float = 0.0) -> BehaviorClassification:
    """Classify flexible rule/count features into one canonical Reddit behavior.

    Direct labels such as ``action=comment1`` win over numeric evidence. Numeric
    features are scored by log-count, share, or bounded score rules from
    ``FEATURE_RULES``. The returned confidence is a soft margin in ``[0, 1]``.
    """

    normalized = {_normalize_feature_key(key): value for key, value in features.items()}
    scores = {label: 0.0 for label in REDDIT_BEHAVIOR_LABELS}
    contributions: list[RuleContribution] = []

    direct_label = _direct_label(normalized)
    if direct_label != UNKNOWN_BEHAVIOR:
        direct_label = BROAD_TO_ACTION_LABEL.get(direct_label, direct_label)
    if direct_label in scores:
        scores[direct_label] += 10.0
        contributions.append(
            RuleContribution(
                behavior=direct_label,
                feature="direct_label",
                value=direct_label,
                score=10.0,
                reason="direct behavior/action label",
            )
        )

    candidate_source = normalized.get("candidate_source", normalized.get("source", ""))
    if str(candidate_source).strip().lower() == "notification":
        scores["revisit"] += 4.0
        contributions.append(
            RuleContribution(
                behavior="revisit",
                feature="candidate_source",
                value=str(candidate_source),
                score=4.0,
                reason="notification source implies revisit pressure",
            )
        )

    for behavior, rules in FEATURE_RULES.items():
        for feature, weight, mode in rules:
            if feature not in normalized:
                continue
            score = _feature_score(normalized[feature], weight, mode)
            if score <= 0.0:
                continue
            scores[behavior] += score
            contributions.append(
                RuleContribution(
                    behavior=behavior,
                    feature=feature,
                    value=normalized[feature],
                    score=score,
                    reason=f"{mode} rule",
                )
            )

    label, confidence = _best_label(scores)
    if confidence < min_confidence:
        label = UNKNOWN_BEHAVIOR
    return BehaviorClassification(
        label=label,
        confidence=confidence,
        scores={key: round(value, 6) for key, value in scores.items()},
        contributions=tuple(contributions),
    )


def behavior_counts_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    label_key: str = "behavior_label",
    weight_key: str | None = None,
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for record in records:
        if label_key in record:
            label = normalize_behavior_label(record.get(label_key))
        else:
            label = classify_rule_features(record).label
        if label == UNKNOWN_BEHAVIOR:
            continue
        weight = _as_float(record.get(weight_key), 1.0) if weight_key else 1.0
        if weight > 0:
            counts[label] += weight
    return {label: float(counts.get(label, 0.0)) for label in REDDIT_BEHAVIOR_LABELS}


def behavior_distribution_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    label_key: str = "behavior_label",
    weight_key: str | None = None,
) -> dict[str, float]:
    counts = behavior_counts_from_records(records, label_key=label_key, weight_key=weight_key)
    total = sum(counts.values())
    if total <= 0:
        return {label: 0.0 for label in REDDIT_BEHAVIOR_LABELS}
    return {label: value / total for label, value in counts.items()}


def normalize_broad_behavior_label(value: Any, default: str = UNKNOWN_BEHAVIOR) -> str:
    label = normalize_behavior_label(value, default=default)
    if label in BEHAVIOR_CLASSES:
        return label
    return ACTION_TO_BEHAVIOR_CLASS.get(label, default)


def behavior_label_descriptions() -> dict[str, str]:
    return dict(BEHAVIOR_CLASS_DESCRIPTIONS)


def classify_from_features(features: Mapping[str, Any], min_confidence: float = 0.0) -> BehaviorClassification:
    """Classify a trajectory summary into a broad PEBA behavior class."""

    normalized = _expand_feature_payload(features)
    scores = {label: 0.0 for label in BEHAVIOR_CLASSES}
    contributions: list[RuleContribution] = []

    for key in ("behavior_label", "behavior", "dominant_behavior"):
        if key not in normalized:
            continue
        direct = normalize_broad_behavior_label(normalized.get(key))
        if direct != UNKNOWN_BEHAVIOR:
            scores[direct] += 10.0
            contributions.append(
                RuleContribution(
                    behavior=direct,
                    feature=key,
                    value=normalized[key],
                    score=10.0,
                    reason="direct broad behavior label",
                )
            )
            break

    action_label = behavior_from_action(normalized.get("action") or normalized.get("action_type"), normalized.get("candidate_source"))
    broad_action = ACTION_TO_BEHAVIOR_CLASS.get(action_label)
    if broad_action:
        scores[broad_action] += 5.0
        contributions.append(
            RuleContribution(
                behavior=broad_action,
                feature="action",
                value=action_label,
                score=5.0,
                reason="direct action mapped to broad behavior",
            )
        )

    _add_derived_shares(normalized)
    for behavior, rules in BROAD_FEATURE_RULES.items():
        for feature, weight, mode in rules:
            if feature not in normalized:
                continue
            score = _feature_score(normalized[feature], weight, mode)
            if score <= 0.0:
                continue
            scores[behavior] += score
            contributions.append(
                RuleContribution(
                    behavior=behavior,
                    feature=feature,
                    value=normalized[feature],
                    score=score,
                    reason=f"{mode} rule",
                )
            )

    if _as_float(normalized.get("action_events"), 0.0) <= 0 and _as_float(normalized.get("browse_item_count"), 0.0) > 0:
        scores["lurker_skipper"] += 2.5
        contributions.append(
            RuleContribution(
                behavior="lurker_skipper",
                feature="browse_item_count",
                value=normalized.get("browse_item_count"),
                score=2.5,
                reason="browse without visible actions",
            )
        )

    reciprocal = _as_float(normalized.get("reciprocal_neighbor_count_proxy"), 0.0)
    unique_targets = _as_float(normalized.get("unique_target_authors"), 0.0)
    if unique_targets > 0 and reciprocal / max(1.0, unique_targets) <= 0.2:
        scores["weak_tie_interactor"] += 2.0
        contributions.append(
            RuleContribution(
                behavior="weak_tie_interactor",
                feature="reciprocal_neighbor_count_proxy",
                value=reciprocal,
                score=2.0,
                reason="many targets with low reciprocity",
            )
        )

    label, confidence = _best_label(scores)
    if confidence < min_confidence:
        label = UNKNOWN_BEHAVIOR
    return BehaviorClassification(
        label=label,
        confidence=confidence,
        scores={key: round(value, 6) for key, value in scores.items()},
        contributions=tuple(contributions),
    )


def taxonomy_rows() -> list[dict[str, Any]]:
    return [
        {
            "label": spec.label,
            "display_name": spec.display_name,
            "description": spec.description,
            "aliases": list(spec.aliases),
            "action_names": list(spec.action_names),
        }
        for spec in BEHAVIOR_TAXONOMY.values()
    ]


def broad_taxonomy_rows() -> list[dict[str, str]]:
    return [{"label": label, "description": BEHAVIOR_CLASS_DESCRIPTIONS[label]} for label in BEHAVIOR_CLASSES]


def _normalize_feature_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _expand_feature_payload(features: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {_normalize_feature_key(key): value for key, value in features.items()}
    action_counts = _json_mapping(normalized.get("action_counts_json"))
    for action, value in action_counts.items():
        label = behavior_from_action(action)
        if label != UNKNOWN_BEHAVIOR:
            normalized.setdefault(f"{label}_count", value)
    source_counts = _json_mapping(normalized.get("source_counts_json"))
    source_total = sum(max(0.0, _as_float(value)) for value in source_counts.values())
    for source, value in source_counts.items():
        key = _normalize_feature_key(f"{source}_source_share")
        if source_total > 0:
            normalized.setdefault(key, _as_float(value) / source_total)
    return normalized


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, Mapping):
        return {}
    return {str(key): item for key, item in parsed.items()}


def _add_derived_shares(features: dict[str, Any]) -> None:
    action_keys = {
        "post": "post_count",
        "comment1": "comment1_count",
        "comment2": "comment2_count",
        "revisit": "revisit_count",
        "skip": "skip_count",
    }
    total = _as_float(features.get("action_events"), 0.0)
    if total <= 0.0:
        total = sum(_as_float(features.get(key), 0.0) for key in action_keys.values())
    if total > 0.0:
        for label, key in action_keys.items():
            features.setdefault(f"{label}_share", _as_float(features.get(key), 0.0) / total)
    comment_count = _as_float(features.get("comment_count"), 0.0)
    if comment_count <= 0.0:
        comment_count = _as_float(features.get("comment1_count"), 0.0) + _as_float(features.get("comment2_count"), 0.0)
        features.setdefault("comment_count", comment_count)
    reciprocal = _as_float(features.get("reciprocal_neighbor_count_proxy"), 0.0)
    unique_targets = _as_float(features.get("unique_target_authors"), 0.0)
    if unique_targets > 0.0:
        features.setdefault("low_reciprocity_proxy", max(0.0, 1.0 - reciprocal / max(1.0, unique_targets)))


def _direct_label(features: Mapping[str, Any]) -> str:
    for key in DIRECT_LABEL_KEYS:
        if key in features:
            label = normalize_behavior_label(features.get(key))
            if label != UNKNOWN_BEHAVIOR:
                return label
    action = features.get("action")
    if action is not None:
        return behavior_from_action(action, features.get("candidate_source"))
    return UNKNOWN_BEHAVIOR


def _feature_score(value: Any, weight: float, mode: str) -> float:
    numeric = _as_float(value, 0.0)
    if numeric <= 0.0:
        return 0.0
    if mode == "count":
        return weight * math.log1p(numeric)
    if mode == "share":
        return weight * min(1.0, numeric)
    if mode == "score":
        return weight * min(1.0, numeric)
    return weight * numeric


def _best_label(scores: Mapping[str, float]) -> tuple[str, float]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ordered or ordered[0][1] <= 0.0:
        return UNKNOWN_BEHAVIOR, 0.0
    best_label, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = (best_score - second_score) / max(best_score, 1e-12)
    return best_label, max(0.0, min(1.0, confidence))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


__all__ = [
    "ACTION_TO_BEHAVIOR_CLASS",
    "BEHAVIOR_TAXONOMY",
    "BEHAVIOR_CLASSES",
    "BEHAVIOR_CLASS_DESCRIPTIONS",
    "BROAD_BEHAVIOR_ALIASES",
    "BROAD_TO_ACTION_LABEL",
    "BROAD_FEATURE_RULES",
    "DIRECT_LABEL_KEYS",
    "FEATURE_RULES",
    "REDDIT_BEHAVIOR_LABELS",
    "UNKNOWN_BEHAVIOR",
    "BehaviorClassification",
    "BehaviorSpec",
    "RuleContribution",
    "behavior_label_descriptions",
    "behavior_counts_from_records",
    "behavior_distribution_from_records",
    "behavior_from_action",
    "broad_taxonomy_rows",
    "classify_from_features",
    "classify_rule_features",
    "normalize_broad_behavior_label",
    "normalize_behavior_label",
    "taxonomy_rows",
]
