from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:
    from .behavior_taxonomy import (
        BEHAVIOR_CLASSES,
        REDDIT_BEHAVIOR_LABELS,
        UNKNOWN_BEHAVIOR,
        classify_from_features,
        classify_rule_features,
        normalize_broad_behavior_label,
        normalize_behavior_label,
    )
    from .peba_metrics import DistributionComparison, compare_distributions
except ImportError:  # pragma: no cover - supports direct module execution from this directory.
    from behavior_taxonomy import (
        BEHAVIOR_CLASSES,
        REDDIT_BEHAVIOR_LABELS,
        UNKNOWN_BEHAVIOR,
        classify_from_features,
        classify_rule_features,
        normalize_broad_behavior_label,
        normalize_behavior_label,
    )
    from peba_metrics import DistributionComparison, compare_distributions


DEFAULT_AGENT_ID_KEYS = ("user_id", "agent_id", "id", "user_name", "name")


@dataclass(frozen=True)
class ClassifiedAgent:
    agent_id: str
    index: int
    source_behavior: str
    confidence: float
    scores: dict[str, float]
    record: Mapping[str, Any]


@dataclass(frozen=True)
class BehaviorAssignment:
    agent_id: str
    agent_index: int
    source_behavior: str
    target_behavior: str
    priority: float
    source_gap: float
    target_gap: float
    confidence: float
    reason: str


@dataclass(frozen=True)
class CreditAssignmentPlan:
    assignments: tuple[BehaviorAssignment, ...]
    comparison: DistributionComparison
    overrepresented: tuple[str, ...]
    underrepresented: tuple[str, ...]
    classified_agents: tuple[ClassifiedAgent, ...]


def classify_agents(
    agents: Iterable[Mapping[str, Any]],
    *,
    agent_id_keys: Sequence[str] = DEFAULT_AGENT_ID_KEYS,
    label_key: str = "behavior_label",
    classifier: str = "action",
) -> list[ClassifiedAgent]:
    classified: list[ClassifiedAgent] = []
    for index, record in enumerate(agents):
        if label_key in record:
            if classifier == "broad":
                label = normalize_broad_behavior_label(record.get(label_key))
            else:
                label = normalize_behavior_label(record.get(label_key))
            confidence = 1.0 if label != UNKNOWN_BEHAVIOR else 0.0
            score_labels = BEHAVIOR_CLASSES if classifier == "broad" else REDDIT_BEHAVIOR_LABELS
            scores = {behavior: 0.0 for behavior in score_labels}
            if label in scores:
                scores[label] = 1.0
        else:
            result = classify_from_features(record) if classifier == "broad" else classify_rule_features(record)
            label = result.label
            confidence = result.confidence
            scores = result.scores
        classified.append(
            ClassifiedAgent(
                agent_id=_agent_id(record, index, agent_id_keys),
                index=index,
                source_behavior=label,
                confidence=confidence,
                scores=scores,
                record=record,
            )
        )
    return classified


def observed_distribution_from_agents(
    classified_agents: Iterable[ClassifiedAgent],
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    agents = list(classified_agents)
    if labels is None:
        inferred = sorted({agent.source_behavior for agent in agents if agent.source_behavior != UNKNOWN_BEHAVIOR})
        labels = tuple(inferred) if inferred else REDDIT_BEHAVIOR_LABELS
    counts = {label: 0.0 for label in labels}
    for agent in agents:
        if agent.source_behavior in counts:
            counts[agent.source_behavior] += 1.0
    total = sum(counts.values())
    if total <= 0.0:
        return {label: 0.0 for label in labels}
    return {label: count / total for label, count in counts.items()}


def assign_behavior_credits(
    agents: Iterable[Mapping[str, Any]],
    target_distribution: Mapping[str, float],
    *,
    observed_distribution: Mapping[str, float] | None = None,
    labels: Sequence[str] | None = None,
    max_assignments: int | None = None,
    min_gap: float = 0.0,
    seed: int | None = None,
    agent_id_keys: Sequence[str] = DEFAULT_AGENT_ID_KEYS,
    label_key: str = "behavior_label",
) -> CreditAssignmentPlan:
    """Assign agents from overrepresented behavior classes to missing targets.

    ``signed_gap`` is ``target - observed``. Agents are selected from negative
    gap classes and assigned to positive gap target behaviors. The function only
    returns a plan; it does not mutate agent records or runner state.
    """

    agent_records = list(agents)
    classifier = _classifier_for_target(target_distribution, labels)
    classified = classify_agents(agent_records, agent_id_keys=agent_id_keys, label_key=label_key, classifier=classifier)
    resolved_labels = _resolve_labels(target_distribution, observed_distribution, classified, labels, classifier)
    observed = observed_distribution or observed_distribution_from_agents(classified, resolved_labels)
    comparison = compare_distributions(observed, target_distribution, resolved_labels, gap_tolerance=min_gap)
    under = tuple(sorted(comparison.underrepresented, key=lambda label: (-comparison.signed_gap[label], label)))
    over = tuple(sorted(comparison.overrepresented, key=lambda label: (comparison.signed_gap[label], label)))

    candidates = select_over_behavior_agents(classified, comparison, labels=over)
    target_slots = _target_slots(comparison, len(classified), under, max_assignments=max_assignments)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(candidates)
        candidates.sort(key=lambda agent: (-_source_pressure(agent, comparison), -agent.confidence, agent.agent_id))

    assignments: list[BehaviorAssignment] = []
    for agent, target_behavior in zip(candidates, target_slots):
        source_gap = comparison.signed_gap.get(agent.source_behavior, 0.0)
        target_gap = comparison.signed_gap.get(target_behavior, 0.0)
        priority = abs(source_gap) * max(0.0, target_gap) * max(agent.confidence, 0.05)
        assignments.append(
            BehaviorAssignment(
                agent_id=agent.agent_id,
                agent_index=agent.index,
                source_behavior=agent.source_behavior,
                target_behavior=target_behavior,
                priority=priority,
                source_gap=source_gap,
                target_gap=target_gap,
                confidence=agent.confidence,
                reason=f"{agent.source_behavior} overrepresented; {target_behavior} underrepresented",
            )
        )

    assignments.sort(key=lambda item: (-item.priority, item.agent_index, item.target_behavior))
    return CreditAssignmentPlan(
        assignments=tuple(assignments),
        comparison=comparison,
        overrepresented=over,
        underrepresented=under,
        classified_agents=tuple(classified),
    )


def select_over_behavior_agents(
    classified_agents: Iterable[ClassifiedAgent],
    comparison: DistributionComparison,
    *,
    labels: Sequence[str] | None = None,
) -> list[ClassifiedAgent]:
    over_labels = set(labels or comparison.overrepresented)
    selected = [agent for agent in classified_agents if agent.source_behavior in over_labels]
    return sorted(selected, key=lambda agent: (-_source_pressure(agent, comparison), -agent.confidence, agent.agent_id))


def assignment_rows(plan: CreditAssignmentPlan) -> list[dict[str, Any]]:
    return [
        {
            "agent_id": item.agent_id,
            "agent_index": item.agent_index,
            "source_behavior": item.source_behavior,
            "target_behavior": item.target_behavior,
            "priority": item.priority,
            "source_gap": item.source_gap,
            "target_gap": item.target_gap,
            "confidence": item.confidence,
            "reason": item.reason,
        }
        for item in plan.assignments
    ]


def _target_slots(
    comparison: DistributionComparison,
    agent_count: int,
    under_labels: Sequence[str],
    *,
    max_assignments: int | None,
) -> list[str]:
    slots: list[str] = []
    for label in under_labels:
        gap = max(0.0, comparison.signed_gap.get(label, 0.0))
        count = int(round(gap * max(1, agent_count)))
        if gap > 0.0 and count <= 0:
            count = 1
        slots.extend([label] * count)

    if not slots and under_labels:
        slots.append(under_labels[0])

    slots.sort(key=lambda label: (-comparison.signed_gap.get(label, 0.0), label))
    if max_assignments is not None:
        slots = slots[: max(0, max_assignments)]
    return slots


def _source_pressure(agent: ClassifiedAgent, comparison: DistributionComparison) -> float:
    gap = comparison.signed_gap.get(agent.source_behavior, 0.0)
    score = agent.scores.get(agent.source_behavior, 0.0)
    return max(0.0, -gap) * max(score, 0.05)


def _agent_id(record: Mapping[str, Any], index: int, keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return f"agent_{index}"


def _classifier_for_target(target_distribution: Mapping[str, float], labels: Sequence[str] | None) -> str:
    label_set = {str(label) for label in labels or ()} | {str(label) for label in target_distribution}
    if label_set & set(BEHAVIOR_CLASSES):
        return "broad"
    return "action"


def _resolve_labels(
    target_distribution: Mapping[str, float],
    observed_distribution: Mapping[str, float] | None,
    classified_agents: Sequence[ClassifiedAgent],
    labels: Sequence[str] | None,
    classifier: str,
) -> tuple[str, ...]:
    if labels is not None:
        return tuple(str(label) for label in labels)
    inferred = {str(label) for label in target_distribution}
    if observed_distribution is not None:
        inferred.update(str(label) for label in observed_distribution)
    inferred.update(agent.source_behavior for agent in classified_agents if agent.source_behavior != UNKNOWN_BEHAVIOR)
    if inferred:
        preferred = BEHAVIOR_CLASSES if classifier == "broad" else REDDIT_BEHAVIOR_LABELS
        return tuple(label for label in preferred if label in inferred) + tuple(sorted(inferred - set(preferred)))
    return BEHAVIOR_CLASSES if classifier == "broad" else REDDIT_BEHAVIOR_LABELS


def assign_persona_update_targets(
    labels,
    gap: Mapping[str, float],
    *,
    seed: int,
    max_update_fraction: float = 0.35,
    min_abs_over_gap: float = 0.01,
):
    """Compatibility wrapper used by the PEvo loop.

    The main loop passes a behavior-label DataFrame plus signed gap
    ``target - observed``. This wrapper returns the CSV-friendly assignment
    schema expected by PersonaWriter.
    """
    import pandas as pd

    if labels is None or len(labels) == 0:
        return pd.DataFrame(columns=["user_id", "current_behavior", "target_behavior", "selection_reason", "gap_from", "gap_to"])
    label_records = labels.to_dict(orient="records") if hasattr(labels, "to_dict") else list(labels)
    target_distribution = {key: max(0.0, float(value)) for key, value in gap.items() if float(value) > min_abs_over_gap}
    if not target_distribution:
        return pd.DataFrame(columns=["user_id", "current_behavior", "target_behavior", "selection_reason", "gap_from", "gap_to"])
    observed_distribution = {key: max(0.0, -float(value)) for key, value in gap.items() if float(value) < -min_abs_over_gap}
    if not observed_distribution:
        return pd.DataFrame(columns=["user_id", "current_behavior", "target_behavior", "selection_reason", "gap_from", "gap_to"])
    max_assignments = max(1, int(round(len(label_records) * max_update_fraction)))
    plan = assign_behavior_credits(
        label_records,
        target_distribution,
        observed_distribution=observed_distribution,
        labels=tuple(gap.keys()),
        max_assignments=max_assignments,
        min_gap=min_abs_over_gap,
        seed=seed,
        label_key="behavior_label",
    )
    rows = []
    for item in plan.assignments:
        rows.append(
            {
                "user_id": item.agent_id,
                "current_behavior": item.source_behavior,
                "target_behavior": item.target_behavior,
                "selection_reason": item.reason,
                "gap_from": item.source_gap,
                "gap_to": item.target_gap,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "BehaviorAssignment",
    "ClassifiedAgent",
    "CreditAssignmentPlan",
    "DEFAULT_AGENT_ID_KEYS",
    "assignment_rows",
    "assign_behavior_credits",
    "assign_persona_update_targets",
    "classify_agents",
    "observed_distribution_from_agents",
    "select_over_behavior_agents",
]
