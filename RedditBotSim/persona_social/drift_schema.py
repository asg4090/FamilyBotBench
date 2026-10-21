from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


DRIFT_AXES: tuple[str, ...] = (
    "language",
    "topic_content",
    "persona_population",
    "action_behavior",
    "graph_structure",
    "temporal_rhythm",
    "memory_revisit",
    "llm_generation",
)


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=repr)]
    return value


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class DriftAxisSpec:
    axis: str
    role: str = "watch"
    min_target_score: float = 0.15
    max_non_target_score: float = 0.10
    human_gate_profile: str = "broad"

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass(frozen=True)
class MutationPlan:
    family: str
    slug: str
    generation: int
    attempt: int
    seed: int
    plan_id: str
    parent_family_version: str
    child_family_version: str
    target_axes: list[str]
    auxiliary_axes: list[str] = field(default_factory=list)
    frozen_axes: list[str] = field(default_factory=list)
    axis_specs: list[DriftAxisSpec] = field(default_factory=list)
    axis_intensity: dict[str, float] = field(default_factory=dict)
    target_metrics: dict[str, Any] = field(default_factory=dict)
    model_pool: list[dict[str, Any]] = field(default_factory=list)
    feedback: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)

    def to_json(self, **kwargs: Any) -> str:
        opts = {"ensure_ascii": False, "indent": 2}
        opts.update(kwargs)
        return json.dumps(self.to_dict(), **opts)


@dataclass(frozen=True)
class DriftMetric:
    family: str
    axis: str
    role: str
    metric: str
    parent_value: float | str | None
    child_value: float | str | None
    human_reference: float | str | None
    distance: float
    drift_score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass(frozen=True)
class DriftAuditResult:
    family: str
    slug: str
    plan_id: str
    attempt: int
    passed: bool
    target_pass_rate: float
    severe_non_target_count: int
    alignment_passed: bool
    axis_scores: dict[str, float]
    metrics: list[DriftMetric] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass(frozen=True)
class AlignmentGateResult:
    family: str
    slug: str
    plan_id: str
    attempt: int
    passed: bool
    target_pass_rate: float
    severe_non_target_count: int
    failed_target_axes: list[str] = field(default_factory=list)
    severe_non_target_axes: list[str] = field(default_factory=list)
    human_gate_failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)
