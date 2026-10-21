from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .drift_schema import AlignmentGateResult, DriftAuditResult, MutationPlan


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except EmptyDataError:
        return pd.DataFrame()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_alignment_gate(child_dir: Path, plan: MutationPlan, audit: DriftAuditResult) -> AlignmentGateResult:
    profiles = _read_csv(child_dir / "generated_persona_profiles.csv")
    comments = _read_csv(child_dir / "generated_comments.csv")
    behavior = _read_csv(child_dir / "behavior_audit_by_user.csv")
    graph = _read_csv(child_dir / "graph_proxy_audit.csv")

    failed_target_axes = [axis for axis in plan.target_axes if audit.axis_scores.get(axis, 0.0) < 0.15]
    severe_non_target_axes = [
        spec.axis
        for spec in plan.axis_specs
        if spec.role not in {"target", "auxiliary"}
        and audit.axis_scores.get(spec.axis, 0.0) > spec.max_non_target_score
        and _enough_non_target_evidence(child_dir, spec.axis)
    ]
    human_failures = []

    comment_words = []
    if not comments.empty and "comment_body" in comments:
        comment_words = [len(str(text).split()) for text in comments["comment_body"].tolist() if str(text).strip()]
    mean_words = float(np.mean(comment_words)) if comment_words else 0.0
    if mean_words < 3 or mean_words > 120:
        human_failures.append(f"mean_comment_words_out_of_broad_range:{mean_words:.2f}")

    if behavior.empty:
        human_failures.append("missing_behavior_audit")
    else:
        comments_per_user = np.array([_safe_float(v) for v in behavior.get("comment_count", [])], dtype=float)
        posts_per_user = np.array([_safe_float(v) for v in behavior.get("post_count", [])], dtype=float)
        if comments_per_user.size and float(comments_per_user.mean()) < 0.5:
            human_failures.append("too_few_comments_per_user")
        if posts_per_user.size and float(posts_per_user.mean()) > 8:
            human_failures.append("too_many_posts_per_user")

    if not graph.empty and "direction_balance_proxy" in graph:
        direction = np.array([_safe_float(v) for v in graph["direction_balance_proxy"]], dtype=float)
        if direction.size and not (0.15 <= float(direction.mean()) <= 1.0):
            human_failures.append(f"direction_balance_broad_gate_failed:{float(direction.mean()):.3f}")

    if profiles.empty:
        human_failures.append("missing_profiles")

    target_pass_rate = audit.target_pass_rate
    severe_non_target_count = len(severe_non_target_axes)
    passed = target_pass_rate >= 0.8 and severe_non_target_count == 0 and not human_failures
    notes = []
    if not passed:
        notes.append("Rejected from qualified detector set; keep under rejected attempts for analysis.")
    else:
        notes.append("Qualified for detector sanity evaluation.")
    return AlignmentGateResult(
        family=plan.family,
        slug=plan.slug,
        plan_id=plan.plan_id,
        attempt=plan.attempt,
        passed=passed,
        target_pass_rate=round(target_pass_rate, 4),
        severe_non_target_count=severe_non_target_count,
        failed_target_axes=failed_target_axes,
        severe_non_target_axes=severe_non_target_axes,
        human_gate_failures=human_failures,
        notes=notes,
    )


def _non_target_evidence_count(child_dir: Path, axis: str) -> int:
    profiles = _read_csv(child_dir / "generated_persona_profiles.csv")
    comments = _read_csv(child_dir / "generated_comments.csv")
    posts = _read_csv(child_dir / "generated_posts.csv")
    behavior = _read_csv(child_dir / "behavior_audit_by_user.csv")
    graph = _read_csv(child_dir / "graph_proxy_audit.csv")
    if axis in {"language", "topic_content"}:
        return len(comments)
    if axis in {"persona_population", "llm_generation"}:
        return len(profiles)
    if axis == "action_behavior":
        return len(behavior)
    if axis == "graph_structure":
        return len(graph)
    if axis == "temporal_rhythm":
        return len(comments) + len(posts)
    if axis == "memory_revisit":
        trace_path = child_dir / "run" / "decision_trace.jsonl"
        if not trace_path.exists():
            return 0
        return sum(1 for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return 0


def _min_non_target_evidence(axis: str) -> int:
    if axis == "temporal_rhythm":
        return 60
    if axis == "memory_revisit":
        return 60
    if axis in {"language", "topic_content"}:
        return 20
    return 20


def _enough_non_target_evidence(child_dir: Path, axis: str) -> bool:
    return _non_target_evidence_count(child_dir, axis) >= _min_non_target_evidence(axis)
