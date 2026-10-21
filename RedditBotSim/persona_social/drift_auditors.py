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
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .drift_schema import DRIFT_AXES, DriftAuditResult, DriftMetric, MutationPlan, clamp


ORAL_MARKERS = {"yeah", "lol", "honestly", "idk", "like", "nah", "imo", "tbh", "basically", "wild", "weird"}
ABSTRACT_MARKERS = {
    "implications",
    "accountability",
    "landscape",
    "ongoing",
    "ramifications",
    "critical",
    "strategy",
    "dialogue",
    "multifaceted",
    "geopolitical",
}
TOPIC_KEYWORDS = {
    "US politics": ["trump", "biden", "republican", "democrat", "congress", "election", "vote", "court"],
    "Russia-Ukraine war": ["ukraine", "russia", "putin", "zelensky", "nato", "kyiv", "moscow"],
    "Israel-Palestine": ["israel", "gaza", "hamas", "palestinian", "idf", "rafah", "hostage"],
    "world news": ["global", "country", "government", "minister", "police", "protest", "climate"],
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tokens(text: str) -> list[str]:
    return [item.lower() for item in TOKEN_RE.findall(str(text or ""))]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _numeric_series(values: list[float] | pd.Series) -> np.ndarray:
    arr = np.asarray([_safe_float(v, 0.0) for v in list(values)], dtype=float)
    if arr.size == 0:
        return np.zeros(1, dtype=float)
    return arr


def _hist(values: list[str]) -> dict[str, float]:
    counter = Counter(str(v) for v in values if str(v) != "")
    total = sum(counter.values()) or 1
    return {key: value / total for key, value in counter.items()}


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    pv = np.array([float(p.get(k, 0.0)) for k in keys], dtype=float)
    qv = np.array([float(q.get(k, 0.0)) for k in keys], dtype=float)
    pv = pv / max(pv.sum(), 1e-12)
    qv = qv / max(qv.sum(), 1e-12)
    mv = 0.5 * (pv + qv)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))).sum())

    return float(clamp(0.5 * _kl(pv, mv) + 0.5 * _kl(qv, mv), 0.0, 1.0))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a.astype(float))
    b = np.sort(b.astype(float))
    if a.size == 0 or b.size == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein_proxy(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a.astype(float))
    b = np.sort(b.astype(float))
    if a.size == 0 or b.size == 0:
        return 0.0
    n = max(a.size, b.size)
    qa = np.quantile(a, np.linspace(0, 1, n))
    qb = np.quantile(b, np.linspace(0, 1, n))
    scale = max(float(np.std(np.concatenate([a, b]))), 1.0)
    return float(clamp(np.mean(np.abs(qa - qb)) / scale, 0.0, 1.0))


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(values == 0):
        return 0.0
    values = np.sort(np.maximum(values, 0.0))
    n = values.size
    return float((2 * np.arange(1, n + 1).dot(values) / (n * values.sum())) - (n + 1) / n)


def infer_topic(text: str) -> str:
    lowered = str(text or "").lower()
    scores = {topic: sum(1 for keyword in keywords if keyword in lowered) for topic, keywords in TOPIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "world news"


class RunSnapshot:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.comments = _read_csv(run_dir / "generated_comments.csv")
        self.posts = _read_csv(run_dir / "generated_posts.csv")
        self.profiles = _read_csv(run_dir / "generated_persona_profiles.csv")
        self.behavior = _read_csv(run_dir / "behavior_audit_by_user.csv")
        self.graph = _read_csv(run_dir / "graph_proxy_audit.csv")
        self.summary = _read_json(run_dir / "persona_social_summary.json")
        self.family_manifest = _read_csv(run_dir / "family_manifest.csv")
        self.decision_trace = self._read_trace(run_dir / "run" / "decision_trace.jsonl")
        self.memory_trace = self._read_trace(run_dir / "run" / "memory_trace.jsonl")

    @staticmethod
    def _read_trace(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return pd.DataFrame(rows)


def audit_drift(parent_dir: Path, child_dir: Path, plan: MutationPlan) -> DriftAuditResult:
    parent = RunSnapshot(parent_dir)
    child = RunSnapshot(child_dir)
    metrics: list[DriftMetric] = []
    metrics.extend(_language_metrics(parent, child, plan))
    metrics.extend(_topic_metrics(parent, child, plan))
    metrics.extend(_persona_metrics(parent, child, plan))
    metrics.extend(_action_metrics(parent, child, plan))
    metrics.extend(_graph_metrics(parent, child, plan))
    metrics.extend(_temporal_metrics(parent, child, plan))
    metrics.extend(_memory_metrics(parent, child, plan))
    metrics.extend(_llm_metrics(parent, child, plan))
    axis_scores = {}
    for axis in DRIFT_AXES:
        axis_metrics = [m for m in metrics if m.axis == axis]
        axis_scores[axis] = round(float(np.mean([m.drift_score for m in axis_metrics])) if axis_metrics else 0.0, 4)
    target_axes = set(plan.target_axes)
    target_passes = [axis_scores.get(axis, 0.0) >= 0.15 for axis in target_axes]
    target_pass_rate = float(sum(target_passes) / max(1, len(target_passes)))
    severe_non_target = 0
    for spec in plan.axis_specs:
        if spec.role in {"target", "auxiliary"}:
            continue
        if axis_scores.get(spec.axis, 0.0) > spec.max_non_target_score and _enough_non_target_evidence(parent, child, spec.axis):
            severe_non_target += 1
    failure_reasons = []
    if target_pass_rate < 0.8:
        failure_reasons.append(f"target_pass_rate={target_pass_rate:.2f} < 0.80")
    if severe_non_target:
        failure_reasons.append(f"severe_non_target_count={severe_non_target}")
    return DriftAuditResult(
        family=plan.family,
        slug=plan.slug,
        plan_id=plan.plan_id,
        attempt=plan.attempt,
        passed=not failure_reasons,
        target_pass_rate=round(target_pass_rate, 4),
        severe_non_target_count=severe_non_target,
        alignment_passed=True,
        axis_scores=axis_scores,
        metrics=metrics,
        failure_reasons=failure_reasons,
    )


def _axis_evidence_count(parent: RunSnapshot, child: RunSnapshot, axis: str) -> int:
    if axis in {"language", "topic_content"}:
        return min(len(_comment_texts(parent)), len(_comment_texts(child)))
    if axis in {"persona_population", "llm_generation"}:
        return min(len(parent.profiles), len(child.profiles))
    if axis == "action_behavior":
        return min(len(parent.behavior), len(child.behavior))
    if axis == "graph_structure":
        return min(len(parent.graph), len(child.graph))
    if axis == "temporal_rhythm":
        parent_events = len(parent.comments) + len(parent.posts)
        child_events = len(child.comments) + len(child.posts)
        return min(parent_events, child_events)
    if axis == "memory_revisit":
        return min(len(parent.decision_trace), len(child.decision_trace))
    return 0


def _min_non_target_evidence(axis: str) -> int:
    if axis == "temporal_rhythm":
        return 60
    if axis == "memory_revisit":
        return 60
    if axis in {"language", "topic_content"}:
        return 20
    return 20


def _enough_non_target_evidence(parent: RunSnapshot, child: RunSnapshot, axis: str) -> bool:
    return _axis_evidence_count(parent, child, axis) >= _min_non_target_evidence(axis)


def audit_to_frames(result: DriftAuditResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = [metric.to_dict() for metric in result.metrics]
    by_metric = pd.DataFrame(metric_rows)
    by_axis = pd.DataFrame(
        [
            {
                "family": result.family,
                "slug": result.slug,
                "plan_id": result.plan_id,
                "attempt": result.attempt,
                "axis": axis,
                "drift_score": score,
                "role": _axis_role(axis, result),
                "is_target": axis in {m.axis for m in result.metrics if m.role == "target"},
            }
            for axis, score in result.axis_scores.items()
        ]
    )
    return by_metric, by_axis


def _axis_role(axis: str, result: DriftAuditResult) -> str:
    roles = [metric.role for metric in result.metrics if metric.axis == axis]
    return roles[0] if roles else "watch"


def _role(plan: MutationPlan, axis: str) -> str:
    for spec in plan.axis_specs:
        if spec.axis == axis:
            return spec.role
    return "watch"


def _metric(plan: MutationPlan, axis: str, metric: str, parent: Any, child: Any, distance: float, details: dict[str, Any] | None = None) -> DriftMetric:
    role = _role(plan, axis)
    threshold = 0.15 if role == "target" else 0.28 if role == "auxiliary" else 0.10
    passed = distance >= threshold if role == "target" else distance <= threshold
    return DriftMetric(
        family=plan.family,
        axis=axis,
        role=role,
        metric=metric,
        parent_value=_render_value(parent),
        child_value=_render_value(child),
        human_reference=None,
        distance=round(float(distance), 4),
        drift_score=round(float(clamp(distance, 0.0, 1.0)), 4),
        passed=bool(passed),
        details=details or {},
    )


def _render_value(value: Any) -> float | str | None:
    if isinstance(value, (int, float, np.floating)):
        return round(float(value), 4)
    if value is None:
        return None
    text = str(value)
    return text[:180]


def _comment_texts(snapshot: RunSnapshot) -> list[str]:
    if snapshot.comments.empty or "comment_body" not in snapshot.comments:
        return []
    return [str(item) for item in snapshot.comments["comment_body"].tolist() if str(item).strip()]


def _language_features(snapshot: RunSnapshot) -> dict[str, Any]:
    texts = _comment_texts(snapshot)
    toks = [_tokens(text) for text in texts]
    flat = [token for row in toks for token in row]
    lengths = np.array([len(row) for row in toks], dtype=float) if toks else np.zeros(1)
    vocab = set(flat)
    oral = sum(1 for token in flat if token in ORAL_MARKERS)
    abstract = sum(1 for token in flat if token in ABSTRACT_MARKERS)
    total = max(1, len(flat))
    punct = sum(text.count("?") + text.count("!") for text in texts)
    return {
        "mean_words": float(lengths.mean()),
        "median_words": float(np.median(lengths)),
        "lengths": lengths,
        "type_token_ratio": len(vocab) / total,
        "oral_rate": oral / total,
        "abstract_rate": abstract / total,
        "question_exclaim_rate": punct / max(1, len(texts)),
        "token_hist": _hist(flat[:5000]),
    }


def _language_metrics(parent: RunSnapshot, child: RunSnapshot, plan: MutationPlan) -> list[DriftMetric]:
    p = _language_features(parent)
    c = _language_features(child)
    length_shift = wasserstein_proxy(p["lengths"], c["lengths"])
    lexical_js = js_divergence(p["token_hist"], c["token_hist"])
    rate_shift = np.mean(
        [
            abs(p["type_token_ratio"] - c["type_token_ratio"]),
            abs(p["oral_rate"] - c["oral_rate"]) * 10,
            abs(p["abstract_rate"] - c["abstract_rate"]) * 10,
            abs(p["question_exclaim_rate"] - c["question_exclaim_rate"]) / 2,
        ]
    )
    return [
        _metric(plan, "language", "length_wasserstein", p["mean_words"], c["mean_words"], length_shift),
        _metric(plan, "language", "lexical_js", "token_hist", "token_hist", lexical_js),
        _metric(plan, "language", "style_rate_shift", p["oral_rate"], c["oral_rate"], clamp(float(rate_shift))),
    ]


def _topic_metrics(parent: RunSnapshot, child: RunSnapshot, plan: MutationPlan) -> list[DriftMetric]:
    p_topics = [infer_topic(text) for text in _comment_texts(parent)]
    c_topics = [infer_topic(text) for text in _comment_texts(child)]
    p_subs = parent.comments["subreddit"].tolist() if "subreddit" in parent.comments else []
    c_subs = child.comments["subreddit"].tolist() if "subreddit" in child.comments else []
    p_pref = parent.profiles["Preference"].tolist() if "Preference" in parent.profiles else []
    c_pref = child.profiles["Preference"].tolist() if "Preference" in child.profiles else []
    return [
        _metric(plan, "topic_content", "comment_topic_js", "topic_hist", "topic_hist", js_divergence(_hist(p_topics), _hist(c_topics))),
        _metric(plan, "topic_content", "subreddit_js", "subreddit_hist", "subreddit_hist", js_divergence(_hist(p_subs), _hist(c_subs))),
        _metric(plan, "topic_content", "profile_preference_js", "preference_hist", "preference_hist", js_divergence(_hist(p_pref), _hist(c_pref))),
    ]


def _persona_metrics(parent: RunSnapshot, child: RunSnapshot, plan: MutationPlan) -> list[DriftMetric]:
    p_age = _numeric_series(parent.profiles.get("Age", []))
    c_age = _numeric_series(child.profiles.get("Age", []))
    items = [
        _metric(plan, "persona_population", "age_ks", float(p_age.mean()), float(c_age.mean()), ks_statistic(p_age, c_age)),
    ]
    for col in ["Gender", "Region", "EducationLevel", "Ideology", "Preference"]:
        items.append(_metric(plan, "persona_population", f"{col.lower()}_js", f"{col}_hist", f"{col}_hist", js_divergence(_hist(parent.profiles.get(col, [])), _hist(child.profiles.get(col, [])))))
    return items


def _summary_counts(snapshot: RunSnapshot, key: str) -> dict[str, float]:
    summary = snapshot.summary.get("behavior_summary", {}) if snapshot.summary else {}
    raw = summary.get(key, {}) if isinstance(summary, dict) else {}
    if not isinstance(raw, dict):
        return {}
    total = sum(_safe_float(v, 0.0) for v in raw.values()) or 1.0
    return {str(k): _safe_float(v, 0.0) / total for k, v in raw.items()}


def _action_metrics(parent: RunSnapshot, child: RunSnapshot, plan: MutationPlan) -> list[DriftMetric]:
    p_action = _summary_counts(parent, "action_counts")
    c_action = _summary_counts(child, "action_counts")
    p_comments = _numeric_series(parent.behavior.get("comment_count", []))
    c_comments = _numeric_series(child.behavior.get("comment_count", []))
    p_posts = _numeric_series(parent.behavior.get("post_count", []))
    c_posts = _numeric_series(child.behavior.get("post_count", []))
    return [
        _metric(plan, "action_behavior", "action_distribution_js", "action_hist", "action_hist", js_divergence(p_action, c_action)),
        _metric(plan, "action_behavior", "comments_per_user_ks", float(p_comments.mean()), float(c_comments.mean()), ks_statistic(p_comments, c_comments)),
        _metric(plan, "action_behavior", "posts_per_user_ks", float(p_posts.mean()), float(c_posts.mean()), ks_statistic(p_posts, c_posts)),
        _metric(plan, "action_behavior", "activity_gini_shift", gini(p_comments + p_posts), gini(c_comments + c_posts), abs(gini(p_comments + p_posts) - gini(c_comments + c_posts))),
    ]


def _graph_metrics(parent: RunSnapshot, child: RunSnapshot, plan: MutationPlan) -> list[DriftMetric]:
    cols = [
        "total_out_weight_proxy",
        "total_in_weight_proxy",
        "direction_balance_proxy",
        "reciprocal_share_proxy",
        "unique_target_authors",
        "unique_inbound_authors",
    ]
    metrics = []
    for col in cols:
        p = _numeric_series(parent.graph.get(col, []))
        c = _numeric_series(child.graph.get(col, []))
        metrics.append(_metric(plan, "graph_structure", f"{col}_ks", float(p.mean()), float(c.mean()), ks_statistic(p, c)))
    return metrics


def _parse_times(values: list[Any]) -> list[pd.Timestamp]:
    out = []
    for value in values:
        try:
            out.append(pd.to_datetime(str(value)))
        except Exception:
            continue
    return out


def _time_features(snapshot: RunSnapshot) -> dict[str, Any]:
    comment_times = _parse_times(snapshot.comments.get("created_utc", []).tolist() if not snapshot.comments.empty else [])
    post_times = _parse_times(snapshot.posts.get("created_utc", []).tolist() if not snapshot.posts.empty else [])
    all_times = sorted(comment_times + post_times)
    if len(all_times) >= 2:
        deltas = np.array([(all_times[i] - all_times[i - 1]).total_seconds() / 3600 for i in range(1, len(all_times))], dtype=float)
    else:
        deltas = np.zeros(1)
    hours = [str(ts.hour) for ts in all_times]
    return {"deltas": deltas, "hours": hours, "mean_delta": float(deltas.mean())}


def _temporal_metrics(parent: RunSnapshot, child: RunSnapshot, plan: MutationPlan) -> list[DriftMetric]:
    p = _time_features(parent)
    c = _time_features(child)
    return [
        _metric(plan, "temporal_rhythm", "interarrival_ks", p["mean_delta"], c["mean_delta"], ks_statistic(p["deltas"], c["deltas"])),
        _metric(plan, "temporal_rhythm", "hour_js", "hour_hist", "hour_hist", js_divergence(_hist(p["hours"]), _hist(c["hours"]))),
        _metric(plan, "temporal_rhythm", "burstiness_shift", float(np.std(p["deltas"])), float(np.std(c["deltas"])), clamp(abs(float(np.std(p["deltas"]) - np.std(c["deltas"]))) / 24)),
    ]


def _memory_metrics(parent: RunSnapshot, child: RunSnapshot, plan: MutationPlan) -> list[DriftMetric]:
    p_sources = _summary_counts(parent, "candidate_source_counts")
    c_sources = _summary_counts(child, "candidate_source_counts")
    p_revisit = _safe_float(parent.summary.get("revisit_written", 0.0))
    c_revisit = _safe_float(child.summary.get("revisit_written", 0.0))
    p_note = _safe_float(parent.summary.get("notifications_created", 0.0))
    c_note = _safe_float(child.summary.get("notifications_created", 0.0))
    scale = max(1.0, p_note + c_note)
    return [
        _metric(plan, "memory_revisit", "candidate_source_js", "source_hist", "source_hist", js_divergence(p_sources, c_sources)),
        _metric(plan, "memory_revisit", "revisit_count_shift", p_revisit, c_revisit, clamp(abs(p_revisit - c_revisit) / max(1.0, p_revisit + 1))),
        _metric(plan, "memory_revisit", "notification_count_shift", p_note, c_note, clamp(abs(p_note - c_note) / scale)),
    ]


def _llm_metrics(parent: RunSnapshot, child: RunSnapshot, plan: MutationPlan) -> list[DriftMetric]:
    p_provider = _hist(parent.family_manifest.get("provider", []))
    c_provider = _hist(child.family_manifest.get("provider", []))
    p_model = _hist(parent.family_manifest.get("model", []))
    c_model = _hist(child.family_manifest.get("model", []))
    p_temp = _numeric_series(parent.family_manifest.get("temperature", []))
    c_temp = _numeric_series(child.family_manifest.get("temperature", []))
    return [
        _metric(plan, "llm_generation", "provider_js", "provider_hist", "provider_hist", js_divergence(p_provider, c_provider)),
        _metric(plan, "llm_generation", "model_js", "model_hist", "model_hist", js_divergence(p_model, c_model)),
        _metric(plan, "llm_generation", "temperature_ks", float(p_temp.mean()), float(c_temp.mean()), ks_statistic(p_temp, c_temp)),
    ]
