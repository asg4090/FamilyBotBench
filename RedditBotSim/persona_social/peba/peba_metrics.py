from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


EPSILON = 1e-12


@dataclass(frozen=True)
class DistributionComparison:
    labels: tuple[str, ...]
    observed: dict[str, float]
    target: dict[str, float]
    signed_gap: dict[str, float]
    tv: float
    js: float
    kl_observed_to_target: float
    kl_target_to_observed: float
    entropy_observed: float
    entropy_target: float
    entropy_gap: float
    overrepresented: tuple[str, ...]
    underrepresented: tuple[str, ...]


def normalize_distribution(
    distribution: Mapping[str, float],
    labels: Sequence[str] | None = None,
    *,
    fill_value: float = 0.0,
    alpha: float = 0.0,
) -> dict[str, float]:
    ordered_labels = tuple(labels) if labels is not None else tuple(sorted(str(key) for key in distribution))
    raw = {label: max(0.0, _as_float(distribution.get(label, fill_value)) + alpha) for label in ordered_labels}
    total = sum(raw.values())
    if total <= 0.0:
        return {label: 0.0 for label in ordered_labels}
    return {label: value / total for label, value in raw.items()}


def signed_gap(
    observed: Mapping[str, float],
    target: Mapping[str, float],
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """Return ``target - observed`` for each label.

    Positive values are underrepresented behavior classes; negative values are
    overrepresented behavior classes.
    """

    ordered_labels = _labels(observed, target, labels)
    obs = normalize_distribution(observed, ordered_labels)
    tgt = normalize_distribution(target, ordered_labels)
    return {label: tgt[label] - obs[label] for label in ordered_labels}


def total_variation_distance(
    observed: Mapping[str, float],
    target: Mapping[str, float],
    labels: Sequence[str] | None = None,
) -> float:
    gaps = signed_gap(observed, target, labels)
    return 0.5 * sum(abs(value) for value in gaps.values())


def kl_divergence(
    p: Mapping[str, float],
    q: Mapping[str, float],
    labels: Sequence[str] | None = None,
    *,
    base: float = 2.0,
    epsilon: float = EPSILON,
) -> float:
    ordered_labels = _labels(p, q, labels)
    pp = normalize_distribution(p, ordered_labels)
    qq = normalize_distribution(q, ordered_labels)
    log_base = math.log(base) if base and base > 0 else 1.0
    value = 0.0
    for label in ordered_labels:
        pv = pp[label]
        if pv <= 0.0:
            continue
        qv = max(qq[label], epsilon)
        value += pv * (math.log(pv / qv) / log_base)
    return max(0.0, value)


def js_divergence(
    observed: Mapping[str, float],
    target: Mapping[str, float],
    labels: Sequence[str] | None = None,
    *,
    base: float = 2.0,
) -> float:
    ordered_labels = _labels(observed, target, labels)
    obs = normalize_distribution(observed, ordered_labels)
    tgt = normalize_distribution(target, ordered_labels)
    midpoint = {label: 0.5 * (obs[label] + tgt[label]) for label in ordered_labels}
    return 0.5 * kl_divergence(obs, midpoint, ordered_labels, base=base) + 0.5 * kl_divergence(tgt, midpoint, ordered_labels, base=base)


def entropy(
    distribution: Mapping[str, float],
    labels: Sequence[str] | None = None,
    *,
    base: float = 2.0,
    normalized: bool = False,
) -> float:
    ordered_labels = tuple(labels) if labels is not None else tuple(sorted(str(key) for key in distribution))
    probs = normalize_distribution(distribution, ordered_labels)
    log_base = math.log(base) if base and base > 0 else 1.0
    value = 0.0
    for prob in probs.values():
        if prob > 0.0:
            value -= prob * (math.log(prob) / log_base)
    if normalized and len(ordered_labels) > 1:
        value /= math.log(len(ordered_labels)) / log_base
    return value


def entropy_gap(
    observed: Mapping[str, float],
    target: Mapping[str, float],
    labels: Sequence[str] | None = None,
    *,
    base: float = 2.0,
    normalized: bool = False,
) -> float:
    """Return target entropy minus observed entropy.

    Positive values mean the target distribution is more behavior-diverse than
    the observed distribution.
    """

    ordered_labels = _labels(observed, target, labels)
    return entropy(target, ordered_labels, base=base, normalized=normalized) - entropy(observed, ordered_labels, base=base, normalized=normalized)


def compare_distributions(
    observed: Mapping[str, float],
    target: Mapping[str, float],
    labels: Sequence[str] | None = None,
    *,
    gap_tolerance: float = 1e-9,
    entropy_normalized: bool = False,
) -> DistributionComparison:
    ordered_labels = _labels(observed, target, labels)
    obs = normalize_distribution(observed, ordered_labels)
    tgt = normalize_distribution(target, ordered_labels)
    gaps = {label: tgt[label] - obs[label] for label in ordered_labels}
    ent_obs = entropy(obs, ordered_labels, normalized=entropy_normalized)
    ent_tgt = entropy(tgt, ordered_labels, normalized=entropy_normalized)
    under = tuple(label for label in ordered_labels if gaps[label] > gap_tolerance)
    over = tuple(label for label in ordered_labels if gaps[label] < -gap_tolerance)
    return DistributionComparison(
        labels=tuple(ordered_labels),
        observed=obs,
        target=tgt,
        signed_gap=gaps,
        tv=total_variation_distance(obs, tgt, ordered_labels),
        js=js_divergence(obs, tgt, ordered_labels),
        kl_observed_to_target=kl_divergence(obs, tgt, ordered_labels),
        kl_target_to_observed=kl_divergence(tgt, obs, ordered_labels),
        entropy_observed=ent_obs,
        entropy_target=ent_tgt,
        entropy_gap=ent_tgt - ent_obs,
        overrepresented=over,
        underrepresented=under,
    )


def metric_dict(comparison: DistributionComparison) -> dict[str, Any]:
    return {
        "labels": list(comparison.labels),
        "observed": dict(comparison.observed),
        "target": dict(comparison.target),
        "signed_gap": dict(comparison.signed_gap),
        "tv": comparison.tv,
        "js": comparison.js,
        "kl_observed_to_target": comparison.kl_observed_to_target,
        "kl_target_to_observed": comparison.kl_target_to_observed,
        "entropy_observed": comparison.entropy_observed,
        "entropy_target": comparison.entropy_target,
        "entropy_gap": comparison.entropy_gap,
        "overrepresented": list(comparison.overrepresented),
        "underrepresented": list(comparison.underrepresented),
    }


def _coerce_distribution_payload(value: Mapping[str, float] | Iterable[str], labels: Sequence[str]) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {str(key): _as_float(raw) for key, raw in value.items()}
    counts = Counter(str(item) for item in value)
    return {label: float(counts.get(label, 0.0)) for label in labels}


def distribution_gap(
    observed: Mapping[str, float] | Iterable[str],
    target: Mapping[str, float],
    labels: Sequence[str] | None = None,
    *,
    alpha: float = 0.0,
    gap_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Return a serializable PEBA distribution gap report.

    Positive signed gaps mean the target behavior is underrepresented in the
    observed population; negative signed gaps identify overrepresented sources.
    """

    ordered_labels = _labels(_coerce_distribution_payload(observed, labels or tuple(target)), target, labels)
    observed_payload = _coerce_distribution_payload(observed, ordered_labels)
    obs = normalize_distribution(observed_payload, ordered_labels, alpha=alpha)
    tgt = normalize_distribution(target, ordered_labels, alpha=alpha)
    comparison = compare_distributions(obs, tgt, ordered_labels, gap_tolerance=gap_tolerance)
    payload = metric_dict(comparison)
    payload["p_sim"] = dict(payload["observed"])
    payload["p_real"] = dict(payload["target"])
    payload["gap"] = dict(payload["signed_gap"])
    payload["metrics"] = {
        "kl": payload["kl_observed_to_target"],
        "js": payload["js"],
        "tv": payload["tv"],
        "entropy_gap": payload["entropy_gap"],
    }
    return payload


def _labels(
    left: Mapping[str, float],
    right: Mapping[str, float],
    labels: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if labels is not None:
        return tuple(str(label) for label in labels)
    return tuple(sorted({str(key) for key in left} | {str(key) for key in right}))


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
    "DistributionComparison",
    "EPSILON",
    "compare_distributions",
    "distribution_gap",
    "entropy",
    "entropy_gap",
    "js_divergence",
    "kl_divergence",
    "metric_dict",
    "normalize_distribution",
    "signed_gap",
    "total_variation_distance",
]
