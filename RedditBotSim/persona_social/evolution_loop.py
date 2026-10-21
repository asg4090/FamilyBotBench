from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import hashlib
import json
import random
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .family_mutation import (
    FamilyMutation,
    build_mutation_plan,
    evolve_family_personas,
    mutate_family_spec,
    mutation_summary,
)
from .family_registry import get_family_spec, normalize_family_name
from .family_schema import FamilyManifest, FamilySpec, build_family_manifest, export_persona_list
from .family_sampler import build_evolved_family_bundle, sample_family_personas, sample_model_pool
from .models import Persona


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    return value


def _stable_seed(*parts: object) -> int:
    payload = "|".join(json.dumps(part, ensure_ascii=False, sort_keys=True, default=str) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class EvolutionRound:
    generation: int
    family: str
    family_version: str
    mutation_id: str
    mutation_parent: str | None
    mutation_reason: str
    seed: int | None = None
    feedback: dict[str, float] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    mutation: dict[str, Any] = field(default_factory=dict)
    family_spec: dict[str, Any] = field(default_factory=dict)
    personas: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class EvolutionResult:
    family: str
    rounds: list[EvolutionRound] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "rounds": [round_item.to_dict() for round_item in self.rounds],
            "summary": _json_safe(self.summary),
        }


def normalize_feedback(feedback: dict[str, float] | None) -> dict[str, float]:
    if not feedback:
        return {}
    out: dict[str, float] = {}
    for key, value in feedback.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def aggregate_feedback(
    feedback_records: Iterable[dict[str, float]] | None,
    *,
    default: dict[str, float] | None = None,
) -> dict[str, float]:
    records = list(feedback_records or [])
    if not records:
        return normalize_feedback(default)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for record in records:
        normalized = normalize_feedback(record)
        for key, value in normalized.items():
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    averaged = {key: sums[key] / max(1, counts[key]) for key in sums}
    if default:
        for key, value in default.items():
            averaged.setdefault(key, float(value))
    return averaged


def normalize_detector_feedback(
    metrics: dict[str, float] | None,
    *,
    family: str,
) -> dict[str, float]:
    normalized = normalize_feedback(metrics)
    if not normalized:
        return {}
    out = dict(normalized)
    recall = out.get("bot_recall", out.get("recall", out.get("score", 0.0)))
    out.setdefault("pressure", _clamp(recall, 0.0, 1.0))
    out.setdefault("detector_pressure", out["pressure"])
    out.setdefault("text_pressure", out["pressure"])
    out.setdefault("behavior_pressure", out["pressure"] * 0.85)
    out.setdefault("graph_pressure", out["pressure"] * 0.8)
    out.setdefault("time_pressure", out["pressure"] * 0.65)
    out["family"] = family
    return out


def build_family_evolution_feedback(
    family: str,
    rounds_metrics: Iterable[dict[str, float]],
    *,
    family_bias: dict[str, float] | None = None,
) -> dict[str, float]:
    family_name = normalize_family_name(family)
    family_bias = normalize_feedback(family_bias)
    recs = [normalize_feedback(metrics) for metrics in rounds_metrics]
    if not recs:
        return family_bias
    text_values = []
    graph_values = []
    behavior_values = []
    time_values = []
    for rec in recs:
        if "text_pressure" in rec:
            text_values.append(rec["text_pressure"])
        if "behavior_pressure" in rec:
            behavior_values.append(rec["behavior_pressure"])
        if "graph_pressure" in rec:
            graph_values.append(rec["graph_pressure"])
        if "time_pressure" in rec:
            time_values.append(rec["time_pressure"])
        if "pressure" in rec:
            text_values.append(rec["pressure"])
            behavior_values.append(rec["pressure"])
            graph_values.append(rec["pressure"])
            time_values.append(rec["pressure"])
    def _avg(values: list[float], fallback: float = 0.0) -> float:
        return sum(values) / len(values) if values else fallback
    pressure = _clamp(_avg(text_values + behavior_values + graph_values + time_values, family_bias.get("pressure", 0.0)), 0.0, 1.0)
    out = {
        "pressure": pressure,
        "detector_pressure": pressure,
        "text_pressure": _clamp(_avg(text_values, pressure), 0.0, 1.0),
        "behavior_pressure": _clamp(_avg(behavior_values, pressure * 0.85), 0.0, 1.0),
        "graph_pressure": _clamp(_avg(graph_values, pressure * 0.8), 0.0, 1.0),
        "time_pressure": _clamp(_avg(time_values, pressure * 0.65), 0.0, 1.0),
        "family": family_name,
    }
    out.update({key: float(value) for key, value in family_bias.items() if key not in out})
    return out


def _manifest_family_name(value: Any, fallback: str) -> str:
    if isinstance(value, dict):
        return str(value.get("family", fallback))
    return str(getattr(value, "family", fallback))


def _manifest_family_version(value: Any, fallback: str = "base") -> str:
    if isinstance(value, dict):
        return str(value.get("family_version", fallback))
    return str(getattr(value, "family_version", fallback))


def _manifest_lineage(value: Any) -> list[str]:
    if isinstance(value, dict):
        raw = value.get("lineage", [])
    else:
        raw = getattr(value, "lineage", [])
    return [str(item) for item in (raw or [])]


def evolve_one_generation(
    family: str | FamilySpec,
    generation: int,
    personas: Iterable[Persona | dict[str, Any]] | int,
    *,
    model_pool: list[dict[str, Any]] | None = None,
    feedback: dict[str, float] | None = None,
    seed: int | None = None,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> tuple[list[Persona], FamilySpec, FamilyMutation, FamilyManifest]:
    spec = get_family_spec(family) if isinstance(family, str) else family
    evolved_personas, evolved_spec, mutation = evolve_family_personas(
        spec,
        generation,
        personas,
        model_pool=model_pool or sample_model_pool(seed or 0),
        feedback=feedback,
        seed=seed,
        parent_version=parent_version,
        lineage=lineage,
    )
    manifest = build_family_manifest(
        family=evolved_spec.family,
        generation=generation,
        personas=evolved_personas,
        model_pool=model_pool or sample_model_pool(seed or 0),
        feedback=feedback,
        seed=seed,
        family_spec=evolved_spec,
        mutation=mutation,
        family_version=mutation.family_version,
        dimension_axes=mutation.dimension_axes,
        time_profile=mutation.time_profile,
        lineage=mutation.lineage,
    )
    return evolved_personas, evolved_spec, mutation, manifest


def run_evolution_loop(
    family: str | FamilySpec,
    initial_personas: Iterable[Persona | dict[str, Any]] | int,
    *,
    generations: int,
    model_pool: list[dict[str, Any]] | None = None,
    feedback_sequence: Iterable[dict[str, float]] | None = None,
    seed: int | None = None,
    family_feedback: dict[str, float] | None = None,
) -> EvolutionResult:
    spec = get_family_spec(family) if isinstance(family, str) else family
    pool = model_pool or sample_model_pool(seed or 0)
    feedback_list = list(feedback_sequence or [])
    rounds: list[EvolutionRound] = []
    current_personas: list[Persona] | int = initial_personas
    current_feedback = normalize_feedback(family_feedback)
    lineage: list[str] = [f"{spec.family}:base"]
    parent_version = "base"

    for generation in range(1, generations + 1):
        feedback = aggregate_feedback(
            feedback_list[generation - 1 : generation] if generation - 1 < len(feedback_list) else [],
            default=current_feedback,
        )
        current_personas, evolved_spec, mutation, manifest = evolve_one_generation(
            spec,
            generation,
            current_personas,
            model_pool=pool,
            feedback=feedback,
            seed=seed,
            parent_version=parent_version,
            lineage=lineage,
        )
        parent_version = mutation.family_version
        lineage = list(mutation.lineage)
        current_feedback = build_family_evolution_feedback(spec.family, [feedback], family_bias=current_feedback)
        rounds.append(
            EvolutionRound(
                generation=generation,
                family=evolved_spec.family,
                family_version=mutation.family_version,
                mutation_id=mutation.mutation_id,
                mutation_parent=mutation.mutation_parent,
                mutation_reason=mutation.mutation_reason,
                seed=mutation.seed,
                feedback=dict(feedback),
                manifest=manifest.to_dict(),
                mutation=mutation.to_dict(),
                family_spec=evolved_spec.to_dict(),
                personas=export_persona_list(current_personas),
            )
        )
    summary = {
        "family": spec.family,
        "generations": generations,
        "final_family_version": rounds[-1].family_version if rounds else "base",
        "final_mutation_id": rounds[-1].mutation_id if rounds else None,
        "lineage": lineage,
        "seed": seed,
        "mutation_summary": mutation_summary(rounds[-1].mutation) if rounds else "",
    }
    return EvolutionResult(family=spec.family, rounds=rounds, summary=summary)


def evolve_from_feedback_records(
    family: str | FamilySpec,
    personas: Iterable[Persona | dict[str, Any]],
    feedback_records: Iterable[dict[str, float]],
    *,
    generation: int = 1,
    model_pool: list[dict[str, Any]] | None = None,
    seed: int | None = None,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> tuple[list[Persona], FamilySpec, FamilyMutation, FamilyManifest]:
    feedback = aggregate_feedback(feedback_records)
    return evolve_one_generation(
        family,
        generation,
        personas,
        model_pool=model_pool,
        feedback=feedback,
        seed=seed,
        parent_version=parent_version,
        lineage=lineage,
    )


def build_evolution_report(result: EvolutionResult) -> dict[str, Any]:
    return result.to_dict()


def build_evolution_manifest(
    family: str | FamilySpec,
    generation: int,
    personas: Iterable[Persona | dict[str, Any]],
    *,
    model_pool: list[dict[str, Any]] | None = None,
    feedback: dict[str, float] | None = None,
    seed: int | None = None,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> dict[str, Any]:
    persona_list = list(personas)
    spec = get_family_spec(family) if isinstance(family, str) else family
    evolved_spec, mutation = mutate_family_spec(
        spec,
        generation,
        feedback=feedback,
        seed=seed,
        parent_version=parent_version,
        lineage=lineage,
    )
    mutated_personas = evolve_family_personas(
        evolved_spec,
        generation,
        persona_list,
        model_pool=model_pool or sample_model_pool(seed or 0),
        feedback=feedback,
        seed=seed,
        parent_version=mutation.family_version,
        lineage=mutation.lineage,
    )
    manifest = build_family_manifest(
        family=evolved_spec.family,
        generation=generation,
        personas=mutated_personas[0],
        model_pool=model_pool or sample_model_pool(seed or 0),
        feedback=feedback,
        seed=seed,
        family_spec=evolved_spec,
        mutation=mutation,
        family_version=mutation.family_version,
        dimension_axes=mutation.dimension_axes,
        time_profile=mutation.time_profile,
        lineage=mutation.lineage,
    )
    return manifest.to_dict()
