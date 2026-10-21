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
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Iterable

from .family_registry import get_family_spec, family_time_profile, normalize_family_name
from .family_schema import FamilySpec, build_family_manifest as build_manifest_payload
from .models import Persona
from .personas import build_family_personas as _legacy_build_family_personas


MUTATION_SCHEMA_VERSION = "phase4"


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


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _stable_seed(*parts: object) -> int:
    payload = "|".join(json.dumps(part, ensure_ascii=False, sort_keys=True, default=str) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _normalise_feedback(feedback: dict[str, float] | None) -> dict[str, float]:
    if not feedback:
        return {}
    out: dict[str, float] = {}
    for key, value in feedback.items():
        if value is None:
            continue
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _dominant_key(values: dict[str, float], default: str = "language") -> str:
    if not values:
        return default
    return max(values, key=lambda key: values[key])


def _family_axis_weights(spec: FamilySpec) -> dict[str, float]:
    axes = dict(spec.dimension_axes or spec.axes or {})
    base = {"language": 0.25, "behavior": 0.25, "graph": 0.25, "time": 0.25}
    if axes:
        base.update({str(key): _safe_float(value, base.get(str(key), 0.25)) for key, value in axes.items()})
    return base


def _extract_pressures(feedback: dict[str, float]) -> dict[str, float]:
    text_keys = (
        "text_pressure",
        "language_pressure",
        "lexical_pressure",
        "style_pressure",
        "text_recall",
        "detector_text_recall",
        "bot_text_recall",
    )
    behavior_keys = (
        "behavior_pressure",
        "action_pressure",
        "policy_pressure",
        "behavior_recall",
        "detector_behavior_recall",
        "bot_behavior_recall",
    )
    graph_keys = (
        "graph_pressure",
        "network_pressure",
        "edge_pressure",
        "graph_recall",
        "detector_graph_recall",
        "bot_graph_recall",
    )
    time_keys = (
        "time_pressure",
        "temporal_pressure",
        "rhythm_pressure",
        "time_recall",
        "detector_time_recall",
        "bot_time_recall",
    )
    overall_keys = (
        "pressure",
        "detector_pressure",
        "bot_recall",
        "recall",
        "score",
    )

    def _max_value(keys: tuple[str, ...]) -> float:
        vals = [_safe_float(feedback.get(key), 0.0) for key in keys if key in feedback]
        return max(vals) if vals else 0.0

    text = _max_value(text_keys)
    behavior = _max_value(behavior_keys)
    graph = _max_value(graph_keys)
    time_pressure = _max_value(time_keys)
    overall = _max_value(overall_keys)

    if overall > 0:
        text = max(text, overall * 0.8)
        behavior = max(behavior, overall * 0.75)
        graph = max(graph, overall * 0.75)
        time_pressure = max(time_pressure, overall * 0.65)

    if not any((text, behavior, graph, time_pressure)):
        text = behavior = graph = time_pressure = 0.18

    return {
        "language": _clamp(text, 0.0, 1.0),
        "behavior": _clamp(behavior, 0.0, 1.0),
        "graph": _clamp(graph, 0.0, 1.0),
        "time": _clamp(time_pressure, 0.0, 1.0),
    }


def _build_time_profile(
    family: str,
    pressures: dict[str, float],
    rng: random.Random,
) -> dict[str, float]:
    family_key = normalize_family_name(family)
    base = family_time_profile(family_key)
    if not base:
        base = {
            "burstiness": 0.35,
            "silence_ratio": 0.35,
            "revisit_delay": 0.35,
            "day_cycle": 0.35,
            "notification_window": 0.35,
        }
    language = pressures.get("language", 0.0)
    behavior = pressures.get("behavior", 0.0)
    graph = pressures.get("graph", 0.0)
    time_pressure = pressures.get("time", 0.0)
    drift = 0.03 + 0.12 * (language + behavior + graph + time_pressure) / 4.0
    profile = dict(base)
    profile["burstiness"] = _clamp(_safe_float(profile.get("burstiness"), 0.35) + drift + rng.uniform(-0.05, 0.08), 0.05, 0.98)
    profile["silence_ratio"] = _clamp(_safe_float(profile.get("silence_ratio"), 0.35) + 0.07 * time_pressure + rng.uniform(-0.03, 0.06), 0.05, 0.98)
    profile["revisit_delay"] = _clamp(_safe_float(profile.get("revisit_delay"), 0.35) + 0.08 * graph + rng.uniform(-0.04, 0.05), 0.05, 0.98)
    profile["day_cycle"] = _clamp(_safe_float(profile.get("day_cycle"), 0.35) + 0.06 * behavior + rng.uniform(-0.02, 0.04), 0.05, 0.98)
    profile["notification_window"] = _clamp(_safe_float(profile.get("notification_window"), 0.35) + 0.07 * graph + rng.uniform(-0.03, 0.05), 0.05, 0.98)
    return {key: round(value, 4) for key, value in profile.items()}


def _dimension_deltas(spec: FamilySpec, pressures: dict[str, float], rng: random.Random) -> dict[str, float]:
    axes = _family_axis_weights(spec)
    family = normalize_family_name(spec.family)
    family_bias = {
        "Family A: Language-Persona": {"language": 0.55, "behavior": 0.18, "graph": 0.10, "time": 0.17},
        "Family C: Graph-Aware": {"language": 0.18, "behavior": 0.24, "graph": 0.42, "time": 0.16},
        "Family E: Emergent": {"language": 0.25, "behavior": 0.25, "graph": 0.25, "time": 0.25},
    }.get(family, {"language": 0.25, "behavior": 0.25, "graph": 0.25, "time": 0.25})
    deltas: dict[str, float] = {}
    for axis in ("language", "behavior", "graph", "time"):
        pressure = pressures.get(axis, 0.0)
        leverage = 0.45 + 0.45 * axes.get(axis, 0.25) + 0.15 * family_bias.get(axis, 0.25)
        noise = rng.uniform(-0.015, 0.04)
        if family == "Family E: Emergent":
            noise += rng.uniform(0.0, 0.025)
        delta = 0.02 + leverage * (0.18 * pressure + 0.04 * sum(pressures.values()) / 4.0) + noise
        deltas[axis] = round(_clamp(delta, 0.005, 0.22), 4)
    return deltas


def _mutation_reason(spec: FamilySpec, pressures: dict[str, float], deltas: dict[str, float]) -> str:
    dominant = _dominant_key(pressures)
    dominant_delta = _dominant_key(deltas)
    family_name = normalize_family_name(spec.family)
    if family_name == "Family A: Language-Persona":
        return f"language-led drift after {dominant} pressure; expand lexical variation on {dominant_delta}"
    if family_name == "Family C: Graph-Aware":
        return f"graph-led drift after {dominant} pressure; widen thread-awareness and revisit behavior"
    if family_name == "Family E: Emergent":
        return f"feedback-sensitive drift after {dominant} pressure; let all axes co-evolve"
    return f"mutation driven by {dominant} pressure"


def _mutation_depth(lineage: list[str]) -> int:
    return max(0, len(lineage) - 1)


@dataclass(frozen=True)
class FamilyMutation:
    family: str
    family_version: str
    mutation_id: str
    mutation_parent: str | None = None
    mutation_reason: str = ""
    generation: int = 0
    mutation_depth: int = 0
    seed: int | None = None
    feedback: dict[str, float] = field(default_factory=dict)
    dimension_axes: dict[str, float] = field(default_factory=dict)
    dimension_deltas: dict[str, float] = field(default_factory=dict)
    time_profile: dict[str, float] = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)
    family_seed: int | None = None
    family_schema_version: str = MUTATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def build_mutation_plan(
    family: str | FamilySpec,
    generation: int,
    *,
    feedback: dict[str, float] | None = None,
    seed: int | None = None,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> FamilyMutation:
    spec = get_family_spec(family) if isinstance(family, str) else family
    normalized_feedback = _normalise_feedback(feedback)
    pressures = _extract_pressures(normalized_feedback)
    lineage_list = list(lineage or [])
    if not lineage_list:
        lineage_list = [f"{spec.family}:base"]
    base_seed = seed if seed is not None else _stable_seed(spec.family, generation, parent_version or "base", normalized_feedback)
    rng = random.Random(base_seed)
    deltas = _dimension_deltas(spec, pressures, rng)
    axes = _family_axis_weights(spec)
    mutated_axes = {
        axis: round(_clamp(axes.get(axis, 0.25) + deltas.get(axis, 0.02), 0.01, 0.99), 4)
        for axis in ("language", "behavior", "graph", "time")
    }
    time_profile = _build_time_profile(spec.family, pressures, rng)
    mutation_id = hashlib.sha256(
        json.dumps(
            {
                "family": spec.family,
                "generation": generation,
                "parent_version": parent_version,
                "seed": base_seed,
                "feedback": normalized_feedback,
                "deltas": deltas,
                "axes": mutated_axes,
                "time_profile": time_profile,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:12]
    family_version = f"g{generation:03d}-{mutation_id[:8]}"
    mutation_parent = parent_version if parent_version not in (None, "") else (lineage_list[-1] if lineage_list else None)
    mutation_reason = _mutation_reason(spec, pressures, deltas)
    resolved_lineage = list(lineage_list)
    resolved_lineage.append(f"{spec.family}:{family_version}")
    return FamilyMutation(
        family=spec.family,
        family_version=family_version,
        mutation_id=mutation_id,
        mutation_parent=mutation_parent,
        mutation_reason=mutation_reason,
        generation=generation,
        mutation_depth=_mutation_depth(resolved_lineage),
        seed=base_seed,
        feedback=normalized_feedback,
        dimension_axes=mutated_axes,
        dimension_deltas=deltas,
        time_profile=time_profile,
        lineage=resolved_lineage,
        family_seed=base_seed,
    )


def mutate_family_spec(
    family: str | FamilySpec,
    generation: int,
    *,
    feedback: dict[str, float] | None = None,
    seed: int | None = None,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> tuple[FamilySpec, FamilyMutation]:
    base_spec = get_family_spec(family) if isinstance(family, str) else family
    mutation = build_mutation_plan(
        base_spec,
        generation,
        feedback=feedback,
        seed=seed,
        parent_version=parent_version,
        lineage=lineage,
    )
    base_payload = base_spec.to_dict()
    base_payload.update(
        {
            "dimension_axes": mutation.dimension_axes,
            "time_profile": mutation.time_profile,
            "family_version": mutation.family_version,
            "mutation_parent": mutation.mutation_parent,
            "mutation_reason": mutation.mutation_reason,
            "mutation_id": mutation.mutation_id,
            "mutation_depth": mutation.mutation_depth,
            "mutation_deltas": mutation.dimension_deltas,
            "lineage": mutation.lineage,
            "feedback": mutation.feedback,
            "family_seed": mutation.family_seed,
        }
    )
    if mutation.mutation_reason:
        base_payload["description"] = f"{base_spec.description} Mutation note: {mutation.mutation_reason}."
        prompt_notes = list(base_payload.get("prompt_notes", []))
        prompt_notes.append(f"Mutation {mutation.family_version}: {mutation.mutation_reason}")
        base_payload["prompt_notes"] = prompt_notes
        behavior_notes = list(base_payload.get("behavior_notes", []))
        behavior_notes.append(
            f"Generation {generation} feedback redirects behavior through {max(mutation.dimension_deltas, key=mutation.dimension_deltas.get)}."
        )
        base_payload["behavior_notes"] = behavior_notes
    evolved_spec = FamilySpec(**base_payload)
    return evolved_spec, mutation


def _persona_to_dict(persona: Persona | dict[str, Any]) -> dict[str, Any]:
    if isinstance(persona, Persona):
        payload = dict(persona.__dict__)
    elif isinstance(persona, dict):
        payload = dict(persona)
    else:
        raise TypeError(f"Unsupported persona payload type: {type(persona)!r}")
    return payload


def mutate_persona_payload(
    persona: Persona | dict[str, Any],
    mutation: FamilyMutation,
    *,
    index: int = 0,
    family: str | None = None,
) -> dict[str, Any]:
    payload = _persona_to_dict(persona)
    family_name = family or mutation.family
    payload["family"] = family_name
    payload["family_generation"] = mutation.generation
    payload["family_round"] = mutation.generation
    payload["family_version"] = mutation.family_version
    payload["mutation_parent"] = mutation.mutation_parent
    payload["mutation_reason"] = mutation.mutation_reason
    payload["mutation_id"] = mutation.mutation_id
    payload["mutation_depth"] = mutation.mutation_depth
    payload["family_seed"] = mutation.family_seed
    payload["dimension_axes"] = dict(mutation.dimension_axes)
    payload["time_profile"] = dict(mutation.time_profile)
    payload["mutation_deltas"] = dict(mutation.dimension_deltas)
    payload["lineage"] = list(mutation.lineage)
    payload["feedback"] = dict(mutation.feedback)

    rng = random.Random(_stable_seed(mutation.mutation_id, payload.get("user_id", index), family_name, index))

    base_style = str(payload.get("style", ""))
    base_stance = str(payload.get("stance", ""))
    base_description = str(payload.get("description", ""))

    if mutation.dimension_deltas.get("language", 0.0) > 0.0:
        style_suffixes = [
            "more conversational and uneven",
            "more blunt and casual",
            "more hedged and thread-aware",
            "more specific and low-drama",
            "more chatty and slightly messy",
        ]
        stance_suffixes = [
            "leans on lived reaction over neat framing",
            "asks for the point that matters to regular readers",
            "pushes back on polished language more aggressively",
            "keeps replies short but pointed",
            "lets small details steer the take",
        ]
        payload["style"] = f"{base_style}, {rng.choice(style_suffixes)}".strip(", ")
        payload["stance"] = f"{base_stance}; {rng.choice(stance_suffixes)}".strip("; ")
        payload["brevity"] = round(_clamp(_safe_float(payload.get("brevity"), 0.5) + rng.uniform(-0.07, 0.09), 0.2, 0.98), 3)

    if mutation.dimension_deltas.get("behavior", 0.0) > 0.0:
        payload["post_tendency"] = round(_clamp(_safe_float(payload.get("post_tendency"), 0.2) + rng.uniform(-0.02, 0.08), 0.02, 0.95), 3)
        payload["reply_tendency"] = round(_clamp(_safe_float(payload.get("reply_tendency"), 0.4) + rng.uniform(0.0, 0.12), 0.02, 0.98), 3)
        payload["exploration_tendency"] = round(_clamp(_safe_float(payload.get("exploration_tendency"), 0.3) + rng.uniform(-0.01, 0.09), 0.02, 0.98), 3)

    if mutation.dimension_deltas.get("graph", 0.0) > 0.0:
        payload["revisit_tendency"] = round(_clamp(_safe_float(payload.get("revisit_tendency"), 0.2) + rng.uniform(0.01, 0.11), 0.02, 0.98), 3)
        payload["graph_affinity"] = round(_clamp(_safe_float(payload.get("graph_affinity"), 0.5) + rng.uniform(0.03, 0.13), 0.02, 0.99), 3)
        payload["notification_reactivity"] = round(
            _clamp(_safe_float(payload.get("notification_reactivity"), 0.5) + rng.uniform(0.03, 0.13), 0.02, 0.99),
            3,
        )

    if mutation.dimension_deltas.get("time", 0.0) > 0.0:
        payload["exploration_tendency"] = round(_clamp(_safe_float(payload.get("exploration_tendency"), 0.3) + rng.uniform(0.0, 0.1), 0.02, 0.98), 3)
        payload["revisit_tendency"] = round(_clamp(_safe_float(payload.get("revisit_tendency"), 0.2) + rng.uniform(0.0, 0.08), 0.02, 0.98), 3)

    if base_description:
        payload["description"] = f"{base_description} Evolves with {mutation.mutation_reason.lower()}."

    interests = dict(payload.get("interest_weights", {}) or {})
    if interests:
        topic_keys = list(interests.keys())
        for topic_index, topic in enumerate(topic_keys):
            raw = _safe_float(interests.get(topic), 0.0)
            topic_bias = 0.0
            if mutation.dimension_deltas.get("language", 0.0) > 0.0:
                topic_bias += 0.02 * ((topic_index + index) % 3 - 1)
            if mutation.dimension_deltas.get("graph", 0.0) > 0.0:
                topic_bias += 0.03 if topic_index % 2 == 0 else -0.01
            if mutation.dimension_deltas.get("time", 0.0) > 0.0:
                topic_bias += 0.02 if topic_index == (index % max(1, len(topic_keys))) else 0.0
            interests[topic] = round(_clamp(raw * (1.0 + topic_bias), 0.01, 0.99), 4)
        total = sum(max(0.01, value) for value in interests.values())
        payload["interest_weights"] = {key: round(max(0.01, value) / total, 4) for key, value in interests.items()}

    model_pool = payload.get("model_pool")
    if isinstance(model_pool, list) and model_pool:
        model_choice = model_pool[index % len(model_pool)]
        if isinstance(model_choice, dict):
            payload["llm_provider"] = str(model_choice.get("provider", payload.get("llm_provider", "openai")))
            payload["llm_model"] = str(model_choice.get("model", payload.get("llm_model", "gpt-5.4")))
            payload["llm_temperature"] = _safe_float(model_choice.get("temperature"), _safe_float(payload.get("llm_temperature"), 0.4))
            payload["llm_reasoning_effort"] = str(model_choice.get("reasoning_effort", payload.get("llm_reasoning_effort", "medium")))

    return payload


def mutate_personas(
    personas: Iterable[Persona | dict[str, Any]],
    mutation: FamilyMutation,
    *,
    family: str | None = None,
) -> list[Persona]:
    mutated: list[Persona] = []
    for index, persona in enumerate(personas):
        payload = mutate_persona_payload(persona, mutation, index=index, family=family)
        persona_obj = Persona(**{k: payload[k] for k in Persona.__dataclass_fields__.keys() if k in payload})
        for extra_key in (
            "family_version",
            "mutation_parent",
            "mutation_reason",
            "mutation_id",
            "mutation_depth",
            "family_seed",
            "dimension_axes",
            "time_profile",
            "mutation_deltas",
            "lineage",
            "feedback",
        ):
            if extra_key in payload:
                object.__setattr__(persona_obj, extra_key, payload[extra_key])
        mutated.append(persona_obj)
    return mutated


def build_evolved_family_bundle(
    family: str | FamilySpec,
    generation: int,
    personas: Iterable[Persona | dict[str, Any]] | int,
    *,
    model_pool: list[dict[str, Any]] | None = None,
    feedback: dict[str, float] | None = None,
    seed: int | None = None,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> dict[str, Any]:
    spec = get_family_spec(family) if isinstance(family, str) else family
    evolved_spec, mutation = mutate_family_spec(
        spec,
        generation,
        feedback=feedback,
        seed=seed,
        parent_version=parent_version,
        lineage=lineage,
    )
    if isinstance(personas, int):
        persona_records = mutate_personas(
            _legacy_build_family_personas(
                spec.family,
                generation,
                personas,
                model_pool or [],
                feedback=feedback,
            ),
            mutation,
            family=evolved_spec.family,
        )
    else:
        persona_records = mutate_personas(list(personas), mutation, family=evolved_spec.family)
    manifest = build_manifest_payload(
        family=evolved_spec.family,
        generation=generation,
        personas=persona_records,
        model_pool=model_pool,
        feedback=feedback or mutation.feedback,
        seed=seed if seed is not None else mutation.seed,
        family_spec=evolved_spec,
        mutation=mutation,
        family_version=mutation.family_version,
        dimension_axes=mutation.dimension_axes,
        time_profile=mutation.time_profile,
        lineage=mutation.lineage,
    )
    return {"spec": evolved_spec, "mutation": mutation.to_dict(), "manifest": manifest.to_dict()}


def evolve_family_personas(
    family: str | FamilySpec,
    generation: int,
    personas: Iterable[Persona | dict[str, Any]] | int,
    *,
    model_pool: list[dict[str, Any]] | None = None,
    feedback: dict[str, float] | None = None,
    seed: int | None = None,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> tuple[list[Persona], FamilySpec, FamilyMutation]:
    spec = get_family_spec(family) if isinstance(family, str) else family
    evolved_spec, mutation = mutate_family_spec(
        spec,
        generation,
        feedback=feedback,
        seed=seed,
        parent_version=parent_version,
        lineage=lineage,
    )
    if isinstance(personas, int):
        base_personas = _legacy_build_family_personas(
            spec.family,
            generation,
            personas,
            model_pool or [],
            feedback=feedback,
        )
        mutated = mutate_personas(base_personas, mutation, family=evolved_spec.family)
    else:
        mutated = mutate_personas(personas, mutation, family=evolved_spec.family)
    return mutated, evolved_spec, mutation


def plan_from_feedback(
    family: str | FamilySpec,
    generation: int,
    feedback: dict[str, float] | None = None,
    *,
    seed: int | None = None,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> dict[str, Any]:
    _, mutation = mutate_family_spec(
        family,
        generation,
        feedback=feedback,
        seed=seed,
        parent_version=parent_version,
        lineage=lineage,
    )
    return mutation.to_dict()


def mutation_summary(mutation: FamilyMutation | dict[str, Any]) -> str:
    payload = mutation.to_dict() if isinstance(mutation, FamilyMutation) else dict(mutation)
    return (
        f"{payload.get('family', '')} {payload.get('family_version', '')} "
        f"parent={payload.get('mutation_parent', '')} reason={payload.get('mutation_reason', '')}"
    ).strip()
