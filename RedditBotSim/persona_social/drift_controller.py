from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import hashlib
import os
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .drift_schema import DRIFT_AXES, DriftAxisSpec, MutationPlan, clamp, json_safe
from .family_schema import build_family_manifest
from .models import Persona
from .personas import load_default_personas


DEFAULT_MODEL_POOL: list[dict[str, Any]] = [
    {"provider": "openai", "model": "gpt-5.4", "temperature": 0.4, "reasoning_effort": "medium"},
    {"provider": "openai", "model": "gpt-5.4", "temperature": 0.8, "reasoning_effort": "medium"},
    {"provider": "gemini", "model": "gemini-3-flash-preview", "temperature": 0.4, "reasoning_effort": "medium"},
    {"provider": "gemini", "model": "gemini-3-pro-preview", "temperature": 0.8, "reasoning_effort": "medium"},
]


DRIFT_FAMILIES: dict[str, dict[str, Any]] = {
    "A_language_drift": {
        "slug": "a_language_drift",
        "family": "A_language_drift",
        "target_axes": ["language", "topic_content"],
        "auxiliary_axes": ["persona_population"],
        "frozen_axes": ["graph_structure", "temporal_rhythm"],
    },
    "C_graph_drift": {
        "slug": "c_graph_drift",
        "family": "C_graph_drift",
        "target_axes": ["graph_structure", "memory_revisit"],
        "auxiliary_axes": ["action_behavior", "temporal_rhythm"],
        "frozen_axes": ["language"],
    },
    "T_temporal_behavior_drift": {
        "slug": "t_temporal_behavior_drift",
        "family": "T_temporal_behavior_drift",
        "target_axes": ["temporal_rhythm", "action_behavior"],
        "auxiliary_axes": ["memory_revisit"],
        "frozen_axes": ["topic_content"],
    },
    "P_population_drift": {
        "slug": "p_population_drift",
        "family": "P_population_drift",
        "target_axes": ["persona_population", "llm_generation"],
        "auxiliary_axes": ["topic_content", "language"],
        "frozen_axes": ["graph_structure"],
    },
    "E_mixed_emergent_drift": {
        "slug": "e_mixed_emergent_drift",
        "family": "E_mixed_emergent_drift",
        "target_axes": ["language", "action_behavior", "graph_structure"],
        "auxiliary_axes": ["topic_content", "temporal_rhythm", "memory_revisit"],
        "frozen_axes": [],
    },
}


STYLE_VARIANTS = [
    "short, casual, a little uneven",
    "blunt, specific, mildly sarcastic",
    "questioning, phone-scrolling, not polished",
    "dry, concrete, sometimes annoyed",
    "chatty but still specific",
]

STANCE_VARIANTS = [
    "reacts to one detail instead of summarizing the whole story",
    "asks what this changes for normal people",
    "pushes back on neat framing",
    "trusts specifics more than official language",
    "does not mind leaving a rough first reaction",
]

REGION_VARIANTS = [
    "American, Ohio",
    "American, Arizona",
    "American, Florida",
    "Canadian, Toronto",
    "British, Manchester",
    "Australian, Sydney",
    "American, Oregon",
    "American, Wisconsin",
]

SUBREDDIT_VARIANTS = [
    ["politics", "news", "worldnews"],
    ["worldnews", "InternationalNews", "GlobalTalk"],
    ["news", "UpliftingNews", "politics"],
    ["GlobalTalk", "worldnews", "InternationalNews"],
    ["politics", "worldnews", "news", "GlobalTalk"],
]

TOPIC_KEYS = ["US politics", "world news", "Russia-Ukraine war", "Israel-Palestine"]
TOPIC_SUBREDDITS = {
    "US politics": [["politics", "news"], ["politics", "PoliticalDiscussion", "news"]],
    "world news": [["worldnews", "InternationalNews"], ["GlobalTalk", "worldnews", "news"]],
    "Russia-Ukraine war": [["worldnews", "InternationalNews", "GlobalTalk"], ["worldnews", "news"]],
    "Israel-Palestine": [["worldnews", "InternationalNews"], ["news", "GlobalTalk", "worldnews"]],
}


def stable_seed(*parts: object) -> int:
    payload = "|".join(json.dumps(part, ensure_ascii=False, sort_keys=True, default=str) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def family_config(name: str) -> dict[str, Any]:
    if name in DRIFT_FAMILIES:
        return dict(DRIFT_FAMILIES[name])
    lowered = name.lower()
    for cfg in DRIFT_FAMILIES.values():
        if lowered in {cfg["family"].lower(), cfg["slug"].lower()}:
            return dict(cfg)
    raise KeyError(f"Unknown drift family: {name}")


def family_names() -> list[str]:
    return list(DRIFT_FAMILIES)


def filtered_model_pool(providers: list[str] | None = None) -> list[dict[str, Any]]:
    pool = [dict(item) for item in DEFAULT_MODEL_POOL]
    for item in pool:
        model = os.environ.get(str(item["provider"]).upper() + "_MODEL")
        if model:
            item["model"] = model
    if not providers:
        return pool
    allowed = {provider.strip().lower() for provider in providers if provider.strip()}
    selected = [item for item in pool if str(item["provider"]).lower() in allowed]
    if not selected:
        raise ValueError("No supported drift provider selected; use openai and/or gemini.")
    return selected


def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.01, float(value)) for key, value in weights.items()}
    total = sum(cleaned.values()) or 1.0
    return {key: round(value / total, 4) for key, value in cleaned.items()}


def _axis_specs(target_axes: list[str], auxiliary_axes: list[str], frozen_axes: list[str]) -> list[DriftAxisSpec]:
    specs = []
    for axis in DRIFT_AXES:
        if axis in target_axes:
            role = "target"
        elif axis in auxiliary_axes:
            role = "auxiliary"
        elif axis in frozen_axes:
            role = "frozen"
        else:
            role = "watch"
        max_non_target = 0.22 if role == "frozen" else 0.35 if role == "watch" else 0.45
        specs.append(DriftAxisSpec(axis=axis, role=role, max_non_target_score=max_non_target))
    return specs


def build_mutation_plan(
    family_name: str,
    generation: int,
    attempt: int,
    seed: int,
    *,
    providers: list[str] | None = None,
    feedback: dict[str, float] | None = None,
    parent_family_version: str = "parent",
) -> MutationPlan:
    cfg = family_config(family_name)
    rng = random.Random(stable_seed(cfg["slug"], generation, attempt, seed, feedback or {}))
    feedback = dict(feedback or {})
    target_axes = list(cfg["target_axes"])
    auxiliary_axes = list(cfg.get("auxiliary_axes", []))
    if cfg["family"] == "E_mixed_emergent_drift":
        rng.shuffle(target_axes)
        target_axes = target_axes[: rng.choice([2, 3])]
    frozen_axes = list(cfg.get("frozen_axes", []))
    pressure_terms = []
    for key, value in feedback.items():
        key_text = str(key).lower()
        if any(token in key_text for token in ("pressure", "recall", "drift", "score")):
            try:
                pressure_terms.append(float(value))
            except (TypeError, ValueError):
                continue
    feedback_pressure = clamp(sum(pressure_terms) / len(pressure_terms), 0.0, 1.0) if pressure_terms else 0.0
    generation_gain = clamp(0.22 * max(0, generation - 1), 0.0, 0.44)
    pressure_gain = clamp(0.28 * feedback_pressure, 0.0, 0.28)
    axis_intensity = {}
    for axis in DRIFT_AXES:
        if axis in target_axes:
            low = clamp(0.66 + generation_gain + pressure_gain, 0.6, 0.94)
            high = clamp(0.9 + generation_gain + pressure_gain, low + 0.02, 0.99)
            axis_intensity[axis] = round(rng.uniform(low, high), 4)
        elif axis in auxiliary_axes:
            low = clamp(0.16 + 0.35 * generation_gain + 0.3 * pressure_gain, 0.1, 0.36)
            high = clamp(0.3 + 0.5 * generation_gain + 0.4 * pressure_gain, low + 0.02, 0.54)
            axis_intensity[axis] = round(rng.uniform(low, high), 4)
        elif axis in frozen_axes:
            axis_intensity[axis] = round(rng.uniform(0.0, 0.04), 4)
        else:
            low = clamp(0.03 + 0.12 * generation_gain + 0.12 * pressure_gain, 0.02, 0.12)
            high = clamp(0.1 + 0.22 * generation_gain + 0.2 * pressure_gain, low + 0.02, 0.22)
            axis_intensity[axis] = round(rng.uniform(low, high), 4)
    plan_payload = {
        "family": cfg["family"],
        "generation": generation,
        "attempt": attempt,
        "seed": seed,
        "target_axes": target_axes,
        "auxiliary_axes": auxiliary_axes,
        "axis_intensity": axis_intensity,
        "feedback": feedback,
    }
    plan_id = hashlib.sha256(json.dumps(plan_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    child_version = f"g{generation:03d}-a{attempt:02d}-{plan_id[:8]}"
    return MutationPlan(
        family=cfg["family"],
        slug=cfg["slug"],
        generation=generation,
        attempt=attempt,
        seed=seed,
        plan_id=plan_id,
        parent_family_version=parent_family_version,
        child_family_version=child_version,
        target_axes=target_axes,
        auxiliary_axes=auxiliary_axes,
        frozen_axes=frozen_axes,
        axis_specs=_axis_specs(target_axes, auxiliary_axes, frozen_axes),
        axis_intensity=axis_intensity,
        target_metrics=_target_metrics(axis_intensity, target_axes),
        model_pool=filtered_model_pool(providers),
        feedback=feedback,
        notes=[
            "Mutation is statistics-driven; it does not minimize detector probability.",
            "Target axes must pass drift audit before entering detector evaluation.",
            f"Generation gain={generation_gain:.3f}, feedback gain={pressure_gain:.3f}, feedback pressure={feedback_pressure:.3f}.",
            "Target axes are intentionally amplified to make family-level drift more visible across generations.",
        ],
    )


def _target_metrics(axis_intensity: dict[str, float], target_axes: list[str]) -> dict[str, Any]:
    metrics = {}
    for axis in target_axes:
        strength = axis_intensity.get(axis, 0.0)
        metrics[axis] = {
            "target_drift_score_min": 0.15,
            "nominal_strength": strength,
        }
    return metrics


def build_parent_personas(
    family_name: str,
    count: int,
    *,
    seed: int,
    providers: list[str] | None = None,
) -> list[Persona]:
    cfg = family_config(family_name)
    pool = filtered_model_pool(providers)
    base_personas = load_default_personas(count)
    personas: list[Persona] = []
    for idx, persona in enumerate(base_personas, start=1):
        model_cfg = pool[(idx - 1) % len(pool)] if pool else {}
        user_id = f"{cfg['slug']}_parent_{idx:03d}"
        traits = dict(persona.traits)
        traits.update(
            {
                "drift_role": "parent_control",
                "sentence_shape": "baseline mixed",
                "thread_memory": "ordinary",
                "temporal_profile": "baseline",
            }
        )
        personas.append(
            replace(
                persona,
                user_id=user_id,
                user_name=f"{persona.user_name}_{cfg['slug']}_parent",
                family=cfg["family"],
                family_generation=0,
                family_round=0,
                llm_provider=str(model_cfg.get("provider", persona.llm_provider or "openai")),
                llm_model=str(model_cfg.get("model", persona.llm_model or "gpt-5.2")),
                llm_temperature=float(model_cfg.get("temperature", persona.llm_temperature or 0.4)),
                llm_reasoning_effort=str(model_cfg.get("reasoning_effort", persona.llm_reasoning_effort or "medium")),
                traits=traits,
            )
        )
    return personas


def mutate_personas(parent_personas: list[Persona], plan: MutationPlan) -> list[Persona]:
    rng = random.Random(stable_seed(plan.plan_id, "mutate"))
    pool = plan.model_pool or DEFAULT_MODEL_POOL
    personas: list[Persona] = []
    for idx, persona in enumerate(parent_personas, start=1):
        local_rng = random.Random(stable_seed(plan.plan_id, persona.user_id, idx))
        payload = asdict(persona)
        payload["user_id"] = f"{plan.slug}_{plan.child_family_version}_{idx:03d}".replace("-", "_")
        payload["user_name"] = f"{persona.user_name.split('_')[0]}_{plan.slug}_g{plan.generation}"
        payload["family"] = plan.family
        payload["family_generation"] = plan.generation
        payload["family_round"] = plan.generation
        payload["traits"] = dict(payload.get("traits") or {})
        payload["traits"].update(
            {
                "drift_role": "child_candidate",
                "mutation_plan_id": plan.plan_id,
                "target_axes": ",".join(plan.target_axes),
                "child_family_version": plan.child_family_version,
                "family_drift_brief": f"{plan.family} generation {plan.generation} targeting {', '.join(plan.target_axes)}",
                "family_axes": json.dumps(plan.axis_intensity, ensure_ascii=False, sort_keys=True),
                "family_feedback": json.dumps(plan.feedback, ensure_ascii=False, sort_keys=True),
                "family_generation": plan.generation,
            }
        )
        _apply_language(payload, plan, local_rng)
        _apply_topic(payload, plan, local_rng, idx)
        _apply_population(payload, plan, local_rng, idx)
        _apply_action_behavior(payload, plan, local_rng, idx)
        _apply_graph(payload, plan, local_rng)
        _apply_temporal(payload, plan, local_rng)
        _apply_memory(payload, plan, local_rng)
        _apply_llm(payload, plan, local_rng, idx)
        personas.append(Persona(**{key: payload[key] for key in Persona.__dataclass_fields__ if key in payload}))
    rng.shuffle(personas)
    return personas


def write_manifest(path: Path, family: str, generation: int, personas: list[Persona], plan: MutationPlan | None = None) -> None:
    manifest = build_family_manifest(
        family=family,
        generation=generation,
        personas=personas,
        model_pool=plan.model_pool if plan else filtered_model_pool(),
        feedback=plan.feedback if plan else {},
        seed=plan.seed if plan else None,
        family_version=plan.child_family_version if plan else "parent",
        dimension_axes=plan.axis_intensity if plan else {},
    ).to_dict()
    if plan:
        manifest["mutation_plan"] = plan.to_dict()
    path.write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")


def _intensity(plan: MutationPlan, axis: str) -> float:
    return float(plan.axis_intensity.get(axis, 0.0))


def _apply_language(payload: dict[str, Any], plan: MutationPlan, rng: random.Random) -> None:
    strength = _intensity(plan, "language")
    if strength <= 0.05:
        return
    payload["style"] = f"{rng.choice(STYLE_VARIANTS)}, drift strength {strength:.2f}"
    payload["stance"] = rng.choice(STANCE_VARIANTS)
    payload["brevity"] = round(clamp(float(payload["brevity"]) + rng.uniform(-0.34, 0.26) * strength, 0.08, 0.99), 3)
    payload["debate_tendency"] = round(clamp(float(payload["debate_tendency"]) + rng.uniform(-0.18, 0.26) * strength, 0.02, 0.99), 3)
    payload["traits"].update(
        {
            "sentence_shape": rng.choice(["short fragments", "uneven phone comments", "question-heavy", "dry one-detail reactions"]),
            "reply_texture": rng.choice(["not polished", "specific and blunt", "casual pushback", "low-drama"]),
            "formality": round(clamp(0.42 - 0.34 * strength + rng.uniform(-0.08, 0.08)), 3),
            "hedging": round(clamp(rng.uniform(0.08, 0.88)), 3),
            "sarcasm": round(clamp(rng.uniform(0.08, 0.8) * strength), 3),
            "question_rate": round(clamp(rng.uniform(0.02, 0.78) * strength + 0.04), 3),
            "stance_strength": round(clamp(0.28 + rng.uniform(0.1, 0.56) * strength), 3),
            "target_mean_words": int(rng.choice([6, 9, 14, 22, 34, 48]) * (1.0 + 0.45 * (1.0 - strength))),
        }
    )


def _apply_topic(payload: dict[str, Any], plan: MutationPlan, rng: random.Random, idx: int) -> None:
    strength = _intensity(plan, "topic_content")
    if strength <= 0.05:
        return
    dominant_idx = stable_seed(plan.plan_id, "dominant_topic") % len(TOPIC_KEYS)
    dominant = TOPIC_KEYS[dominant_idx]
    if idx % 5 in {0, 3}:
        focus = TOPIC_KEYS[(dominant_idx + idx) % len(TOPIC_KEYS)]
    else:
        focus = dominant
    spillover = TOPIC_KEYS[(TOPIC_KEYS.index(focus) + 1 + idx) % len(TOPIC_KEYS)]
    weights = {topic: 0.02 + rng.random() * 0.02 for topic in TOPIC_KEYS}
    weights[focus] = 0.74 + 0.18 * strength
    weights[spillover] = max(weights[spillover], 0.08 + 0.12 * (1.0 - strength))
    payload["interest_weights"] = _normalise_weights(weights)
    payload["subreddits"] = list(rng.choice(TOPIC_SUBREDDITS.get(focus, SUBREDDIT_VARIANTS)))
    payload["exploration_tendency"] = round(clamp(float(payload["exploration_tendency"]) + 0.24 * strength, 0.02, 0.98), 3)
    payload["traits"].update(
        {
            "topic_focus": focus,
            "dominant_topic": dominant,
            "topic_drift_strength": round(strength, 3),
            "topic_sampling": "dominant-cluster-with-spillover",
        }
    )


def _apply_population(payload: dict[str, Any], plan: MutationPlan, rng: random.Random, idx: int) -> None:
    strength = _intensity(plan, "persona_population")
    if strength <= 0.05:
        return
    payload["age"] = int(clamp(int(payload["age"]) + rng.choice([-13, -8, -4, 5, 9, 14]) * strength, 18, 72))
    payload["region"] = rng.choice(REGION_VARIANTS)
    if idx % 5 == 0:
        payload["ideology"] = rng.choice(["Liberal", "Moderate", "Conservative", "Independent"])
    payload["description"] = f"{payload['description']} Has a more distinct browsing rhythm and topic lens."
    payload["traits"].update(
        {
            "population_cluster": rng.choice(["low_active", "ordinary_active", "argumentative", "lurker_poster", "thread_returner"]),
            "persona_diversity_strength": round(strength, 3),
        }
    )


def _apply_action_behavior(payload: dict[str, Any], plan: MutationPlan, rng: random.Random, idx: int) -> None:
    strength = _intensity(plan, "action_behavior")
    if strength <= 0.05:
        return
    active_cluster = idx % 4
    if active_cluster == 0:
        post_delta, reply_delta = 0.34, -0.08
    elif active_cluster == 1:
        post_delta, reply_delta = -0.08, 0.34
    elif active_cluster == 2:
        post_delta, reply_delta = -0.18, -0.14
    else:
        post_delta, reply_delta = 0.16, 0.22
    payload["post_tendency"] = round(clamp(float(payload["post_tendency"]) + post_delta * strength, 0.01, 0.92), 3)
    payload["reply_tendency"] = round(clamp(float(payload["reply_tendency"]) + reply_delta * strength, 0.02, 0.98), 3)
    payload["exploration_tendency"] = round(clamp(float(payload["exploration_tendency"]) + rng.uniform(0.08, 0.32) * strength, 0.02, 0.98), 3)
    payload["debate_tendency"] = round(clamp(float(payload["debate_tendency"]) + rng.uniform(-0.14, 0.26) * strength, 0.02, 0.98), 3)
    payload["traits"].update({"activity_cluster": active_cluster, "skip_bias": round(clamp(0.08 + (0.26 if active_cluster == 2 else 0.04) * strength), 3)})


def _apply_graph(payload: dict[str, Any], plan: MutationPlan, rng: random.Random) -> None:
    strength = _intensity(plan, "graph_structure")
    if strength <= 0.05:
        return
    payload["graph_affinity"] = round(clamp(float(payload["graph_affinity"]) + rng.uniform(0.18, 0.42) * strength, 0.04, 0.99), 3)
    payload["reply_tendency"] = round(clamp(float(payload["reply_tendency"]) + rng.uniform(0.08, 0.24) * strength, 0.04, 0.99), 3)
    payload["traits"].update(
        {
            "graph_target": rng.choice(["weak_ties", "more_human_edges", "thread_locality", "lower_reciprocity", "wider_author_mix"]),
            "target_bot_authored_cap": 1 if strength >= 0.85 else 2,
        }
    )


def _apply_temporal(payload: dict[str, Any], plan: MutationPlan, rng: random.Random) -> None:
    strength = _intensity(plan, "temporal_rhythm")
    if strength <= 0.05:
        return
    profile = {
        "post_delay_min": int(rng.choice([0, 5, 20, 45, 90]) * strength),
        "post_delay_max": int(20 + rng.choice([60, 180, 420, 900, 1800]) * strength),
        "comment_delay_min": int(2 + rng.choice([8, 25, 60, 180]) * strength),
        "comment_delay_max": int(60 + rng.choice([210, 600, 1440, 2880]) * strength),
        "burstiness": round(clamp(rng.uniform(0.35, 1.0) * strength), 3),
        "silence_ratio": round(clamp(rng.uniform(0.1, 0.55) * strength), 3),
    }
    payload["traits"].update({"temporal_profile": profile})


def _apply_memory(payload: dict[str, Any], plan: MutationPlan, rng: random.Random) -> None:
    strength = _intensity(plan, "memory_revisit")
    if strength <= 0.05:
        return
    payload["revisit_tendency"] = round(clamp(float(payload["revisit_tendency"]) + rng.uniform(0.18, 0.42) * strength, 0.02, 0.99), 3)
    payload["notification_reactivity"] = round(clamp(float(payload["notification_reactivity"]) + rng.uniform(0.18, 0.4) * strength, 0.02, 0.99), 3)
    payload["traits"].update(
        {
            "thread_memory": rng.choice(["checks replies later", "sometimes returns but not always", "notices repeat authors", "keeps loose thread memory"]),
            "notification_window": round(clamp(0.16 + 0.66 * strength + rng.uniform(-0.1, 0.1)), 3),
        }
    )


def _apply_llm(payload: dict[str, Any], plan: MutationPlan, rng: random.Random, idx: int) -> None:
    strength = _intensity(plan, "llm_generation")
    pool = list(plan.model_pool or DEFAULT_MODEL_POOL)
    if not pool:
        return
    if "llm_generation" not in set(plan.target_axes) | set(plan.auxiliary_axes):
        return
    if strength > 0.05:
        rng.shuffle(pool)
        model_cfg = pool[(idx + int(strength * 10)) % len(pool)]
        temperature = float(model_cfg.get("temperature", 0.4))
        temperature = clamp(temperature + rng.uniform(-0.15, 0.25) * strength, 0.15, 1.1)
    else:
        model_cfg = pool[(idx - 1) % len(pool)]
        temperature = float(model_cfg.get("temperature", 0.4))
    payload["llm_provider"] = str(model_cfg.get("provider", "openai"))
    payload["llm_model"] = str(model_cfg.get("model", "gpt-5.2"))
    payload["llm_temperature"] = round(float(temperature), 3)
    payload["llm_reasoning_effort"] = str(model_cfg.get("reasoning_effort", "medium"))


def manifest_records(personas: list[Persona]) -> list[dict[str, Any]]:
    return [json_safe(asdict(persona)) for persona in personas]
