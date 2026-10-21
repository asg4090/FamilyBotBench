from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import re
from typing import Any

from .family_schema import FamilySpec, SCHEMA_VERSION


def _norm_key(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return " ".join(cleaned.split())


FAMILY_A_LANGUAGE_PERSONA = FamilySpec(
    family="Family A: Language-Persona",
    slug="family_a_language_persona",
    description="Language-first family that mainly varies tone, wording, hedging, and comment length while keeping the same topical world.",
    axes={"language": 0.95, "behavior": 0.55, "graph": 0.20, "emergence": 0.35},
    dimension_axes={"language": 0.95, "behavior": 0.55, "graph": 0.20, "time": 0.30},
    prompt_notes=[
        "Bias toward conversational phrasing and uneven, human-like sentence shape.",
        "Let brevity, sarcasm, hedging, and informal follow-ups vary strongly across members.",
        "Keep topic distribution stable enough that differences mostly come from wording and stance.",
    ],
    behavior_notes=[
        "Reply and debate rates should vary more than posting rate.",
        "Prefer visible lexical style change over structural graph change.",
    ],
    graph_notes=[
        "Keep graph behavior close to the base distribution.",
        "Do not force extra mutual-reply structure in this family.",
    ],
    time_notes=[
        "Keep daily rhythm close to baseline with only mild drift.",
        "Let time-of-day effects be secondary to language variation.",
    ],
    time_profile={"burstiness": 0.28, "silence_ratio": 0.24, "revisit_delay": 0.22, "day_cycle": 0.32, "notification_window": 0.24},
    mutation_notes=[
        "This family is the cleanest language-only baseline for detector generalization tests.",
    ],
)

FAMILY_C_GRAPH_AWARE = FamilySpec(
    family="Family C: Graph-Aware",
    slug="family_c_graph_aware",
    description="Graph-first family that reacts to thread position, notifications, revisits, and local interaction structure.",
    axes={"language": 0.45, "behavior": 0.70, "graph": 0.95, "emergence": 0.45},
    dimension_axes={"language": 0.45, "behavior": 0.70, "graph": 0.95, "time": 0.65},
    prompt_notes=[
        "Keep text natural but let network context matter more than style polish.",
        "Commenting should depend on thread structure, not just topic match.",
    ],
    behavior_notes=[
        "Increase variation in revisit, reply, and post likelihood.",
        "Support occasional second-level replies and follow-up after notifications.",
    ],
    graph_notes=[
        "Bias toward thread-aware browsing and local interaction memory.",
        "Preserve weak ties but allow more graph-shaped behavior than Family A.",
    ],
    time_notes=[
        "Use time rhythm to support revisit and notification-driven replies.",
        "Allow response delay and burstiness to reflect graph activity.",
    ],
    time_profile={"burstiness": 0.42, "silence_ratio": 0.31, "revisit_delay": 0.48, "day_cycle": 0.35, "notification_window": 0.52},
    mutation_notes=[
        "This family is the structural shift family for graph-detector robustness tests.",
    ],
)

FAMILY_E_EMERGENT = FamilySpec(
    family="Family E: Emergent",
    slug="family_e_emergent",
    description="Feedback-sensitive family whose behavior drifts across rounds and reacts to prior detector pressure.",
    axes={"language": 0.65, "behavior": 0.75, "graph": 0.70, "emergence": 0.95},
    dimension_axes={"language": 0.65, "behavior": 0.75, "graph": 0.70, "time": 0.85},
    prompt_notes=[
        "Allow more self-consistent drift across rounds instead of a fixed template.",
        "Blend language, behavior, and graph changes together.",
    ],
    behavior_notes=[
        "Use feedback-sensitive variation across follow-up generations.",
        "Allow the population to diversify as detector pressure increases.",
    ],
    graph_notes=[
        "Let interaction structure mutate alongside text and topic preferences.",
        "Use round-to-round changes to emulate an evolving population.",
    ],
    time_notes=[
        "Make rhythm a first-class mutation axis instead of a fixed cadence.",
        "Let burstiness and silence windows drift across generations.",
    ],
    time_profile={"burstiness": 0.52, "silence_ratio": 0.38, "revisit_delay": 0.45, "day_cycle": 0.44, "notification_window": 0.56},
    mutation_notes=[
        "This family is the evolutionary baseline for held-out family evaluation.",
    ],
)

FAMILY_REGISTRY: dict[str, FamilySpec] = {
    FAMILY_A_LANGUAGE_PERSONA.family: FAMILY_A_LANGUAGE_PERSONA,
    FAMILY_C_GRAPH_AWARE.family: FAMILY_C_GRAPH_AWARE,
    FAMILY_E_EMERGENT.family: FAMILY_E_EMERGENT,
}

FAMILY_ALIAS_MAP: dict[str, str] = {
    _norm_key("A"): FAMILY_A_LANGUAGE_PERSONA.family,
    _norm_key("Family A"): FAMILY_A_LANGUAGE_PERSONA.family,
    _norm_key("Family A Language Persona"): FAMILY_A_LANGUAGE_PERSONA.family,
    _norm_key("Language-Persona"): FAMILY_A_LANGUAGE_PERSONA.family,
    _norm_key("Language Persona"): FAMILY_A_LANGUAGE_PERSONA.family,
    _norm_key("C"): FAMILY_C_GRAPH_AWARE.family,
    _norm_key("Family C"): FAMILY_C_GRAPH_AWARE.family,
    _norm_key("Family C Graph Aware"): FAMILY_C_GRAPH_AWARE.family,
    _norm_key("Graph-Aware"): FAMILY_C_GRAPH_AWARE.family,
    _norm_key("Graph Aware"): FAMILY_C_GRAPH_AWARE.family,
    _norm_key("E"): FAMILY_E_EMERGENT.family,
    _norm_key("Family E"): FAMILY_E_EMERGENT.family,
    _norm_key("Family E Emergent"): FAMILY_E_EMERGENT.family,
    _norm_key("Emergent"): FAMILY_E_EMERGENT.family,
}

FAMILY_ORDER: tuple[str, ...] = (
    FAMILY_A_LANGUAGE_PERSONA.family,
    FAMILY_C_GRAPH_AWARE.family,
    FAMILY_E_EMERGENT.family,
)


def normalize_family_name(family: str) -> str:
    key = _norm_key(family)
    if key in FAMILY_ALIAS_MAP:
        return FAMILY_ALIAS_MAP[key]
    for canonical in FAMILY_ORDER:
        if key == _norm_key(canonical):
            return canonical
    raise KeyError(f"Unknown family: {family}")


def get_family_spec(family: str) -> FamilySpec:
    canonical = normalize_family_name(family)
    return FAMILY_REGISTRY[canonical]


def list_family_specs() -> list[FamilySpec]:
    return [FAMILY_REGISTRY[name] for name in FAMILY_ORDER]


def family_dimension_axes(family: str) -> dict[str, float]:
    spec = get_family_spec(family)
    axes = dict(spec.dimension_axes or spec.axes)
    return axes


def family_time_profile(family: str) -> dict[str, float]:
    spec = get_family_spec(family)
    return dict(spec.time_profile)


def export_family_registry() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "families": [spec.to_dict() for spec in list_family_specs()],
        "aliases": dict(sorted(FAMILY_ALIAS_MAP.items())),
    }
