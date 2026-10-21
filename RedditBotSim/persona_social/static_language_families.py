from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import json
import random
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .models import Persona
from .trait_prompting import render_static_family_guidance, sample_trait_profile, stable_seed, trait_profile_to_columns


STATIC_LANGUAGE_FAMILIES = {
    "F1": "F1_original_architecture",
    "F2": "F2_tfidf_informed_language",
    "F3": "F3_human_anchor_local_tfidf",
    "F4": "F4_peba_trait_prompt",
    "F5": "F5_gemini_trait_prompt",
}

SCHEMA_VERSION = "static-language-families-v1"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"
FAMILY_IDS = tuple(STATIC_LANGUAGE_FAMILIES)

HIGH_DIM_TRAIT_FIELDS = (
    "impulse_control",
    "attention_style",
    "emotional_reactivity",
    "certainty_style",
    "conflict_style",
    "lived_experience_lens",
    "sentence_shape",
    "reply_texture",
    "debate_tendency",
    "post_tendency",
    "reply_tendency",
    "revisit_tendency",
    "exploration_tendency",
    "brevity",
    "graph_affinity",
    "notification_reactivity",
    "formality",
    "hedging",
    "sarcasm",
    "detail_orientation",
    "evidence_demand",
    "stance_strength",
    "question_rate",
    "thread_memory",
)

FAMILY_DISPLAY_NAMES = {
    "F1_original_architecture": "Family 1: Original Architecture",
    "F2_tfidf_informed_language": "Family 2: TF-IDF-Informed Language",
    "F3_human_anchor_local_tfidf": "Family 3: Human-Anchor Local TF-IDF",
    "F4_peba_trait_prompt": "Family 4: PEBA Trait Prompt",
    "F5_gemini_trait_prompt": "Family 5: Gemini Trait Prompt",
}

TOPICS = ["US politics", "world news", "Russia-Ukraine war", "Israel-Palestine"]
SUBREDDIT_POOLS = [
    ["politics", "news", "worldnews"],
    ["worldnews", "InternationalNews", "GlobalTalk"],
    ["news", "worldnews", "UpliftingNews"],
    ["politics", "worldnews", "GlobalTalk"],
    ["InternationalNews", "news", "worldnews"],
    ["politics", "news", "InternationalNews"],
]
REGIONS = [
    "American, Ohio",
    "American, Pennsylvania",
    "American, Arizona",
    "American, Michigan",
    "American, Oregon",
    "Canadian, Toronto",
    "British, Manchester",
    "American, Illinois",
    "American, Florida",
    "American, California",
    "Australian, Sydney",
    "American, Wisconsin",
    "American, Georgia",
    "American, Massachusetts",
    "American, Texas",
    "American, Nevada",
    "British, Leeds",
    "American, Colorado",
    "Irish, Dublin",
    "American, Washington",
    "Canadian, Vancouver",
    "British, Bristol",
    "American, Minnesota",
    "American, North Carolina",
]
EDUCATION = ["High School Diploma", "Some college", "Associate Degree", "Bachelor's Degree", "Master's Degree", "College student"]
GENDER = ["Female", "Male", "Non-binary"]
IDEOLOGY = ["Liberal", "Moderate", "Conservative", "Independent"]
USERNAME_LEFT = [
    "Sidewalk",
    "Diner",
    "BusStop",
    "Lake",
    "PaperBag",
    "Rainy",
    "Cheap",
    "Tiny",
    "Checkout",
    "HalfCharged",
    "LateBus",
    "Porch",
    "Spare",
    "Notebook",
    "GroupChat",
    "Parking",
    "Kitchen",
    "SideQuest",
    "TeaBreak",
    "Faded",
    "Northbound",
    "Corner",
    "Mug",
    "NightShift",
    "Plain",
]
USERNAME_RIGHT = [
    "Sam",
    "Coffee",
    "Mia",
    "Guy",
    "Lunch",
    "Window",
    "Headphones",
    "Fan",
    "Lane",
    "Phone",
    "Rider",
    "Light",
    "Change",
    "Margin",
    "Thread",
    "Meter",
    "Radio",
    "Dad",
    "Nora",
    "Sticker",
    "Commuter",
    "Receipt",
    "Thermos",
    "Clock",
    "Reader",
]


def normalize_family_id(family: str) -> str:
    key = str(family).strip()
    return STATIC_LANGUAGE_FAMILIES.get(key.upper(), key)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.01, float(value)) for value in weights.values())
    return {key: round(max(0.01, float(value)) / total, 4) for key, value in weights.items()}


def _sample_interest_weights(rng: random.Random, family_id: str) -> dict[str, float]:
    weights = {topic: rng.uniform(0.12, 0.42) for topic in TOPICS}
    if family_id == "F1_original_architecture":
        weights["US politics"] += 0.12
        weights["world news"] += 0.08
    elif family_id == "F2_tfidf_informed_language":
        weights[rng.choice(TOPICS)] += 0.16
    elif family_id == "F3_human_anchor_local_tfidf":
        weights[rng.choice(TOPICS)] += 0.1
    elif family_id in {"F4_peba_trait_prompt", "F5_gemini_trait_prompt"}:
        weights[rng.choice(TOPICS)] += rng.uniform(0.08, 0.22)
    return _normalize_weights(weights)


def _unique_username(family_id: str, index: int, rng: random.Random) -> str:
    left = USERNAME_LEFT[(index + rng.randrange(len(USERNAME_LEFT))) % len(USERNAME_LEFT)]
    right = USERNAME_RIGHT[(index * 3 + rng.randrange(len(USERNAME_RIGHT))) % len(USERNAME_RIGHT)]
    suffix = family_id.split("_", 1)[0].lower()
    return f"{left}{right}_{suffix}_{index:03d}"


def _model_profile(family_id: str, index: int, rng: random.Random, *, openai_model: str, gemini_model: str) -> dict[str, Any]:
    if family_id == "F5_gemini_trait_prompt":
        return {
            "llm_provider": "gemini",
            "llm_model": gemini_model,
            "llm_temperature": rng.choice([0.4, 0.8]),
            "llm_reasoning_effort": "medium",
        }
    return {
        "llm_provider": "openai",
        "llm_model": openai_model,
        "llm_temperature": 0.55 if family_id == "F1_original_architecture" else rng.choice([0.4, 0.7, 0.9]),
        "llm_reasoning_effort": "medium",
    }


def build_static_language_personas(
    family: str,
    count: int,
    *,
    seed: int = 0,
    anchor_blocks: Mapping[str, str] | None = None,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> tuple[list[Persona], pd.DataFrame]:
    """Build unique personas for a static language family.

    This intentionally does not cycle a static Excel/persona sheet. Every bot is
    sampled from a parameter space and gets a unique id, username, trait profile,
    demographic fields, behavior tendencies, and model configuration.
    """
    family_id = normalize_family_id(family)
    anchor_blocks = dict(anchor_blocks or {})
    personas: list[Persona] = []
    manifest_rows: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        rng = random.Random(stable_seed("static-language", family_id, seed, index))
        user_id = f"{family_id.lower()}_{index:03d}"
        trait = sample_trait_profile(stable_seed("trait", family_id, seed, index), family_id=family_id)
        desc_add, style, stance = render_static_family_guidance(family_id, trait, anchor_blocks.get(user_id))
        if family_id == "F3_human_anchor_local_tfidf":
            anchor_note = anchor_blocks.get(user_id, "").strip()
            desc_add = f"{desc_add} Anchor profile reserve description: {anchor_note or 'reserved for source profile description.'}"
            style = f"{style}; anchor profile reserve style: {anchor_note or 'reserved for source style cues.'}"
            stance = f"{stance}; anchor profile reserve stance: {anchor_note or 'reserved for source stance cues.'}"
        model_cfg = _model_profile(family_id, index, rng, openai_model=openai_model, gemini_model=gemini_model)
        age = rng.randint(19, 64)
        row = {
            "user_id": user_id,
            "user_name": _unique_username(family_id, index, rng),
            "age": age,
            "gender": rng.choice(GENDER),
            "region": rng.choice(REGIONS),
            "education": rng.choice(EDUCATION),
            "ideology": rng.choice(IDEOLOGY),
            "description": desc_add,
            "subreddits": rng.choice(SUBREDDIT_POOLS),
            "interest_weights": _sample_interest_weights(rng, family_id),
            "style": style,
            "stance": stance,
            "debate_tendency": round(rng.uniform(0.22, 0.82), 3),
            "post_tendency": round(rng.uniform(0.08, 0.28), 3),
            "reply_tendency": round(rng.uniform(0.34, 0.68), 3),
            "revisit_tendency": round(rng.uniform(0.08, 0.32), 3),
            "exploration_tendency": round(rng.uniform(0.22, 0.62), 3),
            "brevity": round(rng.uniform(0.42, 0.9), 3),
            "family": family_id,
            "family_round": 0,
            "family_generation": 0,
            "graph_affinity": round(rng.uniform(0.25, 0.68), 3),
            "notification_reactivity": round(rng.uniform(0.18, 0.54), 3),
            **model_cfg,
        }
        if family_id == "F1_original_architecture":
            row["brevity"] = round(rng.uniform(0.28, 0.58), 3)
            row["debate_tendency"] = round(rng.uniform(0.48, 0.82), 3)
        elif family_id == "F2_tfidf_informed_language":
            row["brevity"] = round(rng.uniform(0.66, 0.94), 3)
            row["post_tendency"] = round(rng.uniform(0.06, 0.2), 3)
        elif family_id == "F3_human_anchor_local_tfidf":
            row["brevity"] = round(rng.uniform(0.58, 0.9), 3)
        elif family_id in {"F4_peba_trait_prompt", "F5_gemini_trait_prompt"}:
            row["debate_tendency"] = round(rng.uniform(0.18, 0.9), 3)
            row["exploration_tendency"] = round(rng.uniform(0.25, 0.72), 3)
        personas.append(Persona(**row))
        manifest_rows.append(
            {
                **{k: v for k, v in row.items() if k not in {"subreddits", "interest_weights"}},
                "family_display": FAMILY_DISPLAY_NAMES.get(family_id, family_id),
                "SubRedditList": json.dumps(row["subreddits"], ensure_ascii=False),
                "interest_weights": json.dumps(row["interest_weights"], ensure_ascii=False, sort_keys=True),
                "trait_profile_json": trait.to_json(),
                **{f"trait_{k}": v for k, v in trait_profile_to_columns(trait).items()},
                "high_dim_traits_json": json.dumps(_high_dim_traits(row, trait), ensure_ascii=False, sort_keys=True),
                "anchor_profile_attached": bool(anchor_blocks.get(user_id)),
            }
        )
    return personas, pd.DataFrame(manifest_rows)


def _scale(value: Any, center: float = 0.5, spread: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = center
    if spread == 0:
        return 0.0
    return round(max(-1.0, min(1.0, (numeric - center) / spread)), 3)


def _high_dim_traits(row: Mapping[str, Any], trait: Any | None = None) -> dict[str, Any]:
    trait_fields = trait_profile_to_columns(trait) if trait is not None else {}
    debate = float(row.get("debate_tendency", 0.5))
    reply = float(row.get("reply_tendency", 0.5))
    revisit = float(row.get("revisit_tendency", 0.2))
    exploration = float(row.get("exploration_tendency", 0.4))
    brevity = float(row.get("brevity", 0.65))
    family_id = str(row.get("family", ""))
    family_bias = {
        "F1_original_architecture": {"formality": 0.55, "detail_orientation": 0.2, "hedging": -0.05},
        "F2_tfidf_informed_language": {"formality": -0.25, "detail_orientation": 0.55, "hedging": 0.12},
        "F3_human_anchor_local_tfidf": {"formality": -0.2, "detail_orientation": 0.42, "hedging": 0.18},
        "F4_peba_trait_prompt": {"formality": -0.45, "detail_orientation": 0.18, "hedging": 0.08},
        "F5_gemini_trait_prompt": {"formality": -0.18, "detail_orientation": 0.32, "hedging": 0.14},
    }.get(family_id, {"formality": 0.0, "detail_orientation": 0.0, "hedging": 0.0})
    numeric_traits = {
        "debate_tendency": round(debate, 3),
        "post_tendency": round(float(row.get("post_tendency", 0.15)), 3),
        "reply_tendency": round(reply, 3),
        "revisit_tendency": round(revisit, 3),
        "exploration_tendency": round(exploration, 3),
        "brevity": round(brevity, 3),
        "graph_affinity": round(float(row.get("graph_affinity", 0.5)), 3),
        "notification_reactivity": round(float(row.get("notification_reactivity", 0.35)), 3),
        "formality": round(family_bias["formality"] - 0.25 * brevity, 3),
        "hedging": round(family_bias["hedging"] + 0.35 * (0.55 - debate), 3),
        "sarcasm": round(0.22 * debate - 0.08 * float(row.get("post_tendency", 0.15)), 3),
        "detail_orientation": round(family_bias["detail_orientation"] + 0.25 * exploration, 3),
        "evidence_demand": round(0.35 * debate + family_bias["detail_orientation"], 3),
        "stance_strength": round(0.85 * _scale(debate, center=0.45, spread=0.45), 3),
        "question_rate": round(0.22 + 0.28 * exploration + 0.16 * (0.6 - debate), 3),
        "thread_memory": round(0.18 + 0.55 * revisit + 0.18 * reply, 3),
    }
    return {**trait_fields, **numeric_traits}


def _persona_uid(persona: Mapping[str, Any]) -> str:
    payload = {
        "user_id": persona.get("user_id"),
        "family": persona.get("family"),
        "description": persona.get("description"),
        "style": persona.get("style"),
        "stance": persona.get("stance"),
        "traits": persona.get("traits"),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _persona_dict(persona: Persona, manifest_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = asdict(persona)
    row = dict(manifest_row or {})
    if row.get("high_dim_traits_json"):
        payload["traits"] = json.loads(str(row["high_dim_traits_json"]))
    else:
        payload["traits"] = _high_dim_traits(payload)
    payload["trait_schema_version"] = SCHEMA_VERSION
    payload["family_id"] = next((short_id for short_id, full_id in STATIC_LANGUAGE_FAMILIES.items() if full_id == payload["family"]), payload["family"])
    payload["family_display"] = FAMILY_DISPLAY_NAMES.get(payload["family"], payload["family"])
    payload["persona_uid"] = _persona_uid(payload)
    return payload


def build_static_language_manifest(
    family: str,
    count: int,
    *,
    seed: int = 0,
    anchor_blocks: Mapping[str, str] | None = None,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> dict[str, Any]:
    personas, manifest_rows = build_static_language_personas(
        family,
        count,
        seed=seed,
        anchor_blocks=anchor_blocks,
        openai_model=openai_model,
        gemini_model=gemini_model,
    )
    records = [
        _persona_dict(persona, row)
        for persona, row in zip(personas, manifest_rows.to_dict(orient="records"), strict=True)
    ]
    persona_uids = [record["persona_uid"] for record in records]
    if len(persona_uids) != len(set(persona_uids)):
        raise RuntimeError(f"duplicate persona manifest uid detected for {family}")
    family_id = normalize_family_id(family)
    first_model = records[0] if records else _model_profile(
        family_id,
        0,
        random.Random(stable_seed("empty-model", family_id, seed)),
        openai_model=openai_model,
        gemini_model=gemini_model,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "family_id": next((short_id for short_id, full_id in STATIC_LANGUAGE_FAMILIES.items() if full_id == family_id), family_id),
        "family": family_id,
        "family_display": FAMILY_DISPLAY_NAMES.get(family_id, family_id),
        "generation": 0,
        "count": len(records),
        "seed": seed,
        "llm_provider": first_model["llm_provider"],
        "llm_model": first_model["llm_model"],
        "llm_temperature": first_model["llm_temperature"],
        "llm_reasoning_effort": first_model["llm_reasoning_effort"],
        "trait_fields": list(HIGH_DIM_TRAIT_FIELDS),
        "persona_uids": persona_uids,
        "personas": records,
    }


def build_all_static_language_manifests(
    count_per_family: int,
    *,
    seed: int = 0,
    anchor_blocks: Mapping[str, str] | None = None,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> dict[str, Any]:
    manifests = [
        build_static_language_manifest(
            family,
            count_per_family,
            seed=seed,
            anchor_blocks=anchor_blocks,
            openai_model=openai_model,
            gemini_model=gemini_model,
        )
        for family in FAMILY_IDS
    ]
    persona_uids = [uid for manifest in manifests for uid in manifest["persona_uids"]]
    if len(persona_uids) != len(set(persona_uids)):
        raise RuntimeError("duplicate persona uid detected across static language families")
    return {
        "schema_version": SCHEMA_VERSION,
        "count_per_family": count_per_family,
        "total_personas": len(persona_uids),
        "families": manifests,
    }


def export_persona_manifest_json(personas: list[Persona], path: str | Path, *, family: str, manifest_rows: pd.DataFrame | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_records = manifest_rows.to_dict(orient="records") if manifest_rows is not None else []
    payload = {
        "family": normalize_family_id(family),
        "generation": 0,
        "personas": [
            _persona_dict(persona, manifest_records[index] if index < len(manifest_records) else None)
            for index, persona in enumerate(personas)
        ],
    }
    if manifest_rows is not None:
        payload["manifest_rows"] = manifest_records
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_persona_manifest_csv(manifest_rows: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows.to_csv(output, index=False)
