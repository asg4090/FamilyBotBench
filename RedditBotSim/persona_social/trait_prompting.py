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
from dataclasses import asdict, dataclass
from typing import Any


IMPULSE = ["impulsive", "quick-reacting", "measured", "careful", "hesitant"]
ATTENTION = ["headline-reactive", "detail-oriented", "number-focused", "timeline-focused", "tone-sensitive"]
EMOTION = ["annoyed", "weary", "dry", "curious", "skeptical", "sympathetic", "blunt"]
CERTAINTY = ["confident", "half-sure", "questioning", "hedged", "contrarian"]
CONFLICT = ["argumentative", "conflict-avoidant", "asks follow-up questions", "drops one-liners", "pushes back politely"]
LENS = [
    "shift worker",
    "college student",
    "parent",
    "local politics watcher",
    "veteran-adjacent family member",
    "immigrant family perspective",
    "teacher",
    "small business employee",
    "healthcare worker",
    "public transit commuter",
]


@dataclass(frozen=True)
class TraitProfile:
    impulse_control: str
    attention_style: str
    emotional_reactivity: str
    certainty_style: str
    conflict_style: str
    lived_experience_lens: str
    sentence_shape: str
    reply_texture: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def stable_seed(*parts: object) -> int:
    payload = "|".join(json.dumps(part, ensure_ascii=False, sort_keys=True, default=str) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def sample_trait_profile(seed: int | random.Random, *, family_id: str = "") -> TraitProfile:
    rng = seed if isinstance(seed, random.Random) else random.Random(seed)
    if family_id == "F4_peba_trait_prompt" or family_id == "F5_gemini_trait_prompt":
        impulse = rng.choice(IMPULSE)
        attention = rng.choice(ATTENTION)
        emotion = rng.choice(EMOTION)
        certainty = rng.choice(CERTAINTY)
        conflict = rng.choice(CONFLICT)
    elif family_id == "F2_tfidf_informed_language":
        impulse = rng.choice(["quick-reacting", "measured", "half-sure"])
        attention = rng.choice(["detail-oriented", "headline-reactive", "tone-sensitive"])
        emotion = rng.choice(["dry", "skeptical", "annoyed", "curious"])
        certainty = rng.choice(["questioning", "half-sure", "hedged"])
        conflict = rng.choice(["asks follow-up questions", "drops one-liners", "pushes back politely"])
    elif family_id == "F3_human_anchor_local_tfidf":
        impulse = rng.choice(["quick-reacting", "measured", "hesitant"])
        attention = rng.choice(["headline-reactive", "detail-oriented", "tone-sensitive"])
        emotion = rng.choice(["dry", "weary", "skeptical", "curious"])
        certainty = rng.choice(["half-sure", "questioning", "hedged"])
        conflict = rng.choice(["asks follow-up questions", "conflict-avoidant", "pushes back politely"])
    else:
        impulse = rng.choice(["measured", "careful", "confident"])
        attention = rng.choice(["abstract-policy minded", "timeline-focused", "headline-reactive"])
        emotion = rng.choice(["serious", "skeptical", "concerned"])
        certainty = rng.choice(["confident", "contrarian", "hedged"])
        conflict = rng.choice(["argumentative", "pushes back politely", "asks follow-up questions"])
    return TraitProfile(
        impulse_control=impulse,
        attention_style=attention,
        emotional_reactivity=emotion,
        certainty_style=certainty,
        conflict_style=conflict,
        lived_experience_lens=rng.choice(LENS),
        sentence_shape=rng.choice(["short fragments", "mixed short and medium sentences", "one direct point then a caveat", "question-first replies"]),
        reply_texture=rng.choice(["phone-typed", "group-chat", "dry aside", "specific complaint", "small correction"]),
    )


def render_trait_persona_card(profile: TraitProfile) -> str:
    return (
        f"Acts from a {profile.lived_experience_lens} lens; "
        f"{profile.impulse_control}, {profile.attention_style}, {profile.emotional_reactivity}; "
        f"{profile.certainty_style} certainty and {profile.conflict_style}; "
        f"usually writes with {profile.sentence_shape} and a {profile.reply_texture} texture."
    )


def render_static_family_guidance(family_id: str, profile: TraitProfile, anchor_block: str | None = None) -> tuple[str, str, str]:
    """Return description/style/stance additions that fit Persona fields."""
    trait_card = render_trait_persona_card(profile)
    if family_id == "F1_original_architecture":
        return (
            f"{trait_card} Tends to write broader civic or geopolitical comments, like the original BotSim generation style.",
            "formal-leaning, explanatory, abstract, policy-aware",
            "tries to connect the thread to wider political stakes and institutional consequences",
        )
    if family_id == "F2_tfidf_informed_language":
        return (
            f"{trait_card} Keeps comments grounded in one concrete detail rather than broad political framing.",
            "plain, specific, casual, not polished; avoids abstract essay wording",
            "reacts with questions, small complaints, and everyday phrasing instead of broad summaries",
        )
    if family_id == "F3_human_anchor_local_tfidf":
        anchor_note = (anchor_block or "").strip()
        return (
            f"{trait_card} Before commenting, aligns with a few private human anchor language profiles as aggregate statistics only. {anchor_note}",
            "locally anchored, ordinary, uneven, guided by anchor-level word habits without copying",
            "tries to sound like the small anchor set in rhythm and concreteness, not like a general assistant",
        )
    if family_id == "F4_peba_trait_prompt":
        return (
            f"{trait_card} Persona traits indirectly shape the comment; no explicit detector-avoidance rule is used.",
            "trait-driven, natural, sometimes inconsistent, context-reactive",
            "lets temperament and lived-experience lens decide whether to argue, ask, joke, or stay narrow",
        )
    if family_id == "F5_gemini_trait_prompt":
        return (
            f"{trait_card} Same trait system as F4, generated through a Gemini-compatible model.",
            "trait-driven, natural, context-reactive, model-shifted",
            "uses the same persona trait logic as F4 while leaving wording to the Gemini model",
        )
    return trait_card, "plain", "reacts naturally"


def trait_profile_to_columns(profile: TraitProfile) -> dict[str, Any]:
    return asdict(profile)


PROMPT_SCHEMA_VERSION = "trait-prompting-v1"


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ordered_trait_names(traits: dict[str, Any]) -> list[str]:
    preferred = [
        "formality",
        "hedging",
        "sarcasm",
        "detail_orientation",
        "emotional_intensity",
        "argument_depth",
        "brevity_control",
        "lexical_diversity",
        "topic_focus",
        "stance_strength",
        "question_rate",
        "reply_reactivity",
        "thread_memory",
        "risk_aversion",
        "evidence_demand",
        "humor_dryness",
        "politeness",
        "certainty",
        "personal_anecdote_rate",
        "counterargument_rate",
        "signature_jitter",
    ]
    return [name for name in preferred if name in traits] + sorted(set(traits) - set(preferred))


def render_high_dim_trait_block(traits: dict[str, Any]) -> str:
    lines = [f"- {name}: {traits[name]}" for name in _ordered_trait_names(traits)]
    return "\n".join(lines)


def build_trait_prompt(persona: dict[str, Any], *, task: str = "write a Reddit-style comment") -> str:
    """Render a static-language persona dictionary into a compact LLM prompt."""
    traits = dict(persona.get("traits") or persona.get("high_dim_traits") or {})
    identity = {
        "user_id": persona.get("user_id"),
        "user_name": persona.get("user_name"),
        "persona_uid": persona.get("persona_uid"),
        "family_id": persona.get("family_id"),
        "family": persona.get("family"),
        "generation": persona.get("family_generation", persona.get("generation", 0)),
        "provider": persona.get("llm_provider"),
        "model": persona.get("llm_model"),
    }
    lines = [
        f"Prompt schema: {PROMPT_SCHEMA_VERSION}",
        f"Task: {task}",
        f"Identity: {_compact_json(identity)}",
        f"Description: {persona.get('description', '')}",
        f"Style: {persona.get('style', '')}",
        f"Stance: {persona.get('stance', '')}",
        f"Subreddits: {_compact_json(persona.get('subreddits', []))}",
        f"Interest weights: {_compact_json(persona.get('interest_weights', {}))}",
        "High-dimensional traits:",
        render_high_dim_trait_block(traits),
        "Behavior controls:",
        f"- debate_tendency: {persona.get('debate_tendency')}",
        f"- post_tendency: {persona.get('post_tendency')}",
        f"- reply_tendency: {persona.get('reply_tendency')}",
        f"- revisit_tendency: {persona.get('revisit_tendency')}",
        f"- exploration_tendency: {persona.get('exploration_tendency')}",
        f"- brevity: {persona.get('brevity')}",
        "Generation instruction: preserve the persona traits, avoid reusable boilerplate, and keep this bot distinct from other static-family members.",
    ]
    return "\n".join(line for line in lines if line != "")


def persona_prompt_payload(persona: dict[str, Any], *, task: str = "write a Reddit-style comment") -> dict[str, Any]:
    """Return model routing metadata and the rendered trait prompt."""
    return {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "user_id": persona.get("user_id"),
        "persona_uid": persona.get("persona_uid"),
        "family_id": persona.get("family_id"),
        "provider": persona.get("llm_provider"),
        "model": persona.get("llm_model"),
        "temperature": persona.get("llm_temperature"),
        "reasoning_effort": persona.get("llm_reasoning_effort"),
        "prompt": build_trait_prompt(persona, task=task),
    }


def manifest_prompt_payloads(manifest: dict[str, Any], *, task: str = "write a Reddit-style comment") -> list[dict[str, Any]]:
    """Build one prompt payload per persona in either a family or all-family manifest."""
    if "families" in manifest:
        personas = [
            persona
            for family_manifest in manifest.get("families", [])
            for persona in family_manifest.get("personas", [])
        ]
    else:
        personas = list(manifest.get("personas", []))
    return [persona_prompt_payload(persona, task=task) for persona in personas]


def trait_manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """Summarize trait coverage and model routing choices for generated manifests."""
    payloads = manifest_prompt_payloads(manifest)
    trait_names: set[str] = set()
    if "families" in manifest:
        for family_manifest in manifest.get("families", []):
            for persona in family_manifest.get("personas", []):
                trait_names.update((persona.get("traits") or {}).keys())
    else:
        for persona in manifest.get("personas", []):
            trait_names.update((persona.get("traits") or {}).keys())
    return {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "persona_count": len(payloads),
        "providers": sorted({str(payload.get("provider")) for payload in payloads}),
        "models": sorted({str(payload.get("model")) for payload in payloads}),
        "trait_fields": sorted(trait_names),
        "persona_uids": [payload.get("persona_uid") for payload in payloads],
    }
