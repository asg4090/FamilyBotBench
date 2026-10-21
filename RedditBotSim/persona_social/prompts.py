from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import json
from collections.abc import Mapping
from typing import Any

from .models import Candidate, Persona


def _persona_value(persona: Any, key: str, default: Any = None) -> Any:
    if isinstance(persona, Mapping):
        return persona.get(key, default)
    return getattr(persona, key, default)


def _persona_float(persona: Any, key: str, default: float) -> float:
    value = _persona_value(persona, key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _persona_int(persona: Any, key: str, default: int) -> int:
    value = _persona_value(persona, key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _persona_traits(persona: Any) -> dict[str, Any]:
    raw = _persona_value(persona, "traits", {}) or {}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        raw = parsed
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value for key, value in raw.items()}


def _render_trait_lines(traits: dict[str, Any], *, max_items: int = 24) -> str:
    preferred = [
        "impulse_control",
        "attention_style",
        "emotional_reactivity",
        "certainty_style",
        "conflict_style",
        "lived_experience_lens",
        "sentence_shape",
        "reply_texture",
        "formality",
        "hedging",
        "sarcasm",
        "detail_orientation",
        "evidence_demand",
        "stance_strength",
        "question_rate",
        "thread_memory",
        "debate_tendency",
        "post_tendency",
        "reply_tendency",
        "revisit_tendency",
        "exploration_tendency",
        "brevity",
        "graph_affinity",
        "notification_reactivity",
    ]
    keys = [key for key in preferred if key in traits]
    keys.extend(sorted(set(traits) - set(keys)))
    lines = []
    for key in keys[:max_items]:
        value = traits[key]
        if isinstance(value, float):
            rendered = f"{value:.3f}"
        else:
            rendered = str(value)
        lines.append(f"- {key}: {rendered}")
    return "\n".join(lines)


def _render_jsonish(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value)
    if not text.strip():
        return ""
    try:
        parsed = json.loads(text)
    except Exception:
        return text
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True)


def persona_block(persona: Persona) -> str:
    interest_weights = _persona_value(persona, "interest_weights", {}) or {}
    if not isinstance(interest_weights, Mapping):
        interest_weights = {}
    formatted_interests = []
    for k, v in interest_weights.items():
        try:
            formatted_interests.append(f"{k}: {float(v):.2f}")
        except (TypeError, ValueError):
            continue
    interests = ", ".join(formatted_interests)
    return f"""Identity:
- username: {_persona_value(persona, "user_name", "")}
- age/gender/region: {_persona_value(persona, "age", "")}, {_persona_value(persona, "gender", "")}, {_persona_value(persona, "region", "")}
- education: {_persona_value(persona, "education", "")}
- ideology: {_persona_value(persona, "ideology", "")}
- description: {_persona_value(persona, "description", "")}
- interests: {interests}
- style: {_persona_value(persona, "style", "")}
- stance: {_persona_value(persona, "stance", "")}
- debate tendency: {_persona_float(persona, "debate_tendency", 0.0):.2f}
- post tendency: {_persona_float(persona, "post_tendency", 0.0):.2f}
- reply tendency: {_persona_float(persona, "reply_tendency", 0.0):.2f}
- revisit tendency: {_persona_float(persona, "revisit_tendency", 0.0):.2f}
- graph affinity: {_persona_float(persona, "graph_affinity", 0.5):.2f}
- notification reactivity: {_persona_float(persona, "notification_reactivity", 0.5):.2f}
- brevity: {_persona_float(persona, "brevity", 0.0):.2f}
"""


def family_context_block(persona: Persona) -> str:
    family = str(_persona_value(persona, "family", "baseline") or "baseline")
    generation = _persona_int(persona, "family_generation", 0)
    round_id = _persona_int(persona, "family_round", 0)
    traits = _persona_traits(persona)
    rendered_traits = _render_trait_lines(traits)
    drift_brief = str(traits.get("family_drift_brief", "") or "").strip()
    family_axes = _render_jsonish(traits.get("family_axes", ""))
    temporal_profile = _render_jsonish(traits.get("temporal_profile", ""))
    target_axes = str(traits.get("target_axes", "") or "").strip()
    if family == "baseline" and generation == 0 and round_id == 0 and not rendered_traits:
        return ""
    extra_lines = []
    if drift_brief:
        extra_lines.append(f"- drift brief: {drift_brief}")
    if target_axes:
        extra_lines.append(f"- target axes: {target_axes}")
    if family_axes:
        extra_lines.append(f"- family axis intensity: {family_axes}")
    if temporal_profile:
        extra_lines.append(f"- temporal profile: {temporal_profile}")
    extra_block = "\n".join(extra_lines)
    if extra_block:
        extra_block = "\n" + extra_block
    trait_block = ""
    if rendered_traits:
        trait_block = f"""
Active persona trait card:
{rendered_traits}

Use these traits as active behavioral controls, not metadata:
- let attention_style and detail_orientation decide what part of the thread you notice;
- let impulse_control, brevity, sentence_shape, and reply_texture shape the wording;
- let conflict_style, certainty_style, hedging, sarcasm, and question_rate shape the tone;
- let thread_memory, graph_affinity, and notification_reactivity shape whether you reply, skip, revisit, or post.
- do not turn these traits into rigid rules; they should influence natural choices."""
    return f"""Family context:
- family: {family}
- generation: {generation}
- round: {round_id}
- graph affinity: {_persona_float(persona, "graph_affinity", 0.5):.2f}
- notification reactivity: {_persona_float(persona, "notification_reactivity", 0.5):.2f}
{extra_block}
{trait_block}
"""


STYLE_RULES = """Writing rules:
- Write like a normal Reddit user, not an assistant.
- Be concrete and opinionated when it fits. It is okay to be skeptical, blunt, casual, or mildly messy.
- Do not imitate another user's exact style.
- Do not write a balanced essay or a neutral summary.
- Do not over-explain your identity. Keep the reply natural and in-character.
- If the action is a post, it should feel like an actual person starting a thread, not a prompt completion.
- If the action is a reply, it should feel responsive to the specific target and to the surrounding thread.
- Before finalizing, reread the sentence once for fluency: the final content must be grammatical, readable, and not broken by awkward deletions.
- Avoid abstract AI-ish words such as implications, accountability, landscape, ongoing, ramifications, critical, strategy, dialogue, multifaceted, underscores, highlights.
- Do not mention bots, simulation, AI, prompts, models, or generation.
- Avoid square brackets in the content.
"""


def social_context_block(persona: Persona, memory: Any, browse_items: list[dict[str, Any]]) -> str:
    own_posts = list(getattr(memory, "own_posts", []) or [])[-8:]
    own_c1 = list(getattr(memory, "own_comment1", []) or [])[-8:]
    own_c2 = list(getattr(memory, "own_comment2", []) or [])[-8:]
    notifications = []
    for note in list(getattr(memory, "notifications", []) or [])[-6:]:
        note_id = str(note.get("notification_id", ""))
        if note_id and note_id in set(getattr(memory, "revisited_notifications", set()) or set()):
            continue
        notifications.append(note)
    recent_topics = list(getattr(memory, "recent_topics", []) or [])[-8:]
    recent_actions = list(getattr(memory, "recent_actions", []) or [])[-8:]
    recent_sources = list(getattr(memory, "recent_candidate_sources", []) or [])[-8:]
    return f"""Social context:
- own posts: {own_posts}
- own comment1 ids: {own_c1}
- own comment2 ids: {own_c2}
- pending notifications: {json.dumps(notifications, ensure_ascii=False)}
- recent topics: {recent_topics}
- recent actions: {recent_actions}
- recent candidate sources: {recent_sources}
"""


def build_action_prompt(
    persona: Persona,
    action: str,
    candidate: Candidate | None,
    planned_time: str,
    browse_items: list[dict[str, Any]],
    memory: Any | None = None,
) -> str:
    if action == "browse_decide":
        action_instruction = """You are not being assigned a target. You are browsing these thread snapshots and must decide what a real user like you would do.

Choose one action:
- skip: say nothing if none of the threads pull you in.
- comment1: write a top-level comment on one post.
- comment2: reply to one visible first-level comment.
- revisit: respond to a visible first-level comment in a thread connected to your own post/comment or notification.
- post: start your own short Reddit post after browsing.

Rules for target selection:
- For comment1, set target_post_id to the post you chose and leave target_comment_id empty.
- For comment2 or revisit, set target_post_id and target_comment_id to a visible first-level comment_id from that thread.
- For post, leave both target ids empty.
- For skip, content must be empty.
- Do not pretend you saw context that is not in the thread snapshots.
- If you reply to a comment, respond to that specific comment and the surrounding thread, not just the headline."""
        schema = {
            "action": "skip | comment1 | comment2 | revisit | post",
            "target_post_id": "post id or empty",
            "target_comment_id": "first-level comment id or empty",
            "reason": "brief private reason for the choice",
            "content": "post/comment text or empty for skip",
        }
    elif action == "post":
        action_instruction = """You decided to make your own Reddit post after browsing. Write one natural Reddit post body as a short text. It should sound like a real person starting a thread, not a task answer or an essay. If you mention your own experience or opinion, keep it grounded and specific."""
        schema = {"content": "post text"}
    elif action in {"comment1", "comment2", "revisit"}:
        target = candidate.to_prompt_item(0) if candidate else {}
        action_label = "reply to this first-level comment" if action in {"comment2", "revisit"} else "comment on this post"
        action_instruction = f"""You decided to {action_label}. React as yourself. Keep it natural, specific, and slightly conversational. Short is fine, but do not sound canned.

Target:
{json.dumps(target, ensure_ascii=False, indent=2)}
"""
        schema = {"content": "comment text"}
    else:
        action_instruction = "You decided not to comment. Return an empty content string."
        schema = {"content": ""}

    family_block = family_context_block(persona)
    social_block = social_context_block(persona, memory, browse_items) if memory is not None else ""
    return f"""{persona_block(persona)}
{family_block}
{social_block}

You are browsing Reddit in an offline simulation at {planned_time}. You saw these candidate items:
{json.dumps(browse_items, ensure_ascii=False, indent=2)}

Decision:
- action: {action}

{action_instruction}

{STYLE_RULES}

Return valid JSON only, exactly in this schema:
{json.dumps(schema, ensure_ascii=False)}
"""
