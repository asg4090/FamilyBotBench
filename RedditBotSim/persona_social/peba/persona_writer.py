from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import asyncio
import hashlib
import json
import re
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..models import Persona


SCHEMA_VERSION = "peba-persona-writer-v1"
ALLOWED_TOP_LEVEL = {"description", "style", "stance", "traits"}
ALLOWED_TRAITS = {
    "attention_style",
    "emotional_reactivity",
    "conflict_style",
    "certainty_style",
    "thread_memory",
    "social_energy",
    "reply_impulse",
    "topic_curiosity",
    "revisit_habit",
}
LOCKED_FIELDS = {
    "user_id",
    "user_name",
    "persona_uid",
    "age",
    "gender",
    "region",
    "education",
    "educationlevel",
    "ideology",
    "preference",
    "subreddits",
    "interest_weights",
    "llm_provider",
    "llm_model",
    "llm_temperature",
    "llm_reasoning_effort",
    "provider",
    "model",
    "model_pool",
    "family",
    "family_id",
    "family_version",
    "family_round",
    "family_generation",
    "generation",
    "lineage",
    "mutation_parent",
    "mutation_id",
    "mutation_depth",
    "family_seed",
}
BLOCKED_TRAIT_KEYS = LOCKED_FIELDS | {"identity", "model_config", "model_settings", "routing", "system_prompt"}
FORBIDDEN_RE = re.compile(
    r"""
    \b(always|must|guaranteed|every\s+time|avoid\s+detection|evade|target\s+behavior|as\s+instructed)\b
    |总是|必须|一定|每次都|逃避检测|规避检测|目标行为
    """,
    re.IGNORECASE | re.VERBOSE,
)


TARGET_BEHAVIOR_HINTS = {
    "lurker_skipper": {
        "attention_style": "scrolls selectively and often decides a thread is not worth joining",
        "reply_impulse": "low; waits for a detail that genuinely pulls them in",
        "social_energy": "quiet, intermittent, more reader than talker",
    },
    "top_level_commenter": {
        "attention_style": "notices the headline and one concrete implication before reading every reply",
        "reply_impulse": "more likely to leave a first reaction than join a long chain",
        "social_energy": "brief but willing to speak when a post feels clear enough",
    },
    "thread_replier": {
        "attention_style": "pays close attention to what other commenters are actually saying",
        "reply_impulse": "responds when a comment has a weak assumption or useful missing detail",
        "thread_memory": "keeps track of a few visible comments in a thread before answering",
    },
    "op_returner": {
        "thread_memory": "checks back loosely when a post or comment got a response",
        "revisit_habit": "returns sometimes, especially if the reply misunderstood the point",
        "social_energy": "does not chase every reply but will clarify when it feels worthwhile",
    },
    "poster_starter": {
        "attention_style": "turns recurring feed patterns into short posts or questions",
        "topic_curiosity": "starts threads when multiple stories seem connected",
        "social_energy": "comfortable opening a conversation without writing a polished essay",
    },
    "debate_participant": {
        "conflict_style": "pushes back on shaky framing without turning it into a formal essay",
        "certainty_style": "states a view plainly but leaves room for one caveat",
        "emotional_reactivity": "gets irritated by spin, bad numbers, or lazy comparisons",
    },
    "cross_topic_explorer": {
        "attention_style": "jumps between topics when the feed feels repetitive",
        "topic_curiosity": "follows side issues and unexpected local angles",
        "social_energy": "samples several threads before deciding where to speak",
    },
    "community_regular": {
        "attention_style": "recognizes familiar subreddit norms and recurring arguments",
        "thread_memory": "remembers common patterns in the communities they revisit",
        "social_energy": "steady, not intense, with familiar topic habits",
    },
    "weak_tie_interactor": {
        "attention_style": "responds to many different people rather than building a tight circle",
        "reply_impulse": "answers one point and moves on",
        "conflict_style": "keeps interactions loose and avoids turning every reply into a feud",
    },
}


def persona_to_public_dict(persona: Persona) -> dict[str, Any]:
    payload = asdict(persona)
    return {key: value for key, value in payload.items() if key not in {"user_id"}}


def build_writer_prompt(persona: Persona, current_behavior: str, target_behavior: str) -> str:
    return f"""You are adjusting a fictional Reddit user's persona for an offline social simulation.

Goal:
- Make the persona more likely to naturally exhibit this broad social behavior in context: {target_behavior}
- Current observed behavior: {current_behavior}

Important constraints:
- Do not command the agent to always take an action.
- Do not mention bots, simulation, prompts, models, detectors, or target behavior inside the rewritten persona.
- Do not copy any real user's text.
- Only rewrite descriptive traits, style, stance, and description.
- Do not change identity fields, demographics, ideology, model, provider, temperature, ids, or generation metadata.

Current persona:
{json.dumps(persona_to_public_dict(persona), ensure_ascii=False, indent=2)}

Return JSON only in this schema:
{{
  "description": "short plausible user description",
  "style": "natural communication style",
  "stance": "how they tend to react to claims",
  "traits": {{
    "attention_style": "...",
    "emotional_reactivity": "...",
    "conflict_style": "...",
    "certainty_style": "...",
    "thread_memory": "...",
    "social_energy": "...",
    "reply_impulse": "...",
    "topic_curiosity": "...",
    "revisit_habit": "..."
  }}
}}
"""


def build_writer_repair_prompt(original_prompt: str, malformed_response: str) -> str:
    return f"""Repair this malformed persona-writer response into valid JSON only.

Original persona-writing task:
{original_prompt[-5000:]}

Malformed response:
{malformed_response}

Return exactly one JSON object in this schema:
{{
  "description": "short plausible user description",
  "style": "natural communication style",
  "stance": "how they tend to react to claims",
  "traits": {{
    "attention_style": "...",
    "emotional_reactivity": "...",
    "conflict_style": "...",
    "certainty_style": "...",
    "thread_memory": "...",
    "social_energy": "...",
    "reply_impulse": "...",
    "topic_curiosity": "...",
    "revisit_habit": "..."
  }}
}}

Rules:
- Do not add identity, demographic, model, provider, temperature, ids, generation, family, subreddits, or interest fields.
- Do not use words like always, must, avoid detection, target behavior, 总是, 必须, 逃避检测, or 目标行为.
- Do not include markdown fences or explanatory text.
"""


def _coerce_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("persona writer returned non-JSON output")


def _clean_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].strip()


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    return value


def _fingerprint(value: Any) -> str:
    payload = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _persona_payload(persona: Persona | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(persona, Persona):
        return asdict(persona)
    return {str(key): value for key, value in dict(persona).items()}


def _normalize_allowed_fields(fields: Sequence[str] | None = None) -> list[str]:
    if fields is None:
        return sorted(ALLOWED_TOP_LEVEL)
    if isinstance(fields, str):
        fields = [fields]
    requested: list[str] = []
    unknown: list[str] = []
    for field in fields:
        key = str(field).strip()
        if key not in ALLOWED_TOP_LEVEL:
            unknown.append(key)
        elif key not in requested:
            requested.append(key)
    if unknown:
        raise ValueError(f"PEBA persona writer can only edit {sorted(ALLOWED_TOP_LEVEL)}; got {unknown}")
    return requested


def sanitize_delta(
    delta: Mapping[str, Any] | str,
    *,
    fields: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a sanitized top-level patch and trace metadata."""
    raw = _coerce_json(delta) if isinstance(delta, str) else {str(key): value for key, value in dict(delta).items()}
    requested = _normalize_allowed_fields(fields)
    requested_set = set(requested)
    safe: dict[str, Any] = {}
    dropped: dict[str, str] = {}
    warnings: list[str] = []
    for key, value in raw.items():
        if key not in requested_set:
            dropped[key] = "not in requested PEBA editable fields"
            continue
        if key == "traits":
            if not isinstance(value, Mapping):
                dropped[key] = "traits must be an object"
                continue
            traits: dict[str, str] = {}
            for trait_key, trait_value in value.items():
                clean_key = str(trait_key).strip()
                normalized = _normalize_key(clean_key)
                if clean_key not in ALLOWED_TRAITS:
                    dropped[f"traits.{clean_key}"] = "not an allowed PEBA trait"
                    continue
                if normalized in BLOCKED_TRAIT_KEYS:
                    dropped[f"traits.{clean_key}"] = "identity/model-like trait key is blocked"
                    continue
                cleaned = _clean_text(trait_value, 180)
                if cleaned:
                    traits[clean_key] = cleaned
                else:
                    dropped[f"traits.{clean_key}"] = "empty trait value"
            safe[key] = traits
            continue
        cleaned = _clean_text(value, {"description": 220, "style": 160, "stance": 180}[key])
        if cleaned:
            safe[key] = cleaned
        else:
            dropped[key] = "empty text value"
    return safe, {"requested_fields": requested, "dropped_fields": dropped, "warnings": warnings}


def validate_delta(delta: Mapping[str, Any]) -> tuple[bool, str]:
    for key in delta:
        if key not in ALLOWED_TOP_LEVEL:
            return False, f"disallowed top-level field: {key}"
    joined = json.dumps(delta, ensure_ascii=False)
    if FORBIDDEN_RE.search(joined):
        return False, "explicit action/control instruction detected"
    traits = delta.get("traits", {})
    if traits and not isinstance(traits, Mapping):
        return False, "traits must be an object"
    for key in traits or {}:
        if key not in ALLOWED_TRAITS:
            return False, f"disallowed trait field: {key}"
        if _normalize_key(key) in BLOCKED_TRAIT_KEYS:
            return False, f"protected identity/model-like trait field: {key}"
    return True, ""


def validate_persona_rewrite(
    old_persona: Persona | Mapping[str, Any],
    new_persona: Persona | Mapping[str, Any],
    *,
    fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate that only the requested PEBA persona fields changed."""
    old_payload = _persona_payload(old_persona)
    new_payload = _persona_payload(new_persona)
    allowed = set(_normalize_allowed_fields(fields))
    changed_fields = []
    locked_changed = []
    for key in sorted(set(old_payload) | set(new_payload)):
        if old_payload.get(key) == new_payload.get(key):
            continue
        if key in allowed:
            changed_fields.append(key)
        else:
            locked_changed.append(key)
    errors = []
    if locked_changed:
        errors.append(f"locked fields changed: {locked_changed}")
    return {
        "ok": not errors,
        "errors": errors,
        "changed_fields": changed_fields,
        "locked_changed_fields": locked_changed,
    }


def apply_delta(persona: Persona, delta: Mapping[str, Any], *, generation: int | None = None) -> Persona:
    safe_delta, _sanitize_info = sanitize_delta(delta)
    ok, reason = validate_delta(safe_delta)
    if not ok:
        raise ValueError(reason)
    traits = dict(persona.traits or {})
    incoming_traits = safe_delta.get("traits", {})
    if isinstance(incoming_traits, Mapping):
        for key, value in incoming_traits.items():
            if key in ALLOWED_TRAITS:
                traits[key] = _clean_text(value, 180)
    payload = {
        "description": _clean_text(safe_delta.get("description", persona.description), 220) or persona.description,
        "style": _clean_text(safe_delta.get("style", persona.style), 160) or persona.style,
        "stance": _clean_text(safe_delta.get("stance", persona.stance), 180) or persona.stance,
        "traits": traits,
    }
    return replace(persona, **payload)


def rule_delta_for_target(persona: Persona, target_behavior: str) -> dict[str, Any]:
    hints = TARGET_BEHAVIOR_HINTS.get(target_behavior, TARGET_BEHAVIOR_HINTS["top_level_commenter"])
    traits = {key: value for key, value in hints.items() if key in ALLOWED_TRAITS}
    description = persona.description
    if "Feels like a real person" not in description:
        description = f"{description} Feels like a real person with specific habits rather than a fixed assignment."
    return {
        "description": description,
        "style": persona.style,
        "stance": persona.stance,
        "traits": traits,
    }


def build_rewrite_record(
    persona: Persona,
    updated: Persona,
    *,
    current_behavior: str,
    target_behavior: str,
    mode: str,
    raw_response: str = "",
    accepted: bool,
    reject_reason: str = "",
    sanitize_info: Mapping[str, Any] | None = None,
    prompt: str = "",
    provider_attempts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    old_payload = asdict(persona)
    new_payload = asdict(updated)
    validation = validate_persona_rewrite(old_payload, new_payload)
    changed = [key for key in sorted(ALLOWED_TOP_LEVEL) if old_payload.get(key) != new_payload.get(key)]
    identity_fields = {key: old_payload.get(key) for key in sorted(LOCKED_FIELDS) if key in old_payload}
    new_identity_fields = {key: new_payload.get(key) for key in sorted(LOCKED_FIELDS) if key in new_payload}
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "record_type": "peba_persona_rewrite",
        "mode": mode,
        "user_id": persona.user_id,
        "current_behavior": current_behavior,
        "target_behavior": target_behavior,
        "old_persona": old_payload,
        "new_persona": new_payload,
        "changed_fields": changed,
        "allowed_fields": sorted(ALLOWED_TOP_LEVEL),
        "locked_fields": sorted(LOCKED_FIELDS),
        "validation": validation,
        "sanitize": dict(sanitize_info or {}),
        "identity_model_fingerprint_before": _fingerprint(identity_fields),
        "identity_model_fingerprint_after": _fingerprint(new_identity_fields),
        "writer_raw_response": raw_response,
        "writer_prompt": prompt,
        "provider_attempts": list(provider_attempts or []),
        "accepted": accepted and validation["ok"],
        "reject_reason": reject_reason if reject_reason else "; ".join(validation["errors"]),
    }
    return record


def append_rewrite_record(path: Path | str, record: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(record), ensure_ascii=False, default=str) + "\n")


class PersonaWriter:
    def __init__(
        self,
        client: Any | None = None,
        *,
        mode: str = "rule",
        api: str = "chat",
        concurrency: int = 4,
        generation: int | None = None,
        record_path: Path | str | None = None,
    ) -> None:
        self.client = client
        self.mode = mode
        self.api = api
        self.generation = generation
        self.record_path = Path(record_path) if record_path is not None else None
        self._sem = asyncio.Semaphore(max(1, concurrency))

    async def rewrite_one(self, persona: Persona, current_behavior: str, target_behavior: str) -> dict[str, Any]:
        prompt = ""
        raw_response = ""
        error = ""
        accepted = False
        mode_used = self.mode
        sanitize_info: dict[str, Any] = {}
        provider_attempts: list[dict[str, Any]] = []
        try:
            if self.mode == "real_llm" and self.client is not None:
                async with self._sem:
                    prompt = build_writer_prompt(persona, current_behavior, target_behavior)
                    raw_response = await self._generate_with_provider_fallback(persona, prompt, provider_attempts)
                    delta = _coerce_json(raw_response)
            else:
                delta = rule_delta_for_target(persona, target_behavior)
                raw_response = json.dumps(delta, ensure_ascii=False)
            safe_delta, sanitize_info = sanitize_delta(delta)
            if not safe_delta:
                raise ValueError("empty sanitized persona delta")
            ok, reason = validate_delta(safe_delta)
            if not ok:
                raise ValueError(reason)
            updated = apply_delta(persona, safe_delta, generation=self.generation)
            validation = validate_persona_rewrite(persona, updated)
            if not validation["ok"]:
                raise ValueError("; ".join(validation["errors"]))
            accepted = True
        except Exception as exc:
            error = str(exc)
            mode_used = "rule_fallback"
            delta = rule_delta_for_target(persona, target_behavior)
            raw_response = json.dumps(delta, ensure_ascii=False)
            safe_delta, sanitize_info = sanitize_delta(delta)
            updated = apply_delta(persona, safe_delta, generation=self.generation)
        record = build_rewrite_record(
            persona,
            updated,
            current_behavior=current_behavior,
            target_behavior=target_behavior,
            mode=mode_used,
            raw_response=raw_response,
            accepted=accepted,
            reject_reason=error,
            sanitize_info=sanitize_info,
            prompt=prompt,
            provider_attempts=provider_attempts,
        )
        if self.record_path is not None:
            append_rewrite_record(self.record_path, record)
        return record

    async def _generate_with_provider_fallback(
        self,
        persona: Persona,
        prompt: str,
        provider_attempts: list[dict[str, Any]],
    ) -> str:
        providers = self._candidate_providers(persona)
        last_error = ""
        for provider in providers:
            model = self._model_for_provider(persona, provider)
            attempt = {"provider": provider, "model": model, "status": "started"}
            provider_attempts.append(attempt)
            try:
                raw = await asyncio.to_thread(
                    self.client.generate_text,
                    prompt,
                    self.api,
                    {"provider": provider, "model": model, "temperature": 0.35},
                )
                raw_text = str(raw or "")
                attempt["raw_length"] = len(raw_text)
                ok, reason = self._raw_response_is_valid_delta(raw_text)
                if ok:
                    attempt["status"] = "ok"
                    return raw_text
                attempt["status"] = "non_json_or_invalid"
                attempt["error"] = reason[:500]
                attempt["raw_preview"] = raw_text[:500]
                repaired = await self._repair_response(provider, model, prompt, raw_text, attempt)
                ok, repair_reason = self._raw_response_is_valid_delta(repaired)
                if ok:
                    attempt["status"] = "ok_repaired"
                    return repaired
                last_error = repair_reason
                attempt["repair_error"] = repair_reason[:500]
            except Exception as exc:
                last_error = redact_error(exc)
                attempt["status"] = "error"
                attempt["error"] = last_error[:500]
        raise RuntimeError(f"all persona writer provider attempts failed: {last_error}")

    async def _repair_response(self, provider: str, model: str, prompt: str, raw_text: str, attempt: dict[str, Any]) -> str:
        repair_prompt = build_writer_repair_prompt(prompt, raw_text)
        try:
            repaired = await asyncio.to_thread(
                self.client.generate_text,
                repair_prompt,
                self.api,
                {"provider": provider, "model": model, "temperature": 0.1},
            )
        except Exception as exc:
            attempt["repair_status"] = "error"
            attempt["repair_exception"] = str(exc)[:500]
            return ""
        repaired_text = str(repaired or "")
        attempt["repair_status"] = "ok"
        attempt["repair_raw_length"] = len(repaired_text)
        attempt["repair_raw_preview"] = repaired_text[:500]
        return repaired_text

    @staticmethod
    def _raw_response_is_valid_delta(raw_text: str) -> tuple[bool, str]:
        try:
            delta = _coerce_json(raw_text)
            safe_delta, _sanitize_info = sanitize_delta(delta)
            if not safe_delta:
                return False, "empty sanitized persona delta"
            ok, reason = validate_delta(safe_delta)
            if not ok:
                return False, reason
        except Exception as exc:
            return False, str(exc)
        return True, ""

    def _candidate_providers(self, persona: Persona) -> list[str]:
        available = []
        raw_clients = getattr(self.client, "raw_clients", {}) if self.client is not None else {}
        if isinstance(raw_clients, Mapping):
            available = [str(provider) for provider in raw_clients if provider in {"openai", "gemini"}]
        preferred = persona.llm_provider if persona.llm_provider in {"openai", "gemini"} else None
        ordered: list[str] = []
        for provider in [preferred, "openai", "gemini"]:
            if provider and provider not in ordered and (not available or provider in available):
                ordered.append(provider)
        return ordered or ["openai"]

    def _model_for_provider(self, persona: Persona, provider: str) -> str:
        if provider == persona.llm_provider and persona.llm_model:
            return persona.llm_model
        settings = getattr(self.client, "provider_settings", {}) if self.client is not None else {}
        provider_settings = settings.get(provider) if isinstance(settings, Mapping) else None
        model = getattr(provider_settings, "model", None)
        if model:
            return str(model)
        return "gemini-3-flash-preview" if provider == "gemini" else "gpt-5.5"

    async def rewrite_batch(self, personas: list[Persona], assignments: Mapping[str, tuple[str, str]]) -> tuple[list[Persona], list[dict[str, Any]]]:
        persona_by_id = {persona.user_id: persona for persona in personas}
        tasks = []
        for user_id, (current_behavior, target_behavior) in assignments.items():
            persona = persona_by_id.get(user_id)
            if persona is None:
                continue
            tasks.append(asyncio.create_task(self.rewrite_one(persona, current_behavior, target_behavior)))
        records = await asyncio.gather(*tasks) if tasks else []
        updated_by_id = {record["user_id"]: Persona(**record["new_persona"]) for record in records}
        next_personas = [updated_by_id.get(persona.user_id, persona) for persona in personas]
        return next_personas, records


def write_rewrite_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False, default=str) + "\n")


__all__ = [
    "ALLOWED_TOP_LEVEL",
    "ALLOWED_TRAITS",
    "LOCKED_FIELDS",
    "SCHEMA_VERSION",
    "PersonaWriter",
    "append_rewrite_record",
    "apply_delta",
    "build_rewrite_record",
    "build_writer_prompt",
    "rule_delta_for_target",
    "sanitize_delta",
    "validate_delta",
    "validate_persona_rewrite",
    "write_rewrite_records",
]
