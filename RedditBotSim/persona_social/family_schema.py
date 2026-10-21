from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Iterable

from .models import Persona


SCHEMA_VERSION = "phase0-2"
PERSONA_FIELDS = (
    "user_id",
    "user_name",
    "age",
    "gender",
    "region",
    "education",
    "ideology",
    "description",
    "subreddits",
    "interest_weights",
    "style",
    "stance",
    "debate_tendency",
    "post_tendency",
    "reply_tendency",
    "revisit_tendency",
    "exploration_tendency",
    "brevity",
    "family",
    "family_round",
    "family_generation",
    "llm_provider",
    "llm_model",
    "llm_temperature",
    "llm_reasoning_effort",
    "graph_affinity",
    "notification_reactivity",
)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    return value


@dataclass(frozen=True)
class FamilySpec:
    family: str
    slug: str
    description: str
    axes: dict[str, float] = field(default_factory=dict)
    dimension_axes: dict[str, float] = field(default_factory=dict)
    prompt_notes: list[str] = field(default_factory=list)
    behavior_notes: list[str] = field(default_factory=list)
    graph_notes: list[str] = field(default_factory=list)
    time_notes: list[str] = field(default_factory=list)
    mutation_notes: list[str] = field(default_factory=list)
    family_version: str = "base"
    mutation_parent: str | None = None
    mutation_reason: str | None = None
    mutation_id: str | None = None
    mutation_depth: int = 0
    mutation_deltas: dict[str, float] = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)
    time_profile: dict[str, float] = field(default_factory=dict)
    feedback: dict[str, float] = field(default_factory=dict)
    family_seed: int | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class FamilyManifest:
    family: str
    generation: int
    count: int
    seed: int | None = None
    family_version: str = "base"
    mutation_parent: str | None = None
    mutation_reason: str | None = None
    mutation_id: str | None = None
    mutation_depth: int = 0
    mutation_deltas: dict[str, float] = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)
    dimension_axes: dict[str, float] = field(default_factory=dict)
    time_profile: dict[str, float] = field(default_factory=dict)
    model_pool: list[dict[str, Any]] = field(default_factory=list)
    feedback: dict[str, float] = field(default_factory=dict)
    mutation: dict[str, Any] = field(default_factory=dict)
    family_spec: dict[str, Any] = field(default_factory=dict)
    personas: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def to_json(self, **kwargs: Any) -> str:
        options = {"ensure_ascii": False, "indent": 2}
        options.update(kwargs)
        return json.dumps(self.to_dict(), **options)


def persona_to_record(persona: Persona | dict[str, Any]) -> dict[str, Any]:
    extra_fields = (
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
    )
    if isinstance(persona, dict):
        payload = {field: persona.get(field) for field in PERSONA_FIELDS if field in persona}
        for field in extra_fields:
            if field in persona:
                payload[field] = persona[field]
    else:
        payload = {field: getattr(persona, field) for field in PERSONA_FIELDS}
        for field in extra_fields:
            if hasattr(persona, field):
                payload[field] = getattr(persona, field)
    payload["subreddits"] = list(payload["subreddits"])
    payload["interest_weights"] = dict(payload["interest_weights"])
    return _json_safe(payload)


def personas_to_records(personas: Iterable[Persona | dict[str, Any]]) -> list[dict[str, Any]]:
    return [persona_to_record(persona) for persona in personas]


def build_family_manifest(
    family: str,
    generation: int,
    personas: Iterable[Persona | dict[str, Any]],
    *,
    model_pool: list[dict[str, Any]] | None = None,
    feedback: dict[str, float] | None = None,
    seed: int | None = None,
    family_spec: FamilySpec | dict[str, Any] | None = None,
    mutation: Any | None = None,
    family_version: str | None = None,
    dimension_axes: dict[str, float] | None = None,
    time_profile: dict[str, float] | None = None,
    lineage: list[str] | None = None,
) -> FamilyManifest:
    def _payload(value: Any | None) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return _json_safe(dict(value))
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return _json_safe(value.to_dict())
        if is_dataclass(value):
            return _json_safe(asdict(value))
        return {}

    spec_payload = _payload(family_spec)
    mutation_payload = _payload(mutation)
    resolved_family_version = str(
        family_version
        or spec_payload.get("family_version")
        or mutation_payload.get("family_version")
        or "base"
    )
    resolved_mutation_parent = mutation_payload.get("mutation_parent", spec_payload.get("mutation_parent"))
    resolved_mutation_reason = mutation_payload.get("mutation_reason", spec_payload.get("mutation_reason"))
    resolved_mutation_id = mutation_payload.get("mutation_id", spec_payload.get("mutation_id"))
    resolved_mutation_depth = int(
        mutation_payload.get("mutation_depth", spec_payload.get("mutation_depth", 0)) or 0
    )
    resolved_mutation_deltas = dict(mutation_payload.get("dimension_deltas", spec_payload.get("mutation_deltas", {})) or {})
    resolved_lineage = list(lineage if lineage is not None else mutation_payload.get("lineage", spec_payload.get("lineage", [])) or [])
    resolved_dimension_axes = dict(
        dimension_axes
        or mutation_payload.get("dimension_axes")
        or spec_payload.get("dimension_axes")
        or {}
    )
    resolved_time_profile = dict(
        time_profile
        or mutation_payload.get("time_profile")
        or spec_payload.get("time_profile")
        or {}
    )
    records = personas_to_records(personas)
    return FamilyManifest(
        family=family,
        generation=generation,
        count=len(records),
        seed=seed,
        family_version=resolved_family_version,
        mutation_parent=str(resolved_mutation_parent) if resolved_mutation_parent not in (None, "") else None,
        mutation_reason=str(resolved_mutation_reason) if resolved_mutation_reason not in (None, "") else None,
        mutation_id=str(resolved_mutation_id) if resolved_mutation_id not in (None, "") else None,
        mutation_depth=resolved_mutation_depth,
        mutation_deltas=resolved_mutation_deltas,
        lineage=resolved_lineage,
        dimension_axes=resolved_dimension_axes,
        time_profile=resolved_time_profile,
        model_pool=[dict(item) for item in (model_pool or [])],
        feedback=dict(feedback or {}),
        mutation=mutation_payload,
        family_spec=spec_payload,
        personas=records,
    )


def family_manifest_payload(
    family: str,
    generation: int,
    personas: Iterable[Persona | dict[str, Any]],
    *,
    model_pool: list[dict[str, Any]] | None = None,
    feedback: dict[str, float] | None = None,
    seed: int | None = None,
    family_spec: FamilySpec | dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_family_manifest(
        family=family,
        generation=generation,
        personas=personas,
        model_pool=model_pool,
        feedback=feedback,
        seed=seed,
        family_spec=family_spec,
    ).to_dict()


def export_persona_list(personas: Iterable[Persona | dict[str, Any]]) -> list[dict[str, Any]]:
    return personas_to_records(personas)
