from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .family_registry import get_family_spec, normalize_family_name
from .family_schema import FamilyManifest, FamilySpec, family_manifest_payload, export_persona_list
from .family_mutation import FamilyMutation, mutate_family_spec, mutate_personas
from .models import Persona
from .personas import build_family_personas as _legacy_build_family_personas


def sample_model_pool(seed: int | random.Random = 0) -> list[dict[str, Any]]:
    rng = seed if isinstance(seed, random.Random) else random.Random(seed)
    base_pool = [
        {"provider": "openai", "model": "gpt-5.4", "temperature": 0.4, "reasoning_effort": "medium"},
        {"provider": "gemini", "model": "gemini-3-flash-preview", "temperature": 0.6, "reasoning_effort": "medium"},
        {"provider": "openai", "model": "gpt-5.4", "temperature": 0.8, "reasoning_effort": "medium"},
        {"provider": "gemini", "model": "gemini-3-pro-preview", "temperature": 0.8, "reasoning_effort": "medium"},
    ]
    pool: list[dict[str, Any]] = []
    for entry in base_pool:
        item = dict(entry)
        item["temperature"] = round(max(0.05, min(1.2, float(item["temperature"]) + rng.choice([-0.1, -0.05, 0.0, 0.05, 0.1]))), 3)
        pool.append(item)
    return pool


def sample_family_personas(
    family: str,
    generation: int,
    count: int,
    model_pool: list[dict[str, object]],
    feedback: dict[str, float] | None = None,
    *,
    evolve: bool = False,
    seed: int | None = None,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> list[Persona]:
    canonical_family = normalize_family_name(family)
    spec = get_family_spec(canonical_family)
    actual_model_pool = model_pool or sample_model_pool(seed or 0)
    personas = _legacy_build_family_personas(canonical_family, generation, count, actual_model_pool, feedback=feedback)
    if not evolve:
        return personas
    _, mutation = mutate_family_spec(
        spec,
        generation,
        feedback=feedback,
        seed=seed,
        parent_version=parent_version,
        lineage=lineage,
    )
    return mutate_personas(personas, mutation, family=canonical_family)


build_family_personas = sample_family_personas


def build_family_manifest(
    family: str,
    generation: int,
    personas: Iterable[Persona | dict[str, Any]] | int,
    *,
    model_pool: list[dict[str, Any]] | None = None,
    feedback: dict[str, float] | None = None,
    seed: int | None = None,
    evolve: bool = False,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> FamilyManifest:
    spec = get_family_spec(family)
    actual_model_pool = model_pool or sample_model_pool(seed or 0)
    mutation: FamilyMutation | None = None
    manifest_spec: FamilySpec = spec
    if isinstance(personas, int):
        if evolve:
            manifest_spec, mutation = mutate_family_spec(
                spec,
                generation,
                feedback=feedback,
                seed=seed,
                parent_version=parent_version,
                lineage=lineage,
            )
        persona_list = sample_family_personas(
            family,
            generation,
            personas,
            actual_model_pool,
            feedback=feedback,
            evolve=False,
            seed=seed,
            parent_version=parent_version,
            lineage=lineage,
        )
        if mutation is not None:
            persona_list = mutate_personas(persona_list, mutation, family=spec.family)
    else:
        persona_list = list(personas)
        if evolve:
            manifest_spec, mutation = mutate_family_spec(
                spec,
                generation,
                feedback=feedback,
                seed=seed,
                parent_version=parent_version,
                lineage=lineage,
            )
            persona_list = mutate_personas(persona_list, mutation, family=spec.family)
    return FamilyManifest(
        family=spec.family,
        generation=generation,
        count=len(persona_list),
        seed=seed,
        family_version=mutation.family_version if mutation else "base",
        mutation_parent=mutation.mutation_parent if mutation else None,
        mutation_reason=mutation.mutation_reason if mutation else None,
        mutation_id=mutation.mutation_id if mutation else None,
        mutation_depth=mutation.mutation_depth if mutation else 0,
        mutation_deltas=dict(mutation.dimension_deltas) if mutation else {},
        lineage=list(mutation.lineage) if mutation else [],
        dimension_axes=dict(mutation.dimension_axes) if mutation else dict(manifest_spec.dimension_axes or manifest_spec.axes),
        time_profile=dict(mutation.time_profile) if mutation else dict(manifest_spec.time_profile),
        model_pool=[dict(item) for item in actual_model_pool],
        feedback=dict(feedback or {}),
        mutation=mutation.to_dict() if mutation else {},
        family_spec=manifest_spec.to_dict(),
        personas=export_persona_list(persona_list),
    )


def export_family_manifest(
    family: str,
    generation: int,
    count: int,
    model_pool: list[dict[str, Any]],
    feedback: dict[str, float] | None = None,
    seed: int | None = None,
    evolve: bool = False,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> dict[str, Any]:
    if evolve:
        manifest = build_family_manifest(
            family,
            generation,
            count,
            model_pool=model_pool,
            feedback=feedback,
            seed=seed,
            evolve=True,
            parent_version=parent_version,
            lineage=lineage,
        )
        return manifest.to_dict()
    personas = sample_family_personas(family, generation, count, model_pool, feedback=feedback)
    return family_manifest_payload(
        family=normalize_family_name(family),
        generation=generation,
        personas=personas,
        model_pool=model_pool,
        feedback=feedback,
        seed=seed,
        family_spec=get_family_spec(family),
    )


def export_persona_list_for_family(
    family: str,
    generation: int,
    count: int,
    model_pool: list[dict[str, Any]],
    feedback: dict[str, float] | None = None,
    evolve: bool = False,
    seed: int | None = None,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> list[dict[str, Any]]:
    personas = sample_family_personas(
        family,
        generation,
        count,
        model_pool,
        feedback=feedback,
        evolve=evolve,
        seed=seed,
        parent_version=parent_version,
        lineage=lineage,
    )
    return export_persona_list(personas)


def build_family_bundle(
    family: str,
    generation: int,
    count: int,
    model_pool: list[dict[str, Any]],
    feedback: dict[str, float] | None = None,
    seed: int | None = None,
    evolve: bool = False,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> dict[str, Any]:
    manifest = build_family_manifest(
        family=family,
        generation=generation,
        personas=count,
        model_pool=model_pool,
        feedback=feedback,
        seed=seed,
        evolve=evolve,
        parent_version=parent_version,
        lineage=lineage,
    )
    return {"manifest": manifest.to_dict(), "personas": manifest.personas}


def build_evolved_family_bundle(
    family: str,
    generation: int,
    count: int,
    model_pool: list[dict[str, Any]],
    feedback: dict[str, float] | None = None,
    seed: int | None = None,
    parent_version: str | None = None,
    lineage: list[str] | None = None,
) -> dict[str, Any]:
    manifest = build_family_manifest(
        family=family,
        generation=generation,
        personas=count,
        model_pool=model_pool,
        feedback=feedback,
        seed=seed,
        evolve=True,
        parent_version=parent_version,
        lineage=lineage,
    )
    return {"manifest": manifest.to_dict(), "personas": manifest.personas, "mutation": manifest.mutation}
