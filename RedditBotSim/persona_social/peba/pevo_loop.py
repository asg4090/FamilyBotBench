from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from ..models import Persona
from ..personas import load_default_personas
from .behavior_classifier import classify_trajectories_llm, classify_trajectories_rule
from .behavior_taxonomy import BEHAVIOR_CLASSES
from .credit_assignment import assign_persona_update_targets
from .detector_eval import run_detector_sanity
from .peba_metrics import distribution_gap
from .persona_writer import PersonaWriter, write_rewrite_records
from .reference_distribution import build_human_reference, load_reference
from .report import write_final_report, write_iteration_report
from .trajectory_summarizer import summarize_trajectories


DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"


def _load_local_env(reddit_dir: Path) -> None:
    env_file = reddit_dir / ".env.local"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _model_pool(reddit_dir: Path) -> list[dict[str, Any]]:
    _load_local_env(reddit_dir)
    openai_model = os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    gemini_model = os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    return [
        {"provider": "openai", "model": openai_model, "temperature": 0.4, "reasoning_effort": "medium"},
        {"provider": "openai", "model": openai_model, "temperature": 0.8, "reasoning_effort": "medium"},
        {"provider": "gemini", "model": gemini_model, "temperature": 0.4, "reasoning_effort": "medium"},
        {"provider": "gemini", "model": gemini_model, "temperature": 0.8, "reasoning_effort": "medium"},
    ]


@dataclass
class PEvoConfig:
    project_root: Path
    reddit_dir: Path
    output_dir: Path
    family: str = "pevo_global"
    reference_slice: str = "global"
    bots: int = 50
    iterations: int = 5
    rounds_per_simulation: int = 6
    browse_count: int = 10
    decision_mode: str = "llm_first"
    providers: tuple[str, ...] = ("openai", "gemini")
    classifier: str = "rule"
    persona_writer: str = "rule"
    api: str = "responses"
    concurrency: int = 8
    retries: int = 2
    llm_timeout: float = 90.0
    seed: int = 20260606
    js_epsilon: float = 0.08
    min_improvement: float = 0.01
    dry_run: bool = False
    skip_detectors: bool = False


def _provider_pool(providers: tuple[str, ...], reddit_dir: Path) -> list[dict[str, Any]]:
    allowed = {provider.strip().lower() for provider in providers}
    pool = _model_pool(reddit_dir)
    return [dict(item) for item in pool if item["provider"] in allowed] or [dict(pool[0])]


def _initial_personas(config: PEvoConfig) -> list[Persona]:
    pool = _provider_pool(config.providers, config.reddit_dir)
    personas = []
    for idx, persona in enumerate(load_default_personas(config.bots), start=1):
        model_cfg = pool[(idx - 1) % len(pool)]
        traits = dict(persona.traits or {})
        traits.update(
            {
                "attention_style": traits.get("attention_style", "notices one or two concrete details in a thread"),
                "emotional_reactivity": traits.get("emotional_reactivity", "moderate, context dependent"),
                "conflict_style": traits.get("conflict_style", "pushes back when something sounds too neat"),
                "certainty_style": traits.get("certainty_style", "plain but not absolute"),
                "thread_memory": traits.get("thread_memory", "loose memory of recent threads"),
                "social_energy": traits.get("social_energy", "ordinary intermittent Reddit participation"),
                "reply_impulse": traits.get("reply_impulse", "sometimes replies when a comment gives them a hook"),
                "topic_curiosity": traits.get("topic_curiosity", "follows a few adjacent topics"),
                "revisit_habit": traits.get("revisit_habit", "checks back occasionally, not mechanically"),
            }
        )
        personas.append(
            replace(
                persona,
                user_id=f"{config.family}_g000_{idx:03d}",
                user_name=f"{persona.user_name}_{config.family}_g0",
                family=config.family,
                family_generation=0,
                family_round=0,
                llm_provider=model_cfg["provider"],
                llm_model=model_cfg["model"],
                llm_temperature=float(model_cfg["temperature"]),
                llm_reasoning_effort=str(model_cfg["reasoning_effort"]),
                traits=traits,
            )
        )
    return personas


def _write_personas(path: Path, personas: list[Persona], *, family: str, generation: int) -> None:
    personas = _stamp_generation(personas, generation)
    payload = {
        "family": family,
        "generation": generation,
        "personas": [asdict(persona) for persona in personas],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_personas(path: Path) -> list[Persona]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("personas", payload if isinstance(payload, list) else [])
    return [Persona(**row) for row in rows]


def _stamp_generation(personas: list[Persona], generation: int) -> list[Persona]:
    return [replace(persona, family_generation=generation, family_round=generation) for persona in personas]


def _run_simulation(config: PEvoConfig, iteration_dir: Path, manifest_path: Path) -> None:
    run_dir = iteration_dir / "run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        sys.executable,
        str(config.reddit_dir / "persona_social_runner.py"),
        "--run-dir",
        str(run_dir),
        "--config",
        str(config.reddit_dir / "config.yml"),
        "--max-agents",
        str(config.bots),
        "--rounds",
        str(config.rounds_per_simulation),
        "--browse-count",
        str(config.browse_count),
        "--concurrency",
        str(config.concurrency),
        "--retries",
        str(config.retries),
        "--llm-timeout",
        str(config.llm_timeout),
        "--api",
        config.api,
        "--reasoning-effort",
        "medium",
        "--seed",
        str(config.seed),
        "--clean",
        "--family-manifest",
        str(manifest_path),
        "--decision-mode",
        config.decision_mode,
    ]
    if config.dry_run:
        cmd.append("--dry-run")
    subprocess.run(cmd, cwd=config.reddit_dir, check=True)


def _load_writer_client(config: PEvoConfig):
    if config.persona_writer != "real_llm":
        return None
    if str(config.reddit_dir) not in sys.path:
        sys.path.insert(0, str(config.reddit_dir))
    from llm_client import build_llm_client

    client, _settings = build_llm_client(allowed_providers=set(config.providers))
    return client


def _load_classifier_client(config: PEvoConfig):
    if config.classifier != "llm":
        return None
    if str(config.reddit_dir) not in sys.path:
        sys.path.insert(0, str(config.reddit_dir))
    from llm_client import build_llm_client

    client, _settings = build_llm_client(allowed_providers=set(config.providers))
    return client


async def run_pevo(config: PEvoConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = config.output_dir / "human_reference"
    if not (reference_dir / "reference_index.json").exists():
        build_human_reference(config.reddit_dir / "config.yml", reference_dir)
    reference = load_reference(reference_dir, config.reference_slice)
    p_real = reference["behavior_distribution"]
    writer_client = _load_writer_client(config)
    classifier_client = _load_classifier_client(config)

    personas = _initial_personas(config)
    history_rows: list[dict[str, Any]] = []
    previous_js: float | None = None
    final_iteration_dir: Path | None = None

    for iteration in range(config.iterations):
        iteration_dir = config.output_dir / f"iteration_{iteration:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        personas = _stamp_generation(personas, iteration)
        manifest_path = iteration_dir / "personas.json"
        _write_personas(manifest_path, personas, family=config.family, generation=iteration)
        _run_simulation(config, iteration_dir, manifest_path)

        summaries = summarize_trajectories(iteration_dir, iteration_dir / "trajectory_summary.jsonl")
        if config.classifier == "llm" and classifier_client is not None:
            labels = classify_trajectories_llm(summaries, classifier_client, api=config.api)
        else:
            labels = classify_trajectories_rule(summaries)
        labels.to_csv(iteration_dir / "behavior_labels.csv", index=False)
        gap_payload = distribution_gap(labels["behavior_label"].tolist(), p_real, BEHAVIOR_CLASSES, alpha=0.5)
        (iteration_dir / "p_sim.json").write_text(json.dumps(gap_payload["p_sim"], ensure_ascii=False, indent=2), encoding="utf-8")
        (iteration_dir / "distribution_gap.json").write_text(json.dumps(gap_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        metrics = gap_payload["metrics"]
        js = float(metrics["js"])
        tv = float(metrics["tv"])
        kl = float(metrics["kl"])
        assignments = assign_persona_update_targets(
            labels,
            gap_payload["gap"],
            seed=config.seed + iteration * 1009,
            max_update_fraction=0.35,
        )
        assignments.to_csv(iteration_dir / "persona_update_targets.csv", index=False)
        assignment_map = {
            row["user_id"]: (row["current_behavior"], row["target_behavior"])
            for row in assignments.to_dict(orient="records")
        }
        should_stop = js <= config.js_epsilon
        if previous_js is not None and previous_js - js < config.min_improvement and iteration > 0:
            should_stop = True

        rewrites: list[dict[str, Any]] = []
        if not should_stop and assignment_map:
            writer = PersonaWriter(
                writer_client,
                mode=config.persona_writer,
                api=config.api,
                concurrency=min(8, config.concurrency),
                generation=iteration + 1,
            )
            personas, rewrites = await writer.rewrite_batch(personas, assignment_map)
        write_rewrite_records(iteration_dir / "persona_rewrites.jsonl", rewrites)
        _write_personas(iteration_dir / "next_personas.json", personas, family=config.family, generation=iteration + 1)
        write_iteration_report(iteration_dir, gap_payload, labels, rewrites)

        history_rows.append(
            {
                "iteration": iteration,
                "js": js,
                "tv": tv,
                "kl": kl,
                "entropy_gap": float(metrics["entropy_gap"]),
                "rewrite_count": len(rewrites),
                "reference_slice": config.reference_slice,
                "classifier": config.classifier,
                "persona_writer": config.persona_writer,
                "stopped": should_stop,
            }
        )
        final_iteration_dir = iteration_dir
        previous_js = js
        if should_stop:
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(config.output_dir / "pevo_alignment_history.csv", index=False)
    _write_personas(config.output_dir / "qualified_personas_final.json", personas, family=config.family, generation=len(history_rows))
    detector_summary = {}
    if not config.skip_detectors and final_iteration_dir is not None:
        detector_summary = run_detector_sanity(config.project_root, final_iteration_dir, config.output_dir)
    write_final_report(config.output_dir, history, detector_summary)
    return {
        "output_dir": str(config.output_dir),
        "iterations_completed": int(len(history_rows)),
        "final_metrics": history_rows[-1] if history_rows else {},
        "detectors": detector_summary,
    }
