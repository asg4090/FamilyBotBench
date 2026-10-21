#!/usr/bin/env python3
"""Persona-aware RedditBotSim runner for offline social-behavior experiments."""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import argparse
import asyncio
import ast
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


REDDIT_BOT_SIM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REDDIT_BOT_SIM_DIR.parent
for rel in ["Action", "Environment", "Utils", "AgentDesicionCenter"]:
    sys.path.insert(0, str(REDDIT_BOT_SIM_DIR / rel))

from Entity import RedditComment1, RedditComment2, RedditPost, RedditUser  # noqa: E402
from env import RedditEnv  # noqa: E402
from llm_client import build_llm_client, load_llm_settings  # noqa: E402
from safe_runner import DryRunClient, _as_project_relative, _copy_csv, _resolve_path  # noqa: E402
from utils import generate_reddit_id, load_config, set_logger  # noqa: E402

from persona_social.audit import build_behavior_audit, export_generated_comments, export_generated_posts, graph_proxy_audit  # noqa: E402
from persona_social.family_sampler import build_family_manifest as build_family_manifest_bundle, sample_model_pool  # noqa: E402
from persona_social.decision import DecisionEngine  # noqa: E402
from persona_social.llm import AsyncLLMGenerator  # noqa: E402
from persona_social.models import Candidate, Decision, MemoryState, Persona  # noqa: E402
from persona_social.personas import load_default_personas  # noqa: E402
from persona_social.prompts import build_action_prompt  # noqa: E402
from persona_social.recommender import PersonaRecommender  # noqa: E402
from persona_social.trace import JsonlWriter  # noqa: E402


@lru_cache(maxsize=1)
def _default_persona_template() -> Persona:
    return load_default_personas(1)[0]


def _persona_value(persona: Any, key: str, default: Any = None) -> Any:
    if isinstance(persona, Mapping):
        return persona.get(key, default)
    return getattr(persona, key, default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(default)
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return list(parsed)
            except Exception:
                pass
        return [text]
    try:
        return list(value)
    except Exception:
        return list(default)


def _safe_dict(value: Any, default: dict[str, float]) -> dict[str, float]:
    if isinstance(value, dict):
        out: dict[str, float] = {}
        for key, raw in value.items():
            try:
                out[str(key)] = float(raw)
            except (TypeError, ValueError):
                continue
        return out or dict(default)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return dict(default)
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, dict):
                    out: dict[str, float] = {}
                    for key, raw in parsed.items():
                        try:
                            out[str(key)] = float(raw)
                        except (TypeError, ValueError):
                            continue
                    return out or dict(default)
            except Exception:
                pass
    return dict(default)


def _safe_any_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): raw for key, raw in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, Mapping):
                    return {str(key): raw for key, raw in parsed.items()}
            except Exception:
                pass
    return {}


def _coerce_persona(persona_like: Any) -> Persona:
    if isinstance(persona_like, Persona):
        return persona_like
    if not isinstance(persona_like, Mapping):
        raise TypeError(f"Unsupported persona payload type: {type(persona_like)!r}")
    payload = dict(persona_like)
    nested = payload.get("persona")
    if isinstance(nested, Mapping):
        payload = {**dict(nested), **{k: v for k, v in payload.items() if k != "persona"}}
    template = _default_persona_template()
    family = _persona_value(payload, "family", _persona_value(payload, "family_name", template.family))
    family_generation = _safe_int(
        _persona_value(payload, "family_generation", _persona_value(payload, "generation", template.family_generation)),
        template.family_generation,
    )
    family_round = _safe_int(_persona_value(payload, "family_round", family_generation), template.family_round)
    llm_provider = _persona_value(payload, "llm_provider", _persona_value(payload, "provider", template.llm_provider))
    llm_model = _persona_value(payload, "llm_model", _persona_value(payload, "model", template.llm_model))
    llm_temperature = _persona_value(payload, "llm_temperature", _persona_value(payload, "temperature", template.llm_temperature))
    llm_reasoning_effort = _persona_value(
        payload,
        "llm_reasoning_effort",
        _persona_value(payload, "reasoning_effort", template.llm_reasoning_effort),
    )
    return Persona(
        user_id=str(_persona_value(payload, "user_id", template.user_id)),
        user_name=str(_persona_value(payload, "user_name", template.user_name)),
        age=_safe_int(_persona_value(payload, "age", template.age), template.age),
        gender=str(_persona_value(payload, "gender", template.gender)),
        region=str(_persona_value(payload, "region", template.region)),
        education=str(_persona_value(payload, "education", template.education)),
        ideology=str(_persona_value(payload, "ideology", template.ideology)),
        description=str(_persona_value(payload, "description", template.description)),
        subreddits=[str(item) for item in _safe_list(_persona_value(payload, "subreddits", template.subreddits), template.subreddits)],
        interest_weights=_safe_dict(_persona_value(payload, "interest_weights", template.interest_weights), template.interest_weights),
        style=str(_persona_value(payload, "style", template.style)),
        stance=str(_persona_value(payload, "stance", template.stance)),
        debate_tendency=_safe_float(_persona_value(payload, "debate_tendency", template.debate_tendency), template.debate_tendency),
        post_tendency=_safe_float(_persona_value(payload, "post_tendency", template.post_tendency), template.post_tendency),
        reply_tendency=_safe_float(_persona_value(payload, "reply_tendency", template.reply_tendency), template.reply_tendency),
        revisit_tendency=_safe_float(_persona_value(payload, "revisit_tendency", template.revisit_tendency), template.revisit_tendency),
        exploration_tendency=_safe_float(_persona_value(payload, "exploration_tendency", template.exploration_tendency), template.exploration_tendency),
        brevity=_safe_float(_persona_value(payload, "brevity", template.brevity), template.brevity),
        family=str(family or template.family),
        family_round=family_round,
        family_generation=family_generation,
        llm_provider=str(llm_provider) if llm_provider not in (None, "") else None,
        llm_model=str(llm_model) if llm_model not in (None, "") else None,
        llm_temperature=_safe_float(llm_temperature, template.llm_temperature if template.llm_temperature is not None else 0.4)
        if llm_temperature not in (None, "")
        else None,
        llm_reasoning_effort=str(llm_reasoning_effort) if llm_reasoning_effort not in (None, "") else None,
        graph_affinity=_safe_float(_persona_value(payload, "graph_affinity", template.graph_affinity), template.graph_affinity),
        notification_reactivity=_safe_float(
            _persona_value(payload, "notification_reactivity", template.notification_reactivity),
            template.notification_reactivity,
        ),
        traits=_safe_any_dict(_persona_value(payload, "traits", _persona_value(payload, "high_dim_traits", template.traits))),
    )


def _load_persona_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    if suffix in {".jsonl", ".ndjson"}:
        records = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, Mapping):
                records.append(dict(obj))
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        top_defaults = dict(payload)
        records = top_defaults.pop("personas", top_defaults.pop("records", top_defaults.pop("items", None)))
        if records is None:
            return [top_defaults]
        if not isinstance(records, list):
            return [top_defaults]
        merged_records = []
        for item in records:
            if isinstance(item, Mapping):
                merged = dict(top_defaults)
                merged.update(item)
                if "generation" in top_defaults and "family_generation" not in merged:
                    merged["family_generation"] = top_defaults["generation"]
                if "generation" in top_defaults and "family_round" not in merged:
                    merged["family_round"] = top_defaults["generation"]
                merged_records.append(merged)
        return merged_records
    raise ValueError(f"Unsupported family manifest format: {path}")


def _load_personas_from_manifest(path: Path) -> list[Persona]:
    return [_coerce_persona(item) for item in _load_persona_records(path)]


def _local_build_family_personas(family: str, generation: int, max_agents: int, model_pool: list[dict[str, object]]) -> list[Persona]:
    base_personas = [_coerce_persona(item) for item in load_default_personas(max_agents)]
    if not base_personas:
        return []
    out: list[Persona] = []
    for index, persona in enumerate(base_personas):
        model_cfg = model_pool[index % len(model_pool)] if model_pool else {}
        out.append(
            Persona(
                **{
                    **persona.__dict__,
                    "family": family,
                    "family_round": generation,
                    "family_generation": generation,
                    "llm_provider": str(model_cfg.get("provider", persona.llm_provider or "openai")) if model_cfg else persona.llm_provider,
                    "llm_model": str(model_cfg.get("model", persona.llm_model or "gpt-5.4")) if model_cfg else persona.llm_model,
                    "llm_temperature": float(model_cfg.get("temperature", persona.llm_temperature if persona.llm_temperature is not None else 0.4)) if model_cfg else persona.llm_temperature,
                    "llm_reasoning_effort": str(model_cfg.get("reasoning_effort", persona.llm_reasoning_effort or "medium")) if model_cfg else persona.llm_reasoning_effort,
                }
            )
        )
    return out


def _persona_family_block(persona: Persona) -> str:
    if (persona.family or "baseline") == "baseline" and persona.family_generation == 0 and persona.family_round == 0:
        return ""
    return f"""Family context:
- family: {persona.family}
- generation: {persona.family_generation}
- round: {persona.family_round}
- graph affinity: {persona.graph_affinity:.2f}
- notification reactivity: {persona.notification_reactivity:.2f}
"""


def prepare_run_config(base_config_path: Path, run_dir: Path, personas: list[Persona]) -> dict[str, Any]:
    config = load_config(str(base_config_path))
    run_data_dir = run_dir / "data"
    run_data_dir.mkdir(parents=True, exist_ok=True)
    for key, dst_name in {
        "reddit_posts": "Posts.csv",
        "reddit_comments1": "Comment_1.csv",
        "reddit_comments2": "Comment_2.csv",
        "reddit_users": "User.csv",
    }.items():
        dst = run_data_dir / dst_name
        _copy_csv(_resolve_path(config["paths"][key]), dst)
        config["paths"][key] = str(dst.resolve())
    profile_path = run_dir / "generated_persona_profiles.csv"
    write_profiles(personas, profile_path)
    config["paths"]["agent_bots"] = str(profile_path.resolve())
    config["agents_num"] = len(personas)
    config.setdefault("log", {})
    config["log"]["log_file"] = str(run_dir / "log" / "persona_social_runner.log")
    return config


def build_family_manifest(personas: list[Persona]) -> pd.DataFrame:
    rows = []
    for persona in personas:
        persona = _coerce_persona(persona)
        rows.append(
            {
                "user_id": persona.user_id,
                "user_name": persona.user_name,
                "family": persona.family,
                "family_generation": persona.family_generation,
                "family_round": persona.family_round,
                "provider": persona.llm_provider or "",
                "model": persona.llm_model or "",
                "temperature": persona.llm_temperature if persona.llm_temperature is not None else "",
                "reasoning_effort": persona.llm_reasoning_effort or "",
                "graph_affinity": persona.graph_affinity,
                "notification_reactivity": persona.notification_reactivity,
                "post_tendency": persona.post_tendency,
                "reply_tendency": persona.reply_tendency,
                "revisit_tendency": persona.revisit_tendency,
                "exploration_tendency": persona.exploration_tendency,
                "traits_json": json.dumps(persona.traits, ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def write_profiles(personas: list[Persona], path: Path) -> None:
    rows = []
    for persona in personas:
        persona = _coerce_persona(persona)
        rows.append(
            {
                "user_id": persona.user_id,
                "user_name": persona.user_name,
                "Age": persona.age,
                "Gender": persona.gender,
                "Region": persona.region,
                "EducationLevel": persona.education,
                "Ideology": persona.ideology,
                "Description": persona.description,
                "Preference": persona.primary_interest,
                "SubRedditNum": len(persona.subreddits),
                "SubRedditList": repr(persona.subreddits),
                "PostNum": 0,
                "Comment1Num": 0,
                "Comment2Num": 0.0,
                "Formulate_Post": "[]",
                "Formulate_Comment1": "[]",
                "Formulate_Comment2": "[]",
                "PostLengthList": "[]",
                "interest_weights": json.dumps(persona.interest_weights, ensure_ascii=False),
                "style": persona.style,
                "stance": persona.stance,
                "debate_tendency": persona.debate_tendency,
                "post_tendency": persona.post_tendency,
                "reply_tendency": persona.reply_tendency,
                "revisit_tendency": persona.revisit_tendency,
                "exploration_tendency": persona.exploration_tendency,
                "brevity": persona.brevity,
                "family": persona.family,
                "family_round": persona.family_round,
                "family_generation": persona.family_generation,
                "llm_provider": persona.llm_provider or "",
                "llm_model": persona.llm_model or "",
                "llm_temperature": persona.llm_temperature if persona.llm_temperature is not None else "",
                "llm_reasoning_effort": persona.llm_reasoning_effort or "",
                "graph_affinity": persona.graph_affinity,
                "notification_reactivity": persona.notification_reactivity,
                "traits_json": json.dumps(persona.traits, ensure_ascii=False, sort_keys=True),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def create_persona_user(env: RedditEnv, persona: Persona) -> RedditUser:
    persona = _coerce_persona(persona)
    user = RedditUser(
        author_id=persona.user_id,
        author_name=persona.user_name,
        description=persona.description,
        subreddit=repr(persona.subreddits),
        submission_num="0",
        comment_num="0",
        character_setting=persona.primary_interest,
        comment_num_1="0",
        comment_num_2="0",
    )
    env.get_env_reddit_data().users[persona.user_id] = user
    user.save_to_file(env.config["paths"]["reddit_users"])
    return user


def _temporal_bounds(persona: Persona | None, prefix: str, default_min: int, default_max: int) -> tuple[int, int]:
    if persona is None:
        return default_min, default_max
    traits = _safe_any_dict(_persona_value(persona, "traits", {}))
    profile = _safe_any_dict(traits.get("temporal_profile"))
    low = _safe_int(profile.get(f"{prefix}_delay_min"), default_min)
    high = _safe_int(profile.get(f"{prefix}_delay_max"), default_max)
    high = max(low + 1, high)
    return max(0, low), high


def next_comment_time(base_time: str, rng: random.Random, persona: Persona | None = None) -> str:
    dt = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
    low, high = _temporal_bounds(persona, "comment", 5, 210)
    delta = timedelta(minutes=rng.randint(low, high), seconds=rng.randint(0, 59))
    return (dt + delta).strftime("%Y-%m-%d %H:%M:%S")


def next_post_time(round_time: str, rng: random.Random, persona: Persona | None = None) -> str:
    dt = datetime.strptime(round_time, "%Y-%m-%d %H:%M:%S")
    low, high = _temporal_bounds(persona, "post", 0, 45)
    delta = timedelta(minutes=rng.randint(low, high), seconds=rng.randint(0, 59))
    return (dt + delta).strftime("%Y-%m-%d %H:%M:%S")


def fallback_content(decision: Decision) -> str:
    persona = decision.persona
    if decision.action == "browse_decide":
        return ""
    if decision.action == "post":
        return f"{persona.primary_interest}: this story feels like it needs more context"
    if decision.action in {"comment1", "comment2", "revisit"}:
        options = [
            "Yeah, that part is doing a lot of work.",
            "I don't totally buy that framing.",
            "This is exactly the detail people keep skipping.",
            "Feels like there is more going on here than the headline says.",
            "Honestly, I just want the plain version of what happened.",
        ]
        return options[hash(persona.user_id + decision.action) % len(options)]
    return ""


def build_round_times(start: str, end: str, rounds: int) -> list[datetime]:
    start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S") + timedelta(days=140)
    # Keep rounds close together so newly created posts/comments remain visible
    # to later browsing rounds and can produce weak bot-bot interaction.
    return [start_dt + timedelta(days=i * 2, hours=(i % 3) * 3) for i in range(rounds)]


def decision_to_record(decision: Decision, browse_items: list[dict[str, Any]]) -> dict[str, Any]:
    cand = decision.candidate
    persona = decision.persona
    cand_source = cand.source if cand else ("skip" if decision.action == "skip" else "")
    return {
        "user_id": persona.user_id,
        "user_name": persona.user_name,
        "family": persona.family,
        "family_generation": persona.family_generation,
        "family_round": persona.family_round,
        "graph_affinity": persona.graph_affinity,
        "notification_reactivity": persona.notification_reactivity,
        "llm_provider": persona.llm_provider or "",
        "llm_model": persona.llm_model or "",
        "llm_temperature": persona.llm_temperature if persona.llm_temperature is not None else "",
        "llm_reasoning_effort": persona.llm_reasoning_effort or "",
        "round_id": decision.round_id,
        "action": decision.action,
        "reason": decision.reason,
        "planned_time": decision.planned_time,
        "candidate_id": cand.candidate_id if cand else "",
        "candidate_source": cand_source,
        "post_id": cand.post_id if cand else "",
        "comment_id": cand.comment_id if cand else "",
        "subreddit": cand.subreddit if cand else "",
        "topic": cand.topic if cand else decision.persona.primary_interest,
        "browse_count": len(browse_items),
        "browse_items": browse_items,
        "llm_selected_action": decision.parsed.get("action", "") if isinstance(decision.parsed, dict) else "",
        "llm_selected_post_id": decision.parsed.get("target_post_id", "") if isinstance(decision.parsed, dict) else "",
        "llm_selected_comment_id": decision.parsed.get("target_comment_id", "") if isinstance(decision.parsed, dict) else "",
    }


class PersonaSocialSimulation:
    def __init__(
        self,
        config: dict[str, Any],
        personas: list[Persona],
        run_dir: Path,
        llm: AsyncLLMGenerator,
        rng: random.Random,
        browse_count: int,
        rounds: int,
        decision_mode: str = "system_first",
    ) -> None:
        self.config = config
        self.personas = personas
        self.run_dir = run_dir
        self.llm = llm
        self.rng = rng
        self.browse_count = browse_count
        self.rounds = rounds
        self.env = RedditEnv(config)
        self.env.reset()
        self.recommender = PersonaRecommender(self.env, rng)
        self.decision_mode = decision_mode
        self.decider = DecisionEngine(rng, mode=decision_mode)
        self.personas = [_coerce_persona(p) for p in personas]
        self.memories = {p.user_id: MemoryState(p.user_id) for p in self.personas}
        self.users = {p.user_id: create_persona_user(self.env, p) for p in self.personas}
        self.decision_trace = JsonlWriter(run_dir / "decision_trace.jsonl")
        self.llm_trace = JsonlWriter(run_dir / "llm_trace.jsonl")
        self.memory_trace = JsonlWriter(run_dir / "memory_trace.jsonl")
        self.summary: dict[str, int] = {
            "agents_created": len(self.personas),
            "browse_events": 0,
            "post_written": 0,
            "comment1_written": 0,
            "comment2_written": 0,
            "revisit_written": 0,
            "skip_events": 0,
            "llm_calls": 0,
            "llm_fallbacks": 0,
            "notifications_created": 0,
        }

    async def run(self) -> dict[str, int]:
        for round_id, round_time in enumerate(build_round_times(self.config["time"]["start_time"], self.config["time"]["end_time"], self.rounds), start=1):
            await self.run_round(round_id, round_time)
        return self.summary

    async def run_round(self, round_id: int, round_time: datetime) -> None:
        decisions: list[tuple[Decision, list[dict[str, Any]], list[Candidate]]] = []
        for persona in self.personas:
            memory = self.memories[persona.user_id]
            candidates = self.recommender.build_candidates(persona, memory, round_time, round_id, self.browse_count)
            browse_items = [candidate.to_prompt_item(i) for i, candidate in enumerate(candidates)]
            memory.seen_posts.update(candidate.post_id for candidate in candidates)
            decision = self.decider.choose(persona, memory, round_id, round_time, candidates)
            if decision.action == "skip":
                memory.record_action("skip")
            self.summary["browse_events"] += 1
            self.decision_trace.write("browse", **decision_to_record(decision, browse_items))
            decisions.append((decision, browse_items, candidates))

        tasks = []
        for decision, browse_items, _candidates in decisions:
            if decision.action == "skip":
                tasks.append(None)
                continue
            prompt = build_action_prompt(
                decision.persona,
                decision.action,
                decision.candidate,
                decision.planned_time,
                browse_items,
                memory=self.memories[decision.persona.user_id],
            )
            decision.prompt = prompt
            tasks.append(asyncio.create_task(self._generate_for_decision(decision, prompt)))

        real_tasks = [task for task in tasks if task is not None]
        real_results = await asyncio.gather(*real_tasks) if real_tasks else []
        result_iter = iter(real_results)
        results = [None if task is None else next(result_iter) for task in tasks]

        for (decision, browse_items, candidates), result in zip(decisions, results):
            if decision.action == "skip":
                self.summary["skip_events"] += 1
                self.decision_trace.write("decision", **decision_to_record(decision, browse_items))
                continue
            self.summary["llm_calls"] += 0 if self.llm.dry_run else 1
            if result is None:
                decision.content = fallback_content(decision)
                decision.fallback = True
            else:
                decision.raw_response = result.raw_text
                decision.parsed = result.parsed
                decision.content = result.content or fallback_content(decision)
                decision.fallback = bool(result.error)
            if decision.action == "browse_decide":
                self._apply_llm_thread_choice(decision, candidates)
            if decision.fallback:
                self.summary["llm_fallbacks"] += 1
            self.llm_trace.write(
                "llm_result",
                user_id=decision.persona.user_id,
                action=decision.action,
                round_id=decision.round_id,
                prompt=decision.prompt,
                raw_response=decision.raw_response,
                parsed=decision.parsed,
                content=decision.content,
                error=result.error if result is not None else "missing_result",
                fallback=decision.fallback,
                meta=result.meta if result is not None else {},
            )
            self.decision_trace.write("decision", **decision_to_record(decision, browse_items))
            if decision.action == "skip":
                self.summary["skip_events"] += 1
                self.memories[decision.persona.user_id].record_action("skip")
                continue
            self.apply_decision(decision)

    def _apply_llm_thread_choice(self, decision: Decision, candidates: list[Candidate]) -> None:
        parsed = decision.parsed if isinstance(decision.parsed, dict) else {}
        action = str(parsed.get("action", "") or "").strip().lower()
        if action == "reply":
            action = "comment2"
        if action == "comment":
            action = "comment1"
        if action not in {"skip", "comment1", "comment2", "revisit", "post"}:
            decision.action = "skip"
            decision.candidate = None
            decision.reason = "safe_skip_after_invalid_llm_thread_choice"
            decision.content = ""
            return
        decision.action = action
        decision.reason = str(parsed.get("reason", "") or f"llm_thread_choice_{action}")
        selected = self._resolve_candidate_from_choice(parsed, candidates, action)
        decision.candidate = selected
        if action == "skip":
            decision.content = ""
            return
        if action in {"comment1", "comment2", "revisit"} and selected is None:
            decision.action = "skip"
            decision.candidate = None
            decision.reason = "safe_skip_after_missing_llm_target"
            decision.content = ""
            return
        if not decision.content:
            decision.content = fallback_content(decision)

    def _resolve_candidate_from_choice(self, parsed: dict[str, Any], candidates: list[Candidate], action: str) -> Candidate | None:
        if action == "post":
            return None
        target_post_id = str(parsed.get("target_post_id", "") or "").strip()
        target_comment_id = str(parsed.get("target_comment_id", "") or "").strip()
        by_post = {c.post_id: c for c in candidates}
        if action in {"comment2", "revisit"} and target_comment_id:
            for candidate in candidates:
                if target_post_id and candidate.post_id != target_post_id:
                    continue
                if str(candidate.comment_id or "") == target_comment_id:
                    return replace(candidate, action_hint="comment2")
            for candidate in candidates:
                if target_post_id and candidate.post_id != target_post_id:
                    continue
                for item in candidate.thread_comments:
                    if str(item.get("comment_id", "")) != target_comment_id:
                        continue
                    return replace(
                        candidate,
                        action_hint="comment2",
                        comment_id=target_comment_id,
                        comment_time=str(item.get("time", "") or candidate.post_time),
                        comment_author_id=str(item.get("author_id", "") or ""),
                        comment_author_name=str(item.get("author", "") or ""),
                        comment_text=str(item.get("text", "") or ""),
                    )
        base = by_post.get(target_post_id) if target_post_id else None
        if base is None and candidates:
            base = candidates[0]
        if base is None:
            return None
        if action == "comment1" or not target_comment_id:
            return replace(base, action_hint="comment1", comment_id=None, comment_time=None, comment_author_id=None, comment_author_name=None, comment_text=None)
        return replace(base, action_hint="comment1", comment_id=None, comment_time=None, comment_author_id=None, comment_author_name=None, comment_text=None)

    async def _generate_for_decision(self, decision: Decision, prompt: str):
        settings_override = dict(decision.persona.llm_profile())
        settings_override.update(
            {
                "family": decision.persona.family,
                "family_generation": decision.persona.family_generation,
                "family_round": decision.persona.family_round,
                "graph_affinity": decision.persona.graph_affinity,
                "notification_reactivity": decision.persona.notification_reactivity,
            }
        )
        return await self.llm.generate(prompt, fallback_content(decision), settings_override=settings_override)

    def apply_decision(self, decision: Decision) -> None:
        if decision.action == "post":
            self._write_post(decision)
        elif decision.action == "comment1":
            self._write_comment1(decision)
        elif decision.action in {"comment2", "revisit"}:
            self._write_comment2(decision)

    def _write_post(self, decision: Decision) -> None:
        data = self.env.get_env_reddit_data()
        existing = list(data.posts.keys())
        post_id = generate_reddit_id(existing)
        persona = decision.persona
        subreddit = self.rng.choice(persona.subreddits)
        post = RedditPost(
            submission_id=post_id,
            author_id=persona.user_id,
            author_name=persona.user_name,
            posts=decision.content or fallback_content(decision),
            score=0,
            num_comments=0,
            upvote_ratio=0.0,
            time=next_post_time(decision.planned_time, self.rng, persona),
            subreddit=subreddit,
        )
        data.posts[post_id] = post
        data.users[persona.user_id].add_user_post_by_post(post)
        post.save_to_file(self.config["paths"]["reddit_posts"])
        self.memories[persona.user_id].own_posts.append(post_id)
        self.summary["post_written"] += 1
        decision.result = f"Post Success! The Post ID is {post_id}"
        self.memories[persona.user_id].record_action("post")
        self.memory_trace.write("post_written", user_id=persona.user_id, post_id=post_id, subreddit=subreddit, content=post.posts)

    def _write_comment1(self, decision: Decision) -> None:
        cand = decision.candidate
        if cand is None:
            return
        data = self.env.get_env_reddit_data()
        comment_id = generate_reddit_id([c.comment1_id for c in data.comments1])
        comment_time = next_comment_time(cand.post_time, self.rng, decision.persona)
        comment = RedditComment1(
            comment1_id=comment_id,
            comment1_author_name=decision.persona.user_name,
            comment1_author_id=decision.persona.user_id,
            comment1_score=0,
            comment1_body=decision.content or fallback_content(decision),
            link_id=f"t3_{cand.post_id}",
            parent_id=f"t3_{cand.post_id}",
            subreddit=cand.subreddit,
            comment1_time=comment_time,
            level="1",
        )
        data.comments1.append(comment)
        if cand.post_id in data.posts:
            data.posts[cand.post_id].add_comment1(comment)
        data.users[decision.persona.user_id].add_user_comment_by_commet1(comment)
        comment.save_to_file(self.config["paths"]["reddit_comments1"])
        memory = self.memories[decision.persona.user_id]
        memory.own_comment1.append(comment_id)
        memory.remember_author(cand.post_author_id)
        memory.record_action("comment1", cand)
        self._maybe_notify_post_owner(cand.post_author_id, decision.persona.user_id, cand.post_id, comment_id)
        self.summary["comment1_written"] += 1
        decision.result = "Comment1 Success"
        self.memory_trace.write("comment1_written", user_id=decision.persona.user_id, post_id=cand.post_id, comment_id=comment_id, target_author=cand.post_author_id)

    def _write_comment2(self, decision: Decision) -> None:
        cand = decision.candidate
        if cand is None or cand.comment_id is None:
            return
        data = self.env.get_env_reddit_data()
        comment_id = generate_reddit_id([c.comment2_id for c in data.comments2])
        base_time = cand.comment_time or cand.post_time
        comment_time = next_comment_time(base_time, self.rng, decision.persona)
        comment = RedditComment2(
            comment2_id=comment_id,
            comment2_author_name=decision.persona.user_name,
            comment2_author_id=decision.persona.user_id,
            comment2_score=0,
            comment2_body=decision.content or fallback_content(decision),
            link_id=f"t3_{cand.post_id}",
            parent_id=f"t1_{cand.comment_id}",
            subreddit=cand.subreddit,
            comment2_time=comment_time,
            level="2",
        )
        data.comments2.append(comment)
        if cand.post_id in data.posts:
            data.posts[cand.post_id].add_comment2(comment)
        data.users[decision.persona.user_id].add_user_comment_by_commet2(comment)
        comment.save_to_file(self.config["paths"]["reddit_comments2"])
        memory = self.memories[decision.persona.user_id]
        memory.own_comment2.append(comment_id)
        memory.remember_author(cand.comment_author_id or cand.post_author_id)
        memory.record_action(decision.action, cand)
        self._maybe_notify_comment_owner(cand.comment_author_id, decision.persona.user_id, cand.post_id, cand.comment_id, comment_id)
        if decision.action == "revisit":
            memory.revisited_notifications.add(cand.candidate_id.replace("notification:", ""))
            for note in memory.notifications:
                if note.get("notification_id") == cand.candidate_id.replace("notification:", ""):
                    note["resolved"] = True
                    break
            self.summary["revisit_written"] += 1
        else:
            self.summary["comment2_written"] += 1
        decision.result = "Comment2 Success"
        self.memory_trace.write(
            "comment2_written",
            user_id=decision.persona.user_id,
            post_id=cand.post_id,
            parent_comment_id=cand.comment_id,
            comment_id=comment_id,
            target_author=cand.comment_author_id,
            action=decision.action,
        )

    def _maybe_notify_post_owner(self, owner_id: str, actor_id: str, post_id: str, comment_id: str) -> bool:
        if owner_id not in self.memories:
            return False
        if owner_id == actor_id:
            return False
        notification_id = f"post:{post_id}:comment:{comment_id}"
        self.memories[owner_id].notifications.append(
            {
                "notification_id": notification_id,
                "type": "post_reply",
                "post_id": post_id,
                "comment_id": comment_id,
                "resolved": False,
            }
        )
        self.summary["notifications_created"] += 1
        return True

    def _maybe_notify_comment_owner(self, owner_id: str | None, actor_id: str, post_id: str, parent_comment_id: str, reply_comment_id: str) -> bool:
        if owner_id not in self.memories:
            return False
        if owner_id == actor_id:
            return False
        notification_id = f"comment:{parent_comment_id}:reply:{reply_comment_id}"
        self.memories[owner_id].notifications.append(
            {
                "notification_id": notification_id,
                "type": "comment_reply",
                "post_id": post_id,
                "comment_id": parent_comment_id,
                "reply_comment_id": reply_comment_id,
                "resolved": False,
            }
        )
        self.summary["notifications_created"] += 1
        return True


def write_summary(run_dir: Path, summary: dict[str, Any], config: dict[str, Any], personas: list[Persona], dry_run: bool, api: str) -> None:
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    with (run_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Persona-Aware RedditBotSim v2 run\n\n")
        f.write(f"- dry_run: `{dry_run}`\n")
        f.write(f"- api: `{api}`\n")
        f.write(f"- agents_num: `{len(personas)}`\n")
        f.write(f"- browse_count: `{summary.get('browse_count')}`\n")
        f.write(f"- decision_mode: `{summary.get('decision_mode', 'system_first')}`\n")
        f.write("- trace files: `decision_trace.jsonl`, `llm_trace.jsonl`, `memory_trace.jsonl`\n")
        f.write("- generated CSV files are isolated under `data/`.\n")


async def async_main(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    run_dir = Path(args.run_dir) if args.run_dir else PROJECT_ROOT / "outputs" / "persona_social"
    if args.clean and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    family_manifest_bundle = None
    if args.family_manifest:
        personas = _load_personas_from_manifest(Path(args.family_manifest))
    elif args.family:
        model_pool = json.loads(Path(args.model_pool).read_text(encoding="utf-8")) if args.model_pool else sample_model_pool(random.Random(args.seed))
        family_manifest_bundle = build_family_manifest_bundle(
            args.family,
            args.family_generation,
            args.max_agents,
            seed=args.seed,
            model_pool=model_pool,
        )
        personas = [_coerce_persona(item) for item in family_manifest_bundle.personas]
    else:
        personas = [_coerce_persona(item) for item in load_default_personas(args.max_agents)]
    config = prepare_run_config(Path(args.config), run_dir, personas)
    logger = set_logger(config, name="persona_social_runner")
    logger.info("Persona social runner started")

    if args.dry_run:
        client = DryRunClient()
        settings = load_llm_settings(
            {"llm": {"api_key_env": "OPENAI_API_KEY"}},
            provider_override=args.provider,
            model_override=args.model,
            temperature_override=args.temperature,
        ) if (REDDIT_BOT_SIM_DIR / ".env.disabled").exists() or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") else None
        model = settings.model if settings else "dry-run"
        temperature = settings.temperature if settings else 0.0
        reasoning_effort = settings.reasoning_effort if settings else "none"
    else:
        client, settings = build_llm_client(
            config,
            allowed_providers={str(getattr(p, "llm_provider", None) or args.provider or "openai") for p in personas} | {args.provider or "openai"},
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
        )
        model = settings.model
        temperature = settings.temperature
        reasoning_effort = settings.reasoning_effort
    if args.model:
        model = args.model
    if args.reasoning_effort:
        reasoning_effort = args.reasoning_effort
    if args.temperature is not None:
        temperature = args.temperature

    llm = AsyncLLMGenerator(
        client=client,
        model=model,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        api=args.api,
        concurrency=args.concurrency,
        retries=args.retries,
        dry_run=args.dry_run,
        timeout=args.llm_timeout,
    )
    sim = PersonaSocialSimulation(config, personas, run_dir, llm, rng, args.browse_count, args.rounds, decision_mode=args.decision_mode)
    summary = await sim.run()
    bot_ids = {p.user_id for p in personas}
    comments = export_generated_comments(run_dir, run_dir.parent / "generated_comments.csv", bot_ids)
    posts = export_generated_posts(run_dir, run_dir.parent / "generated_posts.csv", bot_ids)
    per_user, behavior_summary = build_behavior_audit(run_dir, bot_ids)
    graph_proxy = graph_proxy_audit(run_dir, bot_ids)
    per_user.to_csv(run_dir.parent / "behavior_audit_by_user.csv", index=False)
    graph_proxy.to_csv(run_dir.parent / "graph_proxy_audit.csv", index=False)
    build_family_manifest(personas).to_csv(run_dir.parent / "family_manifest.csv", index=False)
    if family_manifest_bundle is not None:
        (run_dir.parent / "family_manifest.json").write_text(family_manifest_bundle.to_json(), encoding="utf-8")
    shutil.copy2(run_dir / "generated_persona_profiles.csv", run_dir.parent / "generated_persona_profiles.csv")
    provider_counts = Counter((p.llm_provider or (settings.provider if settings else args.provider or "dry-run")) for p in personas)
    model_counts = Counter((p.llm_model or model) for p in personas)
    temperature_counts = Counter(str(p.llm_temperature if p.llm_temperature is not None else temperature) for p in personas)
    provider_summary = next(iter(provider_counts)) if len(provider_counts) == 1 else "mixed_per_persona"
    model_summary = next(iter(model_counts)) if len(model_counts) == 1 else "mixed_per_persona"
    full_summary = {
        **summary,
        "dry_run": args.dry_run,
        "api": args.api,
        "provider": provider_summary,
        "model": model_summary,
        "provider_counts": dict(provider_counts),
        "model_counts": dict(model_counts),
        "temperature_counts": dict(temperature_counts),
        "reasoning_effort": reasoning_effort,
        "browse_count": args.browse_count,
        "rounds": args.rounds,
        "decision_mode": args.decision_mode,
        "generated_post_rows": int(len(posts)),
        "generated_comment_rows": int(len(comments)),
        "behavior_summary": behavior_summary,
        "run_dir": str(run_dir),
    }
    (run_dir.parent / "persona_social_summary.json").write_text(json.dumps(full_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(run_dir, full_summary, config, personas, args.dry_run, args.api)
    print(json.dumps(full_summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run persona-aware RedditBotSim v2 social simulation.")
    parser.add_argument("--config", default=str(REDDIT_BOT_SIM_DIR / "config.yml"))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-agents", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--browse-count", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--llm-timeout", type=float, default=90.0)
    parser.add_argument("--api", choices=["responses", "chat"], default="responses")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--family", default=None)
    parser.add_argument("--family-generation", type=int, default=0)
    parser.add_argument("--family-manifest", default=None)
    parser.add_argument("--model-pool", default=None)
    parser.add_argument("--decision-mode", choices=["system_first", "llm_first"], default="system_first")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
