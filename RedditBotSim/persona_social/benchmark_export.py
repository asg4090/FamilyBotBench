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
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


BENCHMARK_SCHEMA_VERSION = "phase5-v1"

TEXT_SAMPLE_COLUMNS = [
    "benchmark_round_id",
    "round_index",
    "family",
    "family_generation",
    "family_round",
    "label",
    "origin_kind",
    "source_table",
    "sample_type",
    "sample_id",
    "text_id",
    "user_id",
    "user_name",
    "provider",
    "model",
    "temperature",
    "reasoning_effort",
    "subreddit",
    "topic",
    "post_id",
    "parent_text_id",
    "parent_text_type",
    "root_post_id",
    "created_utc",
    "text_body",
    "text_length",
    "word_count",
]

USER_SAMPLE_COLUMNS = [
    "benchmark_round_id",
    "round_index",
    "family",
    "family_slug",
    "family_generation",
    "family_round",
    "label",
    "source_kind",
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
    "provider",
    "model",
    "temperature",
    "reasoning_effort",
    "graph_affinity",
    "notification_reactivity",
    "submission_num",
    "comment_num",
    "comment_num_1",
    "comment_num_2",
    "post_count",
    "comment1_count",
    "comment2_count",
    "total_text_count",
]

EDGE_COLUMNS = [
    "benchmark_round_id",
    "round_index",
    "family",
    "family_generation",
    "family_round",
    "edge_id",
    "edge_type",
    "edge_level",
    "source_text_id",
    "source_text_type",
    "source_user_id",
    "source_user_name",
    "source_label",
    "source_family",
    "source_family_generation",
    "target_text_id",
    "target_text_type",
    "target_user_id",
    "target_user_name",
    "target_label",
    "target_family",
    "target_family_generation",
    "root_post_id",
    "subreddit",
    "created_utc",
    "source_kind",
]

FAMILY_MAP_COLUMNS = [
    "benchmark_round_id",
    "round_index",
    "family",
    "family_slug",
    "family_generation",
    "family_round",
    "label",
    "user_id",
    "user_name",
    "source_kind",
    "provider",
    "model",
    "temperature",
    "reasoning_effort",
    "graph_affinity",
    "notification_reactivity",
    "post_tendency",
    "reply_tendency",
    "revisit_tendency",
    "exploration_tendency",
    "brevity",
]

ROUND_MAP_COLUMNS = [
    "benchmark_round_id",
    "round_index",
    "family",
    "family_generation",
    "family_round",
    "run_key",
    "run_dir",
    "benchmark_dir",
    "simulation_rounds",
    "browse_count",
    "agents_num",
    "seed",
    "provider",
    "model",
    "temperature",
    "reasoning_effort",
    "dry_run",
    "start_time",
    "end_time",
    "duration_seconds",
    "bot_count",
    "human_count",
    "text_sample_count",
    "user_sample_count",
    "edge_count",
    "source_count",
    "summary_json",
    "feedback_json",
    "family_manifest_path",
]

SOURCE_MAP_COLUMNS = [
    "benchmark_round_id",
    "round_index",
    "family",
    "family_generation",
    "family_round",
    "source_id",
    "source_kind",
    "record_domain",
    "source_file",
    "exists",
    "row_count",
    "description",
]

INDEX_COLUMNS = [
    "record_type",
    "record_id",
    "benchmark_round_id",
    "round_index",
    "family",
    "family_generation",
    "family_round",
    "user_id",
    "user_name",
    "label",
    "source_kind",
    "text",
    "edge_type",
    "path",
]


def _json_default(value: Any) -> str:
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _stable_slug(value: str) -> str:
    cleaned = [ch.lower() if ch.isalnum() else "_" for ch in value]
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "benchmark"


def _stable_id(prefix: str, *parts: Any, length: int = 12) -> str:
    payload = "|".join(_safe_text(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _first_present(row: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in row:
            value = row[key]
            if value not in (None, ""):
                return value
    return default


def _word_count(text: Any) -> int:
    return len(_safe_text(text).split())


def _normalize_profile_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    out = {
        "user_id": _safe_text(_first_present(payload, "user_id", "author_id")),
        "user_name": _safe_text(_first_present(payload, "user_name", "author_name")),
        "age": _safe_int(_first_present(payload, "Age", "age")),
        "gender": _safe_text(_first_present(payload, "Gender", "gender")),
        "region": _safe_text(_first_present(payload, "Region", "region")),
        "education": _safe_text(_first_present(payload, "EducationLevel", "education")),
        "ideology": _safe_text(_first_present(payload, "Ideology", "ideology")),
        "description": _safe_text(_first_present(payload, "Description", "description")),
        "subreddits": _safe_text(_first_present(payload, "SubRedditList", "subreddits", default="[]")),
        "interest_weights": _safe_text(_first_present(payload, "interest_weights", default="{}")),
        "style": _safe_text(_first_present(payload, "style", default="")),
        "stance": _safe_text(_first_present(payload, "stance", default="")),
        "debate_tendency": _safe_float(_first_present(payload, "debate_tendency", default=0.0)),
        "post_tendency": _safe_float(_first_present(payload, "post_tendency", default=0.0)),
        "reply_tendency": _safe_float(_first_present(payload, "reply_tendency", default=0.0)),
        "revisit_tendency": _safe_float(_first_present(payload, "revisit_tendency", default=0.0)),
        "exploration_tendency": _safe_float(_first_present(payload, "exploration_tendency", default=0.0)),
        "brevity": _safe_float(_first_present(payload, "brevity", default=0.0)),
        "family": _safe_text(_first_present(payload, "family", default="human")),
        "family_slug": _safe_text(_first_present(payload, "family_slug", default="human")),
        "family_generation": _safe_int(_first_present(payload, "family_generation", default=0)),
        "family_round": _safe_int(_first_present(payload, "family_round", default=0)),
        "provider": _safe_text(_first_present(payload, "llm_provider", "provider")),
        "model": _safe_text(_first_present(payload, "llm_model", "model")),
        "temperature": _safe_float(_first_present(payload, "llm_temperature", "temperature", default="")),
        "reasoning_effort": _safe_text(_first_present(payload, "llm_reasoning_effort", "reasoning_effort")),
        "graph_affinity": _safe_float(_first_present(payload, "graph_affinity", default=0.0)),
        "notification_reactivity": _safe_float(_first_present(payload, "notification_reactivity", default=0.0)),
        "submission_num": _safe_int(_first_present(payload, "submission_num", "PostNum", default=0)),
        "comment_num": _safe_int(_first_present(payload, "comment_num", "CommentNum", default=0)),
        "comment_num_1": _safe_int(_first_present(payload, "comment_num_1", "Comment1Num", default=0)),
        "comment_num_2": _safe_int(_first_present(payload, "comment_num_2", "Comment2Num", default=0)),
        "source_kind": _safe_text(_first_present(payload, "source_kind", default="profile")),
    }
    return out


def _load_profile_map(run_dir: Path, family_manifest_path: Path | None = None) -> dict[str, dict[str, Any]]:
    profile_map: dict[str, dict[str, Any]] = {}
    generated_profiles = _read_csv(run_dir / "generated_persona_profiles.csv")
    if not generated_profiles.empty:
        for _, row in generated_profiles.iterrows():
            payload = _normalize_profile_row(row.to_dict())
            if payload["user_id"]:
                payload["family"] = payload["family"] or "bot"
                payload["family_slug"] = payload["family_slug"] or _stable_slug(payload["family"])
                profile_map[payload["user_id"]] = payload

    manifest_sources = []
    if family_manifest_path and family_manifest_path.exists():
        manifest_sources.append(family_manifest_path)
    for candidate in [run_dir.parent / "family_manifest.json", run_dir / "family_manifest.json"]:
        if candidate.exists() and candidate not in manifest_sources:
            manifest_sources.append(candidate)
    for manifest_path in manifest_sources:
        payload = _read_json(manifest_path)
        for person in payload.get("personas", []) if isinstance(payload, dict) else []:
            if not isinstance(person, dict):
                continue
            normalized = _normalize_profile_row(person)
            if normalized["user_id"]:
                normalized["family"] = normalized["family"] or _safe_text(payload.get("family", "bot"))
                normalized["family_slug"] = normalized["family_slug"] or _stable_slug(normalized["family"])
                profile_map[normalized["user_id"]] = normalized
    return profile_map


def _load_base_users(run_dir: Path) -> pd.DataFrame:
    user_path = run_dir / "data" / "User.csv"
    if not user_path.exists():
        user_path = run_dir / "data" / "Users.csv"
    return _read_csv(user_path)


def _load_posts_comments(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    posts = _read_csv(run_dir / "data" / "Posts.csv")
    comments1 = _read_csv(run_dir / "data" / "Comment_1.csv")
    comments2 = _read_csv(run_dir / "data" / "Comment_2.csv")
    return posts, comments1, comments2


def _resolve_round_meta(round_meta: Mapping[str, Any] | None, run_dir: Path) -> dict[str, Any]:
    payload = dict(round_meta or {})
    family = _safe_text(payload.get("family", "human")) or "human"
    generation = _safe_int(payload.get("family_generation", payload.get("generation", 0)))
    family_round = _safe_int(payload.get("family_round", generation))
    benchmark_round_id = _safe_text(
        payload.get(
            "benchmark_round_id",
            payload.get("round_id", f"{_stable_slug(family)}_g{generation:03d}"),
        )
    )
    payload.setdefault("benchmark_round_id", benchmark_round_id)
    payload.setdefault("round_index", _safe_int(payload.get("round_index", generation)))
    payload.setdefault("family", family)
    payload.setdefault("family_slug", _safe_text(payload.get("family_slug", _stable_slug(family))))
    payload.setdefault("family_generation", generation)
    payload.setdefault("family_round", family_round)
    payload.setdefault("run_key", _safe_text(payload.get("run_key", run_dir.name or benchmark_round_id)))
    payload.setdefault("run_dir", str(run_dir.resolve()))
    payload.setdefault("benchmark_dir", _safe_text(payload.get("benchmark_dir", "")))
    payload.setdefault("simulation_rounds", _safe_int(payload.get("simulation_rounds", payload.get("rounds", 0))))
    payload.setdefault("browse_count", _safe_int(payload.get("browse_count", 0)))
    payload.setdefault("agents_num", _safe_int(payload.get("agents_num", 0)))
    payload.setdefault("seed", _safe_int(payload.get("seed", 0)))
    payload.setdefault("provider", _safe_text(payload.get("provider", "")))
    payload.setdefault("model", _safe_text(payload.get("model", "")))
    payload.setdefault("temperature", _safe_float(payload.get("temperature", 0.0)))
    payload.setdefault("reasoning_effort", _safe_text(payload.get("reasoning_effort", "")))
    payload.setdefault("dry_run", bool(payload.get("dry_run", False)))
    payload.setdefault("start_time", _safe_text(payload.get("start_time", "")))
    payload.setdefault("end_time", _safe_text(payload.get("end_time", "")))
    payload.setdefault("duration_seconds", _safe_float(payload.get("duration_seconds", 0.0)))
    payload.setdefault("summary", dict(payload.get("summary", {})) if isinstance(payload.get("summary", {}), Mapping) else {})
    payload.setdefault("feedback", dict(payload.get("feedback", {})) if isinstance(payload.get("feedback", {}), Mapping) else {})
    payload.setdefault("family_manifest_path", _safe_text(payload.get("family_manifest_path", "")))
    return payload


def _profile_record(profile_map: Mapping[str, dict[str, Any]], user_id: str) -> dict[str, Any]:
    profile = profile_map.get(user_id)
    if profile:
        return dict(profile)
    return {
        "user_id": user_id,
        "user_name": "",
        "age": 0,
        "gender": "",
        "region": "",
        "education": "",
        "ideology": "",
        "description": "",
        "subreddits": "[]",
        "interest_weights": "{}",
        "style": "",
        "stance": "",
        "debate_tendency": 0.0,
        "post_tendency": 0.0,
        "reply_tendency": 0.0,
        "revisit_tendency": 0.0,
        "exploration_tendency": 0.0,
        "brevity": 0.0,
        "family": "human",
        "family_slug": "human",
        "family_generation": 0,
        "family_round": 0,
        "provider": "",
        "model": "",
        "temperature": 0.0,
        "reasoning_effort": "",
        "graph_affinity": 0.0,
        "notification_reactivity": 0.0,
        "submission_num": 0,
        "comment_num": 0,
        "comment_num_1": 0,
        "comment_num_2": 0,
        "source_kind": "base_user",
    }


def _text_origin(profile_map: Mapping[str, dict[str, Any]], user_id: str) -> tuple[str, dict[str, Any]]:
    profile = _profile_record(profile_map, user_id)
    label = "bot" if profile.get("family", "human") not in {"", "human", "baseline"} else "human"
    return label, profile


def _build_text_samples(
    run_dir: Path,
    round_meta: Mapping[str, Any],
    profile_map: Mapping[str, dict[str, Any]],
) -> pd.DataFrame:
    posts, comments1, comments2 = _load_posts_comments(run_dir)
    rows: list[dict[str, Any]] = []

    def add_row(
        *,
        sample_type: str,
        source_table: str,
        text_id: str,
        text_body: str,
        user_id: str,
        user_name: str,
        created_utc: str,
        subreddit: str,
        parent_text_id: str = "",
        parent_text_type: str = "",
        root_post_id: str = "",
        post_id: str = "",
        topic: str = "",
    ) -> None:
        label, profile = _text_origin(profile_map, user_id)
        row = {
            "benchmark_round_id": round_meta["benchmark_round_id"],
            "round_index": round_meta["round_index"],
            "family": profile.get("family", "human") if label == "bot" else "human",
            "family_generation": _safe_int(profile.get("family_generation", 0)) if label == "bot" else 0,
            "family_round": _safe_int(profile.get("family_round", 0)) if label == "bot" else 0,
            "label": label,
            "origin_kind": profile.get("source_kind", "base_user"),
            "source_table": source_table,
            "sample_type": sample_type,
            "sample_id": f"{sample_type}:{text_id}",
            "text_id": text_id,
            "user_id": user_id,
            "user_name": user_name,
            "provider": profile.get("provider", ""),
            "model": profile.get("model", ""),
            "temperature": _safe_float(profile.get("temperature", 0.0)),
            "reasoning_effort": profile.get("reasoning_effort", ""),
            "subreddit": subreddit,
            "topic": topic,
            "post_id": post_id or (text_id if sample_type == "post" else root_post_id),
            "parent_text_id": parent_text_id,
            "parent_text_type": parent_text_type,
            "root_post_id": root_post_id or (text_id if sample_type == "post" else ""),
            "created_utc": created_utc,
            "text_body": text_body,
            "text_length": len(_safe_text(text_body)),
            "word_count": _word_count(text_body),
        }
        rows.append(row)

    for _, row in posts.iterrows():
        payload = row.to_dict()
        post_id = _safe_text(_first_present(payload, "submission_id", "post_id"))
        user_id = _safe_text(_first_present(payload, "author_id", "user_id"))
        user_name = _safe_text(_first_present(payload, "author_name", "user_name"))
        add_row(
            sample_type="post",
            source_table="Posts.csv",
            text_id=post_id,
            text_body=_safe_text(_first_present(payload, "posts", "post_body", "body")),
            user_id=user_id,
            user_name=user_name,
            created_utc=_safe_text(_first_present(payload, "time", "created_utc")),
            subreddit=_safe_text(_first_present(payload, "subreddit")),
            parent_text_id="",
            parent_text_type="",
            root_post_id=post_id,
            post_id=post_id,
            topic=_safe_text(_first_present(payload, "topic")),
        )

    for _, row in comments1.iterrows():
        payload = row.to_dict()
        comment_id = _safe_text(_first_present(payload, "comment_id", "comment1_id"))
        post_id = _safe_text(_first_present(payload, "link_id")).replace("t3_", "")
        parent_text_id = f"post:{post_id}"
        add_row(
            sample_type="comment1",
            source_table="Comment_1.csv",
            text_id=comment_id,
            text_body=_safe_text(_first_present(payload, "comment_body", "body")),
            user_id=_safe_text(_first_present(payload, "comment_author_id", "author_id")),
            user_name=_safe_text(_first_present(payload, "comment_author_name", "author_name")),
            created_utc=_safe_text(_first_present(payload, "created_utc", "comment1_time", "time")),
            subreddit=_safe_text(_first_present(payload, "subreddit")),
            parent_text_id=parent_text_id,
            parent_text_type="post",
            root_post_id=post_id,
            post_id=post_id,
            topic=_safe_text(_first_present(payload, "topic")),
        )

    for _, row in comments2.iterrows():
        payload = row.to_dict()
        comment_id = _safe_text(_first_present(payload, "comment_id", "comment2_id"))
        post_id = _safe_text(_first_present(payload, "link_id")).replace("t3_", "")
        parent_comment_id = _safe_text(_first_present(payload, "parent_id")).replace("t1_", "")
        parent_text_id = f"comment1:{parent_comment_id}"
        add_row(
            sample_type="comment2",
            source_table="Comment_2.csv",
            text_id=comment_id,
            text_body=_safe_text(_first_present(payload, "comment_body", "body")),
            user_id=_safe_text(_first_present(payload, "comment_author_id", "author_id")),
            user_name=_safe_text(_first_present(payload, "comment_author_name", "author_name")),
            created_utc=_safe_text(_first_present(payload, "created_utc", "comment2_time", "time")),
            subreddit=_safe_text(_first_present(payload, "subreddit")),
            parent_text_id=parent_text_id,
            parent_text_type="comment1",
            root_post_id=post_id,
            post_id=post_id,
            topic=_safe_text(_first_present(payload, "topic")),
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["created_utc", "sample_type", "sample_id"], kind="stable")
    return df.reindex(columns=TEXT_SAMPLE_COLUMNS)


def _build_user_samples(
    run_dir: Path,
    round_meta: Mapping[str, Any],
    profile_map: Mapping[str, dict[str, Any]],
) -> pd.DataFrame:
    base_users = _load_base_users(run_dir)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    if not base_users.empty:
        for _, row in base_users.iterrows():
            payload = row.to_dict()
            user_id = _safe_text(_first_present(payload, "author_id", "user_id"))
            if not user_id:
                continue
            profile = _profile_record(profile_map, user_id)
            label = "bot" if profile.get("family", "human") not in {"", "human", "baseline"} else "human"
            seen.add(user_id)
            rows.append(
                {
                    "benchmark_round_id": round_meta["benchmark_round_id"],
                    "round_index": round_meta["round_index"],
                    "family": profile.get("family", "human") if label == "bot" else "human",
                    "family_slug": profile.get("family_slug", "human") if label == "bot" else "human",
                    "family_generation": _safe_int(profile.get("family_generation", 0)) if label == "bot" else 0,
                    "family_round": _safe_int(profile.get("family_round", 0)) if label == "bot" else 0,
                    "label": label,
                    "source_kind": profile.get("source_kind", "base_user"),
                    "user_id": user_id,
                    "user_name": _safe_text(_first_present(payload, "author_name", "user_name", default=profile.get("user_name", ""))),
                    "age": profile.get("age", 0),
                    "gender": profile.get("gender", ""),
                    "region": profile.get("region", ""),
                    "education": profile.get("education", ""),
                    "ideology": profile.get("ideology", ""),
                    "description": profile.get("description", _safe_text(_first_present(payload, "description", default=""))),
                    "subreddits": profile.get("subreddits", "[]"),
                    "interest_weights": profile.get("interest_weights", "{}"),
                    "style": profile.get("style", ""),
                    "stance": profile.get("stance", ""),
                    "debate_tendency": profile.get("debate_tendency", 0.0),
                    "post_tendency": profile.get("post_tendency", 0.0),
                    "reply_tendency": profile.get("reply_tendency", 0.0),
                    "revisit_tendency": profile.get("revisit_tendency", 0.0),
                    "exploration_tendency": profile.get("exploration_tendency", 0.0),
                    "brevity": profile.get("brevity", 0.0),
                    "provider": profile.get("provider", ""),
                    "model": profile.get("model", ""),
                    "temperature": profile.get("temperature", 0.0),
                    "reasoning_effort": profile.get("reasoning_effort", ""),
                    "graph_affinity": profile.get("graph_affinity", 0.0),
                    "notification_reactivity": profile.get("notification_reactivity", 0.0),
                    "submission_num": _safe_int(_first_present(payload, "submission_num", default=profile.get("submission_num", 0))),
                    "comment_num": _safe_int(_first_present(payload, "comment_num", default=profile.get("comment_num", 0))),
                    "comment_num_1": _safe_int(_first_present(payload, "comment_num_1", default=profile.get("comment_num_1", 0))),
                    "comment_num_2": _safe_int(_first_present(payload, "comment_num_2", default=profile.get("comment_num_2", 0))),
                    "post_count": _safe_int(_first_present(payload, "submission_num", default=profile.get("submission_num", 0))),
                    "comment1_count": _safe_int(_first_present(payload, "comment_num_1", default=profile.get("comment_num_1", 0))),
                    "comment2_count": _safe_int(_first_present(payload, "comment_num_2", default=profile.get("comment_num_2", 0))),
                    "total_text_count": _safe_int(_first_present(payload, "comment_num", default=profile.get("comment_num", 0))) + _safe_int(_first_present(payload, "submission_num", default=profile.get("submission_num", 0))),
                }
            )

    for user_id, profile in profile_map.items():
        if user_id in seen:
            continue
        rows.append(
            {
                "benchmark_round_id": round_meta["benchmark_round_id"],
                "round_index": round_meta["round_index"],
                "family": profile.get("family", "human"),
                "family_slug": profile.get("family_slug", "human"),
                "family_generation": _safe_int(profile.get("family_generation", 0)),
                "family_round": _safe_int(profile.get("family_round", 0)),
                "label": "bot" if profile.get("family", "human") not in {"", "human", "baseline"} else "human",
                "source_kind": profile.get("source_kind", "profile"),
                "user_id": user_id,
                "user_name": profile.get("user_name", ""),
                "age": profile.get("age", 0),
                "gender": profile.get("gender", ""),
                "region": profile.get("region", ""),
                "education": profile.get("education", ""),
                "ideology": profile.get("ideology", ""),
                "description": profile.get("description", ""),
                "subreddits": profile.get("subreddits", "[]"),
                "interest_weights": profile.get("interest_weights", "{}"),
                "style": profile.get("style", ""),
                "stance": profile.get("stance", ""),
                "debate_tendency": profile.get("debate_tendency", 0.0),
                "post_tendency": profile.get("post_tendency", 0.0),
                "reply_tendency": profile.get("reply_tendency", 0.0),
                "revisit_tendency": profile.get("revisit_tendency", 0.0),
                "exploration_tendency": profile.get("exploration_tendency", 0.0),
                "brevity": profile.get("brevity", 0.0),
                "provider": profile.get("provider", ""),
                "model": profile.get("model", ""),
                "temperature": profile.get("temperature", 0.0),
                "reasoning_effort": profile.get("reasoning_effort", ""),
                "graph_affinity": profile.get("graph_affinity", 0.0),
                "notification_reactivity": profile.get("notification_reactivity", 0.0),
                "submission_num": profile.get("submission_num", 0),
                "comment_num": profile.get("comment_num", 0),
                "comment_num_1": profile.get("comment_num_1", 0),
                "comment_num_2": profile.get("comment_num_2", 0),
                "post_count": profile.get("submission_num", 0),
                "comment1_count": profile.get("comment_num_1", 0),
                "comment2_count": profile.get("comment_num_2", 0),
                "total_text_count": _safe_int(profile.get("submission_num", 0)) + _safe_int(profile.get("comment_num", 0)),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["family", "family_generation", "user_id"], kind="stable")
    return df.reindex(columns=USER_SAMPLE_COLUMNS)


def _build_interaction_edges(
    run_dir: Path,
    round_meta: Mapping[str, Any],
    profile_map: Mapping[str, dict[str, Any]],
) -> pd.DataFrame:
    posts, comments1, comments2 = _load_posts_comments(run_dir)
    post_author = {
        _safe_text(_first_present(row.to_dict(), "submission_id", "post_id")): _safe_text(_first_present(row.to_dict(), "author_id", "user_id"))
        for _, row in posts.iterrows()
    }
    post_author_name = {
        _safe_text(_first_present(row.to_dict(), "submission_id", "post_id")): _safe_text(_first_present(row.to_dict(), "author_name", "user_name"))
        for _, row in posts.iterrows()
    }
    c1_author = {
        _safe_text(_first_present(row.to_dict(), "comment_id", "comment1_id")): _safe_text(_first_present(row.to_dict(), "comment_author_id", "author_id"))
        for _, row in comments1.iterrows()
    }
    c1_author_name = {
        _safe_text(_first_present(row.to_dict(), "comment_id", "comment1_id")): _safe_text(_first_present(row.to_dict(), "comment_author_name", "author_name"))
        for _, row in comments1.iterrows()
    }
    rows: list[dict[str, Any]] = []

    def profile_bits(user_id: str) -> dict[str, Any]:
        profile = _profile_record(profile_map, user_id)
        label = "bot" if profile.get("family", "human") not in {"", "human", "baseline"} else "human"
        return {
            "label": label,
            "family": profile.get("family", "human") if label == "bot" else "human",
            "family_generation": _safe_int(profile.get("family_generation", 0)) if label == "bot" else 0,
        }

    for _, row in comments1.iterrows():
        payload = row.to_dict()
        comment_id = _safe_text(_first_present(payload, "comment_id", "comment1_id"))
        post_id = _safe_text(_first_present(payload, "link_id")).replace("t3_", "")
        source_user_id = _safe_text(_first_present(payload, "comment_author_id", "author_id"))
        source_user_name = _safe_text(_first_present(payload, "comment_author_name", "author_name"))
        target_user_id = _safe_text(post_author.get(post_id, ""))
        target_user_name = _safe_text(post_author_name.get(post_id, ""))
        source_bits = profile_bits(source_user_id)
        target_bits = profile_bits(target_user_id)
        rows.append(
            {
                "benchmark_round_id": round_meta["benchmark_round_id"],
                "round_index": round_meta["round_index"],
                "family": source_bits["family"],
                "family_generation": source_bits["family_generation"],
                "family_round": _safe_int(profile_map.get(source_user_id, {}).get("family_round", 0)) if source_bits["label"] == "bot" else 0,
                "edge_id": _stable_id("e1_", round_meta["benchmark_round_id"], comment_id, post_id, source_user_id, target_user_id),
                "edge_type": "reply_to_post",
                "edge_level": 1,
                "source_text_id": f"comment1:{comment_id}",
                "source_text_type": "comment1",
                "source_user_id": source_user_id,
                "source_user_name": source_user_name,
                "source_label": source_bits["label"],
                "source_family": source_bits["family"],
                "source_family_generation": source_bits["family_generation"],
                "target_text_id": f"post:{post_id}",
                "target_text_type": "post",
                "target_user_id": target_user_id,
                "target_user_name": target_user_name,
                "target_label": target_bits["label"],
                "target_family": target_bits["family"],
                "target_family_generation": target_bits["family_generation"],
                "root_post_id": post_id,
                "subreddit": _safe_text(_first_present(payload, "subreddit")),
                "created_utc": _safe_text(_first_present(payload, "created_utc", "comment1_time", "time")),
                "source_kind": "Comment_1.csv",
            }
        )

    for _, row in comments2.iterrows():
        payload = row.to_dict()
        comment_id = _safe_text(_first_present(payload, "comment_id", "comment2_id"))
        post_id = _safe_text(_first_present(payload, "link_id")).replace("t3_", "")
        parent_comment_id = _safe_text(_first_present(payload, "parent_id")).replace("t1_", "")
        source_user_id = _safe_text(_first_present(payload, "comment_author_id", "author_id"))
        source_user_name = _safe_text(_first_present(payload, "comment_author_name", "author_name"))
        target_user_id = _safe_text(c1_author.get(parent_comment_id, ""))
        target_user_name = _safe_text(c1_author_name.get(parent_comment_id, ""))
        source_bits = profile_bits(source_user_id)
        target_bits = profile_bits(target_user_id)
        rows.append(
            {
                "benchmark_round_id": round_meta["benchmark_round_id"],
                "round_index": round_meta["round_index"],
                "family": source_bits["family"],
                "family_generation": source_bits["family_generation"],
                "family_round": _safe_int(profile_map.get(source_user_id, {}).get("family_round", 0)) if source_bits["label"] == "bot" else 0,
                "edge_id": _stable_id("e2_", round_meta["benchmark_round_id"], comment_id, parent_comment_id, source_user_id, target_user_id),
                "edge_type": "reply_to_comment",
                "edge_level": 2,
                "source_text_id": f"comment2:{comment_id}",
                "source_text_type": "comment2",
                "source_user_id": source_user_id,
                "source_user_name": source_user_name,
                "source_label": source_bits["label"],
                "source_family": source_bits["family"],
                "source_family_generation": source_bits["family_generation"],
                "target_text_id": f"comment1:{parent_comment_id}",
                "target_text_type": "comment1",
                "target_user_id": target_user_id,
                "target_user_name": target_user_name,
                "target_label": target_bits["label"],
                "target_family": target_bits["family"],
                "target_family_generation": target_bits["family_generation"],
                "root_post_id": post_id,
                "subreddit": _safe_text(_first_present(payload, "subreddit")),
                "created_utc": _safe_text(_first_present(payload, "created_utc", "comment2_time", "time")),
                "source_kind": "Comment_2.csv",
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["created_utc", "edge_level", "edge_id"], kind="stable")
    return df.reindex(columns=EDGE_COLUMNS)


def _build_family_map(user_samples: pd.DataFrame) -> pd.DataFrame:
    if user_samples.empty:
        return pd.DataFrame(columns=FAMILY_MAP_COLUMNS)
    columns = [
        "benchmark_round_id",
        "round_index",
        "family",
        "family_slug",
        "family_generation",
        "family_round",
        "label",
        "user_id",
        "user_name",
        "source_kind",
        "provider",
        "model",
        "temperature",
        "reasoning_effort",
        "graph_affinity",
        "notification_reactivity",
        "post_tendency",
        "reply_tendency",
        "revisit_tendency",
        "exploration_tendency",
        "brevity",
    ]
    return user_samples.reindex(columns=columns)


def _build_round_map(round_meta: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    summary = round_meta.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    feedback = round_meta.get("feedback", {})
    if not isinstance(feedback, Mapping):
        feedback = {}
    row = {
        "benchmark_round_id": round_meta["benchmark_round_id"],
        "round_index": round_meta["round_index"],
        "family": round_meta["family"],
        "family_generation": _safe_int(round_meta["family_generation"]),
        "family_round": _safe_int(round_meta["family_round"]),
        "run_key": round_meta["run_key"],
        "run_dir": round_meta["run_dir"],
        "benchmark_dir": round_meta["benchmark_dir"],
        "simulation_rounds": _safe_int(round_meta["simulation_rounds"]),
        "browse_count": _safe_int(round_meta["browse_count"]),
        "agents_num": _safe_int(round_meta["agents_num"]),
        "seed": _safe_int(round_meta["seed"]),
        "provider": _safe_text(round_meta["provider"]),
        "model": _safe_text(round_meta["model"]),
        "temperature": _safe_float(round_meta["temperature"]),
        "reasoning_effort": _safe_text(round_meta["reasoning_effort"]),
        "dry_run": bool(round_meta["dry_run"]),
        "start_time": _safe_text(round_meta["start_time"]),
        "end_time": _safe_text(round_meta["end_time"]),
        "duration_seconds": _safe_float(round_meta["duration_seconds"]),
        "bot_count": int((tables["user_samples"]["label"] == "bot").sum()) if "user_samples" in tables and not tables["user_samples"].empty else 0,
        "human_count": int((tables["user_samples"]["label"] == "human").sum()) if "user_samples" in tables and not tables["user_samples"].empty else 0,
        "text_sample_count": int(len(tables.get("text_samples", []))),
        "user_sample_count": int(len(tables.get("user_samples", []))),
        "edge_count": int(len(tables.get("interaction_edges", []))),
        "source_count": int(len(tables.get("source_map", []))),
        "summary_json": _json_dumps(summary),
        "feedback_json": _json_dumps(feedback),
        "family_manifest_path": _safe_text(round_meta.get("family_manifest_path", "")),
    }
    return pd.DataFrame([row], columns=ROUND_MAP_COLUMNS)


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix.lower() == ".jsonl":
        return len(_read_jsonl(path))
    if path.suffix.lower() == ".csv":
        try:
            return int(pd.read_csv(path, dtype=str).shape[0])
        except Exception:
            return 0
    return 1


def _build_source_map(run_dir: Path, round_meta: Mapping[str, Any]) -> pd.DataFrame:
    source_specs = [
        ("base_users", "input", "user", run_dir / "data" / "User.csv", "Original users copied into the run directory."),
        ("base_posts", "input", "text", run_dir / "data" / "Posts.csv", "All post rows available for this benchmark run."),
        ("base_comments1", "input", "text", run_dir / "data" / "Comment_1.csv", "All first-level comments available for this benchmark run."),
        ("base_comments2", "input", "text", run_dir / "data" / "Comment_2.csv", "All second-level comments available for this benchmark run."),
        ("generated_persona_profiles", "generated", "user", run_dir / "generated_persona_profiles.csv", "Persona profiles used to generate the bot side."),
        ("decision_trace", "trace", "decision", run_dir / "decision_trace.jsonl", "Decision trace emitted by the simulation."),
        ("llm_trace", "trace", "llm", run_dir / "llm_trace.jsonl", "LLM prompts and responses."),
        ("memory_trace", "trace", "memory", run_dir / "memory_trace.jsonl", "Memory and notification updates."),
        ("summary", "summary", "round", run_dir / "summary.csv", "Round-level summary metrics."),
        ("readme", "manifest", "round", run_dir / "README.md", "Round README emitted by the simulation runner."),
    ]
    rows = []
    for source_id, source_kind, record_domain, source_file, description in source_specs:
        rows.append(
            {
                "benchmark_round_id": round_meta["benchmark_round_id"],
                "round_index": round_meta["round_index"],
                "family": round_meta["family"],
                "family_generation": _safe_int(round_meta["family_generation"]),
                "family_round": _safe_int(round_meta["family_round"]),
                "source_id": source_id,
                "source_kind": source_kind,
                "record_domain": record_domain,
                "source_file": str(source_file.resolve()),
                "exists": bool(source_file.exists()),
                "row_count": _count_rows(source_file),
                "description": description,
            }
        )
    if round_meta.get("family_manifest_path"):
        source_file = Path(round_meta["family_manifest_path"])
        rows.append(
            {
                "benchmark_round_id": round_meta["benchmark_round_id"],
                "round_index": round_meta["round_index"],
                "family": round_meta["family"],
                "family_generation": _safe_int(round_meta["family_generation"]),
                "family_round": _safe_int(round_meta["family_round"]),
                "source_id": "family_manifest",
                "source_kind": "manifest",
                "record_domain": "round",
                "source_file": str(source_file.resolve()),
                "exists": bool(source_file.exists()),
                "row_count": _count_rows(source_file),
                "description": "Family manifest used to seed the round.",
            }
        )
    return pd.DataFrame(rows, columns=SOURCE_MAP_COLUMNS)


def build_benchmark_tables(
    run_dir: Path,
    round_meta: Mapping[str, Any] | None = None,
    *,
    family_manifest_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    round_info = _resolve_round_meta(round_meta, run_dir)
    profile_map = _load_profile_map(run_dir, family_manifest_path or (Path(round_info["family_manifest_path"]) if round_info.get("family_manifest_path") else None))

    text_samples = _build_text_samples(run_dir, round_info, profile_map)
    user_samples = _build_user_samples(run_dir, round_info, profile_map)
    interaction_edges = _build_interaction_edges(run_dir, round_info, profile_map)
    family_map = _build_family_map(user_samples)
    source_map = _build_source_map(run_dir, round_info)
    round_map = _build_round_map(round_info, {
        "text_samples": text_samples,
        "user_samples": user_samples,
        "interaction_edges": interaction_edges,
        "source_map": source_map,
    })

    index_rows: list[dict[str, Any]] = []
    for _, row in text_samples.iterrows():
        index_rows.append(
            {
                "record_type": "text",
                "record_id": row["sample_id"],
                "benchmark_round_id": row["benchmark_round_id"],
                "round_index": row["round_index"],
                "family": row["family"],
                "family_generation": row["family_generation"],
                "family_round": row["family_round"],
                "user_id": row["user_id"],
                "user_name": row["user_name"],
                "label": row["label"],
                "source_kind": row["source_table"],
                "text": row["text_body"],
                "edge_type": "",
                "path": row["source_table"],
            }
        )
    for _, row in user_samples.iterrows():
        index_rows.append(
            {
                "record_type": "user",
                "record_id": row["user_id"],
                "benchmark_round_id": row["benchmark_round_id"],
                "round_index": row["round_index"],
                "family": row["family"],
                "family_generation": row["family_generation"],
                "family_round": row["family_round"],
                "user_id": row["user_id"],
                "user_name": row["user_name"],
                "label": row["label"],
                "source_kind": row["source_kind"],
                "text": "",
                "edge_type": "",
                "path": "user_samples",
            }
        )
    for _, row in interaction_edges.iterrows():
        index_rows.append(
            {
                "record_type": "edge",
                "record_id": row["edge_id"],
                "benchmark_round_id": row["benchmark_round_id"],
                "round_index": row["round_index"],
                "family": row["family"],
                "family_generation": row["family_generation"],
                "family_round": row["family_round"],
                "user_id": row["source_user_id"],
                "user_name": row["source_user_name"],
                "label": row["source_label"],
                "source_kind": row["source_kind"],
                "text": "",
                "edge_type": row["edge_type"],
                "path": row["source_text_id"],
            }
        )
    benchmark_index = pd.DataFrame(index_rows, columns=INDEX_COLUMNS)

    record_rows: list[dict[str, Any]] = []
    for _, row in text_samples.iterrows():
        record_rows.append(
            {
                "record_type": "text",
                "record_id": row["sample_id"],
                "round_id": round_info["benchmark_round_id"],
                "family": row["family"],
                "family_generation": row["family_generation"],
                "family_round": row["family_round"],
                "user_id": row["user_id"],
                "user_name": row["user_name"],
                "label": row["label"],
                "source_kind": row["source_table"],
                "payload": row.to_dict(),
            }
        )
    for _, row in user_samples.iterrows():
        record_rows.append(
            {
                "record_type": "user",
                "record_id": row["user_id"],
                "round_id": round_info["benchmark_round_id"],
                "family": row["family"],
                "family_generation": row["family_generation"],
                "family_round": row["family_round"],
                "user_id": row["user_id"],
                "user_name": row["user_name"],
                "label": row["label"],
                "source_kind": row["source_kind"],
                "payload": row.to_dict(),
            }
        )
    for _, row in interaction_edges.iterrows():
        record_rows.append(
            {
                "record_type": "edge",
                "record_id": row["edge_id"],
                "round_id": round_info["benchmark_round_id"],
                "family": row["family"],
                "family_generation": row["family_generation"],
                "family_round": row["family_round"],
                "user_id": row["source_user_id"],
                "user_name": row["source_user_name"],
                "label": row["source_label"],
                "source_kind": row["source_kind"],
                "payload": row.to_dict(),
            }
        )
    benchmark_records = pd.DataFrame(record_rows)

    return {
        "text_samples": text_samples,
        "user_samples": user_samples,
        "interaction_edges": interaction_edges,
        "family_map": family_map,
        "round_map": round_map,
        "source_map": source_map,
        "benchmark_index": benchmark_index,
        "benchmark_records": benchmark_records,
    }


def _write_dataframe(df: pd.DataFrame, path: Path, columns: Iterable[str]) -> Path:
    _ensure_dir(path.parent)
    frame = df.copy() if not df.empty else pd.DataFrame(columns=list(columns))
    if not frame.empty:
        frame = frame.reindex(columns=list(columns))
    else:
        frame = pd.DataFrame(columns=list(columns))
    frame.to_csv(path, index=False)
    return path


def _write_jsonl(df: pd.DataFrame, path: Path) -> Path:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            payload = row.to_dict()
            f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
    return path


def write_benchmark_bundle(
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    round_meta: Mapping[str, Any],
) -> dict[str, Path]:
    _ensure_dir(output_dir)
    table_dir = _ensure_dir(output_dir / "tables")
    written: dict[str, Path] = {}
    written["text_samples"] = _write_dataframe(tables["text_samples"], table_dir / "text_samples.csv", TEXT_SAMPLE_COLUMNS)
    written["user_samples"] = _write_dataframe(tables["user_samples"], table_dir / "user_samples.csv", USER_SAMPLE_COLUMNS)
    written["interaction_edges"] = _write_dataframe(tables["interaction_edges"], table_dir / "interaction_edges.csv", EDGE_COLUMNS)
    written["family_map"] = _write_dataframe(tables["family_map"], table_dir / "family_map.csv", FAMILY_MAP_COLUMNS)
    written["round_map"] = _write_dataframe(tables["round_map"], table_dir / "round_map.csv", ROUND_MAP_COLUMNS)
    written["source_map"] = _write_dataframe(tables["source_map"], table_dir / "source_map.csv", SOURCE_MAP_COLUMNS)
    written["benchmark_index"] = _write_dataframe(tables["benchmark_index"], table_dir / "benchmark_index.csv", INDEX_COLUMNS)
    written["benchmark_records"] = _write_jsonl(tables["benchmark_records"], table_dir / "benchmark_records.jsonl")

    manifest = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_round_id": round_meta["benchmark_round_id"],
        "round_index": round_meta["round_index"],
        "family": round_meta["family"],
        "family_generation": _safe_int(round_meta["family_generation"]),
        "family_round": _safe_int(round_meta["family_round"]),
        "run_dir": round_meta["run_dir"],
        "benchmark_dir": str(output_dir.resolve()),
        "tables": {
            name: {
                "path": str(path.relative_to(output_dir)),
                "rows": int(len(tables[name])),
            }
            for name, path in written.items()
            if name in tables
        },
        "round_meta": {
            key: value
            for key, value in round_meta.items()
            if key not in {"summary", "feedback"}
        },
        "summary": round_meta.get("summary", {}),
        "feedback": round_meta.get("feedback", {}),
        "schemas": {
            "text_samples": TEXT_SAMPLE_COLUMNS,
            "user_samples": USER_SAMPLE_COLUMNS,
            "interaction_edges": EDGE_COLUMNS,
            "family_map": FAMILY_MAP_COLUMNS,
            "round_map": ROUND_MAP_COLUMNS,
            "source_map": SOURCE_MAP_COLUMNS,
            "benchmark_index": INDEX_COLUMNS,
        },
    }
    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    written["manifest"] = manifest_path

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Benchmark export",
                "",
                f"- schema_version: `{BENCHMARK_SCHEMA_VERSION}`",
                f"- benchmark_round_id: `{round_meta['benchmark_round_id']}`",
                f"- family: `{round_meta['family']}`",
                f"- family_generation: `{round_meta['family_generation']}`",
                f"- family_round: `{round_meta['family_round']}`",
                "- tables: `text_samples.csv`, `user_samples.csv`, `interaction_edges.csv`, `family_map.csv`, `round_map.csv`, `source_map.csv`, `benchmark_index.csv`, `benchmark_records.jsonl`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    written["readme"] = readme
    return written


def export_run_benchmark(
    run_dir: Path,
    output_dir: Path,
    round_meta: Mapping[str, Any] | None = None,
    *,
    family_manifest_path: Path | None = None,
) -> dict[str, Any]:
    tables = build_benchmark_tables(run_dir, round_meta, family_manifest_path=family_manifest_path)
    resolved_meta = _resolve_round_meta(round_meta, run_dir)
    written = write_benchmark_bundle(output_dir, tables, resolved_meta)
    return {
        "tables": tables,
        "written": written,
        "round_meta": resolved_meta,
    }


def merge_round_exports(export_dirs: Iterable[Path], output_dir: Path) -> dict[str, Path]:
    export_dirs = [Path(path) for path in export_dirs]
    _ensure_dir(output_dir)
    table_dir = _ensure_dir(output_dir / "tables")
    aggregated: dict[str, list[pd.DataFrame]] = {
        "text_samples": [],
        "user_samples": [],
        "interaction_edges": [],
        "family_map": [],
        "round_map": [],
        "source_map": [],
        "benchmark_index": [],
    }
    records_files: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []

    for export_dir in export_dirs:
        export_dir = Path(export_dir)
        manifest_path = export_dir / "benchmark_manifest.json"
        manifest = _read_json(manifest_path)
        if manifest:
            manifest_rows.append(
                {
                    "benchmark_round_id": manifest.get("benchmark_round_id", ""),
                    "family": manifest.get("family", ""),
                    "family_generation": manifest.get("family_generation", 0),
                    "family_round": manifest.get("family_round", 0),
                    "round_index": manifest.get("round_index", 0),
                    "export_dir": str(export_dir.resolve()),
                    "manifest_path": str(manifest_path.resolve()),
                }
            )
        for table_name in aggregated:
            path = export_dir / "tables" / f"{table_name}.csv"
            if path.exists():
                aggregated[table_name].append(_read_csv(path))
        record_path = export_dir / "tables" / "benchmark_records.jsonl"
        if record_path.exists():
            records_files.append(record_path)

    written: dict[str, Path] = {}
    for table_name, frames in aggregated.items():
        if frames:
            combined = pd.concat(frames, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=globals()[f"{table_name.upper()}_COLUMNS"] if f"{table_name.upper()}_COLUMNS" in globals() else [])
        columns = globals().get(f"{table_name.upper()}_COLUMNS", list(combined.columns))
        written[table_name] = _write_dataframe(combined, table_dir / f"{table_name}.csv", columns)

    combined_records: list[dict[str, Any]] = []
    for record_path in records_files:
        combined_records.extend(_read_jsonl(record_path))
    records_df = pd.DataFrame(combined_records)
    if records_df.empty:
        records_df = pd.DataFrame(columns=["record_type", "record_id", "round_id", "family", "family_generation", "family_round", "user_id", "user_name", "label", "source_kind", "payload"])
    _write_jsonl(records_df, table_dir / "benchmark_records.jsonl")
    written["benchmark_records"] = table_dir / "benchmark_records.jsonl"

    manifest = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "round_count": len(export_dirs),
        "round_exports": manifest_rows,
        "tables": {
            name: {
                "path": str((table_dir / f"{name}.csv").relative_to(output_dir)),
                "rows": int(len(_read_csv(table_dir / f"{name}.csv"))),
            }
            for name in aggregated
        },
        "records": {
            "path": str((table_dir / "benchmark_records.jsonl").relative_to(output_dir)),
            "rows": int(len(_read_jsonl(table_dir / "benchmark_records.jsonl"))),
        },
    }
    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    written["manifest"] = manifest_path
    return written


def derive_feedback_from_round(
    summary: Mapping[str, Any] | None,
    user_samples: pd.DataFrame | None = None,
    interaction_edges: pd.DataFrame | None = None,
) -> dict[str, float]:
    summary = summary or {}
    user_samples = user_samples if user_samples is not None else pd.DataFrame()
    interaction_edges = interaction_edges if interaction_edges is not None else pd.DataFrame()
    mean_comment_words = _safe_float(summary.get("mean_comment_words", 0.0), 0.0)
    bot_to_bot_total = _safe_float(summary.get("bot_to_bot_total", 0.0), 0.0)
    bot_to_human_total = _safe_float(summary.get("bot_to_human_total", 0.0), 0.0)
    total_mentions = bot_to_bot_total + bot_to_human_total
    bot_share = bot_to_bot_total / max(1.0, total_mentions)
    reciprocity = 0.0
    if not user_samples.empty and "family" in user_samples and "label" in user_samples:
        bot_rows = user_samples[user_samples["label"] == "bot"]
        if not bot_rows.empty and "graph_affinity" in bot_rows:
            reciprocity = float(bot_rows["graph_affinity"].fillna(0.0).mean())
    if interaction_edges is not None and not interaction_edges.empty:
        c1 = interaction_edges[interaction_edges["edge_type"] == "reply_to_post"] if "edge_type" in interaction_edges else pd.DataFrame()
        c2 = interaction_edges[interaction_edges["edge_type"] == "reply_to_comment"] if "edge_type" in interaction_edges else pd.DataFrame()
        if not c1.empty or not c2.empty:
            graph_depth = float(len(c2)) / max(1.0, float(len(c1) + len(c2)))
        else:
            graph_depth = 0.0
    else:
        graph_depth = 0.0
    return {
        "text_pressure": round(max(0.0, min(1.0, 1.0 - mean_comment_words / 18.0)), 4),
        "graph_pressure": round(max(0.0, min(1.0, bot_share + 0.25 * graph_depth)), 4),
        "alignment_pressure": round(max(0.0, min(1.0, 1.0 - reciprocity)), 4),
        "recovery_bias": round(max(0.0, min(1.0, 0.45 + 0.02 * _safe_float(summary.get("mean_posts_per_bot", 0.0), 0.0))), 4),
    }

