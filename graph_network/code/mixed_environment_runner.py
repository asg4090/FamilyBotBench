#!/usr/bin/env python3
"""First runnable mixed dynamic environment runner for the static graph dataset.

This runner produces a complete dry-run dataset without calling external APIs.
It prioritizes the project-local BotSim snapshot JSON as the background
environment and schedules G1/G2/G3 bots in one shared, time-interleaved queue.
"""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import argparse
import csv
import heapq
import importlib
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "graph_static_dataset_v1.yaml"
DEFAULT_SNAPSHOT = PROJECT_DIR / "inputs" / "original_botsim_snapshot" / "BotSim-24-Dataset"

FAMILIES = [
    "G1_population_calibrated",
    "G2_anchor_behavior",
    "G3_contextual_interactive",
]

USER_FIELDS = [
    "user_id",
    "user_name",
    "source",
    "label",
    "family",
    "is_generated",
    "is_replay_human",
    "created_by_run",
]

POST_FIELDS = [
    "post_id",
    "author_id",
    "author_source",
    "family",
    "subreddit",
    "topic",
    "created_utc",
    "title",
    "body",
    "score",
    "num_comments",
    "upvote_ratio",
    "is_generated",
    "is_replay_context",
]

COMMENT_FIELDS = [
    "comment_id",
    "author_id",
    "author_source",
    "family",
    "post_id",
    "parent_id",
    "parent_type",
    "level",
    "created_utc",
    "body",
    "score",
    "is_generated",
    "is_replay_human",
    "action_type",
    "target_author_id",
]

LABEL_FIELDS = [
    "node_id",
    "node_type",
    "label",
    "label_known",
    "source",
    "family",
    "split_group",
]

PERSONA_FIELDS = [
    "user_id",
    "family",
    "profile_source",
    "anchor_user_ids",
    "llm_provider",
    "llm_model",
    "llm_temperature",
    "llm_reasoning_effort",
    "post_tendency",
    "reply_tendency",
    "comment1_tendency",
    "comment2_tendency",
    "revisit_tendency",
    "skip_tendency",
    "exploration_tendency",
    "notification_reactivity",
    "topic_entropy_target",
    "active_rhythm_type",
    "preferred_topics",
    "context_sensitivity",
    "diversity_preference",
    "memory_decay",
    "language_style_summary",
    "behavior_style_summary",
    "metadata_json",
]

FAMILY_MANIFEST_FIELDS = [
    "family",
    "family_label",
    "bot_count",
    "policy_source",
    "dry_run",
    "live_llm",
    "description",
]

ANCHOR_FIELDS = ["bot_user_id", "family", "anchor_user_ids", "binding_reason"]

EDGE_FIELDS = [
    "edge_id",
    "src_id",
    "dst_id",
    "src_type",
    "dst_type",
    "edge_type",
    "time",
    "weight",
    "source_event_id",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _truth(value: Any) -> str:
    return "true" if bool(value) else "false"


def _safe_text(value: Any, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _parse_time(value: Any, fallback: datetime) -> datetime:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=None)
        except ValueError:
            pass
    return fallback


def _format_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _load_yaml_or_minimal(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            return loaded if isinstance(loaded, dict) else {}
    except Exception:
        # Minimal fallback for the known config. It keeps dry-run usable even
        # under an interpreter without PyYAML.
        data: Dict[str, Any] = {"paths": {}, "runtime": {}, "generation": {"time": {}, "human_reply_replay": {}}}
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("seed:"):
                data["runtime"]["seed"] = _as_int(stripped.split(":", 1)[1].strip(), 20260603)
            elif stripped.startswith("browse_count:"):
                data["runtime"]["browse_count"] = _as_int(stripped.split(":", 1)[1].strip(), 10)
            elif stripped.startswith("run_name:"):
                data["generation"]["run_name"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("windows:"):
                data["generation"]["time"]["windows"] = _as_int(stripped.split(":", 1)[1].strip(), 8)
            elif stripped.startswith("start_time:"):
                data["generation"]["time"]["start_time"] = stripped.split(":", 1)[1].strip().strip('"')
            elif stripped.startswith("window_hours:"):
                data["generation"]["time"]["window_hours"] = _as_int(stripped.split(":", 1)[1].strip(), 12)
        return data


def _config_get(config: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    current: Any = config
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _normalize_post_id(raw_id: Any) -> str:
    raw = str(raw_id or "").strip()
    raw = raw[3:] if raw.startswith("t3_") else raw
    return f"p_{raw}" if raw else ""


def _normalize_comment_id(raw_id: Any) -> str:
    raw = str(raw_id or "").strip()
    raw = raw[3:] if raw.startswith("t1_") else raw
    return f"c_{raw}" if raw else ""


def _parent_type_and_id(raw_parent: Any) -> Tuple[str, str]:
    parent = str(raw_parent or "").strip()
    if parent.startswith("t1_"):
        return "comment", _normalize_comment_id(parent)
    if parent.startswith("t3_"):
        return "post", _normalize_post_id(parent)
    if parent:
        return "comment", _normalize_comment_id(parent)
    return "", ""


def _parse_subreddits(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            pass
    return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]


@dataclass
class BotState:
    user_id: str
    user_name: str
    family: str
    anchors: List[str]
    post_tendency: float
    reply_tendency: float
    revisit_tendency: float
    exploration_tendency: float
    notification_reactivity: float
    topic_entropy_target: float
    active_rhythm_type: str
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_reasoning_effort: str
    comment1_tendency: float
    comment2_tendency: float
    skip_tendency: float
    preferred_topics: List[str]
    context_sensitivity: float
    diversity_preference: float
    memory_decay: float
    notification_queue: List[Dict[str, Any]] = field(default_factory=list)
    own_history: List[Dict[str, Any]] = field(default_factory=list)


class PolicyAdapter:
    """Loose adapter for a future social_family_policies.py implementation."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.module = None
        self.policy_source = "local_fallback"
        try:
            sys.path.insert(0, str(CODE_DIR))
            self.module = importlib.import_module("social_family_policies")
            self.policy_source = "social_family_policies.py"
        except Exception:
            self.module = None

    def decide(self, bot: BotState, candidates: List[Dict[str, Any]], now: datetime) -> Dict[str, Any]:
        if self.module is not None:
            choose_action = getattr(self.module, "choose_action", None)
            social_persona_cls = getattr(self.module, "SocialPersona", None)
            if callable(choose_action) and social_persona_cls is not None:
                try:
                    persona = self._to_social_persona(bot, candidates, social_persona_cls)
                    memory = self._to_policy_memory(bot)
                    decision = choose_action(persona, candidates, memory, self.rng)
                    converted = self._from_action_decision(decision, candidates)
                    if converted.get("action_type"):
                        return converted
                except Exception as exc:
                    return self._fallback_decide(bot, candidates, reason=f"policy_adapter_failed: {exc}")
        return self._fallback_decide(bot, candidates)

    def _to_social_persona(self, bot: BotState, candidates: List[Dict[str, Any]], social_persona_cls: Any) -> Any:
        topics = []
        for candidate in candidates:
            topic = str(candidate.get("subreddit") or candidate.get("topic") or "").strip()
            if topic and topic not in topics:
                topics.append(topic)
        if not topics:
            topics = ["politics", "worldnews", "news"]
        context_sensitivity = 0.35
        diversity_preference = 0.45
        if bot.family.startswith("G2"):
            context_sensitivity = 0.42
            diversity_preference = 0.35
        elif bot.family.startswith("G3"):
            context_sensitivity = 0.78
            diversity_preference = 0.60
        return social_persona_cls(
            user_id=bot.user_id,
            family=bot.family,
            seed=self.rng.randint(1, 2_147_483_647),
            profile_source="mixed_environment_runner",
            anchor_user_ids=tuple(bot.anchors),
            provider_model=f"{bot.llm_provider}:{bot.llm_model}",
            post_tendency=bot.post_tendency,
            reply_tendency=bot.reply_tendency,
            comment1_tendency=bot.comment1_tendency,
            comment2_tendency=bot.comment2_tendency,
            revisit_tendency=bot.revisit_tendency,
            exploration_tendency=bot.exploration_tendency,
            notification_reactivity=bot.notification_reactivity,
            skip_tendency=bot.skip_tendency,
            topic_entropy_target=bot.topic_entropy_target,
            active_rhythm_type=bot.active_rhythm_type,
            preferred_topics=tuple(bot.preferred_topics or topics[:5]),
            behavior_style_summary=f"runner-integrated {bot.family} policy",
            context_sensitivity=bot.context_sensitivity or context_sensitivity,
            controversy_tolerance=0.42 if bot.family.startswith("G3") else 0.28,
            diversity_preference=bot.diversity_preference or diversity_preference,
            memory_decay=bot.memory_decay,
        )

    def _to_policy_memory(self, bot: BotState) -> Dict[str, Any]:
        participated_posts = []
        participated_comments = []
        recent_targets = []
        for item in bot.own_history[-12:]:
            post_id = str(item.get("post_id") or "")
            comment_id = str(item.get("comment_id") or "")
            target_id = str(item.get("target_id") or post_id or comment_id or "")
            if post_id:
                participated_posts.append(post_id)
            if comment_id:
                participated_comments.append(comment_id)
            if target_id:
                recent_targets.append(target_id)
        return {
            "notifications": list(bot.notification_queue),
            "notification_queue": list(bot.notification_queue),
            "participated_post_ids": participated_posts,
            "participated_comment_ids": participated_comments,
            "recent_target_ids": recent_targets[-8:],
            "seen_target_ids": recent_targets[-20:],
        }

    def _from_action_decision(self, decision: Any, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if hasattr(decision, "__dataclass_fields__"):
            action_type = str(getattr(decision, "action_type", "skip"))
            target_id = str(getattr(decision, "target_id", ""))
            target_post_id = str(getattr(decision, "target_post_id", ""))
            target_comment_id = str(getattr(decision, "target_comment_id", ""))
            reason = str(getattr(decision, "reason", "social_family_policy"))
            sampled = getattr(decision, "sampled_probabilities", {})
            target = self._match_candidate(candidates, target_id, target_post_id, target_comment_id)
            return {
                "action_type": action_type,
                "target": target,
                "policy_reason": reason,
                "policy_score": getattr(decision, "score", ""),
                "sampled_probabilities": sampled,
            }
        if isinstance(decision, dict):
            return decision
        return {"action_type": "skip", "target": {}, "policy_reason": "unsupported_policy_decision"}

    @staticmethod
    def _match_candidate(
        candidates: List[Dict[str, Any]],
        target_id: str,
        target_post_id: str,
        target_comment_id: str,
    ) -> Dict[str, Any]:
        for candidate in candidates:
            if target_id and str(candidate.get("target_id")) == target_id:
                return candidate
            if target_post_id and str(candidate.get("post_id")) == target_post_id:
                if not target_comment_id or str(candidate.get("comment_id")) == target_comment_id:
                    return candidate
            if target_comment_id and str(candidate.get("comment_id")) == target_comment_id:
                return candidate
        return {}

    def _fallback_decide(self, bot: BotState, candidates: List[Dict[str, Any]], reason: str | None = None) -> Dict[str, Any]:
        notification_targets = [c for c in candidates if c["candidate_type"] == "notification"]
        own_targets = [c for c in candidates if c["candidate_type"] == "own_history"]
        post_targets = [c for c in candidates if c["target_type"] == "post"]
        comment_targets = [c for c in candidates if c["target_type"] == "comment"]

        weights = {
            "skip": max(0.08, 0.30 - bot.reply_tendency * 0.10),
            "post": bot.post_tendency,
            "comment1": bot.reply_tendency,
            "comment2": bot.reply_tendency * 0.45,
            "revisit": bot.revisit_tendency,
        }
        if bot.family.startswith("G1"):
            weights.update({"skip": 0.24, "post": bot.post_tendency, "comment1": bot.reply_tendency, "comment2": 0.12})
        elif bot.family.startswith("G2"):
            weights.update({"skip": 0.16, "post": bot.post_tendency * 0.8, "comment1": bot.reply_tendency, "comment2": 0.22})
        elif bot.family.startswith("G3"):
            weights.update({"skip": 0.10, "post": bot.post_tendency * 0.7, "comment1": bot.reply_tendency, "comment2": 0.30, "revisit": bot.revisit_tendency * 1.4})

        if not post_targets:
            weights["comment1"] = 0.0
        if not comment_targets:
            weights["comment2"] = 0.0
        if not notification_targets and not own_targets:
            weights["revisit"] = 0.0

        actions, action_weights = zip(*[(a, max(0.0, w)) for a, w in weights.items()])
        action = self.rng.choices(actions, weights=action_weights, k=1)[0]
        if action == "comment1":
            target = self.rng.choice(post_targets)
        elif action == "comment2":
            target = self.rng.choice(comment_targets)
        elif action == "revisit":
            pool = notification_targets + own_targets + comment_targets + post_targets
            target = self.rng.choice(pool) if pool else {}
        else:
            target = {}
        return {
            "action_type": action,
            "target": target,
            "policy_reason": reason or f"fallback dry-run policy for {bot.family}",
        }


class MixedEnvironment:
    def __init__(
        self,
        *,
        config: Dict[str, Any],
        output_dir: Path,
        bots_per_family: int,
        time_windows: int,
        dry_run: bool,
        seed: int,
        family_counts: Optional[Dict[str, int]] = None,
        live_llm: bool = False,
        run_name: str = "",
    ):
        self.config = config
        self.output_dir = output_dir
        self.bots_per_family = bots_per_family
        self.family_counts = family_counts or {family: bots_per_family for family in FAMILIES}
        self.time_windows = time_windows
        self.live_llm = live_llm
        self.dry_run = bool(dry_run and not live_llm)
        self.seed = seed
        self.rng = random.Random(seed)
        self.run_name = str(run_name or _config_get(config, ["generation", "run_name"], output_dir.name) or output_dir.name)
        self.browse_count = _as_int(_config_get(config, ["runtime", "browse_count"], 10), 10)
        self.start_time = _parse_time(
            _config_get(config, ["generation", "time", "start_time"], "2024-01-01 08:00:00"),
            datetime(2024, 1, 1, 8, 0, 0),
        )
        self.window_hours = _as_int(_config_get(config, ["generation", "time", "window_hours"], 12), 12)
        self.policy = PolicyAdapter(self.rng)
        self.event_counter = 0
        self.edge_counter = 0
        self.post_counter = 0
        self.comment_counter = 0

        self.users: List[Dict[str, Any]] = []
        self.posts: List[Dict[str, Any]] = []
        self.comments: List[Dict[str, Any]] = []
        self.labels: List[Dict[str, Any]] = []
        self.personas: List[Dict[str, Any]] = []
        self.family_manifest: List[Dict[str, Any]] = []
        self.anchor_bindings: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.bots: List[BotState] = []

        self.user_by_id: Dict[str, Dict[str, Any]] = {}
        self.post_by_id: Dict[str, Dict[str, Any]] = {}
        self.comment_by_id: Dict[str, Dict[str, Any]] = {}
        self.source_humans: List[Dict[str, Any]] = []
        self.source_original_bots: List[Dict[str, Any]] = []
        self.llm_client: Any = None
        self.llm_call_counter = 0
        if self.live_llm:
            try:
                sys.path.insert(0, str(CODE_DIR))
                from llm_client import GraphLLMClient

                self.llm_client = GraphLLMClient(
                    max_tokens=_as_int(_config_get(config, ["llm", "max_tokens"], 2048), 2048),
                    timeout=float(_config_get(config, ["runtime", "llm_timeout_seconds"], 90) or 90),
                    retries=_as_int(_config_get(config, ["runtime", "retries"], 2), 2),
                    reasoning_effort=str(_config_get(config, ["llm", "openai", "reasoning_effort"], "medium") or "medium"),
                    api=str(_config_get(config, ["llm", "openai", "api"], "responses") or "responses"),
                )
            except Exception as exc:
                raise RuntimeError(f"--live-llm requested but LLM client could not initialize: {exc}") from exc

    def run(self) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._clear_run_files()
        dataset_dir = self._resolve_dataset_dir()
        self._load_background(dataset_dir)
        self._create_bots()
        self._run_schedule()
        self._write_outputs(dataset_dir)
        replay_summary = self._apply_replay()
        self._write_manifest(dataset_dir, replay_summary)
        return {
            "output_dir": str(self.output_dir),
            "background_posts": len([p for p in self.posts if not _as_bool_like(p.get("is_generated"))]),
            "background_comments": len([c for c in self.comments if not _as_bool_like(c.get("is_generated"))]),
            "generated_bots": len(self.bots),
            "replay": replay_summary,
            "policy_source": self.policy.policy_source,
        }

    def _clear_run_files(self) -> None:
        for name in (
            "events.jsonl",
            "action_trace.jsonl",
            "memory_trace.jsonl",
            "llm_trace.jsonl",
            "human_replay_manifest.csv",
        ):
            path = self.output_dir / name
            if path.exists():
                path.unlink()

    def _resolve_dataset_dir(self) -> Path:
        configured = Path(str(_config_get(self.config, ["paths", "original_dataset_dir"], "")))
        if (configured / "user_post_comment.json").exists():
            return configured
        reddit_data = PROJECT_DIR / "inputs" / "original_botsim_snapshot" / "RedditData"
        return reddit_data

    def _load_background(self, dataset_dir: Path) -> None:
        json_path = dataset_dir / "user_post_comment.json"
        users_path = dataset_dir / "Users.csv"
        if json_path.exists():
            self._load_background_from_json(json_path, users_path)
            return
        self._load_background_from_csv(dataset_dir)

    def _load_background_from_json(self, json_path: Path, users_path: Path) -> None:
        user_rows = _read_csv(users_path)
        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict[user_id] in {json_path}")

        human_ids = list(raw.keys())[:1907]
        user_meta = {row.get("user_id", ""): row for row in user_rows}
        for idx, row in enumerate(user_rows):
            user_id = row.get("user_id", "")
            is_human = user_id in human_ids if user_id else idx < 1907
            source = "original_human" if is_human else "original_bot"
            label = "0" if is_human else "1"
            user = {
                "user_id": user_id,
                "user_name": row.get("name", user_id),
                "source": source,
                "label": label,
                "family": "original_human" if is_human else "original_bot",
                "is_generated": "false",
                "is_replay_human": "false",
                "created_by_run": "source_dataset",
            }
            self._add_user(user, label_known="1", split_group=source)

        for user_id in human_ids:
            if user_id not in self.user_by_id:
                self._add_user(
                    {
                        "user_id": user_id,
                        "user_name": user_meta.get(user_id, {}).get("name", user_id),
                        "source": "original_human",
                        "label": "0",
                        "family": "original_human",
                        "is_generated": "false",
                        "is_replay_human": "false",
                        "created_by_run": "source_dataset",
                    },
                    label_known="1",
                    split_group="original_human",
                )

        for user_id in human_ids:
            payload = raw.get(user_id) or {}
            meta = user_meta.get(user_id, {})
            for post in payload.get("posts", []) or []:
                subreddit = post.get("subreddit") or (self._sample_subreddit(meta) or "unknown")
                post_id = _normalize_post_id(post.get("submission_id"))
                if not post_id or post_id in self.post_by_id:
                    continue
                row = {
                    "post_id": post_id,
                    "author_id": user_id,
                    "author_source": "original_human",
                    "family": "original_human",
                    "subreddit": subreddit,
                    "topic": subreddit,
                    "created_utc": post.get("created_utc", ""),
                    "title": _safe_text(post.get("posts") or post.get("title") or "", 400),
                    "body": _safe_text(post.get("body") or "", 1000),
                    "score": post.get("score", "0"),
                    "num_comments": post.get("num_comments", "0"),
                    "upvote_ratio": post.get("upvote_ratio", "0"),
                    "is_generated": "false",
                    "is_replay_context": "true",
                }
                self._add_post(row, source_event_id="")

        pending_comments: List[Tuple[str, Dict[str, Any]]] = []
        for user_id in human_ids:
            payload = raw.get(user_id) or {}
            for key in ("comment_1", "comment_2"):
                for comment in payload.get(key, []) or []:
                    pending_comments.append((user_id, comment))

        for author_id, comment in pending_comments:
            comment_id = _normalize_comment_id(comment.get("comment_id"))
            if not comment_id or comment_id in self.comment_by_id:
                continue
            parent_type, parent_id = _parent_type_and_id(comment.get("parent_id"))
            post_id = _normalize_post_id(comment.get("link_id")) or (parent_id if parent_type == "post" else "")
            level = _as_int(comment.get("level"), 1)
            target_author_id = self._target_author_id(parent_type, parent_id)
            row = {
                "comment_id": comment_id,
                "author_id": author_id,
                "author_source": "original_human",
                "family": "original_human",
                "post_id": post_id,
                "parent_id": parent_id,
                "parent_type": parent_type or ("post" if level == 1 else "comment"),
                "level": str(level),
                "created_utc": comment.get("created_utc", ""),
                "body": _safe_text(comment.get("comment_body") or comment.get("body") or "", 1000),
                "score": comment.get("comment_score", "0"),
                "is_generated": "false",
                "is_replay_human": "false",
                "action_type": "comment1" if level == 1 else "comment2",
                "target_author_id": target_author_id,
            }
            self._add_comment(row, source_event_id="")

    def _load_background_from_csv(self, reddit_dir: Path) -> None:
        users = _read_csv(reddit_dir / "HumanUser.csv") or _read_csv(reddit_dir / "User.csv")
        for row in users:
            user_id = row.get("user_id") or row.get("id") or row.get("author_id") or row.get("name")
            if not user_id:
                continue
            self._add_user(
                {
                    "user_id": user_id,
                    "user_name": row.get("name", user_id),
                    "source": "original_human",
                    "label": "0",
                    "family": "original_human",
                    "is_generated": "false",
                    "is_replay_human": "false",
                    "created_by_run": "source_dataset_csv_fallback",
                },
                label_known="1",
                split_group="original_human",
            )
        for row in _read_csv(reddit_dir / "Posts.csv"):
            post_id = _normalize_post_id(row.get("submission_id") or row.get("post_id") or row.get("id"))
            author_id = row.get("author_id") or row.get("user_id") or row.get("author")
            if not post_id or not author_id:
                continue
            if author_id not in self.user_by_id:
                self._add_user(
                    {
                        "user_id": author_id,
                        "user_name": row.get("author_name", author_id),
                        "source": "original_human",
                        "label": "0",
                        "family": "original_human",
                        "is_generated": "false",
                        "is_replay_human": "false",
                        "created_by_run": "source_dataset_csv_fallback",
                    },
                    label_known="1",
                    split_group="original_human",
                )
            subreddit = row.get("subreddit", "unknown")
            self._add_post(
                {
                    "post_id": post_id,
                    "author_id": author_id,
                    "author_source": "original_human",
                    "family": "original_human",
                    "subreddit": subreddit,
                    "topic": subreddit,
                    "created_utc": row.get("created_utc", ""),
                    "title": _safe_text(row.get("posts") or row.get("title") or "", 400),
                    "body": _safe_text(row.get("body") or "", 1000),
                    "score": row.get("score", "0"),
                    "num_comments": row.get("num_comments", "0"),
                    "upvote_ratio": row.get("upvote_ratio", "0"),
                    "is_generated": "false",
                    "is_replay_context": "true",
                },
                source_event_id="",
            )

    def _add_user(self, user: Dict[str, Any], *, label_known: str, split_group: str) -> None:
        user_id = str(user.get("user_id", ""))
        if not user_id or user_id in self.user_by_id:
            return
        self.users.append(user)
        self.user_by_id[user_id] = user
        if user.get("source") == "original_human":
            self.source_humans.append(user)
        elif user.get("source") == "original_bot":
            self.source_original_bots.append(user)
        self.labels.append(
            {
                "node_id": user_id,
                "node_type": "user",
                "label": user.get("label", "-1"),
                "label_known": label_known,
                "source": user.get("source", ""),
                "family": user.get("family", ""),
                "split_group": split_group,
            }
        )

    def _add_post(self, row: Dict[str, Any], *, source_event_id: str) -> None:
        post_id = str(row.get("post_id", ""))
        if not post_id or post_id in self.post_by_id:
            return
        self.posts.append(row)
        self.post_by_id[post_id] = row
        self._add_edge(row["author_id"], post_id, "user", "post", "user_writes_post", row.get("created_utc", ""), source_event_id)

    def _add_comment(self, row: Dict[str, Any], *, source_event_id: str) -> None:
        comment_id = str(row.get("comment_id", ""))
        if not comment_id or comment_id in self.comment_by_id:
            return
        self.comments.append(row)
        self.comment_by_id[comment_id] = row
        post_id = str(row.get("post_id") or "")
        if source_event_id and post_id in self.post_by_id:
            self.post_by_id[post_id]["num_comments"] = str(_as_int(self.post_by_id[post_id].get("num_comments"), 0) + 1)
        created = row.get("created_utc", "")
        self._add_edge(row["author_id"], comment_id, "user", "comment", "user_writes_comment", created, source_event_id)
        parent_type = row.get("parent_type")
        parent_id = row.get("parent_id")
        if parent_type == "post":
            self._add_edge(comment_id, parent_id, "comment", "post", "comment_replies_post", created, source_event_id)
            self._add_edge(row["author_id"], row.get("target_author_id", ""), "user", "user", "user_comments_on_user_post", created, source_event_id)
        elif parent_type == "comment":
            self._add_edge(comment_id, parent_id, "comment", "comment", "comment_replies_comment", created, source_event_id)
            self._add_edge(row["author_id"], row.get("target_author_id", ""), "user", "user", "user_replies_to_user_comment", created, source_event_id)

    def _add_edge(
        self,
        src_id: str,
        dst_id: str,
        src_type: str,
        dst_type: str,
        edge_type: str,
        time: str,
        source_event_id: str,
    ) -> None:
        if not src_id or not dst_id:
            return
        self.edge_counter += 1
        self.edges.append(
            {
                "edge_id": f"e_{self.run_name}_{self.edge_counter:08d}",
                "src_id": src_id,
                "dst_id": dst_id,
                "src_type": src_type,
                "dst_type": dst_type,
                "edge_type": edge_type,
                "time": time,
                "weight": "1",
                "source_event_id": source_event_id,
            }
        )

    def _create_bots(self) -> None:
        family_descriptions = {
            "G1_population_calibrated": "population calibrated dry-run social actor",
            "G2_anchor_behavior": "anchor behavior dry-run social actor",
            "G3_contextual_interactive": "contextual interactive dry-run social actor",
        }
        for family in FAMILIES:
            family_count = int(self.family_counts.get(family, self.bots_per_family))
            for index in range(family_count):
                anchors = self._sample_anchors(family)
                bot = self._make_bot(family, index, anchors)
                self.bots.append(bot)
                self._add_user(
                    {
                        "user_id": bot.user_id,
                        "user_name": bot.user_name,
                        "source": "generated_bot",
                        "label": "1",
                        "family": family,
                        "is_generated": "true",
                        "is_replay_human": "false",
                        "created_by_run": self.run_name,
                    },
                    label_known="1",
                    split_group=family,
                )
                self.personas.append(
                    {
                        "user_id": bot.user_id,
                        "family": family,
                        "profile_source": "dry_run_template" if not anchors else "dry_run_anchor_profile",
                        "anchor_user_ids": "|".join(anchors),
                        "llm_provider": bot.llm_provider,
                        "llm_model": bot.llm_model,
                        "llm_temperature": f"{bot.llm_temperature:.3f}",
                        "llm_reasoning_effort": bot.llm_reasoning_effort,
                        "post_tendency": f"{bot.post_tendency:.3f}",
                        "reply_tendency": f"{bot.reply_tendency:.3f}",
                        "comment1_tendency": f"{bot.comment1_tendency:.3f}",
                        "comment2_tendency": f"{bot.comment2_tendency:.3f}",
                        "revisit_tendency": f"{bot.revisit_tendency:.3f}",
                        "skip_tendency": f"{bot.skip_tendency:.3f}",
                        "exploration_tendency": f"{bot.exploration_tendency:.3f}",
                        "notification_reactivity": f"{bot.notification_reactivity:.3f}",
                        "topic_entropy_target": f"{bot.topic_entropy_target:.3f}",
                        "active_rhythm_type": bot.active_rhythm_type,
                        "preferred_topics": "|".join(bot.preferred_topics),
                        "context_sensitivity": f"{bot.context_sensitivity:.3f}",
                        "diversity_preference": f"{bot.diversity_preference:.3f}",
                        "memory_decay": f"{bot.memory_decay:.3f}",
                        "language_style_summary": f"dry-run text mentions {family} and visible context ids",
                        "behavior_style_summary": f"{family_descriptions[family]}; anchors={','.join(anchors) if anchors else 'none'}",
                        "metadata_json": json.dumps({"live_llm": self.live_llm}, ensure_ascii=False),
                    }
                )
                if anchors:
                    self.anchor_bindings.append(
                        {
                            "bot_user_id": bot.user_id,
                            "family": family,
                            "anchor_user_ids": "|".join(anchors),
                            "binding_reason": "sampled from original human users for behavior-rate calibration",
                        }
                    )
        for family in FAMILIES:
            self.family_manifest.append(
                {
                    "family": family,
                    "family_label": family.split("_", 1)[0],
                    "bot_count": str(self.family_counts.get(family, self.bots_per_family)),
                    "policy_source": self.policy.policy_source,
                    "dry_run": _truth(self.dry_run),
                    "live_llm": _truth(self.live_llm),
                    "description": family_descriptions[family],
                }
            )

    def _sample_anchors(self, family: str) -> List[str]:
        if not self.source_humans or not family.startswith("G2"):
            return []
        count = self.rng.randint(1, 3)
        return [row["user_id"] for row in self.rng.sample(self.source_humans, k=min(count, len(self.source_humans)))]

    def _make_bot(self, family: str, index: int, anchors: List[str]) -> BotState:
        base_post = self.rng.uniform(0.16, 0.34)
        base_reply = self.rng.uniform(0.35, 0.62)
        base_revisit = self.rng.uniform(0.08, 0.22)
        if family.startswith("G1"):
            base_post *= 1.0
            base_reply *= 0.9
        elif family.startswith("G2"):
            anchor_rates = self._anchor_rates(anchors)
            base_post = anchor_rates[0]
            base_reply = anchor_rates[1]
            base_revisit = anchor_rates[2]
        elif family.startswith("G3"):
            base_post *= 0.75
            base_reply *= 1.15
            base_revisit *= 1.8
        provider, model, temperature = self._llm_profile_for_bot(family, index)
        comment1_tendency = max(0.05, min(0.85, base_reply * (0.70 if not family.startswith("G3") else 0.76)))
        comment2_tendency = max(0.03, min(0.70, base_reply * (0.34 if not family.startswith("G2") else 0.42)))
        skip_tendency = max(0.04, min(0.45, 0.30 - min(base_reply, 0.8) * 0.18))
        preferred_topics = self._preferred_topics_for_bot(family, anchors)
        return BotState(
            user_id=f"gb_{family[:2].lower()}_{self.run_name}_{index:04d}",
            user_name=f"{family[:2].lower()}_dryrun_{index:04d}",
            family=family,
            anchors=anchors,
            post_tendency=min(base_post, 0.8),
            reply_tendency=min(base_reply, 0.85),
            revisit_tendency=min(base_revisit, 0.75),
            exploration_tendency=self.rng.uniform(0.35, 0.85),
            notification_reactivity=self.rng.uniform(0.35, 0.9) if family.startswith("G3") else self.rng.uniform(0.1, 0.55),
            topic_entropy_target=self.rng.uniform(0.35, 0.95),
            active_rhythm_type=self.rng.choice(["bursty", "steady", "evening_peak", "weekend_light"]),
            llm_provider=provider,
            llm_model=model,
            llm_temperature=temperature,
            llm_reasoning_effort=str(_config_get(self.config, ["llm", "openai", "reasoning_effort"], "medium") or "medium"),
            comment1_tendency=comment1_tendency,
            comment2_tendency=comment2_tendency,
            skip_tendency=skip_tendency,
            preferred_topics=preferred_topics,
            context_sensitivity=0.78 if family.startswith("G3") else (0.42 if family.startswith("G2") else 0.35),
            diversity_preference=0.60 if family.startswith("G3") else (0.35 if family.startswith("G2") else 0.45),
            memory_decay=self.rng.uniform(0.25, 0.85),
        )

    def _llm_profile_for_bot(self, family: str, index: int) -> Tuple[str, str, float]:
        openai_model = str(_config_get(self.config, ["llm", "openai", "model"], "gpt-5.4") or "gpt-5.4")
        gemini_model = str(_config_get(self.config, ["llm", "gemini", "model"], "gemini-3.5-flash") or "gemini-3.5-flash")
        temperatures = [0.4, 0.8]
        try:
            openai_temps = _config_get(self.config, ["llm", "openai", "temperatures"], None)
            gemini_temps = _config_get(self.config, ["llm", "gemini", "temperatures"], None)
            if isinstance(openai_temps, list) and openai_temps:
                temperatures = [float(x) for x in openai_temps]
            if isinstance(gemini_temps, list) and gemini_temps and index % 2 == 1:
                temperatures = [float(x) for x in gemini_temps]
        except Exception:
            temperatures = [0.4, 0.8]
        provider = "openai" if (FAMILIES.index(family) + index) % 2 == 0 else "gemini"
        model = openai_model if provider == "openai" else gemini_model
        temperature = temperatures[index % len(temperatures)]
        return provider, model, temperature

    def _preferred_topics_for_bot(self, family: str, anchors: List[str]) -> List[str]:
        topics: List[str] = []
        for anchor in anchors:
            user = self.user_by_id.get(anchor, {})
            topics.extend(_parse_subreddits(user.get("subreddit")))
        if not topics:
            sample_posts = self.rng.sample(self.posts, k=min(5, len(self.posts))) if self.posts else []
            topics.extend(str(p.get("subreddit") or p.get("topic") or "") for p in sample_posts)
        topics = [t for t in topics if t]
        if not topics:
            topics = ["worldnews", "news", "InternationalNews"]
        self.rng.shuffle(topics)
        return list(dict.fromkeys(topics))[:5]

    def _anchor_rates(self, anchors: List[str]) -> Tuple[float, float, float]:
        if not anchors:
            return 0.24, 0.48, 0.15
        post_counts = []
        reply_counts = []
        for anchor in anchors:
            posts = [p for p in self.posts if p.get("author_id") == anchor]
            comments = [c for c in self.comments if c.get("author_id") == anchor]
            post_counts.append(len(posts))
            reply_counts.append(len(comments))
        post_avg = sum(post_counts) / max(1, len(post_counts))
        reply_avg = sum(reply_counts) / max(1, len(reply_counts))
        total = max(1.0, post_avg + reply_avg)
        post_tendency = 0.12 + 0.45 * (post_avg / total)
        reply_tendency = 0.25 + 0.55 * (reply_avg / total)
        revisit = 0.10 + min(0.25, reply_avg / 80.0)
        return post_tendency, reply_tendency, revisit

    def _run_schedule(self) -> None:
        queue: List[Tuple[datetime, int, BotState]] = []
        sequence = 0
        for window in range(self.time_windows):
            start = self.start_time + timedelta(hours=window * self.window_hours)
            for bot_index, bot in enumerate(self.bots):
                family_offset = FAMILIES.index(bot.family) * (self.window_hours * 3600 / max(6, len(FAMILIES) * 4))
                jitter = self.rng.randint(0, max(1, int(self.window_hours * 3600 / 2)))
                bot_offset = (bot_index % max(1, self.bots_per_family)) * 17
                scheduled = start + timedelta(seconds=int((family_offset + jitter + bot_offset) % (self.window_hours * 3600)))
                heapq.heappush(queue, (scheduled, sequence, bot))
                sequence += 1

        while queue:
            now, _, bot = heapq.heappop(queue)
            self._step_bot(bot, now)

    def _step_bot(self, bot: BotState, now: datetime) -> None:
        candidates = self._browse_candidates(bot)
        browse_event_id = self._next_event_id()
        self._write_event(
            {
                "event_id": browse_event_id,
                "event_type": "browse",
                "time": _format_time(now),
                "actor_user_id": bot.user_id,
                "actor_family": bot.family,
                "target_post_id": "",
                "target_comment_id": "",
                "target_author_id": "",
                "content_id": "",
                "policy_reason": f"browse_count={self.browse_count}; shared mixed environment",
                "llm_call_id": "",
                "candidate_count": len(candidates),
                "candidate_summary": [
                    {
                        "candidate_type": c["candidate_type"],
                        "target_type": c["target_type"],
                        "target_id": c["target_id"],
                        "author_id": c.get("author_id", ""),
                        "family": c.get("family", ""),
                    }
                    for c in candidates
                ],
            }
        )
        self._write_memory(
            {
                "event_id": browse_event_id,
                "time": _format_time(now),
                "actor_user_id": bot.user_id,
                "actor_family": bot.family,
                "memory_type": "browse_context",
                "notification_queue_size": len(bot.notification_queue),
                "own_history_size": len(bot.own_history),
                "candidates": candidates,
            }
        )

        decision = self.policy.decide(bot, candidates, now)
        action_type = str(decision.get("action_type", "skip"))
        if action_type not in {"skip", "post", "comment1", "comment2", "revisit"}:
            action_type = "skip"
        target = decision.get("target") if isinstance(decision.get("target"), dict) else {}
        if action_type == "post":
            self._action_post(bot, now, decision)
        elif action_type in {"comment1", "comment2", "revisit"}:
            self._action_comment(bot, now, action_type, target, decision)
        else:
            self._action_skip(bot, now, decision)

    def _browse_candidates(self, bot: BotState) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        for item in bot.notification_queue[:2]:
            candidates.append({**item, "candidate_type": "notification"})
        for item in bot.own_history[-2:]:
            candidates.append({**item, "candidate_type": "own_history"})

        bot_posts = [p for p in self.posts if p.get("author_source") == "generated_bot" and p.get("author_id") != bot.user_id]
        bot_comments = [c for c in self.comments if c.get("author_source") == "generated_bot" and c.get("author_id") != bot.user_id]
        human_posts = [p for p in self.posts if p.get("author_source") == "original_human"]
        human_comments = [c for c in self.comments if c.get("author_source") == "original_human"]

        pools = [
            ("bot_new_post", "post", bot_posts),
            ("bot_new_comment", "comment", bot_comments),
            ("original_human_post", "post", human_posts),
            ("original_human_comment", "comment", human_comments),
        ]
        for candidate_type, target_type, pool in pools:
            if not pool:
                continue
            sample_count = min(max(1, self.browse_count // 4), len(pool))
            for row in self.rng.sample(pool, k=sample_count):
                candidates.append(self._candidate_from_row(candidate_type, target_type, row))

        self.rng.shuffle(candidates)
        return candidates[: self.browse_count]

    def _candidate_from_row(self, candidate_type: str, target_type: str, row: Dict[str, Any]) -> Dict[str, Any]:
        if target_type == "post":
            return {
                "candidate_type": candidate_type,
                "target_type": "post",
                "target_id": row.get("post_id", ""),
                "post_id": row.get("post_id", ""),
                "comment_id": "",
                "author_id": row.get("author_id", ""),
                "family": row.get("family", ""),
                "subreddit": row.get("subreddit", ""),
                "excerpt": _safe_text(row.get("title") or row.get("body"), 180),
            }
        return {
            "candidate_type": candidate_type,
            "target_type": "comment",
            "target_id": row.get("comment_id", ""),
            "post_id": row.get("post_id", ""),
            "comment_id": row.get("comment_id", ""),
            "author_id": row.get("author_id", ""),
            "family": row.get("family", ""),
            "subreddit": self.post_by_id.get(row.get("post_id", ""), {}).get("subreddit", ""),
            "excerpt": _safe_text(row.get("body"), 180),
        }

    def _action_post(self, bot: BotState, now: datetime, decision: Dict[str, Any]) -> None:
        event_id = self._next_event_id()
        self.post_counter += 1
        subreddit = self._bot_subreddit(bot)
        post_id = f"p_gen_{self.run_name}_{self.post_counter:06d}"
        context_ref = self._context_reference(bot)
        title, body, llm_call_id, llm_status = self._generate_post_text(bot, subreddit, context_ref)
        post = {
            "post_id": post_id,
            "author_id": bot.user_id,
            "author_source": "generated_bot",
            "family": bot.family,
            "subreddit": subreddit,
            "topic": subreddit,
            "created_utc": _format_time(now),
            "title": title,
            "body": body,
            "score": "1",
            "num_comments": "0",
            "upvote_ratio": "0.50",
            "is_generated": "true",
            "is_replay_context": "false",
        }
        self._add_post(post, source_event_id=event_id)
        history_item = self._candidate_from_row("own_history", "post", post)
        bot.own_history.append(history_item)
        decision = {**decision, "llm_call_id": llm_call_id, "llm_status": llm_status}
        self._write_action_and_event(bot, now, event_id, "post", decision, post_id, "", "", post_id)

    def _action_comment(
        self,
        bot: BotState,
        now: datetime,
        action_type: str,
        target: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> None:
        target_type = target.get("target_type")
        if action_type == "comment1" and target_type != "post":
            post_candidates = [c for c in self._browse_candidates(bot) if c["target_type"] == "post"]
            target = self.rng.choice(post_candidates) if post_candidates else {}
            target_type = target.get("target_type")
        if action_type in {"comment2", "revisit"} and target_type not in {"comment", "post"}:
            fallback = bot.notification_queue[:1] or bot.own_history[-1:] or self._browse_candidates(bot)
            target = fallback[0] if fallback else {}
            target_type = target.get("target_type")
        if not target:
            self._action_skip(bot, now, {"policy_reason": f"{action_type} had no valid target"})
            return

        if action_type == "comment1" or target_type == "post":
            parent_type = "post"
            parent_id = target.get("post_id") or target.get("target_id")
            post_id = parent_id
            level = "1"
        else:
            parent_type = "comment"
            parent_id = target.get("comment_id") or target.get("target_id")
            parent_comment = self.comment_by_id.get(parent_id, {})
            post_id = target.get("post_id") or parent_comment.get("post_id", "")
            level = "2"

        if not post_id:
            self._action_skip(bot, now, {"policy_reason": f"{action_type} missing post_id"})
            return

        event_id = self._next_event_id()
        self.comment_counter += 1
        comment_id = f"c_gen_{self.run_name}_{self.comment_counter:06d}"
        target_author_id = target.get("author_id") or self._target_author_id(parent_type, parent_id)
        body, llm_call_id, llm_status = self._generate_comment_text(
            bot=bot,
            action_type=action_type,
            target=target,
            parent_type=parent_type,
            parent_id=parent_id,
            post_id=post_id,
            target_author_id=target_author_id,
        )
        comment = {
            "comment_id": comment_id,
            "author_id": bot.user_id,
            "author_source": "generated_bot",
            "family": bot.family,
            "post_id": post_id,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "level": level,
            "created_utc": _format_time(now),
            "body": body,
            "score": "1",
            "is_generated": "true",
            "is_replay_human": "false",
            "action_type": action_type,
            "target_author_id": target_author_id,
        }
        self._add_comment(comment, source_event_id=event_id)
        history_item = self._candidate_from_row("own_history", "comment", comment)
        bot.own_history.append(history_item)
        self._maybe_notify_bot(target_author_id, bot, comment, now, event_id)
        decision = {**decision, "llm_call_id": llm_call_id, "llm_status": llm_status}
        self._write_action_and_event(bot, now, event_id, action_type, decision, post_id, parent_id if parent_type == "comment" else "", target_author_id, comment_id)

    def _action_skip(self, bot: BotState, now: datetime, decision: Dict[str, Any]) -> None:
        event_id = self._next_event_id()
        self._write_action_and_event(bot, now, event_id, "skip", decision, "", "", "", "")

    def _maybe_notify_bot(
        self,
        target_author_id: str,
        actor: BotState,
        comment: Dict[str, Any],
        now: datetime,
        event_id: str,
    ) -> None:
        if not target_author_id or target_author_id == actor.user_id:
            return
        target_bot = next((bot for bot in self.bots if bot.user_id == target_author_id), None)
        if not target_bot:
            return
        item = self._candidate_from_row("notification", "comment", comment)
        item["from_user_id"] = actor.user_id
        target_bot.notification_queue.append(item)
        notify_event_id = self._next_event_id()
        self._write_event(
            {
                "event_id": notify_event_id,
                "event_type": "notification_created",
                "time": _format_time(now),
                "actor_user_id": target_author_id,
                "actor_family": target_bot.family,
                "target_post_id": comment.get("post_id", ""),
                "target_comment_id": comment.get("comment_id", ""),
                "target_author_id": actor.user_id,
                "content_id": comment.get("comment_id", ""),
                "policy_reason": f"generated reply from {actor.family}",
                "llm_call_id": "",
                "source_event_id": event_id,
            }
        )

    def _write_action_and_event(
        self,
        bot: BotState,
        now: datetime,
        event_id: str,
        action_type: str,
        decision: Dict[str, Any],
        target_post_id: str,
        target_comment_id: str,
        target_author_id: str,
        content_id: str,
    ) -> None:
        policy_reason = str(decision.get("policy_reason", f"dry-run {action_type}"))
        self._write_event(
            {
                "event_id": event_id,
                "event_type": action_type,
                "time": _format_time(now),
                "actor_user_id": bot.user_id,
                "actor_family": bot.family,
                "target_post_id": target_post_id,
                "target_comment_id": target_comment_id,
                "target_author_id": target_author_id,
                "content_id": content_id,
                "policy_reason": policy_reason,
                "llm_call_id": str(decision.get("llm_call_id", "")),
                "llm_status": str(decision.get("llm_status", "")),
            }
        )
        self._write_action(
            {
                "event_id": event_id,
                "time": _format_time(now),
                "actor_user_id": bot.user_id,
                "actor_family": bot.family,
                "action_type": action_type,
                "target_post_id": target_post_id,
                "target_comment_id": target_comment_id,
                "target_author_id": target_author_id,
                "content_id": content_id,
                "dry_run": self.dry_run,
                "policy_reason": policy_reason,
                "llm_call_id": str(decision.get("llm_call_id", "")),
                "llm_status": str(decision.get("llm_status", "")),
            }
        )
        self._write_memory(
            {
                "event_id": event_id,
                "time": _format_time(now),
                "actor_user_id": bot.user_id,
                "actor_family": bot.family,
                "memory_type": "post_action_state",
                "action_type": action_type,
                "notification_queue_size": len(bot.notification_queue),
                "own_history_size": len(bot.own_history),
            }
        )

    def _write_event(self, row: Dict[str, Any]) -> None:
        _append_jsonl(self.output_dir / "events.jsonl", row)

    def _write_action(self, row: Dict[str, Any]) -> None:
        _append_jsonl(self.output_dir / "action_trace.jsonl", row)

    def _write_memory(self, row: Dict[str, Any]) -> None:
        _append_jsonl(self.output_dir / "memory_trace.jsonl", row)

    def _next_event_id(self) -> str:
        self.event_counter += 1
        return f"evt_{self.run_name}_{self.event_counter:08d}"

    def _target_author_id(self, parent_type: str, parent_id: str) -> str:
        if parent_type == "post":
            return str(self.post_by_id.get(parent_id, {}).get("author_id", ""))
        if parent_type == "comment":
            return str(self.comment_by_id.get(parent_id, {}).get("author_id", ""))
        return ""

    def _sample_subreddit(self, user: Dict[str, Any]) -> str:
        values = _parse_subreddits(user.get("subreddit"))
        return self.rng.choice(values) if values else "unknown"

    def _bot_subreddit(self, bot: BotState) -> str:
        if bot.anchors:
            anchor = self.user_by_id.get(bot.anchors[0], {})
            sampled = self._sample_subreddit(anchor)
            if sampled != "unknown":
                return sampled
        if self.posts:
            return str(self.rng.choice(self.posts).get("subreddit", "mixed"))
        return "mixed"

    def _context_reference(self, bot: BotState) -> str:
        if bot.notification_queue:
            return bot.notification_queue[-1].get("target_id", "notification")
        if bot.own_history:
            return bot.own_history[-1].get("target_id", "own_history")
        if self.posts:
            return str(self.rng.choice(self.posts).get("post_id", "background"))
        return "empty_environment"

    def _next_llm_call_id(self) -> str:
        self.llm_call_counter += 1
        return f"llm_{self.run_name}_{self.llm_call_counter:06d}"

    def _persona_prompt_block(self, bot: BotState) -> str:
        return (
            f"Bot id: {bot.user_id}\n"
            f"Family: {bot.family}\n"
            f"Anchors: {', '.join(bot.anchors) if bot.anchors else 'none'}\n"
            f"Behavior tendencies: post={bot.post_tendency:.2f}, reply={bot.reply_tendency:.2f}, "
            f"comment1={bot.comment1_tendency:.2f}, comment2={bot.comment2_tendency:.2f}, "
            f"revisit={bot.revisit_tendency:.2f}, skip={bot.skip_tendency:.2f}\n"
            f"Style: casual Reddit-like, specific, not essay-like. Preferred topics: {', '.join(bot.preferred_topics)}.\n"
            "Write like a normal social media user. Be natural, concise, opinionated when appropriate, and avoid polished policy-summary language."
        )

    def _generate_post_text(self, bot: BotState, subreddit: str, context_ref: str) -> Tuple[str, str, str, str]:
        fallback_title = f"Question about {subreddit}"
        fallback_body = f"i keep seeing this come up around {subreddit}. curious what people here actually think about it."
        if not self.live_llm or self.llm_client is None:
            return (
                f"{bot.family} dry-run post about {subreddit} after browsing {context_ref}",
                f"Dry-run generated by {bot.family}; context_reference={context_ref}; anchors={','.join(bot.anchors) or 'none'}.",
                "",
                "dry_run",
            )
        prompt = (
            "Return only JSON with keys title and body.\n"
            "You are writing a new Reddit-style post for an offline simulation dataset.\n"
            "Do not mention bots, simulation, prompts, datasets, or AI.\n"
            f"{self._persona_prompt_block(bot)}\n"
            f"Subreddit/topic: {subreddit}\n"
            f"Recent context reference: {context_ref}\n"
            "Title should be one natural post title. Body should be 1-3 casual sentences."
        )
        call_id = self._next_llm_call_id()
        result = self.llm_client.generate(
            provider=bot.llm_provider,
            model=bot.llm_model,
            temperature=bot.llm_temperature,
            prompt=prompt,
            fallback=json.dumps({"title": fallback_title, "body": fallback_body}, ensure_ascii=False),
            preferred_keys=("title", "body", "text", "content"),
        )
        title, body = fallback_title, fallback_body
        try:
            parsed = json.loads(result.parsed_text)
            if isinstance(parsed, dict):
                title = _safe_text(parsed.get("title") or fallback_title, 180)
                body = _safe_text(parsed.get("body") or fallback_body, 600)
            else:
                body = _safe_text(result.parsed_text or fallback_body, 600)
        except Exception:
            body = _safe_text(result.parsed_text or fallback_body, 600)
        self._write_llm_trace(call_id, "post", bot, prompt, result)
        return title, body, call_id, result.status

    def _generate_comment_text(
        self,
        *,
        bot: BotState,
        action_type: str,
        target: Dict[str, Any],
        parent_type: str,
        parent_id: str,
        post_id: str,
        target_author_id: str,
    ) -> Tuple[str, str, str]:
        fallback = "yeah, i don't think it's that simple. feels like there is a lot more context here than people are admitting."
        if not self.live_llm or self.llm_client is None:
            return (
                f"{bot.family} dry-run {action_type} referencing {parent_type} {parent_id} "
                f"in post {post_id}; context_excerpt={_safe_text(target.get('excerpt'), 140)}; anchors={','.join(bot.anchors) or 'none'}.",
                "",
                "dry_run",
            )
        post = self.post_by_id.get(post_id, {})
        parent_comment = self.comment_by_id.get(parent_id, {}) if parent_type == "comment" else {}
        own_participation = "yes" if any(item.get("post_id") == post_id for item in bot.own_history) else "no"
        prompt = (
            "Return only JSON with key text.\n"
            "You are writing one Reddit-style comment for an offline simulation dataset.\n"
            "Do not mention bots, simulation, prompts, datasets, or AI.\n"
            f"{self._persona_prompt_block(bot)}\n"
            f"Action type: {action_type}\n"
            f"Post title: {_safe_text(post.get('title'), 240)}\n"
            f"Post body: {_safe_text(post.get('body'), 420)}\n"
            f"Target excerpt: {_safe_text(target.get('excerpt'), 240)}\n"
            f"Parent comment: {_safe_text(parent_comment.get('body'), 300)}\n"
            f"Target author id: {target_author_id}\n"
            f"Has this bot participated in this thread before: {own_participation}\n"
            "Write 1-3 sentences. Make it sound like a real comment, not a formal analysis."
        )
        call_id = self._next_llm_call_id()
        result = self.llm_client.generate(
            provider=bot.llm_provider,
            model=bot.llm_model,
            temperature=bot.llm_temperature,
            prompt=prompt,
            fallback=fallback,
            preferred_keys=("text", "comment", "body", "content"),
        )
        body = _safe_text(result.parsed_text or fallback, 700)
        self._write_llm_trace(call_id, action_type, bot, prompt, result)
        return body, call_id, result.status

    def _write_llm_trace(self, call_id: str, call_type: str, bot: BotState, prompt: str, result: Any) -> None:
        _append_jsonl(
            self.output_dir / "llm_trace.jsonl",
            {
                "llm_call_id": call_id,
                "call_type": call_type,
                "actor_user_id": bot.user_id,
                "actor_family": bot.family,
                "provider": result.provider,
                "model": result.model,
                "temperature": result.temperature,
                "status": result.status,
                "error": result.error,
                "fallback_reason": result.fallback_reason,
                "prompt": prompt,
                "raw_response": result.raw_response,
                "parsed_text": result.parsed_text,
            },
        )

    def _write_outputs(self, dataset_dir: Path) -> None:
        _write_csv(self.output_dir / "users.csv", self.users, USER_FIELDS)
        _write_csv(self.output_dir / "posts.csv", self.posts, POST_FIELDS)
        _write_csv(self.output_dir / "comments.csv", self.comments, COMMENT_FIELDS)
        _write_csv(self.output_dir / "persona_profiles.csv", self.personas, PERSONA_FIELDS)
        _write_csv(self.output_dir / "family_manifest.csv", self.family_manifest, FAMILY_MANIFEST_FIELDS)
        _write_csv(self.output_dir / "labels.csv", self.labels, LABEL_FIELDS)
        _write_csv(self.output_dir / "human_anchor_bindings.csv", self.anchor_bindings, ANCHOR_FIELDS)
        _write_csv(self.output_dir / "edges.csv", self.edges, EDGE_FIELDS)
        _write_csv(self.output_dir / "human_replay_manifest.csv", [], [
            "replay_comment_id",
            "replay_author_id",
            "target_type",
            "target_id",
            "target_author_id",
            "target_family",
            "post_id",
            "source_pattern_comment_id",
            "source_pattern_author_id",
            "source_pattern_level",
            "source_pattern_body_excerpt",
            "created_utc",
            "seed",
        ])
        if not self.live_llm:
            _append_jsonl(
                self.output_dir / "llm_trace.jsonl",
                {
                    "run_name": self.run_name,
                    "dry_run": self.dry_run,
                    "live_llm": self.live_llm,
                    "llm_call_id": "",
                    "provider": "",
                    "model": "",
                    "note": "dry-run generated by templates; no external API call",
                },
            )

    def _apply_replay(self) -> Dict[str, Any]:
        replay_cfg = _config_get(self.config, ["generation", "human_reply_replay"], {}) or {}
        if replay_cfg.get("enabled", True) is False:
            return {"created": 0, "reason": "disabled"}
        try:
            sys.path.insert(0, str(CODE_DIR))
            from human_reply_replay import apply_human_reply_replay

            return apply_human_reply_replay(
                self.output_dir,
                seed=self.seed + 101,
                max_replay_per_bot_post=_as_int(replay_cfg.get("max_replay_per_bot_post"), 2),
                max_replay_per_bot_comment=_as_int(replay_cfg.get("max_replay_per_bot_comment"), 1),
                run_name=self.run_name,
            )
        except Exception as exc:
            return {"created": 0, "reason": f"replay_failed: {exc}"}

    def _write_manifest(self, dataset_dir: Path, replay_summary: Dict[str, Any]) -> None:
        _write_json(
            self.output_dir / "dataset_manifest.json",
            {
                "schema_version": str(_config_get(self.config, ["schema_version"], "static-graph-network-dataset-v1")),
                "run_name": self.run_name,
                "created_at": _format_time(datetime.now()),
                "bot_families": FAMILIES,
                "bots_per_family": self.bots_per_family,
                "family_counts": self.family_counts,
                "environment": "mixed_dynamic_reddit_offline",
                "source_dataset": str(dataset_dir),
                "dry_run": self.dry_run,
                "live_llm": self.live_llm,
                "seed": self.seed,
                "time_windows": self.time_windows,
                "browse_count": self.browse_count,
                "policy_source": self.policy.policy_source,
                "human_reply_replay": replay_summary,
                "notes": "First runnable mixed runner; background is loaded from project-local BotSim JSON snapshot when available.",
            },
        )


def _as_bool_like(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a dry-run mixed dynamic static-graph dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to graph_static_dataset_v1.yaml.")
    parser.add_argument("--output-dir", required=True, help="Directory for all run outputs.")
    parser.add_argument("--run-name", default="", help="Run name used in generated ids and manifests.")
    parser.add_argument("--bots-per-family", type=int, default=None, help="Override generated bot count per family.")
    parser.add_argument("--family-counts", default="", help="Optional comma-separated family counts, e.g. G1=4,G2=3,G3=3.")
    parser.add_argument("--time-windows", type=int, default=None, help="Override number of time windows.")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Generate template text without API calls. Default: true.")
    parser.add_argument("--live-llm", action="store_true", help="Call real GPT/Gemini APIs for generated text.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args(argv)


def parse_family_counts(raw: str, default_count: int) -> Dict[str, int]:
    if not raw.strip():
        return {family: default_count for family in FAMILIES}
    alias = {
        "G1": "G1_population_calibrated",
        "G2": "G2_anchor_behavior",
        "G3": "G3_contextual_interactive",
    }
    counts = {family: 0 for family in FAMILIES}
    for part in raw.split(","):
        if "=" not in part:
            raise ValueError(f"Invalid family count segment: {part!r}")
        key, value = part.split("=", 1)
        family = alias.get(key.strip(), key.strip())
        if family not in counts:
            raise ValueError(f"Unknown family in --family-counts: {key}")
        counts[family] = _as_int(value, 0)
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config)
    bots_per_family = args.bots_per_family
    if bots_per_family is None:
        family_cfg = _config_get(config, ["generation", "families"], {}) or {}
        counts = [_as_int((family_cfg.get(family) or {}).get("bots"), 50) for family in FAMILIES]
        bots_per_family = min(counts) if counts else 50
    time_windows = args.time_windows
    if time_windows is None:
        time_windows = _as_int(_config_get(config, ["generation", "time", "windows"], 8), 8)
    seed = args.seed if args.seed is not None else _as_int(_config_get(config, ["runtime", "seed"], 20260603), 20260603)
    family_counts = parse_family_counts(args.family_counts, bots_per_family)

    runner = MixedEnvironment(
        config=config,
        output_dir=Path(args.output_dir),
        bots_per_family=bots_per_family,
        time_windows=time_windows,
        dry_run=args.dry_run,
        seed=seed,
        family_counts=family_counts,
        live_llm=args.live_llm,
        run_name=args.run_name,
    )
    summary = runner.run()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
