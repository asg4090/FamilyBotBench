#!/usr/bin/env python3
"""Materialize a partial live run from JSONL traces.

The live mixed runner writes trace files during execution, but only writes the
CSV dataset tables after the full schedule finishes. This utility reconstructs
the standard dataset entry files from a trace snapshot so an interrupted or
early-stopped run can still be exported and audited.
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
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from mixed_environment_runner import (  # noqa: E402
    COMMENT_FIELDS,
    DEFAULT_CONFIG,
    FAMILIES,
    MixedEnvironment,
    _as_int,
    _config_get,
    _load_yaml_or_minimal,
    _safe_text,
    parse_family_counts,
)


TRACE_NAMES = ("action_trace.jsonl", "events.jsonl", "llm_trace.jsonl", "memory_trace.jsonl")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path, help="Directory containing trace JSONL files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write materialized CSV files.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), type=Path, help="Runner config YAML.")
    parser.add_argument("--source-run-name", default="", help="Original run_name embedded in generated ids.")
    parser.add_argument("--family-counts", default="G1=50,G2=50,G3=50", help="Family counts used by the original run.")
    parser.add_argument("--bots-per-family", type=int, default=1, help="Original --bots-per-family value.")
    parser.add_argument("--time-windows", type=int, default=24, help="Original time window count.")
    parser.add_argument("--seed", type=int, default=20260609, help="Original seed.")
    parser.add_argument("--copy-traces", action="store_true", help="Copy trace files from trace-dir into output-dir first.")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    if not path.exists():
        return rows, [{"line_no": 0, "error": "missing_file", "path": str(path)}]
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append({"line_no": line_no, "error": str(exc), "excerpt": stripped[:240]})
                continue
            if isinstance(item, dict):
                rows.append(item)
            else:
                errors.append({"line_no": line_no, "error": "json_value_is_not_object", "excerpt": stripped[:240]})
    return rows, errors


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def strip_fenced_json(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def maybe_json_dict(text: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_fenced_json(text)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def text_from_llm(llm_row: Dict[str, Any], keys: Iterable[str], fallback: str) -> str:
    parsed_text = str(llm_row.get("parsed_text") or "").strip()
    raw_response = str(llm_row.get("raw_response") or "").strip()
    for candidate in (parsed_text, raw_response):
        payload = maybe_json_dict(candidate)
        if payload:
            for key in keys:
                value = payload.get(key)
                if value:
                    return str(value)
        cleaned = strip_fenced_json(candidate)
        if cleaned:
            return cleaned
    return fallback


def post_text_from_llm(llm_row: Dict[str, Any]) -> Tuple[str, str]:
    fallback_title = "Question about this topic"
    fallback_body = "i keep seeing this come up and wanted to hear what other people think."
    parsed_text = str(llm_row.get("parsed_text") or "").strip()
    raw_response = str(llm_row.get("raw_response") or "").strip()
    for candidate in (parsed_text, raw_response):
        payload = maybe_json_dict(candidate)
        if payload:
            title = _safe_text(payload.get("title") or fallback_title, 180)
            body = _safe_text(payload.get("body") or payload.get("text") or fallback_body, 800)
            return title, body
    return fallback_title, _safe_text(parsed_text or raw_response or fallback_body, 800)


def subreddit_from_prompt(llm_row: Dict[str, Any]) -> str:
    prompt = str(llm_row.get("prompt") or "")
    match = re.search(r"Subreddit/topic:\s*(.+)", prompt)
    if match:
        value = match.group(1).splitlines()[0].strip()
        return _safe_text(value, 80) or "mixed"
    return "mixed"


def infer_source_run_name(actions: Sequence[Dict[str, Any]]) -> str:
    patterns = [
        re.compile(r"^evt_(.+)_\d{8}$"),
        re.compile(r"^p_gen_(.+)_\d{6}$"),
        re.compile(r"^c_gen_(.+)_\d{6}$"),
        re.compile(r"^llm_(.+)_\d{6}$"),
    ]
    for row in actions:
        for key in ("event_id", "content_id", "llm_call_id"):
            value = str(row.get(key) or "")
            for pattern in patterns:
                match = pattern.match(value)
                if match:
                    return match.group(1)
    return "partial_trace_run"


def copy_trace_files(trace_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in TRACE_NAMES:
        src = trace_dir / name
        dst = output_dir / name
        if src.resolve() == dst.resolve():
            continue
        if src.exists():
            shutil.copy2(src, dst)


def build_runner(args: argparse.Namespace, config: Dict[str, Any], source_run_name: str) -> MixedEnvironment:
    family_counts = parse_family_counts(args.family_counts, args.bots_per_family)
    runner = MixedEnvironment(
        config=config,
        output_dir=args.output_dir,
        bots_per_family=args.bots_per_family,
        time_windows=args.time_windows,
        dry_run=False,
        seed=args.seed,
        family_counts=family_counts,
        live_llm=False,
        run_name=source_run_name,
    )
    runner.live_llm = True
    runner.dry_run = False
    dataset_dir = runner._resolve_dataset_dir()
    runner._load_background(dataset_dir)
    runner._create_bots()
    return runner


def materialize_generated_rows(
    runner: MixedEnvironment,
    actions: Sequence[Dict[str, Any]],
    llm_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    missing_llm: List[Dict[str, str]] = []
    missing_targets: List[Dict[str, str]] = []

    for action in actions:
        action_type = str(action.get("action_type") or "").strip().lower()
        if action_type not in {"post", "comment1", "comment2", "revisit"}:
            counts[action_type or "unknown"] += 1
            continue

        event_id = str(action.get("event_id") or "")
        actor_user_id = str(action.get("actor_user_id") or "")
        actor_family = str(action.get("actor_family") or "")
        created = str(action.get("time") or "")
        content_id = str(action.get("content_id") or "")
        llm_call_id = str(action.get("llm_call_id") or "")
        llm_row = llm_by_id.get(llm_call_id, {})
        if not llm_row and llm_call_id:
            missing_llm.append({"event_id": event_id, "llm_call_id": llm_call_id})

        if action_type == "post":
            if not content_id:
                missing_targets.append({"event_id": event_id, "reason": "post_missing_content_id"})
                continue
            title, body = post_text_from_llm(llm_row)
            subreddit = subreddit_from_prompt(llm_row)
            runner._add_post(
                {
                    "post_id": content_id,
                    "author_id": actor_user_id,
                    "author_source": "generated_bot",
                    "family": actor_family,
                    "subreddit": subreddit,
                    "topic": subreddit,
                    "created_utc": created,
                    "title": title,
                    "body": body,
                    "score": "1",
                    "num_comments": "0",
                    "upvote_ratio": "0.50",
                    "is_generated": "true",
                    "is_replay_context": "false",
                },
                source_event_id=event_id,
            )
            counts["post_materialized"] += 1
            continue

        post_id = str(action.get("target_post_id") or "")
        target_comment_id = str(action.get("target_comment_id") or "")
        if action_type == "comment1" or not target_comment_id:
            parent_type = "post"
            parent_id = post_id
            level = "1"
        else:
            parent_type = "comment"
            parent_id = target_comment_id
            level = "2"
            if not post_id:
                post_id = str(runner.comment_by_id.get(parent_id, {}).get("post_id", ""))
        if not content_id or not post_id or not parent_id:
            missing_targets.append(
                {
                    "event_id": event_id,
                    "reason": "comment_missing_content_or_parent",
                    "content_id": content_id,
                    "post_id": post_id,
                    "parent_id": parent_id,
                }
            )
            continue
        target_author_id = str(action.get("target_author_id") or runner._target_author_id(parent_type, parent_id))
        body = _safe_text(
            text_from_llm(
                llm_row,
                ("text", "comment", "body", "content"),
                "yeah, i don't think it's that simple. there's more context here than people are admitting.",
            ),
            900,
        )
        runner._add_comment(
            {
                "comment_id": content_id,
                "author_id": actor_user_id,
                "author_source": "generated_bot",
                "family": actor_family,
                "post_id": post_id,
                "parent_id": parent_id,
                "parent_type": parent_type,
                "level": level,
                "created_utc": created,
                "body": body,
                "score": "1",
                "is_generated": "true",
                "is_replay_human": "false",
                "action_type": action_type,
                "target_author_id": target_author_id,
            },
            source_event_id=event_id,
        )
        counts[f"{action_type}_materialized"] += 1

    return {
        "materialized_counts": dict(counts),
        "missing_llm": missing_llm[:50],
        "missing_llm_count": len(missing_llm),
        "missing_targets": missing_targets[:50],
        "missing_targets_count": len(missing_targets),
    }


def write_partial_note(output_dir: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Partial Trace Materialization Report",
        "",
        f"- output_dir: `{output_dir}`",
        f"- source_run_name: `{report.get('source_run_name')}`",
        f"- created_at: `{report.get('created_at')}`",
        f"- actions_read: `{report.get('actions_read')}`",
        f"- llm_rows_read: `{report.get('llm_rows_read')}`",
        f"- generated_posts: `{report.get('generated_posts')}`",
        f"- generated_comments: `{report.get('generated_comments')}`",
        f"- generated_users: `{report.get('generated_users')}`",
        f"- invalid_jsonl_lines: `{report.get('invalid_jsonl_line_count')}`",
        f"- missing_llm_count: `{report.get('missing_llm_count')}`",
        f"- missing_targets_count: `{report.get('missing_targets_count')}`",
        "",
        "## Action Counts",
        "",
    ]
    for key, value in sorted((report.get("action_counts") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Materialized Counts", ""])
    for key, value in sorted((report.get("materialized_counts") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    (output_dir / "partial_materialization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.trace_dir = args.trace_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_traces:
        copy_trace_files(args.trace_dir, args.output_dir)

    action_rows, action_errors = read_jsonl(args.output_dir / "action_trace.jsonl")
    llm_rows, llm_errors = read_jsonl(args.output_dir / "llm_trace.jsonl")
    event_rows, event_errors = read_jsonl(args.output_dir / "events.jsonl")
    memory_rows, memory_errors = read_jsonl(args.output_dir / "memory_trace.jsonl")
    errors = {
        "action_trace": action_errors,
        "llm_trace": llm_errors,
        "events": event_errors,
        "memory_trace": memory_errors,
    }

    source_run_name = args.source_run_name or infer_source_run_name(action_rows)
    config = _load_yaml_or_minimal(args.config)
    runner = build_runner(args, config, source_run_name)
    dataset_dir = runner._resolve_dataset_dir()
    llm_by_id = {str(row.get("llm_call_id")): row for row in llm_rows if row.get("llm_call_id")}
    action_counts = Counter(str(row.get("action_type") or "") for row in action_rows)
    materialized = materialize_generated_rows(runner, action_rows, llm_by_id)

    runner._write_outputs(dataset_dir)
    replay_summary = {"created": 0, "reason": "partial materialization from live trace; human replay not applied"}
    runner._write_manifest(dataset_dir, replay_summary)

    generated_users = [row for row in runner.users if row.get("source") == "generated_bot"]
    generated_posts = [row for row in runner.posts if str(row.get("is_generated")).lower() == "true"]
    generated_comments = [row for row in runner.comments if str(row.get("is_generated")).lower() == "true"]
    invalid_jsonl_line_count = sum(len(value) for value in errors.values())
    report = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_run_name": source_run_name,
        "trace_dir": str(args.trace_dir),
        "output_dir": str(args.output_dir),
        "config": str(args.config),
        "seed": args.seed,
        "time_windows": args.time_windows,
        "family_counts": parse_family_counts(args.family_counts, args.bots_per_family),
        "actions_read": len(action_rows),
        "events_read": len(event_rows),
        "memory_rows_read": len(memory_rows),
        "llm_rows_read": len(llm_rows),
        "action_counts": dict(action_counts),
        "generated_users": len(generated_users),
        "generated_posts": len(generated_posts),
        "generated_comments": len(generated_comments),
        "invalid_jsonl_line_count": invalid_jsonl_line_count,
        "jsonl_errors": errors,
        **materialized,
    }
    write_json(args.output_dir / "partial_materialization_report.json", report)
    write_partial_note(args.output_dir, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
