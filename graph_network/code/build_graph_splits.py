#!/usr/bin/env python3
"""Build static graph generalization splits for one run directory."""

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
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REQUIRED_INPUTS = (
    "users.csv",
    "posts.csv",
    "comments.csv",
    "events.jsonl",
    "labels.csv",
)

DEFAULT_FAMILIES = (
    "G1_population_calibrated",
    "G2_anchor_behavior",
    "G3_contextual_interactive",
    "G4_slot_conditioned",
)

LABEL_SCARCITY_RATIOS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)

CURRICULUM_ORDERS = (
    ("G1_population_calibrated", "G2_anchor_behavior", "G3_contextual_interactive"),
    ("G2_anchor_behavior", "G1_population_calibrated", "G3_contextual_interactive"),
    ("G3_contextual_interactive", "G1_population_calibrated", "G2_anchor_behavior"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build zero-shot, LOFO, label scarcity, time-forward, and curriculum splits."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Run directory containing users.csv/posts.csv/comments.csv/events.jsonl/labels.csv.",
    )
    parser.add_argument("--seed", type=int, default=20260603, help="Deterministic sampling seed.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of training candidates moved to validation.",
    )
    return parser.parse_args()


def require_run_file(run_dir: Path, name: str) -> Path:
    path = (run_dir / name).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing inside run-dir: {name}")
    if run_dir.resolve() not in path.parents:
        raise ValueError(f"Resolved input path escaped run-dir: {path}")
    return path


def ensure_output_dir(run_dir: Path, name: str) -> Path:
    path = (run_dir / name).resolve()
    if run_dir.resolve() not in path.parents:
        raise ValueError(f"Resolved output path escaped run-dir: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in reader]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path.name}:{line_no}: {exc}") from exc
            if isinstance(item, dict):
                events.append(item)
    return events


def first_nonempty(row: Dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key, "")
        if value is not None and str(value) != "":
            return str(value)
    return default


def normalize_label(value: Any) -> int:
    text = "" if value is None else str(value).strip().lower()
    if text in {"1", "bot", "true"}:
        return 1
    if text in {"0", "human", "false"}:
        return 0
    if text in {"-1", "unknown", "unlabeled", ""}:
        return -1
    try:
        return int(float(text))
    except ValueError:
        return -1


def normalize_known(value: Any, label: int) -> int:
    text = "" if value is None else str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    if text in {"0", "false", "no"}:
        return 0
    return 1 if label in {0, 1} else 0


def parse_time(value: Any) -> Optional[datetime]:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def split_known_train_val(
    node_ids: Sequence[str],
    records: Dict[str, Dict[str, Any]],
    rng: random.Random,
    val_fraction: float,
) -> Tuple[List[str], List[str]]:
    by_label: Dict[int, List[str]] = {}
    for node_id in node_ids:
        label = int(records[node_id]["label"])
        by_label.setdefault(label, []).append(node_id)

    train: List[str] = []
    val: List[str] = []
    for label_ids in by_label.values():
        shuffled = sorted(label_ids)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            train.extend(shuffled)
            continue
        val_count = int(round(len(shuffled) * val_fraction))
        val_count = max(1, min(val_count, len(shuffled) - 1))
        val.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:])
    return sorted(train), sorted(val)


def without_overlap(
    all_user_ids: Sequence[str],
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str],
) -> List[str]:
    used = set(train) | set(val) | set(test)
    return sorted(node_id for node_id in all_user_ids if node_id not in used)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def family_prefix(name: str) -> str:
    return name.split("_", 1)[0].lower()


def ratio_name(ratio: float) -> str:
    return f"{int(round(ratio * 100)):03d}pct"


def build_user_records(
    users: Sequence[Dict[str, str]],
    labels: Sequence[Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    label_lookup: Dict[str, Dict[str, Any]] = {}
    for row in labels:
        node_id = first_nonempty(row, ("node_id", "user_id"))
        if not node_id:
            continue
        label = normalize_label(row.get("label"))
        label_lookup[node_id] = {
            "label": label,
            "label_known": normalize_known(row.get("label_known"), label),
            "source": first_nonempty(row, ("source",)),
            "family": first_nonempty(row, ("family",)),
            "split_group": first_nonempty(row, ("split_group",)),
        }

    records: Dict[str, Dict[str, Any]] = {}
    for user in users:
        node_id = first_nonempty(user, ("user_id", "node_id"))
        if not node_id:
            continue
        label_info = label_lookup.get(node_id, {})
        label = normalize_label(label_info.get("label", user.get("label", -1)))
        records[node_id] = {
            "node_id": node_id,
            "label": label,
            "label_known": normalize_known(label_info.get("label_known"), label),
            "source": first_nonempty(label_info, ("source",), first_nonempty(user, ("source",))),
            "family": first_nonempty(label_info, ("family",), first_nonempty(user, ("family",))),
        }
    return records


def known(records: Dict[str, Dict[str, Any]], node_ids: Iterable[str]) -> List[str]:
    return sorted(
        node_id
        for node_id in node_ids
        if int(records[node_id]["label_known"]) == 1 and int(records[node_id]["label"]) in {0, 1}
    )


def original_nodes(records: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted(
        node_id
        for node_id, rec in records.items()
        if str(rec["source"]).startswith("original_")
        or str(rec["family"]).lower() in {"", "original", "background"}
    )


def nodes_in_families(records: Dict[str, Dict[str, Any]], families: Iterable[str]) -> List[str]:
    family_set = set(families)
    return sorted(node_id for node_id, rec in records.items() if rec["family"] in family_set)


def generated_family_nodes(records: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted(
        node_id
        for node_id, rec in records.items()
        if str(rec["family"]).startswith("G")
        and int(rec["label_known"]) == 1
        and int(rec["label"]) == 1
    )


def discover_families(records: Dict[str, Dict[str, Any]]) -> List[str]:
    found = sorted({str(rec["family"]) for rec in records.values() if str(rec["family"]).startswith("G")})
    ordered: List[str] = [family for family in DEFAULT_FAMILIES if family in found]
    ordered.extend(family for family in found if family not in ordered)
    return ordered or list(DEFAULT_FAMILIES)


def find_family_by_prefix(families: Sequence[str], prefix: str) -> str:
    for family in families:
        if family_prefix(family) == prefix:
            return family
    for default in DEFAULT_FAMILIES:
        if family_prefix(default) == prefix:
            return default
    raise ValueError(f"Cannot resolve family prefix: {prefix}")


def resolve_curriculum_orders(families: Sequence[str]) -> List[Tuple[str, ...]]:
    """Resolve configured orders and append newly discovered families such as G4."""
    resolved_orders: List[Tuple[str, ...]] = []
    for order in CURRICULUM_ORDERS:
        resolved: List[str] = []
        for family in order:
            try:
                candidate = find_family_by_prefix(families, family_prefix(family))
            except ValueError:
                continue
            if candidate in families and candidate not in resolved:
                resolved.append(candidate)
        for family in families:
            if family not in resolved:
                resolved.append(family)
        if resolved and tuple(resolved) not in resolved_orders:
            resolved_orders.append(tuple(resolved))
    natural = tuple(families)
    if natural and natural not in resolved_orders:
        resolved_orders.append(natural)
    return resolved_orders


def split_payload(
    split_name: str,
    setting: str,
    all_user_ids: Sequence[str],
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str],
    holdout_families: Sequence[str],
    notes: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "split_name": split_name,
        "setting": setting,
        "train_node_ids": sorted(train),
        "val_node_ids": sorted(val),
        "test_node_ids": sorted(test),
        "unlabeled_visible_node_ids": without_overlap(all_user_ids, train, val, test),
        "holdout_families": list(holdout_families),
        "notes": notes,
    }
    if extra:
        payload.update(extra)
    return payload


def collect_user_activity_times(
    posts: Sequence[Dict[str, str]],
    comments: Sequence[Dict[str, str]],
    events: Sequence[Dict[str, Any]],
) -> Dict[str, List[datetime]]:
    activity: Dict[str, List[datetime]] = {}

    def add(user_id: str, value: Any) -> None:
        parsed = parse_time(value)
        if user_id and parsed is not None:
            activity.setdefault(user_id, []).append(parsed)

    for post in posts:
        add(first_nonempty(post, ("author_id", "user_id")), first_nonempty(post, ("created_utc", "time", "created_at")))
    for comment in comments:
        add(
            first_nonempty(comment, ("author_id", "user_id")),
            first_nonempty(comment, ("created_utc", "time", "created_at")),
        )
    for event in events:
        add(
            first_nonempty(event, ("actor_user_id", "author_id", "user_id")),
            first_nonempty(event, ("time", "created_utc", "created_at")),
        )
    return activity


def build_time_forward_split(
    records: Dict[str, Dict[str, Any]],
    all_user_ids: Sequence[str],
    posts: Sequence[Dict[str, str]],
    comments: Sequence[Dict[str, str]],
    events: Sequence[Dict[str, Any]],
    rng: random.Random,
    val_fraction: float,
) -> Dict[str, Any]:
    activity = collect_user_activity_times(posts, comments, events)
    flattened = sorted(t for times in activity.values() for t in times)

    if not flattened:
        ordered = sorted(all_user_ids)
        cutoff_index = int(len(ordered) * 0.6)
        train_candidates = known(records, ordered[:cutoff_index])
        test_candidates = known(records, [node_id for node_id in ordered[cutoff_index:] if records[node_id]["label"] == 1])
        cutoff_value = "fallback_node_order_60pct"
    else:
        cutoff_index = max(0, min(len(flattened) - 1, int((len(flattened) - 1) * 0.6)))
        cutoff = flattened[cutoff_index]
        cutoff_value = cutoff.isoformat()
        train_pool = []
        test_pool = []
        for node_id in all_user_ids:
            times = activity.get(node_id, [])
            if not times:
                continue
            first_seen = min(times)
            last_seen = max(times)
            if first_seen <= cutoff and last_seen <= cutoff:
                train_pool.append(node_id)
            elif last_seen > cutoff and int(records[node_id]["label"]) == 1:
                test_pool.append(node_id)
        train_candidates = known(records, train_pool)
        test_candidates = known(records, test_pool)

    train, val = split_known_train_val(train_candidates, records, rng, val_fraction)
    test = sorted(node_id for node_id in test_candidates if node_id not in set(train) | set(val))
    return split_payload(
        split_name="time_forward",
        setting="temporal_transductive",
        all_user_ids=all_user_ids,
        train=train,
        val=val,
        test=test,
        holdout_families=[],
        notes="Train labels come from the first 60 percent of observed activity time; test nodes are later active known bots.",
        extra={"time_train_fraction": 0.6, "cutoff": cutoff_value},
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"--run-dir does not exist or is not a directory: {run_dir}")

    paths = {name: require_run_file(run_dir, name) for name in REQUIRED_INPUTS}
    splits_dir = ensure_output_dir(run_dir, "splits")
    rng = random.Random(args.seed)

    users = read_csv_rows(paths["users.csv"])
    posts = read_csv_rows(paths["posts.csv"])
    comments = read_csv_rows(paths["comments.csv"])
    labels = read_csv_rows(paths["labels.csv"])
    events = read_jsonl(paths["events.jsonl"])

    records = build_user_records(users, labels)
    all_user_ids = sorted(records)
    families = discover_families(records)
    original = known(records, original_nodes(records))
    generated = known(records, generated_family_nodes(records))

    written: List[Dict[str, Any]] = []

    def write_split(payload: Dict[str, Any]) -> None:
        path = splits_dir / f"{payload['split_name']}.json"
        write_json(path, payload)
        written.append(
            {
                "split_name": payload["split_name"],
                "path": str(path),
                "train": len(payload["train_node_ids"]),
                "val": len(payload["val_node_ids"]),
                "test": len(payload["test_node_ids"]),
                "unlabeled_visible": len(payload["unlabeled_visible_node_ids"]),
            }
        )

    train, val = split_known_train_val(original, records, rng, args.val_fraction)
    test = generated
    write_split(
        split_payload(
            split_name="original_only_zero_shot",
            setting="inductive_family_holdout",
            all_user_ids=all_user_ids,
            train=train,
            val=val,
            test=test,
            holdout_families=families,
            notes="Train on original_human/original_bot only; test on all generated G-family users.",
        )
    )

    for holdout in families:
        prefix = family_prefix(holdout)
        holdout_nodes = nodes_in_families(records, [holdout])
        train_candidates = known(
            records,
            [
                node_id
                for node_id in all_user_ids
                if node_id not in set(holdout_nodes)
                and (
                    str(records[node_id]["source"]).startswith("original_")
                    or str(records[node_id]["family"]).startswith("G")
                )
            ],
        )
        train, val = split_known_train_val(train_candidates, records, rng, args.val_fraction)
        test = known(records, holdout_nodes)
        write_split(
            split_payload(
                split_name=f"lofo_{prefix}",
                setting="inductive_family_holdout",
                all_user_ids=all_user_ids,
                train=train,
                val=val,
                test=test,
                holdout_families=[holdout],
                notes=f"Leave-one-family-out split; {holdout} is excluded from labeled training.",
            )
        )

    generated_known = generated
    for ratio in LABEL_SCARCITY_RATIOS:
        selected: List[str] = []
        for family in families:
            family_nodes = [node_id for node_id in generated_known if records[node_id]["family"] == family]
            shuffled = sorted(family_nodes)
            rng.shuffle(shuffled)
            if ratio >= 1.0:
                take = len(shuffled)
            elif shuffled:
                take = max(1, int(round(len(shuffled) * ratio)))
            else:
                take = 0
            selected.extend(shuffled[:take])
        train_candidates = sorted(set(original) | set(selected))
        train, val = split_known_train_val(train_candidates, records, rng, args.val_fraction)
        test = sorted(node_id for node_id in generated_known if node_id not in set(train) | set(val))
        write_split(
            split_payload(
                split_name=f"label_scarcity_{ratio_name(ratio)}",
                setting="transductive_label_scarcity",
                all_user_ids=all_user_ids,
                train=train,
                val=val,
                test=test,
                holdout_families=[],
                notes="Only a sampled share of generated bot labels is visible for training.",
                extra={"label_scarcity_ratio": ratio},
            )
        )

    write_split(
        build_time_forward_split(records, all_user_ids, posts, comments, events, rng, args.val_fraction)
    )

    for order_index, resolved_order in enumerate(resolve_curriculum_orders(families), start=1):
        current_train_families: List[str] = []
        steps: List[Dict[str, Any]] = []
        initial_train_candidates = original
        initial_train, initial_val = split_known_train_val(initial_train_candidates, records, rng, args.val_fraction)
        initial_test = known(records, nodes_in_families(records, [resolved_order[0]]))

        for step_index, family in enumerate(resolved_order, start=1):
            train_candidates = sorted(
                set(original)
                | set(known(records, nodes_in_families(records, current_train_families)))
            )
            step_train, step_val = split_known_train_val(train_candidates, records, rng, args.val_fraction)
            step_test = known(records, nodes_in_families(records, [family]))
            steps.append(
                {
                    "step": step_index,
                    "add_family_after_evaluation": family,
                    "train_node_ids": step_train,
                    "val_node_ids": step_val,
                    "test_node_ids": step_test,
                    "unlabeled_visible_node_ids": without_overlap(all_user_ids, step_train, step_val, step_test),
                    "expected_metrics": [
                        "before_recall",
                        "after_recall",
                        "recovery_gain",
                        "final_generalization",
                    ],
                }
            )
            current_train_families.append(family)

        write_split(
            split_payload(
                split_name=f"incremental_curriculum_order{order_index}",
                setting="incremental_curriculum",
                all_user_ids=all_user_ids,
                train=initial_train,
                val=initial_val,
                test=initial_test,
                holdout_families=list(resolved_order),
                notes="Top-level ids describe the initial evaluation state; per-step ids are in curriculum_steps.",
                extra={"curriculum_order": list(resolved_order), "curriculum_steps": steps},
            )
        )

    summary = {
        "run_dir": str(run_dir),
        "splits_dir": str(splits_dir),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "counts": {
            "users": len(all_user_ids),
            "known_users": len(known(records, all_user_ids)),
            "original_known_users": len(original),
            "generated_known_users": len(generated),
            "families": {family: len(nodes_in_families(records, [family])) for family in families},
            "splits": len(written),
        },
        "splits": written,
        "notes": "family is used only to construct split membership; split files expose node id lists for train/val/test/unlabeled_visible.",
    }
    summary_path = splits_dir / "split_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
