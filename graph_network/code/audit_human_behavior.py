#!/usr/bin/env python3
"""Audit human behavior statistics from BotSim input snapshots.

The primary input is BotSim-24-Dataset/user_post_comment.json, whose outer
dict key is the true author_id for posts/comment_1/comment_2 entries. CSV
tables under RedditData are treated as a fallback when the JSON is missing.
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
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


REQUIRED_METRICS = [
    "posts_per_user",
    "c1_per_user",
    "c2_per_user",
    "total_comments",
    "comment2_share",
    "unique_target_authors",
    "inbound_count",
    "outbound_count",
    "direction_balance",
    "reciprocal_neighbor_count",
    "reciprocal_share",
    "topic_entropy_proxy",
    "active_span_proxy",
]

QUANTILE_POINTS = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_input_root() -> Path:
    return project_root() / "inputs" / "original_botsim_snapshot"


def normalize_col_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    normalized = {normalize_col_name(col): col for col in df.columns}
    for candidate in candidates:
        col = normalized.get(normalize_col_name(candidate))
        if col is not None:
            return col
    return None


def value_as_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def strip_reddit_prefix(value: Any) -> str:
    """Normalize Reddit ids such as t3_abc and t1_xyz to abc/xyz."""
    text = value_as_str(value)
    if len(text) > 3 and text[0] == "t" and text[1].isdigit() and text[2] == "_":
        return text[3:]
    return text


def read_csv_if_exists(path: Optional[Path], warnings: List[str]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        warnings.append(f"Missing CSV table: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - exact parser errors depend on data.
        warnings.append(f"Failed to read CSV table {path}: {exc}")
        return pd.DataFrame()


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def discover_inputs(input_root: Path, warnings: List[str]) -> Dict[str, Optional[Path]]:
    """Find the preferred JSON and fallback CSV tables without requiring one layout."""
    roots: List[Path] = []
    if input_root.exists():
        roots.append(input_root)
    else:
        warnings.append(
            f"Input root does not exist inside this project: {input_root}"
        )

    dataset_roots: List[Path] = []
    reddit_roots: List[Path] = []
    for root in roots:
        dataset_roots.extend([root, root / "BotSim-24-Dataset"])
        reddit_roots.extend([root, root / "RedditData", root / "RedditBotSim" / "Data" / "RedditData"])

    # Keep search bounded and deterministic; these files are expected near the roots.
    json_path = first_existing(root / "user_post_comment.json" for root in dataset_roots)
    users_path = first_existing(
        [*(root / "Users.csv" for root in dataset_roots), *(root / "User.csv" for root in reddit_roots)]
    )
    posts_path = first_existing(root / "Posts.csv" for root in reddit_roots)
    c1_path = first_existing(root / "Comment_1.csv" for root in reddit_roots)
    c2_path = first_existing(root / "Comment_2.csv" for root in reddit_roots)

    if json_path is None:
        warnings.append("BotSim JSON user_post_comment.json not found; falling back to CSV activity tables.")
    if users_path is None:
        warnings.append("No User.csv/Users.csv found; human labels may be unavailable.")
    for label, path in [("Posts.csv", posts_path), ("Comment_1.csv", c1_path), ("Comment_2.csv", c2_path)]:
        if path is None:
            warnings.append(f"{label} not found; related CSV fallback table will be empty.")

    return {
        "json": json_path,
        "users": users_path,
        "posts": posts_path,
        "comment_1": c1_path,
        "comment_2": c2_path,
    }


def load_json_activity(path: Path, warnings: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flatten BotSim's dict[user_id] activity JSON into three DataFrames."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        warnings.append(f"Failed to read JSON activity {path}: {exc}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not isinstance(data, dict):
        warnings.append(f"JSON activity is not a dict[user_id] object: {path}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    posts: List[Dict[str, Any]] = []
    c1_rows: List[Dict[str, Any]] = []
    c2_rows: List[Dict[str, Any]] = []

    for user_id, payload in data.items():
        if not isinstance(payload, dict):
            continue
        for item in payload.get("posts", []) or []:
            row = dict(item)
            row["author_id"] = str(user_id)
            row["post_id"] = strip_reddit_prefix(row.get("post_id") or row.get("submission_id") or row.get("id"))
            posts.append(row)
        for item in (payload.get("comment_1") or payload.get("comment1") or []):
            row = dict(item)
            row["author_id"] = str(user_id)
            row["comment_id"] = strip_reddit_prefix(row.get("comment_id") or row.get("id"))
            row["level"] = 1
            c1_rows.append(row)
        for item in (payload.get("comment_2") or payload.get("comment2") or []):
            row = dict(item)
            row["author_id"] = str(user_id)
            row["comment_id"] = strip_reddit_prefix(row.get("comment_id") or row.get("id"))
            row["level"] = 2
            c2_rows.append(row)

    return pd.DataFrame(posts), pd.DataFrame(c1_rows), pd.DataFrame(c2_rows)


def standardize_posts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["post_id", "author_id", "topic", "created_utc"])
    post_col = find_col(df, ["post_id", "submission_id", "id"])
    author_col = find_col(df, ["author_id", "user_id", "poster_id", "submission_author_id"])
    topic_col = find_col(df, ["topic", "subreddit", "community"])
    time_col = find_col(df, ["created_utc", "created_at", "time", "timestamp"])

    out = pd.DataFrame()
    out["post_id"] = df[post_col].map(strip_reddit_prefix) if post_col else ""
    out["author_id"] = df[author_col].map(value_as_str) if author_col else ""
    out["topic"] = df[topic_col].map(value_as_str) if topic_col else ""
    out["created_utc"] = df[time_col] if time_col else ""
    return out


def standardize_comments(df: pd.DataFrame, level: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["comment_id", "author_id", "post_id", "parent_id", "level", "topic", "created_utc", "target_author_id"]
        )
    comment_col = find_col(df, ["comment_id", "id"])
    author_col = find_col(df, ["author_id", "comment_author_id", "user_id"])
    link_col = find_col(df, ["post_id", "submission_id", "link_id"])
    parent_col = find_col(df, ["parent_id", "parent_comment_id"])
    topic_col = find_col(df, ["topic", "subreddit", "community"])
    time_col = find_col(df, ["created_utc", "created_at", "time", "timestamp"])
    target_col = find_col(df, ["target_author_id", "target_user_id", "parent_author_id"])

    out = pd.DataFrame()
    out["comment_id"] = df[comment_col].map(strip_reddit_prefix) if comment_col else ""
    out["author_id"] = df[author_col].map(value_as_str) if author_col else ""
    out["post_id"] = df[link_col].map(strip_reddit_prefix) if link_col else ""
    out["parent_id"] = df[parent_col].map(strip_reddit_prefix) if parent_col else ""
    out["level"] = level
    out["topic"] = df[topic_col].map(value_as_str) if topic_col else ""
    out["created_utc"] = df[time_col] if time_col else ""
    out["target_author_id"] = df[target_col].map(value_as_str) if target_col else ""
    return out


def load_csv_activity(paths: Dict[str, Optional[Path]], warnings: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    posts = standardize_posts(read_csv_if_exists(paths.get("posts"), warnings))
    c1 = standardize_comments(read_csv_if_exists(paths.get("comment_1"), warnings), level=1)
    c2 = standardize_comments(read_csv_if_exists(paths.get("comment_2"), warnings), level=2)
    return posts, c1, c2


def load_users(path: Optional[Path], warnings: List[str]) -> pd.DataFrame:
    users = read_csv_if_exists(path, warnings)
    if users.empty:
        return pd.DataFrame(columns=["user_id", "label", "source", "row_index"])
    user_col = find_col(users, ["user_id", "id", "author_id"])
    if user_col is None:
        warnings.append(f"Users table lacks a user id column: {path}")
        return pd.DataFrame(columns=["user_id", "label", "source", "row_index"])
    label_col = find_col(users, ["label", "class", "is_bot", "is_human"])
    source_col = find_col(users, ["source", "origin", "user_source"])

    out = pd.DataFrame()
    out["user_id"] = users[user_col].map(value_as_str)
    out["label"] = users[label_col].map(value_as_str) if label_col else ""
    out["source"] = users[source_col].map(value_as_str) if source_col else ""
    out["row_index"] = range(len(users))
    return out[out["user_id"] != ""].copy()


def infer_human_users(users: pd.DataFrame, activity_user_order: Sequence[str], warnings: List[str]) -> Tuple[set, str]:
    """Use explicit labels/sources first, then BotSim's first-1907 convention."""
    if not users.empty:
        label_norm = users["label"].astype(str).str.strip().str.lower()
        source_norm = users["source"].astype(str).str.strip().str.lower()

        explicit_human = users[
            label_norm.isin({"human", "0", "false"}) | source_norm.isin({"original_human", "human"})
        ]["user_id"]
        explicit_bot = users[
            label_norm.isin({"bot", "1", "true"}) | source_norm.isin({"original_bot", "generated_bot", "bot"})
        ]["user_id"]

        if len(explicit_human) > 0 and len(explicit_human) + len(explicit_bot) > 0:
            return set(explicit_human), "explicit_label_or_source"

        fallback = set(users.sort_values("row_index").head(1907)["user_id"])
        warnings.append(
            "Human labels are unclear; used BotSim fallback rule: first 1907 rows of Users.csv/User.csv are human."
        )
        return fallback, "botsim_first_1907_users_fallback"

    fallback_order = [user_id for user_id in activity_user_order if user_id]
    fallback = set(fallback_order[:1907])
    warnings.append(
        "Users table missing; used first 1907 user ids from activity input order as a weak human fallback."
    )
    return fallback, "activity_order_first_1907_weak_fallback"


def parse_time(value: Any) -> Optional[datetime]:
    text = value_as_str(value)
    if not text:
        return None
    if text.replace(".", "", 1).isdigit():
        try:
            return datetime.fromtimestamp(float(text), tz=timezone.utc).replace(tzinfo=None)
        except (OverflowError, ValueError):
            return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:19] if "T" in text else text, fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(text, errors="coerce").to_pydatetime()
    except Exception:
        return None


def entropy(values: Sequence[str]) -> float:
    filtered = [value for value in values if value]
    if not filtered:
        return 0.0
    counts = Counter(filtered)
    total = sum(counts.values())
    return float(-sum((count / total) * math.log(count / total) for count in counts.values()))


def add_comment_targets(posts: pd.DataFrame, c1: pd.DataFrame, c2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Infer comment target authors from post and parent-comment ownership."""
    post_author = dict(zip(posts["post_id"].map(strip_reddit_prefix), posts["author_id"].map(value_as_str)))
    all_comments = pd.concat([c1, c2], ignore_index=True) if not c1.empty or not c2.empty else pd.DataFrame()
    comment_author = (
        dict(zip(all_comments["comment_id"].map(strip_reddit_prefix), all_comments["author_id"].map(value_as_str)))
        if not all_comments.empty
        else {}
    )

    c1 = c1.copy()
    c2 = c2.copy()
    if "target_author_id" not in c1.columns:
        c1["target_author_id"] = ""
    if "target_author_id" not in c2.columns:
        c2["target_author_id"] = ""

    c1_missing = c1["target_author_id"].map(value_as_str) == ""
    c1.loc[c1_missing, "target_author_id"] = c1.loc[c1_missing, "post_id"].map(lambda x: post_author.get(strip_reddit_prefix(x), ""))

    c2_missing = c2["target_author_id"].map(value_as_str) == ""
    c2.loc[c2_missing, "target_author_id"] = c2.loc[c2_missing, "parent_id"].map(
        lambda x: comment_author.get(strip_reddit_prefix(x), "")
    )

    coverage = {
        "comment1_rows": int(len(c1)),
        "comment1_target_author_known": int((c1["target_author_id"].map(value_as_str) != "").sum()) if not c1.empty else 0,
        "comment2_rows": int(len(c2)),
        "comment2_target_author_known": int((c2["target_author_id"].map(value_as_str) != "").sum()) if not c2.empty else 0,
    }
    return c1, c2, coverage


def compute_behavior_reference(
    users: pd.DataFrame,
    posts: pd.DataFrame,
    c1: pd.DataFrame,
    c2: pd.DataFrame,
    human_users: set,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    known_user_ids = set(users["user_id"]) if not users.empty else set()
    activity_user_ids = set(posts["author_id"]) | set(c1["author_id"]) | set(c2["author_id"])
    all_user_ids = sorted((known_user_ids | activity_user_ids) & human_users)

    comments = pd.concat([c1, c2], ignore_index=True) if not c1.empty or not c2.empty else pd.DataFrame()
    outbound_neighbors: Dict[str, Counter] = defaultdict(Counter)
    inbound_count: Counter = Counter()

    if not comments.empty:
        for _, row in comments.iterrows():
            src = value_as_str(row.get("author_id"))
            dst = value_as_str(row.get("target_author_id"))
            if src and dst and src != dst:
                outbound_neighbors[src][dst] += 1
                inbound_count[dst] += 1

    for user_id in all_user_ids:
        user_posts = posts[posts["author_id"] == user_id] if not posts.empty else pd.DataFrame()
        user_c1 = c1[c1["author_id"] == user_id] if not c1.empty else pd.DataFrame()
        user_c2 = c2[c2["author_id"] == user_id] if not c2.empty else pd.DataFrame()
        user_comments = pd.concat([user_c1, user_c2], ignore_index=True) if not user_c1.empty or not user_c2.empty else pd.DataFrame()

        outbound = sum(outbound_neighbors[user_id].values())
        inbound = inbound_count[user_id]
        outbound_set = set(outbound_neighbors[user_id])
        reciprocal = {dst for dst in outbound_set if outbound_neighbors.get(dst, Counter()).get(user_id, 0) > 0}

        topic_values: List[str] = []
        time_values: List[datetime] = []
        for frame in [user_posts, user_c1, user_c2]:
            if frame.empty:
                continue
            topic_values.extend(frame["topic"].map(value_as_str).tolist())
            time_values.extend(t for t in frame["created_utc"].map(parse_time).tolist() if t is not None)

        total_comments = len(user_c1) + len(user_c2)
        direction_denominator = outbound + inbound
        span_days = 0.0
        if len(time_values) >= 2:
            span_days = (max(time_values) - min(time_values)).total_seconds() / 86400.0

        rows.append(
            {
                "user_id": user_id,
                "posts_per_user": int(len(user_posts)),
                "c1_per_user": int(len(user_c1)),
                "c2_per_user": int(len(user_c2)),
                "total_comments": int(total_comments),
                "comment2_share": float(len(user_c2) / total_comments) if total_comments else 0.0,
                "unique_target_authors": int(len(outbound_set)),
                "inbound_count": int(inbound),
                "outbound_count": int(outbound),
                "direction_balance": float((outbound - inbound) / direction_denominator) if direction_denominator else 0.0,
                "reciprocal_neighbor_count": int(len(reciprocal)),
                "reciprocal_share": float(len(reciprocal) / len(outbound_set)) if outbound_set else 0.0,
                "topic_entropy_proxy": entropy(topic_values),
                "active_span_proxy": float(span_days),
            }
        )

    return pd.DataFrame(rows, columns=["user_id", *REQUIRED_METRICS])


def quantile(sorted_values: List[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight)


def summarize_metric(values: Sequence[Any]) -> Dict[str, Any]:
    numeric = sorted(float(v) for v in values if v is not None and not pd.isna(v))
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "quantiles": {f"q{int(q * 100):02d}": None for q in QUANTILE_POINTS},
        }
    mean = sum(numeric) / len(numeric)
    variance = sum((v - mean) ** 2 for v in numeric) / (len(numeric) - 1) if len(numeric) > 1 else 0.0
    return {
        "count": len(numeric),
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "min": float(numeric[0]),
        "max": float(numeric[-1]),
        "quantiles": {f"q{int(q * 100):02d}": quantile(numeric, q) for q in QUANTILE_POINTS},
    }


def build_quantiles(reference: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric in REQUIRED_METRICS:
        summary = summarize_metric(reference[metric].tolist() if metric in reference.columns else [])
        row = {
            "metric": metric,
            "count": summary["count"],
            "mean": summary["mean"],
            "std": summary["std"],
            "min": summary["min"],
            **summary["quantiles"],
            "max": summary["max"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def compact_histogram(values: Sequence[Any], max_bins: int = 30) -> List[Dict[str, Any]]:
    counts = Counter(float(v) for v in values if v is not None and not pd.isna(v))
    if len(counts) <= max_bins:
        return [{"value": value, "count": int(count)} for value, count in sorted(counts.items())]
    sorted_values = sorted(counts.elements())
    cuts = [quantile(sorted_values, q / max_bins) for q in range(max_bins + 1)]
    bins: List[Dict[str, Any]] = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        count = sum(1 for value in sorted_values if left <= value <= right)
        bins.append({"left": left, "right": right, "count": int(count)})
    return bins


def build_action_distribution(reference: pd.DataFrame) -> Dict[str, Any]:
    per_user: Dict[str, Any] = {}
    for metric in REQUIRED_METRICS:
        values = reference[metric].tolist() if metric in reference.columns else []
        per_user[metric] = {
            **summarize_metric(values),
            "histogram": compact_histogram(values),
        }
    return {
        "human_user_count": int(len(reference)),
        "action_totals": {
            "posts": int(reference["posts_per_user"].sum()) if not reference.empty else 0,
            "comment_1": int(reference["c1_per_user"].sum()) if not reference.empty else 0,
            "comment_2": int(reference["c2_per_user"].sum()) if not reference.empty else 0,
            "total_comments": int(reference["total_comments"].sum()) if not reference.empty else 0,
        },
        "per_user_metrics": per_user,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def run_audit(input_root: Path, output_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    warnings: List[str] = []
    paths = discover_inputs(input_root, warnings)
    users = load_users(paths.get("users"), warnings)

    activity_source = "json" if paths.get("json") else "csv"
    if paths.get("json"):
        posts_raw, c1_raw, c2_raw = load_json_activity(paths["json"], warnings)  # type: ignore[index]
        posts, c1, c2 = standardize_posts(posts_raw), standardize_comments(c1_raw, 1), standardize_comments(c2_raw, 2)
    else:
        posts, c1, c2 = load_csv_activity(paths, warnings)

    c1, c2, target_coverage = add_comment_targets(posts, c1, c2)
    activity_order = pd.concat([posts["author_id"], c1["author_id"], c2["author_id"]], ignore_index=True).drop_duplicates().tolist()
    human_users, human_rule = infer_human_users(users, activity_order, warnings)
    reference = compute_behavior_reference(users, posts, c1, c2, human_users)
    quantiles = build_quantiles(reference)
    action_distribution = build_action_distribution(reference)

    summary = {
        "schema_version": "human-behavior-audit-v1",
        "input_root_requested": str(input_root),
        "output_dir": str(output_dir),
        "dry_run": bool(dry_run),
        "activity_source": activity_source,
        "paths": {key: str(value) if value else None for key, value in paths.items()},
        "human_selection_rule": human_rule,
        "human_fallback_note": (
            "When label/source is unclear, this script uses the BotSim convention that the first 1907 rows "
            "of Users.csv/User.csv are human."
        ),
        "table_rows": {
            "users": int(len(users)),
            "posts": int(len(posts)),
            "comment_1": int(len(c1)),
            "comment_2": int(len(c2)),
        },
        "human_user_count": int(len(reference)),
        "target_author_coverage": target_coverage,
        "output_files": [
            "human_behavior_reference.csv",
            "human_graph_quantiles.csv",
            "human_action_distribution.json",
            "audit_summary.json",
        ],
        "warnings": warnings,
    }

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        reference.to_csv(output_dir / "human_behavior_reference.csv", index=False)
        quantiles.to_csv(output_dir / "human_graph_quantiles.csv", index=False)
        with (output_dir / "human_action_distribution.json").open("w", encoding="utf-8") as f:
            json.dump(json_safe(action_distribution), f, ensure_ascii=False, indent=2)
        with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
            json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit BotSim human behavior reference statistics.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_input_root(),
        help="Input snapshot root. Defaults to inputs/original_botsim_snapshot; if missing, BotSim source dirs are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "outputs" / "human_behavior_audit",
        help="Directory for human behavior audit outputs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load and compute the audit without writing output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(args.input_root.resolve(), args.output_dir.resolve(), dry_run=args.dry_run)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
