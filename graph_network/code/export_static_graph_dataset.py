#!/usr/bin/env python3
"""Export mixed_environment_runner CSV/JSONL outputs to static graph files."""

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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REQUIRED_INPUTS = (
    "users.csv",
    "posts.csv",
    "comments.csv",
    "events.jsonl",
    "labels.csv",
)

NODE_TABLE_FIELDS = (
    "node_id",
    "node_type",
    "label",
    "label_known",
    "source",
    "family",
    "created_utc",
)

EDGE_TABLE_FIELDS = (
    "edge_id",
    "src_id",
    "dst_id",
    "src_type",
    "dst_type",
    "edge_type",
    "time",
    "weight",
    "source_event_id",
)

USER_GRAPH_EDGE_TYPES = (
    "user_comments_on_user_post",
    "user_replies_to_user_comment",
    "user_receives_reply",
    "user_revisits_thread",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export static graph tables and a homogeneous user graph from one run directory."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Run directory containing users.csv/posts.csv/comments.csv/events.jsonl/labels.csv.",
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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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


def build_label_lookup(label_rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in label_rows:
        node_id = first_nonempty(row, ("node_id", "user_id"))
        if not node_id:
            continue
        label = normalize_label(row.get("label"))
        lookup[node_id] = {
            "node_type": first_nonempty(row, ("node_type",), "user"),
            "label": label,
            "label_known": normalize_known(row.get("label_known"), label),
            "source": first_nonempty(row, ("source",)),
            "family": first_nonempty(row, ("family",)),
            "split_group": first_nonempty(row, ("split_group",)),
        }
    return lookup


def build_node_table(
    users: Sequence[Dict[str, str]],
    posts: Sequence[Dict[str, str]],
    comments: Sequence[Dict[str, str]],
    label_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for user in users:
        user_id = first_nonempty(user, ("user_id", "node_id"))
        if not user_id:
            continue
        label_info = label_lookup.get(user_id, {})
        label = normalize_label(label_info.get("label", user.get("label", -1)))
        rows.append(
            {
                "node_id": user_id,
                "node_type": "user",
                "label": label,
                "label_known": normalize_known(label_info.get("label_known"), label),
                "source": first_nonempty(label_info, ("source",), first_nonempty(user, ("source",))),
                "family": first_nonempty(label_info, ("family",), first_nonempty(user, ("family",))),
                "created_utc": first_nonempty(user, ("created_utc", "created_at", "time")),
            }
        )

    for post in posts:
        post_id = first_nonempty(post, ("post_id", "node_id"))
        if not post_id:
            continue
        rows.append(
            {
                "node_id": post_id,
                "node_type": "post",
                "label": -1,
                "label_known": 0,
                "source": first_nonempty(post, ("author_source", "source")),
                "family": first_nonempty(post, ("family",)),
                "created_utc": first_nonempty(post, ("created_utc", "created_at", "time")),
            }
        )

    for comment in comments:
        comment_id = first_nonempty(comment, ("comment_id", "node_id"))
        if not comment_id:
            continue
        rows.append(
            {
                "node_id": comment_id,
                "node_type": "comment",
                "label": -1,
                "label_known": 0,
                "source": first_nonempty(comment, ("author_source", "source")),
                "family": first_nonempty(comment, ("family",)),
                "created_utc": first_nonempty(comment, ("created_utc", "created_at", "time")),
            }
        )

    return rows


class EdgeBuilder:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self._seen: set[Tuple[str, str, str, str, str, str]] = set()

    def add(
        self,
        src_id: str,
        dst_id: str,
        src_type: str,
        dst_type: str,
        edge_type: str,
        time: str = "",
        source_event_id: str = "",
        weight: float = 1.0,
    ) -> None:
        if not src_id or not dst_id:
            return
        key = (src_id, dst_id, src_type, dst_type, edge_type, source_event_id)
        if key in self._seen:
            return
        self._seen.add(key)
        self.rows.append(
            {
                "edge_id": f"e_{len(self.rows) + 1:08d}",
                "src_id": src_id,
                "dst_id": dst_id,
                "src_type": src_type,
                "dst_type": dst_type,
                "edge_type": edge_type,
                "time": time,
                "weight": weight,
                "source_event_id": source_event_id,
            }
        )


def build_edge_table(
    posts: Sequence[Dict[str, str]],
    comments: Sequence[Dict[str, str]],
    events: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    post_by_id = {
        first_nonempty(post, ("post_id", "node_id")): post
        for post in posts
        if first_nonempty(post, ("post_id", "node_id"))
    }
    comment_by_id = {
        first_nonempty(comment, ("comment_id", "node_id")): comment
        for comment in comments
        if first_nonempty(comment, ("comment_id", "node_id"))
    }

    builder = EdgeBuilder()

    for post in posts:
        post_id = first_nonempty(post, ("post_id", "node_id"))
        author_id = first_nonempty(post, ("author_id", "user_id"))
        builder.add(
            author_id,
            post_id,
            "user",
            "post",
            "user_writes_post",
            time=first_nonempty(post, ("created_utc", "created_at", "time")),
        )

    for comment in comments:
        comment_id = first_nonempty(comment, ("comment_id", "node_id"))
        author_id = first_nonempty(comment, ("author_id", "user_id"))
        post_id = first_nonempty(comment, ("post_id",))
        parent_id = first_nonempty(comment, ("parent_id",))
        parent_type = first_nonempty(comment, ("parent_type",)).lower()
        created_time = first_nonempty(comment, ("created_utc", "created_at", "time"))
        action_type = first_nonempty(comment, ("action_type",)).lower()

        builder.add(author_id, comment_id, "user", "comment", "user_writes_comment", time=created_time)

        if parent_type == "comment" and parent_id:
            builder.add(
                comment_id,
                parent_id,
                "comment",
                "comment",
                "comment_replies_comment",
                time=created_time,
            )
        elif post_id:
            builder.add(comment_id, post_id, "comment", "post", "comment_replies_post", time=created_time)

        post_author_id = first_nonempty(post_by_id.get(post_id, {}), ("author_id", "user_id"))
        if post_author_id and author_id and author_id != post_author_id:
            builder.add(
                author_id,
                post_author_id,
                "user",
                "user",
                "user_comments_on_user_post",
                time=created_time,
            )

        target_author_id = first_nonempty(comment, ("target_author_id",))
        if not target_author_id and parent_type == "comment" and parent_id in comment_by_id:
            target_author_id = first_nonempty(comment_by_id[parent_id], ("author_id", "user_id"))

        if parent_type == "comment" and target_author_id and author_id and author_id != target_author_id:
            builder.add(
                author_id,
                target_author_id,
                "user",
                "user",
                "user_replies_to_user_comment",
                time=created_time,
            )

        if target_author_id and author_id and author_id != target_author_id:
            builder.add(
                author_id,
                target_author_id,
                "user",
                "user",
                "user_receives_reply",
                time=created_time,
            )
        elif post_author_id and author_id and author_id != post_author_id:
            builder.add(
                author_id,
                post_author_id,
                "user",
                "user",
                "user_receives_reply",
                time=created_time,
            )

        if action_type == "revisit":
            revisit_dst = post_id or parent_id
            revisit_dst_type = "post" if post_id else "comment"
            builder.add(
                author_id,
                revisit_dst,
                "user",
                revisit_dst_type,
                "user_revisits_thread",
                time=created_time,
            )

    for index, event in enumerate(events, start=1):
        event_type = str(event.get("event_type", "")).strip().lower()
        event_id = str(event.get("event_id") or f"event_line_{index}")
        event_time = str(event.get("time") or event.get("created_utc") or "")
        actor_id = str(event.get("actor_user_id") or event.get("author_id") or "")
        target_author_id = str(event.get("target_author_id") or "")
        target_post_id = str(event.get("target_post_id") or "")
        target_comment_id = str(event.get("target_comment_id") or "")

        if event_type in {"notification_created", "human_replay_reply"}:
            if actor_id and target_author_id and actor_id != target_author_id:
                builder.add(
                    actor_id,
                    target_author_id,
                    "user",
                    "user",
                    "user_receives_reply",
                    time=event_time,
                    source_event_id=event_id,
                )

        if event_type == "revisit":
            if target_post_id:
                builder.add(
                    actor_id,
                    target_post_id,
                    "user",
                    "post",
                    "user_revisits_thread",
                    time=event_time,
                    source_event_id=event_id,
                )
            elif target_comment_id:
                builder.add(
                    actor_id,
                    target_comment_id,
                    "user",
                    "comment",
                    "user_revisits_thread",
                    time=event_time,
                    source_event_id=event_id,
                )
            elif target_author_id and actor_id != target_author_id:
                builder.add(
                    actor_id,
                    target_author_id,
                    "user",
                    "user",
                    "user_revisits_thread",
                    time=event_time,
                    source_event_id=event_id,
                )

    return builder.rows


def user_author_for_node(
    node_id: str,
    posts_by_id: Dict[str, Dict[str, str]],
    comments_by_id: Dict[str, Dict[str, str]],
) -> Optional[str]:
    if node_id in posts_by_id:
        return first_nonempty(posts_by_id[node_id], ("author_id", "user_id")) or None
    if node_id in comments_by_id:
        return first_nonempty(comments_by_id[node_id], ("author_id", "user_id")) or None
    return None


def build_user_graph_arrays(
    users: Sequence[Dict[str, str]],
    labels: Dict[str, Dict[str, Any]],
    posts: Sequence[Dict[str, str]],
    comments: Sequence[Dict[str, str]],
    edges: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    user_ids = [first_nonempty(user, ("user_id", "node_id")) for user in users]
    user_ids = [user_id for user_id in user_ids if user_id]
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    posts_by_id = {
        first_nonempty(post, ("post_id", "node_id")): post
        for post in posts
        if first_nonempty(post, ("post_id", "node_id"))
    }
    comments_by_id = {
        first_nonempty(comment, ("comment_id", "node_id")): comment
        for comment in comments
        if first_nonempty(comment, ("comment_id", "node_id"))
    }

    edge_type_names = list(USER_GRAPH_EDGE_TYPES)
    edge_type_to_id = {name: idx for idx, name in enumerate(edge_type_names)}
    pairs: List[Tuple[int, int]] = []
    type_ids: List[int] = []

    for edge in edges:
        edge_type = str(edge.get("edge_type", ""))
        if edge_type not in edge_type_to_id:
            continue
        src_id = str(edge.get("src_id", ""))
        dst_id = str(edge.get("dst_id", ""))
        dst_type = str(edge.get("dst_type", ""))
        if edge.get("src_type") != "user":
            continue
        if dst_type != "user":
            dst_id = user_author_for_node(dst_id, posts_by_id, comments_by_id) or ""
        if src_id in user_to_idx and dst_id in user_to_idx:
            pairs.append((user_to_idx[src_id], user_to_idx[dst_id]))
            type_ids.append(edge_type_to_id[edge_type])

    label_values = []
    label_known_values = []
    for user_id in user_ids:
        label_info = labels.get(user_id, {})
        label = normalize_label(label_info.get("label", -1))
        label_values.append(label)
        label_known_values.append(normalize_known(label_info.get("label_known"), label))

    if pairs:
        edge_index = [[src for src, _ in pairs], [dst for _, dst in pairs]]
    else:
        edge_index = [[], []]

    return {
        "node_ids": user_ids,
        "node_id_to_index": user_to_idx,
        "edge_index": edge_index,
        "edge_type": type_ids,
        "edge_type_names": edge_type_names,
        "edge_type_to_id": edge_type_to_id,
        "labels": label_values,
        "label_known": label_known_values,
    }


def write_user_graph_pt(path: Path, graph: Dict[str, Any]) -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        return False

    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    edge_type = torch.tensor(graph["edge_type"], dtype=torch.long)
    labels = torch.tensor(graph["labels"], dtype=torch.long)
    label_known = torch.tensor(graph["label_known"], dtype=torch.bool)
    torch.save(
        {
            "node_ids": graph["node_ids"],
            "node_id_to_index": graph["node_id_to_index"],
            "edge_index": edge_index,
            "edge_type": edge_type,
            "edge_type_names": graph["edge_type_names"],
            "edge_type_to_id": graph["edge_type_to_id"],
            "labels": labels,
            "label_known": label_known,
            "notes": "family is intentionally not stored as a model feature in this graph object.",
        },
        path,
    )
    return True


def write_user_graph_npz(path: Path, graph: Dict[str, Any]) -> None:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is unavailable and numpy is required for .npz fallback") from exc

    np.savez_compressed(
        path,
        node_ids=np.array(graph["node_ids"], dtype=str),
        edge_index=np.array(graph["edge_index"], dtype=np.int64),
        edge_type=np.array(graph["edge_type"], dtype=np.int64),
        edge_type_names=np.array(graph["edge_type_names"], dtype=str),
        labels=np.array(graph["labels"], dtype=np.int64),
        label_known=np.array(graph["label_known"], dtype=np.bool_),
    )


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"--run-dir does not exist or is not a directory: {run_dir}")

    paths = {name: require_run_file(run_dir, name) for name in REQUIRED_INPUTS}
    graph_dir = ensure_output_dir(run_dir, "graph")

    users = read_csv_rows(paths["users.csv"])
    posts = read_csv_rows(paths["posts.csv"])
    comments = read_csv_rows(paths["comments.csv"])
    labels = read_csv_rows(paths["labels.csv"])
    events = read_jsonl(paths["events.jsonl"])

    label_lookup = build_label_lookup(labels)
    node_rows = build_node_table(users, posts, comments, label_lookup)
    edge_rows = build_edge_table(posts, comments, events)
    user_graph = build_user_graph_arrays(users, label_lookup, posts, comments, edge_rows)

    node_table_path = graph_dir / "node_table.csv"
    edge_table_path = graph_dir / "edge_table.csv"
    write_csv(node_table_path, node_rows, NODE_TABLE_FIELDS)
    write_csv(edge_table_path, edge_rows, EDGE_TABLE_FIELDS)

    pt_path = graph_dir / "homogeneous_user_graph.pt"
    npz_path = graph_dir / "homogeneous_user_graph.npz"
    graph_artifact = pt_path
    graph_format = "pt"
    if not write_user_graph_pt(pt_path, user_graph):
        write_user_graph_npz(npz_path, user_graph)
        graph_artifact = npz_path
        graph_format = "npz"

    summary = {
        "run_dir": str(run_dir),
        "graph_dir": str(graph_dir),
        "inputs": {name: str(path) for name, path in paths.items()},
        "outputs": {
            "node_table": str(node_table_path),
            "edge_table": str(edge_table_path),
            "homogeneous_user_graph": str(graph_artifact),
            "homogeneous_user_graph_format": graph_format,
        },
        "counts": {
            "users": len(users),
            "posts": len(posts),
            "comments": len(comments),
            "events": len(events),
            "labels": len(labels),
            "nodes": len(node_rows),
            "edges": len(edge_rows),
            "homogeneous_user_nodes": len(user_graph["node_ids"]),
            "homogeneous_user_edges": len(user_graph["edge_type"]),
        },
        "edge_type_counts": dict(Counter(row["edge_type"] for row in edge_rows)),
        "required_edge_types_present": {
            edge_type: any(row["edge_type"] == edge_type for row in edge_rows)
            for edge_type in (
                "user_writes_post",
                "user_writes_comment",
                "comment_replies_post",
                "comment_replies_comment",
                "user_comments_on_user_post",
                "user_replies_to_user_comment",
                "user_receives_reply",
                "user_revisits_thread",
            )
        },
        "notes": "family is copied to audit tables only; homogeneous graph tensors do not include family features.",
    }
    summary_path = graph_dir / "export_summary.json"
    write_summary(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
