#!/usr/bin/env python3
"""GraphStats LR/RF/optional EBM baselines for the static user graph dataset."""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import argparse
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FORBIDDEN_FEATURE_HINTS = ("user_id", "user_name", "family", "provider", "model")
DEFAULT_SEED = 20260603


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {label}; tried: {', '.join(str(p) for p in paths)}")


def normalize_label(value: Any) -> int:
    if pd.isna(value):
        return -1
    text = str(value).strip().lower()
    if text in {"1", "bot", "true", "yes"}:
        return 1
    if text in {"0", "human", "false", "no"}:
        return 0
    return -1


def choose_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find {label} column; candidates={candidates}; columns={df.columns.tolist()}")


def sanitize_feature_token(value: Any) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_")
    return token[:48] or "unknown"


def load_dataset(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    graph_dir = run_dir / "graph"
    node_path = first_existing(
        [graph_dir / "node_table.csv", graph_dir / "users.csv", run_dir / "users.csv"],
        "node table",
    )
    edge_path = first_existing(
        [graph_dir / "edge_table.csv", graph_dir / "edges.csv", run_dir / "edges.csv"],
        "edge table",
    )
    manifest_path = run_dir / "dataset_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    nodes = pd.read_csv(node_path, dtype=str).fillna("")
    id_col = choose_column(nodes, ["user_id", "node_id", "id"], "user id")
    nodes = nodes.copy()
    nodes["user_id"] = nodes[id_col].astype(str)
    if "node_type" in nodes.columns:
        nodes = nodes[nodes["node_type"].astype(str).str.lower().isin(["", "user"])].copy()
    if "label" not in nodes.columns:
        label_path = first_existing([run_dir / "labels.csv", graph_dir / "labels.csv"], "labels")
        labels = pd.read_csv(label_path, dtype=str).fillna("")
        label_id_col = choose_column(labels, ["node_id", "user_id", "id"], "label node id")
        nodes = nodes.merge(labels[[label_id_col, "label"]], left_on="user_id", right_on=label_id_col, how="left")
    nodes["label"] = nodes["label"].map(normalize_label).astype(int)
    for col, default in {
        "user_name": "",
        "family": "",
        "source": "",
        "is_generated": "",
        "is_replay_human": "",
    }.items():
        if col not in nodes.columns:
            nodes[col] = default
    nodes = nodes.drop_duplicates("user_id").reset_index(drop=True)

    edges = pd.read_csv(edge_path, dtype=str).fillna("")
    src_col = choose_column(edges, ["src_id", "source_id", "src", "source", "from"], "edge source")
    dst_col = choose_column(edges, ["dst_id", "target_id", "dst", "target", "to"], "edge target")
    edges = edges.copy()
    edges["src_id"] = edges[src_col].astype(str)
    edges["dst_id"] = edges[dst_col].astype(str)
    if "edge_type" not in edges.columns:
        edges["edge_type"] = "interaction"
    if "weight" not in edges.columns:
        edges["weight"] = "1"
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(1.0).astype(float)
    user_ids = set(nodes["user_id"])
    edges = edges[edges["src_id"].isin(user_ids) & edges["dst_id"].isin(user_ids)].copy()
    return nodes, edges, manifest


def load_splits(run_dir: Path, nodes: pd.DataFrame, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    split_dir = run_dir / "splits"
    split_paths = sorted(
        path for path in split_dir.glob("*.json") if path.name != "split_summary.json"
    ) if split_dir.exists() else []
    if split_paths:
        return [read_json(path) | {"_path": str(path)} for path in split_paths]

    known = nodes.index[nodes["label"].isin([0, 1])].to_numpy()
    labels = nodes.loc[known, "label"].to_numpy()
    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for label in [0, 1]:
        idx = known[labels == label].copy()
        rng.shuffle(idx)
        n_train = max(1, int(len(idx) * 0.6))
        n_val = max(1, int(len(idx) * 0.2)) if len(idx) >= 5 else 0
        train_ids.extend(nodes.loc[idx[:n_train], "user_id"].tolist())
        val_ids.extend(nodes.loc[idx[n_train : n_train + n_val], "user_id"].tolist())
        test_ids.extend(nodes.loc[idx[n_train + n_val :], "user_id"].tolist())
    return [
        {
            "split_name": "auto_stratified_smoke",
            "setting": "auto",
            "train_node_ids": train_ids,
            "val_node_ids": val_ids,
            "test_node_ids": test_ids,
            "unlabeled_visible_node_ids": [],
            "holdout_families": [],
            "notes": "Auto-created because splits/*.json was absent.",
        }
    ]


def ids_to_indices(ids: list[Any], id_to_idx: dict[str, int]) -> np.ndarray:
    return np.array([id_to_idx[str(node_id)] for node_id in ids if str(node_id) in id_to_idx], dtype=int)


def split_indices(split: dict[str, Any], nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    id_to_idx = {uid: i for i, uid in enumerate(nodes["user_id"].astype(str))}
    return {
        "train": ids_to_indices(split.get("train_node_ids", []), id_to_idx),
        "val": ids_to_indices(split.get("val_node_ids", []), id_to_idx),
        "test": ids_to_indices(split.get("test_node_ids", []), id_to_idx),
        "unlabeled_visible": ids_to_indices(split.get("unlabeled_visible_node_ids", []), id_to_idx),
    }


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.divide(num, np.maximum(den, 1e-12))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def entropy_by_group(keys: np.ndarray, values: np.ndarray, num_nodes: int) -> np.ndarray:
    counters: list[Counter[str]] = [Counter() for _ in range(num_nodes)]
    for key, value in zip(keys, values):
        if value:
            counters[int(key)][str(value)] += 1
    out = np.zeros(num_nodes, dtype=float)
    for idx, counter in enumerate(counters):
        total = sum(counter.values())
        if total <= 1:
            continue
        probs = np.array(list(counter.values()), dtype=float) / total
        out[idx] = float(-(probs * np.log(probs)).sum() / math.log(len(probs))) if len(probs) > 1 else 0.0
    return out


def build_graph_features(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    n = len(nodes)
    id_to_idx = {uid: i for i, uid in enumerate(nodes["user_id"].astype(str))}
    if edges.empty:
        src = np.array([], dtype=int)
        dst = np.array([], dtype=int)
        weight = np.array([], dtype=float)
        edge_type = np.array([], dtype=object)
    else:
        src = edges["src_id"].map(id_to_idx).to_numpy(dtype=int)
        dst = edges["dst_id"].map(id_to_idx).to_numpy(dtype=int)
        weight = edges["weight"].to_numpy(dtype=float)
        edge_type = edges["edge_type"].astype(str).to_numpy(dtype=object)

    features: dict[str, np.ndarray] = {}
    features["total_out_degree"] = np.bincount(src, minlength=n).astype(float)
    features["total_in_degree"] = np.bincount(dst, minlength=n).astype(float)
    features["total_out_weight"] = np.bincount(src, weights=weight, minlength=n).astype(float)
    features["total_in_weight"] = np.bincount(dst, weights=weight, minlength=n).astype(float)
    features["total_degree"] = features["total_out_degree"] + features["total_in_degree"]
    features["total_weight"] = features["total_out_weight"] + features["total_in_weight"]
    features["direction_balance"] = safe_div(features["total_out_weight"] - features["total_in_weight"], features["total_weight"])
    features["log_total_degree"] = np.log1p(features["total_degree"])
    features["log_total_weight"] = np.log1p(features["total_weight"])
    features["avg_out_weight"] = safe_div(features["total_out_weight"], features["total_out_degree"])
    features["avg_in_weight"] = safe_div(features["total_in_weight"], features["total_in_degree"])
    features["has_any_edge"] = (features["total_degree"] > 0).astype(float)

    for name, groups in [("unique_target_authors", zip(src, dst)), ("unique_inbound_authors", zip(dst, src))]:
        buckets: list[set[int]] = [set() for _ in range(n)]
        for key, value in groups:
            buckets[int(key)].add(int(value))
        features[name] = np.array([len(bucket) for bucket in buckets], dtype=float)

    type_tokens = np.array([sanitize_feature_token(t) for t in edge_type], dtype=object)
    for token in sorted(set(type_tokens.tolist())):
        mask = type_tokens == token
        out_count = np.bincount(src[mask], minlength=n).astype(float)
        in_count = np.bincount(dst[mask], minlength=n).astype(float)
        out_weight = np.bincount(src[mask], weights=weight[mask], minlength=n).astype(float)
        in_weight = np.bincount(dst[mask], weights=weight[mask], minlength=n).astype(float)
        prefix = f"etype_{token}"
        features[f"{prefix}_out_degree"] = out_count
        features[f"{prefix}_in_degree"] = in_count
        features[f"{prefix}_degree_share"] = safe_div(out_count + in_count, features["total_degree"])
        features[f"{prefix}_weight_share"] = safe_div(out_weight + in_weight, features["total_weight"])

    def type_rate(pattern: str, denominator: np.ndarray) -> np.ndarray:
        mask = np.array([bool(re.search(pattern, str(t), flags=re.I)) for t in edge_type])
        return safe_div(np.bincount(src[mask], minlength=n).astype(float), denominator)

    features["comment1_rate"] = type_rate(r"C1|comment1|comments_on_user_post", features["total_out_degree"])
    features["comment2_rate"] = type_rate(r"C2|comment2|repl", features["total_out_degree"])
    features["post_rate"] = type_rate(r"post|writes_post", features["total_out_degree"])
    features["revisit_rate"] = type_rate(r"revisit", features["total_out_degree"])
    features["receive_rate"] = type_rate(r"receive|receives_reply", features["total_in_degree"])

    directed = sparse.coo_matrix((weight, (src, dst)), shape=(n, n)).tocsr()
    binary = sparse.coo_matrix((np.ones_like(weight), (src, dst)), shape=(n, n)).tocsr()
    undirected = binary.maximum(binary.T).tocsr()
    reciprocal = binary.multiply(binary.T).tocsr()
    features["reciprocal_share"] = safe_div(np.asarray(reciprocal.sum(axis=1)).reshape(-1), features["total_degree"])

    if "topic" in edges.columns:
        features["topic_entropy"] = entropy_by_group(src, edges["topic"].astype(str).to_numpy(), n)
    else:
        features["topic_entropy"] = np.zeros(n, dtype=float)

    n_components, component_id = csgraph.connected_components(undirected, directed=False)
    component_sizes = np.bincount(component_id, minlength=max(n_components, 1)).astype(float)
    features["component_size"] = component_sizes[component_id] if n else np.array([], dtype=float)
    features["log_component_size"] = np.log1p(features["component_size"])
    features["is_isolated"] = (np.asarray(undirected.sum(axis=1)).reshape(-1) == 0).astype(float)

    graph = nx.from_scipy_sparse_array(undirected, create_using=nx.Graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    digraph = nx.from_scipy_sparse_array(directed, create_using=nx.DiGraph, edge_attribute="weight")
    features["clustering"] = np.array([v for _, v in sorted(nx.clustering(graph).items())], dtype=float) if n else np.array([])
    try:
        pagerank = nx.pagerank(digraph, alpha=0.85, weight="weight", max_iter=200) if n else {}
    except Exception:
        pagerank = {i: 0.0 for i in range(n)}
    pr = np.array([pagerank.get(i, 0.0) for i in range(n)], dtype=float)
    features["pagerank"] = pr
    features["log_pagerank"] = np.log1p(pr * max(n, 1))

    df = pd.DataFrame(features).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    bad_cols = [c for c in df.columns if any(hint in c.lower() for hint in FORBIDDEN_FEATURE_HINTS)]
    if bad_cols:
        raise AssertionError(f"Forbidden feature names generated: {bad_cols}")
    audit_arrays = {"component_id": component_id if n else np.array([], dtype=int)}
    return df, audit_arrays


def bot_eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    known = np.isin(y_true, [0, 1])
    y_true = y_true[known]
    y_pred = y_pred[known]
    y_prob = y_prob[known]
    if len(y_true) == 0:
        return {"size": 0, "bot_count": 0, "accuracy": np.nan, "bot_precision": np.nan, "bot_recall": np.nan, "bot_f1": np.nan, "macro_f1": np.nan, "bot_mean_probability": np.nan, "mean_probability": np.nan}
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    bot_mask = y_true == 1
    return {
        "size": int(len(y_true)),
        "bot_count": int(bot_mask.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "human_precision": float(precision[0]),
        "bot_precision": float(precision[1]),
        "human_recall": float(recall[0]),
        "bot_recall": float(recall[1]),
        "human_f1": float(f1[0]),
        "bot_f1": float(f1[1]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "bot_mean_probability": float(y_prob[bot_mask].mean()) if bot_mask.any() else np.nan,
        "mean_probability": float(y_prob.mean()) if len(y_prob) else np.nan,
    }


def fit_graphstats_models(x: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, seed: int, max_ebm_rounds: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_idx = np.array([i for i in train_idx if y[i] in (0, 1)], dtype=int)
    statuses: list[dict[str, Any]] = []
    if len(train_idx) == 0 or len(np.unique(y[train_idx])) < 2:
        return {}, [{"model": "ALL", "status": "skipped", "reason": "training labels need both classes"}]

    model_specs: dict[str, Any] = {
        "GraphStats_LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)),
        "GraphStats_RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
    }
    if importlib.util.find_spec("interpret") is None:
        statuses.append({"model": "GraphStats_EBM", "status": "skipped", "reason": "interpret package unavailable"})
    else:
        try:
            from interpret.glassbox import ExplainableBoostingClassifier

            model_specs["GraphStats_EBM"] = ExplainableBoostingClassifier(
                feature_names=x.columns.tolist(),
                interactions=8,
                max_rounds=max_ebm_rounds,
                early_stopping_rounds=40,
                validation_size=0.15,
                learning_rate=0.03,
                outer_bags=4,
                inner_bags=0,
                n_jobs=-2,
                random_state=seed,
            )
        except Exception as exc:
            statuses.append({"model": "GraphStats_EBM", "status": "skipped", "reason": f"interpret import failed: {exc}"})

    fitted: dict[str, Any] = {}
    x_np = x.to_numpy(dtype=np.float64)
    for name, model in model_specs.items():
        try:
            model.fit(x_np[train_idx], y[train_idx])
            fitted[name] = model
            statuses.append({"model": name, "status": "ok", "reason": ""})
        except Exception as exc:
            statuses.append({"model": name, "status": "skipped", "reason": str(exc)})
    return fitted, statuses


def predict_proba(model: Any, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    prob = model.predict_proba(x.to_numpy(dtype=np.float64))[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pred, prob


def feature_importance_rows(model_name: str, model: Any, feature_names: list[str], split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if model_name == "GraphStats_RF" and hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif model_name == "GraphStats_LR":
        lr = model.named_steps.get("logisticregression") if hasattr(model, "named_steps") else model
        values = np.abs(np.asarray(lr.coef_).reshape(-1))
    elif model_name == "GraphStats_EBM" and hasattr(model, "term_importances"):
        for rank, (term, importance) in enumerate(sorted(zip(model.term_names_, model.term_importances()), key=lambda x: x[1], reverse=True), start=1):
            rows.append({"split_name": split_name, "model": model_name, "rank": rank, "feature": term, "importance": float(importance), "importance_kind": "ebm_term"})
        return rows
    else:
        return rows
    for rank, idx in enumerate(np.argsort(values)[::-1], start=1):
        rows.append({"split_name": split_name, "model": model_name, "rank": rank, "feature": feature_names[idx], "importance": float(values[idx]), "importance_kind": "feature"})
    return rows


def add_metric_rows(rows: list[dict[str, Any]], split: dict[str, Any], model_name: str, backend: str, y: np.ndarray, pred: np.ndarray, prob: np.ndarray, indices: np.ndarray, nodes: pd.DataFrame, eval_group: str, status: str = "ok", reason: str = "") -> None:
    if len(indices) == 0:
        return
    base = {
        "split_name": split.get("split_name", "unknown"),
        "setting": split.get("setting", ""),
        "holdout_families": ";".join(map(str, split.get("holdout_families", []))),
        "model": model_name,
        "backend": backend,
        "eval_group": eval_group,
        "family": "ALL",
        "status": status,
        "reason": reason,
    }
    rows.append(base | bot_eval_metrics(y[indices], pred[indices], prob[indices]))
    for family, fam_df in nodes.iloc[indices].groupby(nodes.iloc[indices]["family"].replace("", "unknown")):
        fam_idx = fam_df.index.to_numpy()
        if not np.isin(y[fam_idx], [0, 1]).any():
            continue
        rows.append(base | {"family": family} | bot_eval_metrics(y[fam_idx], pred[fam_idx], prob[fam_idx]))


def shortcut_audit_rows(split: dict[str, Any], nodes: pd.DataFrame, features: pd.DataFrame, audit_arrays: dict[str, np.ndarray], indices_by_group: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    y = nodes["label"].to_numpy(dtype=int)
    comp = audit_arrays["component_id"]
    comp_bot_only = np.zeros(len(nodes), dtype=bool)
    for cid in sorted(set(comp.tolist())) if len(comp) else []:
        mask = comp == cid
        known = np.isin(y[mask], [0, 1])
        comp_bot_only[mask] = bool(known.any() and np.all(y[mask][known] == 1))
    flags = pd.DataFrame(
        {
            "zero_inbound": features["total_in_degree"] <= 0,
            "zero_outbound": features["total_out_degree"] <= 0,
            "over_directional": features["direction_balance"].abs() >= 0.95,
            "low_reciprocity": features["reciprocal_share"] <= 0.001,
            "isolated": features["is_isolated"] > 0,
            "tiny_component": features["component_size"] <= 2,
            "bot_only_component": comp_bot_only,
        }
    )
    rows: list[dict[str, Any]] = []
    forbidden_cols = [c for c in nodes.columns if any(h in c.lower() for h in FORBIDDEN_FEATURE_HINTS)]
    for group_name, indices in indices_by_group.items():
        if len(indices) == 0:
            continue
        families = nodes.iloc[indices]["family"].replace("", "unknown")
        for family in ["ALL", *sorted(set(families.tolist()))]:
            fam_indices = indices if family == "ALL" else nodes.iloc[indices][families == family].index.to_numpy()
            if len(fam_indices) == 0:
                continue
            row = {
                "split_name": split.get("split_name", "unknown"),
                "setting": split.get("setting", ""),
                "eval_group": group_name,
                "family": family,
                "size": int(len(fam_indices)),
                "bot_count": int((y[fam_indices] == 1).sum()),
                "forbidden_metadata_columns_present": ";".join(forbidden_cols),
                "forbidden_metadata_used_as_features": False,
            }
            for col in flags.columns:
                row[f"{col}_share"] = float(flags.iloc[fam_indices][col].mean()) if len(fam_indices) else np.nan
            rows.append(row)
    return rows


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    result_dir = ensure_dir(Path(args.result_dir).resolve())
    nodes, edges, manifest = load_dataset(run_dir)
    features, audit_arrays = build_graph_features(nodes, edges)
    splits = load_splits(run_dir, nodes, args.seed)
    y = nodes["label"].to_numpy(dtype=int)

    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for split in splits:
        idx = split_indices(split, nodes)
        models, statuses = fit_graphstats_models(features, y, idx["train"], args.seed, args.max_ebm_rounds)
        status_map = {row["model"]: row for row in statuses}
        for status in statuses:
            if status["status"] != "ok":
                metrics_rows.append(
                    {
                        "split_name": split.get("split_name", "unknown"),
                        "setting": split.get("setting", ""),
                        "holdout_families": ";".join(map(str, split.get("holdout_families", []))),
                        "model": status["model"],
                        "backend": "sklearn" if status["model"] != "GraphStats_EBM" else "interpret",
                        "eval_group": "test",
                        "family": "ALL",
                        "status": "skipped",
                        "reason": status["reason"],
                    }
                )
        for model_name, model in models.items():
            pred, prob = predict_proba(model, features)
            model_status = status_map.get(model_name, {"status": "ok", "reason": ""})
            backend = "interpret" if model_name == "GraphStats_EBM" else "sklearn"
            for group in ["val", "test"]:
                add_metric_rows(metrics_rows, split, model_name, backend, y, pred, prob, idx[group], nodes, group, model_status["status"], model_status["reason"])
            importance_rows.extend(feature_importance_rows(model_name, model, features.columns.tolist(), split.get("split_name", "unknown")))
            for i, row in nodes.iterrows():
                prediction_rows.append(
                    {
                        "split_name": split.get("split_name", "unknown"),
                        "model": model_name,
                        "node_id": row["user_id"],
                        "true_label": int(row["label"]),
                        "predicted_label": int(pred[i]),
                        "bot_probability": float(prob[i]),
                        "family": row.get("family", ""),
                        "source": row.get("source", ""),
                        "in_train": bool(i in set(idx["train"].tolist())),
                        "in_val": bool(i in set(idx["val"].tolist())),
                        "in_test": bool(i in set(idx["test"].tolist())),
                    }
                )
        audit_rows.extend(shortcut_audit_rows(split, nodes, features, audit_arrays, {"train": idx["train"], "val": idx["val"], "test": idx["test"], "all_known": np.where(np.isin(y, [0, 1]))[0]}))

    pd.DataFrame(metrics_rows).to_csv(result_dir / "graphstats_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(result_dir / "graphstats_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(audit_rows).to_csv(result_dir / "shortcut_audit.csv", index=False, encoding="utf-8")
    pd.DataFrame(importance_rows).to_csv(result_dir / "feature_importance.csv", index=False, encoding="utf-8")
    with (result_dir / "graphstats_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "node_count": int(len(nodes)),
                "edge_count": int(len(edges)),
                "feature_count": int(features.shape[1]),
                "split_count": int(len(splits)),
                "manifest_run_name": manifest.get("run_name", ""),
                "forbidden_feature_hints": FORBIDDEN_FEATURE_HINTS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Dataset run directory. Read-only.")
    parser.add_argument("--result-dir", required=True, help="Directory for result CSV/JSON outputs.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-ebm-rounds", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
