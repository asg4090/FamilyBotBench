#!/usr/bin/env python3
"""R-GCN transductive baseline with PyG and torch/sklearn fallback."""

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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_graphstats_baselines import (
    DEFAULT_SEED,
    add_metric_rows,
    build_graph_features,
    ensure_dir,
    load_dataset,
    load_splits,
    sanitize_feature_token,
    split_indices,
)


def pyg_backend_label() -> str:
    return "pyg_rgcnconv" if importlib.util.find_spec("torch_geometric") else "torch_manual_rgcn_fallback"


def sklearn_fallback(features: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    train_idx = np.array([i for i in train_idx if y[i] in (0, 1)], dtype=int)
    if len(train_idx) == 0 or len(np.unique(y[train_idx])) < 2:
        prob = np.full(len(y), 0.5, dtype=float)
        return (prob >= 0.5).astype(int), prob, "sklearn_constant_fallback"
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))
    model.fit(features.to_numpy(dtype=float)[train_idx], y[train_idx])
    prob = model.predict_proba(features.to_numpy(dtype=float))[:, 1]
    return (prob >= 0.5).astype(int), prob, "sklearn_fallback"


def build_relation_adjs(edges: pd.DataFrame, id_to_idx: dict[str, int], num_nodes: int):
    import torch

    rel_edges: dict[str, list[tuple[int, int]]] = {}
    for _, row in edges.iterrows():
        src = id_to_idx.get(str(row["src_id"]))
        dst = id_to_idx.get(str(row["dst_id"]))
        if src is None or dst is None:
            continue
        rel = sanitize_feature_token(row.get("edge_type", "interaction"))
        rel_edges.setdefault(rel, []).append((src, dst))
        rel_edges.setdefault(f"{rel}_rev", []).append((dst, src))
    adjs = []
    for rel in sorted(rel_edges):
        pairs = rel_edges[rel]
        if not pairs:
            continue
        dst = np.array([p[1] for p in pairs], dtype=np.int64)
        deg = np.bincount(dst, minlength=num_nodes).astype(np.float32)
        vals = np.array([1.0 / max(deg[d], 1.0) for d in dst], dtype=np.float32)
        idx = torch.tensor([[p[1] for p in pairs], [p[0] for p in pairs]], dtype=torch.long)
        adj = torch.sparse_coo_tensor(idx, torch.tensor(vals), (num_nodes, num_nodes)).coalesce()
        adjs.append(adj)
    return adjs


def build_relation_edge_index(edges: pd.DataFrame, id_to_idx: dict[str, int]):
    import torch

    rel_to_id: dict[str, int] = {}
    src_list: list[int] = []
    dst_list: list[int] = []
    rel_list: list[int] = []
    for _, row in edges.iterrows():
        src = id_to_idx.get(str(row["src_id"]))
        dst = id_to_idx.get(str(row["dst_id"]))
        if src is None or dst is None:
            continue
        rel = sanitize_feature_token(row.get("edge_type", "interaction"))
        rev_rel = f"{rel}_rev"
        if rel not in rel_to_id:
            rel_to_id[rel] = len(rel_to_id)
        if rev_rel not in rel_to_id:
            rel_to_id[rev_rel] = len(rel_to_id)
        src_list.append(src)
        dst_list.append(dst)
        rel_list.append(rel_to_id[rel])
        src_list.append(dst)
        dst_list.append(src)
        rel_list.append(rel_to_id[rev_rel])
    if not src_list:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long), 0
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(rel_list, dtype=torch.long)
    return edge_index, edge_type, len(rel_to_id)


def pyg_rgcn(features: pd.DataFrame, edges: pd.DataFrame, nodes: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import RGCNConv

    train_idx = np.array([i for i in train_idx if y[i] in (0, 1)], dtype=int)
    if len(train_idx) == 0 or len(np.unique(y[train_idx])) < 2:
        return sklearn_fallback(features, y, train_idx, args.seed)

    scaler = StandardScaler().fit(features.to_numpy(dtype=float)[train_idx])
    x = torch.tensor(scaler.transform(features.to_numpy(dtype=float)), dtype=torch.float32)
    labels = torch.tensor(np.where(np.isin(y, [0, 1]), y, 0), dtype=torch.long)
    train_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx = np.array([i for i in val_idx if y[i] in (0, 1)], dtype=int)
    val_t = torch.tensor(val_idx, dtype=torch.long) if len(val_idx) else train_t
    id_to_idx = {uid: i for i, uid in enumerate(nodes["user_id"].astype(str))}
    edge_index, edge_type, num_relations = build_relation_edge_index(edges, id_to_idx)
    if num_relations == 0:
        return sklearn_fallback(features, y, train_idx, args.seed)

    class PyGRGCN(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, rel_count: int):
            super().__init__()
            self.conv1 = RGCNConv(in_dim, hidden_dim, rel_count)
            self.conv2 = RGCNConv(hidden_dim, 2, rel_count)
            self.dropout = nn.Dropout(args.dropout)

        def forward(self, h):
            h = F.relu(self.conv1(h, edge_index, edge_type))
            h = self.dropout(h)
            return self.conv2(h, edge_index, edge_type)

    torch.manual_seed(args.seed)
    model = PyGRGCN(x.shape[1], args.hidden_dim, num_relations)
    class_counts = np.bincount(y[train_idx].astype(int), minlength=2).astype(float)
    class_weight = torch.tensor(class_counts.sum() / np.maximum(class_counts, 1.0), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    patience_left = args.patience
    for _epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[train_t], labels[train_t], weight=class_weight)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(x)[val_t], labels[val_t], weight=class_weight).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
    return (prob >= 0.5).astype(int), prob, "pyg_rgcnconv"


def torch_rgcn(features: pd.DataFrame, edges: pd.DataFrame, nodes: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    backend = "torch_manual_rgcn_fallback"
    train_idx = np.array([i for i in train_idx if y[i] in (0, 1)], dtype=int)
    if len(train_idx) == 0 or len(np.unique(y[train_idx])) < 2:
        return sklearn_fallback(features, y, train_idx, args.seed)

    scaler = StandardScaler().fit(features.to_numpy(dtype=float)[train_idx])
    x = torch.tensor(scaler.transform(features.to_numpy(dtype=float)), dtype=torch.float32)
    labels = torch.tensor(np.where(np.isin(y, [0, 1]), y, 0), dtype=torch.long)
    train_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx = np.array([i for i in val_idx if y[i] in (0, 1)], dtype=int)
    val_t = torch.tensor(val_idx, dtype=torch.long) if len(val_idx) else train_t
    id_to_idx = {uid: i for i, uid in enumerate(nodes["user_id"].astype(str))}
    adjs = build_relation_adjs(edges, id_to_idx, len(nodes))

    class RGCNLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, rel_count: int):
            super().__init__()
            self.self_lin = nn.Linear(in_dim, out_dim)
            self.rel_lins = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(rel_count)])

        def forward(self, h):
            out = self.self_lin(h)
            for adj, lin in zip(adjs, self.rel_lins):
                out = out + lin(torch.sparse.mm(adj, h))
            return out

    class SimpleRGCN(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, rel_count: int):
            super().__init__()
            self.layer1 = RGCNLayer(in_dim, hidden_dim, rel_count)
            self.layer2 = RGCNLayer(hidden_dim, 2, rel_count)
            self.dropout = nn.Dropout(args.dropout)

        def forward(self, h):
            h = F.relu(self.layer1(h))
            h = self.dropout(h)
            return self.layer2(h)

    torch.manual_seed(args.seed)
    model = SimpleRGCN(x.shape[1], args.hidden_dim, len(adjs))
    class_counts = np.bincount(y[train_idx].astype(int), minlength=2).astype(float)
    class_weight = torch.tensor(class_counts.sum() / np.maximum(class_counts, 1.0), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    patience_left = args.patience
    for _epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[train_t], labels[train_t], weight=class_weight)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(x)[val_t], labels[val_t], weight=class_weight).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
    return (prob >= 0.5).astype(int), prob, backend


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    result_dir = ensure_dir(Path(args.result_dir).resolve())
    nodes, edges, _manifest = load_dataset(run_dir)
    features, _audit = build_graph_features(nodes, edges)
    splits = load_splits(run_dir, nodes, args.seed)
    y = nodes["label"].to_numpy(dtype=int)

    metrics_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    torch_available = importlib.util.find_spec("torch") is not None
    pyg_available = importlib.util.find_spec("torch_geometric") is not None
    for split in splits:
        idx = split_indices(split, nodes)
        if torch_available and pyg_available:
            pred, prob, backend = pyg_rgcn(features, edges, nodes, y, idx["train"], idx["val"], args)
        elif torch_available:
            pred, prob, backend = torch_rgcn(features, edges, nodes, y, idx["train"], idx["val"], args)
        else:
            pred, prob, backend = sklearn_fallback(features, y, idx["train"], args.seed)
        for group in ["val", "test"]:
            add_metric_rows(metrics_rows, split, "RGCN_Transductive", backend, y, pred, prob, idx[group], nodes, group)
        for i, row in nodes.iterrows():
            pred_rows.append(
                {
                    "split_name": split.get("split_name", "unknown"),
                    "model": "RGCN_Transductive",
                    "backend": backend,
                    "node_id": row["user_id"],
                    "true_label": int(row["label"]),
                    "predicted_label": int(pred[i]),
                    "bot_probability": float(prob[i]),
                    "family": row.get("family", ""),
                    "in_train": bool(i in set(idx["train"].tolist())),
                    "in_val": bool(i in set(idx["val"].tolist())),
                    "in_test": bool(i in set(idx["test"].tolist())),
                }
            )
    pd.DataFrame(metrics_rows).to_csv(result_dir / "rgcn_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(pred_rows).to_csv(result_dir / "rgcn_predictions.csv", index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
