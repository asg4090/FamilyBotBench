#!/usr/bin/env python3
"""
Graph-only BotSim-24 bot detection experiment.

The experiment intentionally excludes user text and user metadata as model
inputs. The only supervised label source is the dataset README rule: first
1907 users are human and the last 1000 users are bots.

Allowed model inputs:
- edge_index.pt
- edge_type.pt
- edge_weight.pt
- graph-derived node statistics computed from those three tensors

Disallowed model inputs:
- Users.csv columns as features
- user_post_comment.json text
- text.json
- comment/post counts from metadata
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
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix, csgraph
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def configure_matplotlib_fonts() -> None:
    preferred_fonts = [
        "Arial Unicode MS",
        "Hiragino Sans GB",
        "PingFang SC",
        "Songti SC",
        "Heiti SC",
        "Noto Sans CJK SC",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_fonts()


EDGE_TYPE_NAMES = {
    0: "C1-P",
    1: "C2-P",
    2: "C2-C1",
}

MODEL_NAMES_ZH = {
    "GraphStats_LR": "图结构统计 + 逻辑回归",
    "GraphStats_RF": "图结构统计 + 随机森林",
    "Constant_RGCN": "常数节点特征 + RGCN",
    "GraphStats_RGCN": "图结构统计 + RGCN",
}


@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class SparseRGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.rel_linears = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)]
        )
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        out = self.self_linear(x)
        for adj, linear in zip(adjs, self.rel_linears):
            out = out + linear(torch.sparse.mm(adj, x))
        return out + self.bias


class GraphOnlyRGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_relations: int = 3,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.input = nn.Linear(in_dim, hidden_dim)
        self.rgcn1 = SparseRGCNLayer(hidden_dim, hidden_dim, num_relations)
        self.rgcn2 = SparseRGCNLayer(hidden_dim, hidden_dim, num_relations)
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        h = F.relu(self.input(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.rgcn1(h, adjs))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.rgcn2(h, adjs))
        return self.out(h)


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "BotSim-24-Dataset").exists() and (parent / "BotSim-24-mini-sample").exists():
            return parent
    raise RuntimeError("Cannot locate BotSim-main project root.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_split(labels: np.ndarray, seed: int) -> Split:
    all_idx = np.arange(len(labels))
    train_idx, holdout_idx = train_test_split(
        all_idx,
        test_size=0.30,
        stratify=labels,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=1 / 3,
        stratify=labels[holdout_idx],
        random_state=seed,
    )
    return Split(np.sort(train_idx), np.sort(val_idx), np.sort(test_idx))


def load_graph(root: Path) -> tuple[pd.DataFrame, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    users = pd.read_csv(root / "BotSim-24-Dataset" / "Users.csv")
    labels = np.zeros(len(users), dtype=np.int64)
    labels[1907:] = 1
    edge_index = torch.load(
        root / "BotSim-24-mini-sample" / "edge_index.pt",
        map_location="cpu",
        weights_only=True,
    ).long()
    edge_type = torch.load(
        root / "BotSim-24-mini-sample" / "edge_type.pt",
        map_location="cpu",
        weights_only=True,
    ).long()
    edge_weight = torch.load(
        root / "BotSim-24-mini-sample" / "edge_weight.pt",
        map_location="cpu",
        weights_only=True,
    ).reshape(-1).float()
    return users, labels, edge_index, edge_type, edge_weight


def make_sparse_adjs(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    num_relations: int = 3,
    use_weight: bool = True,
) -> list[torch.Tensor]:
    src = edge_index[:, 0].long()
    dst = edge_index[:, 1].long()
    base_weight = edge_weight.float() if use_weight else torch.ones_like(edge_weight).float()
    adjs: list[torch.Tensor] = []
    for rel in range(num_relations):
        mask = edge_type == rel
        rel_src = src[mask]
        rel_dst = dst[mask]
        rel_weight = base_weight[mask]
        if rel_src.numel() == 0:
            indices = torch.empty((2, 0), dtype=torch.long)
            values = torch.empty((0,), dtype=torch.float32)
        else:
            deg = torch.zeros(num_nodes, dtype=torch.float32)
            deg.index_add_(0, rel_dst, rel_weight)
            norm_weight = rel_weight / deg[rel_dst].clamp_min(1.0)
            indices = torch.stack([rel_dst, rel_src], dim=0)
            values = norm_weight
        adjs.append(torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce())
    return adjs


def make_train_only_graph(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
    train_idx: np.ndarray,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    in_train = torch.zeros(num_nodes, dtype=torch.bool)
    in_train[torch.tensor(train_idx, dtype=torch.long)] = True
    mask = in_train[edge_index[:, 0]] & in_train[edge_index[:, 1]]
    return edge_index[mask], edge_type[mask], edge_weight[mask]


def graph_stat_features(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    num_relations: int = 3,
) -> pd.DataFrame:
    src_np = edge_index[:, 0].numpy()
    dst_np = edge_index[:, 1].numpy()
    type_np = edge_type.numpy()
    weight_np = edge_weight.numpy().astype(float)
    rows: dict[str, np.ndarray] = {}

    for rel in range(num_relations):
        mask = type_np == rel
        rel_name = EDGE_TYPE_NAMES.get(rel, str(rel)).replace("-", "_")
        src = src_np[mask]
        dst = dst_np[mask]
        weight = weight_np[mask]
        out_count = np.bincount(src, minlength=num_nodes).astype(float)
        in_count = np.bincount(dst, minlength=num_nodes).astype(float)
        out_weight = np.bincount(src, weights=weight, minlength=num_nodes).astype(float)
        in_weight = np.bincount(dst, weights=weight, minlength=num_nodes).astype(float)
        rows[f"{rel_name}_out_degree"] = out_count
        rows[f"{rel_name}_in_degree"] = in_count
        rows[f"{rel_name}_out_weight"] = out_weight
        rows[f"{rel_name}_in_weight"] = in_weight

    total_out = np.bincount(src_np, minlength=num_nodes).astype(float)
    total_in = np.bincount(dst_np, minlength=num_nodes).astype(float)
    total_out_w = np.bincount(src_np, weights=weight_np, minlength=num_nodes).astype(float)
    total_in_w = np.bincount(dst_np, weights=weight_np, minlength=num_nodes).astype(float)
    rows["total_out_degree"] = total_out
    rows["total_in_degree"] = total_in
    rows["total_degree"] = total_in + total_out
    rows["total_out_weight"] = total_out_w
    rows["total_in_weight"] = total_in_w
    rows["total_weight"] = total_out_w + total_in_w
    rows["direction_balance"] = (total_out_w - total_in_w) / np.maximum(total_out_w + total_in_w, 1.0)
    rows["has_any_edge"] = ((total_in + total_out) > 0).astype(float)

    return pd.DataFrame(rows)


def structural_audit(
    labels: np.ndarray,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    edge_np = edge_index.numpy()
    type_np = edge_type.numpy()
    weight_np = edge_weight.numpy()
    src_label = labels[edge_np[:, 0]]
    dst_label = labels[edge_np[:, 1]]

    pair_rows = []
    for rel in sorted(np.unique(type_np).tolist()):
        rel_mask = type_np == rel
        for src_class, src_name in [(0, "human"), (1, "bot")]:
            for dst_class, dst_name in [(0, "human"), (1, "bot")]:
                mask = rel_mask & (src_label == src_class) & (dst_label == dst_class)
                pair_rows.append(
                    {
                        "edge_type": int(rel),
                        "edge_type_name": EDGE_TYPE_NAMES.get(int(rel), str(rel)),
                        "source_label": src_name,
                        "target_label": dst_name,
                        "edges": int(mask.sum()),
                        "weight_sum": float(weight_np[mask].sum()) if mask.any() else 0.0,
                    }
                )
    pair_df = pd.DataFrame(pair_rows)

    num_nodes = len(labels)
    data = np.ones(edge_np.shape[0] * 2, dtype=float)
    row = np.concatenate([edge_np[:, 0], edge_np[:, 1]])
    col = np.concatenate([edge_np[:, 1], edge_np[:, 0]])
    adj = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    n_components, component_id = csgraph.connected_components(adj, directed=False)
    degree = np.asarray(adj.sum(axis=1)).reshape(-1)

    comp_rows = []
    for comp in range(n_components):
        idx = np.where(component_id == comp)[0]
        bots = int(labels[idx].sum())
        humans = int(len(idx) - bots)
        comp_rows.append(
            {
                "component_id": int(comp),
                "nodes": int(len(idx)),
                "human_nodes": humans,
                "bot_nodes": bots,
                "bot_rate": bots / len(idx),
                "is_pure": bool(bots == 0 or humans == 0),
            }
        )
    comp_df = pd.DataFrame(comp_rows).sort_values("nodes", ascending=False).reset_index(drop=True)

    summary = {
        "num_nodes": int(num_nodes),
        "num_humans": int((labels == 0).sum()),
        "num_bots": int((labels == 1).sum()),
        "num_edges": int(edge_np.shape[0]),
        "edge_index_min": int(edge_np.min()),
        "edge_index_max": int(edge_np.max()),
        "self_loops": int((edge_np[:, 0] == edge_np[:, 1]).sum()),
        "unique_directed_pairs": int(len(set(map(tuple, edge_np.tolist())))),
        "edge_type_counts": {
            EDGE_TYPE_NAMES.get(int(rel), str(rel)): int((type_np == rel).sum())
            for rel in sorted(np.unique(type_np).tolist())
        },
        "edge_weight_min": float(weight_np.min()),
        "edge_weight_mean": float(weight_np.mean()),
        "edge_weight_max": float(weight_np.max()),
        "human_to_bot_edges": int(((src_label == 0) & (dst_label == 1)).sum()),
        "bot_to_human_edges": int(((src_label == 1) & (dst_label == 0)).sum()),
        "human_to_human_edges": int(((src_label == 0) & (dst_label == 0)).sum()),
        "bot_to_bot_edges": int(((src_label == 1) & (dst_label == 1)).sum()),
        "connected_components": int(n_components),
        "pure_components": int(comp_df["is_pure"].sum()),
        "mixed_components": int((~comp_df["is_pure"]).sum()),
        "isolated_nodes": int((degree == 0).sum()),
        "isolated_humans": int(((degree == 0) & (labels == 0)).sum()),
        "isolated_bots": int(((degree == 0) & (labels == 1)).sum()),
        "largest_component_nodes": int(comp_df.iloc[0]["nodes"]),
        "largest_component_humans": int(comp_df.iloc[0]["human_nodes"]),
        "largest_component_bots": int(comp_df.iloc[0]["bot_nodes"]),
    }
    return summary, pair_df, comp_df


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        out["roc_auc"] = float("nan")
    return out


def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor, idx: np.ndarray) -> dict[str, float]:
    y_true = labels[idx].cpu().numpy()
    prob = torch.softmax(logits[idx], dim=1)[:, 1].detach().cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    return score_predictions(y_true, pred, prob)


def train_rgcn(
    model: nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    split: Split,
    train_adjs: list[torch.Tensor],
    eval_adjs: list[torch.Tensor],
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[pd.DataFrame, torch.Tensor]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -math.inf
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_logits = model(x, train_adjs)
        loss = F.cross_entropy(train_logits[split.train_idx], labels[split.train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(x, eval_adjs)
            train_scores = evaluate_logits(eval_logits, labels, split.train_idx)
            val_scores = evaluate_logits(eval_logits, labels, split.val_idx)
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "train_f1": train_scores["f1"],
                "val_f1": val_scores["f1"],
                "train_accuracy": train_scores["accuracy"],
                "val_accuracy": val_scores["accuracy"],
            }
        )
        if val_scores["f1"] > best_val_f1:
            best_val_f1 = val_scores["f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, eval_adjs)
    return pd.DataFrame(history), logits


def run_ml_models(
    features: pd.DataFrame,
    labels: np.ndarray,
    split: Split,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    x = features.to_numpy(dtype=np.float32)
    rows: list[dict[str, object]] = []
    probs: dict[str, np.ndarray] = {}
    models = {
        "GraphStats_LR": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
        ),
        "GraphStats_RF": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    for model_name, model in models.items():
        model.fit(x[split.train_idx], labels[split.train_idx])
        all_prob = model.predict_proba(x)[:, 1]
        probs[model_name] = all_prob
        for split_name, idx in [
            ("train", split.train_idx),
            ("validation", split.val_idx),
            ("test", split.test_idx),
        ]:
            pred = (all_prob[idx] >= 0.5).astype(int)
            scores = score_predictions(labels[idx], pred, all_prob[idx])
            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "model_zh": MODEL_NAMES_ZH[model_name],
                    "split": split_name,
                    **scores,
                }
            )
    return rows, probs


def run_rgcn_models(
    features: pd.DataFrame,
    labels_np: np.ndarray,
    split: Split,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
    seed: int,
    epochs: int,
    hidden_dim: int,
) -> tuple[list[dict[str, object]], dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    num_nodes = len(labels_np)
    labels = torch.tensor(labels_np, dtype=torch.long)
    full_adjs = make_sparse_adjs(edge_index, edge_type, edge_weight, num_nodes)
    tr_ei, tr_et, tr_ew = make_train_only_graph(edge_index, edge_type, edge_weight, split.train_idx, num_nodes)
    train_adjs = make_sparse_adjs(tr_ei, tr_et, tr_ew, num_nodes)

    scaler = StandardScaler()
    x_graph_np = scaler.fit(features.iloc[split.train_idx]).transform(features).astype(np.float32)
    x_graph = torch.tensor(x_graph_np, dtype=torch.float32)
    x_constant = torch.ones((num_nodes, 1), dtype=torch.float32)

    runs = {
        "Constant_RGCN": (x_constant, GraphOnlyRGCN(1, hidden_dim=hidden_dim)),
        "GraphStats_RGCN": (x_graph, GraphOnlyRGCN(x_graph.shape[1], hidden_dim=hidden_dim)),
    }

    rows: list[dict[str, object]] = []
    histories: dict[str, pd.DataFrame] = {}
    probs: dict[str, np.ndarray] = {}
    for model_name, (x, model) in runs.items():
        history, logits = train_rgcn(
            model,
            x,
            labels,
            split,
            train_adjs=train_adjs,
            eval_adjs=full_adjs,
            epochs=epochs,
            lr=1e-3,
            weight_decay=1e-3,
        )
        histories[model_name] = history
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs[model_name] = prob
        for split_name, idx in [
            ("train", split.train_idx),
            ("validation", split.val_idx),
            ("test", split.test_idx),
        ]:
            pred = (prob[idx] >= 0.5).astype(int)
            scores = score_predictions(labels_np[idx], pred, prob[idx])
            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "model_zh": MODEL_NAMES_ZH[model_name],
                    "split": split_name,
                    **scores,
                }
            )
    return rows, histories, probs


def plot_metrics(summary: pd.DataFrame, path: Path) -> None:
    test = summary[summary["split"] == "test"].copy()
    test = test.sort_values("f1_mean", ascending=False)
    labels = test["model_zh"].tolist()
    x = np.arange(len(test))
    width = 0.35
    plt.figure(figsize=(10.4, 5.2))
    plt.bar(
        x - width / 2,
        test["accuracy_mean"],
        width,
        yerr=test["accuracy_std"],
        capsize=4,
        label="Accuracy",
        color="#4C78A8",
    )
    plt.bar(
        x + width / 2,
        test["f1_mean"],
        width,
        yerr=test["f1_std"],
        capsize=4,
        label="Bot F1",
        color="#E45756",
    )
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("5 seeds mean score")
    plt.title("Graph-only bot detection performance")
    plt.legend()
    for i, row in enumerate(test.itertuples()):
        plt.text(i - width / 2, row.accuracy_mean + 0.025, f"{row.accuracy_mean:.3f}", ha="center", fontsize=8)
        plt.text(i + width / 2, row.f1_mean + 0.025, f"{row.f1_mean:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_edge_pairs(pair_df: pd.DataFrame, path: Path) -> None:
    pivot = pair_df.copy()
    pivot["pair"] = pivot["source_label"] + " -> " + pivot["target_label"]
    type_order = [EDGE_TYPE_NAMES[k] for k in sorted(EDGE_TYPE_NAMES)]
    pair_order = ["human -> human", "human -> bot", "bot -> human", "bot -> bot"]
    values = {
        pair: [
            int(
                pivot[
                    (pivot["edge_type_name"] == edge_type)
                    & (pivot["pair"] == pair)
                ]["edges"].sum()
            )
            for edge_type in type_order
        ]
        for pair in pair_order
    }
    x = np.arange(len(type_order))
    bottom = np.zeros(len(type_order))
    colors = {
        "human -> human": "#4C78A8",
        "human -> bot": "#72B7B2",
        "bot -> human": "#E45756",
        "bot -> bot": "#F58518",
    }
    plt.figure(figsize=(8.2, 5.2))
    for pair in pair_order:
        vals = np.array(values[pair])
        plt.bar(x, vals, bottom=bottom, label=pair, color=colors[pair])
        bottom += vals
    plt.xticks(x, type_order)
    plt.ylabel("Edges")
    plt.title("Edge label-pair distribution by relation type")
    plt.legend()
    for i, total in enumerate(bottom):
        plt.text(i, total + 500, f"{int(total):,}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_component_sizes(comp_df: pd.DataFrame, path: Path) -> None:
    top = comp_df.head(20).copy()
    x = np.arange(len(top))
    plt.figure(figsize=(10.5, 5.0))
    plt.bar(x, top["human_nodes"], label="human", color="#4C78A8")
    plt.bar(x, top["bot_nodes"], bottom=top["human_nodes"], label="bot", color="#E45756")
    plt.yscale("log")
    plt.xlabel("Connected components sorted by size")
    plt.ylabel("Nodes, log scale")
    plt.title("Largest undirected connected components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion(labels: np.ndarray, prob: np.ndarray, idx: np.ndarray, model_zh: str, path: Path) -> None:
    pred = (prob[idx] >= 0.5).astype(int)
    cm = confusion_matrix(labels[idx], pred, labels=[0, 1])
    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["human", "bot"])
    plt.yticks([0, 1], ["human", "bot"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Test confusion matrix: {model_zh}")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_probability_hist(labels: np.ndarray, prob: np.ndarray, idx: np.ndarray, model_zh: str, path: Path) -> None:
    y = labels[idx]
    p = prob[idx]
    bins = np.linspace(0, 1, 21)
    plt.figure(figsize=(7.4, 4.7))
    plt.hist(p[y == 0], bins=bins, alpha=0.75, label="true human", color="#4C78A8")
    plt.hist(p[y == 1], bins=bins, alpha=0.75, label="true bot", color="#E45756")
    plt.xlabel("Predicted bot probability")
    plt.ylabel("Users")
    plt.title(f"Test probability distribution: {model_zh}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_training_curve(history: pd.DataFrame, model_zh: str, path: Path) -> None:
    plt.figure(figsize=(7.8, 4.8))
    plt.plot(history["epoch"], history["train_f1"], label="train F1")
    plt.plot(history["epoch"], history["val_f1"], label="validation F1")
    plt.plot(history["epoch"], history["loss"], label="loss", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Score / loss")
    plt.title(f"Training curve: {model_zh}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_report(
    out_dir: Path,
    metrics_summary: pd.DataFrame,
    graph_summary: dict[str, object],
    best_model: str,
) -> None:
    test = metrics_summary[metrics_summary["split"] == "test"].copy()
    test = test.sort_values("f1_mean", ascending=False)
    best_row = test.iloc[0]
    report_path = out_dir / "GRAPH_ONLY_REPORT_CN.md"
    lines = [
        "# BotSim-24 仅图关系 Bot 识别实验报告",
        "",
        "## 1. 实验口径",
        "",
        "这次实验只使用图关系，不使用评论文本、帖子文本、用户简介、用户名、发帖数、评论数、subreddit 等用户元数据。标签只按官方 README 给出的顺序生成：前 1907 个用户是 human，后 1000 个用户是 bot。",
        "",
        "输入图来自 `BotSim-24-mini-sample/edge_index.pt`、`edge_type.pt`、`edge_weight.pt`。虽然目录名叫 mini-sample，但该边数为 46,518，和论文表格中的 BotSim-24 图边数一致。",
        "",
        "本实验比较四类模型：",
        "",
        "- `图结构统计 + 逻辑回归`：每个用户只看入度、出度、不同边类型的加权/非加权度数等结构统计。",
        "- `图结构统计 + 随机森林`：同样只看结构统计，但允许非线性规则。",
        "- `常数节点特征 + RGCN`：所有节点初始特征都是 1，模型只能靠邻居聚合、边类型和图拓扑产生差异。",
        "- `图结构统计 + RGCN`：先给每个节点图结构统计，再让 RGCN 按关系类型聚合邻居结构。",
        "",
        "## 2. 图本身的关键结构",
        "",
        f"- 节点数：{graph_summary['num_nodes']}，human={graph_summary['num_humans']}，bot={graph_summary['num_bots']}。",
        f"- 有向边数：{graph_summary['num_edges']}，边类型分布：{graph_summary['edge_type_counts']}。",
        f"- human -> bot 边：{graph_summary['human_to_bot_edges']}；bot -> human 边：{graph_summary['bot_to_human_edges']}；human -> human 边：{graph_summary['human_to_human_edges']}；bot -> bot 边：{graph_summary['bot_to_bot_edges']}。",
        f"- 无向连通分量数：{graph_summary['connected_components']}；最大分量含 {graph_summary['largest_component_nodes']} 个节点，其中 human={graph_summary['largest_component_humans']}，bot={graph_summary['largest_component_bots']}。",
        f"- 孤立节点：{graph_summary['isolated_nodes']}，其中 human={graph_summary['isolated_humans']}，bot={graph_summary['isolated_bots']}。",
        "",
        "最值得注意的是：图中没有 human -> bot 边，但有大量 bot -> human 边。这不是文本信号，而是图构造中的方向性结构信号。换句话说，如果 bot 主要去评论/回复人类内容，而人类几乎不指向 bot，那么模型只看边的方向和类型也可能获得很强的区分能力。",
        "",
        "## 3. 五个随机种子的测试集结果",
        "",
        "| 模型 | Accuracy均值 | Accuracy标准差 | Bot F1均值 | Bot F1标准差 | Bot Recall均值 | Bot Precision均值 | ROC-AUC均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test.itertuples():
        lines.append(
            f"| {row.model_zh} | {row.accuracy_mean:.4f} | {row.accuracy_std:.4f} | "
            f"{row.f1_mean:.4f} | {row.f1_std:.4f} | {row.recall_mean:.4f} | "
            f"{row.precision_mean:.4f} | {row.roc_auc_mean:.4f} |"
        )
    lines.extend(
        [
            "",
            f"表现最好的模型是 **{best_row.model_zh}**，测试集 Bot F1={best_row.f1_mean:.4f}±{best_row.f1_std:.4f}，Accuracy={best_row.accuracy_mean:.4f}±{best_row.accuracy_std:.4f}。",
            "",
            "## 4. 结果怎么理解",
            "",
            "第一，图-only 不是“模型什么都不知道”。它虽然看不到文本，但能看到谁指向谁、通过哪类互动指向、互动权重是多少。对 BotSim-24 来说，这些结构本身就带有强烈的人机差异。",
            "",
            "第二，结构统计模型如果已经表现很好，说明很多区分信号不需要深层图神经网络也能抓到。例如 bot 的出边、入边、按类型拆分后的入出权重、方向平衡等，已经是强特征。",
            "",
            "第三，RGCN 的意义是把邻居信息也纳入判断。常数节点特征 RGCN 是最严格的 GNN 版本，因为每个节点一开始完全一样，只能通过图拓扑和关系类型逐层聚合产生不同表示。如果它仍能超过随机水平，就说明图结构本身确实含有可学习信号。",
            "",
            "第四，本实验是 transductive node classification 风格：评估节点所在的整张图在推理时可见，训练损失只使用训练节点标签。为了降低训练阶段的测试节点信息污染，脚本训练时的 RGCN 聚合只使用 train-train 边；验证/测试推理时再在全图上计算节点表示。这仍然不等价于“遇到全新平台、全新用户图”的归纳泛化。",
            "",
            "## 5. 和原文策略的差别",
            "",
            "原文的 GNN 模型不是纯图模型。开源 `BotRGCN.py` 里先把数值元数据特征和文本向量拼接，再送入 RGCN。因此原文结果回答的是“文本 + 元数据 + 图关系”的检测能力，而本实验回答的是“只看图关系还有多少检测能力”。这两个问题不同，不能直接把本实验分数当作论文表格的复现。",
            "",
            "## 6. 输出文件",
            "",
            "- `metrics_all_seeds.csv`：每个随机种子、每个模型、每个 split 的完整指标。",
            "- `metrics_summary.csv`：5 个随机种子的均值和标准差。",
            "- `graph_structural_audit.json`：图结构审计摘要。",
            "- `edge_label_pair_counts.csv`：不同边类型下 human/bot 源目标组合。",
            "- `component_summary.csv`：连通分量规模和 bot 比例。",
            "- `figures/`：指标柱状图、边组合图、混淆矩阵、概率分布和训练曲线。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    root = find_project_root()
    out_dir = root / "baselines" / "graph" / "graph_only_run"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    users, labels, edge_index, edge_type, edge_weight = load_graph(root)
    graph_summary, pair_df, comp_df = structural_audit(labels, edge_index, edge_type, edge_weight)
    graph_features = graph_stat_features(edge_index, edge_type, edge_weight, len(labels))

    graph_features.to_csv(out_dir / "graph_only_node_features.csv", index_label="node_id")
    pair_df.to_csv(out_dir / "edge_label_pair_counts.csv", index=False)
    comp_df.to_csv(out_dir / "component_summary.csv", index=False)
    (out_dir / "graph_structural_audit.json").write_text(
        json.dumps(graph_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_rows: list[dict[str, object]] = []
    all_probs_seed0: dict[str, np.ndarray] = {}
    histories_seed0: dict[str, pd.DataFrame] = {}
    split_seed0: Split | None = None

    for seed in args.seeds:
        set_seed(seed)
        split = make_split(labels, seed)
        ml_rows, ml_probs = run_ml_models(graph_features, labels, split, seed)
        rgcn_rows, histories, rgcn_probs = run_rgcn_models(
            graph_features,
            labels,
            split,
            edge_index,
            edge_type,
            edge_weight,
            seed=seed,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
        )
        all_rows.extend(ml_rows)
        all_rows.extend(rgcn_rows)
        if seed == args.seeds[0]:
            all_probs_seed0 = {**ml_probs, **rgcn_probs}
            histories_seed0 = histories
            split_seed0 = split
            pred_df = users[["user_id", "name"]].copy()
            pred_df["true_label"] = np.where(labels == 1, "bot", "human")
            pred_df["split"] = "unused"
            pred_df.loc[split.train_idx, "split"] = "train"
            pred_df.loc[split.val_idx, "split"] = "validation"
            pred_df.loc[split.test_idx, "split"] = "test"
            for model_name, prob in all_probs_seed0.items():
                pred_df[f"{model_name}_bot_probability"] = prob
                pred_df[f"{model_name}_predicted_label"] = np.where(prob >= 0.5, "bot", "human")
            pred_df.to_csv(out_dir / f"seed_{seed}_predictions.csv", index=False)

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(out_dir / "metrics_all_seeds.csv", index=False)
    summary = (
        metrics.groupby(["model", "model_zh", "split"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
        )
        .sort_values(["split", "f1_mean"], ascending=[True, False])
    )
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    plot_metrics(summary, fig_dir / "01_graph_only_test_metrics.png")
    plot_edge_pairs(pair_df, fig_dir / "02_edge_label_pair_distribution.png")
    plot_component_sizes(comp_df, fig_dir / "03_connected_components.png")

    if split_seed0 is None:
        raise RuntimeError("No seed was run.")
    best_model = (
        summary[summary["split"] == "test"].sort_values("f1_mean", ascending=False).iloc[0]["model"]
    )
    for model_name, prob in all_probs_seed0.items():
        safe_name = model_name.lower()
        plot_confusion(
            labels,
            prob,
            split_seed0.test_idx,
            MODEL_NAMES_ZH[model_name],
            fig_dir / f"04_{safe_name}_confusion_matrix_seed{args.seeds[0]}.png",
        )
        plot_probability_hist(
            labels,
            prob,
            split_seed0.test_idx,
            MODEL_NAMES_ZH[model_name],
            fig_dir / f"05_{safe_name}_probability_hist_seed{args.seeds[0]}.png",
        )
    for model_name, history in histories_seed0.items():
        history.to_csv(out_dir / f"{model_name}_seed{args.seeds[0]}_history.csv", index=False)
        plot_training_curve(
            history,
            MODEL_NAMES_ZH[model_name],
            fig_dir / f"06_{model_name.lower()}_training_curve_seed{args.seeds[0]}.png",
        )

    save_report(out_dir, summary, graph_summary, str(best_model))
    print(f"Done. Outputs written to: {out_dir}")
    print(summary[summary["split"] == "test"].sort_values("f1_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
