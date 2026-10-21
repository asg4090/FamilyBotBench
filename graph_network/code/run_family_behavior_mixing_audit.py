#!/usr/bin/env python3
"""Audit generated family behavior, mixing, community structure, and human alignment."""

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
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, confusion_matrix, f1_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_graphstats_baselines import build_graph_features, load_dataset  # noqa: E402


FAMILIES = [
    "G1_population_calibrated",
    "G2_anchor_behavior",
    "G3_contextual_interactive",
    "G4_slot_conditioned",
]
MAIN_FAMILIES = ["original_human", *FAMILIES]
FAMILY_LABELS = {
    "original_human": "Human",
    "G1_population_calibrated": "G1",
    "G2_anchor_behavior": "G2",
    "G3_contextual_interactive": "G3",
    "G4_slot_conditioned": "G4",
}
ACTION_ORDER = ["post", "comment1", "comment2", "revisit", "skip"]
KEY_HUMAN_ALIGNMENT_FEATURES = [
    "total_degree",
    "total_out_degree",
    "total_in_degree",
    "direction_balance",
    "reciprocal_share",
    "comment1_rate",
    "comment2_rate",
    "receive_rate",
    "component_size",
    "unique_target_authors",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_parse_error": True, "_line_no": line_no, "_raw": line[:500]})
    return pd.DataFrame(rows)


def family_label(value: Any) -> str:
    return FAMILY_LABELS.get(str(value), str(value))


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def load_inputs(run_dir: Path) -> dict[str, Any]:
    users = pd.read_csv(run_dir / "users.csv", dtype=str).fillna("")
    posts = pd.read_csv(run_dir / "posts.csv", dtype=str, low_memory=False).fillna("")
    comments = pd.read_csv(run_dir / "comments.csv", dtype=str, low_memory=False).fillna("")
    profiles = pd.read_csv(run_dir / "persona_profiles.csv", dtype=str).fillna("")
    nodes, user_edges, manifest = load_dataset(run_dir)
    all_edges = pd.read_csv(run_dir / "graph" / "edge_table.csv", dtype=str, low_memory=False).fillna("")
    actions = read_jsonl(run_dir / "action_trace.jsonl")
    events = read_jsonl(run_dir / "events.jsonl")
    memory = read_jsonl(run_dir / "memory_trace.jsonl")
    integrity = read_json(run_dir.parent.parent / "results" / "partial_1202actions_stopped_live_20260604" / "dataset_integrity_summary.json")
    export_summary = read_json(run_dir / "graph" / "export_summary.json")
    return {
        "users": users,
        "posts": posts,
        "comments": comments,
        "profiles": profiles,
        "nodes": nodes,
        "user_edges": user_edges,
        "all_edges": all_edges,
        "actions": actions,
        "events": events,
        "memory": memory,
        "manifest": manifest,
        "integrity": integrity,
        "export_summary": export_summary,
    }


def add_action_time_fields(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return actions
    actions = actions.copy()
    actions["time_dt"] = pd.to_datetime(actions.get("time", ""), errors="coerce")
    actions["active_hour"] = actions["time_dt"].dt.floor("h").astype(str)
    return actions


def behavior_by_bot(data: dict[str, Any], result_dir: Path, fig_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = data["users"]
    actions = add_action_time_fields(data["actions"])
    user_edges = data["user_edges"].copy()
    bot_users = users[users["family"].isin(FAMILIES)][["user_id", "family"]].drop_duplicates()

    if actions.empty:
        action_counts = pd.DataFrame(index=bot_users["user_id"])
    else:
        action_counts = pd.crosstab(actions["actor_user_id"], actions["action_type"])
    for action in ACTION_ORDER:
        if action not in action_counts.columns:
            action_counts[action] = 0
    action_counts = action_counts[ACTION_ORDER].rename(columns={a: f"{a}_count" for a in ACTION_ORDER})

    action_meta = pd.DataFrame(index=bot_users["user_id"])
    if not actions.empty:
        active = actions.groupby("actor_user_id")["active_hour"].nunique()
        threads = actions.groupby("actor_user_id")["target_post_id"].apply(lambda s: len({x for x in s.astype(str) if x}))
        targets = actions.groupby("actor_user_id")["target_author_id"].apply(lambda s: len({x for x in s.astype(str) if x}))
        action_meta["active_window_count"] = active
        action_meta["unique_threads_joined"] = threads
        action_meta["unique_action_target_authors"] = targets

    id_to_family = users.set_index("user_id")["family"].to_dict()
    user_edges["src_family"] = user_edges["src_id"].map(id_to_family).fillna("unknown")
    user_edges["dst_family"] = user_edges["dst_id"].map(id_to_family).fillna("unknown")
    user_edges["weight_num"] = clean_numeric(user_edges.get("weight", pd.Series(1, index=user_edges.index)))

    edge_meta = pd.DataFrame(index=bot_users["user_id"])
    outbound = user_edges[user_edges["src_id"].isin(bot_users["user_id"])]
    inbound = user_edges[user_edges["dst_id"].isin(bot_users["user_id"])]
    if not outbound.empty:
        edge_meta["out_user_edges"] = outbound.groupby("src_id")["weight_num"].sum()
        edge_meta["bot_to_bot_edges"] = outbound[outbound["dst_family"].isin(FAMILIES)].groupby("src_id")["weight_num"].sum()
        edge_meta["bot_to_human_edges"] = outbound[outbound["dst_family"].eq("original_human")].groupby("src_id")["weight_num"].sum()
        edge_meta["unique_users_interacted"] = outbound.groupby("src_id")["dst_id"].nunique()
        edge_meta["sent_reply_count"] = outbound[outbound["edge_type"].str.contains("replies_to_user_comment", na=False)].groupby("src_id")["weight_num"].sum()
    if not inbound.empty:
        edge_meta["in_user_edges"] = inbound.groupby("dst_id")["weight_num"].sum()
        edge_meta["human_to_bot_edges"] = inbound[inbound["src_family"].eq("original_human")].groupby("dst_id")["weight_num"].sum()
        edge_meta["bot_to_bot_in_edges"] = inbound[inbound["src_family"].isin(FAMILIES)].groupby("dst_id")["weight_num"].sum()
        edge_meta["received_reply_count"] = inbound[inbound["edge_type"].str.contains("receives_reply", na=False)].groupby("dst_id")["weight_num"].sum()

    behavior = bot_users.set_index("user_id").join(action_counts, how="left").join(action_meta, how="left").join(edge_meta, how="left").fillna(0).reset_index()
    count_cols = [c for c in behavior.columns if c.endswith("_count") or c.endswith("_edges") or c in ["active_window_count", "unique_threads_joined", "unique_users_interacted", "unique_action_target_authors"]]
    for col in count_cols:
        behavior[col] = clean_numeric(behavior[col])
    action_total = behavior[[f"{a}_count" for a in ACTION_ORDER]].sum(axis=1).replace(0, np.nan)
    for action in ACTION_ORDER:
        behavior[f"{action}_share"] = behavior[f"{action}_count"] / action_total
    behavior = behavior.fillna(0)
    behavior["bot_to_bot_share_out"] = behavior["bot_to_bot_edges"] / behavior["out_user_edges"].replace(0, np.nan)
    behavior["bot_to_human_share_out"] = behavior["bot_to_human_edges"] / behavior["out_user_edges"].replace(0, np.nan)
    behavior = behavior.fillna(0)

    summary_aggs: dict[str, list[str]] = {}
    for col in behavior.columns:
        if col in {"user_id", "family"}:
            continue
        if pd.api.types.is_numeric_dtype(behavior[col]):
            summary_aggs[col] = ["mean", "median", "std", "min", "max", "sum"]
    summary = behavior.groupby("family").agg(summary_aggs)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()

    behavior.to_csv(result_dir / "behavior_by_bot.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(result_dir / "behavior_by_family.csv", index=False, encoding="utf-8-sig")

    action_summary = behavior.groupby("family")[[f"{a}_count" for a in ACTION_ORDER]].sum()
    action_share = action_summary.div(action_summary.sum(axis=1), axis=0).fillna(0)
    ax = action_share.rename(index=family_label).plot(kind="bar", stacked=True, figsize=(10, 5), colormap="tab20")
    ax.set_title("Action distribution by generated family")
    ax.set_ylabel("Share of actions")
    ax.set_xlabel("Family")
    ax.legend([a.replace("_count", "") for a in action_share.columns], bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    savefig(fig_dir / "action_distribution_by_family.png")

    temp = behavior[["family", "active_window_count", "post_count", "comment1_count", "comment2_count", "revisit_count"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    temp.boxplot(column="active_window_count", by="family", ax=axes[0], grid=False)
    axes[0].set_title("Active windows per bot")
    axes[0].set_xlabel("Family")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)
    action_mean = temp.groupby("family")[["post_count", "comment1_count", "comment2_count", "revisit_count"]].mean()
    action_mean.rename(index=family_label).plot(kind="bar", ax=axes[1])
    axes[1].set_title("Mean active action counts")
    axes[1].set_xlabel("Family")
    axes[1].set_ylabel("Mean count")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("")
    savefig(fig_dir / "temporal_activity_by_family.png")
    return behavior, summary


def family_mixing(data: dict[str, Any], result_dir: Path, fig_dir: Path, seed: int, permutations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = data["users"]
    edges = data["user_edges"].copy()
    id_to_family = users.set_index("user_id")["family"].to_dict()
    edges["src_family"] = edges["src_id"].map(id_to_family).fillna("unknown")
    edges["dst_family"] = edges["dst_id"].map(id_to_family).fillna("unknown")
    edges["weight_num"] = clean_numeric(edges.get("weight", pd.Series(1, index=edges.index)))
    main_edges = edges[edges["src_family"].isin(MAIN_FAMILIES) & edges["dst_family"].isin(MAIN_FAMILIES)].copy()

    matrix = (
        main_edges.groupby(["src_family", "dst_family"])["weight_num"]
        .sum()
        .reset_index(name="edge_weight")
    )
    pivot = matrix.pivot_table(index="src_family", columns="dst_family", values="edge_weight", fill_value=0, aggfunc="sum")
    pivot = pivot.reindex(index=MAIN_FAMILIES, columns=MAIN_FAMILIES, fill_value=0)
    long = pivot.reset_index().melt(id_vars="src_family", var_name="dst_family", value_name="edge_weight")
    long["src_label"] = long["src_family"].map(family_label)
    long["dst_label"] = long["dst_family"].map(family_label)
    long.to_csv(result_dir / "family_mixing_matrix.csv", index=False, encoding="utf-8-sig")

    bot_edges = main_edges[main_edges["src_family"].isin(FAMILIES)].copy()
    bot_to_bot = bot_edges[bot_edges["dst_family"].isin(FAMILIES)].copy()
    observed = float((bot_to_bot["src_family"] == bot_to_bot["dst_family"]).mean()) if len(bot_to_bot) else np.nan

    rng = np.random.default_rng(seed)
    bot_ids = users[users["family"].isin(FAMILIES)]["user_id"].astype(str).to_numpy()
    family_values = users[users["family"].isin(FAMILIES)]["family"].astype(str).to_numpy()
    src_ids = bot_to_bot["src_id"].astype(str).to_numpy()
    dst_ids = bot_to_bot["dst_id"].astype(str).to_numpy()
    random_values: list[float] = []
    if len(bot_to_bot) and len(bot_ids):
        base_map = dict(zip(bot_ids, family_values))
        for _ in range(permutations):
            shuffled = family_values.copy()
            rng.shuffle(shuffled)
            shuffled_map = dict(zip(bot_ids, shuffled))
            same = np.array([shuffled_map.get(s, base_map.get(s)) == shuffled_map.get(d, base_map.get(d)) for s, d in zip(src_ids, dst_ids)])
            random_values.append(float(same.mean()))
    random_arr = np.array(random_values, dtype=float)
    random_mean = float(random_arr.mean()) if len(random_arr) else np.nan
    random_std = float(random_arr.std(ddof=1)) if len(random_arr) > 1 else np.nan
    z = float((observed - random_mean) / random_std) if random_std and not math.isnan(random_std) else np.nan
    if len(random_arr) and not math.isnan(observed):
        empirical_p = float((np.sum(random_arr >= observed) + 1) / (len(random_arr) + 1))
    else:
        empirical_p = np.nan
    perm = pd.DataFrame(
        [
            {
                "scope": "generated_bot_to_generated_bot_edges",
                "edge_count": int(len(bot_to_bot)),
                "observed": observed,
                "random_mean": random_mean,
                "random_std": random_std,
                "z_score": z,
                "empirical_p": empirical_p,
                "permutations": int(permutations),
            }
        ]
    )
    perm.to_csv(result_dir / "same_family_permutation_test.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_pivot = pivot.rename(index=family_label, columns=family_label)
    im = ax.imshow(plot_pivot.to_numpy(dtype=float), cmap="Blues")
    ax.set_xticks(np.arange(len(plot_pivot.columns)), labels=plot_pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(plot_pivot.index)), labels=plot_pivot.index)
    ax.set_title("Family mixing matrix, user-user edge weights")
    for i in range(plot_pivot.shape[0]):
        for j in range(plot_pivot.shape[1]):
            ax.text(j, i, f"{plot_pivot.iloc[i, j]:.0f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig_dir / "family_mixing_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(random_arr):
        ax.hist(random_arr, bins=30, alpha=0.75, color="#4C78A8", label="random label baseline")
    if not math.isnan(observed):
        ax.axvline(observed, color="#E45756", linewidth=2, label=f"observed={observed:.3f}")
    ax.set_title("Same-family edge share permutation test")
    ax.set_xlabel("Same-family share among generated bot-to-bot edges")
    ax.set_ylabel("Permutation count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    savefig(fig_dir / "same_family_edge_permutation_test.png")
    return long, perm


def community_audit(data: dict[str, Any], result_dir: Path, fig_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = data["users"]
    edges = data["user_edges"].copy()
    id_to_family = users.set_index("user_id")["family"].to_dict()
    edges["src_family"] = edges["src_id"].map(id_to_family).fillna("unknown")
    edges["dst_family"] = edges["dst_id"].map(id_to_family).fillna("unknown")
    main_nodes = users[users["family"].isin(MAIN_FAMILIES)]["user_id"].astype(str).tolist()
    main_set = set(main_nodes)
    main_edges = edges[edges["src_id"].isin(main_set) & edges["dst_id"].isin(main_set)].copy()

    graph = nx.Graph()
    graph.add_nodes_from(main_nodes)
    for row in main_edges.itertuples(index=False):
        src = str(row.src_id)
        dst = str(row.dst_id)
        if src == dst:
            continue
        if graph.has_edge(src, dst):
            graph[src][dst]["weight"] += 1.0
        else:
            graph.add_edge(src, dst, weight=1.0)

    method = "none"
    communities: list[set[str]]
    if graph.number_of_edges() == 0:
        communities = [{node} for node in graph.nodes]
    else:
        try:
            communities = [set(c) for c in nx.algorithms.community.louvain_communities(graph, weight="weight", seed=20260605)]
            method = "networkx_louvain"
        except Exception:
            communities = [set(c) for c in nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")]
            method = "greedy_modularity"

    node_to_comm: dict[str, int] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = cid

    rows: list[dict[str, Any]] = []
    for cid, comm in enumerate(communities):
        families = [id_to_family.get(node, "unknown") for node in comm]
        counts = Counter(families)
        main_count = sum(counts.get(fam, 0) for fam in MAIN_FAMILIES)
        bot_count = sum(counts.get(fam, 0) for fam in FAMILIES)
        dominant_family, dominant_count = counts.most_common(1)[0] if counts else ("unknown", 0)
        row = {
            "community_id": cid,
            "method": method,
            "size": int(len(comm)),
            "main_family_node_count": int(main_count),
            "bot_count": int(bot_count),
            "bot_share": float(bot_count / len(comm)) if comm else np.nan,
            "dominant_family": dominant_family,
            "dominant_family_share": float(dominant_count / len(comm)) if comm else np.nan,
            "is_bot_only": bool(len(comm) > 0 and bot_count == len(comm)),
        }
        for fam in MAIN_FAMILIES:
            row[f"{fam}_count"] = int(counts.get(fam, 0))
            row[f"{fam}_share"] = float(counts.get(fam, 0) / len(comm)) if comm else np.nan
        rows.append(row)
    comp = pd.DataFrame(rows).sort_values(["size", "community_id"], ascending=[False, True])
    comp.to_csv(result_dir / "community_family_composition.csv", index=False, encoding="utf-8-sig")

    node_order = [node for node in main_nodes if node in node_to_comm]
    community_labels = [node_to_comm[node] for node in node_order]
    family_labels = [id_to_family.get(node, "unknown") for node in node_order]
    bot_node_order = [node for node in node_order if id_to_family.get(node) in FAMILIES]
    bot_comm_labels = [node_to_comm[node] for node in bot_node_order]
    bot_family_labels = [id_to_family.get(node, "unknown") for node in bot_node_order]

    family_partition = []
    for fam in MAIN_FAMILIES:
        nodes = {node for node in main_nodes if id_to_family.get(node) == fam and node in graph}
        if nodes:
            family_partition.append(nodes)
    try:
        family_modularity = nx.algorithms.community.modularity(graph, family_partition, weight="weight") if graph.number_of_edges() else np.nan
    except Exception:
        family_modularity = np.nan

    summary = pd.DataFrame(
        [
            {
                "community_method": method,
                "node_count": int(graph.number_of_nodes()),
                "edge_count": int(graph.number_of_edges()),
                "community_count": int(len(communities)),
                "largest_community_size": int(comp["size"].max()) if not comp.empty else 0,
                "mean_community_purity": float(comp["dominant_family_share"].mean()) if not comp.empty else np.nan,
                "weighted_community_purity": float((comp["dominant_family_share"] * comp["size"]).sum() / comp["size"].sum()) if not comp.empty and comp["size"].sum() else np.nan,
                "bot_only_community_count": int(comp["is_bot_only"].sum()) if not comp.empty else 0,
                "bot_only_node_share": float(comp.loc[comp["is_bot_only"], "size"].sum() / comp["size"].sum()) if not comp.empty and comp["size"].sum() else np.nan,
                "bot_heavy_community_count_80pct": int((comp["bot_share"] >= 0.8).sum()) if not comp.empty else 0,
                "max_bot_share_in_community": float(comp["bot_share"].max()) if not comp.empty else np.nan,
                "max_bot_share_community_size": int(comp.loc[comp["bot_share"].idxmax(), "size"]) if not comp.empty else 0,
                "max_bot_share_community_bot_count": int(comp.loc[comp["bot_share"].idxmax(), "bot_count"]) if not comp.empty else 0,
                "family_modularity": float(family_modularity) if not pd.isna(family_modularity) else np.nan,
                "nmi_all_main": float(normalized_mutual_info_score(family_labels, community_labels)) if node_order else np.nan,
                "ari_all_main": float(adjusted_rand_score(family_labels, community_labels)) if node_order else np.nan,
                "nmi_generated_only": float(normalized_mutual_info_score(bot_family_labels, bot_comm_labels)) if bot_node_order else np.nan,
                "ari_generated_only": float(adjusted_rand_score(bot_family_labels, bot_comm_labels)) if bot_node_order else np.nan,
            }
        ]
    )
    summary.to_csv(result_dir / "community_summary.csv", index=False, encoding="utf-8-sig")

    top = comp.head(20).copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(top))
    x = np.arange(len(top))
    for fam in MAIN_FAMILIES:
        values = top[f"{fam}_share"].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=family_label(fam))
        bottom += values
    ax.set_title("Top communities by family composition")
    ax.set_xlabel("Community rank by size")
    ax.set_ylabel("Family share")
    ax.set_xticks(x, [str(i + 1) for i in range(len(top))])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    savefig(fig_dir / "community_family_purity.png")
    return comp, summary


def feature_separability(data: dict[str, Any], result_dir: Path, fig_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = data["nodes"].copy()
    edges = data["user_edges"].copy()
    features, _audit = build_graph_features(nodes, edges)
    feature_df = pd.concat([nodes[["user_id", "family", "label", "source"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    feature_df.to_csv(result_dir / "graph_feature_by_bot.csv", index=False, encoding="utf-8-sig")

    bot_mask = feature_df["family"].isin(FAMILIES)
    bot_features = feature_df.loc[bot_mask, features.columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y_text = feature_df.loc[bot_mask, "family"].astype(str).to_numpy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    metrics_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    if len(np.unique(y)) >= 2 and len(y) >= 15:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        models = {
            "Family_LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)),
            "Family_RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        }
        x_np = bot_features.to_numpy(dtype=float)
        for model_name, model in models.items():
            scores = cross_validate(model, x_np, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
            pred = cross_val_predict(model, x_np, y, cv=cv, n_jobs=-1)
            metrics_rows.append(
                {
                    "model": model_name,
                    "cv_folds": 5,
                    "accuracy_mean": float(scores["test_accuracy"].mean()),
                    "accuracy_std": float(scores["test_accuracy"].std(ddof=1)),
                    "macro_f1_mean": float(scores["test_f1_macro"].mean()),
                    "macro_f1_std": float(scores["test_f1_macro"].std(ddof=1)),
                    "overall_macro_f1_from_oof": float(f1_score(y, pred, average="macro")),
                }
            )
            cm = confusion_matrix(y, pred, labels=np.arange(len(label_encoder.classes_)))
            for i, actual in enumerate(label_encoder.classes_):
                for j, predicted in enumerate(label_encoder.classes_):
                    confusion_rows.append(
                        {
                            "model": model_name,
                            "actual_family": actual,
                            "predicted_family": predicted,
                            "count": int(cm[i, j]),
                        }
                    )

    metrics = pd.DataFrame(metrics_rows)
    confusion = pd.DataFrame(confusion_rows)
    metrics.to_csv(result_dir / "family_classifier_metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(result_dir / "family_classifier_confusion_matrix.csv", index=False, encoding="utf-8-sig")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(bot_features.to_numpy(dtype=float)) if len(bot_features) else np.empty((0, 0))
    if len(bot_features) >= 2:
        coords = PCA(n_components=2, random_state=seed).fit_transform(x_scaled)
    else:
        coords = np.zeros((len(bot_features), 2))
    pca_df = pd.DataFrame(
        {
            "user_id": feature_df.loc[bot_mask, "user_id"].to_numpy(),
            "family": y_text,
            "pca1": coords[:, 0] if len(coords) else [],
            "pca2": coords[:, 1] if len(coords) else [],
        }
    )
    pca_df.to_csv(result_dir / "graph_feature_pca_by_bot.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 6))
    for fam in FAMILIES:
        sub = pca_df[pca_df["family"] == fam]
        ax.scatter(sub["pca1"], sub["pca2"], s=36, alpha=0.8, label=family_label(fam))
    ax.set_title("PCA of generated bot graph features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(alpha=0.25)
    savefig(fig_dir / "graph_feature_pca_by_family.png")

    if not confusion.empty:
        preferred = "Family_RF" if "Family_RF" in set(confusion["model"]) else confusion["model"].iloc[0]
        view = confusion[confusion["model"] == preferred].pivot_table(index="actual_family", columns="predicted_family", values="count", fill_value=0, aggfunc="sum")
        view = view.reindex(index=FAMILIES, columns=FAMILIES, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(view.to_numpy(dtype=float), cmap="Purples")
        ax.set_xticks(np.arange(len(view.columns)), [family_label(c) for c in view.columns], rotation=30, ha="right")
        ax.set_yticks(np.arange(len(view.index)), [family_label(c) for c in view.index])
        ax.set_title(f"Family classifier confusion matrix ({preferred})")
        for i in range(view.shape[0]):
            for j in range(view.shape[1]):
                ax.text(j, i, str(int(view.iloc[i, j])), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        savefig(fig_dir / "family_classifier_confusion_matrix.png")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "Family classifier skipped", ha="center", va="center")
        ax.axis("off")
        savefig(fig_dir / "family_classifier_confusion_matrix.png")
    return feature_df, metrics, confusion


def human_alignment(feature_df: pd.DataFrame, result_dir: Path, fig_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    human = feature_df[feature_df["family"] == "original_human"]
    bots = feature_df[feature_df["family"].isin(FAMILIES)].copy()
    rows: list[dict[str, Any]] = []
    for bot in bots.itertuples(index=False):
        row = {"user_id": bot.user_id, "family": bot.family}
        for feature in KEY_HUMAN_ALIGNMENT_FEATURES:
            if feature not in feature_df.columns:
                continue
            human_values = pd.to_numeric(human[feature], errors="coerce").dropna().to_numpy(dtype=float)
            value = float(getattr(bot, feature))
            if len(human_values):
                pct = percentileofscore(human_values, value, kind="rank")
            else:
                pct = np.nan
            row[feature] = value
            row[f"{feature}_human_percentile"] = float(pct)
            row[f"{feature}_extreme_1pct"] = bool(pct <= 1.0 or pct >= 99.0) if not math.isnan(pct) else False
        rows.append(row)
    percentiles = pd.DataFrame(rows)
    percentiles.to_csv(result_dir / "human_percentiles_by_bot.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, Any]] = []
    for family, fam_df in percentiles.groupby("family"):
        row: dict[str, Any] = {"family": family, "bot_count": int(len(fam_df))}
        extreme_cols = [c for c in fam_df.columns if c.endswith("_extreme_1pct")]
        row["any_feature_extreme_share"] = float(fam_df[extreme_cols].any(axis=1).mean()) if extreme_cols else np.nan
        for feature in KEY_HUMAN_ALIGNMENT_FEATURES:
            pct_col = f"{feature}_human_percentile"
            ext_col = f"{feature}_extreme_1pct"
            if pct_col in fam_df.columns:
                row[f"{feature}_median_percentile"] = float(fam_df[pct_col].median())
                row[f"{feature}_mean_percentile"] = float(fam_df[pct_col].mean())
            if ext_col in fam_df.columns:
                row[f"{feature}_extreme_share"] = float(fam_df[ext_col].mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(result_dir / "human_alignment_by_family.csv", index=False, encoding="utf-8-sig")

    plot_rows = []
    for family, fam_df in percentiles.groupby("family"):
        for feature in KEY_HUMAN_ALIGNMENT_FEATURES:
            pct_col = f"{feature}_human_percentile"
            if pct_col in fam_df.columns:
                plot_rows.append({"family": family_label(family), "feature": feature, "median_percentile": fam_df[pct_col].median()})
    plot_df = pd.DataFrame(plot_rows)
    pivot = plot_df.pivot_table(index="feature", columns="family", values="median_percentile", aggfunc="mean").reindex(KEY_HUMAN_ALIGNMENT_FEATURES)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlBu_r", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("Median human percentile by generated family")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.0f}" if not pd.isna(val) else "NA", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Human percentile")
    savefig(fig_dir / "human_percentile_by_family.png")
    return percentiles, summary


def write_summary(result_dir: Path, data: dict[str, Any], outputs: dict[str, Any]) -> None:
    summary = {
        "run_dir": str(data["run_dir"]),
        "result_dir": str(result_dir),
        "dataset_integrity_passed": data["integrity"].get("passed"),
        "generated_family_counts": data["users"][data["users"]["family"].isin(FAMILIES)].groupby("family").size().to_dict(),
        "output_files": outputs,
    }
    (result_dir / "family_behavior_mixing_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    result_dir = ensure_dir(Path(args.result_dir).resolve())
    fig_dir = ensure_dir(result_dir / "figures")
    data = load_inputs(run_dir)
    data["run_dir"] = run_dir

    behavior, behavior_summary = behavior_by_bot(data, result_dir, fig_dir)
    mixing, perm = family_mixing(data, result_dir, fig_dir, args.seed, args.permutations)
    community, community_summary = community_audit(data, result_dir, fig_dir)
    feature_df, classifier_metrics, classifier_cm = feature_separability(data, result_dir, fig_dir, args.seed)
    percentiles, alignment = human_alignment(feature_df, result_dir, fig_dir)

    outputs = {
        "behavior_by_bot_rows": int(len(behavior)),
        "behavior_by_family_rows": int(len(behavior_summary)),
        "family_mixing_rows": int(len(mixing)),
        "permutation_rows": int(len(perm)),
        "community_rows": int(len(community)),
        "community_summary_rows": int(len(community_summary)),
        "graph_feature_rows": int(len(feature_df)),
        "classifier_metric_rows": int(len(classifier_metrics)),
        "classifier_confusion_rows": int(len(classifier_cm)),
        "human_percentile_rows": int(len(percentiles)),
        "human_alignment_rows": int(len(alignment)),
    }
    write_summary(result_dir, data, outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--permutations", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
