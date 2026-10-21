#!/usr/bin/env python3
"""Build graph-only generalization report and figures."""

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
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_LABELS = {
    "G1_population_calibrated": "G1 population",
    "G2_anchor_behavior": "G2 anchor",
    "G3_contextual_interactive": "G3 interactive",
    "G4_slot_conditioned": "G4 slot",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_fig_dir(result_dir: Path) -> Path:
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def model_order(name: str) -> int:
    order = {
        "GraphStats_LR": 0,
        "GraphStats_RF": 1,
        "GraphStats_EBM": 2,
        "GraphSAGE_Inductive": 3,
        "RGCN_Transductive": 4,
    }
    return order.get(str(name), 99)


def model_display(name: str) -> str:
    return {
        "GraphStats_LR": "GraphStats LR",
        "GraphStats_RF": "GraphStats RF",
        "GraphStats_EBM": "GraphStats EBM",
        "GraphSAGE_Inductive": "GraphSAGE",
        "RGCN_Transductive": "RGCN",
    }.get(str(name), str(name))


def family_display(name: str) -> str:
    return FAMILY_LABELS.get(str(name), str(name))


def all_metrics(result_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ["graphstats_metrics.csv", "graphsage_metrics.csv", "rgcn_metrics.csv"]:
        df = read_csv(result_dir / name)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in ["bot_recall", "bot_precision", "bot_f1", "macro_f1", "bot_mean_probability"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def test_all(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["eval_group"] == "test") & (df["family"] == "ALL")].copy()


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_zero_shot(result_dir: Path, df: pd.DataFrame, fig_dir: Path) -> Path | None:
    view = df[(df["split_name"] == "original_only_zero_shot") & (df["eval_group"] == "test") & (df["family"] == "ALL")].copy()
    if view.empty:
        return None
    view = view.sort_values("model", key=lambda s: s.map(model_order))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([model_display(x) for x in view["model"]], view["bot_recall"], color="#4C78A8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Bot recall")
    ax.set_title("Zero-shot recall on generated G-families")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = fig_dir / "zero_shot_recall.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_lofo(result_dir: Path, df: pd.DataFrame, fig_dir: Path) -> Path | None:
    view = df[
        df["split_name"].astype(str).str.startswith("lofo_", na=False)
        & (df["eval_group"] == "test")
        & (df["family"] != "ALL")
    ].copy()
    if view.empty:
        return None
    view["model_label"] = view["model"].map(model_display)
    view["family_label"] = view["family"].map(family_display)
    pivot = view.pivot_table(index="model_label", columns="family_label", values="bot_recall", aggfunc="mean")
    ordered_models = [model_display(m) for m in sorted(view["model"].unique(), key=model_order) if model_display(m) in pivot.index]
    ordered_families = [family_display(f) for f in FAMILY_LABELS if family_display(f) in pivot.columns]
    ordered_families.extend([c for c in pivot.columns if c not in ordered_families])
    pivot = pivot.reindex(index=ordered_models, columns=ordered_families)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title("LOFO recall matrix")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="Bot recall")
    fig.tight_layout()
    out = fig_dir / "lofo_recall_matrix.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def extract_ratio(split_name: str) -> float | None:
    match = re.search(r"label_scarcity_(\d+)pct", str(split_name))
    if not match:
        return None
    return int(match.group(1)) / 1000.0


def plot_label_scarcity(df: pd.DataFrame, fig_dir: Path) -> Path | None:
    view = test_all(df[df["split_name"].str.startswith("label_scarcity_", na=False)]).copy()
    if view.empty:
        return None
    view["ratio"] = view["split_name"].map(extract_ratio)
    view = view.dropna(subset=["ratio", "bot_recall"])
    if view.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, group in view.groupby("model"):
        group = group.sort_values("ratio")
        ax.plot(group["ratio"], group["bot_recall"], marker="o", label=model_display(model))
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Generated bot label fraction in training")
    ax.set_ylabel("Held-out generated bot recall")
    ax.set_title("Label scarcity recovery")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "label_scarcity_curve.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_curriculum(curr: pd.DataFrame, fig_dir: Path) -> Path | None:
    view = curr[(curr["eval_group"] == "test") & (curr["family"] == "ALL")].copy()
    if view.empty:
        return None
    view["order_short"] = view["curriculum_order"].astype(str).map(
        lambda s: ">".join(part.split("_", 1)[0] for part in s.split(">"))
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for (model, order), group in view.groupby(["model", "order_short"]):
        if model != "GraphStats_LR":
            continue
        group = group.sort_values("curriculum_step")
        labels = [str(x).split("_", 1)[0] for x in group["add_family_after_evaluation"]]
        ax.plot(group["curriculum_step"], group["bot_recall"], marker="o", label=order)
        for x, y, label in zip(group["curriculum_step"], group["bot_recall"], labels):
            ax.text(x, min(1.02, y + 0.03), label, ha="center", fontsize=8)
    max_step = int(view["curriculum_step"].max()) if not view.empty else 3
    ax.set_xticks(list(range(1, max_step + 1)))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Curriculum step")
    ax.set_ylabel("Blind recall before adding current family")
    ax.set_title("GraphStats LR incremental curriculum")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Order", fontsize=8)
    fig.tight_layout()
    out = fig_dir / "incremental_curriculum_recall.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_shortcut(audit: pd.DataFrame, fig_dir: Path) -> Path | None:
    if audit.empty:
        return None
    metrics = ["zero_inbound_share", "over_directional_share", "low_reciprocity_share", "isolated_share", "bot_only_component_share"]
    view = audit[
        (audit["split_name"] == "original_only_zero_shot")
        & (audit["eval_group"] == "test")
        & (audit["family"] != "ALL")
    ].copy()
    if view.empty:
        return None
    view["family_label"] = view["family"].map(family_display)
    plot_df = view.set_index("family_label")[metrics].fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(plot_df))
    for metric in metrics:
        values = plot_df[metric].to_numpy(dtype=float)
        ax.bar(plot_df.index, values, bottom=bottom, label=metric.replace("_share", ""))
        bottom += values
    ax.set_ylim(0, max(0.1, bottom.max() * 1.2))
    ax.set_ylabel("Share")
    ax.set_title("Shortcut audit shares by family")
    ax.legend(fontsize=7, ncol=2)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out = fig_dir / "shortcut_audit.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "无数据。"
    view = df[cols].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].round(4)
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ensure_fig_dir(result_dir)

    metrics = all_metrics(result_dir)
    graphstats = read_csv(result_dir / "graphstats_metrics.csv")
    curr = read_csv(result_dir / "graphstats_curriculum_step_metrics.csv")
    audit = read_csv(result_dir / "shortcut_audit.csv")
    importance = read_csv(result_dir / "feature_importance.csv")
    manifest = read_json(run_dir / "dataset_manifest.json")
    export_summary = read_json(run_dir / "graph" / "export_summary.json")
    run_users = read_csv(run_dir / "users.csv")
    run_posts = read_csv(run_dir / "posts.csv")
    run_comments = read_csv(run_dir / "comments.csv")
    generated_user_count = int(run_users.get("source", pd.Series(dtype=str)).astype(str).eq("generated_bot").sum()) if not run_users.empty else "unknown"
    generated_post_count = int(run_posts.get("is_generated", pd.Series(dtype=str)).astype(str).str.lower().eq("true").sum()) if not run_posts.empty else "unknown"
    if not run_comments.empty:
        generated_comment_mask = run_comments.get("is_generated", pd.Series(dtype=str)).astype(str).str.lower().eq("true")
        replay_mask = run_comments.get("is_replay_human", pd.Series(dtype=str)).astype(str).str.lower().eq("true")
        generated_comment_count = int((generated_comment_mask & ~replay_mask).sum())
    else:
        generated_comment_count = "unknown"

    zero = metrics[
        (metrics["split_name"] == "original_only_zero_shot")
        & (metrics["eval_group"] == "test")
        & (metrics["family"] == "ALL")
    ].copy().sort_values("model", key=lambda s: s.map(model_order))
    lofo = metrics[
        metrics["split_name"].astype(str).str.startswith("lofo_", na=False)
        & (metrics["eval_group"] == "test")
        & (metrics["family"] != "ALL")
    ].copy().sort_values(["model", "family"], key=lambda s: s.map(model_order) if s.name == "model" else s)
    scarcity = test_all(metrics[metrics["split_name"].str.startswith("label_scarcity_", na=False)]).copy()
    scarcity["label_fraction"] = scarcity["split_name"].map(extract_ratio)
    scarcity = scarcity.dropna(subset=["label_fraction"]).sort_values(["model", "label_fraction"], key=lambda s: s.map(model_order) if s.name == "model" else s)
    curr_view = curr[(curr["eval_group"] == "test") & (curr["family"] == "ALL")].copy()
    if not curr_view.empty:
        curr_view = curr_view.sort_values(["model", "curriculum_order", "curriculum_step"], key=lambda s: s.map(model_order) if s.name == "model" else s)
    lr_features = importance[importance["model"] == "GraphStats_LR"].sort_values(["split_name", "rank"]).groupby("split_name").head(5)

    write_csv(result_dir / "zero_shot_summary.csv", zero)
    write_csv(result_dir / "lofo_family_recall_matrix_long.csv", lofo)
    write_csv(result_dir / "label_scarcity_summary.csv", scarcity)
    write_csv(result_dir / "curriculum_step_summary.csv", curr_view)
    write_csv(result_dir / "top_lr_features_by_split.csv", lr_features)

    figures = {
        "zero_shot": plot_zero_shot(result_dir, metrics, fig_dir),
        "lofo": plot_lofo(result_dir, metrics, fig_dir),
        "label_scarcity": plot_label_scarcity(metrics, fig_dir),
        "curriculum": plot_curriculum(curr, fig_dir),
        "shortcut": plot_shortcut(audit, fig_dir),
    }

    zero_recall_lr = float(zero.loc[zero["model"] == "GraphStats_LR", "bot_recall"].iloc[0]) if not zero[zero["model"] == "GraphStats_LR"].empty else float("nan")
    zero_gnn = zero[zero["model"].isin(["GraphSAGE_Inductive", "RGCN_Transductive"])]
    zero_gnn_text = ", ".join(
        f"{model_display(row.model)}={row.bot_recall:.2f}"
        for row in zero_gnn.itertuples()
    ) or "GraphSAGE/RGCN 无结果"
    lofo_lr = lofo[lofo["model"] == "GraphStats_LR"]
    lofo_lr_mean = float(lofo_lr["bot_recall"].mean()) if not lofo_lr.empty else float("nan")
    curr_lr = curr_view[curr_view["model"] == "GraphStats_LR"]
    step_means = curr_lr.groupby("curriculum_step")["bot_recall"].mean().to_dict() if not curr_lr.empty else {}
    scarcity_lr = scarcity[scarcity["model"] == "GraphStats_LR"].copy()
    scarcity_points = []
    if not scarcity_lr.empty:
        for row in scarcity_lr.sort_values("label_fraction").itertuples():
            scarcity_points.append(f"{float(row.label_fraction):.1%}={float(row.bot_recall):.2f}")
    gnn_backends = sorted(
        str(x)
        for x in metrics[metrics["model"].isin(["GraphSAGE_Inductive", "RGCN_Transductive"])]["backend"].dropna().unique()
    )
    if any(str(x).startswith("pyg_") for x in gnn_backends):
        gnn_note = "当前已安装 `torch_geometric`，GraphSAGE 使用 PyG `SAGEConv`，RGCN 使用 PyG `RGCNConv`。GraphSAGE 是 inductive 参考口径；RGCN 使用完整可见拓扑，属于 observed-transductive 参考结果。GraphStats_LR 仍是本报告主结论，因为它最可解释、最便于定位图结构 shortcut。"
    else:
        gnn_note = "当前 GraphSAGE 与 RGCN 输出中的 backend 为 fallback。这不是文本 fallback，也不是 API fallback，而是项目脚本内置的 PyTorch/sklearn 简化图模型实现。RGCN 使用完整可见拓扑，属于 observed-transductive 参考结果；GraphStats_LR 仍是本报告主结论。"

    lines = [
        "# 图数据泛化性研究报告",
        "",
        "## 研究口径",
        "",
        "本报告只使用图结构特征，不使用评论文本。输入数据是 partial run，因此结论应表述为 `partial_1202actions_stopped_live` 上的图结构泛化实验。",
        "",
        f"- 数据目录：`{run_dir}`",
        f"- run_name：`{manifest.get('run_name', 'unknown')}`",
        f"- generated bots：`{generated_user_count}`",
        f"- generated posts/comments：`{generated_post_count}` / `{generated_comment_count}`",
        f"- user nodes / user-user edges：`{export_summary.get('counts', {}).get('homogeneous_user_nodes', 'unknown')}` / `{export_summary.get('counts', {}).get('homogeneous_user_edges', 'unknown')}`",
        f"- GraphStats feature count：`{read_json(result_dir / 'graphstats_run_summary.json').get('feature_count', 'unknown')}`",
        "",
        "## 核心结论",
        "",
        f"1. **原始 BotSim 图检测器不能 zero-shot 泛化到新 G-family。** GraphStats_LR 在 `original_only_zero_shot` 上的 bot recall 为 `{zero_recall_lr:.2f}`；GNN 参考结果为 {zero_gnn_text}。",
        f"2. **加入其它新 family 后，留一 family 的检出率会发生恢复。** GraphStats_LR 的 LOFO 平均 recall 为 `{lofo_lr_mean:.2f}`，各 family 分别约 `{', '.join(f'{family_display(r.family)}={r.bot_recall:.2f}' for r in lofo_lr.itertuples())}`。",
        f"3. **逐步课程学习显示恢复曲线。** GraphStats_LR 分步骤平均 recall：`{', '.join(f'step {k}={v:.2f}' for k, v in sorted(step_means.items()))}`。",
        "4. **shortcut 需要逐 family 判断。** 重点看 zero-inbound、isolated、bot-only component、方向不平衡、互惠和一级/二级回复边类型占比。",
        "",
        "## Zero-shot：原始图检测器能否检出新 family",
        "",
        markdown_table(zero, ["model", "backend", "bot_count", "bot_recall", "bot_precision", "bot_f1", "bot_mean_probability"]),
        "",
        f"![zero-shot recall]({figures['zero_shot']})" if figures["zero_shot"] else "",
        "",
        "解释：这里训练集只有原始 human + 原始 bot，测试集是所有 generated G-family。若 recall 接近 0，说明原始 BotSim 的图结构判别边界不能直接迁移到新构造的社交模拟 family。",
        "",
        "## LOFO：留出一个 family 的跨家族泛化",
        "",
        markdown_table(lofo, ["split_name", "model", "backend", "family", "bot_count", "bot_recall", "bot_mean_probability"], max_rows=60),
        "",
        f"![LOFO recall matrix]({figures['lofo']})" if figures["lofo"] else "",
        "",
        "解释：只要训练集中包含其它新 family，检测器对 unseen family 的 recall 变化就反映跨家族迁移能力；如果迅速升高，说明不同 family 间存在共享图结构痕迹。",
        "",
        "## Incremental curriculum：逐步加入 family 的恢复曲线",
        "",
        markdown_table(curr_view[curr_view["model"] == "GraphStats_LR"], ["curriculum_order", "curriculum_step", "add_family_after_evaluation", "bot_count", "bot_recall", "bot_mean_probability"], max_rows=30),
        "",
        f"![incremental curriculum recall]({figures['curriculum']})" if figures["curriculum"] else "",
        "",
        "解释：每一步都是先测试当前 family，再把它加入训练集。曲线越快上升，说明少量 family 经验越容易迁移到后续 unseen family。",
        "",
        "## Label scarcity：少量 generated label 是否足够恢复",
        "",
        markdown_table(scarcity[scarcity["model"].isin(["GraphStats_LR", "GraphStats_EBM", "GraphSAGE_Inductive", "RGCN_Transductive"])], ["split_name", "model", "backend", "bot_count", "bot_recall", "bot_mean_probability"], max_rows=60),
        "",
        f"![label scarcity curve]({figures['label_scarcity']})" if figures["label_scarcity"] else "",
        "",
        f"解释：GraphStats_LR 的 label scarcity recall 为：`{', '.join(scarcity_points)}`。这表示只要少量 generated 标签进入训练，图结构检测就会开始恢复，但恢复速度取决于被采样到的 family。",
        "",
        "## Shortcut attribution：模型抓到了什么",
        "",
        "### Shortcut 审计",
        "",
        markdown_table(
            audit[
                (audit["split_name"] == "original_only_zero_shot")
                & (audit["eval_group"] == "test")
                & (audit["family"] != "ALL")
            ],
            ["family", "size", "bot_count", "zero_inbound_share", "over_directional_share", "low_reciprocity_share", "isolated_share", "bot_only_component_share"],
        ),
        "",
        f"![shortcut audit]({figures['shortcut']})" if figures["shortcut"] else "",
        "",
        "### GraphStats_LR top features",
        "",
        markdown_table(lr_features, ["split_name", "rank", "feature", "importance"], max_rows=50),
        "",
        "解释：高权重特征集中在 `user_comments_on_user_post`、`user_replies_to_user_comment`、`user_receives_reply`、`reciprocal_share`、`direction_balance` 等互动结构上。它们不是文本特征，也不是 family/provider/model 这类元数据。",
        "",
        "## GNN 口径说明",
        "",
        gnn_note,
        "",
        "## 产物",
        "",
        f"- `zero_shot_summary.csv`",
        f"- `lofo_family_recall_matrix_long.csv`",
        f"- `label_scarcity_summary.csv`",
        f"- `curriculum_step_summary.csv`",
        f"- `top_lr_features_by_split.csv`",
        f"- `figures/*.png`",
        "",
    ]
    report_path = result_dir / "GRAPH_GENERALIZATION_REPORT_CN.md"
    report_path.write_text("\n".join(str(x) for x in lines if x is not None) + "\n", encoding="utf-8")

    machine = {
        "zero_shot_graphstats_lr_recall": zero_recall_lr,
        "lofo_graphstats_lr_mean_recall": lofo_lr_mean,
        "curriculum_graphstats_lr_step_mean_recall": {str(k): float(v) for k, v in step_means.items()},
        "figures": {k: str(v) if v else "" for k, v in figures.items()},
        "report": str(report_path),
    }
    (result_dir / "graph_generalization_summary.json").write_text(json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(machine, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
