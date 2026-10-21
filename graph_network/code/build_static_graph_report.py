#!/usr/bin/env python3
"""Build the Chinese static graph experiment report from result CSVs."""

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
from pathlib import Path

import pandas as pd


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def metric_table(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"### {title}\n\n未找到对应结果文件。\n"
    cols = [c for c in ["split_name", "setting", "model", "backend", "eval_group", "family", "size", "bot_count", "bot_recall", "bot_precision", "bot_f1", "macro_f1", "bot_mean_probability", "status"] if c in df.columns]
    view = df[(df.get("eval_group", "") == "test") & (df.get("family", "") == "ALL")] if "eval_group" in df.columns and "family" in df.columns else df
    if view.empty:
        view = df
    return f"### {title}\n\n" + view[cols].round(4).to_markdown(index=False) + "\n"


def family_matrix(df: pd.DataFrame, title: str) -> str:
    if df.empty or "family" not in df.columns or "bot_recall" not in df.columns:
        return f"### {title}\n\n暂无 family 分组结果。\n"
    view = df[(df.get("eval_group", "") == "test") & (df["family"] != "ALL")].copy()
    if view.empty:
        return f"### {title}\n\n暂无 test family 分组结果。\n"
    cols = [c for c in ["split_name", "model", "backend", "family", "bot_count", "bot_recall", "bot_mean_probability"] if c in view.columns]
    return f"### {title}\n\n" + view[cols].round(4).to_markdown(index=False) + "\n"


def shortcut_section(audit: pd.DataFrame) -> str:
    if audit.empty:
        return "## Shortcut 风险审计\n\n未找到 `shortcut_audit.csv`。\n"
    view = audit[(audit.get("eval_group", "") == "test") & (audit.get("family", "") == "ALL")].copy()
    if view.empty:
        view = audit.head(20).copy()
    cols = [
        c
        for c in [
            "split_name",
            "setting",
            "family",
            "size",
            "bot_count",
            "zero_inbound_share",
            "over_directional_share",
            "low_reciprocity_share",
            "isolated_share",
            "bot_only_component_share",
            "forbidden_metadata_columns_present",
            "forbidden_metadata_used_as_features",
        ]
        if c in view.columns
    ]
    text = "## Shortcut 风险审计\n\n"
    text += "训练特征仅由 user-user 图结构派生；`family`、`user_id/user_name`、`provider/model` 只保留在分组、预测输出和审计字段中，不进入特征矩阵。\n\n"
    text += view[cols].round(4).to_markdown(index=False) + "\n"
    return text


def top_features_section(importance: pd.DataFrame) -> str:
    if importance.empty:
        return "## 可解释特征\n\n未找到 `feature_importance.csv`。\n"
    view = importance.sort_values(["split_name", "model", "rank"]).groupby(["split_name", "model"]).head(10)
    cols = [c for c in ["split_name", "model", "rank", "feature", "importance", "importance_kind"] if c in view.columns]
    return "## 可解释特征\n\n" + view[cols].round(5).to_markdown(index=False) + "\n"


def split_summary(run_dir: Path) -> str:
    split_dir = run_dir / "splits"
    paths = sorted(
        path for path in split_dir.glob("*.json") if path.name != "split_summary.json"
    ) if split_dir.exists() else []
    if not paths:
        return "未发现 `splits/*.json`；脚本会在缺失时生成仅用于 smoke 的自动分层 split。"
    rows = []
    for path in paths:
        data = read_json_if_exists(path)
        rows.append(
            {
                "split_name": data.get("split_name", path.stem),
                "setting": data.get("setting", ""),
                "train": len(data.get("train_node_ids", [])),
                "val": len(data.get("val_node_ids", [])),
                "test": len(data.get("test_node_ids", [])),
                "unlabeled_visible": len(data.get("unlabeled_visible_node_ids", [])),
                "holdout_families": ";".join(map(str, data.get("holdout_families", []))),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json_if_exists(run_dir / "dataset_manifest.json")
    graphstats = read_csv_if_exists(result_dir / "graphstats_metrics.csv")
    rgcn = read_csv_if_exists(result_dir / "rgcn_metrics.csv")
    sage = read_csv_if_exists(result_dir / "graphsage_metrics.csv")
    audit = read_csv_if_exists(result_dir / "shortcut_audit.csv")
    importance = read_csv_if_exists(result_dir / "feature_importance.csv")
    summary = read_json_if_exists(result_dir / "graphstats_run_summary.json")

    lines = [
        "# graph_network检测实验报告",
        "",
        "## 数据集概况",
        "",
        f"- run_name: `{manifest.get('run_name', summary.get('manifest_run_name', 'unknown'))}`",
        f"- schema_version: `{manifest.get('schema_version', 'unknown')}`",
        f"- environment: `{manifest.get('environment', 'unknown')}`",
        f"- source_dataset: `{manifest.get('source_dataset', 'unknown')}`",
        f"- 节点数: `{summary.get('node_count', 'unknown')}`",
        f"- user-user 边数: `{summary.get('edge_count', 'unknown')}`",
        f"- GraphStats 特征数: `{summary.get('feature_count', 'unknown')}`",
        "",
        "三类生成 family 设计用于区分不同社交拟人机制：G1 偏总体行为分布校准，G2 偏 human anchor 轨迹拟合，G3 偏上下文浏览、通知、回访等交互反馈。报告中的 `family` 仅用于分组解释和泛化矩阵，不作为训练特征。",
        "",
        "## Split 设计",
        "",
        split_summary(run_dir),
        "",
        "协议要求关注 zero-shot、LOFO、label scarcity、time-forward 与 incremental curriculum。当前脚本按 `splits/*.json` 逐个执行；若 smoke 数据只提供少量 split，本报告只展示已运行 split。",
        "",
        "## GraphStats 结果",
        "",
        metric_table(graphstats, "GraphStats 汇总"),
        family_matrix(graphstats, "GraphStats family recall matrix"),
        "",
        "## GNN 结果",
        "",
        metric_table(rgcn, "R-GCN transductive 汇总"),
        metric_table(sage, "GraphSAGE inductive 汇总"),
        family_matrix(pd.concat([rgcn, sage], ignore_index=True) if not rgcn.empty or not sage.empty else pd.DataFrame(), "GNN family recall matrix"),
        "",
        "## Text-only vs Graph-only vs Fusion",
        "",
        "本轮 Worker E 只实现 Graph-only 检测骨架；Text-only 与 Fusion 结果文件缺失时不做数值比较。后续可在相同 split 下接入 TF-IDF/BGE 与文本图融合结果，再判断各 family 是文本更容易检出、图结构更容易检出，还是 Fusion 继承了单一路径 shortcut。",
        "",
        shortcut_section(audit),
        "",
        top_features_section(importance),
        "",
        "## 结论占位",
        "",
        "本报告骨架已经能汇总 smoke dataset 的 GraphStats、R-GCN 与 GraphSAGE 输出。正式实验应补齐全量 family matrix、label scarcity 曲线、time-forward 切分和跨模态比较，并在 shortcut 审计中重点检查零入边、高方向性、低互惠和 bot-only component。", 
        "",
    ]
    (result_dir / "report_cn.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
