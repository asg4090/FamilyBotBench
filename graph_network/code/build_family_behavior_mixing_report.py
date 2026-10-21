#!/usr/bin/env python3
"""Build Chinese report for family behavior and mixing audit."""

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
from typing import Any

import numpy as np
import pandas as pd


FAMILIES = [
    "G1_population_calibrated",
    "G2_anchor_behavior",
    "G3_contextual_interactive",
    "G4_slot_conditioned",
]
FAMILY_LABELS = {
    "original_human": "Human",
    "G1_population_calibrated": "G1 population",
    "G2_anchor_behavior": "G2 anchor",
    "G3_contextual_interactive": "G3 interactive",
    "G4_slot_conditioned": "G4 slot",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int = 20) -> str:
    if df.empty:
        return "_无数据_"
    view = df.copy()
    if cols:
        view = view[[c for c in cols if c in view.columns]]
    view = view.head(max_rows)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].round(4)
    return view.to_markdown(index=False)


def fam(value: Any) -> str:
    return FAMILY_LABELS.get(str(value), str(value))


def image_line(path: Path, alt: str) -> str:
    return f"![{alt}]({path})" if path.exists() else f"_缺少图：{path.name}_"


def wide_behavior_summary(behavior: pd.DataFrame) -> pd.DataFrame:
    if behavior.empty:
        return behavior
    cols = [
        "family",
        "post_count_mean",
        "comment1_count_mean",
        "comment2_count_mean",
        "revisit_count_mean",
        "skip_count_mean",
        "active_window_count_mean",
        "bot_to_bot_share_out_mean",
        "bot_to_human_share_out_mean",
        "received_reply_count_mean",
    ]
    return behavior[[c for c in cols if c in behavior.columns]].copy()


def action_difference_judgment(behavior: pd.DataFrame) -> str:
    if behavior.empty:
        return "行为分布数据为空，无法判断。"
    action_cols = [c for c in ["post_count_mean", "comment1_count_mean", "comment2_count_mean", "revisit_count_mean", "skip_count_mean"] if c in behavior.columns]
    if not action_cols:
        return "行为分布缺少 action 均值列，无法判断。"
    ranges = {}
    for col in action_cols:
        values = pd.to_numeric(behavior[col], errors="coerce")
        ranges[col] = float(values.max() - values.min())
    strongest = max(ranges, key=ranges.get)
    if ranges[strongest] >= 0.8:
        return f"三个 family 的行为分布存在可见差异，最大差异来自 `{strongest.replace('_mean', '')}`，family 均值跨度约 `{ranges[strongest]:.2f}`。"
    return f"三个 family 的行为分布差异较弱，最大 action 均值跨度只有 `{ranges[strongest]:.2f}`，说明 family 机制可能没有充分拉开。"


def mixing_judgment(perm: pd.DataFrame) -> str:
    if perm.empty:
        return "缺少 permutation test，无法判断同 family 抱团。"
    row = perm.iloc[0]
    observed = float(row.get("observed", np.nan))
    random_mean = float(row.get("random_mean", np.nan))
    z_score = float(row.get("z_score", np.nan))
    empirical_p = float(row.get("empirical_p", np.nan))
    if np.isnan(observed):
        return "generated bot-to-bot 边为空或不足，无法判断同 family 抱团。"
    if observed > random_mean and empirical_p <= 0.05:
        return f"存在显著同 family 抱团：真实 same-family share=`{observed:.3f}`，随机基线均值=`{random_mean:.3f}`，z=`{z_score:.2f}`，p=`{empirical_p:.4f}`。"
    if observed > random_mean:
        return f"同 family 互动比例高于随机均值，但未达到强显著：真实=`{observed:.3f}`，随机均值=`{random_mean:.3f}`，p=`{empirical_p:.4f}`。"
    return f"未发现同 family 抱团：真实 same-family share=`{observed:.3f}`，随机基线均值=`{random_mean:.3f}`。"


def directed_environment_judgment(mixing: pd.DataFrame) -> str:
    if mixing.empty:
        return "缺少 mixing matrix，无法判断 human 与 generated bot 的互动方向。"
    human_to_generated = mixing[(mixing["src_family"] == "original_human") & (mixing["dst_family"].isin(FAMILIES))]["edge_weight"].sum()
    generated_to_human = mixing[(mixing["src_family"].isin(FAMILIES)) & (mixing["dst_family"] == "original_human")]["edge_weight"].sum()
    bot_to_bot = mixing[(mixing["src_family"].isin(FAMILIES)) & (mixing["dst_family"].isin(FAMILIES))]["edge_weight"].sum()
    if human_to_generated == 0 and (generated_to_human > 0 or bot_to_bot > 0):
        return f"存在明确环境方向性 shortcut：generated→human 边=`{generated_to_human:.0f}`，generated↔generated 边=`{bot_to_bot:.0f}`，但 human→generated 边=`0`。这说明当前数据还没有人类回到新 bot 内容下互动。"
    return f"human/generated 方向性边统计：human→generated=`{human_to_generated:.0f}`，generated→human=`{generated_to_human:.0f}`，generated↔generated=`{bot_to_bot:.0f}`。"


def community_judgment(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "缺少社区检测汇总，无法判断 bot-only 社区。"
    row = summary.iloc[0]
    bot_only_count = int(row.get("bot_only_community_count", 0))
    bot_only_share = float(row.get("bot_only_node_share", np.nan))
    bot_heavy_count = int(row.get("bot_heavy_community_count_80pct", 0))
    max_bot_share = float(row.get("max_bot_share_in_community", np.nan))
    max_bot_size = int(row.get("max_bot_share_community_size", 0))
    max_bot_count = int(row.get("max_bot_share_community_bot_count", 0))
    modularity = float(row.get("family_modularity", np.nan))
    nmi_gen = float(row.get("nmi_generated_only", np.nan))
    if bot_heavy_count > 0 and not np.isnan(max_bot_share):
        return f"没有严格 bot-only 社区，但存在 bot-heavy 社区：最高 bot_share=`{max_bot_share:.3f}`，该社区 size=`{max_bot_size}`、bot_count=`{max_bot_count}`；family modularity=`{modularity:.3f}`，generated-only NMI=`{nmi_gen:.3f}`。这表示不一定是某个 family 单独抱团，而可能是 generated bot 整体容易集中。"
    if bot_only_count > 0 or (not np.isnan(bot_only_share) and bot_only_share > 0.01):
        return f"存在 bot-only 社区风险：bot-only community=`{bot_only_count}`，节点占比=`{bot_only_share:.3f}`；family modularity=`{modularity:.3f}`，generated-only NMI=`{nmi_gen:.3f}`。"
    return f"未看到明显 bot-only 社区：bot-only community=`{bot_only_count}`；family modularity=`{modularity:.3f}`，generated-only NMI=`{nmi_gen:.3f}`。"


def separability_judgment(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "family classifier 未产生结果，无法判断图特征空间可分性。"
    best = metrics.sort_values("macro_f1_mean", ascending=False).iloc[0]
    macro_f1 = float(best.get("macro_f1_mean", np.nan))
    acc = float(best.get("accuracy_mean", np.nan))
    model = str(best.get("model", "unknown"))
    if macro_f1 >= 0.80:
        return f"各 generated family 在图特征空间中高度可分：最佳模型 `{model}`，CV accuracy=`{acc:.3f}`，macro-F1=`{macro_f1:.3f}`。这说明 family 行为差异明显，但也可能成为检测器可利用的 family signature。"
    if macro_f1 >= 0.55:
        return f"各 generated family 在图特征空间中中等可分：最佳模型 `{model}`，CV accuracy=`{acc:.3f}`，macro-F1=`{macro_f1:.3f}`。"
    return f"各 generated family 在图特征空间中不易区分：最佳模型 `{model}`，CV accuracy=`{acc:.3f}`，macro-F1=`{macro_f1:.3f}`。"


def human_alignment_judgment(alignment: pd.DataFrame) -> str:
    if alignment.empty:
        return "缺少 human alignment 结果，无法判断是否偏离人类分布。"
    if "any_feature_extreme_share" not in alignment.columns:
        return "human alignment 缺少极端分位比例，无法判断。"
    view = alignment[["family", "bot_count", "any_feature_extreme_share"]].copy()
    max_row = view.sort_values("any_feature_extreme_share", ascending=False).iloc[0]
    max_share = float(max_row["any_feature_extreme_share"])
    if max_share >= 0.8:
        return f"存在明显 human 分布偏离：`{fam(max_row.family)}` 中 `{max_share:.2%}` 的 bot 至少一个关键图特征落在人类 1%/99% 极端区间。"
    if max_share >= 0.4:
        return f"存在中等 human 分布偏离：最高为 `{fam(max_row.family)}`，极端特征占比 `{max_share:.2%}`。"
    return f"整体 human alignment 尚可：最高极端特征占比为 `{fam(max_row.family)}` 的 `{max_share:.2%}`。"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    fig_dir = result_dir / "figures"

    run_summary = read_json(result_dir / "family_behavior_mixing_run_summary.json")
    behavior = read_csv(result_dir / "behavior_by_family.csv")
    mixing = read_csv(result_dir / "family_mixing_matrix.csv")
    perm = read_csv(result_dir / "same_family_permutation_test.csv")
    community_comp = read_csv(result_dir / "community_family_composition.csv")
    community_summary = read_csv(result_dir / "community_summary.csv")
    classifier_metrics = read_csv(result_dir / "family_classifier_metrics.csv")
    alignment = read_csv(result_dir / "human_alignment_by_family.csv")

    lines = [
        "# Family Behavior and Mixing Audit",
        "",
        "## 研究口径",
        "",
        "本报告只审计生成数据质量，不重新生成 bot，不做文本检测，也不训练 bot-vs-human 检测器。核心问题是：各 generated family 是否真的形成不同社交行为，以及是否出现 family 抱团或明显偏离 human 图行为分布。",
        "",
        f"- 数据目录：`{run_dir}`",
        f"- 结果目录：`{result_dir}`",
        f"- 数据完整性 passed：`{run_summary.get('dataset_integrity_passed', 'unknown')}`",
        f"- generated family counts：`{run_summary.get('generated_family_counts', {})}`",
        "",
        "## 核心判断",
        "",
        f"1. {action_difference_judgment(behavior)}",
        f"2. {mixing_judgment(perm)}",
        f"3. {directed_environment_judgment(mixing)}",
        f"4. {community_judgment(community_summary)}",
        f"5. {separability_judgment(classifier_metrics)}",
        f"6. {human_alignment_judgment(alignment)}",
        "",
        "## 1. Family 行为分布",
        "",
        table(wide_behavior_summary(behavior), max_rows=10),
        "",
        image_line(fig_dir / "action_distribution_by_family.png", "action distribution by family"),
        "",
        image_line(fig_dir / "temporal_activity_by_family.png", "temporal activity by family"),
        "",
        "解释：这部分看每个 family 是否真的表现为不同社交行动者，而不是只在 prompt/persona 名称上不同。重点关注 post/comment1/comment2/revisit/skip 的比例和活跃窗口数。",
        "",
        "## 2. Family Mixing Matrix 与同类抱团",
        "",
        table(mixing, ["src_label", "dst_label", "edge_weight"], max_rows=20),
        "",
        table(perm, ["scope", "edge_count", "observed", "random_mean", "random_std", "z_score", "empirical_p", "permutations"]),
        "",
        image_line(fig_dir / "family_mixing_matrix.png", "family mixing matrix"),
        "",
        image_line(fig_dir / "same_family_edge_permutation_test.png", "same family permutation test"),
        "",
        "解释：mixing matrix 直接显示谁更常和谁互动；permutation test 用随机 family 标签作为基线。如果 observed same-family share 显著高于随机基线，说明同类 bot 抱团。",
        "",
        "## 3. 社区结构与 bot-only 风险",
        "",
        table(community_summary),
        "",
        table(
            community_comp.sort_values("size", ascending=False),
            ["community_id", "method", "size", "bot_count", "bot_share", "dominant_family", "dominant_family_share", "is_bot_only"],
            max_rows=20,
        ),
        "",
        image_line(fig_dir / "community_family_purity.png", "community family purity"),
        "",
        "解释：社区检测看的是整体图结构是否按 family 分块。family modularity、generated-only NMI/ARI 越高，说明 family 标签越容易从图社区结构中读出来。",
        "",
        "## 4. 图特征空间可分性",
        "",
        table(classifier_metrics),
        "",
        image_line(fig_dir / "graph_feature_pca_by_family.png", "graph feature pca by family"),
        "",
        image_line(fig_dir / "family_classifier_confusion_matrix.png", "family classifier confusion matrix"),
        "",
        "解释：这里不是检测 bot，而是检测 family。如果仅用图结构特征就能高准确率区分各 generated family，说明 family 机制确实造成了分布差异；但如果过高，也意味着存在 family signature。",
        "",
        "## 5. Human Alignment 分位数",
        "",
        table(alignment, max_rows=10),
        "",
        image_line(fig_dir / "human_percentile_by_family.png", "human percentile by family"),
        "",
        "解释：这部分把每个 bot 的图特征映射到 human 用户分布的百分位。大量落在 1% 或 99% 极端区间，说明虽然生成器有互动环境，但仍有不自然行为。",
        "",
        "## 结论使用方式",
        "",
        "如果本报告显示 family 行为差异明显、但同类抱团和 human 极端偏离不严重，那么当前数据适合作为图泛化 benchmark。反之，如果 family classifier 很强、same-family permutation 显著、或 human extreme share 很高，那么后续应先修生成环境，而不是继续扩大检测模型。",
        "",
    ]

    out = result_dir / "FAMILY_BEHAVIOR_MIXING_AUDIT_CN.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(out), "result_dir": str(result_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
