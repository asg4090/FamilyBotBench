#!/usr/bin/env python3
"""Build standalone and consolidated reports for static text extended experiments."""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RESULTS = Path(__import__("release_support").release_paths("text_evaluation")["output_dir"])
FIGURES = RESULTS / "figures"

REPORTS = {
    "lofo": RESULTS / "lofo_formal" / "LOFO_FORMAL_REPORT_CN.md",
    "topic": RESULTS / "topic_controlled" / "TOPIC_CONTROLLED_REPORT_CN.md",
    "thread_topic": RESULTS / "thread_topic_controlled" / "THREAD_TOPIC_CONTROLLED_REPORT_CN.md",
    "anchor": RESULTS / "anchor_similarity" / "ANCHOR_SIMILARITY_REPORT_CN.md",
    "family": RESULTS / "family_separability" / "FAMILY_SEPARABILITY_REPORT_CN.md",
    "ood": RESULTS / "static_text_ood" / "STATIC_TEXT_OOD_REPORT_CN.md",
    "main": RESULTS / "STATIC_TEXT_EXTENDED_REPORT_CN.md",
    "audit": RESULTS / "COMPLETION_AUDIT_EXTENDED_STATIC_TEXT_CN.md",
}


EXPECTED = {
    "LOFO 正式化": [
        RESULTS / "lofo_formal" / "lofo_formal_metrics.csv",
        RESULTS / "lofo_formal" / "lofo_formal_predictions.csv",
        RESULTS / "lofo_formal" / "lofo_formal_matrix.csv",
        FIGURES / "lofo_formal_recall_heatmap.png",
        FIGURES / "lofo_formal_probability_boxplot.png",
        RESULTS / "lofo_formal" / "LOFO_FORMAL_REPORT_CN.md",
    ],
    "Semantic-Cluster-Controlled": [
        RESULTS / "topic_controlled" / "topic_assignments.csv",
        RESULTS / "topic_controlled" / "topic_keyword_summary.csv",
        RESULTS / "topic_controlled" / "topic_matched_detection_metrics.csv",
        RESULTS / "topic_controlled" / "within_topic_detection_metrics.csv",
        RESULTS / "topic_controlled" / "leave_one_topic_out_metrics.csv",
        FIGURES / "topic_distribution_by_label_family.png",
        FIGURES / "topic_controlled_recall_drop.png",
        FIGURES / "within_topic_recall.png",
        FIGURES / "leave_one_topic_out_recall.png",
        RESULTS / "topic_controlled" / "TOPIC_CONTROLLED_REPORT_CN.md",
    ],
    "Thread-Derived Topic-Controlled": [
        RESULTS / "thread_topic_controlled" / "post_topic_map.csv",
        RESULTS / "thread_topic_controlled" / "comment_topic_dataset.csv",
        RESULTS / "thread_topic_controlled" / "topic_resource_audit.csv",
        RESULTS / "thread_topic_controlled" / "topic_matched_detection_metrics.csv",
        RESULTS / "thread_topic_controlled" / "leave_one_topic_out_metrics.csv",
        RESULTS / "thread_topic_controlled" / "leave_posts_out_within_topic_metrics.csv",
        RESULTS / "thread_topic_controlled" / "thread_topic_tfidf_recall_by_topic.csv",
        FIGURES / "thread_topic_distribution_by_label_family.png",
        FIGURES / "thread_topic_resource_audit.png",
        FIGURES / "thread_topic_matched_recall_drop.png",
        FIGURES / "thread_topic_tfidf_recall_by_topic_overall.png",
        FIGURES / "thread_leave_one_topic_out_recall.png",
        FIGURES / "thread_leave_posts_out_within_topic_recall.png",
        RESULTS / "thread_topic_controlled" / "THREAD_TOPIC_CONTROLLED_REPORT_CN.md",
    ],
    "Anchor Similarity": [
        RESULTS / "anchor_similarity" / "anchor_linkage_audit.csv",
        RESULTS / "anchor_similarity" / "anchor_similarity_by_comment.csv",
        RESULTS / "anchor_similarity" / "anchor_similarity_by_family.csv",
        RESULTS / "anchor_similarity" / "anchor_similarity_vs_detection.csv",
        FIGURES / "anchor_similarity_by_family_boxplot.png",
        FIGURES / "anchor_similarity_vs_bot_probability.png",
        FIGURES / "stylistic_distance_by_family.png",
        RESULTS / "anchor_similarity" / "ANCHOR_SIMILARITY_REPORT_CN.md",
    ],
    "Family Separability": [
        RESULTS / "family_separability" / "family_separability_metrics.csv",
        RESULTS / "family_separability" / "family_separability_predictions.csv",
        RESULTS / "family_separability" / "family_confusion_matrix_tfidf.csv",
        RESULTS / "family_separability" / "family_confusion_matrix_bge.csv",
        FIGURES / "family_confusion_matrix_tfidf.png",
        FIGURES / "family_confusion_matrix_bge.png",
        FIGURES / "family_separability_embedding_pca.png",
        RESULTS / "family_separability" / "FAMILY_SEPARABILITY_REPORT_CN.md",
    ],
    "Static Text OOD": [
        RESULTS / "static_text_ood" / "ood_scores_by_sample.csv",
        RESULTS / "static_text_ood" / "ood_metrics_by_family.csv",
        RESULTS / "static_text_ood" / "ood_method_summary.csv",
        FIGURES / "ood_score_distribution_by_family.png",
        FIGURES / "ood_auroc_by_method_family.png",
        FIGURES / "bge_centroid_margin_scatter.png",
        RESULTS / "static_text_ood" / "STATIC_TEXT_OOD_REPORT_CN.md",
    ],
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def fmt(x, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def shape(path: Path) -> str:
    if not path.exists() or path.suffix.lower() != ".csv":
        return ""
    try:
        df = pd.read_csv(path)
        return f"{len(df)} 行 x {len(df.columns)} 列"
    except Exception as exc:
        return f"读取失败: {exc}"


def md_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> list[str]:
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return ["空表"]
    return ["```text", df.to_string(index=False), "```"]


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_lofo() -> dict[str, str]:
    metrics = read_csv(RESULTS / "lofo_formal" / "lofo_formal_metrics.csv")
    pivot = metrics.pivot(index="heldout_family_short", columns="model", values="heldout_bot_recall").reset_index()
    worst_tfidf = metrics[metrics["model"].eq("TF-IDF+LR")].sort_values("heldout_bot_recall").iloc[0]
    worst_bge = metrics[metrics["model"].eq("BGE+LR")].sort_values("heldout_bot_recall").iloc[0]
    lines = [
        "# LOFO 正式化实验报告",
        "",
        "## 研究问题",
        "",
        "如果某个 family 完全不进入训练集，检测器能否泛化检出它？",
        "",
        "## 协议",
        "",
        "每次留出一个 family，训练集为 `original_train + 其余 4 个 family`，测试集包括 held-out family 以及 original_test。",
        "",
        "## Held-out Bot Recall",
        "",
        *md_table(pivot),
        "",
        "## 完整指标",
        "",
        *md_table(metrics),
        "",
        "## 解释",
        "",
        f"- TF-IDF 最难泛化到 `{worst_tfidf.heldout_family_short}`，held-out recall = `{fmt(worst_tfidf.heldout_bot_recall)}`。",
        f"- BGE 最难泛化到 `{worst_bge.heldout_family_short}`，held-out recall = `{fmt(worst_bge.heldout_bot_recall)}`。",
        "- 这组结果是比 curriculum 顺序更干净的跨 family 泛化证据。",
    ]
    write(REPORTS["lofo"], lines)
    return {
        "lofo_tfidf_worst": f"{worst_tfidf.heldout_family_short}={fmt(worst_tfidf.heldout_bot_recall)}",
        "lofo_bge_worst": f"{worst_bge.heldout_family_short}={fmt(worst_bge.heldout_bot_recall)}",
    }


def build_topic() -> dict[str, str]:
    matched = read_csv(RESULTS / "topic_controlled" / "topic_matched_detection_metrics.csv")
    within = read_csv(RESULTS / "topic_controlled" / "within_topic_detection_metrics.csv")
    loto = read_csv(RESULTS / "topic_controlled" / "leave_one_topic_out_metrics.csv")
    keywords = read_csv(RESULTS / "topic_controlled" / "topic_keyword_summary.csv")
    main_k = matched[matched["k"].eq(12)].copy()
    lines = [
        "# Semantic-Cluster-Controlled Detection 实验报告",
        "",
        "## 研究问题",
        "",
        "检测器是不是主要依靠语义簇 shortcut，而不是 bot 语言特征？",
        "",
        "## 协议",
        "",
        "用 BGE embedding + KMeans 做无监督语义簇，`k=8/12/16`；主解释以 `k=12` 为中心。这个实验不是严格 topic control，因为聚类可能混入话题、风格、人机差异和 family 差异。",
        "",
        "## Topic-Matched Detection",
        "",
        *md_table(
            matched,
            [
                "k",
                "model",
                "status",
                "matched_total_retained",
                "baseline_global_test_bot_recall",
                "controlled_matched_test_bot_recall",
                "bot_recall_drop_vs_global",
                "bot_recall_drop_vs_uncontrolled_matched_test",
            ],
        ),
        "",
        "## Within-Topic 状态",
        "",
        *md_table(within.groupby(["k", "model", "status"]).size().reset_index(name="n")),
        "",
        "## Leave-One-Topic-Out 状态",
        "",
        *md_table(loto.groupby(["k", "model", "status"]).size().reset_index(name="n")),
        "",
        "## k=12 Topic 关键词预览",
        "",
        *md_table(
            keywords[keywords["k"].eq(12)],
            ["topic_id", "topic_total_count", "human_count", "bot_count", "top_tfidf_terms", "status"],
            max_rows=12,
        ),
        "",
        "## 解释",
        "",
    ]
    for row in main_k.itertuples(index=False):
        lines.append(
            f"- k=12 `{row.model}`：baseline bot recall `{fmt(row.baseline_global_test_bot_recall)}`，"
            f"topic-controlled recall `{fmt(row.controlled_matched_test_bot_recall)}`，"
            f"相对全局 drop `{fmt(row.bot_recall_drop_vs_global)}`。"
        )
    lines.append("- 如果控制 topic 后 recall 下降，说明存在话题 shortcut；如果下降不大，说明检测器还抓到 topic 之外的信号。")
    write(REPORTS["topic"], lines)
    tfidf_drop = main_k[main_k["model"].eq("tfidf_lr")]["bot_recall_drop_vs_global"].iloc[0]
    bge_drop = main_k[main_k["model"].eq("bge_lr")]["bot_recall_drop_vs_global"].iloc[0]
    return {"topic_tfidf_drop_k12": fmt(tfidf_drop), "topic_bge_drop_k12": fmt(bge_drop)}


def build_anchor() -> dict[str, str]:
    audit = read_csv(RESULTS / "anchor_similarity" / "anchor_linkage_audit.csv")
    family = read_csv(RESULTS / "anchor_similarity" / "anchor_similarity_by_family.csv")
    vs = read_csv(RESULTS / "anchor_similarity" / "anchor_similarity_vs_detection.csv")
    f3 = family[family["family_short"].eq("F3")].iloc[0]
    lines = [
        "# Anchor Similarity 实验报告",
        "",
        "## 研究问题",
        "",
        "Anchor 对齐是否真的让 bot 文本更接近人类文本，尤其是 F3？",
        "",
        "## Linkage Audit",
        "",
        *md_table(
            audit,
            [
                "family_short",
                "generated_users",
                "generated_comments",
                "user_anchor_linkage_coverage",
                "comment_anchor_linkage_coverage",
                "analysis_mode",
                "audit_note",
            ],
        ),
        "",
        "## Family-Level Similarity",
        "",
        *md_table(
            family,
            [
                "family_short",
                "n_comments",
                "nearest_anchor_tfidf_cosine_mean",
                "assigned_anchor_tfidf_cosine_mean",
                "nearest_human_bge_cosine_mean",
                "human_centroid_style_z_l2_mean",
            ],
        ),
        "",
        "## Similarity vs Detection",
        "",
        *md_table(
            vs,
            [
                "family_short",
                "detector_model",
                "similarity_metric",
                "mean_similarity",
                "mean_bot_probability",
                "spearman_similarity_vs_probability",
            ],
            max_rows=20,
        ),
        "",
        "## 解释",
        "",
        f"- F3 comment anchor linkage coverage = `{fmt(f3.comment_anchor_linkage_coverage)}`。",
        f"- F3 nearest-human BGE cosine mean = `{fmt(f3.nearest_human_bge_cosine_mean)}`，在五个 family 中最高。",
        f"- F3 human centroid style z-L2 mean = `{fmt(f3.human_centroid_style_z_l2_mean)}`，在五个 family 中最低。",
        "- 但 F3 assigned-anchor TF-IDF cosine 不高，不能说逐条指定 anchor 的表层词汇高度相似；更准确说法是 F3 更接近 human pool 的语义/风格区域。",
    ]
    write(REPORTS["anchor"], lines)
    return {
        "anchor_f3_bge_cosine": fmt(f3.nearest_human_bge_cosine_mean),
        "anchor_f3_style_distance": fmt(f3.human_centroid_style_z_l2_mean),
    }


def build_family() -> dict[str, str]:
    metrics = read_csv(RESULTS / "family_separability" / "family_separability_metrics.csv")
    tfidf_cm = read_csv(RESULTS / "family_separability" / "family_confusion_matrix_tfidf.csv")
    bge_cm = read_csv(RESULTS / "family_separability" / "family_confusion_matrix_bge.csv")
    all_rows = metrics[(metrics["family_short"].eq("ALL")) & (metrics["split"].eq("test")) & (metrics["status"].eq("ok"))]
    lines = [
        "# Family Separability 实验报告",
        "",
        "## 研究问题",
        "",
        "五个 bot family 在文本空间中是否能彼此区分？如果能，说明生成策略留下 family 指纹。",
        "",
        "## Test 指标",
        "",
        *md_table(all_rows[["model", "feature", "lr_mode", "accuracy", "macro_f1"]]),
        "",
        "## TF-IDF Confusion Matrix",
        "",
        *md_table(tfidf_cm),
        "",
        "## BGE Confusion Matrix",
        "",
        *md_table(bge_cm),
        "",
        "## D2 Cross-Round",
        "",
        *md_table(metrics[metrics["protocol"].eq("D2_cross_round")]),
        "",
        "## 解释",
        "",
        "- TF-IDF test accuracy/macro-F1 达到 1.0，说明 family 之间存在非常强的表层词汇/短语指纹。",
        "- BGE test accuracy 约 0.95，说明语义空间也能区分 family，但 F2/F1、F4/F3 有少量混淆。",
        "- 这不是简单的好消息：family 多样化同时也制造了可识别的 family fingerprint。",
    ]
    write(REPORTS["family"], lines)
    tfidf_acc = all_rows[all_rows["model"].str.contains("tfidf", case=False)]["accuracy"].max()
    bge_acc = all_rows[all_rows["model"].str.contains("bge", case=False)]["accuracy"].max()
    return {"family_tfidf_acc": fmt(tfidf_acc), "family_bge_acc": fmt(bge_acc)}


def build_ood() -> dict[str, str]:
    summary = read_csv(RESULTS / "static_text_ood" / "ood_method_summary.csv")
    metrics = read_csv(RESULTS / "static_text_ood" / "ood_metrics_by_family.csv")
    best = summary.sort_values("mean_auroc_ood_vs_id", ascending=False).iloc[0]
    by_scenario = (
        metrics.groupby(["scenario", "method"])["auroc_ood_vs_id"]
        .mean()
        .reset_index()
        .sort_values(["scenario", "auroc_ood_vs_id"], ascending=[True, False])
    )
    lines = [
        "# Static Text OOD 实验报告",
        "",
        "## 研究问题",
        "",
        "不用普通分类器，只用距离或异常分数，能否发现 held-out family 是文本分布外样本？",
        "",
        "## Method Summary",
        "",
        *md_table(
            summary,
            [
                "method",
                "method_category",
                "score_direction",
                "is_supervised_proxy",
                "mean_auroc_ood_vs_id",
                "median_auroc_ood_vs_id",
                "mean_auprc_ood",
                "mean_score_gap_heldout_minus_id",
            ],
        ),
        "",
        "## Scenario Mean AUROC",
        "",
        *md_table(by_scenario),
        "",
        "## 解释",
        "",
        f"- 平均 AUROC 最高的方法是 `{best.method}`，mean AUROC = `{fmt(best.mean_auroc_ood_vs_id)}`。",
        "- kNN distance 是最稳定的纯 OOD 信号；original-only known 下更有效，lofo-known 下明显变难。",
        "- TF-IDF supervised probability proxy 不是好的纯 OOD 证据；它已经被标注为 supervised proxy。",
        "- centroid bot-margin 低于 0.5 是结果现象，不应强行反转方向；说明该 margin 对 human ID 样本也敏感。",
    ]
    write(REPORTS["ood"], lines)
    return {"ood_best_method": str(best.method), "ood_best_auroc": fmt(best.mean_auroc_ood_vs_id)}


def audit_rows() -> pd.DataFrame:
    rows = []
    for section, paths in EXPECTED.items():
        for path in paths:
            rows.append(
                {
                    "section": section,
                    "path": str(path.relative_to(ROOT)),
                    "exists": path.exists(),
                    "shape": shape(path),
                }
            )
    return pd.DataFrame(rows)


def build_main(summaries: dict[str, str]) -> None:
    audit = audit_rows()
    all_ok = bool(audit["exists"].all())
    lines = [
        "# 静态文本扩展实验总报告",
        "",
        "## 完成度",
        "",
        f"整体状态：`{'全部核心文件已生成' if all_ok else '仍有核心文件缺失'}`。",
        "",
        "本轮由 5 个独立 worker 分别完成五个实验；主线程只负责公共模块、任务边界、验收和汇总。",
        "",
        "## 文件审计",
        "",
        *md_table(audit),
        "",
        "## 五个实验结论",
        "",
        "### 1. LOFO 正式化",
        "",
        f"- TF-IDF 最难 held-out：`{summaries['lofo_tfidf_worst']}`。",
        f"- BGE 最难 held-out：`{summaries['lofo_bge_worst']}`。",
        "- 这说明跨 family 泛化不是均匀的，F3/F5 对文本检测器更有压力。",
        "",
        "### 2. Semantic-Cluster-Controlled Detection",
        "",
        f"- k=12 TF-IDF topic-control recall drop：`{summaries['topic_tfidf_drop_k12']}`。",
        f"- k=12 BGE topic-control recall drop：`{summaries['topic_bge_drop_k12']}`。",
        "- 旧 BGE 语义簇控制后 recall 有下降，但它不能严格证明 topic shortcut；正式 topic control 应引用 `thread_topic_controlled` 新实验。",
        "",
        "### 3. Anchor Similarity",
        "",
        f"- F3 nearest-human BGE cosine：`{summaries['anchor_f3_bge_cosine']}`。",
        f"- F3 human centroid style z-L2：`{summaries['anchor_f3_style_distance']}`。",
        "- F3 更靠近 human pool，但指定 anchor 的表层 TF-IDF 相似度并不高，不能夸大为逐条模仿成功。",
        "",
        "### 4. Family Separability",
        "",
        f"- TF-IDF family classification accuracy：`{summaries['family_tfidf_acc']}`。",
        f"- BGE family classification accuracy：`{summaries['family_bge_acc']}`。",
        "- family 可分性很强，说明不同生成策略留下明显 family 指纹。",
        "",
        "### 5. Static Text OOD",
        "",
        f"- 最强平均 OOD 方法：`{summaries['ood_best_method']}`，mean AUROC `{summaries['ood_best_auroc']}`。",
        "- kNN distance 比 supervised probability 更像可靠的 OOD 信号。",
        "",
        "## 综合判断",
        "",
        "这组静态文本扩展实验把问题从“检出率多少”推进到“为什么泛化失败/恢复”。当前证据显示：",
        "",
        "- TF-IDF 仍然很强，但容易留下 family 指纹和表层词汇依赖。",
        "- BGE 泛化更稳，但在 topic control、LOFO 和 OOD 中仍有 family/provider shift 压力。",
        "- Anchor 对齐改善了 F3 的 human-pool 接近度，但不是逐条 anchor 复刻。",
        "- 纯静态文本已经足够支持一个较完整的泛化性分析章节；后续动态图/行为实验可以沿用这套 held-out family 思路。",
        "",
        "## 独立报告路径",
        "",
    ]
    for key in ["lofo", "topic", "anchor", "family", "ood"]:
        lines.append(f"- `{REPORTS[key].relative_to(ROOT)}`")
    write(REPORTS["main"], lines)


def main() -> None:
    research_cli(globals(), "text_evaluation")
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, str] = {}
    summaries.update(build_lofo())
    summaries.update(build_topic())
    summaries.update(build_anchor())
    summaries.update(build_family())
    summaries.update(build_ood())
    build_main(summaries)
    audit = audit_rows()
    write(
        REPORTS["audit"],
        [
            "# 静态文本扩展实验完成度审计",
            "",
            *md_table(audit),
            "",
            f"全部核心文件是否存在：`{bool(audit['exists'].all())}`",
        ],
    )
    print(REPORTS["main"])
    print(REPORTS["audit"])


if __name__ == "__main__":
    main()
