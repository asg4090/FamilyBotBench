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
from typing import Any, Mapping, Sequence

import pandas as pd

try:
    from .detector_eval import evaluate_run, run_detector_sanity
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from detector_eval import evaluate_run, run_detector_sanity


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _detector_payload(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    payload = summary.get(name, {})
    return payload if isinstance(payload, Mapping) else {}


def detector_summary_lines(detector_summary: Mapping[str, Any]) -> list[str]:
    tfidf = _detector_payload(detector_summary, "tfidf")
    tfidf_light = _detector_payload(detector_summary, "tfidf_lightweight")
    if not tfidf_light and isinstance(tfidf.get("fallback"), Mapping):
        tfidf_light = tfidf["fallback"]
    bge = _detector_payload(detector_summary, "bge")
    graph = _detector_payload(detector_summary, "graphstats")
    rows = [
        ["TF-IDF 原训练探针", tfidf.get("status", "-"), tfidf.get("detector", "-"), _fmt(tfidf.get("mean_bot_probability")), tfidf.get("reason", "-")],
        [
            "TF-IDF 轻量相似度",
            tfidf_light.get("status", "-"),
            tfidf_light.get("detector", "-"),
            _fmt(tfidf_light.get("mean_nearest_neighbor_similarity")),
            tfidf_light.get("reason", "-"),
        ],
        ["BGE 本地语义", bge.get("status", "-"), bge.get("detector", "-"), _fmt(bge.get("mean_nearest_neighbor_cosine")), bge.get("reason", "-")],
        [
            "GraphStats",
            graph.get("status", "-"),
            graph.get("detector", "-"),
            _fmt(graph.get("mean_bot_probability_proxy")),
            graph.get("reason", "-"),
        ],
    ]
    lines = [
        _table(["片段", "状态", "detector", "核心均值", "说明"], rows),
        "",
    ]
    if tfidf_light:
        basic = tfidf_light.get("basic", {}) if isinstance(tfidf_light.get("basic"), Mapping) else {}
        lines.extend(
            [
                "文本基础统计：",
                _table(
                    ["指标", "值"],
                    [
                        ["文本数", basic.get("n", "-")],
                        ["用户数", basic.get("user_count", "-")],
                        ["平均词数", _fmt(basic.get("mean_words"))],
                        ["重复文本占比", _fmt(basic.get("duplicate_document_share"))],
                    ],
                ),
                "",
            ]
        )
    if graph:
        lines.extend(
            [
                "GraphStats 汇总：",
                _table(
                    ["指标", "值"],
                    [
                        ["用户数", graph.get("n", "-")],
                        ["combined 均值", _fmt(graph.get("mean_bot_probability_proxy"))],
                        ["combined 中位数", _fmt(graph.get("median_bot_probability_proxy"))],
                        ["combined 最大值", _fmt(graph.get("max_bot_probability_proxy"))],
                        ["概率 >= 0.66 占比", _fmt(graph.get("share_probability_ge_0_66"))],
                        ["平均出边 proxy", _fmt(graph.get("mean_total_out_weight_proxy"))],
                        ["平均入边 proxy", _fmt(graph.get("mean_total_in_weight_proxy"))],
                        ["平均方向平衡", _fmt(graph.get("mean_direction_balance_proxy"))],
                        ["平均互惠占比", _fmt(graph.get("mean_reciprocal_share_proxy"))],
                    ],
                ),
                "",
            ]
        )
        top_users = graph.get("top_users", [])
        if isinstance(top_users, list) and top_users:
            lines.extend(
                [
                    "GraphStats 高概率用户：",
                    _table(
                        ["user_id", "combined", "LR-like", "RF-like", "stat", "出边", "入边", "触发规则"],
                        [
                            [
                                item.get("user_id", "-"),
                                _fmt(item.get("graph_bot_probability_proxy")),
                                _fmt(item.get("graph_lr_like_probability_proxy")),
                                _fmt(item.get("graph_rf_like_vote_probability_proxy")),
                                _fmt(item.get("graph_statistical_probability_proxy")),
                                _fmt(item.get("total_out_weight_proxy")),
                                _fmt(item.get("total_in_weight_proxy")),
                                item.get("triggered_rules", "-") or "-",
                            ]
                            for item in top_users[:10]
                            if isinstance(item, Mapping)
                        ],
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "解释边界：",
            "- 该 detector summary 是 sanity check，不是正式泛化评测。",
            "- BGE 不联网下载模型；只有本地已有缓存或显式传入本地模型目录时才运行。",
            "- GraphStats 无 human/bot 标签时使用 LR-like 固定公式、RF-like 规则投票和本轮统计百分位组合概率。",
        ]
    )
    return lines


def write_iteration_report(iteration_dir: Path, gap_payload: dict[str, Any], labels: pd.DataFrame, rewrites: list[dict[str, Any]]) -> None:
    metrics = gap_payload.get("metrics", {})
    counts = labels["behavior_label"].value_counts().to_dict() if not labels.empty and "behavior_label" in labels else {}
    lines = [
        f"# PEvo iteration {iteration_dir.name}",
        "",
        "## 分布距离",
        "",
        f"- JS distance: `{metrics.get('js')}`",
        f"- TV distance: `{metrics.get('tv')}`",
        f"- KL divergence: `{metrics.get('kl')}`",
        f"- Entropy gap: `{metrics.get('entropy_gap')}`",
        "",
        "## 本轮行为分布",
        "",
    ]
    for label, count in counts.items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Persona 改写", "", f"- rewrite records: `{len(rewrites)}`"])
    iteration_dir.joinpath(f"alignment_report_{iteration_dir.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(output_dir: Path, history: pd.DataFrame, detector_summary: dict[str, Any]) -> None:
    lines = [
        "# PEBA/PEvo RedditBotSim 对齐实验报告",
        "",
        "## 实验目的",
        "",
        "本实验把旧的规则式漂移改成 PEBA/PEvo 式闭环：固定 Reddit 环境，运行 persona 群体，事后分类轨迹，计算模拟行为分布和 human reference 的差距，再通过 persona writer 改写人物深层设定。",
        "",
        "检测器不参与 persona 优化，只在最终阶段作为外部 sanity evaluation。",
        "",
        "## 对齐历史",
        "",
    ]
    if history.empty:
        lines.append("- 没有可用历史记录。")
    else:
        for _, row in history.iterrows():
            lines.append(
                f"- iteration `{row['iteration']}`: JS=`{row['js']:.4f}`, TV=`{row['tv']:.4f}`, KL=`{row['kl']:.4f}`, rewrites=`{int(row.get('rewrite_count', 0))}`"
            )
    lines.extend(["", "## 检测器 sanity", ""])
    lines.extend(detector_summary_lines(detector_summary))
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- PEvo 的更新目标是缩小 `p_sim` 与 human `p_real` 的行为分布差距，不是降低检测器概率。",
            "- writer 只改描述性 persona 字段，禁止显式写入“必须回访/总是回复/逃避检测”等动作命令。",
            "- 如果后续要做正式泛化评估，应在多个 human reference slice 上生成 family，再做 leave-one-family-out 和 cross-generation 测试。",
        ]
    )
    output_dir.joinpath("peba_pevo_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_detector_report(project_root: Path, iteration_dir: Path, detector_summary: Mapping[str, Any]) -> str:
    lines = [
        "# PEBA detector sanity summary",
        "",
        "## 结论",
        "",
        "该报告面向 final `generated_comments.csv`、`generated_posts.csv`、`graph_proxy_audit.csv` 做轻量 detector sanity 检查。",
        "",
        f"- project_root: `{project_root}`",
        f"- iteration_dir: `{iteration_dir}`",
        "",
    ]
    input_rows = detector_summary.get("input_rows", {})
    if isinstance(input_rows, Mapping) and input_rows:
        lines.extend(
            [
                "## 输入规模",
                "",
                _table(["输入", "行数"], [[key, value] for key, value in input_rows.items()]),
                "",
            ]
        )
    lines.extend(["## Detector sanity", ""])
    if "tfidf_lightweight" in detector_summary or "graphstats" in detector_summary:
        lines.extend(detector_summary_lines(detector_summary))
    else:
        lines.extend(detector_summary_lines({"tfidf": detector_summary.get("tfidf", {}), "graphstats": detector_summary.get("graphstats", {}), "bge": detector_summary.get("bge", {})}))
    lines.append("")
    return "\n".join(lines)


def write_detector_report(
    project_root: Path,
    iteration_dir: Path,
    out_path: Path,
    detector_summary: Mapping[str, Any] | None = None,
    bge_model: str | None = None,
    sample_size: int = 400,
) -> Path:
    summary = detector_summary or evaluate_run(project_root, iteration_dir, bge_model=bge_model, sample_size=sample_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_detector_report(project_root, iteration_dir, summary), encoding="utf-8")
    return out_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Chinese PEBA detector sanity markdown report.")
    parser.add_argument("--project-root", default=".", help="RedditBotSim/project root used by optional detector helpers.")
    parser.add_argument("--iteration-dir", required=True, help="Directory containing final generated outputs.")
    parser.add_argument("--out", help="Markdown report path. Defaults to ITERATION_DIR/peba_detector_sanity_report_cn.md.")
    parser.add_argument("--summary-json", help="Optional precomputed detector_summary.json to render.")
    parser.add_argument("--run-and-write-summary", action="store_true", help="Also write detector CSV/JSON artifacts via run_detector_sanity.")
    parser.add_argument("--out-dir", help="Detector artifact directory when --run-and-write-summary is set.")
    parser.add_argument("--bge-model", help="Optional local BGE/SentenceTransformer path or cached model id.")
    parser.add_argument("--sample-size", type=int, default=400)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    iteration_dir = Path(args.iteration_dir).expanduser().resolve()
    if args.summary_json:
        detector_summary = json.loads(Path(args.summary_json).expanduser().read_text(encoding="utf-8"))
    elif args.run_and_write_summary:
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else iteration_dir / "detector_sanity"
        detector_summary = run_detector_sanity(project_root, iteration_dir, out_dir, bge_model=args.bge_model)
    else:
        detector_summary = evaluate_run(project_root, iteration_dir, bge_model=args.bge_model, sample_size=args.sample_size)
    out_path = Path(args.out).expanduser().resolve() if args.out else iteration_dir / "peba_detector_sanity_report_cn.md"
    path = write_detector_report(project_root, iteration_dir, out_path, detector_summary=detector_summary, bge_model=args.bge_model, sample_size=args.sample_size)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
