#!/usr/bin/env python3
"""Run selected curriculum-order experiments for the static family benchmark."""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
STATIC_RUN = PROJECT_ROOT / "static_language" / "outputs" / "full_5family_50bot_3round_gpt54_20260601"
ANALYSIS_DIR = PROJECT_ROOT / "baselines" / "text"
RESULT_DIR = SCRIPT_DIR / "results"
FIG_DIR = RESULT_DIR / "figures"
apply_paths(globals(), "order_evaluation")

if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from comment_text_only_experiment import split_indices  # noqa: E402


FAMILIES = [
    "F1_original_architecture",
    "F2_tfidf_informed_language",
    "F3_human_anchor_local_tfidf",
    "F4_peba_trait_prompt",
    "F5_gemini_trait_prompt",
]

SHORT = {
    "F1_original_architecture": "F1",
    "F2_tfidf_informed_language": "F2",
    "F3_human_anchor_local_tfidf": "F3",
    "F4_peba_trait_prompt": "F4",
    "F5_gemini_trait_prompt": "F5",
}

FULL = {value: key for key, value in SHORT.items()}

SELECTED_ORDERS = [
    ("engineering_order", ["F1", "F2", "F3", "F4", "F5"]),
    ("hard_to_easy", ["F2", "F3", "F5", "F4", "F1"]),
    ("easy_to_hard", ["F1", "F4", "F5", "F3", "F2"]),
    ("provider_shift_first", ["F5", "F1", "F2", "F3", "F4"]),
    ("anchor_first", ["F3", "F2", "F5", "F4", "F1"]),
    ("trait_first", ["F4", "F5", "F2", "F3", "F1"]),
]


def train_tfidf(docs: list[str], labels: np.ndarray, train_idx: np.ndarray):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    clf.fit(vectorizer.fit_transform([docs[i] for i in train_idx]), labels[train_idx])
    return vectorizer, clf


def predict_tfidf(model, docs: list[str], idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectorizer, clf = model
    x = vectorizer.transform([docs[i] for i in idx])
    return clf.predict(x), clf.predict_proba(x)[:, 1]


def train_bge(embeddings: np.ndarray, labels: np.ndarray, train_idx: np.ndarray):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(embeddings[train_idx], labels[train_idx])
    return clf


def predict_bge(model, embeddings: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(embeddings[idx])
    prob = model.predict_proba(embeddings[idx])[:, 1]
    return pred, prob


def recall_for(labels: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> float:
    y = labels[idx]
    positives = int((y == 1).sum())
    if positives == 0:
        return float("nan")
    return float(((pred == 1) & (y == 1)).sum() / positives)


def family_subset_key(families: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(families))


def load_data():
    combined = pd.read_csv(STATIC_RUN / "combined_sample_index.csv")
    combined["doc"] = combined["doc"].fillna("")
    docs = combined["doc"].tolist()
    labels = combined["label"].to_numpy(dtype=int)
    embeddings = np.load(STATIC_RUN / "combined_bge_embeddings.npy")

    original_idx = combined.index[combined["source"].isin(["original_human", "original_bot"])].to_numpy()
    original_labels = labels[original_idx]
    orig_train_rel, _orig_val_rel, _orig_test_rel = split_indices(original_labels, 20260531)
    original_train_idx = original_idx[orig_train_rel]

    family_indices = {
        family: combined.index[combined["family"] == family].to_numpy()
        for family in FAMILIES
    }
    return combined, docs, labels, embeddings, original_train_idx, family_indices


def build_subset_models(
    docs: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    original_train_idx: np.ndarray,
    family_indices: dict[str, np.ndarray],
):
    subset_models: dict[tuple[str, ...], dict[str, object]] = {}
    for subset_size in range(0, len(FAMILIES) + 1):
        for subset in itertools.combinations(FAMILIES, subset_size):
            key = family_subset_key(subset)
            train_idx = np.concatenate([original_train_idx] + [family_indices[f] for f in key])
            print(f"training subset size={subset_size}: {'+'.join(SHORT[f] for f in key) or 'original_only'}", flush=True)
            subset_models[key] = {
                "tfidf": train_tfidf(docs, labels, train_idx),
                "bge": train_bge(embeddings, labels, train_idx),
            }
    return subset_models


def evaluate_orders(
    docs: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    family_indices: dict[str, np.ndarray],
    subset_models: dict[tuple[str, ...], dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for order_id, (order_name, short_order) in enumerate(SELECTED_ORDERS, start=1):
        order = [FULL[item] for item in short_order]
        seen: list[str] = []
        order_display = ">".join(short_order)
        for step, family in enumerate(order, start=1):
            before_key = family_subset_key(seen)
            after_key = family_subset_key(seen + [family])
            eval_idx = family_indices[family]
            for model_name in ["tfidf", "bge"]:
                if model_name == "tfidf":
                    before_pred, before_prob = predict_tfidf(subset_models[before_key]["tfidf"], docs, eval_idx)
                    after_pred, after_prob = predict_tfidf(subset_models[after_key]["tfidf"], docs, eval_idx)
                else:
                    before_pred, before_prob = predict_bge(subset_models[before_key]["bge"], embeddings, eval_idx)
                    after_pred, after_prob = predict_bge(subset_models[after_key]["bge"], embeddings, eval_idx)
                before_recall = recall_for(labels, before_pred, eval_idx)
                after_recall = recall_for(labels, after_pred, eval_idx)
                rows.append(
                    {
                        "order_id": order_id,
                        "order_name": order_name,
                        "order": order_display,
                        "step": step,
                        "seen_before": ">".join(SHORT[f] for f in seen) or "original_only",
                        "family": family,
                        "family_short": SHORT[family],
                        "model": model_name,
                        "before_recall": before_recall,
                        "after_recall": after_recall,
                        "recovery_gain": after_recall - before_recall,
                        "before_mean_probability": float(before_prob.mean()),
                        "after_mean_probability": float(after_prob.mean()),
                    }
                )
            seen.append(family)
    return pd.DataFrame(rows)


def evaluate_bridge_effects(
    docs: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    family_indices: dict[str, np.ndarray],
    subset_models: dict[tuple[str, ...], dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for order_id, (order_name, short_order) in enumerate(SELECTED_ORDERS, start=1):
        order = [FULL[item] for item in short_order]
        seen: list[str] = []
        order_display = ">".join(short_order)
        for step, added_family in enumerate(order, start=1):
            remaining = order[step:]
            if not remaining:
                seen.append(added_family)
                continue
            before_key = family_subset_key(seen)
            after_key = family_subset_key(seen + [added_family])
            for target_family in remaining:
                eval_idx = family_indices[target_family]
                for model_name in ["tfidf", "bge"]:
                    if model_name == "tfidf":
                        before_pred, before_prob = predict_tfidf(subset_models[before_key]["tfidf"], docs, eval_idx)
                        after_pred, after_prob = predict_tfidf(subset_models[after_key]["tfidf"], docs, eval_idx)
                    else:
                        before_pred, before_prob = predict_bge(subset_models[before_key]["bge"], embeddings, eval_idx)
                        after_pred, after_prob = predict_bge(subset_models[after_key]["bge"], embeddings, eval_idx)
                    before_recall = recall_for(labels, before_pred, eval_idx)
                    after_recall = recall_for(labels, after_pred, eval_idx)
                    rows.append(
                        {
                            "order_id": order_id,
                            "order_name": order_name,
                            "order": order_display,
                            "step_added": step,
                            "seen_before": ">".join(SHORT[f] for f in seen) or "original_only",
                            "added_family": added_family,
                            "added_family_short": SHORT[added_family],
                            "target_unseen_family": target_family,
                            "target_family_short": SHORT[target_family],
                            "model": model_name,
                            "target_recall_before_adding_family": before_recall,
                            "target_recall_after_adding_family": after_recall,
                            "bridge_gain": after_recall - before_recall,
                            "target_probability_before": float(before_prob.mean()),
                            "target_probability_after": float(after_prob.mean()),
                        }
                    )
            seen.append(added_family)
    return pd.DataFrame(rows)


def make_summaries(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order_summary = (
        metrics.groupby(["model", "order_id", "order_name", "order"])
        .agg(
            mean_before_recall=("before_recall", "mean"),
            mean_after_recall=("after_recall", "mean"),
            mean_recovery_gain=("recovery_gain", "mean"),
            min_before_recall=("before_recall", "min"),
            max_before_recall=("before_recall", "max"),
        )
        .reset_index()
    )
    family_summary = (
        metrics.groupby(["model", "family", "family_short"])
        .agg(
            before_recall_mean=("before_recall", "mean"),
            before_recall_std=("before_recall", "std"),
            before_recall_min=("before_recall", "min"),
            before_recall_max=("before_recall", "max"),
            after_recall_mean=("after_recall", "mean"),
            recovery_gain_mean=("recovery_gain", "mean"),
        )
        .reset_index()
    )
    position_summary = (
        metrics.groupby(["model", "family_short", "step"])
        .agg(
            before_recall_mean=("before_recall", "mean"),
            before_recall_std=("before_recall", "std"),
            n=("before_recall", "size"),
        )
        .reset_index()
    )
    return order_summary, family_summary, position_summary


def make_bridge_summary(bridge: pd.DataFrame) -> pd.DataFrame:
    return (
        bridge.groupby(["model", "added_family", "added_family_short"])
        .agg(
            bridge_gain_mean=("bridge_gain", "mean"),
            bridge_gain_std=("bridge_gain", "std"),
            bridge_gain_min=("bridge_gain", "min"),
            bridge_gain_max=("bridge_gain", "max"),
            target_recall_before_mean=("target_recall_before_adding_family", "mean"),
            target_recall_after_mean=("target_recall_after_adding_family", "mean"),
            n=("bridge_gain", "size"),
        )
        .reset_index()
    )


def plot_order_curves(metrics: pd.DataFrame) -> None:
    for recall_col, label in [("before_recall", "before"), ("after_recall", "after")]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        for ax, model in zip(axes, ["tfidf", "bge"]):
            sub = metrics[metrics["model"] == model]
            for order_name, order_sub in sub.groupby("order_name", sort=False):
                order_sub = order_sub.sort_values("step")
                ax.plot(order_sub["step"], order_sub[recall_col], marker="o", linewidth=2, label=order_name)
            ax.set_title(f"{model.upper()} {label} recall by curriculum order")
            ax.set_xlabel("curriculum step")
            ax.set_ylabel("recall")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_ylim(-0.03, 1.05)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"selected_orders_{label}_recall_curves.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_family_boxplot(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model in zip(axes, ["tfidf", "bge"]):
        sub = metrics[metrics["model"] == model]
        data = [sub[sub["family_short"] == family]["before_recall"].to_numpy() for family in ["F1", "F2", "F3", "F4", "F5"]]
        ax.boxplot(data, tick_labels=["F1", "F2", "F3", "F4", "F5"], showmeans=True)
        ax.set_title(f"{model.upper()} selected-order family hardness")
        ax.set_xlabel("family")
        ax.set_ylabel("before recall")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "selected_orders_family_hardness_boxplot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_position_effect(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, model in zip(axes, ["tfidf", "bge"]):
        sub = metrics[metrics["model"] == model]
        for family in ["F1", "F2", "F3", "F4", "F5"]:
            fam = sub[sub["family_short"] == family].sort_values("step")
            ax.scatter(fam["step"], fam["before_recall"], label=family, s=46, alpha=0.85)
            if len(fam) > 1:
                grouped = fam.groupby("step")["before_recall"].mean().reset_index()
                ax.plot(grouped["step"], grouped["before_recall"], linewidth=1.5, alpha=0.7)
        ax.set_title(f"{model.upper()} position effect")
        ax.set_xlabel("position in order")
        ax.set_ylabel("before recall")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(-0.03, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "selected_orders_position_effect.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_bridge_effect(bridge_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model in zip(axes, ["tfidf", "bge"]):
        sub = bridge_summary[bridge_summary["model"] == model].sort_values("added_family_short")
        x = np.arange(len(sub))
        ax.bar(x, sub["bridge_gain_mean"], yerr=sub["bridge_gain_std"].fillna(0), capsize=4)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["added_family_short"])
        ax.set_title(f"{model.upper()} bridge effect")
        ax.set_xlabel("family added to training")
        ax.set_ylabel("mean recall gain on later unseen families")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "selected_orders_bridge_effect.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    order_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    position_summary: pd.DataFrame,
    bridge_summary: pd.DataFrame,
) -> None:
    lines = [
        "# 有意义顺序实验结果报告",
        "",
        "本实验没有枚举 120 个全排列，只按照计划文档中的代表性 curriculum 顺序测试检测器泛化恢复。",
        "",
        "## 实验顺序",
        "",
    ]
    for name, order in SELECTED_ORDERS:
        lines.append(f"- `{name}`: {' -> '.join(order)}")

    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `selected_order_prepost_metrics.csv`：每条顺序、每一步、每个模型的加入前/加入后 recall。",
            "- `selected_order_summary.csv`：每条顺序的平均 before/after recall。",
            "- `selected_family_hardness_summary.csv`：family 在不同顺序下的稳定难度。",
            "- `selected_position_effect_summary.csv`：family 放在不同位置时的 before recall。",
            "- `selected_bridge_effects.csv`：某 family 加入训练后，对后续未见 family recall 的逐项影响。",
            "- `selected_bridge_effect_summary.csv`：每个 family 的平均 bridge effect。",
            "- `figures/selected_orders_before_recall_curves.png`：代表性顺序的 before recall 曲线。",
            "- `figures/selected_orders_after_recall_curves.png`：代表性顺序的 after recall 曲线。",
            "- `figures/selected_orders_family_hardness_boxplot.png`：family 顺序敏感性箱线图。",
            "- `figures/selected_orders_position_effect.png`：位置效应图。",
            "- `figures/selected_orders_bridge_effect.png`：family bridge effect 图。",
            "",
            "## 各顺序平均结果",
            "",
        ]
    )
    for model in ["tfidf", "bge"]:
        lines.append(f"### {model.upper()}")
        sub = order_summary[order_summary["model"] == model].sort_values("order_id")
        lines.append("")
        lines.append("| 顺序 | mean before recall | mean after recall | mean recovery gain | min before recall |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in sub.itertuples(index=False):
            lines.append(
                f"| {row.order_name} | {row.mean_before_recall:.3f} | {row.mean_after_recall:.3f} | "
                f"{row.mean_recovery_gain:.3f} | {row.min_before_recall:.3f} |"
            )
        lines.append("")

    lines.extend(["## Family 稳定难度", ""])
    for model in ["tfidf", "bge"]:
        lines.append(f"### {model.upper()}")
        sub = family_summary[family_summary["model"] == model].sort_values("family_short")
        lines.append("")
        lines.append("| family | before mean | before std | min | max | after mean |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in sub.itertuples(index=False):
            lines.append(
                f"| {row.family_short} | {row.before_recall_mean:.3f} | {row.before_recall_std:.3f} | "
                f"{row.before_recall_min:.3f} | {row.before_recall_max:.3f} | {row.after_recall_mean:.3f} |"
            )
        lines.append("")

    lines.extend(["## Bridge Effect", ""])
    lines.append("Bridge effect 衡量的是：把某个 family 加入训练后，对后续仍未见 family 的 recall 平均提升。")
    lines.append("")
    for model in ["tfidf", "bge"]:
        lines.append(f"### {model.upper()}")
        sub = bridge_summary[bridge_summary["model"] == model].sort_values("added_family_short")
        lines.append("")
        lines.append("| added family | mean bridge gain | std | min | max | n |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in sub.itertuples(index=False):
            lines.append(
                f"| {row.added_family_short} | {row.bridge_gain_mean:.3f} | {row.bridge_gain_std:.3f} | "
                f"{row.bridge_gain_min:.3f} | {row.bridge_gain_max:.3f} | {row.n} |"
            )
        lines.append("")

    lines.extend(["## 初步解释", ""])
    lines.append(
        "before recall 代表当前 family 尚未加入训练集时的检出能力，因此它测的是跨 family 泛化；"
        "after recall 代表把当前 family 加入训练后是否能恢复，因此它测的是模型容量和特征是否足以吸收该 family。"
    )
    lines.append("")
    lines.append(
        "如果一个 family 在多条顺序中 before recall 均值低且方差低，它更像稳定难例；"
        "如果它放后面才变高，说明它依赖前置 family 的覆盖，不一定本身容易。"
    )
    lines.append("")

    RESULT_DIR.joinpath("MEANINGFUL_ORDER_EXPERIMENT_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    research_cli(globals(), "order_evaluation")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    _combined, docs, labels, embeddings, original_train_idx, family_indices = load_data()
    subset_models = build_subset_models(docs, labels, embeddings, original_train_idx, family_indices)
    metrics = evaluate_orders(docs, labels, embeddings, family_indices, subset_models)
    bridge = evaluate_bridge_effects(docs, labels, embeddings, family_indices, subset_models)
    order_summary, family_summary, position_summary = make_summaries(metrics)
    bridge_summary = make_bridge_summary(bridge)

    metrics.to_csv(RESULT_DIR / "selected_order_prepost_metrics.csv", index=False)
    bridge.to_csv(RESULT_DIR / "selected_bridge_effects.csv", index=False)
    order_summary.to_csv(RESULT_DIR / "selected_order_summary.csv", index=False)
    family_summary.to_csv(RESULT_DIR / "selected_family_hardness_summary.csv", index=False)
    position_summary.to_csv(RESULT_DIR / "selected_position_effect_summary.csv", index=False)
    bridge_summary.to_csv(RESULT_DIR / "selected_bridge_effect_summary.csv", index=False)

    plot_order_curves(metrics)
    plot_family_boxplot(metrics)
    plot_position_effect(metrics)
    plot_bridge_effect(bridge_summary)
    write_report(order_summary, family_summary, position_summary, bridge_summary)

    print(f"results: {RESULT_DIR}")
    print(order_summary.to_string(index=False))
    print(family_summary.to_string(index=False))
    print(bridge_summary.to_string(index=False))


if __name__ == "__main__":
    main()
