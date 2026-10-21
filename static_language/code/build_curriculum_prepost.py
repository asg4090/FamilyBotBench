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
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNNER_PATH = ROOT / "run_static_language_family_benchmark.py"
spec = importlib.util.spec_from_file_location("static_runner", RUNNER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {RUNNER_PATH}")
static_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static_runner)


def short_family(name: str) -> str:
    return str(name).split("_", 1)[0]


def train_and_eval(
    combined: pd.DataFrame,
    docs: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray | None,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    target_family: str,
    train_name: str,
    phase: str,
    round_id: int,
    max_features: int,
) -> list[dict[str, object]]:
    args = SimpleNamespace(max_features=max_features)
    models = static_runner.fit_models(docs, labels, train_idx, embeddings, args)
    rows = []
    for model_name, model_obj in models.items():
        pred, prob = static_runner.predict_model(model_name, model_obj, docs, embeddings, eval_idx)
        rows.append(
            {
                "protocol": "curriculum_pre_post",
                "round": round_id,
                "phase": phase,
                "train_family": train_name,
                "eval_family": target_family,
                "model": model_name,
                "train_size": int(len(train_idx)),
                **static_runner.metrics(labels[eval_idx], pred, prob),
            }
        )
    return rows


def plot_prepost(prepost: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for model in sorted(prepost["model"].unique()):
        subset = prepost[prepost["model"].eq(model)].copy()
        subset["family_short"] = subset["eval_family"].map(short_family)
        data = subset.pivot(index="family_short", columns="phase", values="recall").reindex(["F1", "F2", "F3", "F4", "F5"])
        data = data.rename(columns={"before_adding_family": "zero-shot before", "after_adding_family": "after training"})
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        data.plot(kind="bar", ax=ax, color={"zero-shot before": "#9ecae1", "after training": "#fb6a4a"})
        ax.set_title(f"{model.upper()} Five-Round Curriculum: Before vs After Adding Each Family")
        ax.set_xlabel("Round target family")
        ax.set_ylabel("Bot recall")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="")
        ax.set_xticklabels(data.index, rotation=0)
        fig.tight_layout()
        fig.savefig(figures / f"{model}_five_round_prepost_recall.png", dpi=180)
        plt.close(fig)


def append_report_section(output_dir: Path) -> None:
    report = output_dir / "STATIC_LANGUAGE_FAMILY_REPORT_CN.md"
    text = report.read_text(encoding="utf-8")
    marker = "\n## 7. 五轮 Curriculum Pre/Post 泛化测试\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    text += marker
    text += "\n".join(
        [
            "",
            "这组图以 family 为单位构造 5 轮顺序测试。每一轮先测试目标 family 在尚未加入训练集时的 recall，然后把该 family 加入训练集，再测试同一 family 的训练后 recall。",
            "",
            "- `zero-shot before`：训练集只包含 original human/bot 和之前轮次 family。",
            "- `after training`：训练集加入当前轮次目标 family。",
            "",
            "![TF-IDF Five Round Pre/Post](figures/tfidf_five_round_prepost_recall.png)",
            "",
            "![BGE Five Round Pre/Post](figures/bge_five_round_prepost_recall.png)",
            "",
            "明细数据见 `curriculum_prepost_matrix.csv`。",
            "",
        ]
    )
    report.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--truncate-words", type=int, default=600)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    combined = pd.read_csv(output_dir / "combined_sample_index.csv")
    docs = combined["doc"].fillna("").tolist()
    labels = combined["label"].to_numpy(dtype=int)
    embeddings_path = output_dir / "combined_bge_embeddings.npy"
    embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
    _, _, _, original_train_idx, _, _ = static_runner.build_original_rows(args.truncate_words)
    families = [
        "F1_original_architecture",
        "F2_tfidf_informed_language",
        "F3_human_anchor_local_tfidf",
        "F4_peba_trait_prompt",
        "F5_gemini_trait_prompt",
    ]
    family_indices = {
        family: combined.index[combined["family"].eq(family)].to_numpy(dtype=int)
        for family in families
    }
    rows: list[dict[str, object]] = []
    for idx, family in enumerate(families):
        previous = families[:idx]
        train_pre_parts = [original_train_idx] + [family_indices[f] for f in previous]
        train_pre = np.concatenate(train_pre_parts)
        pre_name = "original" if not previous else "original+" + "+".join(previous)
        rows.extend(
            train_and_eval(
                combined,
                docs,
                labels,
                embeddings,
                train_pre,
                family_indices[family],
                family,
                pre_name,
                "before_adding_family",
                idx + 1,
                args.max_features,
            )
        )
        current = families[: idx + 1]
        train_post = np.concatenate([original_train_idx] + [family_indices[f] for f in current])
        post_name = "original+" + "+".join(current)
        rows.extend(
            train_and_eval(
                combined,
                docs,
                labels,
                embeddings,
                train_post,
                family_indices[family],
                family,
                post_name,
                "after_adding_family",
                idx + 1,
                args.max_features,
            )
        )
    prepost = pd.DataFrame(rows)
    prepost.to_csv(output_dir / "curriculum_prepost_matrix.csv", index=False)
    plot_prepost(prepost, output_dir)
    append_report_section(output_dir)
    print(output_dir / "curriculum_prepost_matrix.csv")


if __name__ == "__main__":
    main()
