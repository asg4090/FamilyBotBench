#!/usr/bin/env python3
"""Run multi-generation drift families and evaluate text/graph generalization."""

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
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REDDIT_DIR = PROJECT_ROOT / "RedditBotSim"
DATA_ROOT = PROJECT_ROOT
EMBEDDING_ROOT = PROJECT_ROOT / "baselines/text/modern_embedding_run"
ANALYSIS_DIR = PROJECT_ROOT / "baselines" / "text"
GRAPH_DIR = PROJECT_ROOT / "baselines" / "graph"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "continuous_generalization_benchmark"

for path in [SCRIPT_DIR, REDDIT_DIR, ANALYSIS_DIR, GRAPH_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from comment_text_only_experiment import split_indices, strict_clean_comment, truncate_words  # noqa: E402
from graph_only_bot_detection import Split as GraphSplit, graph_stat_features, run_ml_models, run_rgcn_models, set_seed as set_graph_seed  # noqa: E402
from modern_embedding_text_experiment import encode_docs, pick_device, slugify_model_name  # noqa: E402
from persona_social.alignment_gate import evaluate_alignment_gate  # noqa: E402
from persona_social.drift_auditors import audit_drift, audit_to_frames  # noqa: E402
from persona_social.drift_controller import build_mutation_plan, build_parent_personas, family_names, mutate_personas, write_manifest  # noqa: E402
from persona_social.drift_manifest import write_json  # noqa: E402
from run_dynamic_drift_generation import clean_generated_doc, original_text_data, run_persona_generation  # noqa: E402


FAMILY_BASE_FEEDBACK: dict[str, dict[str, float]] = {
    "A_language_drift": {
        "text_pressure": 0.95,
        "behavior_pressure": 0.38,
        "graph_pressure": 0.12,
        "time_pressure": 0.10,
        "pressure": 0.72,
    },
    "C_graph_drift": {
        "text_pressure": 0.22,
        "behavior_pressure": 0.76,
        "graph_pressure": 0.95,
        "time_pressure": 0.68,
        "pressure": 0.82,
    },
    "E_mixed_emergent_drift": {
        "text_pressure": 0.74,
        "behavior_pressure": 0.78,
        "graph_pressure": 0.73,
        "time_pressure": 0.70,
        "pressure": 0.80,
    },
}


@dataclass(frozen=True)
class RunRecord:
    family: str
    generation: int
    run_dir: Path
    manifest_path: Path
    plan_id: str
    child_family_version: str
    target_axes: tuple[str, ...]
    target_pass_rate: float
    cumulative_target_pass_rate: float
    severe_non_target_count: int
    gate_passed: bool
    selection_status: str
    feedback_in: dict[str, float]
    quick_feedback_out: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--families", default="A_language_drift,C_graph_drift,E_mixed_emergent_drift")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--bots-per-family", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--browse-count", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--llm-timeout", type=float, default=90.0)
    parser.add_argument("--max-resample", type=int, default=3)
    parser.add_argument("--providers", default="openai,gemini")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--api", choices=["responses", "chat"], default="responses")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--graph-epochs", type=int, default=30)
    parser.add_argument("--graph-hidden-dim", type=int, default=64)
    parser.add_argument("--truncate-words", type=int, default=600)
    parser.add_argument("--bge-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-bge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-quick-feedback", action="store_true")
    parser.add_argument("--skip-generalization", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parse_release_args(parser, globals(), "dynamic_drift")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _load_persona_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("personas", payload.get("records", payload.get("items", [])))
        if isinstance(rows, list):
            return [dict(item) for item in rows if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    raise ValueError(f"Unsupported manifest payload: {path}")


def _load_personas_from_manifest(path: Path) -> list[Any]:
    from persona_social_runner import _coerce_persona  # type: ignore

    return [_coerce_persona(item) for item in _load_persona_records(path)]


def _parent_manifest_name(generation: int) -> str:
    return "parent_manifest.json" if generation == 0 else "child_manifest.json"


def prepare_parent_control(family: str, family_dir: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    parent_dir = family_dir / "generation_000_parent"
    manifest_path = parent_dir / "parent_manifest.json"
    if manifest_path.exists() and (parent_dir / "generated_persona_profiles.csv").exists():
        return parent_dir, manifest_path
    if parent_dir.exists():
        shutil.rmtree(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    personas = build_parent_personas(family, args.bots_per_family, seed=args.seed, providers=split_csv(args.providers))
    write_manifest(manifest_path, family, 0, personas, plan=None)
    run_persona_generation(parent_dir, manifest_path, args)
    return parent_dir, manifest_path


def combine_feedback(family: str, generation: int, prior_feedback: dict[str, float] | None) -> dict[str, float]:
    base = dict(FAMILY_BASE_FEEDBACK.get(family, {}))
    gain = min(0.18, 0.08 * max(0, generation - 1))
    for key in ["pressure", "text_pressure", "behavior_pressure", "graph_pressure", "time_pressure"]:
        if key in base:
            base[key] = max(0.0, min(1.0, float(base[key]) + gain))
    if not prior_feedback:
        return base
    out = dict(base)
    for key, value in prior_feedback.items():
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if key not in out:
            out[key] = parsed
        else:
            out[key] = max(out[key], 0.65 * out[key] + 0.35 * parsed)
    return {key: round(float(value), 4) for key, value in out.items()}


def quick_feedback_from_metrics(metric_frames: list[pd.DataFrame]) -> dict[str, float]:
    if not metric_frames:
        return {}
    merged = pd.concat([df for df in metric_frames if not df.empty], ignore_index=True)
    if merged.empty:
        return {}

    def _mean_for(model_names: set[str], column: str = "recall") -> float | None:
        sub = merged[merged["model"].isin(model_names)]
        if sub.empty or column not in sub.columns:
            return None
        vals = pd.to_numeric(sub[column], errors="coerce").dropna()
        return None if vals.empty else float(vals.mean())

    text_recall = _mean_for({"tfidf", "bge"})
    graph_recall = _mean_for({"GraphStats_LR", "GraphStats_RF"})
    if text_recall is None and graph_recall is None:
        return {}
    if text_recall is None:
        text_recall = graph_recall
    if graph_recall is None:
        graph_recall = text_recall
    overall = float(np.mean([value for value in [text_recall, graph_recall] if value is not None]))
    return {
        "pressure": round(overall, 4),
        "detector_pressure": round(overall, 4),
        "text_pressure": round(float(text_recall), 4),
        "behavior_pressure": round(float(0.25 * text_recall + 0.75 * graph_recall), 4),
        "graph_pressure": round(float(graph_recall), 4),
        "time_pressure": round(float(0.15 * text_recall + 0.55 * graph_recall), 4),
    }


def run_quick_feedback(family: str, run_dir: Path, family_out: Path, args: argparse.Namespace) -> dict[str, float]:
    if args.skip_quick_feedback:
        return {}
    quick_dir = family_out / f"quick_feedback_{run_dir.name}"
    quick_dir.mkdir(parents=True, exist_ok=True)
    from run_dynamic_drift_generation import run_bge_detector, run_graphstats_detector, run_tfidf_detector  # noqa: E402

    metric_frames: list[pd.DataFrame] = []
    _, tfidf_metrics = run_tfidf_detector([(family, run_dir)], quick_dir, args)
    metric_frames.append(tfidf_metrics)
    if not args.skip_bge:
        _, bge_metrics = run_bge_detector([(family, run_dir)], quick_dir, args)
        metric_frames.append(bge_metrics if isinstance(bge_metrics, pd.DataFrame) else pd.DataFrame())
    _, graph_metrics = run_graphstats_detector([(family, run_dir)], quick_dir, args)
    metric_frames.append(graph_metrics)
    feedback = quick_feedback_from_metrics(metric_frames)
    write_json(quick_dir / "feedback_summary.json", feedback)
    return feedback


def run_family_generations(family: str, out_dir: Path, args: argparse.Namespace) -> list[RunRecord]:
    family_dir = out_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)
    parent_dir, parent_manifest = prepare_parent_control(family, family_dir, args)
    root_parent_dir = parent_dir
    parent_version = "parent"
    prior_feedback: dict[str, float] = {}
    records: list[RunRecord] = []
    gate_rows: list[dict[str, Any]] = []
    axis_frames: list[pd.DataFrame] = []

    for generation in range(1, args.generations + 1):
        personas = _load_personas_from_manifest(parent_manifest)
        feedback_in = combine_feedback(family, generation, prior_feedback)
        attempts: list[tuple[Path, Any, Any, Any, Any]] = []
        for attempt in range(1, args.max_resample + 1):
            plan = build_mutation_plan(
                family,
                generation,
                attempt,
                args.seed + generation * 1009 + attempt * 17,
                providers=split_csv(args.providers),
                feedback=feedback_in,
                parent_family_version=parent_version,
            )
            child_personas = mutate_personas(personas, plan)
            attempt_dir = family_dir / f"generation_{generation:03d}" / f"attempt_{attempt:02d}_{plan.plan_id}"
            if attempt_dir.exists():
                shutil.rmtree(attempt_dir)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = attempt_dir / "child_manifest.json"
            write_manifest(manifest_path, plan.family, plan.generation, child_personas, plan=plan)
            run_persona_generation(attempt_dir, manifest_path, args)
            local_audit = audit_drift(parent_dir, attempt_dir, plan)
            cumulative_audit = audit_drift(root_parent_dir, attempt_dir, plan)
            gate = evaluate_alignment_gate(attempt_dir, plan, local_audit)
            metric_df, axis_df = audit_to_frames(local_audit)
            metric_df.to_csv(attempt_dir / "drift_audit.csv", index=False)
            axis_df.to_csv(attempt_dir / "drift_audit_by_axis.csv", index=False)
            cumulative_metric_df, cumulative_axis_df = audit_to_frames(cumulative_audit)
            cumulative_metric_df.to_csv(attempt_dir / "drift_audit_cumulative.csv", index=False)
            cumulative_axis_df.to_csv(attempt_dir / "drift_audit_by_axis_cumulative.csv", index=False)
            write_json(attempt_dir / "alignment_gate.json", gate.to_dict())
            attempts.append((attempt_dir, plan, local_audit, cumulative_audit, gate))
            gate_rows.append(gate.to_dict() | {"family": family, "generation": generation, "attempt": attempt})
            axis_frames.append(axis_df.assign(family=family, generation=generation, attempt=attempt, audit_scope="local"))
            axis_frames.append(cumulative_axis_df.assign(family=family, generation=generation, attempt=attempt, audit_scope="cumulative"))
            if gate.passed:
                break

        passed = [item for item in attempts if item[4].passed]
        chosen_dir, chosen_plan, chosen_audit, chosen_cumulative_audit, chosen_gate = passed[0] if passed else max(
            attempts,
            key=lambda item: (item[2].target_pass_rate, -item[2].severe_non_target_count),
        )
        selection_status = "qualified" if chosen_gate.passed else "best_rejected"
        selected_dir = family_dir / f"generation_{generation:03d}_{selection_status}"
        if selected_dir.exists():
            shutil.rmtree(selected_dir)
        shutil.copytree(chosen_dir, selected_dir)
        quick_feedback = run_quick_feedback(family, selected_dir, family_dir, args)
        record = RunRecord(
            family=family,
            generation=generation,
            run_dir=selected_dir,
            manifest_path=selected_dir / "child_manifest.json",
            plan_id=chosen_plan.plan_id,
            child_family_version=chosen_plan.child_family_version,
            target_axes=tuple(chosen_plan.target_axes),
            target_pass_rate=float(chosen_audit.target_pass_rate),
            cumulative_target_pass_rate=float(chosen_cumulative_audit.target_pass_rate),
            severe_non_target_count=int(chosen_audit.severe_non_target_count),
            gate_passed=bool(chosen_gate.passed),
            selection_status=selection_status,
            feedback_in=feedback_in,
            quick_feedback_out=quick_feedback,
        )
        records.append(record)
        parent_dir = selected_dir
        parent_manifest = selected_dir / "child_manifest.json"
        parent_version = chosen_plan.child_family_version
        prior_feedback = quick_feedback

    lineage_df = pd.DataFrame(
        [
            {
                "family": record.family,
                "generation": record.generation,
                "run_dir": str(record.run_dir),
                "plan_id": record.plan_id,
                "child_family_version": record.child_family_version,
                "selection_status": record.selection_status,
                "gate_passed": record.gate_passed,
                "target_axes": ",".join(record.target_axes),
                "target_pass_rate": record.target_pass_rate,
                "cumulative_target_pass_rate": record.cumulative_target_pass_rate,
                "severe_non_target_count": record.severe_non_target_count,
                "feedback_in": json.dumps(record.feedback_in, ensure_ascii=False, sort_keys=True),
                "quick_feedback_out": json.dumps(record.quick_feedback_out, ensure_ascii=False, sort_keys=True),
            }
            for record in records
        ]
    )
    lineage_df.to_csv(family_dir / "family_lineage_summary.csv", index=False)
    if gate_rows:
        pd.DataFrame(gate_rows).to_csv(family_dir / "all_alignment_gates.csv", index=False)
    if axis_frames:
        pd.concat(axis_frames, ignore_index=True).to_csv(family_dir / "all_drift_axis_scores.csv", index=False)
    return records


def collect_generated_docs(records: list[RunRecord], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for record in records:
        comments = pd.read_csv(record.run_dir / "generated_comments.csv", dtype=str).fillna("")
        profiles = pd.read_csv(record.run_dir / "generated_persona_profiles.csv", dtype=str).fillna("")
        docs = clean_generated_doc(comments, profiles, args.truncate_words)
        docs["family"] = record.family
        docs["generation"] = record.generation
        docs["run_dir"] = str(record.run_dir)
        docs["benchmark_id"] = f"{record.family}_g{record.generation}"
        rows.append(docs)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_text_eval_sets(generated_docs: pd.DataFrame, target_family: str, target_generation: int, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_mask = (generated_docs["family"] == target_family) & (generated_docs["generation"] == target_generation)
    if protocol == "incremental_generation":
        train_generated = generated_docs[generated_docs["generation"] < target_generation].copy()
    elif protocol == "heldout_family_same_generation":
        train_generated = generated_docs[(generated_docs["generation"] == target_generation) & (generated_docs["family"] != target_family)].copy()
    else:
        raise KeyError(protocol)
    test_generated = generated_docs[target_mask].copy()
    return train_generated, test_generated


def _combine_train_text(original_df: pd.DataFrame, original_train_idx: np.ndarray, train_generated: pd.DataFrame) -> pd.DataFrame:
    train_original = original_df.iloc[original_train_idx].copy()
    if train_generated.empty:
        return train_original
    out = pd.concat(
        [
            train_original[["label", "doc", "source", "user_id", "family"]],
            train_generated[["label", "doc", "source", "user_id", "family"]],
        ],
        ignore_index=True,
    )
    return out


def _combine_test_text(original_df: pd.DataFrame, original_test_idx: np.ndarray, test_generated: pd.DataFrame) -> pd.DataFrame:
    human_test = original_df.iloc[original_test_idx].copy()
    human_test = human_test[human_test["label"] == 0][["label", "doc", "source", "user_id", "family"]]
    return pd.concat([human_test, test_generated[["label", "doc", "source", "user_id", "family"]]], ignore_index=True)


def evaluate_text_tfidf(records: list[RunRecord], out_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    original_df, train_idx, _val_idx, test_idx = original_text_data(args)
    generated_docs = collect_generated_docs(records, args)
    generated_docs.to_csv(out_dir / "generated_doc_table.csv", index=False)

    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    for protocol in ["incremental_generation", "heldout_family_same_generation"]:
        for record in records:
            train_generated, test_generated = build_text_eval_sets(generated_docs, record.family, record.generation, protocol)
            if test_generated.empty:
                continue
            train_df = _combine_train_text(original_df, train_idx, train_generated)
            test_df = _combine_test_text(original_df, test_idx, test_generated)
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
            x_train = vectorizer.fit_transform(train_df["doc"].fillna("").tolist())
            x_test = vectorizer.transform(test_df["doc"].fillna("").tolist())
            y_train = train_df["label"].to_numpy(dtype=int)
            clf.fit(x_train, y_train)
            prob = clf.predict_proba(x_test)[:, 1]
            pred = clf.predict(x_test)
            eval_df = test_df.copy()
            eval_df["protocol"] = protocol
            eval_df["model"] = "TFIDF_LR"
            eval_df["target_family"] = record.family
            eval_df["target_generation"] = record.generation
            eval_df["bot_probability"] = prob
            eval_df["pred_label"] = pred
            target_mask = (eval_df["source"] == "new_ai_bot")
            human_mask = (eval_df["label"] == 0)
            metric_rows.append(
                {
                    "protocol": protocol,
                    "model": "TFIDF_LR",
                    "target_family": record.family,
                    "target_generation": record.generation,
                    "train_generated_bots": int(len(train_generated)),
                    "test_generated_bots": int(target_mask.sum()),
                    "overall_accuracy": float((pred == eval_df["label"].to_numpy(dtype=int)).mean()),
                    "target_bot_recall": float(pred[target_mask].mean()) if target_mask.any() else float("nan"),
                    "human_test_fpr": float(pred[human_mask].mean()) if human_mask.any() else float("nan"),
                    "target_bot_mean_probability": float(prob[target_mask].mean()) if target_mask.any() else float("nan"),
                }
            )
            pred_rows.append(eval_df)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    metrics.to_csv(out_dir / "text_generalization_tfidf_metrics.csv", index=False)
    predictions.to_csv(out_dir / "text_generalization_tfidf_predictions.csv", index=False)
    return metrics, predictions


def evaluate_text_bge(records: list[RunRecord], out_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.linear_model import LogisticRegression

    slug = slugify_model_name(args.bge_model)
    base_embedding_path = EMBEDDING_ROOT / slug / "strict_clean_truncated_embeddings.npy"
    if args.skip_bge or not base_embedding_path.exists():
        metrics = pd.DataFrame(
            [
                {
                    "protocol": "all",
                    "model": "BGE_LR",
                    "status": "skipped",
                    "reason": "skip_bge_or_missing_base_embeddings",
                }
            ]
        )
        metrics.to_csv(out_dir / "text_generalization_bge_metrics.csv", index=False)
        return metrics, pd.DataFrame()

    original_df, train_idx, _val_idx, test_idx = original_text_data(args)
    base_embeddings = np.load(base_embedding_path)
    generated_docs = collect_generated_docs(records, args)
    if generated_docs.empty:
        return pd.DataFrame(), pd.DataFrame()
    generated_cache = out_dir / "generated_doc_embeddings.npy"
    if generated_cache.exists():
        gen_embeddings = np.load(generated_cache)
    else:
        device = pick_device(args.device)
        gen_embeddings = encode_docs(
            generated_docs["doc"].fillna("").tolist(),
            model_name=args.bge_model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            normalize=True,
        )
        np.save(generated_cache, gen_embeddings)

    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    for protocol in ["incremental_generation", "heldout_family_same_generation"]:
        for record in records:
            train_generated, test_generated = build_text_eval_sets(generated_docs, record.family, record.generation, protocol)
            if test_generated.empty:
                continue
            train_original = original_df.iloc[train_idx].copy()
            train_x = [base_embeddings[train_original["original_row"].to_numpy(dtype=int)]]
            train_y = [train_original["label"].to_numpy(dtype=int)]
            if not train_generated.empty:
                train_x.append(gen_embeddings[train_generated.index.to_numpy(dtype=int)])
                train_y.append(train_generated["label"].to_numpy(dtype=int))
            x_train = np.vstack(train_x)
            y_train = np.concatenate(train_y)
            human_test = original_df.iloc[test_idx].copy()
            human_test = human_test[human_test["label"] == 0]
            x_test = np.vstack(
                [
                    base_embeddings[human_test["original_row"].to_numpy(dtype=int)],
                    gen_embeddings[test_generated.index.to_numpy(dtype=int)],
                ]
            )
            test_df = pd.concat(
                [
                    human_test[["label", "doc", "source", "user_id", "family"]],
                    test_generated[["label", "doc", "source", "user_id", "family"]],
                ],
                ignore_index=True,
            )
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
            clf.fit(x_train, y_train)
            prob = clf.predict_proba(x_test)[:, 1]
            pred = clf.predict(x_test)
            eval_df = test_df.copy()
            eval_df["protocol"] = protocol
            eval_df["model"] = "BGE_LR"
            eval_df["target_family"] = record.family
            eval_df["target_generation"] = record.generation
            eval_df["bot_probability"] = prob
            eval_df["pred_label"] = pred
            target_mask = (eval_df["source"] == "new_ai_bot")
            human_mask = (eval_df["label"] == 0)
            metric_rows.append(
                {
                    "protocol": protocol,
                    "model": "BGE_LR",
                    "target_family": record.family,
                    "target_generation": record.generation,
                    "train_generated_bots": int(len(train_generated)),
                    "test_generated_bots": int(target_mask.sum()),
                    "overall_accuracy": float((pred == eval_df["label"].to_numpy(dtype=int)).mean()),
                    "target_bot_recall": float(pred[target_mask].mean()) if target_mask.any() else float("nan"),
                    "human_test_fpr": float(pred[human_mask].mean()) if human_mask.any() else float("nan"),
                    "target_bot_mean_probability": float(prob[target_mask].mean()) if target_mask.any() else float("nan"),
                }
            )
            pred_rows.append(eval_df)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    metrics.to_csv(out_dir / "text_generalization_bge_metrics.csv", index=False)
    predictions.to_csv(out_dir / "text_generalization_bge_predictions.csv", index=False)
    return metrics, predictions


def build_augmented_graph_records(records: list[RunRecord]) -> tuple[pd.DataFrame, np.ndarray, Any, Any, Any]:
    import torch
    from collections import Counter

    users = pd.read_csv(DATA_ROOT / "BotSim-24-Dataset" / "Users.csv", dtype=str).fillna("")
    labels = np.zeros(len(users), dtype=np.int64)
    labels[1907:] = 1
    node_rows = users[["user_id", "name"]].copy()
    node_rows["source"] = np.where(np.arange(len(node_rows)) >= 1907, "original_bot", "original_human")
    node_rows["family"] = ""
    node_rows["family_generation"] = -1
    node_rows["label"] = labels

    generated_rows = []
    for record in records:
        profiles = pd.read_csv(record.run_dir / "generated_persona_profiles.csv", dtype=str).fillna("")
        nodes = profiles[["user_id", "user_name", "family", "family_generation"]].drop_duplicates("user_id").rename(columns={"user_name": "name"})
        nodes["source"] = "new_ai_bot"
        nodes["label"] = 1
        generated_rows.append(nodes)
    if generated_rows:
        node_rows = pd.concat([node_rows, *generated_rows], ignore_index=True)

    labels = node_rows["label"].to_numpy(dtype=np.int64)
    node_index = {uid: i for i, uid in enumerate(node_rows["user_id"].astype(str))}
    edge_index = torch.load(DATA_ROOT / "BotSim-24-mini-sample" / "edge_index.pt", map_location="cpu", weights_only=True).long()
    if edge_index.ndim == 2 and edge_index.shape[0] == 2 and edge_index.shape[1] != 2:
        edge_index = edge_index.T.contiguous()
    edge_type = torch.load(DATA_ROOT / "BotSim-24-mini-sample" / "edge_type.pt", map_location="cpu", weights_only=True).long()
    edge_weight = torch.load(DATA_ROOT / "BotSim-24-mini-sample" / "edge_weight.pt", map_location="cpu", weights_only=True).reshape(-1).float()
    counter: Counter[tuple[int, int, int]] = Counter()

    def add_edge(source_user: str, target_user: str | None, rel: int) -> None:
        if source_user in node_index and target_user in node_index:
            counter[(node_index[source_user], node_index[str(target_user)], rel)] += 1

    for record in records:
        data_dir = record.run_dir / "run" / "data"
        profiles = pd.read_csv(record.run_dir / "generated_persona_profiles.csv", dtype=str).fillna("")
        bot_ids = set(profiles["user_id"].astype(str))
        posts = pd.read_csv(data_dir / "Posts.csv", dtype=str).fillna("")
        c1 = pd.read_csv(data_dir / "Comment_1.csv", dtype=str).fillna("")
        c2 = pd.read_csv(data_dir / "Comment_2.csv", dtype=str).fillna("")
        post_author = dict(zip(posts["submission_id"].astype(str), posts["author_id"].astype(str)))
        c1_author = dict(zip(c1["comment_id"].astype(str), c1["comment_author_id"].astype(str)))
        for _, row in c1[c1["comment_author_id"].isin(bot_ids)].iterrows():
            add_edge(str(row["comment_author_id"]), post_author.get(str(row["link_id"]).replace("t3_", "")), 0)
        for _, row in c2[c2["comment_author_id"].isin(bot_ids)].iterrows():
            post_id = str(row["link_id"]).replace("t3_", "")
            parent_id = str(row["parent_id"]).replace("t1_", "")
            add_edge(str(row["comment_author_id"]), post_author.get(post_id), 1)
            add_edge(str(row["comment_author_id"]), c1_author.get(parent_id), 2)

    if counter:
        new_edges = np.array([(src, dst) for (src, dst, _rel) in counter], dtype=np.int64)
        new_types = np.array([rel for (_src, _dst, rel) in counter], dtype=np.int64)
        new_weights = np.array(list(counter.values()), dtype=np.float32)
        edge_index = torch.cat([edge_index, torch.tensor(new_edges, dtype=torch.long)], dim=0)
        edge_type = torch.cat([edge_type, torch.tensor(new_types, dtype=torch.long)], dim=0)
        edge_weight = torch.cat([edge_weight, torch.tensor(new_weights, dtype=torch.float32)], dim=0)
    return node_rows, labels, edge_index, edge_type, edge_weight


def _graph_protocol_sets(node_rows: pd.DataFrame, target_family: str, target_generation: int, protocol: str) -> tuple[np.ndarray, np.ndarray]:
    generated_mask = node_rows["source"].eq("new_ai_bot")
    target_mask = generated_mask & node_rows["family"].eq(target_family) & node_rows["family_generation"].astype(int).eq(target_generation)
    if protocol == "incremental_generation":
        train_generated = generated_mask & node_rows["family_generation"].astype(int).lt(target_generation)
    elif protocol == "heldout_family_same_generation":
        train_generated = generated_mask & node_rows["family_generation"].astype(int).eq(target_generation) & ~node_rows["family"].eq(target_family)
    else:
        raise KeyError(protocol)
    return node_rows.index[train_generated].to_numpy(dtype=int), node_rows.index[target_mask].to_numpy(dtype=int)


def evaluate_graph_generalization(records: list[RunRecord], out_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not records:
        return pd.DataFrame(), pd.DataFrame()

    node_rows, labels, edge_index, edge_type, edge_weight = build_augmented_graph_records(records)
    features = graph_stat_features(edge_index, edge_type, edge_weight, len(labels))
    graph_root = out_dir / "graph_generalization"
    graph_root.mkdir(parents=True, exist_ok=True)
    features.to_csv(graph_root / "graph_stat_features.csv", index_label="node_id")
    node_rows.to_csv(graph_root / "graph_nodes.csv", index_label="node_id")

    original_idx = node_rows.index[node_rows["source"] != "new_ai_bot"].to_numpy(dtype=int)
    original_labels = labels[original_idx]
    original_train_rel, original_val_rel, original_test_rel = split_indices(original_labels, args.seed)
    original_train_idx = original_idx[original_train_rel]
    original_val_idx = original_idx[original_val_rel]
    original_test_human_idx = original_idx[original_test_rel][original_labels[original_test_rel] == 0]

    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    for protocol in ["incremental_generation", "heldout_family_same_generation"]:
        for record in records:
            train_generated_idx, target_generated_idx = _graph_protocol_sets(node_rows, record.family, record.generation, protocol)
            if len(target_generated_idx) == 0:
                continue
            split = GraphSplit(
                train_idx=np.concatenate([original_train_idx, train_generated_idx]),
                val_idx=original_val_idx,
                test_idx=np.concatenate([original_test_human_idx, target_generated_idx]),
            )
            set_graph_seed(args.seed + record.generation)
            ml_rows, ml_probs = run_ml_models(features, labels, split, args.seed + record.generation)
            rgcn_rows, _histories, rgcn_probs = run_rgcn_models(
                features,
                labels,
                split,
                edge_index,
                edge_type,
                edge_weight,
                args.seed + record.generation,
                args.graph_epochs,
                args.graph_hidden_dim,
            )
            all_probs = {**ml_probs, **rgcn_probs}
            for model_name, prob in all_probs.items():
                test_df = node_rows.iloc[split.test_idx].copy()
                test_df["protocol"] = protocol
                test_df["model"] = model_name
                test_df["target_family"] = record.family
                test_df["target_generation"] = record.generation
                test_df["true_label"] = labels[split.test_idx]
                test_df["bot_probability"] = prob[split.test_idx]
                test_df["pred_label"] = (test_df["bot_probability"].to_numpy(dtype=float) >= 0.5).astype(int)
                target_mask = test_df["source"].eq("new_ai_bot")
                human_mask = test_df["source"].eq("original_human")
                metric_rows.append(
                    {
                        "protocol": protocol,
                        "model": model_name,
                        "target_family": record.family,
                        "target_generation": record.generation,
                        "train_generated_bots": int(len(train_generated_idx)),
                        "test_generated_bots": int(target_mask.sum()),
                        "overall_accuracy": float((test_df["pred_label"].to_numpy(dtype=int) == test_df["true_label"].to_numpy(dtype=int)).mean()),
                        "target_bot_recall": float(test_df.loc[target_mask, "pred_label"].mean()) if target_mask.any() else float("nan"),
                        "human_test_fpr": float(test_df.loc[human_mask, "pred_label"].mean()) if human_mask.any() else float("nan"),
                        "target_bot_mean_probability": float(test_df.loc[target_mask, "bot_probability"].mean()) if target_mask.any() else float("nan"),
                    }
                )
                pred_rows.append(test_df)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    metrics.to_csv(out_dir / "graph_generalization_metrics.csv", index=False)
    predictions.to_csv(out_dir / "graph_generalization_predictions.csv", index=False)
    return metrics, predictions


def _line_plot(df: pd.DataFrame, protocol: str, path: Path, title: str) -> None:
    sub = df[df["protocol"] == protocol].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    generation_values = sorted(int(value) for value in sub["target_generation"].dropna().unique().tolist())
    family_values = sorted(str(value) for value in sub["target_family"].dropna().unique().tolist())
    for model_name, model_df in sub.groupby("model"):
        line = (
            model_df.groupby("target_generation")["target_bot_recall"]
            .mean()
            .reset_index()
            .sort_values("target_generation")
        )
        ax.plot(line["target_generation"], line["target_bot_recall"], marker="o", label=model_name)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Target generation (1/2/3)")
    ax.set_ylabel("Mean target bot recall over target families")
    if generation_values:
        ax.set_xlim(min(generation_values) - 0.1, max(generation_values) + 0.1)
        ax.xaxis.set_major_locator(FixedLocator(generation_values))
        ax.xaxis.set_major_formatter(FixedFormatter([str(value) for value in generation_values]))
        ax.minorticks_off()
    ax.set_title(title)
    ax.legend(fontsize=8)
    family_label = ", ".join(family_values)
    if protocol == "incremental_generation":
        note = (
            "Each point = mean recall across target families "
            f"[{family_label}] at generation g. "
            "Train bots include all earlier generations (< g) from all three families."
        )
    else:
        note = (
            "Each point = mean recall across target families "
            f"[{family_label}] at generation g."
        )
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _heatmap(df: pd.DataFrame, model_name: str, protocol: str, path: Path, title: str) -> None:
    sub = df[(df["protocol"] == protocol) & (df["model"] == model_name)].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index="target_family", columns="target_generation", values="target_bot_recall", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    im = ax.imshow(pivot.fillna(np.nan), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns.tolist())
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index.tolist())
    ax.set_xlabel("Target generation (1/2/3)")
    ax.set_ylabel("Target family")
    ax.set_title(title)
    for i, family in enumerate(pivot.index):
        for j, generation in enumerate(pivot.columns):
            value = pivot.loc[family, generation]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    if protocol == "heldout_family_same_generation":
        note = (
            "Rows = target family A/C/E. Columns = target generation g. "
            "Train bots use the other families from the same generation g; "
            "this is held-out-family transfer, not zero-shot."
        )
    else:
        note = "Rows = target family. Columns = target generation."
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, lineage_df: pd.DataFrame, text_metrics: pd.DataFrame, graph_metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# 连续漂移泛化性 benchmark 报告",
        "",
        f"- families: `{args.families}`",
        f"- generations: `{args.generations}`",
        f"- bots_per_family: `{args.bots_per_family}`",
        f"- simulation_rounds: `{args.rounds}`",
        f"- browse_count: `{args.browse_count}`",
        f"- providers: `{args.providers}`",
        f"- dry_run: `{args.dry_run}`",
        "",
        "## 图表阅读说明",
        "",
        "- 本实验同时对三个 family 做了连续漂移：`A_language_drift`、`C_graph_drift`、`E_mixed_emergent_drift`，每条 family 都各自从第 1 代漂移到第 3 代。",
        "- 三个 family 都同时参加文本检测和图检测。`A/C/E` 表示漂移设计重点不同，不表示只在某一种检测模态中评估。",
        "- 折线图中的横轴 `target_generation=1/2/3` 表示目标测试样本属于第几代；折线图每个点都是对三个 target family 的 recall 取平均后的结果，不是单独某一个 family 的结果。",
        "- `incremental_generation` 协议：测试第 `g` 代时，训练集中包含所有 family 的更早代 bot（`generation < g`）。",
        "- `heldout_family_same_generation` 协议：测试第 `g` 代某个 target family 时，训练集中只包含同代 `g` 的其他 family，不包含目标 family；它是同代跨 family 泛化，不是 zero-shot。",
        "",
        "## 漂移 lineage",
        "",
        lineage_df.to_markdown(index=False) if not lineage_df.empty else "_empty_",
        "",
        "## 文本泛化汇总",
        "",
    ]
    if not text_metrics.empty:
        text_summary = (
            text_metrics.groupby(["protocol", "model", "target_generation"])
            .agg(
                mean_target_bot_recall=("target_bot_recall", "mean"),
                mean_human_test_fpr=("human_test_fpr", "mean"),
                mean_target_bot_probability=("target_bot_mean_probability", "mean"),
            )
            .reset_index()
        )
        lines.append(text_summary.to_markdown(index=False))
    else:
        lines.append("_empty_")
    lines.extend(["", "## 图泛化汇总", ""])
    if not graph_metrics.empty:
        graph_summary = (
            graph_metrics.groupby(["protocol", "model", "target_generation"])
            .agg(
                mean_target_bot_recall=("target_bot_recall", "mean"),
                mean_human_test_fpr=("human_test_fpr", "mean"),
                mean_target_bot_probability=("target_bot_mean_probability", "mean"),
            )
            .reset_index()
        )
        lines.append(graph_summary.to_markdown(index=False))
    else:
        lines.append("_empty_")
    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- `selection_status=qualified` 表示该代至少有一个 attempt 通过 local gate；`best_rejected` 表示该代没有 attempt 完整通过 gate，但保留了最接近目标的样本用于连续漂移分析。",
            "- `target_pass_rate` 是相对上一代 parent 的 local drift 通过率；`cumulative_target_pass_rate` 是相对第 0 代 parent 的累计漂移通过率。",
            "- `incremental_generation` 看的是训练集只见过更早代 bot 时，对新一代的跨代泛化。",
            "- `heldout_family_same_generation` 看的是同一代内部，训练时没见过该 family 时的跨 family 泛化。",
            "- `target_bot_recall` 越高，说明 detector 越容易抓住该批新 bot；越低，说明泛化更弱。",
            "- `human_test_fpr` 越高，说明 detector 为了抓新 bot 开始误伤原始 human。",
        ]
    )
    (out_dir / "continuous_generalization_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    if (out_dir / "run_config.json").exists() and not args.clean:
        raise SystemExit("This output directory already contains a drift experiment. Choose a new directory, use tools/evaluate_saved_drift.py for evaluation, or explicitly use --clean to replace it. Appending generations is not supported.")
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_families = split_csv(args.families)
    unknown = [family for family in selected_families if family not in family_names()]
    if unknown:
        raise SystemExit(f"Unknown families: {unknown}")

    write_json(out_dir / "run_config.json", {"args": vars(args), "families": selected_families})
    all_records: list[RunRecord] = []
    for family in selected_families:
        all_records.extend(run_family_generations(family, out_dir, args))

    lineage_df = pd.DataFrame(
        [
            {
                "family": record.family,
                "generation": record.generation,
                "run_dir": str(record.run_dir),
                "plan_id": record.plan_id,
                "child_family_version": record.child_family_version,
                "selection_status": record.selection_status,
                "gate_passed": record.gate_passed,
                "target_axes": ",".join(record.target_axes),
                "target_pass_rate": record.target_pass_rate,
                "cumulative_target_pass_rate": record.cumulative_target_pass_rate,
                "severe_non_target_count": record.severe_non_target_count,
            }
            for record in all_records
        ]
    )
    lineage_df.to_csv(out_dir / "lineage_summary.csv", index=False)
    if args.skip_generalization:
        write_report(out_dir, lineage_df, pd.DataFrame(), pd.DataFrame(), args)
        return

    text_tfidf_metrics, _ = evaluate_text_tfidf(all_records, out_dir, args)
    text_bge_metrics, _ = evaluate_text_bge(all_records, out_dir, args)
    text_metrics = pd.concat([df for df in [text_tfidf_metrics, text_bge_metrics] if not df.empty], ignore_index=True)
    text_metrics.to_csv(out_dir / "text_generalization_metrics_all.csv", index=False)

    graph_metrics, _ = evaluate_graph_generalization(all_records, out_dir, args)
    graph_metrics.to_csv(out_dir / "graph_generalization_metrics_all.csv", index=False)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not text_metrics.empty:
        _line_plot(text_metrics, "incremental_generation", fig_dir / "text_incremental_generation_recall.png", "Text detectors: incremental-generation recall")
        _heatmap(text_metrics, "TFIDF_LR", "heldout_family_same_generation", fig_dir / "text_tfidf_heldout_family_heatmap.png", "TF-IDF held-out family recall")
        if (text_metrics["model"] == "BGE_LR").any():
            _heatmap(text_metrics, "BGE_LR", "heldout_family_same_generation", fig_dir / "text_bge_heldout_family_heatmap.png", "BGE held-out family recall")
    if not graph_metrics.empty:
        _line_plot(graph_metrics, "incremental_generation", fig_dir / "graph_incremental_generation_recall.png", "Graph detectors: incremental-generation recall")
        _heatmap(graph_metrics, "GraphStats_RGCN", "heldout_family_same_generation", fig_dir / "graph_rgcn_heldout_family_heatmap.png", "GraphStats RGCN held-out family recall")

    write_report(out_dir, lineage_df, text_metrics, graph_metrics, args)


if __name__ == "__main__":
    main()
