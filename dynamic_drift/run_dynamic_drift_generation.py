#!/usr/bin/env python3
"""Phase 1 dynamic drift generation and qualification runner."""

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
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REDDIT_DIR = PROJECT_ROOT / "RedditBotSim"
DATA_ROOT = PROJECT_ROOT
EMBEDDING_ROOT = PROJECT_ROOT / "baselines/text/modern_embedding_run"
RUNNER = REDDIT_DIR / "persona_social_runner.py"
ANALYSIS_DIR = PROJECT_ROOT / "baselines" / "text"
GRAPH_DIR = PROJECT_ROOT / "baselines" / "graph"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "phase1_qualified_drift"

for path in [REDDIT_DIR, ANALYSIS_DIR, GRAPH_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from comment_text_only_experiment import load_comment_docs, split_indices, strict_clean_comment, truncate_words  # noqa: E402
from modern_embedding_text_experiment import encode_docs, pick_device, slugify_model_name  # noqa: E402
from graph_only_bot_detection import Split as GraphSplit, graph_stat_features, run_ml_models as run_graph_ml_models, set_seed as set_graph_seed  # noqa: E402
from persona_social.alignment_gate import evaluate_alignment_gate  # noqa: E402
from persona_social.drift_auditors import audit_drift, audit_to_frames  # noqa: E402
from persona_social.drift_controller import (  # noqa: E402
    build_mutation_plan,
    build_parent_personas,
    family_names,
    filtered_model_pool,
    mutate_personas,
    write_manifest,
)
from persona_social.drift_manifest import write_attempt_artifacts, write_json, write_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--families", default=",".join(family_names()))
    parser.add_argument("--bots-per-family", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--browse-count", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--llm-timeout", type=float, default=90.0)
    parser.add_argument("--max-resample", type=int, default=3)
    parser.add_argument("--providers", default="openai,gemini")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--api", choices=["responses", "chat"], default="responses")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-detectors", action="store_true")
    parser.add_argument("--skip-bge", action="store_true")
    parser.add_argument("--bge-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--truncate-words", type=int, default=600)
    return parse_release_args(parser, globals(), "dynamic_drift")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def run_persona_generation(run_dir: Path, manifest_path: Path, args: argparse.Namespace) -> None:
    if args.skip_generation:
        return
    providers = split_csv(getattr(args, "providers", "openai")) or ["openai"]
    default_provider = providers[0]
    cmd = [
        sys.executable,
        str(RUNNER),
        "--run-dir",
        str(run_dir / "run"),
        "--config",
        str(args.runtime_config),
        "--max-agents",
        str(args.bots_per_family),
        "--rounds",
        str(args.rounds),
        "--browse-count",
        str(args.browse_count),
        "--concurrency",
        str(args.concurrency),
        "--retries",
        str(args.retries),
        "--llm-timeout",
        str(args.llm_timeout),
        "--api",
        args.api,
        "--provider",
        default_provider,
        "--reasoning-effort",
        args.reasoning_effort,
        "--seed",
        str(args.seed),
        "--clean",
        "--family-manifest",
        str(manifest_path),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(cmd, cwd=REDDIT_DIR, check=True, env=env)


def ensure_parent_run(family: str, family_dir: Path, args: argparse.Namespace) -> Path:
    parent_dir = family_dir / "parent_control"
    parent_manifest_path = parent_dir / "parent_manifest.json"
    if (parent_dir / "generated_comments.csv").exists() and (parent_dir / "persona_social_summary.json").exists():
        return parent_dir
    if parent_dir.exists():
        shutil.rmtree(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    parent_personas = build_parent_personas(family, args.bots_per_family, seed=args.seed, providers=split_csv(args.providers))
    write_manifest(parent_manifest_path, family, 0, parent_personas, plan=None)
    run_persona_generation(parent_dir, parent_manifest_path, args)
    return parent_dir


def generate_child_attempt(family: str, parent_dir: Path, family_dir: Path, attempt: int, args: argparse.Namespace) -> tuple[Path, Any, Any, Any]:
    providers = split_csv(args.providers)
    parent_manifest = json.loads((parent_dir / "parent_manifest.json").read_text(encoding="utf-8"))
    parent_personas_payload = parent_manifest.get("personas", [])
    from persona_social_runner import _coerce_persona  # type: ignore

    parent_personas = [_coerce_persona(item) for item in parent_personas_payload]
    plan = build_mutation_plan(
        family,
        args.generation,
        attempt,
        args.seed + attempt * 101,
        providers=providers,
        feedback={},
        parent_family_version="parent",
    )
    child_personas = mutate_personas(parent_personas, plan)
    attempt_dir = family_dir / f"attempt_{attempt:02d}_{plan.plan_id}"
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = attempt_dir / "child_manifest.json"
    write_manifest(manifest_path, plan.family, plan.generation, child_personas, plan=plan)
    write_json(family_dir / "mutation_plans" / f"{plan.slug}_attempt_{attempt:02d}.json", plan.to_dict())
    run_persona_generation(attempt_dir, manifest_path, args)
    audit = audit_drift(parent_dir, attempt_dir, plan)
    gate = evaluate_alignment_gate(attempt_dir, plan, audit)
    write_attempt_artifacts(attempt_dir, plan, audit, gate)
    metric_df, axis_df = audit_to_frames(audit)
    metric_df.to_csv(attempt_dir / "drift_audit.csv", index=False)
    axis_df.to_csv(attempt_dir / "drift_audit_by_axis.csv", index=False)
    return attempt_dir, plan, audit, gate


def choose_qualified_attempt(attempts: list[tuple[Path, Any, Any, Any]]) -> tuple[Path, Any, Any, Any] | None:
    passed = [item for item in attempts if item[3].passed]
    if passed:
        return passed[0]
    if not attempts:
        return None
    return max(attempts, key=lambda item: (item[2].target_pass_rate, -item[2].severe_non_target_count))


def clean_generated_doc(comments: pd.DataFrame, profiles: pd.DataFrame, truncate_words_n: int) -> pd.DataFrame:
    grouped = comments.groupby("user_id")["comment_body"].apply(list).to_dict() if not comments.empty else {}
    rows = []
    for _, profile in profiles.iterrows():
        bodies = grouped.get(profile["user_id"], [])
        parts = []
        for body in bodies:
            cleaned, reason = strict_clean_comment(body)
            if reason is None and cleaned:
                parts.append(cleaned)
        doc = truncate_words(" ".join(parts), truncate_words_n)
        rows.append(
            {
                "source": "new_ai_bot",
                "label": 1,
                "user_id": profile["user_id"],
                "family": profile.get("family", ""),
                "doc": doc,
                "comment_count": len(bodies),
                "word_count": len(doc.split()),
            }
        )
    return pd.DataFrame(rows)


def original_text_data(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    stats, _raw, _clean, strict_docs = load_comment_docs(DATA_ROOT)
    labels = stats["label"].to_numpy(dtype=int)
    rows = []
    for i, row in stats.iterrows():
        doc = truncate_words(strict_docs[i], args.truncate_words)
        rows.append(
            {
                "source": "original_bot" if labels[i] else "original_human",
                "label": int(labels[i]),
                "original_row": i,
                "user_id": str(row["user_id"]),
                "family": "",
                "doc": doc,
            }
        )
    df = pd.DataFrame(rows)
    train_idx, val_idx, test_idx = split_indices(labels, args.seed)
    return df, train_idx, val_idx, test_idx


def metrics_from_predictions(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"size": 0, "bot_count": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "bot_detected": 0, "bot_mean_probability": float("nan")}
    labels = df["label"].to_numpy(dtype=int) if "label" in df else df["true_label"].to_numpy(dtype=int)
    preds = df["pred_label"].to_numpy(dtype=int)
    probs = df["bot_probability"].to_numpy(dtype=float)
    bot_mask = labels == 1
    tp = int(((preds == 1) & bot_mask).sum())
    return {
        "size": int(len(df)),
        "bot_count": int(bot_mask.sum()),
        "accuracy": float((preds == labels).mean()),
        "precision": float(tp / max(1, int((preds == 1).sum()))),
        "recall": float(tp / max(1, int(bot_mask.sum()))),
        "bot_detected": int(preds[bot_mask].sum()) if bot_mask.any() else 0,
        "bot_mean_probability": float(probs[bot_mask].mean()) if bot_mask.any() else float("nan"),
    }


def run_tfidf_detector(qualified: list[tuple[str, Path]], out_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    original_df, train_idx, _val_idx, _test_idx = original_text_data(args)
    train_df = original_df.iloc[train_idx].copy()
    docs = train_df["doc"].fillna("").tolist()
    labels = train_df["label"].to_numpy(dtype=int)
    vectorizer = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.9, ngram_range=(1, 2), stop_words="english", lowercase=True, strip_accents="unicode")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    clf.fit(vectorizer.fit_transform(docs), labels)
    pred_frames = []
    metric_rows = []
    for family, run_dir in qualified:
        comments = pd.read_csv(run_dir / "generated_comments.csv", dtype=str).fillna("")
        profiles = pd.read_csv(run_dir / "generated_persona_profiles.csv", dtype=str).fillna("")
        eval_df = clean_generated_doc(comments, profiles, args.truncate_words)
        x = vectorizer.transform(eval_df["doc"].fillna("").tolist())
        eval_df["model"] = "tfidf"
        eval_df["bot_probability"] = clf.predict_proba(x)[:, 1]
        eval_df["pred_label"] = clf.predict(x)
        pred_frames.append(eval_df)
        metric_rows.append({"family": family, "model": "tfidf", **metrics_from_predictions(eval_df)})
    pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    pred_df.to_csv(out_dir / "tfidf_predictions.csv", index=False)
    return pred_df, metrics


def run_bge_detector(qualified: list[tuple[str, Path]], out_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.linear_model import LogisticRegression

    slug = slugify_model_name(args.bge_model)
    base_embedding_path = EMBEDDING_ROOT / slug / "strict_clean_truncated_embeddings.npy"
    if not base_embedding_path.exists() or args.skip_bge:
        return pd.DataFrame(), pd.DataFrame([{"model": "bge", "status": "skipped", "reason": "missing_base_embeddings_or_skip_bge"}])
    original_df, train_idx, _val_idx, _test_idx = original_text_data(args)
    base_embeddings = np.load(base_embedding_path)
    train_original = original_df.iloc[train_idx].copy()
    train_emb = base_embeddings[train_original["original_row"].to_numpy(dtype=int)]
    train_labels = train_original["label"].to_numpy(dtype=int)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(train_emb, train_labels)
    pred_frames = []
    metric_rows = []
    device = pick_device(args.device)
    for family, run_dir in qualified:
        comments = pd.read_csv(run_dir / "generated_comments.csv", dtype=str).fillna("")
        profiles = pd.read_csv(run_dir / "generated_persona_profiles.csv", dtype=str).fillna("")
        eval_df = clean_generated_doc(comments, profiles, args.truncate_words)
        cache_path = run_dir / "bge_eval_embeddings.npy"
        if cache_path.exists():
            eval_emb = np.load(cache_path)
        else:
            eval_emb = encode_docs(eval_df["doc"].fillna("").tolist(), model_name=args.bge_model, batch_size=args.batch_size, max_length=args.max_length, device=device, normalize=True)
            np.save(cache_path, eval_emb)
        eval_df["model"] = "bge"
        eval_df["bot_probability"] = clf.predict_proba(eval_emb)[:, 1]
        eval_df["pred_label"] = clf.predict(eval_emb)
        pred_frames.append(eval_df)
        metric_rows.append({"family": family, "model": "bge", **metrics_from_predictions(eval_df)})
    pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    pred_df.to_csv(out_dir / "bge_predictions.csv", index=False)
    return pred_df, metrics


def normalize_edge_index(edge_index: Any) -> Any:
    if edge_index.ndim == 2 and edge_index.shape[0] == 2 and edge_index.shape[1] != 2:
        return edge_index.T.contiguous()
    return edge_index.contiguous()


def build_augmented_graph(run_dirs: list[Path]) -> tuple[pd.DataFrame, np.ndarray, Any, Any, Any]:
    import torch
    from collections import Counter

    users = pd.read_csv(DATA_ROOT / "BotSim-24-Dataset" / "Users.csv", dtype=str).fillna("")
    labels = np.zeros(len(users), dtype=np.int64)
    labels[1907:] = 1
    node_rows = users[["user_id", "name"]].copy()
    node_rows["source"] = np.where(np.arange(len(node_rows)) >= 1907, "original_bot", "original_human")
    node_rows["family"] = ""
    node_rows["label"] = labels
    generated_nodes = []
    for run_dir in run_dirs:
        profiles = pd.read_csv(run_dir / "generated_persona_profiles.csv", dtype=str).fillna("")
        nodes = profiles[["user_id", "user_name", "family"]].drop_duplicates("user_id").rename(columns={"user_name": "name"})
        nodes["source"] = "new_ai_bot"
        nodes["label"] = 1
        generated_nodes.append(nodes)
    if generated_nodes:
        node_rows = pd.concat([node_rows, *generated_nodes], ignore_index=True)
    labels = node_rows["label"].to_numpy(dtype=np.int64)
    node_index = {uid: i for i, uid in enumerate(node_rows["user_id"].astype(str))}
    edge_index = normalize_edge_index(torch.load(DATA_ROOT / "BotSim-24-mini-sample" / "edge_index.pt", map_location="cpu", weights_only=True).long())
    edge_type = torch.load(DATA_ROOT / "BotSim-24-mini-sample" / "edge_type.pt", map_location="cpu", weights_only=True).long()
    edge_weight = torch.load(DATA_ROOT / "BotSim-24-mini-sample" / "edge_weight.pt", map_location="cpu", weights_only=True).reshape(-1).float()
    counter: Counter[tuple[int, int, int]] = Counter()

    def add_edge(source_user: str, target_user: str | None, rel: int) -> None:
        if source_user in node_index and target_user in node_index:
            counter[(node_index[source_user], node_index[str(target_user)], rel)] += 1

    for run_dir in run_dirs:
        data_dir = run_dir / "run" / "data"
        profiles = pd.read_csv(run_dir / "generated_persona_profiles.csv", dtype=str).fillna("")
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


def run_graphstats_detector(qualified: list[tuple[str, Path]], out_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not qualified:
        return pd.DataFrame(), pd.DataFrame()
    node_rows, labels, edge_index, edge_type, edge_weight = build_augmented_graph([run_dir for _family, run_dir in qualified])
    original_idx = node_rows.index[node_rows["source"] != "new_ai_bot"].to_numpy()
    orig_labels = labels[original_idx]
    orig_train_rel, orig_val_rel, _orig_test_rel = split_indices(orig_labels, args.seed)
    generated_idx = node_rows.index[node_rows["source"] == "new_ai_bot"].to_numpy()
    split = GraphSplit(train_idx=original_idx[orig_train_rel], val_idx=original_idx[orig_val_rel], test_idx=generated_idx)
    set_graph_seed(args.seed)
    features = graph_stat_features(edge_index, edge_type, edge_weight, len(labels))
    graph_dir = out_dir / "graphstats"
    graph_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(graph_dir / "graph_stat_features.csv", index_label="node_id")
    node_rows.to_csv(graph_dir / "graph_nodes.csv", index_label="node_id")
    _rows, probs = run_graph_ml_models(features, labels, split, args.seed)
    metric_rows = []
    pred_rows = []
    for model_name, prob in probs.items():
        if model_name not in {"GraphStats_LR", "GraphStats_RF"}:
            continue
        sub = node_rows.iloc[split.test_idx].copy()
        sub["model"] = model_name
        sub["bot_probability"] = prob[split.test_idx]
        sub["pred_label"] = (sub["bot_probability"].to_numpy(dtype=float) >= 0.5).astype(int)
        sub["true_label"] = labels[split.test_idx]
        sub["label"] = sub["true_label"]
        pred_rows.append(sub)
        for family, fam_sub in sub.groupby("family"):
            metric_rows.append({"family": family, "model": model_name, **metrics_from_predictions(fam_sub)})
    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    pred_df.to_csv(out_dir / "graphstats_predictions.csv", index=False)
    return pred_df, metrics


def run_detectors(qualified: list[tuple[str, Path]], out_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    if args.skip_detectors or not qualified:
        return pd.DataFrame()
    metric_frames = []
    _tfidf_pred, tfidf_metrics = run_tfidf_detector(qualified, out_dir, args)
    metric_frames.append(tfidf_metrics)
    _bge_pred, bge_metrics = run_bge_detector(qualified, out_dir, args)
    metric_frames.append(bge_metrics)
    _graph_pred, graph_metrics = run_graphstats_detector(qualified, out_dir, args)
    metric_frames.append(graph_metrics)
    detector_summary = pd.concat([df for df in metric_frames if not df.empty], ignore_index=True) if metric_frames else pd.DataFrame()
    detector_summary.to_csv(out_dir / "detector_summary.csv", index=False)
    return detector_summary


def write_phase_report(out_dir: Path, qualified_rows: list[dict[str, Any]], all_axis: pd.DataFrame, all_gate: pd.DataFrame, detector_summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Phase 1 合格漂移机器人生成框架报告",
        "",
        f"- bots_per_family: `{args.bots_per_family}`",
        f"- rounds: `{args.rounds}`",
        f"- browse_count: `{args.browse_count}`",
        f"- max_resample: `{args.max_resample}`",
        f"- dry_run: `{args.dry_run}`",
        f"- providers: `{args.providers}`",
        "",
        "## 合格 family",
        "",
    ]
    if qualified_rows:
        lines.append(pd.DataFrame(qualified_rows).to_markdown(index=False))
    else:
        lines.append("_没有 family 完全通过 gate；请查看 rejected attempts。_")
    lines.extend(["", "## 漂移轴审计", ""])
    lines.append(all_axis.to_markdown(index=False) if not all_axis.empty else "_empty_")
    lines.extend(["", "## Gate 结果", ""])
    lines.append(all_gate.to_markdown(index=False) if not all_gate.empty else "_empty_")
    lines.extend(["", "## 基础检测", ""])
    lines.append(detector_summary.to_markdown(index=False) if not detector_summary.empty else "_skipped or empty_")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 漂移是否合格由统计审计和 human gate 决定，不由检测器概率决定。",
            "- provider/model/temperature 只用于分组分析，不进入文本或图检测特征。",
            "- 被拒绝的 attempt 会保留在 family 目录下，用于分析失败原因。",
        ]
    )
    write_report(out_dir / "phase1_qualified_drift_report_cn.md", lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "run_config.json",
        {
            "args": vars(args),
            "model_pool": filtered_model_pool(split_csv(args.providers)),
            "families": split_csv(args.families),
        },
    )
    qualified: list[tuple[str, Path]] = []
    qualified_rows: list[dict[str, Any]] = []
    axis_frames = []
    metric_frames = []
    gate_rows = []
    for family in split_csv(args.families):
        family_dir = out_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        parent_dir = ensure_parent_run(family, family_dir, args)
        attempts = []
        for attempt in range(1, args.max_resample + 1):
            attempt_tuple = generate_child_attempt(family, parent_dir, family_dir, attempt, args)
            attempts.append(attempt_tuple)
            attempt_dir, plan, audit, gate = attempt_tuple
            metric_df, axis_df = audit_to_frames(audit)
            metric_frames.append(metric_df)
            axis_frames.append(axis_df)
            gate_rows.append(gate.to_dict())
            if gate.passed:
                break
        chosen = choose_qualified_attempt(attempts)
        if chosen is None:
            continue
        chosen_dir, plan, audit, gate = chosen
        status = "qualified" if gate.passed else "best_rejected"
        if gate.passed:
            qualified.append((plan.family, chosen_dir))
            qualified_root = family_dir / "qualified"
            if qualified_root.exists():
                shutil.rmtree(qualified_root)
            shutil.copytree(chosen_dir, qualified_root)
        qualified_rows.append(
            {
                "family": plan.family,
                "status": status,
                "attempt": plan.attempt,
                "plan_id": plan.plan_id,
                "target_axes": ",".join(plan.target_axes),
                "target_pass_rate": audit.target_pass_rate,
                "severe_non_target_count": audit.severe_non_target_count,
                "axis_scores": json.dumps(audit.axis_scores, ensure_ascii=False, sort_keys=True),
                "run_dir": str(chosen_dir),
            }
        )
    all_axis = pd.concat(axis_frames, ignore_index=True) if axis_frames else pd.DataFrame()
    all_metric = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    all_gate = pd.DataFrame(gate_rows)
    pd.DataFrame(qualified_rows).to_csv(out_dir / "qualified_family_manifest.csv", index=False)
    all_axis.to_csv(out_dir / "drift_audit_by_axis.csv", index=False)
    all_metric.to_csv(out_dir / "drift_audit_by_family.csv", index=False)
    all_gate.to_csv(out_dir / "alignment_gate_results.csv", index=False)
    detector_summary = run_detectors(qualified, out_dir, args)
    write_phase_report(out_dir, qualified_rows, all_axis, all_gate, detector_summary, args)


if __name__ == "__main__":
    main()
