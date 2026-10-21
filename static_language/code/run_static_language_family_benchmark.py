#!/usr/bin/env python3
"""Static language-family benchmark for TF-IDF/BGE generalization."""

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
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
REDDIT_DIR = PROJECT_ROOT / "RedditBotSim"
DATA_ROOT = PROJECT_ROOT
EMBEDDING_ROOT = PROJECT_ROOT / "baselines/text/modern_embedding_run"
ANALYSIS_DIR = PROJECT_ROOT / "baselines" / "text"

for path in [REDDIT_DIR, ANALYSIS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from comment_text_only_experiment import load_comment_docs, split_indices, strict_clean_comment, truncate_words  # noqa: E402
from llm_client import build_llm_client  # noqa: E402
from modern_embedding_text_experiment import encode_docs, pick_device, slugify_model_name  # noqa: E402
from persona_social.anchor_sampler import (  # noqa: E402
    build_anchor_profile_blocks,
    build_human_anchor_pool,
    export_anchor_manifest,
    sample_anchor_bindings,
)
from persona_social.language_audit import audit_documents, centroid_distances, summarize_by_family  # noqa: E402
from persona_social.local_tfidf_rewriter import diagnose_detector_exposure, run_llm_revision_loop, run_revision_loop  # noqa: E402
from persona_social.static_language_families import (  # noqa: E402
    FAMILY_DISPLAY_NAMES,
    STATIC_LANGUAGE_FAMILIES,
    build_static_language_manifest,
    build_static_language_personas,
    export_persona_manifest_csv,
    export_persona_manifest_json,
    normalize_family_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "outputs" / "smoke_5bot"))
    parser.add_argument("--families", default="F1,F2,F3,F4,F5")
    parser.add_argument("--bots-per-family", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--browse-count", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--llm-timeout", type=float, default=90.0)
    parser.add_argument("--api", choices=["responses", "chat"], default="responses")
    parser.add_argument("--openai-model", default="gpt-5.4")
    parser.add_argument("--gemini-model", default="gemini-3-pro-preview")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-bge", action="store_true")
    parser.add_argument("--recompute-bge", action="store_true")
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--truncate-words", type=int, default=600)
    parser.add_argument("--bge-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--anchor-iters", type=int, default=2)
    parser.add_argument("--f3-revision-mode", choices=["llm", "local"], default="llm")
    parser.add_argument("--clean", action="store_true")
    return parse_release_args(parser, globals(), "static_language")


def family_short_id(family_id: str) -> str:
    for short, full in STATIC_LANGUAGE_FAMILIES.items():
        if full == family_id:
            return short
    return family_id


def requested_families(raw: str) -> list[str]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(normalize_family_id(part))
    return out


def train_tfidf(docs: list[str], labels: np.ndarray, train_idx: np.ndarray, max_features: int):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
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


def train_bge(embeddings: np.ndarray, labels: np.ndarray, train_idx: np.ndarray):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(embeddings[train_idx], labels[train_idx])
    return clf


def predict_model(model_name: str, model_obj: Any, docs: list[str], embeddings: np.ndarray | None, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "tfidf":
        vectorizer, clf = model_obj
        x = vectorizer.transform([docs[i] for i in idx])
        return clf.predict(x), clf.predict_proba(x)[:, 1]
    if embeddings is None:
        raise ValueError("BGE embeddings are missing")
    clf = model_obj
    return clf.predict(embeddings[idx]), clf.predict_proba(embeddings[idx])[:, 1]


def metrics(y_true: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    bot_mask = y_true == 1
    return {
        "size": int(len(y_true)),
        "bot_count": int(bot_mask.sum()),
        "accuracy": float(accuracy_score(y_true, pred)) if len(y_true) else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan"),
        "recall": float(recall_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan"),
        "bot_detected": int(pred[bot_mask].sum()) if int(bot_mask.sum()) else 0,
        "bot_mean_probability": float(prob[bot_mask].mean()) if int(bot_mask.sum()) else float("nan"),
        "bot_median_probability": float(np.median(prob[bot_mask])) if int(bot_mask.sum()) else float("nan"),
    }


def build_original_rows(truncate_words_n: int) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stats, _raw_docs, _clean_docs, strict_docs = load_comment_docs(DATA_ROOT)
    docs = [truncate_words(doc, truncate_words_n) for doc in strict_docs]
    labels = stats["label"].to_numpy(dtype=int)
    rows = []
    for i, row in stats.iterrows():
        source = "original_bot" if labels[i] else "original_human"
        rows.append(
            {
                "row_kind": source,
                "source": source,
                "family": source,
                "family_display": source,
                "label": int(labels[i]),
                "user_id": str(row["user_id"]),
                "user_name": str(row["user_id"]),
                "provider": "",
                "model": "",
                "temperature": "",
                "comment_count": int(row["strict_kept_comment_count"]),
                "word_count": len(docs[i].split()),
                "doc": docs[i],
            }
        )
    train_idx, val_idx, test_idx = split_indices(labels, 0)
    return pd.DataFrame(rows), docs, labels, train_idx, val_idx, test_idx


def anchor_rows_from_original(original_df: pd.DataFrame, original_train_idx: np.ndarray) -> list[dict[str, Any]]:
    train_set = set(int(i) for i in original_train_idx)
    rows = []
    for i, row in original_df.iterrows():
        if int(i) not in train_set or int(row["label"]) != 0:
            continue
        rows.append(
            {
                "user_id": row["user_id"],
                "doc": row["doc"],
                "split": "train",
                "label": "human",
                "source": "human",
                "topic": row.get("family", ""),
            }
        )
    return rows


def write_family_manifest(output_dir: Path, family_id: str, args: argparse.Namespace, anchor_blocks: dict[str, str] | None = None) -> Path:
    family_dir = output_dir / "families" / family_id
    family_dir.mkdir(parents=True, exist_ok=True)
    personas, manifest_rows = build_static_language_personas(
        family_id,
        args.bots_per_family,
        seed=args.seed,
        anchor_blocks=anchor_blocks,
        openai_model=args.openai_model,
        gemini_model=args.gemini_model,
    )
    export_persona_manifest_csv(manifest_rows, family_dir / "static_persona_manifest.csv")
    export_persona_manifest_json(personas, family_dir / "family_manifest.json", family=family_id, manifest_rows=manifest_rows)
    full_manifest = build_static_language_manifest(
        family_id,
        args.bots_per_family,
        seed=args.seed,
        anchor_blocks=anchor_blocks,
        openai_model=args.openai_model,
        gemini_model=args.gemini_model,
    )
    (family_dir / "static_family_manifest_full.json").write_text(json.dumps(full_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return family_dir / "family_manifest.json"


def run_generation_for_family(output_dir: Path, family_id: str, manifest_path: Path, args: argparse.Namespace) -> None:
    if args.skip_generation:
        return
    family_dir = output_dir / "families" / family_id
    run_dir = family_dir / "run"
    if args.clean and run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        sys.executable,
        str(REDDIT_DIR / "persona_social_runner.py"),
        "--run-dir",
        str(run_dir),
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
        "gemini" if family_id == "F5_gemini_trait_prompt" else "openai",
        "--model",
        args.gemini_model if family_id == "F5_gemini_trait_prompt" else args.openai_model,
        "--reasoning-effort",
        "medium",
        "--seed",
        str(args.seed),
        "--family-manifest",
        str(manifest_path.resolve()),
        "--clean",
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    subprocess.run(cmd, cwd=REDDIT_DIR, check=True)


def load_anchor_doc_map(bindings: list[Any], pool: list[Any]) -> dict[str, list[str]]:
    pool_map = {anchor.user_id: anchor.doc for anchor in pool}
    out: dict[str, list[str]] = {}
    for binding in bindings:
        out[binding.bot_user_id] = [pool_map[uid] for uid in binding.anchor_user_ids if uid in pool_map]
    return out


def apply_f3_revisions(
    family_dir: Path,
    anchor_doc_map: dict[str, list[str]],
    args: argparse.Namespace,
    *,
    detector: tuple[Any, Any] | None = None,
) -> None:
    comments_path = family_dir / "generated_comments.csv"
    if not comments_path.exists() or comments_path.stat().st_size == 0:
        return
    try:
        comments = pd.read_csv(comments_path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return
    if comments.empty:
        return
    mode = "local" if args.dry_run else args.f3_revision_mode
    trace_name = "local_tfidf_revision_trace.jsonl" if mode == "local" else "llm_tfidf_revision_trace.jsonl"
    trace_path = family_dir / trace_name
    if trace_path.exists():
        trace_path.unlink()
    llm_client = None
    llm_settings = None
    if mode == "llm":
        llm_client, llm_settings = build_llm_client(
            None,
            provider="openai",
            model=args.openai_model,
        )
    revised_by_id: dict[str, str] = {}
    rows = []
    for _, row in comments.iterrows():
        user_id = str(row["user_id"])
        original = str(row["comment_body"])
        final = original
        changed = False
        iterations = 0
        before_probability = None
        after_probability = None
        if detector is not None and original.strip():
            before_probability = diagnose_detector_exposure(detector[0], detector[1], original).get("bot_probability")
        if user_id in anchor_doc_map and original.strip():
            metadata = {
                "comment_id": row.get("comment_id", ""),
                "level": row.get("level", ""),
                "subreddit": row.get("subreddit", ""),
                "parent_id": row.get("parent_id", ""),
                "link_id": row.get("link_id", ""),
            }
            if mode == "llm" and llm_client is not None:
                result = run_llm_revision_loop(
                    anchor_doc_map[user_id],
                    original,
                    llm_client=llm_client,
                    detector=detector,
                    max_iters=args.anchor_iters,
                    trace_path=trace_path,
                    api=args.api,
                    settings_override={
                        "provider": "openai",
                        "model": args.openai_model,
                        "temperature": 0.75,
                        "reasoning_effort": "medium",
                    },
                    metadata=metadata,
                )
                exposure = result.get("final_detector_exposure") or {}
                after_probability = exposure.get("bot_probability")
            else:
                result = run_revision_loop(anchor_doc_map[user_id], original, max_iters=args.anchor_iters, trace_path=trace_path)
            final = result["final_text"]
            changed = bool(result["changed"])
            iterations = int(result.get("iterations", 0))
        if detector is not None and final.strip() and after_probability is None:
            after_probability = diagnose_detector_exposure(detector[0], detector[1], final).get("bot_probability")
        revised_by_id[str(row["comment_id"])] = final
        rows.append(
            {
                **row.to_dict(),
                "comment_body": final,
                "f3_revision_mode": mode,
                "f3_revised": changed,
                "f3_revision_iters": iterations,
                "f3_detector_prob_before": before_probability,
                "f3_detector_prob_after": after_probability,
            }
        )
    pd.DataFrame(rows).to_csv(comments_path, index=False)
    for rel in ["run/data/Comment_1.csv", "run/data/Comment_2.csv"]:
        path = family_dir / rel
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df["comment_body"] = [revised_by_id.get(str(cid), body) for cid, body in zip(df["comment_id"], df["comment_body"])]
        df.to_csv(path, index=False)


def load_family_docs(family_dir: Path, family_id: str, truncate_words_n: int) -> pd.DataFrame:
    comments = read_generated_comments(family_dir / "generated_comments.csv")
    profiles = pd.read_csv(family_dir / "generated_persona_profiles.csv", dtype=str).fillna("")
    grouped = comments.groupby("user_id")["comment_body"].apply(list).to_dict()
    rows = []
    for _, profile in profiles.iterrows():
        bodies = grouped.get(profile["user_id"], [])
        strict_parts = []
        for body in bodies:
            cleaned, reason = strict_clean_comment(body)
            if reason is None and cleaned:
                strict_parts.append(cleaned)
        doc = truncate_words(" ".join(strict_parts), truncate_words_n)
        rows.append(
            {
                "row_kind": "static_language_bot",
                "source": "static_language_bot",
                "family": family_id,
                "family_display": FAMILY_DISPLAY_NAMES.get(family_id, family_id),
                "label": 1,
                "user_id": profile["user_id"],
                "user_name": profile["user_name"],
                "provider": profile.get("llm_provider", ""),
                "model": profile.get("llm_model", ""),
                "temperature": profile.get("llm_temperature", ""),
                "comment_count": len(bodies),
                "word_count": len(doc.split()),
                "doc": doc,
            }
        )
    return pd.DataFrame(rows)


def read_generated_comments(path: Path) -> pd.DataFrame:
    columns = ["level", "user_id", "user_name", "comment_id", "link_id", "parent_id", "subreddit", "created_utc", "comment_body"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df


def read_generated_posts(path: Path) -> pd.DataFrame:
    columns = ["user_id", "user_name", "post_id", "subreddit", "created_utc", "score", "num_comments", "upvote_ratio", "post_body"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df


def combine_family_outputs(output_dir: Path, families: list[str], truncate_words_n: int) -> pd.DataFrame:
    frames = []
    profile_frames = []
    comment_frames = []
    post_frames = []
    for family_id in families:
        family_dir = output_dir / "families" / family_id
        frames.append(load_family_docs(family_dir, family_id, truncate_words_n))
        p = pd.read_csv(family_dir / "generated_persona_profiles.csv", dtype=str).fillna("")
        c = read_generated_comments(family_dir / "generated_comments.csv")
        posts = read_generated_posts(family_dir / "generated_posts.csv")
        p["family"] = family_id
        c["family"] = family_id
        posts["family"] = family_id
        profile_frames.append(p)
        comment_frames.append(c)
        post_frames.append(posts)
    docs = pd.concat(frames, ignore_index=True)
    pd.concat(profile_frames, ignore_index=True).to_csv(output_dir / "generated_profiles_all.csv", index=False)
    pd.concat(comment_frames, ignore_index=True).to_csv(output_dir / "generated_comments_all.csv", index=False)
    pd.concat(post_frames, ignore_index=True).to_csv(output_dir / "generated_posts_all.csv", index=False)
    return docs


def get_embeddings(combined: pd.DataFrame, original_count: int, args: argparse.Namespace, output_dir: Path) -> np.ndarray | None:
    if args.skip_bge:
        return None
    slug = slugify_model_name(args.bge_model)
    base_embedding_path = EMBEDDING_ROOT / slug / "strict_clean_truncated_embeddings.npy"
    if not base_embedding_path.exists():
        print(f"[warn] missing base BGE embeddings: {base_embedding_path}")
        return None
    base_embeddings = np.load(base_embedding_path)
    new_path = output_dir / f"static_language_{slug}_generated_embeddings.npy"
    new_docs = combined.iloc[original_count:]["doc"].fillna("").tolist()
    if new_path.exists() and not args.recompute_bge:
        new_embeddings = np.load(new_path)
        if len(new_embeddings) != len(new_docs):
            new_embeddings = None
    else:
        new_embeddings = None
    if new_embeddings is None:
        device = pick_device(args.device)
        new_embeddings = encode_docs(
            new_docs,
            model_name=args.bge_model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            normalize=True,
        )
        np.save(new_path, new_embeddings)
    embeddings = np.vstack([base_embeddings, new_embeddings])
    np.save(output_dir / "combined_bge_embeddings.npy", embeddings)
    return embeddings


def fit_models(docs: list[str], labels: np.ndarray, train_idx: np.ndarray, embeddings: np.ndarray | None, args: argparse.Namespace) -> dict[str, Any]:
    models = {"tfidf": train_tfidf(docs, labels, train_idx, args.max_features)}
    if embeddings is not None:
        models["bge"] = train_bge(embeddings, labels, train_idx)
    return models


def eval_models(
    protocol: str,
    train_family: str,
    eval_family: str,
    models: dict[str, Any],
    combined: pd.DataFrame,
    docs: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray | None,
    eval_idx: np.ndarray,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    metric_rows = []
    pred_frames = []
    for model_name, model_obj in models.items():
        pred, prob = predict_model(model_name, model_obj, docs, embeddings, eval_idx)
        metric_rows.append(
            {
                "protocol": protocol,
                "train_family": train_family,
                "eval_family": eval_family,
                "model": model_name,
                **metrics(labels[eval_idx], pred, prob),
            }
        )
        pred_frames.append(
            pd.DataFrame(
                {
                    "protocol": protocol,
                    "train_family": train_family,
                    "eval_family": eval_family,
                    "model": model_name,
                    "row": eval_idx.astype(int),
                    "true_label": labels[eval_idx].astype(int),
                    "pred_label": pred.astype(int),
                    "bot_probability": prob.astype(float),
                }
            ).merge(combined.reset_index().rename(columns={"index": "row"}), on="row", how="left")
        )
    return metric_rows, pred_frames


def run_protocols(
    combined: pd.DataFrame,
    docs: list[str],
    labels: np.ndarray,
    original_train_idx: np.ndarray,
    original_val_idx: np.ndarray,
    original_test_idx: np.ndarray,
    family_indices: dict[str, np.ndarray],
    families: list[str],
    embeddings: np.ndarray | None,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []

    models = fit_models(docs, labels, original_train_idx, embeddings, args)
    for family_id, idx in family_indices.items():
        rows, preds = eval_models("original_only_zero_shot", "original_only", family_id, models, combined, docs, labels, embeddings, idx)
        metric_rows.extend(rows)
        pred_frames.extend(preds)

    for train_family in families:
        train_idx = np.concatenate([original_train_idx, family_indices[train_family]])
        models = fit_models(docs, labels, train_idx, embeddings, args)
        for eval_family in families:
            rows, preds = eval_models("single_family_transfer", train_family, eval_family, models, combined, docs, labels, embeddings, family_indices[eval_family])
            metric_rows.extend(rows)
            pred_frames.extend(preds)

    for holdout_family in families:
        train_parts = [original_train_idx] + [family_indices[f] for f in families if f != holdout_family]
        models = fit_models(docs, labels, np.concatenate(train_parts), embeddings, args)
        rows, preds = eval_models("leave_one_family_out", "all_except_" + holdout_family, holdout_family, models, combined, docs, labels, embeddings, family_indices[holdout_family])
        metric_rows.extend(rows)
        pred_frames.extend(preds)

    for stage in range(len(families)):
        seen = families[:stage]
        unseen = families[stage:]
        train_parts = [original_train_idx] + [family_indices[f] for f in seen]
        models = fit_models(docs, labels, np.concatenate(train_parts), embeddings, args)
        train_name = "original" if not seen else "original+" + "+".join(seen)
        for eval_family in unseen:
            rows, preds = eval_models("incremental_curriculum", train_name, eval_family, models, combined, docs, labels, embeddings, family_indices[eval_family])
            metric_rows.extend(rows)
            pred_frames.extend(preds)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(pred_frames, ignore_index=True)
    metrics_df.to_csv(output_dir / "static_language_family_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "static_language_family_predictions.csv", index=False)
    for protocol, name in [
        ("original_only_zero_shot", "original_only_zero_shot_matrix.csv"),
        ("single_family_transfer", "single_family_transfer_matrix.csv"),
        ("leave_one_family_out", "leave_one_family_out_matrix.csv"),
        ("incremental_curriculum", "incremental_curriculum_matrix.csv"),
    ]:
        metrics_df[metrics_df["protocol"] == protocol].to_csv(output_dir / name, index=False)
    return metrics_df, predictions_df


def write_report(output_dir: Path, families: list[str], metrics_df: pd.DataFrame, audit_summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# 静态语言 Family 泛化基准报告",
        "",
        "## 1. 实验设置",
        "",
        f"- family 数量：{len(families)}",
        f"- 每个 family bot 数：{args.bots_per_family}",
        f"- OpenAI-compatible 默认模型：`{args.openai_model}`",
        f"- Gemini-compatible 默认模型：`{args.gemini_model}`",
        f"- dry_run：`{args.dry_run}`",
        f"- BGE：`{'skipped' if args.skip_bge else args.bge_model}`",
        "",
        "## 2. Family 定义",
        "",
    ]
    for family in families:
        lines.append(f"- `{family}`：{FAMILY_DISPLAY_NAMES.get(family, family)}")
    lines += ["", "## 3. 核心泛化结果", ""]
    if not metrics_df.empty:
        pivot = metrics_df.pivot_table(index=["protocol", "eval_family"], columns="model", values="recall", aggfunc="mean").reset_index()
        lines.append(pivot.to_markdown(index=False))
    lines += ["", "## 4. 语言审计摘要", ""]
    if not audit_summary.empty:
        lines.append(audit_summary.to_markdown(index=False))
    lines += [
        "",
        "## 5. 解读口径",
        "",
        "- `original_only_zero_shot` 测旧检测器对新 family 的零样本泛化。",
        "- `single_family_transfer` 测见过某个新 family 后，能否迁移到其他 family。",
        "- `leave_one_family_out` 测见过多数 family 后，对完整未见 family 的检出能力。",
        "- `incremental_curriculum` 测检测器逐步吸收 family 后，对后续未见 family 的恢复曲线。",
        "- F3 的局部 TF-IDF 只使用 2-3 个 human anchor 与当前 bot draft，不使用全局 bot/human 标签。",
    ]
    (output_dir / "STATIC_LANGUAGE_FAMILY_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    families = requested_families(args.families)

    original_df, original_docs, original_labels, orig_train, orig_val, orig_test = build_original_rows(args.truncate_words)
    original_tfidf_detector = train_tfidf(original_docs, original_labels, orig_train, args.max_features)
    anchor_pool = build_human_anchor_pool(anchor_rows_from_original(original_df, orig_train), min_words=20)

    all_anchor_manifest_rows = []
    f3_anchor_doc_map: dict[str, list[str]] = {}
    for family_id in families:
        anchor_blocks = None
        if family_id == "F3_human_anchor_local_tfidf":
            draft_personas, _ = build_static_language_personas(
                family_id,
                args.bots_per_family,
                seed=args.seed,
                openai_model=args.openai_model,
                gemini_model=args.gemini_model,
            )
            bindings = sample_anchor_bindings([p.user_id for p in draft_personas], anchor_pool, seed=args.seed)
            export_anchor_manifest(bindings, output_dir / "anchor_manifest.csv")
            all_anchor_manifest_rows.extend(binding.to_manifest_row() for binding in bindings)
            anchor_blocks = build_anchor_profile_blocks(bindings)
            f3_anchor_doc_map = load_anchor_doc_map(bindings, anchor_pool)
        manifest_path = write_family_manifest(output_dir, family_id, args, anchor_blocks=anchor_blocks)
        run_generation_for_family(output_dir, family_id, manifest_path, args)
        if family_id == "F3_human_anchor_local_tfidf":
            apply_f3_revisions(output_dir / "families" / family_id, f3_anchor_doc_map, args, detector=original_tfidf_detector)

    if all_anchor_manifest_rows and not (output_dir / "anchor_manifest.csv").exists():
        pd.DataFrame(all_anchor_manifest_rows).to_csv(output_dir / "anchor_manifest.csv", index=False)

    generated_docs = combine_family_outputs(output_dir, families, args.truncate_words)
    combined = pd.concat([original_df, generated_docs], ignore_index=True)
    combined.to_csv(output_dir / "combined_sample_index.csv", index_label="row")
    docs = combined["doc"].fillna("").tolist()
    labels = combined["label"].to_numpy(dtype=int)
    original_count = len(original_df)
    family_indices = {
        family_id: combined.index[combined["family"] == family_id].to_numpy(dtype=int)
        for family_id in families
    }

    audit_df = audit_documents(generated_docs.to_dict(orient="records"))
    audit_df.to_csv(output_dir / "language_audit_by_user.csv", index=False)
    audit_summary = summarize_by_family(audit_df)
    audit_summary.to_csv(output_dir / "language_audit_by_family.csv", index=False)

    embeddings = get_embeddings(combined, original_count, args, output_dir)
    if embeddings is not None:
        human_centroid = embeddings[np.where(labels[:original_count] == 0)[0]].mean(axis=0)
        bot_centroid = embeddings[np.where(labels[:original_count] == 1)[0]].mean(axis=0)
        new_embeddings = embeddings[original_count:]
        centroid_distances(new_embeddings, generated_docs["family"].tolist(), human_centroid, bot_centroid).to_csv(
            output_dir / "bge_centroid_distance_by_family.csv",
            index=False,
        )

    metrics_df, _predictions_df = run_protocols(
        combined,
        docs,
        labels,
        orig_train,
        orig_val,
        orig_test,
        family_indices,
        families,
        embeddings,
        args,
        output_dir,
    )
    summary = {
        "families": families,
        "bots_per_family": args.bots_per_family,
        "total_generated_bots": int(len(generated_docs)),
        "dry_run": args.dry_run,
        "openai_model": args.openai_model,
        "gemini_model": args.gemini_model,
        "bge_enabled": embeddings is not None,
        "output_dir": str(output_dir),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, families, metrics_df, audit_summary, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
