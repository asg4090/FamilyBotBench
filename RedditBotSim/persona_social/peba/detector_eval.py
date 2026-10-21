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
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
except Exception as exc:  # optional local detector dependency
    TfidfVectorizer = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
    "you",
}


def _round(value: Any, digits: int = 4) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(parsed) or math.isinf(parsed):
        return 0.0
    return round(parsed, digits)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _text_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _load_generated_text_rows(iteration_dir: Path) -> pd.DataFrame:
    comments = _read_csv(iteration_dir / "generated_comments.csv")
    posts = _read_csv(iteration_dir / "generated_posts.csv")
    rows: list[dict[str, Any]] = []
    comment_col = _text_column(comments, ["comment_body", "text_body", "body", "text"])
    if comment_col:
        for idx, row in comments.iterrows():
            rows.append(
                {
                    "source": "comment",
                    "user_id": row.get("user_id", row.get("comment_author_id", "")),
                    "text_id": row.get("comment_id", idx),
                    "text": row.get(comment_col, ""),
                }
            )
    post_col = _text_column(posts, ["post_body", "posts", "text_body", "body", "text"])
    if post_col:
        for idx, row in posts.iterrows():
            rows.append(
                {
                    "source": "post",
                    "user_id": row.get("user_id", row.get("author_id", "")),
                    "text_id": row.get("post_id", row.get("submission_id", idx)),
                    "text": row.get(post_col, ""),
                }
            )
    return pd.DataFrame(rows)


def _nonempty_texts(rows: pd.DataFrame, sample_size: int = 400) -> list[str]:
    if rows.empty or "text" not in rows:
        return []
    texts = [str(text).strip() for text in rows["text"].tolist() if str(text).strip()]
    if len(texts) <= sample_size:
        return texts
    step = len(texts) / sample_size
    return [texts[int(i * step)] for i in range(sample_size)]


def _basic_text_summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"status": "skipped", "reason": "missing generated_comments/posts text rows", "n": 0}
    texts = rows["text"].astype(str).tolist()
    tokens_by_doc = [[m.group(0).lower() for m in TOKEN_RE.finditer(text)] for text in texts]
    word_counts = [len(tokens) for tokens in tokens_by_doc]
    normalised = [" ".join(str(text).lower().split()) for text in texts if str(text).strip()]
    duplicate_count = sum(count - 1 for count in pd.Series(normalised).value_counts().tolist() if count > 1)
    terms: dict[str, int] = {}
    for tokens in tokens_by_doc:
        for token in tokens:
            if token not in STOPWORDS and len(token) > 2:
                terms[token] = terms.get(token, 0) + 1
    return {
        "status": "ok",
        "n": int(len(rows)),
        "source_counts": rows["source"].value_counts().to_dict() if "source" in rows else {},
        "user_count": int(rows["user_id"].nunique()) if "user_id" in rows else 0,
        "mean_words": _round(np.mean(word_counts) if word_counts else 0.0),
        "median_words": _round(np.median(word_counts) if word_counts else 0.0),
        "duplicate_document_count": int(duplicate_count),
        "duplicate_document_share": _round(duplicate_count / max(1, len(rows))),
        "top_terms": sorted([{"term": key, "count": value} for key, value in terms.items()], key=lambda item: item["count"], reverse=True)[:12],
    }


def _load_generated_docs(iteration_dir: Path) -> pd.DataFrame:
    comments = _read_csv(iteration_dir / "generated_comments.csv")
    profiles = _read_csv(iteration_dir / "generated_persona_profiles.csv")
    grouped = comments.groupby("user_id")["comment_body"].apply(lambda values: " ".join(map(str, values))).to_dict() if not comments.empty else {}
    rows = []
    for _, row in profiles.iterrows():
        user_id = str(row.get("user_id", ""))
        doc = grouped.get(user_id, "")
        rows.append({"user_id": user_id, "family": row.get("family", ""), "doc": doc, "label": 1, "source": "pevo_final"})
    return pd.DataFrame(rows)


def run_lightweight_tfidf_summary(iteration_dir: Path, out_path: Path | None = None, sample_size: int = 400) -> dict[str, Any]:
    rows = _load_generated_text_rows(iteration_dir)
    texts = _nonempty_texts(rows, sample_size=sample_size)
    payload: dict[str, Any] = {"status": "skipped", "detector": "tfidf_self_similarity", "basic": _basic_text_summary(rows)}
    if len(texts) < 2:
        payload.update({"reason": "need at least two generated non-empty texts", "sample_size": len(texts)})
    elif TfidfVectorizer is None:
        payload.update(
            {
                "status": "placeholder",
                "reason": f"sklearn unavailable: {type(SKLEARN_IMPORT_ERROR).__name__}: {SKLEARN_IMPORT_ERROR}",
                "sample_size": len(texts),
            }
        )
    else:
        try:
            vectorizer = TfidfVectorizer(max_features=5000, min_df=1, max_df=1.0, ngram_range=(1, 2), stop_words="english")
            matrix = vectorizer.fit_transform(texts)
            sim = (matrix @ matrix.T).toarray()
            np.fill_diagonal(sim, 0.0)
            nearest = sim.max(axis=1)
            upper = sim[np.triu_indices_from(sim, k=1)]
            payload.update(
                {
                    "status": "ok",
                    "sample_size": int(len(texts)),
                    "feature_count": int(matrix.shape[1]),
                    "mean_nearest_neighbor_similarity": _round(float(nearest.mean())),
                    "p95_nearest_neighbor_similarity": _round(float(np.quantile(nearest, 0.95))),
                    "max_pair_similarity": _round(float(upper.max()) if upper.size else 0.0),
                    "pair_share_similarity_ge_0_85": _round(float((upper >= 0.85).sum() / max(1, upper.size))),
                }
            )
        except Exception as exc:
            payload.update({"status": "placeholder", "reason": f"tfidf failed: {type(exc).__name__}: {exc}", "sample_size": len(texts)})
    if out_path is not None:
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_tfidf_sanity(project_root: Path, iteration_dir: Path, out_path: Path) -> dict[str, Any]:
    analysis_dir = project_root / "baselines" / "text"
    if str(analysis_dir) not in sys.path:
        sys.path.insert(0, str(analysis_dir))
    try:
        from comment_text_only_experiment import load_comment_docs, strict_clean_comment, truncate_words
    except Exception as exc:
        payload = {
            "status": "placeholder",
            "detector": "tfidf_lr_original_train",
            "reason": f"cannot import text experiment helpers: {exc}",
            "fallback": run_lightweight_tfidf_summary(iteration_dir, None),
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    if TfidfVectorizer is None or LogisticRegression is None:
        payload = {
            "status": "placeholder",
            "detector": "tfidf_lr_original_train",
            "reason": f"sklearn unavailable: {type(SKLEARN_IMPORT_ERROR).__name__}: {SKLEARN_IMPORT_ERROR}",
            "fallback": run_lightweight_tfidf_summary(iteration_dir, None),
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    stats, _raw, _clean, strict_docs = load_comment_docs(project_root)
    labels = stats["label"].to_numpy(dtype=int)
    docs = [truncate_words(doc, 600) for doc in strict_docs]
    generated = _load_generated_docs(iteration_dir)
    clean_generated = []
    for doc in generated["doc"].astype(str) if not generated.empty else []:
        parts = []
        for line in doc.split("\n"):
            cleaned, reason = strict_clean_comment(line)
            if reason is None and cleaned:
                parts.append(cleaned)
        clean_generated.append(truncate_words(" ".join(parts), 600))
    if generated.empty:
        payload = {"status": "skipped", "reason": "no generated docs"}
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    vectorizer = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.9, ngram_range=(1, 2), stop_words="english")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    X = vectorizer.fit_transform(docs)
    clf.fit(X, labels)
    probs = clf.predict_proba(vectorizer.transform(clean_generated))[:, 1]
    pred = (probs >= 0.5).astype(int)
    result = generated[["user_id", "family"]].copy()
    result["bot_probability"] = probs
    result["predicted_bot"] = pred
    result["detector"] = "tfidf_lr_original_train"
    result.to_csv(out_path, index=False)
    return {
        "status": "ok",
        "detector": "tfidf_lr_original_train",
        "n": int(len(result)),
        "recall_on_generated": float(pred.mean()) if len(pred) else 0.0,
        "mean_bot_probability": float(np.mean(probs)) if len(probs) else 0.0,
        "fallback": run_lightweight_tfidf_summary(iteration_dir, None),
    }


def _hf_snapshot_dirs(model_id: str) -> list[Path]:
    escaped = "models--" + model_id.replace("/", "--")
    roots = []
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    roots.append(Path("~/.cache/huggingface/hub").expanduser())
    out: list[Path] = []
    for root in roots:
        snapshots = root / escaped / "snapshots"
        if snapshots.exists():
            out.extend(sorted(path for path in snapshots.iterdir() if path.is_dir()))
    return out


def _local_bge_candidates(model: str | None = None) -> list[Path]:
    names = [value for value in [model, os.environ.get("PEBA_BGE_MODEL")] if value]
    names.extend(["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"])
    candidates: list[Path] = []
    for name in names:
        path = Path(str(name)).expanduser()
        if path.exists():
            candidates.append(path.resolve())
            continue
        if "/" in str(name):
            candidates.extend(_hf_snapshot_dirs(str(name)))
            torch_cache = Path("~/.cache/torch/sentence_transformers").expanduser() / str(name).replace("/", "_")
            if torch_cache.exists():
                candidates.append(torch_cache.resolve())
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def run_bge_sanity(iteration_dir: Path, out_path: Path | None = None, bge_model: str | None = None, sample_size: int = 200) -> dict[str, Any]:
    rows = _load_generated_text_rows(iteration_dir)
    texts = _nonempty_texts(rows, sample_size=sample_size)
    if len(texts) < 2:
        payload = {"status": "skipped", "detector": "bge_local_embedding", "reason": "need at least two generated non-empty texts", "sample_size": len(texts)}
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            payload = {
                "status": "placeholder",
                "detector": "bge_local_embedding",
                "reason": f"sentence_transformers unavailable: {type(exc).__name__}: {exc}",
                "sample_size": len(texts),
            }
        else:
            candidates = _local_bge_candidates(bge_model)
            payload = {
                "status": "placeholder",
                "detector": "bge_local_embedding",
                "reason": "no local BGE/SentenceTransformer cache found; no network download attempted",
                "sample_size": len(texts),
            }
            for candidate in candidates:
                try:
                    model = SentenceTransformer(str(candidate), device="cpu")
                    try:
                        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                    except TypeError:
                        embeddings = model.encode(texts, show_progress_bar=False)
                    arr = np.asarray(embeddings, dtype=float)
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    unit = arr / np.maximum(norms, 1e-12)
                    sim = unit @ unit.T
                    np.fill_diagonal(sim, 0.0)
                    nearest = sim.max(axis=1)
                    upper = sim[np.triu_indices_from(sim, k=1)]
                    payload = {
                        "status": "ok",
                        "detector": "bge_local_embedding",
                        "model_path": str(candidate),
                        "sample_size": int(len(texts)),
                        "embedding_dim": int(arr.shape[1]),
                        "mean_nearest_neighbor_cosine": _round(float(nearest.mean())),
                        "p95_nearest_neighbor_cosine": _round(float(np.quantile(nearest, 0.95))),
                        "max_pair_cosine": _round(float(upper.max()) if upper.size else 0.0),
                        "pair_share_cosine_ge_0_92": _round(float((upper >= 0.92).sum() / max(1, upper.size))),
                    }
                    break
                except Exception as exc:
                    payload = {
                        "status": "placeholder",
                        "detector": "bge_local_embedding",
                        "reason": f"local BGE candidate failed: {candidate}: {type(exc).__name__}: {exc}",
                        "sample_size": len(texts),
                    }
    if out_path is not None:
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_graphstats_sanity(iteration_dir: Path, out_path: Path | None = None) -> dict[str, Any]:
    graph_path = iteration_dir / "graph_proxy_audit.csv"
    if not graph_path.exists():
        payload = {"status": "skipped", "reason": "missing graph_proxy_audit.csv"}
        if out_path is not None:
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    df = pd.read_csv(graph_path).fillna(0)
    if df.empty:
        payload = {"status": "skipped", "reason": "empty graph proxy"}
        if out_path is not None:
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    behavior = _read_csv(iteration_dir / "behavior_audit_by_user.csv")
    if not behavior.empty and "user_id" in behavior.columns:
        keep = [column for column in behavior.columns if column == "user_id" or column not in df.columns]
        df = df.merge(behavior[keep], on="user_id", how="left").fillna(0)
    out = df[["user_id"]].copy() if "user_id" in df.columns else pd.DataFrame({"user_id": [str(i) for i in range(len(df))]})
    out_degree = pd.to_numeric(df.get("total_out_weight_proxy", 0), errors="coerce").fillna(0)
    in_degree = pd.to_numeric(df.get("total_in_weight_proxy", 0), errors="coerce").fillna(0)
    direction = pd.to_numeric(df.get("direction_balance_proxy", 0), errors="coerce").fillna(0)
    reciprocal = pd.to_numeric(df.get("reciprocal_share_proxy", 0), errors="coerce").fillna(0)
    unique_targets = pd.to_numeric(df.get("unique_target_authors", 0), errors="coerce").fillna(0)
    unique_inbound = pd.to_numeric(df.get("unique_inbound_authors", 0), errors="coerce").fillna(0)
    comment_count = pd.to_numeric(df.get("comment_count", 0), errors="coerce").fillna(0)
    if comment_count.sum() == 0:
        comment_count = pd.to_numeric(df.get("comment1_count", 0), errors="coerce").fillna(0) + pd.to_numeric(df.get("comment2_count", 0), errors="coerce").fillna(0)
    total_activity = out_degree + in_degree
    if (direction.sum() == 0) and total_activity.sum() > 0:
        direction = out_degree / total_activity.replace(0, np.nan)
        direction = direction.fillna(0)
    imbalance = (direction - 0.5).abs() * 2.0
    no_inbound = ((in_degree <= 0) & (out_degree > 0)).astype(float)

    lr_probs = []
    for idx in df.index:
        logit = (
            -1.15
            + 0.46 * np.log1p(float(out_degree.loc[idx]))
            + 0.22 * np.log1p(float(in_degree.loc[idx]))
            + 0.95 * float(imbalance.loc[idx])
            + 0.70 * float(reciprocal.loc[idx])
            + 0.18 * np.log1p(float(unique_targets.loc[idx]))
            + 0.15 * np.log1p(float(comment_count.loc[idx]))
            + 0.35 * float(no_inbound.loc[idx])
        )
        lr_probs.append(_sigmoid(float(logit)))
    out_q75 = float(out_degree.quantile(0.75)) if len(out_degree) > 1 else float(out_degree.max())
    comment_q75 = float(comment_count.quantile(0.75)) if len(comment_count) > 1 else float(comment_count.max())
    rules = pd.DataFrame(
        {
            "high_activity": total_activity >= max(3.0, out_q75),
            "direction_imbalanced": imbalance >= 0.55,
            "high_reciprocity": reciprocal >= 0.30,
            "outbound_without_inbound": (out_degree >= 2.0) & (in_degree <= 0.0),
            "target_concentrated": (out_degree >= 3.0) & (unique_targets <= 1.0),
            "comment_heavy": comment_count >= max(3.0, comment_q75),
        },
        index=df.index,
    )
    rf_like = rules.astype(float).mean(axis=1)
    rank = lambda values: values.rank(method="average", pct=True).fillna(0.5) if len(values) > 1 else pd.Series([0.5] * len(values), index=values.index)
    statistical = 0.35 * rank(total_activity) + 0.25 * rank(imbalance) + 0.20 * rank(reciprocal) + 0.20 * rank(comment_count)
    out["graph_lr_like_probability_proxy"] = [round(float(value), 4) for value in lr_probs]
    out["graph_rf_like_vote_probability_proxy"] = rf_like.round(4)
    out["graph_statistical_probability_proxy"] = statistical.round(4)
    out["graph_bot_probability_proxy"] = (
        0.45 * out["graph_lr_like_probability_proxy"]
        + 0.35 * out["graph_rf_like_vote_probability_proxy"]
        + 0.20 * out["graph_statistical_probability_proxy"]
    ).round(4)
    out["predicted_bot_proxy"] = (out["graph_bot_probability_proxy"] >= 0.5).astype(int)
    out["detector"] = "graphstats_lr_like_rule_vote_untrained"
    out["total_out_weight_proxy"] = out_degree
    out["total_in_weight_proxy"] = in_degree
    out["direction_balance_proxy"] = direction.round(4)
    out["reciprocal_share_proxy"] = reciprocal.round(4)
    out["unique_target_authors"] = unique_targets
    out["unique_inbound_authors"] = unique_inbound
    out["comment_count"] = comment_count
    out["triggered_rules"] = [",".join(name for name, flag in row.items() if flag) for row in rules.to_dict(orient="records")]
    if out_path is not None:
        out.to_csv(out_path, index=False)
    probs = out["graph_bot_probability_proxy"]
    top_users = out.sort_values("graph_bot_probability_proxy", ascending=False).head(10)
    return {
        "status": "ok",
        "detector": "graphstats_lr_like_rule_vote_untrained",
        "n": int(len(out)),
        "recall_on_generated_proxy": float(out["predicted_bot_proxy"].mean()) if len(out) else 0.0,
        "mean_bot_probability_proxy": float(probs.mean()) if len(out) else 0.0,
        "median_bot_probability_proxy": float(probs.median()) if len(out) else 0.0,
        "max_bot_probability_proxy": float(probs.max()) if len(out) else 0.0,
        "share_probability_ge_0_66": float((probs >= 0.66).mean()) if len(out) else 0.0,
        "mean_total_out_weight_proxy": float(out_degree.mean()) if len(out) else 0.0,
        "mean_total_in_weight_proxy": float(in_degree.mean()) if len(out) else 0.0,
        "mean_direction_balance_proxy": float(direction.mean()) if len(out) else 0.0,
        "mean_reciprocal_share_proxy": float(reciprocal.mean()) if len(out) else 0.0,
        "top_users": top_users[
            [
                "user_id",
                "graph_bot_probability_proxy",
                "graph_lr_like_probability_proxy",
                "graph_rf_like_vote_probability_proxy",
                "graph_statistical_probability_proxy",
                "total_out_weight_proxy",
                "total_in_weight_proxy",
                "direction_balance_proxy",
                "reciprocal_share_proxy",
                "triggered_rules",
            ]
        ].to_dict(orient="records"),
        "reason": "untrained sanity proxy from graph_proxy_audit.csv plus optional behavior_audit_by_user.csv",
    }


def evaluate_run(project_root: Path, iteration_dir: Path, bge_model: str | None = None, sample_size: int = 400) -> dict[str, Any]:
    tfidf_light = run_lightweight_tfidf_summary(iteration_dir, None, sample_size=sample_size)
    graphstats = run_graphstats_sanity(iteration_dir, None)
    bge = run_bge_sanity(iteration_dir, None, bge_model=bge_model, sample_size=min(sample_size, 200))
    comments = _read_csv(iteration_dir / "generated_comments.csv")
    posts = _read_csv(iteration_dir / "generated_posts.csv")
    graph = _read_csv(iteration_dir / "graph_proxy_audit.csv")
    behavior = _read_csv(iteration_dir / "behavior_audit_by_user.csv")
    overall = "ready" if graphstats.get("status") == "ok" and tfidf_light.get("status") in {"ok", "placeholder"} else "partial"
    if graphstats.get("status") != "ok":
        overall = "blocked"
    return {
        "status": overall,
        "project_root": str(project_root),
        "iteration_dir": str(iteration_dir),
        "input_rows": {
            "generated_comments": int(len(comments)),
            "generated_posts": int(len(posts)),
            "graph_proxy_audit": int(len(graph)),
            "behavior_audit_by_user": int(len(behavior)),
        },
        "text_basic": tfidf_light.get("basic", {}),
        "tfidf_lightweight": tfidf_light,
        "bge": bge,
        "graphstats": graphstats,
    }


def run_detector_sanity(project_root: Path, iteration_dir: Path, out_dir: Path, bge_model: str | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tfidf_payload = run_tfidf_sanity(project_root, iteration_dir, out_dir / "detector_predictions_tfidf.csv")
    tfidf_light = tfidf_payload.get("fallback") if isinstance(tfidf_payload.get("fallback"), dict) else run_lightweight_tfidf_summary(
        iteration_dir,
        out_dir / "detector_summary_tfidf_lightweight.json",
    )
    summary = {
        "tfidf": tfidf_payload,
        "tfidf_lightweight": tfidf_light,
        "graphstats": run_graphstats_sanity(iteration_dir, out_dir / "detector_predictions_graphstats.csv"),
        "bge": run_bge_sanity(iteration_dir, out_dir / "detector_summary_bge.json", bge_model=bge_model),
    }
    (out_dir / "detector_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PEBA detector sanity summary for final generated outputs.")
    parser.add_argument("--project-root", default=".", help="RedditBotSim/project root used by optional original TF-IDF helpers.")
    parser.add_argument("--iteration-dir", required=True, help="Directory containing generated_comments/posts and graph_proxy audit CSVs.")
    parser.add_argument("--out-dir", help="Optional output directory. If omitted, prints no-write JSON summary.")
    parser.add_argument("--bge-model", help="Optional local BGE/SentenceTransformer path or cached model id.")
    parser.add_argument("--sample-size", type=int, default=400)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    iteration_dir = Path(args.iteration_dir).expanduser().resolve()
    if args.out_dir:
        result = run_detector_sanity(project_root, iteration_dir, Path(args.out_dir).expanduser().resolve(), bge_model=args.bge_model)
    else:
        result = evaluate_run(project_root, iteration_dir, bge_model=args.bge_model, sample_size=args.sample_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
