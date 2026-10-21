#!/usr/bin/env python3
"""Unsupervised account discovery for original_human vs original_bot.

Labels are kept out of feature construction, similarity search, clustering,
and suspicious-cluster scoring. They are copied only into eval_* columns and
used after scoring to reveal cluster purity and top-cluster precision/recall.
"""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import argparse
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


TARGET_FAMILIES = ("original_human", "original_bot")
BOT_FAMILY = "original_bot"
HUMAN_FAMILY = "original_human"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


@dataclass
class FeatureView:
    name: str
    matrix: Any
    n_features: int
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run account-level unsupervised discovery on original humans and original bots.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(_RELEASE_ROOT / "unsupervised/inputs/static_family"),
        help="Directory containing combined_sample_index.csv and optionally combined_bge_embeddings.npy.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_RELEASE_ROOT / "outputs/unsupervised"),
        help="Directory where CSV outputs will be written.",
    )
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--sample-human", type=int, default=None, help="Optional smoke sample size for humans.")
    parser.add_argument("--sample-bot", type=int, default=None, help="Optional smoke sample size for bots.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--null-repeats", type=int, default=300)
    parser.add_argument(
        "--kmeans-k",
        default="8,12,20,30",
        help="Comma-separated KMeans/MiniBatchKMeans cluster counts.",
    )
    return parse_release_args(parser, globals(), "unsupervised")


def parse_k_values(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k < 2:
            raise ValueError(f"kmeans-k values must be >= 2, got {k}")
        out.append(k)
    if not out:
        raise ValueError("No valid kmeans-k values were provided.")
    return sorted(set(out))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def text_word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def stable_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def infer_account_id(row: pd.Series, input_pos: int) -> str:
    for col in ("user_id", "account_id", "author_id", "user_name", "screen_name", "source_id"):
        if col in row.index:
            val = str(row[col]).strip()
            if val and val.lower() not in {"nan", "none", "null"}:
                return val
    return f"row_{input_pos}"


def dedupe_ids(values: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for value in values:
        count = seen.get(value, 0)
        seen[value] = count + 1
        out.append(value if count == 0 else f"{value}__dup{count + 1}")
    return out


def select_rows(combined: pd.DataFrame, min_words: int, sample_human: int | None, sample_bot: int | None, seed: int) -> pd.DataFrame:
    if "family" not in combined.columns:
        raise ValueError("combined_sample_index.csv must contain a family column.")
    if "doc" not in combined.columns:
        raise ValueError("combined_sample_index.csv must contain a doc column.")

    work = combined.copy()
    work["doc"] = work["doc"].fillna("").astype(str)
    if "word_count" in work.columns:
        observed_word_count = pd.to_numeric(work["word_count"], errors="coerce")
    else:
        observed_word_count = pd.Series(np.nan, index=work.index)
    computed_word_count = work["doc"].map(text_word_count)
    word_count = observed_word_count.fillna(computed_word_count).astype(int)

    mask = (
        work["family"].astype(str).isin(TARGET_FAMILIES)
        & work["doc"].str.strip().ne("")
        & word_count.ge(min_words)
    )
    selected = work.loc[mask].copy()
    selected["_input_pos"] = np.flatnonzero(mask.to_numpy())
    selected["_filter_word_count"] = word_count.loc[mask].to_numpy(dtype=int)

    if selected.empty:
        raise ValueError(
            f"No rows left after filtering family in {TARGET_FAMILIES}, non-empty docs, and word_count >= {min_words}."
        )

    rng = np.random.default_rng(seed)

    def maybe_sample(frame: pd.DataFrame, family: str, n: int | None) -> pd.DataFrame:
        part = frame.loc[frame["family"].astype(str).eq(family)]
        if n is None or n >= len(part):
            return part
        chosen = rng.choice(part.index.to_numpy(), size=n, replace=False)
        return part.loc[np.sort(chosen)]

    if sample_human is not None or sample_bot is not None:
        selected = pd.concat(
            [
                maybe_sample(selected, HUMAN_FAMILY, sample_human),
                maybe_sample(selected, BOT_FAMILY, sample_bot),
            ],
            axis=0,
        ).sort_values("_input_pos")
        if selected.empty:
            raise ValueError("Sampling removed all rows.")

    selected = selected.reset_index(drop=True)
    raw_ids = [infer_account_id(row, int(row["_input_pos"])) for _, row in selected.iterrows()]
    selected["account_id_raw"] = raw_ids
    selected["account_id"] = dedupe_ids(raw_ids)
    return selected


def build_eval_label(selected: pd.DataFrame) -> pd.Series:
    if "label" in selected.columns:
        numeric = pd.to_numeric(selected["label"], errors="coerce")
    else:
        numeric = pd.Series(np.nan, index=selected.index)
    family_map = selected["family"].astype(str).map({HUMAN_FAMILY: 0, BOT_FAMILY: 1})
    return numeric.fillna(family_map).astype(int)


def tokenize_lower(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def style_features(docs: list[str], selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for text in docs:
        text = text or ""
        tokens = tokenize_lower(text)
        token_lengths = np.array([len(t) for t in tokens], dtype=float)
        char_count = len(text)
        nonspace_count = sum(1 for ch in text if not ch.isspace())
        alpha_chars = [ch for ch in text if ch.isalpha()]
        sentence_marks = max(1, text.count(".") + text.count("!") + text.count("?"))
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        rows.append(
            {
                "style_char_count": float(char_count),
                "style_nonspace_count": float(nonspace_count),
                "style_token_count": float(len(tokens)),
                "style_unique_token_ratio": float(unique_ratio),
                "style_mean_token_len": float(token_lengths.mean()) if len(token_lengths) else 0.0,
                "style_std_token_len": float(token_lengths.std(ddof=0)) if len(token_lengths) else 0.0,
                "style_sentence_mark_count": float(sentence_marks),
                "style_mean_tokens_per_sentence": float(len(tokens) / sentence_marks),
                "style_url_count": float(len(URL_RE.findall(text))),
                "style_digit_ratio": float(sum(ch.isdigit() for ch in text) / max(1, char_count)),
                "style_upper_alpha_ratio": float(sum(ch.isupper() for ch in alpha_chars) / max(1, len(alpha_chars))),
                "style_punct_ratio": float(sum((not ch.isalnum()) and (not ch.isspace()) for ch in text) / max(1, char_count)),
                "style_question_ratio": float(text.count("?") / max(1, char_count)),
                "style_exclaim_ratio": float(text.count("!") / max(1, char_count)),
                "style_comma_ratio": float(text.count(",") / max(1, char_count)),
                "style_semicolon_colon_ratio": float((text.count(";") + text.count(":")) / max(1, char_count)),
                "style_quote_ratio": float((text.count('"') + text.count("'")) / max(1, char_count)),
                "style_parenthesis_ratio": float((text.count("(") + text.count(")")) / max(1, char_count)),
                "style_newline_count": float(text.count("\n")),
            }
        )

    features = pd.DataFrame(rows)
    for col in ("comment_count", "word_count"):
        if col in selected.columns:
            features[f"style_input_{col}"] = pd.to_numeric(selected[col], errors="coerce").fillna(0).to_numpy(dtype=float)
    features["style_filter_word_count"] = selected["_filter_word_count"].to_numpy(dtype=float)
    return features.replace([np.inf, -np.inf], 0).fillna(0)


def fit_tfidf(name: str, docs: list[str], analyzer: str, ngram_range: tuple[int, int], max_features: int) -> FeatureView:
    max_df = 1.0 if len(docs) < 3 else 0.95
    common_kwargs: dict[str, Any] = {
        "analyzer": analyzer,
        "ngram_range": ngram_range,
        "max_features": max_features,
        "lowercase": True,
        "strip_accents": "unicode",
        "norm": "l2",
    }
    if analyzer == "word":
        common_kwargs["stop_words"] = "english"
        common_kwargs["token_pattern"] = r"(?u)\b\w\w+\b"

    last_error: Exception | None = None
    for min_df in (2, 1):
        try:
            vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, **common_kwargs)
            matrix = vectorizer.fit_transform(docs)
            return FeatureView(name=name, matrix=matrix.tocsr(), n_features=matrix.shape[1], kind=f"{analyzer}_tfidf_min_df_{min_df}")
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"Could not build {name}: {last_error}") from last_error


def build_feature_views(input_dir: Path, combined: pd.DataFrame, selected: pd.DataFrame, style_df: pd.DataFrame) -> list[FeatureView]:
    docs = selected["doc"].fillna("").astype(str).tolist()
    views = [
        fit_tfidf("word_tfidf", docs, analyzer="word", ngram_range=(1, 2), max_features=50000),
        fit_tfidf("char_wb_tfidf", docs, analyzer="char_wb", ngram_range=(3, 5), max_features=80000),
    ]

    style_scaled = StandardScaler().fit_transform(style_df.to_numpy(dtype=float))
    style_scaled = np.nan_to_num(style_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    style_norm = normalize(style_scaled, norm="l2", copy=False)
    views.append(FeatureView("style_numeric", style_norm.astype(np.float32), style_norm.shape[1], "standardized_text_style"))

    bge_path = input_dir / "combined_bge_embeddings.npy"
    if bge_path.exists():
        embeddings = np.load(bge_path)
        if embeddings.shape[0] == len(combined):
            selected_embeddings = embeddings[selected["_input_pos"].to_numpy(dtype=int)].astype(np.float32, copy=False)
            selected_embeddings = normalize(np.nan_to_num(selected_embeddings), norm="l2", copy=False)
            views.append(FeatureView("bge", selected_embeddings, selected_embeddings.shape[1], "precomputed_bge"))
        else:
            print(
                f"[warn] Skipping BGE view: {bge_path} rows={embeddings.shape[0]} but combined_sample_index rows={len(combined)}.",
                file=sys.stderr,
            )
    return views


def write_account_features(output_dir: Path, selected: pd.DataFrame, style_df: pd.DataFrame, views: list[FeatureView]) -> Path:
    out = pd.DataFrame(
        {
            "account_index": np.arange(len(selected), dtype=int),
            "account_id": selected["account_id"],
            "account_id_raw": selected["account_id_raw"],
            "input_position": selected["_input_pos"].astype(int),
            "input_row": selected["row"] if "row" in selected.columns else selected["_input_pos"].astype(int),
            "doc_sha1": selected["doc"].map(stable_sha1),
        }
    )
    for optional_col in ("user_name", "provider", "model", "temperature"):
        if optional_col in selected.columns:
            out[f"meta_{optional_col}"] = selected[optional_col]
    out = pd.concat([out, style_df.reset_index(drop=True)], axis=1)
    for view in views:
        out[f"view_{view.name}_n_features"] = view.n_features
    path = output_dir / "account_features.csv"
    out.to_csv(path, index=False)
    return path


def write_label_revealed_account_metadata(output_dir: Path, selected: pd.DataFrame, eval_label: pd.Series) -> Path:
    out = pd.DataFrame(
        {
            "account_index": np.arange(len(selected), dtype=int),
            "account_id": selected["account_id"],
            "account_id_raw": selected["account_id_raw"],
            "input_position": selected["_input_pos"].astype(int),
            "eval_family": selected["family"].astype(str),
            "eval_label": eval_label.astype(int),
        }
    )
    for optional_col in ("source", "row_kind", "family_display"):
        if optional_col in selected.columns:
            out[f"eval_{optional_col}"] = selected[optional_col]
    path = output_dir / "label_revealed_account_metadata.csv"
    out.to_csv(path, index=False)
    return path


def make_knn_edges(view: FeatureView, account_ids: np.ndarray, k: int) -> list[dict[str, Any]]:
    n = len(account_ids)
    if n <= 1:
        return []
    n_neighbors = min(n, max(2, k + 1))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    model.fit(view.matrix)
    distances, indices = model.kneighbors(view.matrix, return_distance=True)

    rows: list[dict[str, Any]] = []
    for src_idx in range(n):
        rank = 0
        for dist, dst_idx in zip(distances[src_idx], indices[src_idx]):
            dst_idx = int(dst_idx)
            if dst_idx == src_idx:
                continue
            rank += 1
            if rank > k:
                break
            rows.append(
                {
                    "view": view.name,
                    "source_account_index": src_idx,
                    "source_account_id": account_ids[src_idx],
                    "target_account_index": dst_idx,
                    "target_account_id": account_ids[dst_idx],
                    "rank": rank,
                    "cosine_similarity": float(np.clip(1.0 - dist, -1.0, 1.0)),
                }
            )
    return rows


def write_similarity_edges(output_dir: Path, views: list[FeatureView], account_ids: np.ndarray, k: int) -> Path:
    rows: list[dict[str, Any]] = []
    for view in views:
        print(f"[info] Building kNN edges for view={view.name}, k={k}")
        rows.extend(make_knn_edges(view, account_ids, k))
    path = output_dir / "account_similarity_edges.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def choose_clusterer(view: FeatureView, k: int, random_state: int):
    n_samples, n_features = view.matrix.shape
    use_minibatch = sparse.issparse(view.matrix) or n_samples > 5000 or n_features > 10000
    if use_minibatch:
        return MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
            batch_size=max(1024, k * 32),
            reassignment_ratio=0.01,
        )
    return KMeans(n_clusters=k, random_state=random_state, n_init=20)


def run_clustering(output_dir: Path, views: list[FeatureView], account_ids: np.ndarray, k_values: list[int], random_state: int):
    assignment_rows: list[dict[str, Any]] = []
    cluster_members: dict[tuple[str, int, int], np.ndarray] = {}
    cluster_meta: dict[tuple[str, int, int], dict[str, Any]] = {}
    n = len(account_ids)
    for view in views:
        for k in k_values:
            if k > n:
                print(f"[warn] Skipping view={view.name}, k={k}: only {n} accounts.", file=sys.stderr)
                continue
            print(f"[info] Clustering view={view.name}, k={k}")
            clusterer = choose_clusterer(view, k, random_state)
            cluster_ids = clusterer.fit_predict(view.matrix)
            algorithm = clusterer.__class__.__name__
            for account_index, cluster_id in enumerate(cluster_ids.astype(int)):
                assignment_rows.append(
                    {
                        "account_index": account_index,
                        "account_id": account_ids[account_index],
                        "view": view.name,
                        "k": k,
                        "cluster_id": int(cluster_id),
                        "algorithm": algorithm,
                        "n_features": view.n_features,
                    }
                )
            for cluster_id in sorted(np.unique(cluster_ids.astype(int))):
                key = (view.name, int(k), int(cluster_id))
                members = np.flatnonzero(cluster_ids == cluster_id)
                cluster_members[key] = members.astype(int)
                cluster_meta[key] = {"algorithm": algorithm, "n_features": view.n_features}

    path = output_dir / "cluster_assignments.csv"
    pd.DataFrame(assignment_rows).to_csv(path, index=False)
    return path, cluster_members, cluster_meta


def sum_vector_norm(matrix: Any, indices: np.ndarray) -> float:
    subset = matrix[indices]
    summed = subset.sum(axis=0)
    if sparse.issparse(summed):
        return float(math.sqrt(summed.multiply(summed).sum()))
    arr = np.asarray(summed).ravel()
    return float(np.linalg.norm(arr))


def row_norm_sq_sum(matrix: Any, indices: np.ndarray) -> float:
    subset = matrix[indices]
    if sparse.issparse(subset):
        return float(subset.multiply(subset).sum())
    arr = np.asarray(subset)
    return float(np.einsum("ij,ij->", arr, arr))


def cluster_geometry(matrix: Any, indices: np.ndarray) -> tuple[float, float]:
    size = len(indices)
    if size <= 1:
        return 0.0, 0.0
    norm_sum = sum_vector_norm(matrix, indices)
    norm_sq_rows = row_norm_sq_sum(matrix, indices)
    mean_internal = (norm_sum * norm_sum - norm_sq_rows) / (size * (size - 1))
    mean_internal = float(np.clip(mean_internal, -1.0, 1.0))
    mean_radius = float(max(0.0, 1.0 - norm_sum / max(1, size)))
    return mean_internal, mean_radius


def random_baseline(matrix: Any, size: int, repeats: int, rng: np.random.Generator) -> dict[str, float]:
    n = matrix.shape[0]
    repeats = max(1, int(repeats))
    densities = np.zeros(repeats, dtype=float)
    radii = np.zeros(repeats, dtype=float)
    for i in range(repeats):
        chosen = rng.choice(n, size=size, replace=False)
        densities[i], radii[i] = cluster_geometry(matrix, chosen)
    return {
        "null_density_mean": float(densities.mean()),
        "null_density_std": float(densities.std(ddof=1)) if repeats > 1 else 0.0,
        "null_radius_mean": float(radii.mean()),
        "null_radius_std": float(radii.std(ddof=1)) if repeats > 1 else 0.0,
    }


def safe_z(value: float, mean: float, std: float) -> float:
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0
    return float((value - mean) / std)


def compute_view_support(cluster_members: dict[tuple[str, int, int], np.ndarray], jaccard_threshold: float = 0.5) -> dict[tuple[str, int, int], int]:
    member_sets = {key: set(map(int, members)) for key, members in cluster_members.items()}
    support: dict[tuple[str, int, int], int] = {}
    keys_by_k: dict[int, list[tuple[str, int, int]]] = {}
    for key in member_sets:
        keys_by_k.setdefault(key[1], []).append(key)

    for key, members in member_sets.items():
        supported_views: set[str] = set()
        for other_key in keys_by_k.get(key[1], []):
            other = member_sets[other_key]
            union_size = len(members | other)
            if union_size == 0:
                continue
            jaccard = len(members & other) / union_size
            if jaccard >= jaccard_threshold:
                supported_views.add(other_key[0])
        support[key] = len(supported_views)
    return support


def write_suspicious_scores(
    output_dir: Path,
    views: list[FeatureView],
    cluster_members: dict[tuple[str, int, int], np.ndarray],
    cluster_meta: dict[tuple[str, int, int], dict[str, Any]],
    account_ids: np.ndarray,
    null_repeats: int,
    random_state: int,
) -> tuple[Path, pd.DataFrame]:
    view_map = {view.name: view for view in views}
    rng = np.random.default_rng(random_state)
    baseline_cache: dict[tuple[str, int], dict[str, float]] = {}
    support_count = compute_view_support(cluster_members)
    rows: list[dict[str, Any]] = []

    for key in sorted(cluster_members, key=lambda x: (x[0], x[1], x[2])):
        view_name, k, cluster_id = key
        members = cluster_members[key]
        view = view_map[view_name]
        size = len(members)
        mean_internal, mean_radius = cluster_geometry(view.matrix, members)
        cache_key = (view_name, size)
        if cache_key not in baseline_cache:
            baseline_cache[cache_key] = random_baseline(view.matrix, size, null_repeats, rng)
        baseline = baseline_cache[cache_key]
        density_z = safe_z(mean_internal, baseline["null_density_mean"], baseline["null_density_std"])
        radius_z = safe_z(baseline["null_radius_mean"], mean_radius, baseline["null_radius_std"])
        view_support_count = support_count.get(key, 1)
        suspicious_score = float(density_z + radius_z + 0.5 * max(0, view_support_count - 1) + 0.25 * math.log1p(size))
        member_ids = [str(account_ids[i]) for i in members]
        rows.append(
            {
                "view": view_name,
                "k": int(k),
                "cluster_id": int(cluster_id),
                "algorithm": cluster_meta[key]["algorithm"],
                "n_features": cluster_meta[key]["n_features"],
                "cluster_size": int(size),
                "mean_internal_similarity": float(mean_internal),
                "mean_radius": float(mean_radius),
                "density_z_vs_random": float(density_z),
                "radius_z_vs_random": float(radius_z),
                "view_support_count": int(view_support_count),
                "suspicious_score": suspicious_score,
                "null_repeats": int(null_repeats),
                "null_density_mean": baseline["null_density_mean"],
                "null_density_std": baseline["null_density_std"],
                "null_radius_mean": baseline["null_radius_mean"],
                "null_radius_std": baseline["null_radius_std"],
                "member_account_ids": ";".join(member_ids),
            }
        )

    scores = pd.DataFrame(rows).sort_values(
        ["suspicious_score", "density_z_vs_random", "radius_z_vs_random", "cluster_size"],
        ascending=[False, False, False, False],
    )
    if not scores.empty:
        scores.insert(0, "suspicious_rank", np.arange(1, len(scores) + 1, dtype=int))
    path = output_dir / "suspicious_cluster_scores.csv"
    scores.to_csv(path, index=False)
    return path, scores


def write_label_revealed_evaluation(
    output_dir: Path,
    scores: pd.DataFrame,
    cluster_members: dict[tuple[str, int, int], np.ndarray],
    eval_label: pd.Series,
) -> Path:
    labels = eval_label.to_numpy(dtype=int)
    total_bots = int((labels == 1).sum())
    rows: list[dict[str, Any]] = []

    for score_row in scores.itertuples(index=False):
        key = (str(score_row.view), int(score_row.k), int(score_row.cluster_id))
        members = cluster_members.get(key, np.array([], dtype=int))
        member_labels = labels[members]
        bot_count = int((member_labels == 1).sum())
        human_count = int((member_labels == 0).sum())
        size = int(len(members))
        rows.append(
            {
                "row_type": "cluster",
                "top_cluster_count": np.nan,
                "suspicious_rank": int(score_row.suspicious_rank),
                "view": score_row.view,
                "k": int(score_row.k),
                "cluster_id": int(score_row.cluster_id),
                "suspicious_score": float(score_row.suspicious_score),
                "selected_account_count": size,
                "human_count": human_count,
                "bot_count": bot_count,
                "bot_rate": float(bot_count / size) if size else 0.0,
                "bot_precision": float(bot_count / size) if size else 0.0,
                "bot_recall": float(bot_count / total_bots) if total_bots else 0.0,
            }
        )

    cutoffs = [1, 3, 5, 10, 20, 50]
    cutoffs = [cutoff for cutoff in cutoffs if cutoff <= len(scores)]
    if len(scores) and len(scores) not in cutoffs:
        cutoffs.append(len(scores))
    for cutoff in cutoffs:
        selected_members: set[int] = set()
        for score_row in scores.head(cutoff).itertuples(index=False):
            key = (str(score_row.view), int(score_row.k), int(score_row.cluster_id))
            selected_members.update(map(int, cluster_members.get(key, np.array([], dtype=int))))
        chosen = np.array(sorted(selected_members), dtype=int)
        chosen_labels = labels[chosen] if len(chosen) else np.array([], dtype=int)
        bot_count = int((chosen_labels == 1).sum())
        human_count = int((chosen_labels == 0).sum())
        selected_count = int(len(chosen))
        rows.append(
            {
                "row_type": "top_clusters_union",
                "top_cluster_count": int(cutoff),
                "suspicious_rank": np.nan,
                "view": "ALL",
                "k": np.nan,
                "cluster_id": np.nan,
                "suspicious_score": np.nan,
                "selected_account_count": selected_count,
                "human_count": human_count,
                "bot_count": bot_count,
                "bot_rate": float(bot_count / selected_count) if selected_count else 0.0,
                "bot_precision": float(bot_count / selected_count) if selected_count else 0.0,
                "bot_recall": float(bot_count / total_bots) if total_bots else 0.0,
            }
        )

    path = output_dir / "label_revealed_evaluation.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    k_values = parse_k_values(args.kmeans_k)

    combined_path = input_dir / "combined_sample_index.csv"
    if not combined_path.exists():
        raise FileNotFoundError(f"Missing input file: {combined_path}")

    print(f"[info] Loading {combined_path}")
    combined = pd.read_csv(combined_path)
    selected = select_rows(combined, args.min_words, args.sample_human, args.sample_bot, args.random_state)
    eval_label = build_eval_label(selected)
    docs = selected["doc"].fillna("").astype(str).tolist()
    style_df = style_features(docs, selected)
    views = build_feature_views(input_dir, combined, selected, style_df)

    account_ids = selected["account_id"].astype(str).to_numpy()
    print(
        "[info] Selected accounts: "
        f"n={len(selected)}, humans={(selected['family'].astype(str) == HUMAN_FAMILY).sum()}, "
        f"bots={(selected['family'].astype(str) == BOT_FAMILY).sum()}, views={[view.name for view in views]}"
    )

    output_paths: list[Path] = []
    output_paths.append(write_account_features(output_dir, selected, style_df, views))
    output_paths.append(write_similarity_edges(output_dir, views, account_ids, args.knn_k))
    cluster_path, cluster_members, cluster_meta = run_clustering(output_dir, views, account_ids, k_values, args.random_state)
    output_paths.append(cluster_path)
    score_path, scores = write_suspicious_scores(
        output_dir,
        views,
        cluster_members,
        cluster_meta,
        account_ids,
        args.null_repeats,
        args.random_state,
    )
    output_paths.append(score_path)
    output_paths.append(write_label_revealed_account_metadata(output_dir, selected, eval_label))
    output_paths.append(write_label_revealed_evaluation(output_dir, scores, cluster_members, eval_label))

    print("[info] Wrote outputs:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
