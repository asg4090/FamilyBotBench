#!/usr/bin/env python3
"""Plot and report unsupervised account-discovery outputs.

The discovery stage is label-free. Labels are consumed here only for the final
reveal/evaluation plots and report sections.
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
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "original_human_vs_oldbot"

REQUIRED_FILES = {
    "scores": "suspicious_cluster_scores.csv",
    "label_eval": "label_revealed_evaluation.csv",
    "assignments": "cluster_assignments.csv",
    "features": "account_features.csv",
    "edges": "account_similarity_edges.csv",
}
OPTIONAL_FILES = {
    "account_metadata": "label_revealed_account_metadata.csv",
}

ACCOUNT_ID_CANDIDATES = [
    "account_id",
    "user_id",
    "node_id",
    "author_id",
    "author",
    "user_name",
    "username",
    "id",
]
CLUSTER_CANDIDATES = [
    "mechanism_cluster",
    "suspicious_cluster_id",
    "cluster_id",
    "final_cluster",
    "consensus_cluster",
    "community_id",
    "community",
    "cluster",
]
LABEL_CANDIDATES = [
    "revealed_label",
    "true_label",
    "label",
    "account_label",
    "is_bot",
    "bot",
    "y_true",
]
CLUSTER_KEY_COL = "cluster_key"


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory containing run_unsupervised_account_discovery.py outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-embedding-points",
        type=int,
        default=5000,
        help="Maximum accounts to draw in the embedding scatter plot.",
    )
    return parser.parse_args()


def read_required_outputs(output_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in REQUIRED_FILES.values() if not (output_dir / name).exists()]
    if missing:
        joined = "\n".join(f"- {output_dir / name}" for name in missing)
        raise FileNotFoundError(
            "Missing required output CSVs from run_unsupervised_account_discovery.py:\n"
            f"{joined}"
        )
    data = {key: pd.read_csv(output_dir / filename) for key, filename in REQUIRED_FILES.items()}
    for key, filename in OPTIONAL_FILES.items():
        path = output_dir / filename
        data[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return data


def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    *,
    contains: Iterable[str] | None = None,
    exclude_contains: Iterable[str] | None = None,
) -> str | None:
    if df.empty:
        return None
    by_lower = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        col = by_lower.get(candidate.lower())
        if col is not None:
            return col

    contains = [item.lower() for item in (contains or [])]
    exclude_contains = [item.lower() for item in (exclude_contains or [])]
    if not contains:
        return None

    for col in df.columns:
        lower = str(col).lower()
        if all(item in lower for item in contains) and not any(
            item in lower for item in exclude_contains
        ):
            return str(col)
    return None


def numeric_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def usable_numeric_columns(df: pd.DataFrame, *, drop_metadata: bool = True) -> list[str]:
    cols: list[str] = []
    metadata_tokens = [
        "id",
        "label",
        "cluster",
        "bot",
        "human",
        "rank",
        "family",
        "name",
        "source",
    ]
    for col in df.columns:
        lower = str(col).lower()
        if drop_metadata and any(token in lower for token in metadata_tokens):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= 2 and converted.nunique(dropna=True) > 1:
            cols.append(str(col))
    return cols


def first_numeric_by_tokens(
    df: pd.DataFrame,
    include_tokens: Iterable[str],
    exclude_tokens: Iterable[str] = (),
) -> str | None:
    include_tokens = [token.lower() for token in include_tokens]
    exclude_tokens = [token.lower() for token in exclude_tokens]
    for col in df.columns:
        lower = str(col).lower()
        if not all(token in lower for token in include_tokens):
            continue
        if any(token in lower for token in exclude_tokens):
            continue
        if pd.to_numeric(df[col], errors="coerce").notna().sum() > 0:
            return str(col)
    return None


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, (bool, np.bool_)):
        return "bot" if value else "human"
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if math.isclose(float(value), 1.0):
            return "bot"
        if math.isclose(float(value), 0.0):
            return "human"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    if text in {"1", "true", "yes", "bot", "oldbot", "original_bot"}:
        return "bot"
    if text in {"0", "false", "no", "human", "original_human"}:
        return "human"
    if "bot" in text:
        return "bot"
    if "human" in text:
        return "human"
    return text


def markdown_table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    max_rows: int | None = None,
    digits: int = 4,
) -> str:
    if df.empty:
        return "无可展示记录。"
    view = df.copy()
    if columns is not None:
        view = view[[col for col in columns if col in view.columns]]
    if max_rows is not None:
        view = view.head(max_rows)
    if view.empty:
        return "无可展示记录。"

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    headers = [str(col) for col in view.columns]
    rows = [[fmt(value) for value in row] for row in view.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def get_cluster_column(df: pd.DataFrame) -> str | None:
    return find_column(df, CLUSTER_CANDIDATES, contains=["cluster"])


def get_account_id_column(df: pd.DataFrame) -> str | None:
    return find_column(df, ACCOUNT_ID_CANDIDATES)


def add_cluster_key(df: pd.DataFrame, cluster_col: str | None = None) -> pd.DataFrame:
    keyed = df.copy()
    if CLUSTER_KEY_COL in keyed.columns:
        keyed[CLUSTER_KEY_COL] = keyed[CLUSTER_KEY_COL].astype(str)
        return keyed

    cluster_col = cluster_col or get_cluster_column(keyed)
    if cluster_col is None or cluster_col not in keyed.columns:
        return keyed

    lower_cols = {str(col).lower(): str(col) for col in keyed.columns}
    view_col = lower_cols.get("view")
    k_col = lower_cols.get("k")
    def key_part(value: object) -> str:
        if pd.isna(value):
            return "NA"
        try:
            number = float(value)
            if math.isfinite(number) and number.is_integer():
                return str(int(number))
        except Exception:
            pass
        return str(value)

    if view_col is not None and k_col is not None:
        keyed[CLUSTER_KEY_COL] = (
            keyed[view_col].astype(str)
            + ":k="
            + keyed[k_col].map(key_part)
            + ":c="
            + keyed[cluster_col].map(key_part)
        )
    else:
        keyed[CLUSTER_KEY_COL] = keyed[cluster_col].map(key_part)
    return keyed


def cluster_label_series(df: pd.DataFrame, cluster_col: str | None = None) -> pd.Series:
    keyed = add_cluster_key(df, cluster_col)
    if CLUSTER_KEY_COL in keyed.columns:
        return keyed[CLUSTER_KEY_COL].astype(str)
    return pd.Series([str(i) for i in range(len(df))], index=df.index)


def suspicious_cluster_keys(scores: pd.DataFrame) -> set[str]:
    cluster_col = get_cluster_column(scores)
    keyed = add_cluster_key(scores, cluster_col)
    if CLUSTER_KEY_COL not in keyed.columns:
        return set()

    if cluster_col is None or cluster_col not in scores.columns:
        return set()

    bool_col = None
    for candidate in ["is_suspicious", "selected", "is_selected"]:
        if candidate in keyed.columns:
            bool_col = candidate
            break
    if bool_col is not None:
        values = keyed[bool_col]
        if values.dtype == bool:
            return set(keyed.loc[values, CLUSTER_KEY_COL])
        normalized = values.astype(str).str.lower().str.strip()
        mask = normalized.isin(["1", "true", "yes", "y", "suspicious", "selected"])
        if mask.any():
            return set(keyed.loc[mask, CLUSTER_KEY_COL])

    rank_col = find_column(keyed, ["suspicious_rank", "rank"])
    if rank_col is not None:
        ranks = pd.to_numeric(keyed[rank_col], errors="coerce")
        if ranks.notna().any():
            top_n = min(20, max(1, ranks.notna().sum()))
            top = keyed.loc[ranks <= top_n]
            return set(top[CLUSTER_KEY_COL])

    return set(keyed[CLUSTER_KEY_COL].dropna())


def sorting_score_column(scores: pd.DataFrame) -> str | None:
    candidates = [
        "suspicious_score",
        "mechanism_score",
        "anomaly_score",
        "density_z_score",
        "z_score",
        "score",
    ]
    found = find_column(scores, candidates)
    if found is not None:
        return found
    for token in ["score", "z", "lift"]:
        found = first_numeric_by_tokens(scores, [token], ["p_value", "pvalue"])
        if found is not None:
            return found
    return None


def order_cluster_scores(scores: pd.DataFrame) -> pd.DataFrame:
    ordered = scores.copy()
    score_col = sorting_score_column(ordered)
    if score_col is not None:
        ordered["_plot_sort"] = pd.to_numeric(ordered[score_col], errors="coerce")
        return ordered.sort_values("_plot_sort", ascending=False).drop(columns="_plot_sort")
    density_col = density_column(ordered)
    if density_col is not None:
        ordered["_plot_sort"] = pd.to_numeric(ordered[density_col], errors="coerce")
        return ordered.sort_values("_plot_sort", ascending=False).drop(columns="_plot_sort")
    return ordered


def density_column(scores: pd.DataFrame) -> str | None:
    candidates = [
        "cluster_density",
        "observed_density",
        "density",
        "edge_density",
        "intra_cluster_density",
        "mean_similarity",
        "observed_mean_similarity",
    ]
    found = find_column(scores, candidates)
    if found is not None:
        return found
    return first_numeric_by_tokens(scores, ["density"], ["null"]) or first_numeric_by_tokens(
        scores, ["similarity"], ["null"]
    )


def plot_cluster_density_null_distribution(scores: pd.DataFrame, figures_dir: Path) -> tuple[Path, str]:
    cluster_col = get_cluster_column(scores)
    ordered = order_cluster_scores(scores)
    density_col = density_column(ordered)
    null_mean_col = first_numeric_by_tokens(ordered, ["null", "mean"])
    null_p95_col = (
        first_numeric_by_tokens(ordered, ["null", "p95"])
        or first_numeric_by_tokens(ordered, ["null", "q95"])
        or first_numeric_by_tokens(ordered, ["null", "95"])
    )
    null_sample_cols = [
        col
        for col in ordered.columns
        if "null" in str(col).lower()
        and not any(token in str(col).lower() for token in ["mean", "std", "p95", "q95", "p_value", "pvalue"])
        and pd.to_numeric(ordered[col], errors="coerce").notna().sum() > 0
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    note: str
    if density_col is not None and len(null_sample_cols) >= 3:
        null_values = pd.to_numeric(ordered[null_sample_cols].stack(), errors="coerce").dropna()
        observed = pd.to_numeric(ordered[density_col], errors="coerce").dropna()
        ax.hist(null_values, bins=30, color="#D9E2EC", edgecolor="#829AB1", label="Null density")
        for value in observed:
            ax.axvline(value, color="#D64545", alpha=0.35, linewidth=1.2)
        ax.axvline(observed.mean(), color="#8A1C1C", linewidth=2.0, label="Observed mean")
        ax.set_xlabel("Cluster density / similarity")
        ax.set_ylabel("Null sample count")
        ax.set_title("Cluster density against null distribution")
        ax.legend()
        note = f"使用 `{density_col}` 与 {len(null_sample_cols)} 个 null sample 列绘制分布。"
    elif density_col is not None:
        labels = (
            ordered[cluster_col].astype(str).tolist()
            if cluster_col is not None and cluster_col in ordered.columns
            else [str(i) for i in range(len(ordered))]
        )
        observed = pd.to_numeric(ordered[density_col], errors="coerce")
        x = np.arange(len(ordered))
        ax.bar(x, observed, color="#4C78A8", label=f"Observed `{density_col}`")
        if null_mean_col is not None:
            ax.plot(
                x,
                pd.to_numeric(ordered[null_mean_col], errors="coerce"),
                marker="o",
                color="#7B8794",
                linewidth=1.5,
                label=f"Null mean `{null_mean_col}`",
            )
        if null_p95_col is not None:
            ax.plot(
                x,
                pd.to_numeric(ordered[null_p95_col], errors="coerce"),
                marker="s",
                color="#D64545",
                linewidth=1.5,
                label=f"Null 95% `{null_p95_col}`",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Mechanism cluster")
        ax.set_ylabel("Density / similarity")
        ax.set_title("Observed suspicious-cluster density vs null baseline")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        note = "未发现展开的 null sample 列，改用观测密度与 null summary 对比。"
    else:
        score_col = sorting_score_column(ordered)
        if score_col is None:
            numeric_cols = usable_numeric_columns(ordered, drop_metadata=False)
            if not numeric_cols:
                raise ValueError("suspicious_cluster_scores.csv has no numeric column to plot.")
            score_col = numeric_cols[0]
        labels = (
            ordered[cluster_col].astype(str).tolist()
            if cluster_col is not None and cluster_col in ordered.columns
            else [str(i) for i in range(len(ordered))]
        )
        ax.bar(np.arange(len(ordered)), pd.to_numeric(ordered[score_col], errors="coerce"), color="#4C78A8")
        ax.set_xticks(np.arange(len(ordered)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Mechanism cluster")
        ax.set_ylabel(score_col)
        ax.set_title("Suspicious cluster score (density column not found)")
        ax.grid(axis="y", alpha=0.25)
        note = f"未发现 density/null 字段，使用 `{score_col}` 作为替代绘图。"

    fig.tight_layout()
    path = figures_dir / "cluster_density_null_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, note


def purity_from_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    cluster_col = get_cluster_column(assignments)
    label_col = find_column(assignments, LABEL_CANDIDATES)
    if cluster_col is None or label_col is None:
        return pd.DataFrame()
    tmp = assignments[[cluster_col, label_col]].copy()
    tmp["_label_norm"] = tmp[label_col].map(normalize_label)
    rows = []
    for cluster_id, group in tmp.groupby(cluster_col, dropna=False):
        total = len(group)
        bot_count = int(group["_label_norm"].eq("bot").sum())
        human_count = int(group["_label_norm"].eq("human").sum())
        rows.append(
            {
                "cluster_id": cluster_id,
                "account_count": total,
                "revealed_bot_count": bot_count,
                "revealed_human_count": human_count,
                "revealed_bot_purity": bot_count / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def purity_table(label_eval: pd.DataFrame, assignments: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    cluster_col_eval = get_cluster_column(label_eval)
    if cluster_col_eval is not None:
        table = label_eval.copy()
        bot_purity_col = (
            find_column(
                table,
                [
                    "revealed_bot_purity",
                    "bot_purity",
                    "bot_share",
                    "bot_rate",
                    "purity",
                    "revealed_precision",
                ],
            )
            or first_numeric_by_tokens(table, ["bot", "purity"])
            or first_numeric_by_tokens(table, ["bot", "share"])
            or first_numeric_by_tokens(table, ["purity"])
        )
        if bot_purity_col is not None:
            result = table.rename(columns={cluster_col_eval: "cluster_id", bot_purity_col: "revealed_bot_purity"})
            result = add_cluster_key(result, "cluster_id")
            if "account_count" not in result.columns:
                count_col = (
                    find_column(result, ["account_count", "cluster_size", "size", "n_accounts", "n"])
                    or first_numeric_by_tokens(result, ["count"])
                    or first_numeric_by_tokens(result, ["size"])
                )
                if count_col is not None:
                    result = result.rename(columns={count_col: "account_count"})
            return result

    derived = purity_from_assignments(assignments)
    if not derived.empty:
        return derived

    label_col = find_column(label_eval, LABEL_CANDIDATES)
    cluster_col = get_cluster_column(label_eval)
    if cluster_col is None or label_col is None:
        return pd.DataFrame()
    return purity_from_assignments(label_eval.rename(columns={cluster_col: "cluster_id"}))


def plot_suspicious_cluster_purity(
    label_eval: pd.DataFrame,
    assignments: pd.DataFrame,
    scores: pd.DataFrame,
    figures_dir: Path,
) -> tuple[Path, pd.DataFrame, str]:
    table = purity_table(label_eval, assignments, scores)
    if table.empty or "cluster_id" not in table.columns or "revealed_bot_purity" not in table.columns:
        raise ValueError(
            "Cannot build suspicious_cluster_purity.png: no cluster-level revealed purity "
            "or assignment-level labels were found."
        )
    suspicious_ids = suspicious_cluster_keys(scores)
    plot_table = add_cluster_key(table, "cluster_id")
    if suspicious_ids:
        plot_table = plot_table[plot_table[CLUSTER_KEY_COL].isin(suspicious_ids)].copy()
    if plot_table.empty:
        plot_table = add_cluster_key(table, "cluster_id")

    score_col = sorting_score_column(scores)
    if score_col is not None:
        score_lookup = add_cluster_key(scores, get_cluster_column(scores))
        keep_cols = [CLUSTER_KEY_COL, score_col]
        score_lookup = score_lookup[keep_cols].rename(columns={score_col: "suspicious_score"})
        plot_table = plot_table.merge(score_lookup, on=CLUSTER_KEY_COL, how="left")
    sort_cols = [col for col in ["suspicious_score", "revealed_bot_purity"] if col in plot_table.columns]
    if sort_cols:
        plot_table = plot_table.sort_values(sort_cols, ascending=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = np.arange(len(plot_table))
    purity = pd.to_numeric(plot_table["revealed_bot_purity"], errors="coerce")
    ax.bar(x, purity, color="#F58518")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Revealed bot share")
    ax.set_xlabel("Suspicious mechanism cluster")
    ax.set_title("Label-revealed purity of suspicious mechanism clusters")
    ax.set_xticks(x)
    labels = plot_table[CLUSTER_KEY_COL].astype(str) if CLUSTER_KEY_COL in plot_table.columns else plot_table["cluster_id"].astype(str)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(purity):
        if pd.notna(value):
            ax.text(idx, min(float(value) + 0.03, 0.97), f"{float(value):.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    path = figures_dir / "suspicious_cluster_purity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    note = "标签只在此处用于最后揭示评估，不参与无监督发现和 suspicious score 排序。"
    return path, plot_table, note


def view_cluster_columns(assignments: pd.DataFrame) -> list[str]:
    generic = {candidate.lower() for candidate in CLUSTER_CANDIDATES}
    cols = []
    for col in assignments.columns:
        lower = str(col).lower()
        if "cluster" not in lower:
            continue
        if lower in generic:
            continue
        if any(token in lower for token in ["view", "feature", "text", "graph", "behavior", "profile", "embedding"]):
            cols.append(str(col))
    return cols


def agreement_from_cluster_columns(assignments: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    cols = view_cluster_columns(assignments)
    if len(cols) < 2:
        return pd.DataFrame(), ""
    matrix = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, left in enumerate(cols):
        for j, right in enumerate(cols):
            if i >= j:
                continue
            valid = assignments[[left, right]].dropna()
            if len(valid) < 2:
                score = np.nan
            else:
                score = adjusted_rand_score(valid[left].astype(str), valid[right].astype(str))
            matrix.loc[left, right] = score
            matrix.loc[right, left] = score
    return matrix, "Adjusted Rand Index between view-specific cluster assignments."


def edge_pair_key(row: pd.Series, left_col: str, right_col: str) -> tuple[str, str]:
    left = str(row[left_col])
    right = str(row[right_col])
    return tuple(sorted((left, right)))


def agreement_from_edges(edges: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    view_col = find_column(edges, ["feature_view", "view", "metric", "similarity_view", "edge_view"])
    left_col = find_column(edges, ["source_account_id", "src_account_id", "source", "src", "u", "from"])
    right_col = find_column(edges, ["target_account_id", "dst_account_id", "target", "dst", "v", "to"])
    if view_col is None or left_col is None or right_col is None:
        return pd.DataFrame(), ""
    view_edges: dict[str, set[tuple[str, str]]] = {}
    for view, group in edges.groupby(view_col):
        pairs = {edge_pair_key(row, left_col, right_col) for _, row in group.iterrows()}
        if pairs:
            view_edges[str(view)] = pairs
    views = sorted(view_edges)
    if len(views) < 2:
        return pd.DataFrame(), ""
    matrix = pd.DataFrame(np.eye(len(views)), index=views, columns=views)
    for i, left in enumerate(views):
        for j, right in enumerate(views):
            if i >= j:
                continue
            union = view_edges[left] | view_edges[right]
            score = len(view_edges[left] & view_edges[right]) / len(union) if union else np.nan
            matrix.loc[left, right] = score
            matrix.loc[right, left] = score
    return matrix, "Jaccard overlap between account-similarity edge sets by feature view."


def infer_feature_groups(features: pd.DataFrame) -> dict[str, list[str]]:
    known_prefixes = [
        "text",
        "semantic",
        "embedding",
        "profile",
        "behavior",
        "activity",
        "temporal",
        "graph",
        "network",
        "language",
        "style",
        "content",
    ]
    groups: dict[str, list[str]] = {}
    for col in usable_numeric_columns(features):
        lower = str(col).lower()
        group = None
        for prefix in known_prefixes:
            if lower.startswith(prefix + "_") or lower.startswith(prefix + "__") or lower.startswith(prefix + ":"):
                group = prefix
                break
        if group is None and "__" in lower:
            group = lower.split("__", 1)[0]
        if group is not None:
            groups.setdefault(group, []).append(col)
    return {name: cols for name, cols in groups.items() if cols}


def agreement_from_feature_groups(features: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    groups = infer_feature_groups(features)
    if len(groups) < 2:
        return pd.DataFrame(), ""
    summaries: dict[str, np.ndarray] = {}
    for name, cols in groups.items():
        values = features[cols].apply(pd.to_numeric, errors="coerce")
        values = SimpleImputer(strategy="median").fit_transform(values)
        values = StandardScaler().fit_transform(values)
        if values.shape[1] >= 2 and values.shape[0] >= 3:
            summaries[name] = PCA(n_components=1, random_state=42).fit_transform(values).ravel()
        else:
            summaries[name] = values.mean(axis=1)
    summary_df = pd.DataFrame(summaries)
    matrix = summary_df.corr(method="spearman")
    return matrix, "Spearman correlation between per-view feature summaries."


def plot_feature_view_agreement_heatmap(
    assignments: pd.DataFrame,
    features: pd.DataFrame,
    edges: pd.DataFrame,
    figures_dir: Path,
) -> tuple[Path, str]:
    matrix, note = agreement_from_cluster_columns(assignments)
    if matrix.empty:
        matrix, note = agreement_from_edges(edges)
    if matrix.empty:
        matrix, note = agreement_from_feature_groups(features)
    if matrix.empty:
        matrix = pd.DataFrame([[1.0]], index=["combined"], columns=["combined"])
        note = "未发现多视角聚类列、view-specific 边或可分组特征；绘制 combined 占位热图。"

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    values = matrix.astype(float).to_numpy()
    vmin = -1.0 if np.nanmin(values) < 0 else 0.0
    im = ax.imshow(values, cmap="YlGnBu", vmin=vmin, vmax=1.0)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Feature-view agreement heatmap")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            label = "NA" if pd.isna(value) else f"{float(value):.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="#102A43")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = figures_dir / "feature_view_agreement_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, note


def merged_feature_frame(features: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    feature_id = get_account_id_column(features)
    assign_id = get_account_id_column(assignments)
    if feature_id is not None and assign_id is not None:
        return features.merge(assignments, left_on=feature_id, right_on=assign_id, how="left", suffixes=("", "_assignment"))
    merged = features.copy()
    for col in assignments.columns:
        if col not in merged.columns and len(assignments) == len(merged):
            merged[col] = assignments[col].to_numpy()
    return merged


def compute_embedding(values: np.ndarray) -> tuple[np.ndarray, str]:
    if values.shape[0] < 2:
        raise ValueError("Need at least two accounts to draw embedding scatter.")
    if values.shape[1] < 2:
        values = np.column_stack([values[:, 0], np.zeros(values.shape[0])])

    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, min(15, values.shape[0] - 1)),
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        )
        return reducer.fit_transform(values), "UMAP"
    except Exception:
        reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(values), "PCA fallback; UMAP unavailable"


def plot_account_embedding(
    features: pd.DataFrame,
    account_metadata: pd.DataFrame,
    assignments: pd.DataFrame,
    scores: pd.DataFrame,
    figures_dir: Path,
    max_points: int,
) -> tuple[Path, str]:
    merged = features.copy()
    feature_id = get_account_id_column(merged)
    meta_id = get_account_id_column(account_metadata)
    if feature_id is not None and meta_id is not None and not account_metadata.empty:
        metadata_cols = [
            col
            for col in ["eval_label", "revealed_label", "label", "eval_family"]
            if col in account_metadata.columns
        ]
        if metadata_cols:
            merged = merged.merge(
                account_metadata[[meta_id, *metadata_cols]].drop_duplicates(meta_id),
                left_on=feature_id,
                right_on=meta_id,
                how="left",
                suffixes=("", "_reveal"),
            )
    numeric_cols = usable_numeric_columns(features)
    if len(numeric_cols) < 1:
        raise ValueError("account_features.csv has no usable numeric feature columns for embedding.")

    sampled = merged.copy()
    if len(sampled) > max_points:
        sampled = sampled.sample(max_points, random_state=42)
    values = sampled[numeric_cols].apply(pd.to_numeric, errors="coerce")
    values = SimpleImputer(strategy="median").fit_transform(values)
    values = StandardScaler().fit_transform(values)
    coords, method = compute_embedding(values)

    label_col = find_column(sampled, ["eval_label", "revealed_label", "label"])
    if label_col is not None:
        cluster_values = sampled[label_col].map(normalize_label)
        codes, uniques = pd.factorize(cluster_values)
    else:
        cluster_values = pd.Series(["account"] * len(sampled), index=sampled.index)
        codes = np.zeros(len(sampled), dtype=int)
        uniques = np.array(["account"])
    suspicious_mask = np.zeros(len(sampled), dtype=bool)

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    normal = ~suspicious_mask
    scatter = ax.scatter(
        coords[normal, 0],
        coords[normal, 1],
        c=codes[normal],
        cmap="tab20" if len(uniques) <= 20 else "viridis",
        s=22,
        alpha=0.72,
        linewidths=0,
    )
    if suspicious_mask.any():
        ax.scatter(
            coords[suspicious_mask, 0],
            coords[suspicious_mask, 1],
            c=codes[suspicious_mask],
            cmap="tab20" if len(uniques) <= 20 else "viridis",
            s=42,
            alpha=0.92,
            edgecolors="#D64545",
            linewidths=1.1,
            label="Suspicious mechanism cluster",
        )
        ax.legend(loc="best", frameon=True)

    ax.set_title(f"Account feature embedding ({method})")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(alpha=0.2)
    if len(uniques) <= 12:
        handles = []
        for idx, name in enumerate(uniques):
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=scatter.cmap(scatter.norm(idx)),
                    markeredgecolor="none",
            label=str(name),
                    markersize=6,
                )
            )
        ax.legend(handles=handles, title="cluster", loc="best", frameon=True)
    elif cluster_col is not None:
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("label code")

    fig.tight_layout()
    path = figures_dir / "account_embedding_umap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    if label_col is not None:
        label_note = "颜色来自独立标签揭示元数据，仅用于最后揭示评估，不参与无监督发现。"
    else:
        label_note = "未读取到账户级标签揭示元数据，散点颜色不表示真实标签。"
    note = f"使用 {len(numeric_cols)} 个数值账户特征绘制；降维方法：{method}。{label_note}"
    return path, note


def cluster_label_eval_table(label_eval: pd.DataFrame) -> pd.DataFrame:
    if label_eval.empty or "row_type" not in label_eval.columns:
        return pd.DataFrame()
    rows = label_eval[label_eval["row_type"].astype(str).eq("cluster")].copy()
    if rows.empty:
        return pd.DataFrame()
    rows = add_cluster_key(rows, "cluster_id")
    cols = [
        "suspicious_rank",
        CLUSTER_KEY_COL,
        "view",
        "k",
        "cluster_id",
        "selected_account_count",
        "human_count",
        "bot_count",
        "bot_rate",
        "bot_precision",
        "bot_recall",
    ]
    return rows[[col for col in cols if col in rows.columns]].head(20)


def union_label_eval_table(label_eval: pd.DataFrame) -> pd.DataFrame:
    if label_eval.empty or "row_type" not in label_eval.columns:
        return pd.DataFrame()
    rows = label_eval[label_eval["row_type"].astype(str).eq("top_clusters_union")].copy()
    if rows.empty:
        return pd.DataFrame()
    cols = [
        "top_cluster_count",
        "selected_account_count",
        "human_count",
        "bot_count",
        "bot_rate",
        "bot_precision",
        "bot_recall",
    ]
    return rows[[col for col in cols if col in rows.columns]]


def top_cluster_table(scores: pd.DataFrame, purity: pd.DataFrame) -> pd.DataFrame:
    score_cluster_col = get_cluster_column(scores)
    ordered = add_cluster_key(order_cluster_scores(scores).copy(), score_cluster_col)
    if score_cluster_col is None:
        return ordered.head(12)
    ordered = ordered.rename(columns={score_cluster_col: "cluster_id"})
    cols = [CLUSTER_KEY_COL, "cluster_id"]
    for col in [
        sorting_score_column(ordered),
        density_column(ordered),
        first_numeric_by_tokens(ordered, ["null", "mean"]),
        first_numeric_by_tokens(ordered, ["null", "p95"]) or first_numeric_by_tokens(ordered, ["null", "q95"]),
        find_column(ordered, ["empirical_p_value", "p_value", "pvalue"]),
        find_column(ordered, ["account_count", "cluster_size", "size", "n_accounts"]),
    ]:
        if col is not None and col in ordered.columns and col not in cols:
            cols.append(col)
    if not purity.empty and "revealed_bot_purity" in purity.columns:
        purity_keyed = add_cluster_key(purity, "cluster_id")
        ordered = ordered.merge(
            purity_keyed[[CLUSTER_KEY_COL, "revealed_bot_purity"]].drop_duplicates(CLUSTER_KEY_COL),
            on=CLUSTER_KEY_COL,
            how="left",
        )
        cols.append("revealed_bot_purity")
    return ordered[[col for col in cols if col in ordered.columns]].head(12)


def write_report(
    output_dir: Path,
    data: dict[str, pd.DataFrame],
    figure_notes: dict[str, str],
    purity: pd.DataFrame,
) -> Path:
    scores = data["scores"]
    assignments = data["assignments"]
    features = data["features"]
    edges = data["edges"]
    label_eval = data["label_eval"]
    cluster_col = get_cluster_column(assignments)
    score_cluster_col = get_cluster_column(scores)
    suspicious_ids = suspicious_cluster_keys(scores)
    total_accounts = len(features) if len(features) else len(assignments)
    total_clusters = len(add_cluster_key(assignments, cluster_col)[CLUSTER_KEY_COL].dropna().unique()) if cluster_col else scores.shape[0]

    lines = [
        "# 无监督账户机制簇发现报告",
        "",
        "## 1. 口径与边界",
        "",
        "- 第一阶段数据口径：`original_human` vs `original_bot`，即原始人类账号与旧版原始机器人账号的机制差异发现。",
        "- 无监督发现阶段不使用任何账号标签；聚类、密度/null 对比、相似边和 suspicious score 都必须只由账户特征与相似结构得到。",
        "- `label_revealed_evaluation.csv` 只用于最后揭示评估，用来解释 suspicious mechanism clusters 的标签纯度与召回线索。",
        "- 本报告的结论单位是 `suspicious mechanism clusters`。即使某个簇的 revealed bot share 较高，也不宣称单个账号绝对是 bot。",
        "",
        "## 2. 输入文件",
        "",
        f"- 输出目录：`{output_dir}`",
        f"- suspicious_cluster_scores.csv：`{scores.shape[0]}` 行 x `{scores.shape[1]}` 列",
        f"- label_revealed_evaluation.csv：`{label_eval.shape[0]}` 行 x `{label_eval.shape[1]}` 列",
    f"- cluster_assignments.csv：`{assignments.shape[0]}` 行 x `{assignments.shape[1]}` 列",
    f"- account_features.csv：`{features.shape[0]}` 行 x `{features.shape[1]}` 列",
    f"- account_similarity_edges.csv：`{edges.shape[0]}` 行 x `{edges.shape[1]}` 列",
        f"- label_revealed_account_metadata.csv：`{data['account_metadata'].shape[0]}` 行 x `{data['account_metadata'].shape[1]}` 列，若存在则只用于最终揭示评估",
        "",
        "## 3. 发现摘要",
        "",
        f"- 账户数：`{total_accounts}`",
        f"- 机制簇数：`{total_clusters}`",
        f"- suspicious mechanism clusters 数：`{len(suspicious_ids) if suspicious_ids else scores.shape[0]}`",
        "",
        "### suspicious cluster 排名前列",
        "",
        markdown_table(top_cluster_table(scores, purity), max_rows=12),
        "",
        "### label reveal 单簇评估摘录",
        "",
        "标签只在这里揭示，用来检查无监督 suspicious ranking 排名前列到底对应什么账号组成。",
        "",
        markdown_table(cluster_label_eval_table(label_eval), max_rows=20),
        "",
        "### Top-N suspicious clusters 并集评估",
        "",
        "这一表按 suspicious ranking 取前 N 个簇的账号并集，展示精度和召回之间的取舍。",
        "",
        markdown_table(union_label_eval_table(label_eval), max_rows=10),
        "",
        "## 4. 图表",
        "",
        "### 4.1 Cluster density / null distribution",
        "",
        f"{figure_notes.get('cluster_density_null_distribution.png', '')}",
        "",
        "![cluster density null distribution](figures/cluster_density_null_distribution.png)",
        "",
        "### 4.2 Suspicious cluster purity",
        "",
        f"{figure_notes.get('suspicious_cluster_purity.png', '')}",
        "",
        "![suspicious cluster purity](figures/suspicious_cluster_purity.png)",
        "",
        "### 4.3 Feature-view agreement",
        "",
        f"{figure_notes.get('feature_view_agreement_heatmap.png', '')}",
        "",
        "![feature view agreement heatmap](figures/feature_view_agreement_heatmap.png)",
        "",
        "### 4.4 Account embedding",
        "",
        f"{figure_notes.get('account_embedding_umap.png', '')}",
        "",
        "![account embedding umap](figures/account_embedding_umap.png)",
        "",
        "## 5. 结论表述模板",
        "",
        "- 可以说：无监督流程发现了若干与随机 null baseline 或多视角相似结构相比更集中的 suspicious mechanism clusters。",
        "- 可以说：标签揭示后，这些 suspicious mechanism clusters 中有一部分呈现较高的 bot purity，说明它们捕捉到了 original_bot 的生成机制痕迹。",
        "- 不应说：模型已经在无监督阶段识别出所有 bot，或某个单账号必然是 bot。",
        "- 后续若扩展到新 family，应继续沿用“先无标签发现机制簇，再揭示标签评估”的顺序，避免把 label 信息泄漏进发现阶段。",
    ]
    report_path = output_dir / "UNSUPERVISED_ACCOUNT_DISCOVERY_CN.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    output_dir = Path(args.output_dir).expanduser().resolve()
    data = read_required_outputs(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_notes: dict[str, str] = {}
    density_path, note = plot_cluster_density_null_distribution(data["scores"], figures_dir)
    figure_notes[density_path.name] = note

    purity_path, purity, note = plot_suspicious_cluster_purity(
        data["label_eval"], data["assignments"], data["scores"], figures_dir
    )
    figure_notes[purity_path.name] = note

    agreement_path, note = plot_feature_view_agreement_heatmap(
        data["assignments"], data["features"], data["edges"], figures_dir
    )
    figure_notes[agreement_path.name] = note

    embedding_path, note = plot_account_embedding(
        data["features"],
        data["account_metadata"],
        data["assignments"],
        data["scores"],
        figures_dir,
        max_points=args.max_embedding_points,
    )
    figure_notes[embedding_path.name] = note

    report_path = write_report(output_dir, data, figure_notes, purity)
    print("Generated:")
    for path in [density_path, purity_path, agreement_path, embedding_path, report_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
