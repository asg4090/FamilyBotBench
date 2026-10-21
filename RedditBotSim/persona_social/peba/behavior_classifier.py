from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import json
from typing import Any

import pandas as pd

from .behavior_taxonomy import BEHAVIOR_CLASSES, behavior_label_descriptions, classify_from_features


def classify_trajectories_rule(summaries: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in summaries.to_dict(orient="records"):
        decision = classify_from_features(row)
        rows.append(
            {
                "user_id": row["user_id"],
                "family": row.get("family", ""),
                "family_generation": row.get("family_generation", ""),
                "behavior_label": decision.label,
                "confidence": decision.confidence,
                "classifier": "rule_v1",
                "ranking": json.dumps(decision.ranking, ensure_ascii=False),
                "evidence": json.dumps(decision.evidence, ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def build_classifier_prompt(summary: dict[str, Any]) -> str:
    descriptions = behavior_label_descriptions()
    labels = "\n".join(f"- {label}: {descriptions[label]}" for label in BEHAVIOR_CLASSES)
    visible = {key: summary.get(key) for key in [
        "post_count",
        "comment1_count",
        "comment2_count",
        "revisit_count",
        "skip_count",
        "unique_subreddits",
        "unique_topics",
        "unique_target_authors",
        "reciprocal_neighbor_count_proxy",
        "mean_comment_words",
        "action_counts_json",
        "source_counts_json",
    ]}
    return f"""You are classifying a Reddit user's simulated social trajectory.

Choose exactly one behavior label from this list:
{labels}

Trajectory summary:
{json.dumps(visible, ensure_ascii=False, indent=2)}

Return JSON only:
{{"behavior_label": "...", "confidence": 0.0, "ranking": ["..."], "evidence": "short reason"}}
"""


def classify_trajectories_llm(summaries: pd.DataFrame, client: Any, *, api: str = "chat") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in summaries.to_dict(orient="records"):
        prompt = build_classifier_prompt(summary)
        try:
            raw = client.generate_text(prompt, api=api, settings_override={"temperature": 0.0})
            parsed = json.loads(str(raw).strip())
            label = str(parsed.get("behavior_label", ""))
            if label not in BEHAVIOR_CLASSES:
                raise ValueError(f"invalid behavior label: {label}")
            ranking = parsed.get("ranking") if isinstance(parsed.get("ranking"), list) else [label]
            confidence = float(parsed.get("confidence", 0.5))
            evidence = parsed.get("evidence", "")
            classifier = "llm_v1"
        except Exception:
            fallback = classify_from_features(summary)
            label = fallback.label
            ranking = fallback.ranking
            confidence = fallback.confidence
            evidence = fallback.evidence
            classifier = "llm_v1_fallback_rule"
        rows.append(
            {
                "user_id": summary["user_id"],
                "family": summary.get("family", ""),
                "family_generation": summary.get("family_generation", ""),
                "behavior_label": label,
                "confidence": max(0.0, min(1.0, confidence)),
                "classifier": classifier,
                "ranking": json.dumps(ranking, ensure_ascii=False),
                "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if not isinstance(evidence, str) else evidence,
            }
        )
    return pd.DataFrame(rows)
