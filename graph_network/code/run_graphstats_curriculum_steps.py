#!/usr/bin/env python3
"""Evaluate GraphStats on nested incremental curriculum steps.

The regular baseline runner evaluates each split's top-level train/test ids.
Incremental curriculum split files also contain per-step ids under
``curriculum_steps``. This script expands those steps so we can measure
blind recall before each family is added to the training pool.
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
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_graphstats_baselines import (
    DEFAULT_SEED,
    add_metric_rows,
    build_graph_features,
    ensure_dir,
    fit_graphstats_models,
    load_dataset,
    predict_proba,
    split_indices,
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def curriculum_split_paths(run_dir: Path) -> list[Path]:
    split_dir = run_dir / "splits"
    return sorted(split_dir.glob("incremental_curriculum_order*.json"))


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    result_dir = ensure_dir(Path(args.result_dir).resolve())
    nodes, edges, _manifest = load_dataset(run_dir)
    features, _audit = build_graph_features(nodes, edges)
    y = nodes["label"].to_numpy(dtype=int)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for split_path in curriculum_split_paths(run_dir):
        split = read_json(split_path)
        order_name = split.get("split_name", split_path.stem)
        order = split.get("curriculum_order", [])
        for step in split.get("curriculum_steps", []):
            family = step.get("add_family_after_evaluation", "")
            step_payload = {
                "split_name": f"{order_name}_step{step.get('step')}",
                "setting": "incremental_curriculum_step",
                "holdout_families": [family],
                "train_node_ids": step.get("train_node_ids", []),
                "val_node_ids": step.get("val_node_ids", []),
                "test_node_ids": step.get("test_node_ids", []),
                "unlabeled_visible_node_ids": step.get("unlabeled_visible_node_ids", []),
                "curriculum_order": order,
                "curriculum_step": step.get("step"),
                "add_family_after_evaluation": family,
            }
            idx = split_indices(step_payload, nodes)
            models, statuses = fit_graphstats_models(features, y, idx["train"], args.seed, args.max_ebm_rounds)
            status_map = {row["model"]: row for row in statuses}
            for model_name, model in models.items():
                pred, prob = predict_proba(model, features)
                status = status_map.get(model_name, {"status": "ok", "reason": ""})
                backend = "interpret" if model_name == "GraphStats_EBM" else "sklearn"
                for group in ["val", "test"]:
                    before = len(metric_rows)
                    add_metric_rows(
                        metric_rows,
                        step_payload,
                        model_name,
                        backend,
                        y,
                        pred,
                        prob,
                        idx[group],
                        nodes,
                        group,
                        status.get("status", "ok"),
                        status.get("reason", ""),
                    )
                    for row in metric_rows[before:]:
                        row["curriculum_order"] = ">".join(order)
                        row["curriculum_step"] = step.get("step")
                        row["add_family_after_evaluation"] = family
                idx_sets = {k: set(v.tolist()) for k, v in idx.items()}
                for i, node in nodes.iloc[idx["test"]].iterrows():
                    prediction_rows.append(
                        {
                            "split_name": step_payload["split_name"],
                            "model": model_name,
                            "backend": backend,
                            "node_id": node["user_id"],
                            "true_label": int(node["label"]),
                            "predicted_label": int(pred[i]),
                            "bot_probability": float(prob[i]),
                            "family": node.get("family", ""),
                            "curriculum_order": ">".join(order),
                            "curriculum_step": step.get("step"),
                            "add_family_after_evaluation": family,
                            "in_train": bool(i in idx_sets["train"]),
                            "in_val": bool(i in idx_sets["val"]),
                            "in_test": bool(i in idx_sets["test"]),
                        }
                    )

    pd.DataFrame(metric_rows).to_csv(result_dir / "graphstats_curriculum_step_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(result_dir / "graphstats_curriculum_step_predictions.csv", index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-ebm-rounds", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
