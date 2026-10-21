#!/usr/bin/env python3
"""End-to-end pipeline for the static graph network dataset project."""

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
import subprocess
import sys
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "graph_static_dataset_v1.yaml"
DEFAULT_INPUT = PROJECT_DIR / "inputs" / "original_botsim_snapshot"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs"
DEFAULT_RESULT_ROOT = PROJECT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default="pipeline_smoke")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--bots-per-family", type=int, default=3)
    parser.add_argument("--family-counts", default="")
    parser.add_argument("--time-windows", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--live-llm", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-models", action="store_true", help="Only generate/export/split/check dataset; do not run detectors.")
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--gnn-epochs", type=int, default=20)
    parser.add_argument("--gnn-patience", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def ensure_inside_project(path: Path, label: str) -> Path:
    return path.expanduser().resolve()


def run_cmd(cmd: list[str], *, cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    global DEFAULT_INPUT, DEFAULT_OUTPUT_ROOT, DEFAULT_RESULT_ROOT
    DEFAULT_INPUT = Path(config["paths"]["input_root"])
    DEFAULT_OUTPUT_ROOT = Path(config["paths"]["output_root"])
    DEFAULT_RESULT_ROOT = Path(config["paths"]["result_root"])
    args.config = Path(args.config).resolve()
    run_dir = ensure_inside_project(args.output_dir or (DEFAULT_OUTPUT_ROOT / args.run_name), "output-dir")
    result_dir = ensure_inside_project(args.result_dir or (DEFAULT_RESULT_ROOT / args.run_name), "result-dir")
    audit_dir = ensure_inside_project(DEFAULT_OUTPUT_ROOT / "human_behavior_audit", "audit-dir")
    py = args.python

    if not args.skip_audit:
        run_cmd(
            [
                py,
                str(CODE_DIR / "audit_human_behavior.py"),
                "--input-root",
                str(DEFAULT_INPUT),
                "--output-dir",
                str(audit_dir),
            ],
            cwd=PROJECT_DIR,
        )

    generation_cmd = [
        py,
        str(CODE_DIR / "mixed_environment_runner.py"),
        "--config",
        str(args.config),
        "--output-dir",
        str(run_dir),
        "--run-name",
        str(args.run_name),
        "--bots-per-family",
        str(args.bots_per_family),
        "--time-windows",
        str(args.time_windows),
        "--seed",
        str(args.seed),
    ]
    if args.family_counts:
        generation_cmd.extend(["--family-counts", args.family_counts])
    if args.live_llm:
        generation_cmd.append("--live-llm")
    if args.dry_run:
        generation_cmd.append("--dry-run")
    run_cmd(generation_cmd, cwd=PROJECT_DIR)

    run_cmd([py, str(CODE_DIR / "export_static_graph_dataset.py"), "--run-dir", str(run_dir)], cwd=PROJECT_DIR)
    run_cmd([py, str(CODE_DIR / "build_graph_splits.py"), "--run-dir", str(run_dir), "--seed", str(args.seed)], cwd=PROJECT_DIR)
    check_cmd = [
        py,
        str(CODE_DIR / "check_dataset_integrity.py"),
        "--run-dir",
        str(run_dir),
        "--result-dir",
        str(result_dir),
    ]
    if args.live_llm:
        check_cmd.append("--expect-live-llm")
    run_cmd(check_cmd, cwd=PROJECT_DIR)
    if args.skip_models:
        summary = {
            "run_dir": str(run_dir),
            "result_dir": str(result_dir),
            "dataset_manifest": read_json(run_dir / "dataset_manifest.json"),
            "graph_export_summary": read_json(run_dir / "graph" / "export_summary.json"),
            "split_summary": read_json(run_dir / "splits" / "split_summary.json"),
            "integrity_summary": read_json(result_dir / "dataset_integrity_summary.json"),
            "report": str(result_dir / "dataset_integrity_report_cn.md"),
        }
        (result_dir / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return
    run_cmd(
        [
            py,
            str(CODE_DIR / "run_graphstats_baselines.py"),
            "--run-dir",
            str(run_dir),
            "--result-dir",
            str(result_dir),
            "--seed",
            str(args.seed),
        ],
        cwd=PROJECT_DIR,
    )

    if not args.skip_gnn:
        run_cmd(
            [
                py,
                str(CODE_DIR / "run_rgcn_transductive.py"),
                "--run-dir",
                str(run_dir),
                "--result-dir",
                str(result_dir),
                "--epochs",
                str(args.gnn_epochs),
                "--patience",
                str(args.gnn_patience),
                "--hidden-dim",
                str(args.hidden_dim),
                "--seed",
                str(args.seed),
            ],
            cwd=PROJECT_DIR,
        )
        run_cmd(
            [
                py,
                str(CODE_DIR / "run_graphsage_inductive.py"),
                "--run-dir",
                str(run_dir),
                "--result-dir",
                str(result_dir),
                "--epochs",
                str(args.gnn_epochs),
                "--patience",
                str(args.gnn_patience),
                "--hidden-dim",
                str(args.hidden_dim),
                "--seed",
                str(args.seed),
            ],
            cwd=PROJECT_DIR,
        )

    run_cmd(
        [
            py,
            str(CODE_DIR / "build_static_graph_report.py"),
            "--run-dir",
            str(run_dir),
            "--result-dir",
            str(result_dir),
        ],
        cwd=PROJECT_DIR,
    )

    summary = {
        "run_dir": str(run_dir),
        "result_dir": str(result_dir),
        "dataset_manifest": read_json(run_dir / "dataset_manifest.json"),
        "graph_export_summary": read_json(run_dir / "graph" / "export_summary.json"),
        "split_summary": read_json(run_dir / "splits" / "split_summary.json"),
        "report": str(result_dir / "report_cn.md"),
    }
    (result_dir / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
