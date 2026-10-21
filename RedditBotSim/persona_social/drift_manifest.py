from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import json
from pathlib import Path
from typing import Any

import pandas as pd

from .drift_schema import AlignmentGateResult, DriftAuditResult, MutationPlan, json_safe


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_attempt_artifacts(
    attempt_dir: Path,
    plan: MutationPlan,
    audit: DriftAuditResult,
    gate: AlignmentGateResult,
) -> None:
    write_json(attempt_dir / "mutation_plan.json", plan.to_dict())
    write_json(attempt_dir / "drift_audit.json", audit.to_dict())
    write_json(attempt_dir / "alignment_gate.json", gate.to_dict())


def append_manifest_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([json_safe(row)])
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
