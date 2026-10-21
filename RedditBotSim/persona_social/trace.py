from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import json
from datetime import datetime
from pathlib import Path
from typing import Any


def json_default(value: Any) -> str:
    return str(value)


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")

