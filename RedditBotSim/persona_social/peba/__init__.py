"""PEBA/PEvo-style population alignment tools for RedditBotSim."""

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


from .behavior_taxonomy import BEHAVIOR_CLASSES
from .peba_metrics import distribution_gap

__all__ = ["BEHAVIOR_CLASSES", "distribution_gap"]
