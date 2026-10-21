"""Compatibility exports for the portable provider client."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from provider_client import LLMSettings, UnifiedLLMClient, load_llm_settings, load_local_env, build_llm_client, build_openai_client
