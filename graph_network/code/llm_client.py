#!/usr/bin/env python3
"""Portable provider LLM client for static graph dataset generation."""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib import error, request


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


@dataclass
class LLMCallResult:
    provider: str
    model: str
    temperature: float
    prompt: str
    raw_response: str
    parsed_text: str
    status: str
    error: str = ""
    fallback_reason: str = ""


def first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def strip_code_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_text_response(raw: str, preferred_keys: tuple[str, ...] = ("text", "body", "comment", "content")) -> tuple[str, str]:
    text = strip_code_fence(raw)
    if not text:
        return "", "empty_response"
    try:
        obj = json.loads(text)
        if isinstance(obj, Mapping):
            if "title" in obj and "body" in obj:
                return json.dumps({"title": str(obj["title"]).strip(), "body": str(obj["body"]).strip()}, ensure_ascii=False), ""
            for key in preferred_keys:
                value = obj.get(key)
                if value not in (None, ""):
                    return str(value).strip(), ""
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, Mapping):
                if "title" in obj and "body" in obj:
                    return json.dumps({"title": str(obj["title"]).strip(), "body": str(obj["body"]).strip()}, ensure_ascii=False), ""
                for key in preferred_keys:
                    value = obj.get(key)
                    if value not in (None, ""):
                        return str(value).strip(), ""
        except Exception as exc:
            return text, f"json_extract_error: {exc}"
    return text, ""


class GraphLLMClient:
    def __init__(self, *, max_tokens=2048, timeout=90.0, retries=2, reasoning_effort="medium", api="responses"):
        self.max_tokens=max_tokens; self.timeout=timeout; self.retries=max(0,int(retries))
        self.reasoning_effort=reasoning_effort;self.api=api

    def _api_key(self,provider):
        return os.environ.get(provider.upper()+"_API_KEY", "")

    def provider_available(self,provider):
        return bool(self._api_key(provider))

    def generate(self, *, provider, model, temperature, prompt, fallback, preferred_keys=("text","body","comment","content")):
        from provider_client import build_llm_client
        last_error=""
        for attempt in range(self.retries+1):
            try:
                config={"llm":{"provider":provider,"max_tokens":self.max_tokens,"timeout":self.timeout,"reasoning_effort":self.reasoning_effort}}
                client,_=build_llm_client(config,provider=provider,model=model,temperature=temperature)
                raw=client.generate_text(prompt,api=self.api)
                parsed,parse_error=parse_text_response(raw,preferred_keys)
                if parsed:
                    return LLMCallResult(provider,model,temperature,prompt,raw,parsed,"ok" if attempt==0 else "ok_after_retry",error=parse_error)
                last_error=parse_error or "empty response"
            except Exception as exc:
                last_error=redact_error(exc)
        return LLMCallResult(provider,model,temperature,prompt,"",fallback,"fallback",error=last_error,fallback_reason="api_error")
