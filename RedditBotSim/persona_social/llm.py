from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import asyncio
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResult:
    raw_text: str
    parsed: dict[str, Any] | None
    content: str
    error: str | None = None
    meta: dict[str, Any] | None = None


def parse_json_content(raw_text: str) -> LLMResult:
    text = str(raw_text or "").strip()
    parsed: dict[str, Any] | None = None
    error: str | None = None
    try:
        parsed_obj = json.loads(text)
        if isinstance(parsed_obj, dict):
            parsed = parsed_obj
    except Exception as exc:
        error = f"json_parse_error: {exc}"

    if parsed is None:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed_obj = json.loads(text[start : end + 1])
                if isinstance(parsed_obj, dict):
                    parsed = parsed_obj
                    error = None
            except Exception as exc:
                error = f"json_extract_error: {exc}"

    content = ""
    if parsed is not None:
        value = parsed.get("content", "")
        content = str(value).strip()
    elif text:
        content_match = None
        marker = '"content"'
        marker_pos = text.find(marker)
        if marker_pos >= 0:
            quote_pos = text.find('"', marker_pos + len(marker))
            if quote_pos >= 0:
                content_match = text[quote_pos + 1 :].strip().rstrip('"}').strip()
        content = content_match or text.strip().strip("`").strip()
        parsed = {"content": content}
    return LLMResult(raw_text=text, parsed=parsed, content=content, error=error)


def _extract_schema_from_prompt(prompt: str) -> str:
    marker = "Return valid JSON only, exactly in this schema:"
    if marker not in prompt:
        return '{"content": "text"}'
    return prompt.split(marker, 1)[-1].strip()[:1200] or '{"content": "text"}'


def build_json_repair_prompt(original_prompt: str, malformed_response: str) -> str:
    schema = _extract_schema_from_prompt(original_prompt)
    return f"""You need to repair one malformed JSON response from a Reddit social simulation.

Original task prompt:
{original_prompt[-6000:]}

Malformed response:
{malformed_response}

Return one valid JSON object only.
Required schema:
{schema}

Repair rules:
- Preserve any selected action, target_post_id, and target_comment_id that are already visible in the malformed response.
- If content is missing for a comment or post, write one short natural Reddit-style sentence consistent with the original task and selected target.
- If the selected action is skip, content must be an empty string.
- Do not include markdown fences, commentary, or extra keys except optional "_repair_applied": true.
"""


def _extract_browse_items_from_prompt(prompt: str) -> list[dict[str, Any]]:
    marker = "You saw these candidate items:"
    end_marker = "\n\nDecision:"
    if marker not in prompt or end_marker not in prompt:
        return []
    payload = prompt.split(marker, 1)[1].split(end_marker, 1)[0].strip()
    try:
        parsed = json.loads(payload)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _dry_run_result(prompt: str, fallback: str) -> LLMResult:
    if '"target_post_id"' not in prompt or '"action"' not in prompt:
        raw = json.dumps({"content": fallback}, ensure_ascii=False)
        return LLMResult(raw_text=raw, parsed={"content": fallback, "_dry_run_applied": True}, content=fallback)
    items = _extract_browse_items_from_prompt(prompt)
    selected = items[0] if items else {}
    action = "skip"
    target_post_id = ""
    target_comment_id = ""
    content = ""
    reason = "dry_run_no_candidate"
    for item in items:
        if item.get("first_level_comment_id"):
            selected = item
            action = "comment2"
            target_post_id = str(item.get("post_id", "") or "")
            target_comment_id = str(item.get("first_level_comment_id", "") or "")
            content = "I don't totally buy that framing."
            reason = "dry_run_first_visible_comment"
            break
        comments = item.get("thread_comments")
        if isinstance(comments, list) and comments:
            comment = comments[0] if isinstance(comments[0], dict) else {}
            selected = item
            action = "comment2"
            target_post_id = str(item.get("post_id", "") or "")
            target_comment_id = str(comment.get("comment_id", "") or "")
            content = "Yeah, that part is doing a lot of work."
            reason = "dry_run_thread_comment"
            break
    else:
        if selected:
            action = "comment1"
            target_post_id = str(selected.get("post_id", "") or "")
            content = "Feels like there is more going on here than the headline says."
            reason = "dry_run_first_post"
    parsed = {
        "action": action,
        "target_post_id": target_post_id,
        "target_comment_id": target_comment_id,
        "reason": reason,
        "content": content,
        "_dry_run_applied": True,
    }
    raw = json.dumps(parsed, ensure_ascii=False)
    return LLMResult(raw_text=raw, parsed=parsed, content=content)


class AsyncLLMGenerator:
    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float,
        reasoning_effort: str,
        api: str,
        concurrency: int,
        retries: int,
        dry_run: bool,
        timeout: float | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.api = api
        self.retries = retries
        self.dry_run = dry_run
        self.timeout = timeout
        self._sem = asyncio.Semaphore(max(1, concurrency))

    async def generate(self, prompt: str, fallback: str, settings_override: dict[str, Any] | None = None) -> LLMResult:
        if self.dry_run:
            return _dry_run_result(prompt, fallback)
        async with self._sem:
            last_error: str | None = None
            last_raw = ""
            provider_attempts: list[dict[str, Any]] = []
            provider_overrides = self._candidate_provider_overrides(settings_override)
            for _attempt in range(self.retries + 1):
                for provider_override in provider_overrides:
                    raw = ""
                    attempt_meta = {
                        "attempt": _attempt,
                        "provider": provider_override.get("provider"),
                        "model": provider_override.get("model"),
                        "status": "started",
                    }
                    provider_attempts.append(attempt_meta)
                    try:
                        call = asyncio.to_thread(self._call_sync, prompt, provider_override)
                        raw = await asyncio.wait_for(call, timeout=self.timeout) if self.timeout else await call
                        last_raw = str(raw or "")
                        attempt_meta["raw_length"] = len(last_raw)
                        result = parse_json_content(raw)
                        if result.error is not None and raw.strip():
                            attempt_meta["status"] = "json_error"
                            attempt_meta["error"] = result.error[:500]
                            attempt_meta["raw_preview"] = last_raw[:500]
                            repaired = await self._try_repair(prompt, raw, provider_override)
                            if repaired is not None:
                                attempt_meta["status"] = "ok_repaired"
                                attempt_meta["repair_raw_length"] = len(repaired.raw_text or "")
                                repaired.meta = {"provider_attempts": provider_attempts}
                                return repaired
                        if result.error is not None:
                            last_error = result.error
                            continue
                        if result.parsed is not None and str(result.parsed.get("action", "")).strip().lower() == "skip":
                            attempt_meta["status"] = "ok"
                            result.meta = {"provider_attempts": provider_attempts}
                            return result
                        if result.content:
                            attempt_meta["status"] = "ok"
                            result.meta = {"provider_attempts": provider_attempts}
                            return result
                        last_error = result.error or "empty_content"
                        attempt_meta["status"] = "empty_content"
                        attempt_meta["error"] = last_error[:500]
                    except Exception as exc:
                        last_error = redact_error(exc)
                        attempt_meta["status"] = "error"
                        attempt_meta["error"] = last_error[:500]
                await asyncio.sleep(0.5)
            parsed_fallback = {
                "content": fallback,
                "_fallback_applied": True,
                "_last_error": last_error,
                "_last_raw_preview": last_raw[:1200],
            }
            return LLMResult(
                raw_text=last_raw or json.dumps({"content": fallback}),
                parsed=parsed_fallback,
                content=fallback,
                error=f"fallback_after_error: {last_error}",
                meta={"provider_attempts": provider_attempts},
            )

    async def _try_repair(
        self,
        prompt: str,
        malformed_response: str,
        settings_override: dict[str, Any] | None = None,
    ) -> LLMResult | None:
        repair_prompt = build_json_repair_prompt(prompt, malformed_response)
        repair_settings = dict(settings_override or {})
        repair_settings["temperature"] = min(float(repair_settings.get("temperature", self.temperature)), 0.2)
        try:
            call = asyncio.to_thread(self._call_sync, repair_prompt, repair_settings)
            raw = await asyncio.wait_for(call, timeout=self.timeout) if self.timeout else await call
            result = parse_json_content(raw)
        except Exception:
            return None
        if result.error is not None or result.parsed is None:
            return None
        if not result.content and str(result.parsed.get("action", "")).strip().lower() != "skip":
            return None
        result.parsed["_repair_applied"] = True
        result.parsed["_malformed_preview"] = str(malformed_response or "")[:500]
        result.raw_text = raw
        return result

    def _call_sync(self, prompt: str, settings_override: dict[str, Any] | None = None) -> str:
        effective_model = str(settings_override.get("model", self.model)) if settings_override else self.model
        effective_temperature = float(settings_override.get("temperature", self.temperature)) if settings_override else self.temperature
        effective_reasoning_effort = str(settings_override.get("reasoning_effort", self.reasoning_effort)) if settings_override else self.reasoning_effort
        if hasattr(self.client, "generate_text"):
            return self.client.generate_text(prompt, api=self.api, settings_override=settings_override)
        if self.api == "responses":
            response = self.client.responses.create(
                model=effective_model,
                input=prompt,
                reasoning={"effort": effective_reasoning_effort},
            )
            return response.output_text
        completion = self.client.chat.completions.create(
            model=effective_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=effective_temperature,
        )
        return completion.choices[0].message.content

    def _candidate_provider_overrides(self, settings_override: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        base = dict(settings_override or {})
        raw_clients = getattr(self.client, "raw_clients", {})
        if not isinstance(raw_clients, dict):
            return [base]
        available = [provider for provider in raw_clients if provider in {"openai", "gemini"}]
        preferred = str(base.get("provider", "") or "")
        ordered: list[str] = []
        for provider in [preferred, "openai", "gemini"]:
            if provider and provider in available and provider not in ordered:
                ordered.append(provider)
        if not ordered:
            return [base]
        out = []
        for provider in ordered:
            item = dict(base)
            item["provider"] = provider
            provider_settings = getattr(self.client, "provider_settings", {}).get(provider)
            if provider != preferred or not item.get("model"):
                model = getattr(provider_settings, "model", None)
                if model:
                    item["model"] = str(model)
                elif provider == "gemini":
                    item["model"] = "gemini-3-flash-preview"
                else:
                    item["model"] = "gpt-5.5"
            out.append(item)
        return out
