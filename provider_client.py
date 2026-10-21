"""Environment-only provider configuration; no private gateway or shared-key fallback."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
import os
from release_support import redact_error

DEFAULTS = {
    'openai': ('https://api.openai.com/v1', 'gpt-5.4'),
    'anthropic': ('https://api.anthropic.com', 'claude-sonnet-4-5-20250929'),
    'gemini': ('https://generativelanguage.googleapis.com', 'gemini-3-flash-preview'),
}

@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str = field(repr=False)
    base_url: str
    model: str
    temperature: float
    reasoning_effort: str
    max_tokens: int = 1536
    timeout: float | None = 90.0

def load_local_env(env_path=None):
    """Compatibility shim. Credentials are read only from the process environment."""
    return None

def load_llm_settings(config=None, provider_override=None, model_override=None, temperature_override=None):
    llm = (config or {}).get('llm', {}) or {}
    provider = str(provider_override or os.environ.get('LLM_PROVIDER') or llm.get('provider','openai')).lower()
    if provider not in DEFAULTS: raise ValueError(f'Unsupported LLM provider: {provider}')
    prefix = provider.upper()
    # Each provider has its own credential. YAML may name a variable, never contain its value.
    env_field = 'api_key_env' if provider == 'openai' else provider + '_api_key_env'
    key_name = llm.get(env_field, prefix + '_API_KEY')
    key = os.environ.get(key_name, '')
    if not key: raise RuntimeError(f'Missing {key_name} for provider={provider}.')
    if any(llm.get(k) for k in ['api_key','gemini_api_key','anthropic_api_key']):
        raise ValueError('Literal credentials in YAML are unsupported; set provider environment variables.')
    model_field = 'model' if provider == 'openai' else provider + '_model'
    return LLMSettings(
        provider=provider, api_key=key,
        base_url=(os.environ.get(prefix+'_BASE_URL') or DEFAULTS[provider][0]).rstrip('/'),
        model=str(model_override or os.environ.get(prefix+'_MODEL') or llm.get(model_field) or DEFAULTS[provider][1]),
        temperature=float(temperature_override if temperature_override is not None else os.environ.get(prefix+'_TEMPERATURE',os.environ.get('LLM_TEMPERATURE',llm.get('temperature',0.9)))),
        reasoning_effort=str(os.environ.get(prefix+'_REASONING_EFFORT',llm.get('reasoning_effort','medium'))),
        max_tokens=int(os.environ.get('LLM_MAX_TOKENS',llm.get('max_tokens',1536))),
        timeout=float(os.environ.get('LLM_TIMEOUT',llm.get('timeout',90))),
    )

def _make_client(settings):
    if settings.provider == 'openai':
        from openai import OpenAI
        return OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=settings.timeout, max_retries=0)
    if settings.provider == 'anthropic':
        from anthropic import Anthropic
        return Anthropic(api_key=settings.api_key, base_url=settings.base_url, timeout=settings.timeout, max_retries=0)
    from google import genai
    return genai.Client(api_key=settings.api_key, http_options={'base_url':settings.base_url,'timeout':int(settings.timeout*1000)})

class UnifiedLLMClient:
    def __init__(self, raw_clients, provider_settings, settings):
        self.raw_clients = dict(raw_clients)
        self.provider_settings = dict(provider_settings)
        self.settings = settings
        self.raw_client = self.raw_clients.get(settings.provider)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))
        self.responses = SimpleNamespace(create=self._responses_create)

    def _merged_settings(self, override=None):
        override = override or {}
        provider = override.get('provider',self.settings.provider)
        if provider not in self.provider_settings:
            raise RuntimeError(f'Provider {provider} is not configured; set its own API key.')
        base = self.provider_settings[provider]
        values = {k:override[k] for k in ['model','temperature','reasoning_effort'] if override.get(k) is not None}
        return replace(base, **values)

    def generate_text(self, prompt, api='auto', settings_override=None):
        settings = self._merged_settings(settings_override)
        client = self.raw_clients.get(settings.provider)
        if client is None: raise RuntimeError(f'No configured client for {settings.provider}')
        try:
            if settings.provider == 'openai':
                if api in {'auto','responses'}:
                    kwargs = dict(model=settings.model,input=prompt,max_output_tokens=settings.max_tokens,temperature=settings.temperature)
                    if settings.reasoning_effort and settings.reasoning_effort != 'none':
                        kwargs['reasoning']={'effort':settings.reasoning_effort}
                    try:
                        response = client.responses.create(**kwargs)
                    except Exception as exc:
                        if 'temperature' not in str(exc).lower(): raise
                        kwargs.pop('temperature', None)
                        response = client.responses.create(**kwargs)
                    return str(response.output_text or '')
                completion=client.chat.completions.create(model=settings.model,messages=[{'role':'user','content':prompt}],temperature=settings.temperature,max_tokens=settings.max_tokens)
                return str(completion.choices[0].message.content or '')
            if settings.provider == 'gemini':
                response=client.models.generate_content(model=settings.model,contents=prompt,config={'temperature':settings.temperature,'max_output_tokens':settings.max_tokens,'response_mime_type':'application/json'})
                return str(response.text or '').strip()
            response=client.messages.create(model=settings.model,max_tokens=settings.max_tokens,temperature=settings.temperature,messages=[{'role':'user','content':prompt}])
            return ''.join(str(getattr(block,'text','')) for block in response.content)
        except Exception as exc:
            raise RuntimeError(f'{settings.provider} request failed: {redact_error(exc)}') from None

    def _responses_create(self, **kwargs):
        text=self.generate_text(kwargs.get('input',''),api='responses',settings_override={'model':kwargs.get('model',self.settings.model)})
        return SimpleNamespace(output_text=text)

    def _chat_create(self, **kwargs):
        prompt='\n'.join(str(m.get('content','')) for m in kwargs.get('messages',[]))
        text=self.generate_text(prompt,api='chat',settings_override={'model':kwargs.get('model',self.settings.model),'temperature':kwargs.get('temperature',self.settings.temperature)})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

def build_llm_client(config=None, provider=None, model=None, temperature=None, allowed_providers=None):
    settings=load_llm_settings(config,provider,model,temperature)
    allowed=set(allowed_providers) if allowed_providers is not None else {settings.provider}
    if settings.provider not in allowed: raise ValueError('Default provider is outside allowed_providers.')
    raw={}; by_provider={}
    for name in sorted(allowed):
        current=settings if name==settings.provider else load_llm_settings(config,name)
        raw[name]=_make_client(current);by_provider[name]=current
    return UnifiedLLMClient(raw,by_provider,settings),settings

def build_openai_client(config=None):
    return build_llm_client(config)
