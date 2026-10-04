# src/llm_client.py
"""
The ONE place that talks to an LLM. Everything else calls generate().

Provider is picked in .env (all three speak the OpenAI chat format):

    LLM_PROVIDER=deepseek   DEEPSEEK_API_KEY=sk-...        (default)
    LLM_PROVIDER=ollama     (local; run `ollama serve`, no key needed)
    LLM_PROVIDER=groq       GROQ_API_KEY=gsk_...

    LLM_MODEL=...           optional; overrides the provider's default model
    USE_LLM=0               optional; switches the LLM off
"""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env file from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

PROVIDERS = {
    #  name       base_url                                  key env var          default model
    "deepseek": ("https://api.deepseek.com",               "DEEPSEEK_API_KEY",  "deepseek-flash"),
    "ollama":   ("http://localhost:11434/v1",              None,                "mistral"),
    "groq":     ("https://api.groq.com/openai/v1",         "GROQ_API_KEY",      "openai/gpt-oss-120b"),
}

_client: Optional[OpenAI] = None


def _provider() -> str:
    name = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    if name not in PROVIDERS:
        raise RuntimeError(f"Unknown LLM_PROVIDER={name!r}. Use one of: {', '.join(PROVIDERS)}")
    return name


def _api_key() -> str:
    key_env = PROVIDERS[_provider()][1]
    if key_env is None:
        return "ollama"                      # Ollama ignores the key, but the SDK wants a string
    return os.getenv(key_env, "").strip()


def model_name() -> str:
    return os.getenv("LLM_MODEL") or PROVIDERS[_provider()][2]


def llm_enabled() -> bool:
    """True only if LLM use is switched on (USE_LLM, default on) and a real-looking key is set."""
    if os.getenv("USE_LLM", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    key = _api_key()
    return bool(key) and not key.upper().startswith("YOUR")


def _get_client() -> OpenAI:
    """Lazy-init the client so import never crashes."""
    global _client
    if _client is None:
        if not llm_enabled():
            key_env = PROVIDERS[_provider()][1]
            raise RuntimeError(f"{key_env} is not set. Put it in your .env file.")
        _client = OpenAI(base_url=PROVIDERS[_provider()][0], api_key=_api_key())
    return _client


def generate(
    prompt: str,
    temperature: float = 0.5,
    max_tokens: int = 5000,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """Send one prompt, get the reply text back. Returns None on failure."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = _get_client().chat.completions.create(
            model=model or model_name(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"LLM call failed ({_provider()} / {model or model_name()}): {e}")
    return None
