# src/llm_client.py
"""The one place that picks the LLM: DeepSeek when DEEPSEEK_API_KEY is set, else local Ollama."""
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from dotenv import load_dotenv
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")   # project-root .env

DEEPSEEK_MODEL = "deepseek-flash"
STRUCTURED_ATTEMPTS = 3      # tries per structured call before giving up

T = TypeVar("T", bound=BaseModel)


def llm_info() -> dict:
    """Which provider and model get_llm() uses (shown by the server and the web UI)."""
    if os.getenv("DEEPSEEK_API_KEY"):
        return {"provider": "DeepSeek", "model": DEEPSEEK_MODEL}
    return {"provider": "Ollama", "model": os.getenv("OLLAMA_MODEL", "mistral-nemo:12b")}


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.3):
    info = llm_info()
    if info["provider"] == "DeepSeek":
        # deepseek-flash thinks by default, and thinking mode rejects the forced tool_choice that
        # with_structured_output() sends (and ignores temperature), so switch it off.
        return ChatDeepSeek(model=info["model"], temperature=temperature, timeout=60, max_retries=2,
                            extra_body={"thinking": {"type": "disabled"}})
    return ChatOllama(model=info["model"], temperature=temperature)


# ---------------------------------------------------------------------------
# Structured output that never comes back as None
# ---------------------------------------------------------------------------
class StructuredOutputError(RuntimeError):
    """The model gave no usable answer for a schema, even after retrying."""


def structured_llm(schema: Type[T], temperature: float = 0.3) -> Runnable:
    """
    Like get_llm(temperature).with_structured_output(schema), but a reply LangChain cannot parse is
    repaired or retried instead of silently turning into None: on a long job (a 236-section book)
    the model now and then calls the tool with empty or broken arguments. If every try fails it
    raises StructuredOutputError, so batch(..., return_exceptions=True) reports it like any error.
    Supports invoke / batch / batch_as_completed like any runnable.
    """
    llm = get_llm(temperature).with_structured_output(schema, include_raw=True)

    def call(prompt: Any, config: RunnableConfig) -> T:
        problem = ""
        for _ in range(STRUCTURED_ATTEMPTS):
            out = llm.invoke(prompt, config)
            parsed = out["parsed"] if out["parsed"] is not None else _salvage(schema, out["raw"])
            if parsed is not None:
                return parsed
            problem = _describe(out)
            print(f"[llm] unusable {schema.__name__} reply, retrying: {problem}")
        raise StructuredOutputError(f"No usable {schema.__name__} after {STRUCTURED_ATTEMPTS} tries ({problem})")

    return RunnableLambda(call, name=f"structured_{schema.__name__}")


# a valid "\\" pair, or a lone backslash that JSON does not allow (e.g. "\Theta" or "\lg" from a math book)
_ESCAPES = re.compile(r'\\\\|\\(?![/"bfnrtu])')


def _fix_escapes(text: str) -> str:
    return _ESCAPES.sub(lambda m: m.group(0) if m.group(0) == "\\\\" else "\\\\", text)


def _salvage(schema: Type[T], raw: Any) -> Optional[T]:
    """Second chance for an unparsed reply: tool-call JSON with stray backslashes, arguments wrapped
    in one extra object, or the JSON written as plain text instead of a tool call."""
    candidates: list = [tc.get("args") for tc in getattr(raw, "tool_calls", None) or []]
    candidates += [tc.get("args") for tc in getattr(raw, "invalid_tool_calls", None) or []]
    candidates.append(getattr(raw, "content", None))
    for cand in candidates:
        if isinstance(cand, str):
            text = re.sub(r"^```(?:json)?|```$", "", cand.strip()).strip()
            for fixed in (text, _fix_escapes(text)):
                try:
                    cand = json.loads(fixed, strict=False)
                    break
                except ValueError:
                    continue
        if not isinstance(cand, dict) or not cand:
            continue
        nested = next(iter(cand.values())) if len(cand) == 1 else None
        for data in (cand, nested):
            if isinstance(data, dict):
                try:
                    return schema.model_validate(data)
                except ValueError:                  # pydantic's ValidationError is a ValueError
                    pass
    return None


def _describe(out: dict) -> str:
    raw = out["raw"]
    bits = [f"finish_reason={(getattr(raw, 'response_metadata', None) or {}).get('finish_reason')}"]
    if out.get("parsing_error"):
        bits.append(f"parse error: {str(out['parsing_error'])[:200]}")
    if getattr(raw, "invalid_tool_calls", None):
        bits.append(f"broken tool call: {raw.invalid_tool_calls[0].get('error')}")
    elif not getattr(raw, "tool_calls", None):
        bits.append(f"no tool call, text: {str(getattr(raw, 'content', ''))[:120]!r}")
    else:
        bits.append(f"tool call args: {str(raw.tool_calls[0].get('args'))[:120]}")
    return "; ".join(bits)
