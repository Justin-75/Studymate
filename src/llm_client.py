import os
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Initialize Groq client
_client: Optional[Groq] = None


def llm_enabled() -> bool:
    """True only if LLM use is switched on (USE_LLM, default on) and a real-looking key is set."""
    if os.getenv("USE_LLM", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    key = (GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")).strip()
    return bool(key) and not key.upper().startswith("YOUR")


def _get_client() -> Groq:
    """Lazy-init Groq client so import never crashes."""
    global _client
    if _client is None:
        key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Please set it in your .env file or as an environment variable."
            )
        _client = Groq(api_key=key)
    return _client


def generate_with_groq(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 5000,
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    """
    Call Groq Cloud API to generate text.
    :param prompt: User prompt
    :param model: Model name (default: openai/gpt-oss-120b)
    :param temperature: Temperature for randomness (0~1)
    :param max_tokens: Maximum tokens to generate
    :param system_prompt: Optional system prompt to set the role
    :return: Generated text, or None on failure
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"Groq API call failed: {e}")
    return None