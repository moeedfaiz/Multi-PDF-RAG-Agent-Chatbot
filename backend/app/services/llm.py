import json
import logging
import requests
from typing import Iterator, Optional

from ..config import settings

log = logging.getLogger("llm")

# (connect timeout, read timeout)
DEFAULT_TIMEOUT = (10, 600)

# -----------------------------
# Process-wide cache (avoid re-init per request)
# -----------------------------
_GEMINI_CLIENT = None


def _get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai  # type: ignore
        _GEMINI_CLIENT = genai.Client(api_key=settings.gemini_api_key)
    return _GEMINI_CLIENT


# -----------------------------
# Ollama
# -----------------------------
def ollama_generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    m = model or settings.ollama_model
    url = f"{settings.ollama_base_url}/api/generate"

    payload = {
        "model": m,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }

    r = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "") or ""


def ollama_stream(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> Iterator[str]:
    m = model or settings.ollama_model
    url = f"{settings.ollama_base_url}/api/generate"

    payload = {
        "model": m,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }

    with requests.post(url, json=payload, stream=True, timeout=DEFAULT_TIMEOUT) as r:
        r.raise_for_status()

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("done"):
                break

            token = obj.get("response", "")
            if token:
                yield token


# -----------------------------
# Gemini (google-genai SDK)
# -----------------------------
def gemini_generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    m = model or settings.gemini_model
    client = _get_gemini_client()

    resp = client.models.generate_content(
        model=m,
        contents=prompt,
        config={
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
        },
    )

    text = getattr(resp, "text", None)
    return (text or "").strip()


def gemini_stream(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> Iterator[str]:
    m = model or settings.gemini_model
    client = _get_gemini_client()

    stream = client.models.generate_content_stream(
        model=m,
        contents=prompt,
        config={
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
        },
    )

    for chunk in stream:
        txt = getattr(chunk, "text", None)
        if txt:
            yield txt


# -----------------------------
# Provider switch wrappers (IMPORTANT: must catch mid-stream errors)
# -----------------------------
def llm_generate(
    prompt: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    provider = (settings.llm_provider or "ollama").lower().strip()

    if provider == "gemini":
        try:
            out = gemini_generate(prompt, temperature=temperature, max_tokens=max_tokens)
            if out:
                return out
            raise RuntimeError("Gemini returned empty response.")
        except Exception as e:
            log.exception("Gemini generate failed, falling back to Ollama: %s", e)
            return ollama_generate(prompt, temperature=temperature, max_tokens=max_tokens)

    return ollama_generate(prompt, temperature=temperature, max_tokens=max_tokens)


def llm_stream(
    prompt: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> Iterator[str]:
    """
    IMPORTANT: This is a generator (yields). This allows us to catch errors
    that happen *during* streaming (common for Gemini).
    """
    provider = (settings.llm_provider or "ollama").lower().strip()

    if provider == "gemini":
        try:
            for t in gemini_stream(prompt, temperature=temperature, max_tokens=max_tokens):
                yield t
            return
        except Exception as e:
            log.exception("Gemini stream failed, falling back to Ollama stream: %s", e)
            for t in ollama_stream(prompt, temperature=temperature, max_tokens=max_tokens):
                yield t
            return

    for t in ollama_stream(prompt, temperature=temperature, max_tokens=max_tokens):
        yield t
