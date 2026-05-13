import httpx
from pathlib import Path
from typing import List, Dict
from app.config import get_settings

settings = get_settings()

_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"


def _load_system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def call_llm(
    messages: List[Dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Call the vLLM / OpenAI-compatible chat completions endpoint.

    Args:
        messages: List of message dicts, e.g.
                  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        temperature: Override default temperature from env.
        max_tokens:  Override default max_tokens from env.

    Returns:
        The assistant's reply as a plain string.

    Raises:
        httpx.HTTPStatusError: If the API returns a non-2xx response.
    """
    system_message = {"role": "system", "content": _load_system_prompt()}
    full_messages = [system_message] + messages

    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    payload = {
        "model": settings.model_name,
        "messages": full_messages,
        "temperature": temperature if temperature is not None else settings.temperature,
        "max_tokens": max_tokens if max_tokens is not None else settings.max_tokens,
    }

    with httpx.Client(timeout=120) as client:
        response = client.post(settings.ollama_url, headers=headers, json=payload)
        response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
