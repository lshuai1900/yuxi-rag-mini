import os
from typing import Any

import httpx

from app.core.logging import logger


class LLMProvider:
    """OpenAI-compatible Chat Completions client.

    Works with OpenAI, Azure OpenAI (OpenAI-compatible endpoint), local Ollama
    (`/v1` endpoint) and any service that exposes `POST {base_url}/chat/completions`.
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

        self.endpoint = f"{self.base_url}/chat/completions"
        self.headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def is_configured(self) -> bool:
        """Whether the provider has enough settings to call the LLM."""
        return bool(self.base_url and self.model)

    async def achat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        """Call chat completions and return the assistant message content.

        Args:
            messages: OpenAI-style messages list, e.g.
                [{"role": "system", "content": "..."},
                 {"role": "user", "content": "..."}]
            temperature: sampling temperature
            max_tokens: maximum tokens to generate

        Returns:
            The content string from `choices[0].message.content`.
        """
        if not self.is_configured():
            raise RuntimeError(
                "LLM provider is not configured. Set LLM_BASE_URL and LLM_MODEL."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.endpoint, json=payload, headers=self.headers
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code if e.response is not None else None
            body = e.response.text if e.response is not None else ""
            logger.error(f"LLM HTTP error status={status_code} body={body[:500]}")
            raise RuntimeError(
                f"LLM service returned HTTP {status_code}: {body[:200]}"
            ) from e
        except httpx.HTTPError as e:
            logger.error(f"LLM request failed: {e}")
            raise RuntimeError(f"LLM request failed: {e}") from e

        try:
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"LLM response has no choices: {data}")
            content = choices[0]["message"]["content"]
        except (KeyError, TypeError, IndexError) as e:
            raise RuntimeError(f"Invalid LLM response shape: {data}") from e

        return content or ""


def create_llm_provider() -> LLMProvider:
    """Build an LLMProvider from settings / environment variables.

    Reads (in priority order) environment variables then the project Settings:
        LLM_BASE_URL  -> e.g. https://api.openai.com/v1  or  http://localhost:11434/v1
        LLM_API_KEY   -> OpenAI key, or "ollama" for a local Ollama service
        LLM_MODEL     -> e.g. gpt-4o-mini, qwen2.5:7b
        LLM_TIMEOUT   -> request timeout in seconds (optional)
    """
    from app.core.config import settings

    base_url = os.getenv("LLM_BASE_URL") or settings.LLM_BASE_URL
    api_key = os.getenv("LLM_API_KEY") or settings.LLM_API_KEY
    model = os.getenv("LLM_MODEL") or settings.LLM_MODEL

    timeout_raw = os.getenv("LLM_TIMEOUT")
    if timeout_raw and timeout_raw.isdigit():
        timeout = int(timeout_raw)
    else:
        timeout = settings.LLM_TIMEOUT

    return LLMProvider(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )
