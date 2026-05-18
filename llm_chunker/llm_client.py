from __future__ import annotations

import os
import re
import httpx


class QwenClient:

    DEFAULT_MODEL   = "qwen3.5:4b"
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_NUM_CTX  = 4096

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        temperature: float = 0.0, #temperature of 0.0 for deterministic answers
        timeout: float = 600.0,
        thinking: bool = False,
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> None:
        self.model       = model
        self.base_url    = (base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.temperature = temperature
        self.thinking    = thinking
        self.num_ctx     = num_ctx
        self._client     = httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=timeout, write=30.0, pool=30.0)
        )

    def chat(self, messages: list[dict]) -> str:
        payload = {
            "model":   self.model,
            "messages": messages,
            "stream":  False,
            "think":   self.thinking,
            "options": {
                "temperature": self.temperature,
                "num_ctx":     self.num_ctx,
            },
        }
        try:
            #post request to ollama api
            response = self._client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Ollama not reachable at {self.base_url}. "
            ) from None

        content = response.json()["message"]["content"].strip()
        if not self.thinking:
            content = self._strip_thinking(content)
        return content

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks that some models emit."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()