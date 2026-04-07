from __future__ import annotations

import re
import httpx


class QwenClient:
    """HTTP client for a locally running Qwen model via Ollama."""

    DEFAULT_MODEL = "qwen3.5:4b"
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_NUM_CTX = 4096

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        timeout: float = 600.0,
        thinking: bool = False,
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.thinking = thinking
        self.num_ctx = num_ctx
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=timeout, write=30.0, pool=30.0)
        )

    def chat(self, messages: list) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.thinking,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        content = response.json()["message"]["content"].strip()
        if not self.thinking:
            content = self._strip_thinking(content)
        return content

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()