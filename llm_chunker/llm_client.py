from __future__ import annotations

import re
import httpx


class QwenClient:
    """HTTP client for a locally running Qwen 3.5 4B model via Ollama."""

    DEFAULT_MODEL = "qwen3.5:4b"
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_NUM_CTX = 4096  # small context = less RAM, sufficient for chunking tasks

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        timeout: float = 600.0,  # read timeout for slow CPU/GPU inference
        thinking: bool = False,  # disable thinking to save tokens and RAM
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
        """Send a chat request via /api/chat and return the response text."""
        payload = {
            "model": self.model,
            "messages": self._with_thinking_flag(messages),
            "stream": False,
            "think": self.thinking,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        response = self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def _with_thinking_flag(self, messages: list) -> list:
        """Pass messages unchanged; thinking is controlled via the top-level think field."""
        return messages

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks from the model response."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
