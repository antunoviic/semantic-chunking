from __future__ import annotations

import os
import re
import time

import httpx


class QwenClient:

    DEFAULT_MODEL   = "qwen3.5:4b"
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_NUM_CTX  = 4096
    MAX_RETRIES      = 4       # transiente Ollama-5xx abfangen
    RETRY_BACKOFF    = 3.0     # Sekunden, verdoppelt sich je Versuch

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        temperature: float = 0.0, #temperature of 0.0 for deterministic answers
        seed: int = 42,           # Ollama ist bei T=0 nicht garantiert bitgenau
        timeout: float = 600.0,
        thinking: bool = False,
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> None:
        self.model       = model
        self.base_url    = (base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.temperature = temperature
        self.seed = seed
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
                "seed": self.seed,
                "num_ctx":     self.num_ctx,
            },
        }
        # Ollama liefert gelegentlich transiente 5xx (z.B. beim Modellwechsel unter
        # Speicherdruck). Ohne Wiederholung reisst ein einzelner Fehler einen
        # mehrstuendigen Chunking-Lauf ab — deshalb kurzer Backoff statt Abbruch.
        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                break
            except httpx.ConnectError:
                raise RuntimeError(f"Ollama not reachable at {self.base_url}.") from None
            except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status < 500:
                    raise                      # 4xx ist unser Fehler, nicht wiederholen
                last_error = exc
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_BACKOFF * (2 ** attempt)
                    print(f"[llm] Ollama-Fehler ({status or type(exc).__name__}), "
                          f"Versuch {attempt + 1}/{self.MAX_RETRIES}, warte {wait:.0f}s",
                          flush=True)
                    time.sleep(wait)
        else:
            raise RuntimeError(
                f"Ollama antwortete {self.MAX_RETRIES}x nicht erfolgreich: {last_error}"
            ) from last_error

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