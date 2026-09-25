from __future__ import annotations

import os
import re
import time

import httpx
from ._logging import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Talks to a local Ollama server, with decoding fixed for reproducibility.

    Temperature zero and a fixed seed are deliberate: three full passes over
    the same document produced byte-identical chunks, which is what allows a
    single reported run instead of averaging over repetitions.

    Transport errors and 5xx responses are retried with exponential backoff;
    a 4xx is raised immediately, because a malformed request does not become
    well-formed by waiting. The prompt is checked against the context window
    before it is sent, so an oversized prompt fails with a clear message
    instead of being silently truncated by the server.
    """

    DEFAULT_MODEL   = "qwen3.5:4b"
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_NUM_CTX  = 4096
    MAX_RETRIES      = 4       # attempts on 5xx / transport errors before giving up
    RETRY_BACKOFF    = 3.0     # seconds, doubled per attempt

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        temperature: float = 0.0, # greedy decoding
        seed: int = 42,           # temperature=0 alone is not bit-exact on Ollama; the
                                  # fixed seed made three full runs identical (measured)
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

    CHARS_PER_TOKEN = 4
    CONTEXT_LIMIT_RATIO = 0.9

    def _guard_prompt_size(self, messages: list[dict]) -> None:
        if not self.num_ctx:
            return
        chars = sum(len(m.get("content", "")) for m in messages)
        estimate = chars / self.CHARS_PER_TOKEN
        budget = self.num_ctx * self.CONTEXT_LIMIT_RATIO
        if estimate > budget:
            raise ValueError(
                f"Prompt is about {estimate:.0f} tokens ({chars} characters), which "
                f"exceeds {self.CONTEXT_LIMIT_RATIO:.0%} of num_ctx={self.num_ctx}. "
                f"Ollama would truncate it silently and the model would judge text "
                f"it never saw. Lower max_chunk_chars or max_chunk_sentences, or "
                f"raise num_ctx on the client."
            )

    def chat(self, messages: list[dict]) -> str:
        self._guard_prompt_size(messages)
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
        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                break
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status < 500:
                    raise                      # 4xx is our mistake; retrying cannot fix it
                last_error = exc
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Ollama error (%s), attempt %d/%d, waiting %.0fs",
                        status or type(exc).__name__, attempt + 1, self.MAX_RETRIES, wait)
                    time.sleep(wait)
        else:
            if isinstance(last_error, httpx.ConnectError):
                raise RuntimeError(
                    f"Ollama not reachable at {self.base_url} after "
                    f"{self.MAX_RETRIES} attempts. Is the service running?"
                ) from last_error
            raise RuntimeError(
                f"Ollama failed {self.MAX_RETRIES} times in a row: {last_error}"
            ) from last_error

        body = response.json()
        if "message" not in body:
            raise RuntimeError(f"Unexpected Ollama response without 'message': {body!r}")
        self._check_context_budget(body)
        content = body["message"]["content"].strip()
        if not self.thinking:
            content = self._strip_thinking(content)
        return content

    def _check_context_budget(self, body: dict) -> None:
        used = body.get("prompt_eval_count")
        if not used or not self.num_ctx:
            return
        if used >= self.num_ctx * 0.9:
            logger.warning(
                "Prompt used %d of %d context tokens (%.0f %%). Ollama truncates "
                "silently beyond num_ctx — lower max_chunk_chars or raise num_ctx.",
                used, self.num_ctx, used / self.num_ctx * 100)

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

