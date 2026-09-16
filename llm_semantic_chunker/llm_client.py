from __future__ import annotations

import os
import re
import time

import httpx
from ._logging import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Chat client for a local Ollama server.

    Works with any model Ollama serves — the default is ``qwen3.5:4b``, but
    ``model="llama3.2:3b"`` or anything else pulled locally is equally valid.
    Decoding is near-deterministic by default (``temperature=0.0`` plus a fixed
    ``seed``; on Ollama temperature alone is not bit-exact). Transient failures
    are retried with exponential backoff.
    """

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

    # Grobe Schaetzung: ~4 Zeichen je Token fuer lateinische Schrift. Ungenau
    # (Faktor 3-5 je nach Sprache), aber es geht nur darum, eine Ueberschreitung
    # VOR dem Senden zu bemerken. Ein Tokenizer waere eine weitere Abhaengigkeit
    # fuer eine Bibliothek, die sonst mit drei Paketen auskommt.
    CHARS_PER_TOKEN = 4
    CONTEXT_LIMIT_RATIO = 0.9

    def _guard_prompt_size(self, messages: list[dict]) -> None:
        """Refuse a prompt that would exceed the context window.

        Ollama truncates silently: the model then judges text it never saw, and
        nothing in the response reveals it. A boundary decision made on truncated
        input is worse than no decision at all — so this raises instead of
        letting the run continue on unsound output.

        With the character cap used throughout the thesis runs (1200) the load is
        about 15 %, so this never fires there. It fires for the window mode
        without a cap, or for a `max_chunk_chars` set far above `num_ctx`.
        """
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
        # Ollama liefert gelegentlich transiente Fehler (5xx beim Modellwechsel unter
        # Speicherdruck, abgerissene Verbindungen wenn der Dienst neu startet). Ohne
        # Wiederholung reisst ein einzelner davon einen mehrstuendigen Chunking-Lauf
        # ab — deshalb kurzer Backoff statt Abbruch.
        #
        # httpx.TransportError ist die Oberklasse von ConnectError, ConnectTimeout,
        # ReadError/ReadTimeout, WriteError/WriteTimeout, PoolTimeout und
        # RemoteProtocolError. Frueher wurde ConnectError sofort fatal — also genau
        # der Fall, den die Wiederholung abfangen soll (Ollama vom System beendet und
        # neu gestartet, auf dieser Maschine mehrfach vorgekommen).
        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                break
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status < 500:
                    raise                      # 4xx ist unser Fehler, nicht wiederholen
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
        """Warn when a prompt comes close to the context window.

        Ollama silently truncates prompts that exceed ``num_ctx``: the model then
        judges text it never saw, and nothing in the response says so. No
        tokeniser is needed to notice — the reply carries ``prompt_eval_count``,
        the number of tokens actually processed, which was previously discarded.

        Second safety net behind `_guard_prompt_size`, which refuses oversized
        prompts before sending. This one catches what the character estimate got
        wrong — a language whose tokens are shorter than four characters, for
        instance.
        """
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

