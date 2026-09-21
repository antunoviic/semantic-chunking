"""One retry policy for every HTTP call to Ollama.

Transient failures — a 5xx while the model reloads under memory pressure, a
dropped connection when the runner is restarted — are repeated with exponential
backoff. A 4xx is the caller's mistake and is raised at once. Used by the
embedding function; ``OllamaClient.chat`` carries the same loop inline and will
switch to this helper once the current evaluation run is over (the file lives
in the chunk-cache digest roots, so it is frozen until then).
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import httpx

from ._logging import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 4       # attempts before giving up
RETRY_BACKOFF = 3.0   # seconds before the second attempt, doubled each time


def post_with_retries(
    client: httpx.Client,
    url: str,
    *,
    json: dict,
    retries: int = MAX_RETRIES,
    backoff: float = RETRY_BACKOFF,
    service: str = "Ollama",
    sleep: Optional[Callable[[float], None]] = None,
) -> httpx.Response:
    """POST ``json`` to ``url`` and return the first 2xx response.

    Retries on ``httpx.TransportError`` and on 5xx status codes, waiting
    ``backoff * 2**attempt`` seconds in between. Raises ``httpx.HTTPStatusError``
    immediately for 4xx, and ``RuntimeError`` once ``retries`` attempts failed.
    ``sleep`` exists for tests.
    """
    sleep = sleep or time.sleep
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = client.post(url, json=json)
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500:
                raise
            last_error = exc
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                logger.warning("%s error (%s), attempt %d/%d, waiting %.0fs",
                               service, status or type(exc).__name__,
                               attempt + 1, retries, wait)
                sleep(wait)
    if isinstance(last_error, httpx.ConnectError):
        raise RuntimeError(
            f"{service} not reachable at {url} after {retries} attempts. "
            "Is the service running?") from last_error
    raise RuntimeError(f"{service} failed {retries} times in a row: {last_error}") from last_error
