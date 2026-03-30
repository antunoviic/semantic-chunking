import re
import httpx


class QwenClient:
    """HTTP client for a locally running Qwen 3.5 4B model via Ollama."""

    DEFAULT_MODEL = "qwen3.5:4b"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        timeout: float = 120.0,
        thinking: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.thinking = thinking
        self._client = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict]) -> str:
        """Send a chat request and return the assistant's response text."""
        if self.thinking:
            # Append /think instruction to the last user message
            messages = messages.copy()
            messages[-1] = {
                **messages[-1],
                "content": messages[-1]["content"] + "\n/think",
            }

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        response = self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return self._strip_thinking(content)

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
