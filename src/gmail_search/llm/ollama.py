"""Ollama chat backend. Assumes a user-managed Ollama daemon at :11434."""

from __future__ import annotations

import httpx

URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:latest"
HTTP_TIMEOUT = 120.0


class OllamaBackend:
    model_id = DEFAULT_MODEL

    def chat(
        self,
        client: httpx.Client,
        messages: list[dict],
        *,
        max_tokens: int,
        json_format: bool = False,
    ) -> str:
        body: dict = {
            "model": self.model_id,
            "stream": False,
            "messages": messages,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        }
        if json_format:
            body["format"] = "json"
        resp = client.post(f"{URL}/api/chat", json=body, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return (resp.json().get("message") or {}).get("content", "")

    def __enter__(self) -> "OllamaBackend":
        # Ollama is assumed to already be running (user-managed systemd).
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None
