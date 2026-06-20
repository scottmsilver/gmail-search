"""OpenRouter backend — hosted models via the OpenAI-compatible API.

Used for proposition extraction (amazon/nova-lite-v1), which is pinned to this
backend regardless of the summarizer's LLM_BACKEND. The API key comes from the
OPENROUTER_KEY env var and the base URL is env-overridable — no server URL is
hardcoded as the sole source.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "amazon/nova-lite-v1"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
HTTP_TIMEOUT = 120.0
_RETRYABLE = {429, 500, 502, 503, 504}


class OpenRouterBackend:
    """Backend protocol impl for OpenRouter. Stateless; no spawn lifecycle."""

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.environ.get("PROP_LLM_MODEL", DEFAULT_MODEL)
        # Fail loud if the key is missing rather than emitting unauthenticated calls.
        self._key = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not self._key:
            raise RuntimeError("OPENROUTER_KEY is not set; cannot use the OpenRouter backend")
        # The API key is sent as a bearer to this base URL, so a custom endpoint
        # is a secret-exfiltration risk: require https and an explicit opt-in.
        base = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        if base != DEFAULT_BASE_URL:
            if not base.startswith("https://"):
                raise RuntimeError("OPENROUTER_BASE_URL must be https (the API key is sent to it)")
            if os.environ.get("OPENROUTER_ALLOW_CUSTOM_BASE_URL") != "1":
                raise RuntimeError(
                    "custom OPENROUTER_BASE_URL requires OPENROUTER_ALLOW_CUSTOM_BASE_URL=1 "
                    "(the API key is sent to this host)"
                )
        self._base = base

    # No spawn/teardown — satisfy the optional context-manager protocol as no-ops.
    def __enter__(self) -> "OpenRouterBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

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
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_format:
            body["response_format"] = {"type": "json_object"}
        # NB: never log headers — the Authorization bearer must not leak.
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                resp = client.post(f"{self._base}/chat/completions", json=body, headers=headers, timeout=HTTP_TIMEOUT)
                if resp.status_code in _RETRYABLE:
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                resp.raise_for_status()
                choices = resp.json().get("choices") or []
                return choices[0]["message"]["content"] if choices else ""
            except Exception as e:  # noqa: BLE001 - retry network/5xx, re-raise on exhaustion
                last_exc = e
                if attempt == 5:
                    break
                time.sleep(min(2**attempt, 30))
        raise last_exc if last_exc else RuntimeError("OpenRouter chat failed")
