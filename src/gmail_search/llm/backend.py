"""Backend protocol every LLM provider must implement."""

from __future__ import annotations

from typing import Protocol

import httpx


class Backend(Protocol):
    """Shared interface for Ollama, vLLM, and any future provider.

    Implementations may additionally be used as context managers to own
    a spawn/teardown lifecycle (vLLM does this to avoid pinning the GPU
    when idle). Callers that don't need lifecycle management can call
    `chat` directly.
    """

    # Human-readable identifier stored in message_summaries.model so the
    # backfill query knows whether a row was written under this backend.
    model_id: str

    def chat(
        self,
        client: httpx.Client,
        messages: list[dict],
        *,
        max_tokens: int,
        json_format: bool = False,
    ) -> str:
        """Run a single chat completion. Returns the raw content string."""
        ...

    def __enter__(self) -> "Backend":
        """Default no-op; vLLM overrides to spawn the server process."""
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        """Default no-op; vLLM overrides to shut down the spawned server."""
        ...
