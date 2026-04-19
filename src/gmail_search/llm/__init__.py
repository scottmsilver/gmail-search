"""Local-LLM backend abstraction.

`summarize.py` (and any other caller) asks for a `Backend` via
`get_backend()` and calls `backend.chat(...)` without caring whether
the transport is Ollama, vLLM, or something else. Each backend handles
its own HTTP, request shaping, response parsing, and lifecycle.

Select the backend via the LLM_BACKEND env var:
    export LLM_BACKEND=vllm     # default
    export LLM_BACKEND=ollama
"""

from __future__ import annotations

import os

from gmail_search.llm.backend import Backend


def get_backend() -> Backend:
    """Return the configured backend instance (fresh each call — cheap,
    stateless configuration). Lifecycle (spawn/teardown) is opt-in per
    backend via the context-manager protocol on the returned object.
    """
    name = os.environ.get("LLM_BACKEND", "vllm").lower()
    if name == "ollama":
        from gmail_search.llm.ollama import OllamaBackend

        return OllamaBackend()
    if name == "vllm":
        from gmail_search.llm.vllm import VLLMBackend

        return VLLMBackend()
    raise ValueError(f"unknown LLM_BACKEND: {name!r} (expected 'vllm' or 'ollama')")


__all__ = ["Backend", "get_backend"]
