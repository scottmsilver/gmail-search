"""vLLM chat backend with on-demand spawn/teardown.

vLLM claims ~92% of the GPU while running, so we don't want it pinned
24/7 when nothing's asking for summaries. The backend context-manager
hooks spawn a vllm subprocess on `__enter__` (if none is already up)
and tear it down on `__exit__`. External-managed vLLM (detected by
health check on entry) is left alone.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

URL = "http://127.0.0.1:8001"
DEFAULT_MODEL = "ciocan/gemma-4-E4B-it-W4A16"
# Binary path of the vllm CLI inside the venv we installed it into. If
# users want a different venv they can set VLLM_BIN; we keep a hardcoded
# default that matches our local install for zero-config convenience.
VLLM_BIN_DEFAULT = "/home/ssilver/development/vllm-env/.venv/bin/vllm"
HTTP_TIMEOUT = 120.0
READY_TIMEOUT_SEC = 180.0
TEARDOWN_WAIT_SEC = 15.0


def _launch_args(bin_path: str) -> list[str]:
    """Flags chosen for our 12 GB 3080 Ti:

    - `gptq_marlin`: the log at first-boot recommended switching from
      the buggy `gptq_gemm` kernel to marlin for GPTQ. Faster too.
    - `enforce-eager`: skips CUDA-graph capture. Capture needed more
      VRAM than the quantized weights leave free on our 12 GB card.
    - `max-num-seqs 8`: caps concurrent sequences to fit the KV cache.
    - `max-model-len 2048`: our prompts are ~1.7K tokens including
      output; half the default 4096 doubles KV cache headroom.
    """
    return [
        bin_path,
        "serve",
        DEFAULT_MODEL,
        "--quantization",
        "gptq_marlin",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.92",
        "--enforce-eager",
        "--max-num-seqs",
        "8",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
    ]


class VLLMBackend:
    model_id = DEFAULT_MODEL

    def __init__(self, bin_path: Optional[str] = None):
        import os as _os

        self.bin_path = bin_path or _os.environ.get("VLLM_BIN", VLLM_BIN_DEFAULT)
        self._proc: Optional[subprocess.Popen] = None

    # ── lifecycle ────────────────────────────────────────────────────

    def __enter__(self) -> "VLLMBackend":
        if self._is_healthy():
            logger.info("vLLM already running at %s — reusing", URL)
            self._proc = None  # don't shut down something we didn't start
            return self

        logger.info("spawning vLLM subprocess (model %s)", self.model_id)
        self._proc = subprocess.Popen(
            _launch_args(self.bin_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        if not self._wait_until_ready():
            self._terminate_proc()
            raise RuntimeError(f"vLLM failed to become healthy within {int(READY_TIMEOUT_SEC)}s")
        logger.info("vLLM ready (pid %s)", self._proc.pid)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Only tear down what we started. External-managed vLLM
        # (detected via healthy-before-enter) survives our exit.
        if self._proc is not None:
            self._terminate_proc()
            self._proc = None

    # ── chat ─────────────────────────────────────────────────────────

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
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        if json_format:
            body["response_format"] = {"type": "json_object"}
        resp = client.post(f"{URL}/v1/chat/completions", json=body, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        choices = resp.json().get("choices") or []
        return choices[0]["message"]["content"] if choices else ""

    # ── helpers ──────────────────────────────────────────────────────

    def _is_healthy(self, timeout_sec: float = 2.0) -> bool:
        try:
            r = httpx.get(f"{URL}/v1/models", timeout=timeout_sec)
            return r.status_code == 200
        except Exception:
            return False

    def _wait_until_ready(self) -> bool:
        deadline = time.time() + READY_TIMEOUT_SEC
        while time.time() < deadline:
            if self._is_healthy(timeout_sec=3.0):
                return True
            time.sleep(3)
        return False

    def _terminate_proc(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=TEARDOWN_WAIT_SEC)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except Exception as e:
            logger.warning("error stopping vLLM: %s", e)
