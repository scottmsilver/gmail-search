from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "budget": {
        "max_usd": 5.00,
    },
    "embedding": {
        "model": "gemini-embedding-2-preview",
        "dimensions": 3072,
        "task_type_document": "RETRIEVAL_DOCUMENT",
        "task_type_query": "RETRIEVAL_QUERY",
    },
    "attachments": {
        "max_file_size_mb": 10,
        "max_pdf_pages": 20,
        "max_images_per_message": 10,
        "max_attachment_text_tokens": 50000,
    },
    "download": {
        "batch_size": 25,
        "max_messages": None,
    },
    "search": {
        "default_top_k": 20,
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8080,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Path | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = _deep_merge(DEFAULTS, {})

    if config_path and config_path.exists():
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, file_cfg)

    # Load local overrides (gitignored)
    if config_path:
        local_path = config_path.parent / "config.local.yaml"
        if local_path.exists():
            with open(local_path) as f:
                local_cfg = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, local_cfg)

    if data_dir is None:
        data_dir = Path.cwd() / "data"
    cfg["data_dir"] = str(data_dir)

    return cfg
