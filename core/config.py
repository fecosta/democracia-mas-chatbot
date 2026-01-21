from __future__ import annotations

import os
import json
from typing import Any, Dict

from .paths import config_path, get_data_dir
from .utils import ensure_dirs

DEFAULT_CONFIG: Dict[str, Any] = {
    "chat_model": "claude-3-haiku-20240307",
    "embedding_model": "text-embedding-3-large",
    "temperature": 0.25,
    "top_k": 6,
    "max_history_messages": 10,
    "max_tokens": 1200,
    "default_answer_lang": "auto",  # auto|es|pt|en
}

SUPPORTED_CLAUDE_MODELS = [
    "claude-3-haiku-20240307",
]

ANSWER_LANG_OPTIONS = {
    "Auto": "auto",
    "Español": "es",
    "Português": "pt",
    "English": "en",
}


def load_config() -> Dict[str, Any]:
    data_dir = get_data_dir()
    ensure_dirs(data_dir)
    cfg = DEFAULT_CONFIG.copy()
    path = config_path(data_dir)
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except Exception:
            pass

    if cfg.get("chat_model") not in SUPPORTED_CLAUDE_MODELS:
        cfg["chat_model"] = DEFAULT_CONFIG["chat_model"]
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    data_dir = get_data_dir()
    ensure_dirs(data_dir)
    path = config_path(data_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
