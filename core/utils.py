from __future__ import annotations

import os
import re
import hashlib
from datetime import datetime
from typing import Optional


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def ensure_dirs(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "docs"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "structured"), exist_ok=True)


def safe_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r"\s+", " ", name).strip()
    return name or "file"


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default
