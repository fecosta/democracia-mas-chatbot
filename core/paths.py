from __future__ import annotations

import os


def get_data_dir() -> str:
    return os.environ.get("DPLUS_DATA_DIR", "data")


def docs_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "docs")


def structured_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "structured")


def db_path(data_dir: str) -> str:
    return os.path.join(data_dir, "app.db")


def config_path(data_dir: str) -> str:
    return os.path.join(data_dir, "config.json")
