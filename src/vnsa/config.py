"""Đọc config/settings.toml. Mọi đường dẫn đều tính từ thư mục gốc dự án."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.toml"


@lru_cache(maxsize=1)
def settings() -> dict:
    with SETTINGS_FILE.open("rb") as f:
        return tomllib.load(f)


def path(key: str) -> Path:
    """Đường dẫn tuyệt đối cho một khoá trong mục [data], ví dụ path("warehouse")."""
    return PROJECT_ROOT / settings()["data"][key]
