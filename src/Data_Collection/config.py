"""Shared Data_Collection runtime config helpers.

Loads participant/session identifiers from Data_Collection/config.json and
provides helpers for consistent output filename prefixes across scripts.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_COLLECTION_DIR = Path(__file__).resolve().parent
CONFIG_JSON_PATH = DATA_COLLECTION_DIR / "config.json"


def _sanitize_for_filename(value: Any, default: str) -> str:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        raw = default
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    return sanitized.strip("._") or default


@lru_cache(maxsize=1)
def load_config_json() -> dict[str, Any]:
    if not CONFIG_JSON_PATH.exists():
        raise FileNotFoundError(f"Data_Collection config not found: {CONFIG_JSON_PATH}")
    with CONFIG_JSON_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {CONFIG_JSON_PATH}")
    return raw


def get_participant_id() -> str:
    return _sanitize_for_filename(
        load_config_json().get("participant_id"),
        "unknown_participant",
    )


def get_session_id() -> str:
    return _sanitize_for_filename(
        load_config_json().get("session_id"),
        "unknown_session",
    )


def get_participant_session_prefix() -> str:
    return f"{get_participant_id()}_{get_session_id()}"


def prefixed_filename(filename: str) -> str:
    path = Path(filename)
    return f"{get_participant_session_prefix()}_{path.name}"


def prefixed_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path.with_name(prefixed_filename(path.name))
