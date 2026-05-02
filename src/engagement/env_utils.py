from __future__ import annotations

import os
from pathlib import Path


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if value and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
        value = value[1:-1]
    return key, value


def load_env_files(
    repo_root: Path,
    *,
    filenames: tuple[str, ...] = (".env", ".env.local"),
    override: bool = False,
) -> list[Path]:
    """Load environment variables from repo-local .env files."""

    loaded_paths: list[Path] = []
    for name in filenames:
        path = (repo_root / name).resolve()
        if not path.exists() or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw_line in lines:
            parsed = _parse_env_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            if override or key not in os.environ:
                os.environ[key] = value
        loaded_paths.append(path)
    return loaded_paths
