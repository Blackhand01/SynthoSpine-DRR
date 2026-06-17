"""Small config loader for simple YAML-style key/value files."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path | None, defaults: dict[str, Any]) -> dict[str, Any]:
    """Load a flat YAML-like config and merge it over defaults."""

    config = dict(defaults)
    if path is None:
        return config
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        config[key] = _parse_value(value)
    return config


def _parse_value(value: str) -> Any:
    if "," in value:
        return tuple(part.strip() for part in value.split(",") if part.strip())
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value

