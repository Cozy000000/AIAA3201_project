from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config file and return a dictionary."""
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")

    return data


def get_nested(config: dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Resolve a dotted key path from a nested dictionary."""
    current: Any = config
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
