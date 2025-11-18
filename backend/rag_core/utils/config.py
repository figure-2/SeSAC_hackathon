from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Optional

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "config.yaml"
)


def _parse_simple_yaml(text: str) -> Dict[str, str]:
    """Fallback parser for simple key: value pairs."""
    config: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        # Remove optional quotes
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        config[key] = value
    return config


def load_config(path: Optional[str] = None) -> Dict[str, str]:
    """Load configuration from a YAML file.

    Falls back to a lightweight parser if PyYAML is unavailable.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    contents = cfg_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(contents) or {}
        if not isinstance(data, Mapping):
            raise ValueError("Configuration root must be a mapping.")
        # Convert non-string values to string for consistency
        return {str(k): _stringify_value(v) for k, v in data.items()}
    except ImportError:
        return _parse_simple_yaml(contents)


def _stringify_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(_stringify_value(v) for v in value)
    return str(value)


def ensure_keys(config: Mapping[str, str], required_keys) -> None:
    missing = [key for key in required_keys if key not in config or config[key] == ""]
    if missing:
        raise KeyError(f"Missing required configuration keys: {', '.join(missing)}")


def update_config(path: Optional[str], updates: Mapping[str, str]) -> None:
    """Persist configuration updates to disk (comments may be lost)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config = load_config(cfg_path)
    config.update(updates)

    try:
        import yaml  # type: ignore

        cfg_path.write_text(
            yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except ImportError:
        # Render simple key: value pairs
        lines = []
        for key, value in config.items():
            if re.search(r"\s", value):
                lines.append(f"{key}: \"{value}\"")
            else:
                lines.append(f"{key}: {value}")
        cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
