import json
from pathlib import Path
from typing import Any


def load_json(path: Path | str) -> Any:
    """Load JSON content from a file path."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
