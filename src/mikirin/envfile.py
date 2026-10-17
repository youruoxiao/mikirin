from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path) -> set[str]:
    """Load a small dotenv file without overwriting the process environment."""
    loaded: set[str] = set()
    if not path.exists():
        return loaded
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number}: expected NAME=VALUE")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not name.replace("_", "a").isalnum() or name[0].isdigit():
            raise ValueError(f"Invalid .env variable name on line {line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
            value = value[1:-1]
        if name not in os.environ:
            os.environ[name] = value
            loaded.add(name)
    return loaded
