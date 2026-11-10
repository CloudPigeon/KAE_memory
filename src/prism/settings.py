"""Read operator-provided service settings from an explicit workspace."""
import os
from pathlib import Path

from dotenv import dotenv_values


def load_environment():
    workspace = Path(os.environ.get("PRISM_WORKSPACE") or Path.cwd())
    path = Path(os.environ.get("PRISM_ENV_FILE") or workspace / ".env")
    values = {key: value for key, value in dotenv_values(path).items() if value is not None}
    return {**values, **os.environ}
