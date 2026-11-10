"""Small file helpers for the algorithm examples."""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("PRISM_WORKSPACE") or Path.cwd()).resolve()


def setup():
    cache = ROOT / "cache" / "tiktoken"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)
