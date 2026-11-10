"""Role prompts, kept as plain-text resources beside the method code."""
from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"


def load(name):
    return (RESOURCE_DIR / (name + ".txt")).read_text(encoding="utf-8")


ANSWER = load("ANSWER.en.v7")
PROBE = load("PROBE.en")
PLAN = load("PLAN.en")
CONTROL = load("CONTROL.en")
TOOLS = load("TOOLS.en")
REPLAN = load("REPLAN.en")
JUDGE = load("JUDGE.en")
READY = load("READY.en")
INFER = load("INFER.en")
DURATION = load("DURATION.en")
