"""Project dataset records onto the conversation fields used by RefractMem."""
import re

from .data import sanitize_conversation


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError("conversation IDs use letters, numbers, underscores and hyphens")
    return value


def actor(source):
    if "conversation" in source:
        source = sanitize_conversation(source)
    turn_fields = ("id", "dia_id", "speaker", "listener", "text", "session", "recorded_at")
    return {
        "id": safe_name(source["id"]), "participants": list(source["participants"]),
        "sessions": [{key: session[key] for key in ("id", "recorded_at", "source_ids")}
                     for session in source["sessions"]],
        "turns": [{**{key: turn[key] for key in turn_fields},
                   "caption": turn.get("caption", ""), "image_query": turn.get("image_query", "")}
                  for turn in source["turns"]],
    }


def conversations(payload, selected=None):
    samples = payload if isinstance(payload, list) else [payload]
    return [actor(sample) for sample in samples
            if not selected or sample.get("id", sample.get("sample_id")) in selected]
