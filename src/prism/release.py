"""Commands for preparing data, compiling memory, answering and scoring."""
import argparse
import json
from pathlib import Path

from . import io
from .ingest import conversations
from .pipeline import build, run, workspace_path
from .prompts import RESOURCE_DIR


def load_config(path=None):
    return io.read(path or RESOURCE_DIR / "default_config.json")


def prepare(source, selected=None):
    payload = io.read(source)
    originals = payload if isinstance(payload, list) else [payload]
    actors = conversations(originals, selected)
    selected_ids = {actor["id"] for actor in actors}
    for actor in actors:
        io.write(io.ROOT / "data" / "actors" / (actor["id"] + ".json"), actor)
    questions, labels = [], {}
    for sample in originals:
        conversation_id = sample.get("id", sample.get("sample_id"))
        if conversation_id not in selected_ids:
            continue
        for position, question in enumerate(sample.get("qa", [])):
            if question.get("category") in (1, 2, 3, 4):
                question_id = f"{conversation_id}:qa{position}"
                questions.append({"id": question_id, "conversation": conversation_id, "question": question["question"]})
                labels[question_id] = {"answer": question["answer"], "category": question["category"]}
    io.write(io.ROOT / "data" / "questions.json", questions)
    io.write(io.ROOT / "data" / "evaluation" / "labels.json", labels)
    return {"actors": len(actors), "questions": len(questions)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="KAE_memory / RefractMem algorithm implementation")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--source", required=True, type=Path)
    command.add_argument("--conversations", nargs="+")
    command = sub.add_parser("build")
    command.add_argument("--config", type=Path)
    command.add_argument("--conversations", nargs="+")
    command = sub.add_parser("run")
    command.add_argument("--config", type=Path)
    command.add_argument("--limit", type=int)
    command = sub.add_parser("score")
    command.add_argument("--predictions", required=True, type=Path)
    command.add_argument("--labels", required=True, type=Path)
    command.add_argument("--dataset", choices=("locomo", "hotpotqa"), default="locomo")
    command.add_argument("--bleu1", action="store_true")
    command.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        result = prepare(args.source, args.conversations)
    elif args.command == "build":
        result = build(load_config(args.config), args.conversations)
    elif args.command == "run":
        result = run(load_config(args.config), args.limit)
    else:
        from .metrics import summarize
        result = summarize(io.read(args.predictions), io.read(args.labels), args.dataset, args.bleu1)
        if args.out:
            io.write(workspace_path(args.out), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(bool(result.get("failed", 0)))
