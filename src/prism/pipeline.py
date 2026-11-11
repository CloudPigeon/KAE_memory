"""RefractMem's offline compiler and online research-to-reading algorithm."""
import json
import re
from collections import Counter

from . import io
from .client import Client
from .compilation import compile_multiview_for_cid, fact_records
from .decision import evidence_gate, judge
from .ingest import actor as to_actor, safe_name
from .probe_loop import Workspace
from .probe_plan_loop import broaden_query, research
from .reader import read_answer
from .replan import repair_plan
from .retrieve import Index, memory_records, raw_records
from .settings import load_environment
from .two_stage import build_two_stage

TEMPORAL_QUESTION = re.compile(r'^\s*(when\b|what year\b|how long\b|how many (days|weeks|months|years)\b)', re.I)


def workspace_path(value):
    return io.ROOT / value


def make_clients(config):
    environment = load_environment()
    proxy = environment.get("PRISM_HTTP_PROXY") or None

    def model(role, group, template=True, json_mode=True):
        base = environment.get("PRISM_" + group + "_BASE")
        key = environment.get("PRISM_" + group + "_API_KEY")
        if group == "WRITER":
            base = base or environment["PRISM_MAIN_BASE"]
            key = key or environment["PRISM_MAIN_API_KEY"]
        return Client(base, key, model=config["models"][role], template=template,
                      json_mode=json_mode, proxy=proxy)

    return {
        "writer": model("writer", "WRITER"),
        "planner": model("planner", "MAIN"),
        "reader": model("reader", "MAIN", template=False),
        "probe": model("probe", "AUX", json_mode=False),
        "judge": model("judge", "AUX", template=False),
        "embedding": Client(environment["PRISM_EMBEDDING_BASE"], environment["PRISM_EMBEDDING_API_KEY"],
                            embedding_model=config["models"]["embedding"], proxy=proxy),
    }


def build(config, selected=None):
    """Compile facts and schemas, then align all retrieval views to raw turns."""
    clients = make_clients(config)
    built = []
    for path in sorted(workspace_path(config["actors"]).glob("*.json")):
        if selected and path.stem not in selected:
            continue
        actor = to_actor(io.read(path))
        archive, facts_by_session = build_two_stage(actor, clients["writer"], clients["embedding"], **config["writer"])
        facts = fact_records(actor, facts_by_session)
        organized = memory_records(archive)
        folder = workspace_path(config["store"]) / "memory" / actor["id"]
        io.write(folder / "records.json", facts)
        io.write(folder / "archive.json", archive.to_dict())
        indexes = {}
        for name, records in [("raw", raw_records(actor)), ("prism", facts), ("org", organized)]:
            vectors = clients["embedding"].embed([record["text"] for record in records])
            io.write(folder / (name + "-index.json"), {"vectors": vectors})
            indexes[name] = Index(records, vectors)
        multiview = compile_multiview_for_cid(actor["id"], actor, facts, indexes["prism"].v,
                                             organized, indexes["org"].v, clients["embedding"])
        io.write(folder / "turn-multiview.json", multiview)
        built.append(actor["id"])
    return {"built": len(built), "conversations": built}


def load_store(config, conversation_id):
    conversation_id = safe_name(conversation_id)
    folder = workspace_path(config["store"]) / "memory" / conversation_id
    actor = io.read(workspace_path(config["actors"]) / (conversation_id + ".json"))
    return {"actor": actor,
            "memory": Index(io.read(folder / "records.json"), io.read(folder / "prism-index.json")["vectors"]),
            "raw": Index(raw_records(actor), io.read(folder / "raw-index.json")["vectors"]),
            "multiview": io.read(folder / "turn-multiview.json")}


def answer_question(question, store, clients, settings):
    """Probe -> plan -> search/integrate -> original-evidence reader."""
    workspace = Workspace(store["actor"], store["memory"], store["raw"], clients["embedding"], store["multiview"],
                          context_chars=settings["context_chars"], search_top_k=settings["search_top_k"])
    seen_queries = set()

    def call(prompt, stage, step, max_tokens):
        client = clients["probe"] if stage == "probe" else clients["planner"]
        budget = settings["model_context"] - int(len(prompt) / 2.6) - 256
        if budget < 512:
            keep = max(4000, int((settings["model_context"] - 768) * 2.6))
            head = int(keep * 0.42)
            prompt = prompt[:head] + '\n[middle of the context omitted for length]\n' + prompt[-(keep-head):]
            budget = settings["model_context"] - int(len(prompt) / 2.6) - 256
        text = client.chat(prompt, stage, max_tokens=max(256, min(max_tokens, budget)))
        if stage == "integrate":
            value = io.parse_json(text)
            for action in value.get("next_actions", []):
                if action.get("tool") == "search":
                    query = str(action.get("query") or "")
                    normalized = ' '.join(query.lower().split())
                    if normalized in seen_queries:
                        broader = broaden_query(query)
                        if broader and broader.lower() != normalized:
                            action["query"] = broader
                    seen_queries.add(normalized)
            text = json.dumps(value, ensure_ascii=False)
        return text

    def verify(original_question, state, evidence, step):
        checked, audit = evidence_gate(
            original_question, state,
            lambda ids: evidence.originals(evidence.neighbors(ids, 1)),
            lambda data, questions: judge(data, questions, clients["judge"]),
            threshold=settings["verify_threshold"])
        audit["lan_vetoes"] = sum(need.get("status") == "open" for need in checked["needs"])
        return checked, audit

    steps = settings["temporal_steps"] if TEMPORAL_QUESTION.match(question) else settings["max_steps"]
    trace = research(question, workspace, call, max_steps=steps, verify=verify,
                     replan=lambda q, state, evidence, audit, step: repair_plan(q, state, evidence, audit, step, call))
    return read_answer(question, trace, workspace, clients["reader"]), trace


def run(config, limit=None):
    clients = make_clients(config)
    panel = io.read(workspace_path(config["questions"]))[:limit]
    stores, rows = {}, []
    for item in panel:
        question = {key: item[key] for key in ("id", "conversation", "question")}
        try:
            conversation_id = question["conversation"]
            if conversation_id not in stores:
                stores[conversation_id] = load_store(config, conversation_id)
            answer, trace = answer_question(question["question"], stores[conversation_id], clients, config["research"])
            row = {"id": question["id"], "failed": False, "result": answer,
                   "stop_reason": trace["stop_reason"], "needs": trace["last_state"]["needs"]}
        except Exception as error:
            row = {"id": question["id"], "failed": True, "error": type(error).__name__}
        rows.append(row)
    summary = {"n": len(rows), "failed": sum(row["failed"] for row in rows),
               "stop_reasons": dict(Counter(row.get("stop_reason") for row in rows))}
    io.write(workspace_path(config["run"]) / "rows.json", rows)
    io.write(workspace_path(config["run"]) / "summary.json", summary)
    return summary
