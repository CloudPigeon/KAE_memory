"""Offline release checks using synthetic data and a mock HTTP transport."""
import hashlib
import json
import os
import socket
from pathlib import Path

import httpx
import pytest

from prism import io, release, pipeline
from prism.metrics import locomo_f1, summarize
from prism.settings import load_environment

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    previous_environment = dict(os.environ)
    directory = tmp_path / "workspace"
    directory.mkdir()
    monkeypatch.chdir(directory)
    monkeypatch.setattr(io, "ROOT", directory)
    monkeypatch.setenv("PRISM_WORKSPACE", str(directory))
    monkeypatch.setenv("PRISM_ENV_FILE", str(directory / ".env"))
    for service in ("MAIN", "AUX", "EMBEDDING"):
        monkeypatch.setenv("PRISM_" + service + "_BASE", "https://" + service.lower() + ".invalid/v1")
        monkeypatch.setenv("PRISM_" + service + "_API_KEY", "offline-test-credential")
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("unexpected network request"))
    yield directory
    os.environ.clear()
    os.environ.update(previous_environment)


@pytest.fixture
def mock_models(monkeypatch):
    from prism import probe_loop, retrieve, two_stage

    for module in (probe_loop, retrieve, two_stage):
        monkeypatch.setattr(module, "tokens", lambda text: len(str(text).split()))
    requests = []

    def response(request):
        body = json.loads(request.content)
        requests.append(body)
        if request.url.path.endswith("/embeddings"):
            data = []
            for index, text in enumerate(body["input"]):
                digest = hashlib.sha256(text.encode()).digest()
                data.append({"index": index, "embedding": [(value + 1) / 256 for value in digest[:8]]})
            return httpx.Response(200, request=request, json={"data": data, "usage": {"total_tokens": 1}})
        prompt = body["messages"][0]["content"]
        if "You extract self-contained facts" in prompt[:160]:
            import re
            ids = re.findall(r"example-memory:D([12]):2", prompt)
            session = max(ids)
            source_id = "example-memory:D" + session + ":2"
            text = "Alex moved to Leeds." if session == "1" else "Alex cycles to work."
            result = {"facts": [{"text": text, "time_expression": "", "evidence_ids": [source_id]}]}
        elif "You organize new facts" in prompt[:160]:
            import re
            session = max(re.findall(r"example-memory:D([12]):2", prompt))
            result = {"events": [], "updates": [], "loops": [], "details": [{
                "subject": "Alex", "description": "Lives in Leeds" if session == "1" else "Cycles to work",
                "evidence_ids": ["example-memory:D" + session + ":2"]}]}
        elif "You are an information-gathering planner" in prompt[:160]:
            result = {"scope": "Alex's stated information", "needs": [{"need": "the requested fact", "requires_answer": True}],
                      "next_actions": [{"tool": "read", "source_ids": ["example-memory:D1:2", "example-memory:D2:2"], "radius": 1}]}
        elif "You are an evidence integration" in prompt[:160]:
            result = {"updates": [{"need_id": "n1", "status": "supported", "finding": "Alex lives in Leeds and cycles to work.",
                                   "source_ids": ["example-memory:D1:2", "example-memory:D2:2"]}],
                      "synthesis": "Alex lives in Leeds and cycles to work.", "ready": True, "next_actions": []}
        elif "You are an evidence judge" in prompt[:160] or "question-readiness assessor" in prompt[:160]:
            result = {"verdict": "supported", "quote": "I moved to Leeds last month.", "reason": "The originals state the fact."}
        elif "\nQuestion: " in prompt and body["max_tokens"] == 500:
            result = {"queries": [prompt.rsplit("\nQuestion: ", 1)[1]]}
        else:
            answer = "By bicycle" if prompt.endswith("How does Alex travel to work?") else "Leeds"
            result = {"answer": answer, "evidence_ids": ["example-memory:D1:2", "example-memory:D2:2"]}
        payload = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return httpx.Response(200, request=request, json=payload)

    monkeypatch.setattr(httpx.Client, "send", lambda self, request, **kwargs: response(request))
    return requests


def test_synthetic_algorithm_pipeline(workspace, mock_models):
    source = REPOSITORY / "examples" / "locomo_synthetic.json"
    config = REPOSITORY / "configs" / "qwen14b.json"
    assert release.main(["prepare", "--source", str(source)]) == 0
    actor = io.read(workspace / "data/actors/example-memory.json")
    questions = io.read(workspace / "data/questions.json")
    assert set(actor) == {"id", "participants", "sessions", "turns"}
    assert all(set(question) == {"id", "conversation", "question"} for question in questions)
    assert release.main(["build", "--config", str(config)]) == 0
    store = workspace / "outputs/memory/memory/example-memory"
    assert len(io.read(store / "records.json")) == 2
    views = io.read(store / "turn-multiview.json")
    assert len(views["tids"]) == 4 and len(views["org_vecs"]) == 2
    loaded = pipeline.load_store(release.load_config(config), "example-memory")
    from prism.probe_loop import Workspace
    evidence = Workspace(loaded["actor"], loaded["memory"], loaded["raw"], None, loaded["multiview"])
    assert evidence.rank_turns("Leeds")[0] == "example-memory:D1:2"
    assert release.main(["run", "--config", str(config)]) == 0
    predictions = workspace / "outputs/predictions/rows.json"
    rows = io.read(predictions)
    assert len(rows) == 2 and all(not row["failed"] for row in rows)
    assert all(row["stop_reason"] == "ready" for row in rows)
    assert all(need["source_ids"] for row in rows for need in row["needs"] if need["status"] == "supported")
    assert release.main(["score", "--predictions", str(predictions), "--labels",
                         str(workspace / "data/evaluation/labels.json")]) == 0
    assert not list((workspace / "outputs").rglob("*checkpoint*"))
    assert "offline-test-credential" not in predictions.read_text(encoding="utf-8")


def test_facts_and_notebook_require_real_sources(workspace):
    from prism.two_stage import validate_facts
    from prism.probe_plan_loop import update_notebook
    release.prepare(REPOSITORY / "examples/locomo_synthetic.json")
    actor = io.read(workspace / "data/actors/example-memory.json")
    payload = {"facts": [{"text": "stated fact", "evidence_ids": ["example-memory:D1:2"]},
                         {"text": "future fact", "evidence_ids": ["example-memory:D2:2"]}]}
    facts, rejected = validate_facts(payload, actor, actor["sessions"][0])
    assert len(facts) == 1 and len(rejected) == 1
    state = {"needs": [{"need_id": "n1", "status": "open"}]}
    decision = {"updates": [{"need_id": "n1", "status": "supported", "source_ids": ["invented"]}]}
    updated, feedback = update_notebook(state, decision, {"example-memory:D1:2": {}})
    assert updated["needs"][0]["status"] == "open" and feedback


def test_failed_questions_keep_the_full_denominator(workspace, mock_models, monkeypatch):
    release.prepare(REPOSITORY / "examples/locomo_synthetic.json")
    config = release.load_config(REPOSITORY / "configs/qwen14b.json")
    pipeline.build(config)
    original = pipeline.answer_question

    def answer(question, *args):
        if question.startswith("Where"):
            raise RuntimeError("offline-test-credential")
        return original(question, *args)

    monkeypatch.setattr(pipeline, "answer_question", answer)
    assert pipeline.run(config)["failed"] == 1
    rows = io.read(workspace / "outputs/predictions/rows.json")
    labels = io.read(workspace / "data/evaluation/labels.json")
    assert len(rows) == 2 and rows[0]["error"] == "RuntimeError"
    assert summarize(rows, labels)["n"] == 2
    assert "offline-test-credential" not in json.dumps(rows)


def test_lexical_scores_and_missing_predictions():
    labels = {"a": {"answer": "Leeds", "category": 4}, "b": {"answer": "bicycle", "category": 1}}
    scores = summarize([{"id": "a", "result": {"answer": "Leeds"}}], labels)
    assert scores["n"] == 2 and scores["missing"] == 1 and scores["f1"] == 50
    assert locomo_f1("A long-term plan", "longterm plan") == 1


def test_date_and_quantity_normalization():
    from prism.normalization import present_answer, canonical_count
    assert present_answer("yesterday", "10:00 am on 2 May, 2026") == "1 May 2026"
    assert canonical_count("How many bicycles?", "4 bicycles") == "four bicycles"


def test_environment_loading_uses_the_workspace(workspace, monkeypatch):
    (workspace.parent / ".env").write_text("UNEXPECTED_PARENT_VALUE=parent-value", encoding="utf-8")
    assert "UNEXPECTED_PARENT_VALUE" not in load_environment()
    (workspace / ".env").write_text("EXAMPLE_VALUE=local-value", encoding="utf-8")
    assert load_environment()["EXAMPLE_VALUE"] == "local-value"
