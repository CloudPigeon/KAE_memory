# KAE_memory

This repository contains **PRISM**, a multi-view memory and active evidence-gathering agent for long-term dialogues.

PRISM compiles conversation history into evidence-linked memory, retrieves through aligned views, and answers questions through bounded research over the original turns.

## Reading the algorithm

Start with `src/prism/pipeline.py`. Its `build` and `answer_question` functions connect the method stages.

| Module | Algorithm |
| --- | --- |
| `compilation.py` | Conversational decontextualization and memory-to-turn alignment |
| `two_stage.py`, `memory.py` | Atomic facts, hierarchical organization and temporal memory |
| `retrieve.py`, `probe_loop.py` | Six-view ranking, relevance backprojection and evidence tools |
| `probe_plan_loop.py`, `decision.py`, `replan.py` | Probe, need notebook, integration and bounded research |
| `reader.py`, `normalization.py` | Original-evidence reading and date/count/unit normalization |
| `resources/*.txt` | The role prompts used by these stages |

### Offline memory compilation

1. Bind first-person expressions to the speaker, restore adjacent question context, and retain visual descriptions.
2. Extract self-contained atomic facts from each session with original source IDs.
3. Recall earlier memory for each fact, reserve the fact's own original passages within the source budget, and organize events, states, details and interactions.
4. Align facts and organized schemas to their cited turns. Store their representations alongside the original turn index.

`validate_facts`, `filter_delta` and the memory archive implement the evidence constraints. Records cite visible, non-future sources.

### Six-view retrieval

Each question ranks original turns using six aligned views.

| View | Representation |
| --- | --- |
| Raw dense | Original conversation turn |
| Compiled dense | Decontextualized turn with atomic facts |
| Fact projection | Relevant fact scores projected onto cited turns |
| Schema projection | Relevant organized-memory scores projected onto cited turns |
| Lexical | Plain tokens of compiled turns |
| Stemmed lexical | Compiled turns plus their aligned organized-memory text |

`back_project` transfers relevance through the stored source links. `rrf_order` combines the rankings using `sum_v 1 / (61 + rank_v(turn))`, with ranks starting at zero. Search expands selected anchors by one adjacent turn in the same session.

### Research and reading

The auxiliary model proposes up to three probe queries. The main planner creates a need notebook and selects search, read, thread or session actions. Each round executes up to three actions, integrates their evidence and updates the notebook's source-linked findings. Support and readiness judgments inform the research state.

The default `no_clues` profile uses three research rounds, or two for temporal questions. The final reader receives original turns, with cited neighborhoods placed first and a default 18,000-character utterance budget. Compiled memory contributes through retrieval. The reader path also includes the frozen method's answer refinement and deterministic normalization.

## Run

Use Python 3.10 or later. Run these commands from the repository directory.

```sh
python -m venv .venv
# Activate the environment using your shell.
python -m pip install -e ".[evaluation,test]"
```

Copy `.env.example` to `.env` and fill in the model-service bases and keys. Bases include the API prefix required by your provider.

| Settings | Role |
| --- | --- |
| `PRISM_MAIN_BASE`, `PRISM_MAIN_API_KEY` | Planner and reader |
| `PRISM_AUX_BASE`, `PRISM_AUX_API_KEY` | Probe and evidence judge |
| `PRISM_EMBEDDING_BASE`, `PRISM_EMBEDDING_API_KEY` | Embeddings |
| `PRISM_WRITER_BASE`, `PRISM_WRITER_API_KEY` | Optional writer override, defaulting to the main service |

```sh
kae-memory prepare --source examples/locomo_synthetic.json
kae-memory build --config configs/qwen14b.json
kae-memory run --config configs/qwen14b.json
kae-memory score --predictions outputs/predictions/rows.json --labels data/evaluation/labels.json
```

The example contains two synthetic questions. `build` and `run` call the configured models. Each invocation performs a fresh sequential pass over its selected inputs. Use `run --limit N` to select a question prefix. Failed questions produce a failed row and remain in the scoring denominator.

`configs/qwen14b.json` uses Qwen2.5-14B-Instruct for writing, planning and reading, Qwen3-4B for auxiliary roles, and bge-m3 for embeddings. `configs/gpt4omini.json` selects GPT-4o-mini for planning and reading, with the same offline writer. Role models and method budgets are explicit in the configuration.

The workspace defaults to the current directory. Set `PRISM_WORKSPACE` in the process environment to choose another location. Environment loading uses that workspace's `.env`, or an explicit `PRISM_ENV_FILE`. Local data, stores, predictions and credentials are covered by `.gitignore`.

## Data and scoring

For LoCoMo, pass the original dataset JSON to `prepare`. Add `--conversations` followed by conversation IDs to choose a subset. Preparation writes actors and unlabeled questions separately from evaluation labels. The generated question panel covers categories 1–4. [LoCoMo repository](https://github.com/snap-research/locomo)

LoCoMo F1 uses the audited E-mem normalization and multiset token overlap. Add `--bleu1` for the corresponding NLTK BLEU-1 calculation after installing its tokenizer data.

```sh
python -m nltk.downloader punkt punkt_tab
```

HotpotQA scoring is available through `score --dataset hotpotqa`, using externally prepared predictions and labels. Its lexical scorer preserves the GAM/E-mem convention. Labels map question IDs to objects with `answer` and an optional `category`. Scores are percentages, and missing or failed predictions receive zero within the complete supplied label denominator. [HotpotQA dataset](https://hotpotqa.github.io/)

## Verification and provenance

```sh
python -m pytest -q
```

Tests cover a synthetic build/research/read workflow, source constraints, lexical ranking, answer normalization and the evaluation denominator, using mocked model calls.

The method originates from commit `de03a6a7f66f7e101b2c1409652b4b30bc29d60b`. This distribution presents the active `no_clues` algorithm with direct model calls and a small command interface. The paper's measured results are associated with the frozen experiment implementation at that commit. Distribution validation uses offline algorithm comparisons and mocked model calls.

`RELEASE_MANIFEST.json` records source mappings and file hashes for publication review.

## License

Project code uses the MIT License. The HotpotQA scorer retains its upstream attribution and license notices in `THIRD_PARTY_NOTICES.md` and `third_party/`.
