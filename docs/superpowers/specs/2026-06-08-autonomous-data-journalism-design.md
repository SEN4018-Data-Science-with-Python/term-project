# Autonomous Data Journalism Agent — Design Spec

**Date:** 2026-06-08
**Course:** SEN4018 Term Project
**Status:** Approved — ready for implementation

## 1. Goal

Build a multi-agent AI system that ingests a raw entertainment-industry CSV dataset, autonomously performs exploratory data analysis in a sandboxed Python environment, and generates a fact-verified news article with no manual intervention. Every numerical claim in the final article must be traceable to a Python stdout line from the sandbox.

The design follows the spec in `README.md`, `CLAUDE.md`, and the Medium article (https://medium.com/@emir.genc/an-agentic-system-for-autonomous-data-journalism-with-entertainment-industry-datasets-8e7da6505f29). No deviations.

## 2. Tech Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph |
| Code Execution | E2B Data Analysis Sandbox |
| LLM | OpenAI GPT-4o (via `ChatOpenAI`) |
| UI | Gradio on Hugging Face Spaces |
| Target Dataset | Spotify Most Streamed Songs |

## 3. Architecture

### 3.1 Graph Topology

```
        ┌──────────┐
        │  ingest  │
        └────┬─────┘
             ▼
        ┌──────────┐
   ┌───▶│ analyst  │
   │    └────┬─────┘
   │         ▼
   │   ┌──────────────┐
   │   │  journalist  │
   │   └────┬─────────┘
   │        ▼
   │   ┌────────────┐
   │   │ evaluator  │
   │   └─┬──────┬───┘
   │     │      │
   │ fail│      │pass
   └─────┘      ▼
              END
```

The evaluator routes back to the analyst on failure. A circuit breaker forces END after `revision_count >= 3` even if evaluation still fails (the article is returned with a "best-effort" disclaimer).

### 3.2 State Schema

`src/state.py`:

```python
from typing import TypedDict

class AgentState(TypedDict):
    dataset_path: str            # local path to CSV
    schema: dict                 # {columns, dtypes, describe}
    analysis_plan: list[str]     # questions the analyst will answer
    analysis_results: list[dict] # [{script, stdout, stderr}, ...]
    article_draft: str
    evaluation_errors: list[str] # human-readable failed checks
    revision_count: int          # incremented on each loop back
    final_article: str           # populated only at terminal node
```

### 3.3 Agent Responsibilities

| Agent | File | Input from state | Output to state |
|---|---|---|---|
| Ingest | `src/agents/ingest.py` | `dataset_path` | `schema` |
| Analyst | `src/agents/analyst.py` | `schema`, `analysis_plan`, `evaluation_errors` | `analysis_results` |
| Journalist | `src/agents/journalist.py` | `analysis_results` | `article_draft` |
| Evaluator | `src/agents/evaluator.py` | `article_draft`, `analysis_results` | `evaluation_errors`, `revision_count`, `final_article` (if pass) |

### 3.4 Ingest Agent

Deterministic — no LLM call. Uploads CSV to E2B sandbox, runs:

```python
import pandas as pd
df = pd.read_csv("/data.csv")
print(df.columns.tolist())
print(df.dtypes.to_dict())
print(df.describe(include="all").to_dict())
```

Stdout is parsed into `schema`. **Raw rows never reach the LLM** — context is preserved for the heavy reasoning agents.

### 3.5 Analyst Agent

LLM prompt receives:
- `schema` (compact representation)
- `analysis_plan` (initial set of hypotheses, e.g. "correlate BPM with stream count", "rank top 10 artists by total streams")
- `evaluation_errors` if this is a retry iteration (so it knows what to fix)

LLM returns Python code wrapped in a code block. The code is executed in E2B; stdout/stderr is captured and appended to `analysis_results`. Each run includes the script source for traceability.

System prompt enforces:
- All claims must be printed with `print()` (no `display()`, no notebook magic)
- Use only `pandas`, `numpy`, `scipy`
- Print rankings as ordered lists with explicit indices
- Print correlations with both magnitude and sign

### 3.6 Journalist Agent

LLM prompt receives only `analysis_results` (script + stdout for each run). Drafts a coherent news narrative — Reuters/BBC tone, not blog-style. Required structure: headline, lede, body (3-5 paragraphs), closing.

System prompt enforces:
- Every numerical claim must come from stdout (no invented stats)
- Use ordinal phrasing for ranks ("the second-most-streamed")
- Specify the direction of correlations explicitly

### 3.7 Evaluator Agent — Three-Tier Verification

`src/agents/evaluator.py` runs a **deterministic parser first**, then optionally invokes the LLM for semantic checks.

**Rank Verification.** Regex finds superlative phrasings (`most`, `least`, `top N`, `highest`, `lowest`, ordinals like `first`, `second`). For each, the parser looks up the matching ranked list in `analysis_results` and confirms the named entity occupies the claimed position.

**Sign Verification.** Regex finds correlation phrasings (`positive correlation`, `negatively associated`, `inverse relationship`). For each, the parser finds the corresponding correlation value in stdout and confirms the sign matches.

**Value Verification.** Regex finds numeric tokens (with optional units, %, M, B suffixes). Each is checked against stdout — must appear exactly (allowing for rounding tolerance of ±1% on derived values, exact match for raw counts).

If any check fails, the error message is appended to `evaluation_errors`, `revision_count` is incremented, and the conditional edge routes back to the Analyst. If `revision_count >= 3`, the article is finalized with a `[unverified]` tag on any failed claims.

### 3.8 E2B Sandbox Wrapper

`src/sandbox.py`:

```python
class Sandbox:
    def __init__(self, dataset_path: str): ...
    def upload(self, path: str) -> str: ...      # returns sandbox path
    def run(self, code: str) -> dict:            # {stdout, stderr, exit_code}
        ...
    def close(self): ...
```

The sandbox is created once at graph start (in the ingest node) and reused across all analyst runs. Closed in a `finally` block at graph exit.

### 3.9 Gradio UI

`ui/app.py`:
- File upload component for CSV
- "Run" button triggers `graph.invoke(initial_state)`
- Live log: streams each node's output as it completes (using LangGraph's `stream()` instead of `invoke()`)
- Final article rendered as markdown

Deployable to Hugging Face Spaces with `app.py` at the repo root or via `huggingface_hub` push.

## 4. Project Structure

```
term-project/
├── src/
│   ├── __init__.py
│   ├── state.py
│   ├── graph.py
│   ├── sandbox.py
│   └── agents/
│       ├── __init__.py
│       ├── ingest.py
│       ├── analyst.py
│       ├── journalist.py
│       └── evaluator.py
├── ui/
│   └── app.py
├── datasets/
│   └── spotify_2024.csv
├── docs/superpowers/specs/2026-06-08-autonomous-data-journalism-design.md
├── tests/
│   └── test_evaluator.py   # unit tests for the three-tier parser
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── CLAUDE.md
```

## 5. Dependencies

```
langgraph>=0.2
langchain-openai>=0.2
e2b-code-interpreter>=1.0
gradio>=5.0
pandas
numpy
scipy
python-dotenv
```

## 6. Error Handling & Circuit Breakers

- **E2B execution errors:** stderr is captured and appended to `analysis_results` so the analyst can self-correct on the next iteration.
- **LLM API failures:** Wrapped in retry with exponential backoff (3 attempts).
- **Revision loop limit:** Hard cap at `MAX_REVISIONS = 3`. After that, finalize with `[unverified]` flags.
- **Graph timeout:** Total graph execution capped at 10 minutes via LangGraph's `recursion_limit`.

## 7. Verification

End-to-end smoke test:

```bash
python -m src.graph datasets/spotify_2024.csv
```

Expected: prints final article to stdout, exit code 0.

Unit test (`tests/test_evaluator.py`):
- Feed evaluator a synthetic article with known correct/incorrect numbers
- Confirm Rank/Sign/Value verifiers each catch their respective error types

UI smoke test: launch `python ui/app.py`, upload `spotify_2024.csv`, confirm article renders.

## 8. Out of Scope (Explicit)

- Auth on the Gradio app
- Persistence of past runs / database
- Datasets other than the Spotify CSV in V1
- Streaming token-level output (we stream node-level only)
- Production observability beyond Hugging Face Spaces logs
