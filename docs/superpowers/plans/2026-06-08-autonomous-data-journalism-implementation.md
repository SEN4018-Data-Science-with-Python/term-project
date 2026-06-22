# Autonomous Data Journalism Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 4-agent LangGraph pipeline that ingests the Spotify Most Streamed Songs CSV, performs autonomous EDA in an E2B sandbox, drafts a news article, and verifies every numerical claim against Python stdout — fronted by a Gradio UI.

**Architecture:** LangGraph state machine with deterministic Ingest + LLM-driven Analyst, Journalist, Evaluator nodes. E2B sandbox is the only code-execution environment. Three-tier verification (Rank, Sign, Value) routes the article back to the Analyst on failure, capped at 3 revisions.

**Tech Stack:** Python 3.11+, LangGraph, LangChain-OpenAI (GPT-4o), E2B Code Interpreter, Gradio, pandas/numpy/scipy, pytest

**Spec reference:** `docs/superpowers/specs/2026-06-08-autonomous-data-journalism-design.md`

---

## Task 1: Project Skeleton

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/agents/__init__.py`
- Create: `tests/__init__.py`
- Create: `datasets/.gitkeep`
- Create: `ui/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```
langgraph>=0.2.0
langchain-openai>=0.2.0
langchain-core>=0.3.0
e2b-code-interpreter>=1.0.0
gradio>=5.0.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Create .env.example**

```
OPENAI_API_KEY=sk-...
E2B_API_KEY=e2b_...
```

- [ ] **Step 3: Create empty package files**

```bash
mkdir -p src/agents tests datasets ui
touch src/__init__.py src/agents/__init__.py tests/__init__.py datasets/.gitkeep ui/.gitkeep
```

- [ ] **Step 4: Create local .env (user must paste real keys)**

```bash
cp .env.example .env
echo "Edit .env and add your OPENAI_API_KEY and E2B_API_KEY"
```

- [ ] **Step 5: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install without errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example src/ tests/ datasets/ ui/
git commit -m "chore: add project skeleton and dependencies"
```

---

## Task 2: Agent State Schema

**Files:**
- Create: `src/state.py`

- [ ] **Step 1: Write the state TypedDict**

```python
from typing import TypedDict


class AnalysisRun(TypedDict):
    script: str
    stdout: str
    stderr: str


class AgentState(TypedDict):
    dataset_path: str
    schema: dict
    analysis_plan: list[str]
    analysis_results: list[AnalysisRun]
    article_draft: str
    evaluation_errors: list[str]
    revision_count: int
    final_article: str
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from src.state import AgentState, AnalysisRun; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/state.py
git commit -m "feat: add AgentState TypedDict"
```

---

## Task 3: E2B Sandbox Wrapper

**Files:**
- Create: `src/sandbox.py`

- [ ] **Step 1: Write the Sandbox wrapper**

```python
from __future__ import annotations
from e2b_code_interpreter import Sandbox as E2BSandbox


class Sandbox:
    def __init__(self) -> None:
        self._sbx = E2BSandbox()
        self._dataset_remote_path: str | None = None

    def upload(self, local_path: str, remote_name: str = "data.csv") -> str:
        with open(local_path, "rb") as f:
            self._sbx.files.write(f"/home/user/{remote_name}", f.read())
        self._dataset_remote_path = f"/home/user/{remote_name}"
        return self._dataset_remote_path

    def run(self, code: str) -> dict:
        exec_result = self._sbx.run_code(code)
        stdout = "\n".join(exec_result.logs.stdout)
        stderr = "\n".join(exec_result.logs.stderr)
        if exec_result.error:
            stderr += f"\n{exec_result.error.name}: {exec_result.error.value}"
        return {"stdout": stdout, "stderr": stderr}

    @property
    def dataset_path(self) -> str:
        if not self._dataset_remote_path:
            raise RuntimeError("No dataset uploaded")
        return self._dataset_remote_path

    def close(self) -> None:
        self._sbx.kill()
```

- [ ] **Step 2: Smoke test the sandbox**

Run: `python -c "from src.sandbox import Sandbox; s = Sandbox(); print(s.run('print(2+2)')['stdout']); s.close()"`
Expected: prints `4`.

- [ ] **Step 3: Commit**

```bash
git add src/sandbox.py
git commit -m "feat: add E2B sandbox wrapper"
```

---

## Task 4: Ingest Agent (Deterministic)

**Files:**
- Create: `src/agents/ingest.py`

- [ ] **Step 1: Write the ingest function**

```python
from __future__ import annotations
import json
from src.state import AgentState
from src.sandbox import Sandbox


INGEST_CODE = """
import pandas as pd
import json
df = pd.read_csv("{path}")
schema = {{
    "columns": df.columns.tolist(),
    "dtypes": {{c: str(df[c].dtype) for c in df.columns}},
    "n_rows": len(df),
    "describe": df.describe(include="all").fillna("").to_dict(),
}}
print(json.dumps(schema, default=str))
"""


def ingest_node(state: AgentState, sandbox: Sandbox) -> dict:
    remote = sandbox.upload(state["dataset_path"])
    result = sandbox.run(INGEST_CODE.format(path=remote))
    if result["stderr"]:
        raise RuntimeError(f"Ingest failed: {result['stderr']}")
    schema = json.loads(result["stdout"].strip().splitlines()[-1])
    return {"schema": schema, "analysis_plan": _initial_plan(schema)}


def _initial_plan(schema: dict) -> list[str]:
    cols = schema["columns"]
    return [
        f"Identify the top 10 entries ranked by the most prominent count/streams column among {cols}.",
        f"Compute pairwise correlations between numeric columns and surface the 3 strongest (by |r|).",
        f"Identify any temporal trend column (year, release_date) and describe how key metrics shift across it.",
    ]
```

- [ ] **Step 2: Commit**

```bash
git add src/agents/ingest.py
git commit -m "feat: add deterministic ingest agent"
```

---

## Task 5: Evaluator — Tests First (TDD)

**Files:**
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Write failing tests for all three verification tiers**

```python
from src.agents.evaluator import (
    verify_values,
    verify_signs,
    verify_ranks,
)


def test_value_match_pass():
    article = "The track scored 1,234,567 streams."
    stdout = "streams = 1234567"
    assert verify_values(article, [stdout]) == []


def test_value_mismatch_fails():
    article = "The track scored 1,234,567 streams."
    stdout = "streams = 7654321"
    errors = verify_values(article, [stdout])
    assert any("1,234,567" in e or "1234567" in e for e in errors)


def test_sign_positive_match():
    article = "There is a positive correlation between BPM and energy."
    stdout = "corr(BPM, energy) = 0.42"
    assert verify_signs(article, [stdout]) == []


def test_sign_mismatch_fails():
    article = "There is a positive correlation between BPM and energy."
    stdout = "corr(BPM, energy) = -0.42"
    errors = verify_signs(article, [stdout])
    assert errors  # at least one error reported


def test_rank_top_match():
    article = "Taylor Swift is the most streamed artist."
    stdout = "[('Taylor Swift', 100), ('Drake', 80), ('Bad Bunny', 60)]"
    assert verify_ranks(article, [stdout]) == []


def test_rank_top_mismatch():
    article = "Drake is the most streamed artist."
    stdout = "[('Taylor Swift', 100), ('Drake', 80), ('Bad Bunny', 60)]"
    errors = verify_ranks(article, [stdout])
    assert errors
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_evaluator.py -v`
Expected: ImportError or all tests fail (module doesn't exist yet).

---

## Task 6: Evaluator — Implementation

**Files:**
- Create: `src/agents/evaluator.py`

- [ ] **Step 1: Write the evaluator with three verifiers + node function**

```python
from __future__ import annotations
import re
from src.state import AgentState

MAX_REVISIONS = 3

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_POS_CORR_RE = re.compile(
    r"(positive(?:ly)?\s+(?:correlat|associat|relat))",
    re.IGNORECASE,
)
_NEG_CORR_RE = re.compile(
    r"(negative(?:ly)?\s+(?:correlat|associat|relat)|inverse(?:ly)?\s+(?:correlat|relat))",
    re.IGNORECASE,
)
_SUPERLATIVE_RE = re.compile(
    r"(?P<entity>[A-Z][\w\.&' -]{1,40}?)\s+is\s+the\s+most\s+(?P<metric>[a-z]+)",
    re.IGNORECASE,
)


def _normalize_num(token: str) -> str:
    return token.replace(",", "")


def verify_values(article: str, stdouts: list[str]) -> list[str]:
    errors: list[str] = []
    joined = "\n".join(stdouts)
    stdout_nums = {_normalize_num(t) for t in _NUMBER_RE.findall(joined)}
    for token in _NUMBER_RE.findall(article):
        norm = _normalize_num(token)
        if len(norm) < 2:
            continue
        if norm in stdout_nums:
            continue
        try:
            val = float(norm)
        except ValueError:
            continue
        tolerance = max(abs(val) * 0.01, 0.01)
        if not any(
            abs(val - float(s)) <= tolerance
            for s in stdout_nums
            if _is_floatable(s)
        ):
            errors.append(f"Value '{token}' not found in stdout (no match within 1% tolerance)")
    return errors


def _is_floatable(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def verify_signs(article: str, stdouts: list[str]) -> list[str]:
    errors: list[str] = []
    joined = "\n".join(stdouts)
    pos_in_article = bool(_POS_CORR_RE.search(article))
    neg_in_article = bool(_NEG_CORR_RE.search(article))

    corr_matches = re.findall(r"corr[^=]*=\s*(-?\d*\.?\d+)", joined, re.IGNORECASE)
    if not corr_matches:
        return errors
    signs = {("neg" if float(v) < 0 else "pos") for v in corr_matches}

    if pos_in_article and "pos" not in signs:
        errors.append("Article claims positive correlation but stdout shows none.")
    if neg_in_article and "neg" not in signs:
        errors.append("Article claims negative correlation but stdout shows none.")
    return errors


def verify_ranks(article: str, stdouts: list[str]) -> list[str]:
    errors: list[str] = []
    joined = "\n".join(stdouts)
    matches = list(_SUPERLATIVE_RE.finditer(article))
    if not matches:
        return errors
    list_pattern = re.compile(r"\[\s*\(([^)]+)\)")
    first_items = list_pattern.findall(joined)
    if not first_items:
        return errors
    top_names = []
    for item in first_items:
        parts = [p.strip().strip("'\"") for p in item.split(",")]
        if parts:
            top_names.append(parts[0].lower())
    for m in matches:
        entity = m.group("entity").strip().lower()
        if entity not in top_names:
            errors.append(
                f"Article claims '{m.group('entity')}' is the most {m.group('metric')}, "
                f"but stdout's top entries are: {top_names}"
            )
    return errors


def evaluator_node(state: AgentState) -> dict:
    stdouts = [r["stdout"] for r in state["analysis_results"]]
    article = state["article_draft"]
    errors = (
        verify_values(article, stdouts)
        + verify_signs(article, stdouts)
        + verify_ranks(article, stdouts)
    )
    revision_count = state.get("revision_count", 0)
    if not errors:
        return {"evaluation_errors": [], "final_article": article}
    if revision_count + 1 >= MAX_REVISIONS:
        tagged = article + "\n\n[unverified] " + " ".join(errors)
        return {
            "evaluation_errors": errors,
            "revision_count": revision_count + 1,
            "final_article": tagged,
        }
    return {
        "evaluation_errors": errors,
        "revision_count": revision_count + 1,
    }


def evaluator_route(state: AgentState) -> str:
    if state.get("final_article"):
        return "end"
    return "analyst"
```

- [ ] **Step 2: Run tests to confirm they pass**

Run: `pytest tests/test_evaluator.py -v`
Expected: all 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/agents/evaluator.py tests/test_evaluator.py
git commit -m "feat: add three-tier evaluator with TDD"
```

---

## Task 7: Analyst Agent (LLM)

**Files:**
- Create: `src/agents/analyst.py`

- [ ] **Step 1: Write the analyst node**

```python
from __future__ import annotations
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.sandbox import Sandbox

SYSTEM = """You are a senior data analyst. You write short, focused Python scripts that print results clearly.

Rules:
- Use only pandas, numpy, scipy. Read the dataset from the provided path with pd.read_csv.
- ALL findings must be emitted via print(). Do NOT use display() or notebook magics.
- When ranking, print the full sorted list as a Python list of (entity, value) tuples.
- When computing correlations, print as: corr(col_a, col_b) = 0.42 (always include both columns and the rounded value).
- Print each statistic on its own line with a label.
- Output ONE python code block. No prose.
"""

USER_TEMPLATE = """Dataset path: {path}
Schema (columns and dtypes): {schema}

Analysis tasks:
{tasks}

{revision_notes}

Write a single Python script that runs all the analysis tasks and prints results in the format described.
"""


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def analyst_node(state: AgentState, sandbox: Sandbox) -> dict:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    revision_notes = ""
    if state.get("evaluation_errors"):
        revision_notes = (
            "PREVIOUS ARTICLE FAILED VERIFICATION:\n"
            + "\n".join(f"- {e}" for e in state["evaluation_errors"])
            + "\nProvide additional statistics that resolve these issues."
        )
    user = USER_TEMPLATE.format(
        path=sandbox.dataset_path,
        schema={
            "columns": state["schema"]["columns"],
            "dtypes": state["schema"]["dtypes"],
        },
        tasks="\n".join(f"- {t}" for t in state["analysis_plan"]),
        revision_notes=revision_notes,
    )
    response = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=user)])
    code = _extract_code(response.content)
    result = sandbox.run(code)
    run = {"script": code, "stdout": result["stdout"], "stderr": result["stderr"]}
    return {"analysis_results": state.get("analysis_results", []) + [run]}
```

- [ ] **Step 2: Commit**

```bash
git add src/agents/analyst.py
git commit -m "feat: add analyst agent (GPT-4o + E2B)"
```

---

## Task 8: Journalist Agent (LLM)

**Files:**
- Create: `src/agents/journalist.py`

- [ ] **Step 1: Write the journalist node**

```python
from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState

SYSTEM = """You are a senior data journalist for a Reuters-style wire service.

Rules:
- Every numerical claim MUST come from the provided analysis stdout. Never invent figures.
- Cite ranks with ordinal phrasing ("the second-most-streamed").
- When mentioning correlations, state the direction explicitly ("positive correlation", "negative correlation").
- Tone: factual, concise, news-style. No hype. No editorializing.
- Structure: HEADLINE on the first line, then a lede paragraph, 3-5 body paragraphs, then a closing line.
- Output the article only, no commentary or markdown.
"""

USER_TEMPLATE = """Analysis stdout from the data team:

{analyses}

{revision_notes}

Draft the article.
"""


def journalist_node(state: AgentState) -> dict:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    analyses = "\n\n---\n\n".join(
        f"Script:\n{r['script']}\n\nStdout:\n{r['stdout']}"
        for r in state["analysis_results"]
    )
    revision_notes = ""
    if state.get("evaluation_errors"):
        revision_notes = (
            "PREVIOUS DRAFT HAD ERRORS:\n"
            + "\n".join(f"- {e}" for e in state["evaluation_errors"])
            + "\nRewrite to remove these errors. Use only verified figures."
        )
    user = USER_TEMPLATE.format(analyses=analyses, revision_notes=revision_notes)
    response = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=user)])
    return {"article_draft": response.content.strip()}
```

- [ ] **Step 2: Commit**

```bash
git add src/agents/journalist.py
git commit -m "feat: add journalist agent (GPT-4o)"
```

---

## Task 9: LangGraph Wiring

**Files:**
- Create: `src/graph.py`

- [ ] **Step 1: Write the graph builder + CLI entry**

```python
from __future__ import annotations
import sys
from functools import partial
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.sandbox import Sandbox
from src.agents.ingest import ingest_node
from src.agents.analyst import analyst_node
from src.agents.journalist import journalist_node
from src.agents.evaluator import evaluator_node, evaluator_route


def build_graph(sandbox: Sandbox):
    workflow = StateGraph(AgentState)
    workflow.add_node("ingest", partial(ingest_node, sandbox=sandbox))
    workflow.add_node("analyst", partial(analyst_node, sandbox=sandbox))
    workflow.add_node("journalist", journalist_node)
    workflow.add_node("evaluator", evaluator_node)

    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "analyst")
    workflow.add_edge("analyst", "journalist")
    workflow.add_edge("journalist", "evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        evaluator_route,
        {"analyst": "analyst", "end": END},
    )
    return workflow.compile()


def run(dataset_path: str) -> str:
    load_dotenv()
    sandbox = Sandbox()
    try:
        graph = build_graph(sandbox)
        initial: AgentState = {
            "dataset_path": dataset_path,
            "schema": {},
            "analysis_plan": [],
            "analysis_results": [],
            "article_draft": "",
            "evaluation_errors": [],
            "revision_count": 0,
            "final_article": "",
        }
        final = graph.invoke(initial, config={"recursion_limit": 25})
        return final["final_article"]
    finally:
        sandbox.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.graph <path-to-csv>")
        sys.exit(1)
    article = run(sys.argv[1])
    print("\n=== FINAL ARTICLE ===\n")
    print(article)
```

- [ ] **Step 2: Commit**

```bash
git add src/graph.py
git commit -m "feat: wire LangGraph state machine with revision loop"
```

---

## Task 10: Download Dataset and Smoke-Test the Pipeline

**Files:**
- Modify: `datasets/spotify_2024.csv` (download manually)

- [ ] **Step 1: Download dataset**

Manually download "Spotify Most Streamed Songs" from Kaggle (https://www.kaggle.com/datasets/abdulszz/spotify-most-streamed-songs or similar) and save as `datasets/spotify_2024.csv`.

If Kaggle CLI is configured:
```bash
kaggle datasets download -d abdulszz/spotify-most-streamed-songs -p datasets/ --unzip
mv datasets/Spotify*.csv datasets/spotify_2024.csv
```

- [ ] **Step 2: Verify keys are loaded**

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); assert os.getenv('OPENAI_API_KEY'); assert os.getenv('E2B_API_KEY'); print('keys ok')"
```

Expected: prints `keys ok`.

- [ ] **Step 3: Run the full pipeline**

Run: `python -m src.graph datasets/spotify_2024.csv`
Expected: graph runs through all 4 nodes, prints a complete article at the end.

- [ ] **Step 4: If smoke test fails, debug and re-run**

Common failures:
- E2B import error → check `e2b-code-interpreter` is installed
- OpenAI 401 → check `.env` is loaded
- Analyst code error → inspect `analysis_results[-1]['stderr']`
- Evaluator infinite loop → already handled by `MAX_REVISIONS`

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: smoke test corrections"
```

---

## Task 11: Gradio UI

**Files:**
- Create: `ui/app.py`

- [ ] **Step 1: Write the Gradio app with streaming**

```python
from __future__ import annotations
import os
import tempfile
import gradio as gr
from dotenv import load_dotenv
from functools import partial
from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.sandbox import Sandbox
from src.agents.ingest import ingest_node
from src.agents.analyst import analyst_node
from src.agents.journalist import journalist_node
from src.agents.evaluator import evaluator_node, evaluator_route

load_dotenv()


def _build(sandbox: Sandbox):
    workflow = StateGraph(AgentState)
    workflow.add_node("ingest", partial(ingest_node, sandbox=sandbox))
    workflow.add_node("analyst", partial(analyst_node, sandbox=sandbox))
    workflow.add_node("journalist", journalist_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "analyst")
    workflow.add_edge("analyst", "journalist")
    workflow.add_edge("journalist", "evaluator")
    workflow.add_conditional_edges(
        "evaluator", evaluator_route, {"analyst": "analyst", "end": END}
    )
    return workflow.compile()


def run_pipeline(csv_file):
    if csv_file is None:
        yield "Please upload a CSV first.", ""
        return
    sandbox = Sandbox()
    try:
        graph = _build(sandbox)
        initial: AgentState = {
            "dataset_path": csv_file.name,
            "schema": {},
            "analysis_plan": [],
            "analysis_results": [],
            "article_draft": "",
            "evaluation_errors": [],
            "revision_count": 0,
            "final_article": "",
        }
        log = ""
        article = ""
        for event in graph.stream(initial, config={"recursion_limit": 25}):
            for node, update in event.items():
                log += f"\n▶ {node} completed\n"
                if "analysis_results" in update and update["analysis_results"]:
                    last = update["analysis_results"][-1]
                    log += f"  stdout (truncated):\n  {last['stdout'][:500]}\n"
                if "evaluation_errors" in update and update["evaluation_errors"]:
                    log += f"  ⚠ errors: {update['evaluation_errors']}\n"
                if "final_article" in update and update["final_article"]:
                    article = update["final_article"]
                yield log, article
    finally:
        sandbox.close()


with gr.Blocks(title="Autonomous Data Journalism Agent") as demo:
    gr.Markdown("# Autonomous Data Journalism Agent")
    gr.Markdown(
        "Upload a CSV of entertainment industry data. The agent will perform EDA, draft an article, and verify every numerical claim."
    )
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="CSV dataset", file_types=[".csv"])
            run_btn = gr.Button("Run pipeline", variant="primary")
            log_output = gr.Textbox(label="Live agent log", lines=20, max_lines=40)
        with gr.Column():
            article_output = gr.Markdown(label="Final article")
    run_btn.click(run_pipeline, inputs=[file_input], outputs=[log_output, article_output])


if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Smoke test the UI**

Run: `python ui/app.py`
Expected: Gradio launches at `http://127.0.0.1:7860`. Upload `datasets/spotify_2024.csv`, click Run, watch logs stream, article appears.

- [ ] **Step 3: Commit**

```bash
git add ui/app.py
git commit -m "feat: add Gradio UI with streaming log"
```

---

## Task 12: Hugging Face Spaces Deployment (Optional — if time permits)

**Files:**
- Create: `app.py` (HF Spaces entry — re-exports from `ui/app.py`)
- Create: `README_HF.md` (with Space metadata)

- [ ] **Step 1: Create root-level app.py for HF Spaces**

```python
from ui.app import demo

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Create HF Space metadata file**

`README_HF.md`:
```yaml
---
title: Autonomous Data Journalism Agent
emoji: 📰
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---
```

- [ ] **Step 3: Create HF Space, push, set secrets**

Manual steps:
1. On https://huggingface.co/new-space — choose "Gradio" SDK
2. Clone the space repo locally, copy files over, push
3. In the Space settings → Variables and secrets, add `OPENAI_API_KEY` and `E2B_API_KEY`
4. Wait for build to finish, test by uploading the Spotify CSV

- [ ] **Step 4: Commit HF files**

```bash
git add app.py README_HF.md
git commit -m "feat: add Hugging Face Spaces deployment files"
```

---

## Verification Checklist (Done When All Pass)

- [ ] `pytest tests/ -v` → all evaluator tests pass
- [ ] `python -m src.graph datasets/spotify_2024.csv` → prints final article with no `[unverified]` tag (or at most one)
- [ ] `python ui/app.py` → Gradio UI loads, full pipeline runs, article appears
- [ ] Final article cites at least one rank, one correlation, and one numerical value — and all three pass verification
- [ ] `git log --oneline` shows clean, atomic commits per task

---

## Notes for the Implementer

- **GPT-4o code generation:** occasionally the analyst will emit a code block that uses `df` before reading the CSV — the system prompt and revision-loop both mitigate this. If you see it consistently, lower temperature to `0.1`.
- **E2B billing:** each smoke run uses ~1 sandbox minute. Close sandboxes promptly (already in `finally`).
- **Schema size:** for the Spotify dataset (~24 columns) the schema dict is small enough to pass to GPT-4o without truncation. If you switch to TMDB later, you'll need to summarize `describe()` output.
- **Evaluator false positives:** the regex-based parser is conservative — it sometimes flags real numbers that appear in scientific notation. If you see this, extend `_NUMBER_RE`. Keep the test file updated.
