# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Autonomous Data Journalism Agent** — a multi-agent AI system that ingests raw CSV datasets (entertainment industry), performs autonomous EDA, and generates fact-verified news articles with no manual intervention. All numerical/statistical claims are verified by executing Python in a sandboxed environment before being included in the final article.

Full design article: https://medium.com/@emir.genc/an-agentic-system-for-autonomous-data-journalism-with-entertainment-industry-datasets-8e7da6505f29

## Planned Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (state machine with cyclic loop control) |
| Code Execution | E2B Data Analysis Sandbox (pandas, numpy, scipy) |
| LLMs | Google Gemini (primary) with Qwen fallback via `src/llm.py` `get_llm()` (Analyst, Journalist, Evaluator agents) |
| UI | Gradio on Hugging Face Spaces |

## Agent Architecture

The pipeline is a **LangGraph state machine** with four distinct roles:

1. **Ingestion Agent** — deterministic (no LLM). Reads CSV headers, dtypes, and sampled descriptive stats to map the data schema. Accepts a local CSV, a pre-uploaded `/home/user/...` path, or a `kaggle:owner/slug` ref. For Kaggle refs the dataset is downloaded **inside the sandbox** via `kagglehub` (avoids fragile host→sandbox uploads of huge files); **every** CSV in the dataset is profiled so the analyst can join across tables. For large datasets, only the schema is passed to the LLM (not raw rows).
2. **Analyst Agent** — generates and executes Python scripts inside E2B sandbox; reads stdout/stderr back into the graph state.
3. **Journalist Agent** — drafts a news article from the statistically significant findings.
4. **Evaluator Agent** — deterministic parser + LLM combo. Regex tiers (`verify_signs`, `verify_ranks`, `verify_values`) extract claims; an LLM tier (`verify_semantic`) judges claims that regex cannot (trends, mis-paired correlations, contextual numbers). `select_blocking_errors` combines them: sign + rank + semantic block (force a revision); raw value-regex hits are advisory only (they caused false-positive churn). On a blocking mismatch, routes back to the Analyst (circuit-broken via `MAX_REVISIONS`).

## Key Design Constraints

- **No arithmetic hallucination:** Every number in the final article must trace to a Python stdout line from E2B. The Evaluator enforces three named tiers:
  - **Rank Verification** — confirms superlatives (e.g. "highest grossing") match the computed data
  - **Sign Verification** — validates directional claims (positive vs. negative correlations)
  - **Value Verification** — the LLM semantic tier confirms specific figures trace to Python stdout (rounding/context-aware); the legacy value regex remains as an advisory, non-blocking signal
- **Circuit breakers:** Hard-coded limits on revision loop iterations and API timeout handling to prevent runaway graphs.
- **Sandboxed execution:** All Python runs inside E2B; the host environment never executes user-derived code directly.