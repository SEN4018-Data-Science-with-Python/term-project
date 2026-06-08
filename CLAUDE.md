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
| LLMs | OpenAI / Anthropic / Google models (Analyst, Drafter, Evaluator agents) |
| UI | Gradio on Hugging Face Spaces |

## Agent Architecture

The pipeline is a **LangGraph state machine** with four distinct roles:

1. **Ingestion Agent** — reads CSV headers, dtypes, and descriptive stats to map the data schema. For large datasets, only this schema is passed to the LLM (not raw rows); the agent then generates targeted pandas scripts to answer specific questions.
2. **Analyst Agent** — generates and executes Python scripts inside E2B sandbox; reads stdout/stderr back into the graph state.
3. **Journalist Agent** — drafts a news article from the statistically significant findings.
4. **Evaluator Agent** — deterministic parser + LLM combo that extracts numerical claims from the article and cross-references them against the exact Python output. On mismatch, routes back to the Analyst (circuit-broken to prevent infinite loops).

## Key Design Constraints

- **No arithmetic hallucination:** Every number in the final article must trace to a Python stdout line from E2B. The Evaluator enforces three named tiers:
  - **Rank Verification** — confirms superlatives (e.g. "highest grossing") match the computed data
  - **Sign Verification** — validates directional claims (positive vs. negative correlations)
  - **Value Verification** — ensures specific figures exactly match Python stdout
- **Circuit breakers:** Hard-coded limits on revision loop iterations and API timeout handling to prevent runaway graphs.
- **Sandboxed execution:** All Python runs inside E2B; the host environment never executes user-derived code directly.

## Team

- **Demir Eroğlu (2200678)** — datasets, E2B sandbox integration, Gradio UI
- **Emir İsmail Genç (2202780)** — LangGraph state machine, prompt personas, Evaluator regex parser, revision loop logic
