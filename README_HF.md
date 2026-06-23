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

# Autonomous Data Journalism Agent

Multi-agent AI system that performs autonomous EDA on entertainment industry CSV datasets and produces fact-verified news articles. Built with LangGraph + E2B + Gemini (Qwen fallback).

## Required Space secrets

Set these under **Settings → Variables and secrets** (all are secrets):

| Secret | Purpose |
|---|---|
| `GEMINI_API_KEY` | Primary LLM (Analyst / Journalist / Evaluator) |
| `QWEN_API_KEY` | Fallback LLM — **required even when Gemini is set**, the fallback is built eagerly |
| `E2B_API_KEY` | Sandboxed Python execution |

Optional: `KAGGLE_USERNAME` + `KAGGLE_KEY` (only needed for the `kaggle:owner/slug`
download path), and overrides `GEMINI_MODEL`, `QWEN_MODEL`, `QWEN_BASE_URL`.
