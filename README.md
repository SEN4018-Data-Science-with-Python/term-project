Read the Medium article about our project here: https://medium.com/@emir.genc/an-agentic-system-for-autonomous-data-journalism-with-entertainment-industry-datasets-8e7da6505f29

# Autonomous Data Journalism Agent

## Project Overview
Our project is going to be a multi-agent AI system that is designed to ingest raw datasets related to the different sectors of entertainment industry, perform autonomous EDA (exploratory data analysis) by using Python tool execution and create news articles without any manual intervention. 

The system will rely on a strictly sandboxed Python environment for all mathematical and statistical claims to prevent any arithmetic hallucination from LLMs.

## Technical Architecture & Frameworks

Our pipeline moves away from single-prompt inference and utilizes a multi-step agentic graph.

*   **Orchestration:** `LangGraph` (preferred for strict state management and cyclic loop control). 
*   **Execution Environment:** `E2B Data Analysis Sandbox`. The agent writes and executes `pandas`, `numpy`, and `scipy` scripts here.
*   **Core LLMs:** 
    *   *Analyst & Drafter Agents:* Gemini as the primary provider, with Qwen as the fallback provider for advanced code generation and narrative drafting.
    *   *Evaluator Agent:* A deterministic parser + LLM combo to verify stdout logs against narrative claims.
*   **Deployment & UI:** `Hugging Face Spaces`, running a `Gradio` frontend for users to upload CSVs and view the live agentic process and final article.

## Example datasets

1.  **Spotify Most Listened Songs: 2023 and 2024** 
    *   *Source:* Kaggle.
    *   *Target Variables:* Acousticness, valence, BPM, danceability vs. charting duration.
2.  **Movie Industry: Four decades of movies**
    *   *Source:* Kaggle.
    *   There are 6820 movies in the dataset (220 movies per year, 1986-2016). Three decades of movie data, scraped from IMDb using Python.
    *   *Target Variables:* Production budget, global revenue, genre, and release timing to analyze the changing ROI of mid-budget cinema.
3.  **Full TMDB Movies Dataset 2024 (1M Movies)**
    *   *Source:* Kaggle.
    *   The TMDb (The Movie Database) is a comprehensive movie database that provides information about movies, including details like titles, ratings, release dates, revenue, genres, and much more.

## The Autonomous Execution Loop (Technical Workflow)

1.  **Ingestion & Schema Mapping:** The agent reads the `.csv` headers, dtypes, and basic descriptive statistics to understand the data landscape.
2.  **EDA:** The Analyst Agent generates Python scripts to test correlations, clusters, or time-series trends. It runs the code and reads the output.
3.  **Drafting:** The Journalist Agent compiles the statistically significant output findings into a coherent news narrative.
4.  **Evaluation Framework (The Hard Check):**
    *   *Rank, Sign, and Value Verification:* A programmatic script extracts numerical claims from the text and cross-references them against the exact output dictionary. 
5.  **Circuit-Broken Self-Correction:** If the Evaluator finds a discrepancy, it routes the error back to the Analyst.

## Team Responsibilities

*   **Demir Eroğlu 2200678 - Data & UI**
    *   Sourcing, cleaning, and merging the complex Kaggle datasets.
    *   Implementing the sandboxed Python environment and ensuring it securely returns and error tracebacks to the LLM.
    *   Building the Hugging Face Spaces `Gradio` interface to visualize the agent's real-time terminal outputs and final article.

*   **Emir İsmail Genç 2202780 - Agent Orchestration & QA**
    *   Building the `LangGraph` state machine and defining the prompt personas.
    *   Developing the Evaluation Framework (building the regex/deterministic parser that checks the LLM's text against the Python output).
    *   Implementing the revision loop logic and the hard-coded circuit breakers to handle API timeouts and infinite loops.
