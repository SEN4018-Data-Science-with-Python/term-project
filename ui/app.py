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
