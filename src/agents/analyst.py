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
