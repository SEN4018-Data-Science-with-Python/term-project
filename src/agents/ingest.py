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
