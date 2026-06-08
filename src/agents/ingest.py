from __future__ import annotations
import json
from src.state import AgentState
from src.sandbox import Sandbox


INGEST_CODE = """
import pandas as pd
import json
SAMPLE_ROWS = 5000
CSV_ENCODINGS = ["utf-8", "utf-8-sig", "iso-8859-1", "cp1252"]
path = "{path}"

sample = None
encoding = None
last_error = None
for candidate in CSV_ENCODINGS:
    try:
        sample = pd.read_csv(
            path,
            nrows=SAMPLE_ROWS,
            low_memory=False,
            encoding=candidate,
        )
        encoding = candidate
        break
    except UnicodeDecodeError as exc:
        last_error = exc

if sample is None:
    raise last_error or RuntimeError("Could not read CSV sample with configured encodings")

columns = sample.columns.tolist()
sample_describe = sample.describe(include="all").fillna("").to_dict()
read_csv_kwargs = {{"encoding": encoding, "low_memory": False}}

n_rows = 0
if columns:
    first_col = columns[0]
    for chunk in pd.read_csv(
        path,
        usecols=[first_col],
        chunksize=100000,
        **read_csv_kwargs,
    ):
        n_rows += len(chunk)

schema = {{
    "columns": columns,
    "dtypes": {{c: str(sample[c].dtype) for c in columns}},
    "n_rows": n_rows,
    "describe": sample_describe,
    "encoding": encoding,
    "read_csv_kwargs": read_csv_kwargs,
    "schema_sample_rows": len(sample),
    "ingest_mode": "sampled_schema",
}}
print(json.dumps(schema, default=str))
"""


def ingest_node(state: AgentState, sandbox: Sandbox) -> dict:
    dataset_path = state["dataset_path"]
    if dataset_path.startswith("/home/user/"):
        remote = sandbox.set_dataset_path(dataset_path)
    else:
        remote = sandbox.upload(dataset_path)
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
