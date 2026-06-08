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
