from src.agents.evaluator import (
    verify_values,
    verify_signs,
    verify_ranks,
    verify_semantic,
    parse_llm_verdict,
    select_blocking_errors,
    evaluator_node,
)


class _Response:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return _Response(self._content)


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
    assert errors


def test_rank_top_match():
    article = "Taylor Swift is the most streamed artist."
    stdout = "[('Taylor Swift', 100), ('Drake', 80), ('Bad Bunny', 60)]"
    assert verify_ranks(article, [stdout]) == []


def test_rank_top_mismatch():
    article = "Drake is the most streamed artist."
    stdout = "[('Taylor Swift', 100), ('Drake', 80), ('Bad Bunny', 60)]"
    errors = verify_ranks(article, [stdout])
    assert errors


# --- LLM semantic tier ------------------------------------------------------


def test_parse_llm_verdict_pass():
    assert parse_llm_verdict('{"supported": true}') == []


def test_parse_llm_verdict_extracts_issues():
    text = '{"supported": false, "issues": ["claims a rise with no data", "wrong sign"]}'
    assert parse_llm_verdict(text) == [
        "claims a rise with no data",
        "wrong sign",
    ]


def test_parse_llm_verdict_tolerates_code_fence_and_prose():
    text = 'Here is my verdict:\n```json\n{"supported": false, "issues": ["x"]}\n```'
    assert parse_llm_verdict(text) == ["x"]


def test_parse_llm_verdict_fails_open_on_garbage():
    assert parse_llm_verdict("not json at all") == []


def test_verify_semantic_uses_injected_llm():
    llm = _FakeLLM('{"supported": false, "issues": ["unsupported upward trend"]}')
    errors = verify_semantic("Revenue rose steadily.", ["mean = 5"], llm=llm)
    assert errors == ["unsupported upward trend"]
    assert llm.calls  # the LLM was actually consulted


def test_verify_semantic_skips_llm_when_no_stdout():
    llm = _FakeLLM('{"supported": false, "issues": ["should not run"]}')
    assert verify_semantic("Some article.", [""], llm=llm) == []
    assert not llm.calls


# --- combination policy (your contribution) ---------------------------------


def test_select_blocking_errors_passes_when_all_clean():
    assert select_blocking_errors([], [], [], []) == []


def test_select_blocking_errors_blocks_on_sign_rank_and_semantic():
    result = select_blocking_errors(
        value_errors=[],
        sign_errors=["sign wrong"],
        rank_errors=["rank wrong"],
        semantic_errors=["trend unsupported"],
    )
    assert "sign wrong" in result
    assert "rank wrong" in result
    assert "trend unsupported" in result


# --- groundedness gate: "cannot verify" must FAIL, not pass -----------------


def test_evaluator_fails_when_stdout_empty_even_if_article_looks_complete():
    # Reproduces the silent hallucination: analyst printed nothing, journalist
    # invented a polished article. The evaluator must NOT finalize it.
    state = {
        "analysis_results": [{"script": "x", "stdout": "", "stderr": "Killed"}],
        "article_draft": "Revenue hit $14,944,244 in 2019 and corr was 0.75.",
        "revision_count": 0,
    }
    result = evaluator_node(state)  # empty stdout -> LLM tier is never reached
    assert "final_article" not in result
    assert result["evaluation_errors"]
    assert result["revision_count"] == 1


def test_evaluator_groundedness_failure_still_circuit_breaks():
    state = {
        "analysis_results": [{"script": "x", "stdout": "   ", "stderr": "boom"}],
        "article_draft": "Some ungrounded claims with 123 and 456.",
        "revision_count": 2,  # one more failure trips MAX_REVISIONS
    }
    result = evaluator_node(state)
    assert "final_article" in result
    assert "[unverified]" in result["final_article"]
