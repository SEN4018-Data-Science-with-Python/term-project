from src.agents.evaluator import (
    verify_values,
    verify_signs,
    verify_ranks,
)


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
