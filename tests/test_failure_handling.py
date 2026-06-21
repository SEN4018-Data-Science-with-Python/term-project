import unittest

from src.agents.journalist import journalist_node
from src.agents.analyst import _build_revision_notes


class JournalistGuardTests(unittest.TestCase):
    def test_journalist_refuses_to_fabricate_when_no_stdout(self):
        state = {
            "analysis_results": [{"script": "x", "stdout": "", "stderr": "Killed"}],
        }
        result = journalist_node(state)  # must not call the LLM / invent prose
        self.assertIn("INSUFFICIENT DATA", result["article_draft"])

    def test_journalist_refuses_when_all_runs_blank(self):
        state = {
            "analysis_results": [
                {"script": "a", "stdout": "   ", "stderr": ""},
                {"script": "b", "stdout": "\n", "stderr": "trace"},
            ],
        }
        self.assertIn("INSUFFICIENT DATA", journalist_node(state)["article_draft"])


class AnalystRevisionNotesTests(unittest.TestCase):
    def test_previous_stderr_is_surfaced_to_the_analyst(self):
        state = {
            "analysis_results": [
                {"script": "x", "stdout": "", "stderr": "MemoryError: ..."}
            ],
            "evaluation_errors": [],
        }
        notes = _build_revision_notes(state)
        self.assertIn("MemoryError", notes)
        self.assertIn("PRINTED NOTHING", notes)

    def test_evaluation_errors_are_included(self):
        state = {
            "analysis_results": [{"script": "x", "stdout": "ok", "stderr": ""}],
            "evaluation_errors": ["Article claims X but stdout shows Y"],
        }
        notes = _build_revision_notes(state)
        self.assertIn("FAILED VERIFICATION", notes)
        self.assertIn("Article claims X", notes)

    def test_clean_first_run_has_no_notes(self):
        state = {"analysis_results": [], "evaluation_errors": []}
        self.assertEqual(_build_revision_notes(state), "")


if __name__ == "__main__":
    unittest.main()
