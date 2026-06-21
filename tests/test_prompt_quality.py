import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYST = (ROOT / "src" / "agents" / "analyst.py").read_text()
JOURNALIST = (ROOT / "src" / "agents" / "journalist.py").read_text()
EVALUATOR = (ROOT / "src" / "agents" / "evaluator.py").read_text()


class JournalistPromptTests(unittest.TestCase):
    def test_no_hardcoded_streamed_example(self):
        # The "second-most-streamed" example leaked the Spotify framing into
        # movie articles. Rankings must be named by the real metric.
        self.assertNotIn("second-most-streamed", JOURNALIST)
        self.assertIn("highest-grossing", JOURNALIST)

    def test_requires_metric_fidelity(self):
        self.assertIn("most-streamed", JOURNALIST)  # named as a thing to AVOID
        self.assertIn("actually measures", JOURNALIST.lower())

    def test_forbids_ungrounded_attributions(self):
        # The "Billie Eilish" leak: never add names/attributions not in stdout.
        self.assertIn("attributions", JOURNALIST.lower())
        self.assertIn("do not appear in the analysis stdout", JOURNALIST.lower())


class AnalystPromptTests(unittest.TestCase):
    def test_requires_raw_numbers(self):
        self.assertIn("RAW numbers", ANALYST)
        self.assertIn("no currency symbols", ANALYST.lower())

    def test_requires_dirty_data_filtering(self):
        self.assertIn("DATA IS DIRTY", ANALYST)

    def test_rankings_include_identifying_columns(self):
        # Ground attributions: print artist/director alongside the ranked entity.
        self.assertIn("identifying columns", ANALYST)
        self.assertIn("FIRST element", ANALYST)


class EvaluatorPromptTests(unittest.TestCase):
    def test_accepts_rounding_and_formatting(self):
        # Must not re-flag correct rounding (0.73 for 0.7317) or $-formatting.
        self.assertIn("0.7317", EVALUATOR)
        self.assertIn("SAME number", EVALUATOR)

    def test_flags_ungrounded_attributions_but_spares_generic_nouns(self):
        self.assertIn("Attributing a data point", EVALUATOR)
        self.assertIn("descriptive nouns", EVALUATOR)


if __name__ == "__main__":
    unittest.main()
