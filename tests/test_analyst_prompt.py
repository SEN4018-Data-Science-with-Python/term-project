import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYST_PATH = PROJECT_ROOT / "src" / "agents" / "analyst.py"


class AnalystPromptTests(unittest.TestCase):
    def test_analyst_prompt_includes_read_csv_kwargs_from_schema(self):
        source = ANALYST_PATH.read_text()

        self.assertIn("read_csv_kwargs", source)
        self.assertIn("Read CSV settings", source)


if __name__ == "__main__":
    unittest.main()
