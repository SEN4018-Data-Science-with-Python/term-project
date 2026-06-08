import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
UI_PATH = PROJECT_ROOT / "ui" / "app.py"


class UiProgressLoggingTests(unittest.TestCase):
    def test_run_pipeline_yields_progress_before_creating_sandbox(self):
        source = UI_PATH.read_text()
        tree = ast.parse(source)
        run_pipeline = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run_pipeline"
        )
        sandbox_line = next(
            node.lineno
            for node in ast.walk(run_pipeline)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Sandbox"
        )
        yield_lines_before_sandbox = [
            node.lineno
            for node in ast.walk(run_pipeline)
            if isinstance(node, ast.Yield)
            and node.lineno < sandbox_line
        ]

        self.assertLess(source.index("Starting pipeline"), source.index("Sandbox()"))
        self.assertTrue(yield_lines_before_sandbox)

    def test_run_pipeline_reports_exceptions_to_the_log_output(self):
        source = UI_PATH.read_text()

        self.assertIn("except Exception as exc:", source)
        self.assertIn("Pipeline failed", source)
        self.assertIn("type(exc).__name__", source)

    def test_run_pipeline_logs_uploaded_csv_size(self):
        source = UI_PATH.read_text()

        self.assertIn("os.path.getsize", source)
        self.assertIn("CSV size", source)

    def test_run_pipeline_uploads_csv_before_running_graph(self):
        source = UI_PATH.read_text()

        self.assertIn("Uploading CSV to E2B", source)
        self.assertIn("Upload complete", source)
        self.assertLess(source.index("sandbox.upload"), source.index("graph.stream"))

if __name__ == "__main__":
    unittest.main()
