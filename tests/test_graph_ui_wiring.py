import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
GRAPH_PATH = PROJECT_ROOT / "src" / "graph.py"
UI_PATH = PROJECT_ROOT / "ui" / "app.py"


class GraphUiWiringTests(unittest.TestCase):
    def test_graph_exposes_reusable_initial_state_helper(self):
        tree = ast.parse(GRAPH_PATH.read_text())
        functions = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("initial_state", functions)

    def test_ui_uses_shared_graph_builder_instead_of_rebuilding_graph(self):
        source = UI_PATH.read_text()

        self.assertIn("from src.graph import build_graph, initial_state", source)
        self.assertNotIn("StateGraph", source)
        self.assertNotIn("def _build", source)


if __name__ == "__main__":
    unittest.main()
