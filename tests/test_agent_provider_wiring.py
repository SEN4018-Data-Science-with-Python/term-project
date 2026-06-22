import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT_FILES = [
    PROJECT_ROOT / "src" / "agents" / "analyst.py",
    PROJECT_ROOT / "src" / "agents" / "journalist.py",
]


class AgentProviderWiringTests(unittest.TestCase):
    def test_llm_agents_use_provider_factory_instead_of_chatopenai_directly(self):
        for path in AGENT_FILES:
            with self.subTest(path=path.name):
                source = path.read_text()
                tree = ast.parse(source)
                imported_names = {
                    alias.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    for alias in node.names
                }

                self.assertNotIn("ChatOpenAI", imported_names)
                self.assertIn("get_llm", source)


if __name__ == "__main__":
    unittest.main()
