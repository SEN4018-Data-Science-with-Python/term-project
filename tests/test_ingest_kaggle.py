import json
import unittest

from src.agents import ingest
from src.agents.ingest import ingest_node


def _schema_json(files):
    """Build the JSON the in-sandbox INGEST_CODE would print for `files`,
    a list of (path, size_bytes) tuples."""
    return json.dumps(
        {
            "files": [
                {
                    "path": path,
                    "name": path.rsplit("/", 1)[-1],
                    "size_bytes": size,
                    "columns": ["a"],
                    "dtypes": {"a": "int64"},
                    "n_rows": 1,
                    "describe": {},
                    "encoding": "utf-8",
                    "read_csv_kwargs": {"encoding": "utf-8", "low_memory": False},
                    "schema_sample_rows": 1,
                }
                for path, size in files
            ],
            "ingest_mode": "sampled_schema",
        }
    )


class FakeSandbox:
    """Records calls so we can assert the ingest dispatch without a real VM."""

    def __init__(self, csv_files=None, ingest_stdout="{}"):
        self._csv_files = csv_files or []
        self._ingest_stdout = ingest_stdout
        self.downloaded_ref = None
        self.set_path = None
        self.uploaded = None
        self.ran_code = None

    def download_kaggle(self, dataset_ref):
        self.downloaded_ref = dataset_ref
        return self._csv_files

    def set_dataset_path(self, remote_path):
        self.set_path = remote_path
        return remote_path

    def upload(self, local_path, remote_name="data.csv"):
        self.uploaded = local_path
        return f"/home/user/{remote_name}"

    def run(self, code):
        self.ran_code = code
        return {"stdout": self._ingest_stdout, "stderr": ""}


class PrimaryPathTests(unittest.TestCase):
    def test_primary_path_is_largest_file(self):
        files = [
            {"path": "/a.csv", "size_bytes": 10},
            {"path": "/b.csv", "size_bytes": 999},
            {"path": "/c.csv", "size_bytes": 50},
        ]
        self.assertEqual(ingest._primary_path(files), "/b.csv")


class IngestKaggleDispatchTests(unittest.TestCase):
    def test_kaggle_prefix_downloads_and_profiles_all_csvs(self):
        csv_files = [
            {"path": "/root/lookup.csv", "size_bytes": 1_000},
            {"path": "/root/main.csv", "size_bytes": 50_000_000},
        ]
        stdout = _schema_json(
            [("/root/lookup.csv", 1_000), ("/root/main.csv", 50_000_000)]
        )
        sandbox = FakeSandbox(csv_files=csv_files, ingest_stdout=stdout)

        result = ingest_node({"dataset_path": "kaggle:owner/some-dataset"}, sandbox)

        self.assertEqual(sandbox.downloaded_ref, "owner/some-dataset")
        # Every CSV is profiled, not just one.
        self.assertIn("/root/lookup.csv", sandbox.ran_code)
        self.assertIn("/root/main.csv", sandbox.ran_code)
        self.assertEqual(len(result["schema"]["files"]), 2)
        # The largest table becomes the sandbox's primary handle.
        self.assertEqual(sandbox.set_path, "/root/main.csv")
        self.assertEqual(result["schema"]["primary_path"], "/root/main.csv")
        self.assertIsNone(sandbox.uploaded)
        # The plan mentions both tables so the analyst knows to use them.
        plan_text = " ".join(result["analysis_plan"])
        self.assertIn("lookup.csv", plan_text)
        self.assertIn("main.csv", plan_text)

    def test_local_path_uploads_and_profiles_single_file(self):
        stdout = _schema_json([("/home/user/data.csv", 500)])
        sandbox = FakeSandbox(ingest_stdout=stdout)

        result = ingest_node({"dataset_path": "/tmp/movies.csv"}, sandbox)

        self.assertEqual(sandbox.uploaded, "/tmp/movies.csv")
        self.assertIsNone(sandbox.downloaded_ref)
        self.assertEqual(len(result["schema"]["files"]), 1)
        self.assertEqual(result["schema"]["primary_path"], "/home/user/data.csv")

    def test_preuploaded_path_is_profiled_without_reupload(self):
        stdout = _schema_json([("/home/user/data.csv", 500)])
        sandbox = FakeSandbox(ingest_stdout=stdout)

        ingest_node({"dataset_path": "/home/user/data.csv"}, sandbox)

        self.assertIsNone(sandbox.uploaded)
        self.assertIsNone(sandbox.downloaded_ref)
        self.assertIn("/home/user/data.csv", sandbox.ran_code)


if __name__ == "__main__":
    unittest.main()
