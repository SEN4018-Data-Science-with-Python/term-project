import unittest

from src.sandbox import _parse_kaggle_manifest


class _Err:
    def __init__(self, name, value):
        self.name = name
        self.value = value


MANIFEST = (
    '{"download_dir": "/root/.cache/x", '
    '"csv_files": [{"path": "/root/.cache/x/TMDB_movie_dataset_v11.csv", '
    '"size_bytes": 246000000}]}'
)

# Real tqdm/kagglehub output observed on a *successful* download.
PROGRESS_STDERR = (
    "/usr/local/lib/python3.13/site-packages/tqdm/auto.py:21: TqdmWarning: "
    "IProgress not found. Please update jupyter and ipywidgets.\n"
    "  0%|          | 0.00/246M [00:00<?, ?B/s]\n"
    "100%|██████████| 246M/246M [00:01<00:00, 195MB/s]"
)


class ParseKaggleManifestTests(unittest.TestCase):
    def test_progress_bars_on_stderr_are_not_fatal(self):
        files = _parse_kaggle_manifest(
            stdout=MANIFEST,
            stderr=PROGRESS_STDERR,
            error=None,
            dataset_ref="asaniczka/tmdb-movies-dataset-2023-930k-movies",
        )
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0]["path"].endswith(".csv"))

    def test_real_python_error_raises_with_detail(self):
        with self.assertRaises(RuntimeError) as ctx:
            _parse_kaggle_manifest(
                stdout="",
                stderr="traceback...",
                error=_Err("KaggleApiHTTPError", "403 Forbidden"),
                dataset_ref="owner/private-dataset",
            )
        self.assertIn("403 Forbidden", str(ctx.exception))

    def test_manifest_is_found_even_with_trailing_noise_on_stdout(self):
        stdout = MANIFEST + "\n100%|##########| 246M/246M"
        files = _parse_kaggle_manifest(stdout, "", None, "owner/ds")
        self.assertEqual(len(files), 1)

    def test_empty_csv_list_raises(self):
        with self.assertRaises(RuntimeError):
            _parse_kaggle_manifest(
                stdout='{"download_dir": "/x", "csv_files": []}',
                stderr="",
                error=None,
                dataset_ref="owner/ds",
            )

    def test_missing_manifest_raises(self):
        with self.assertRaises(RuntimeError):
            _parse_kaggle_manifest(
                stdout="no json here", stderr="", error=None, dataset_ref="owner/ds"
            )


if __name__ == "__main__":
    unittest.main()
