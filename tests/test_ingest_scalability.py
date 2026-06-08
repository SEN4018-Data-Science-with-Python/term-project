import unittest
import pathlib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
INGEST_PATH = PROJECT_ROOT / "src" / "agents" / "ingest.py"


class IngestScalabilityTests(unittest.TestCase):
    def test_ingest_uses_sampled_schema_instead_of_full_dataframe_describe(self):
        source = INGEST_PATH.read_text()

        self.assertIn("SAMPLE_ROWS", source)
        self.assertIn("nrows=SAMPLE_ROWS", source)
        self.assertIn("sample_describe", source)
        self.assertNotIn("df.describe(include=\"all\")", source)

    def test_ingest_accepts_preuploaded_e2b_dataset_path(self):
        source = INGEST_PATH.read_text()

        self.assertIn('startswith("/home/user/")', source)
        self.assertIn("set_dataset_path", source)

    def test_ingest_detects_csv_encoding_for_latin1_datasets(self):
        source = INGEST_PATH.read_text()

        self.assertIn("CSV_ENCODINGS", source)
        self.assertIn("iso-8859-1", source)
        self.assertIn('"encoding": encoding', source)
        self.assertIn('"read_csv_kwargs"', source)


if __name__ == "__main__":
    unittest.main()
