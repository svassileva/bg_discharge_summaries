import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


class ConfigurableScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def run_script(self, name: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SRC_DIR / name), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_abbreviation_combiner_uses_configured_paths(self):
        input_dir = self.root / "tables"
        input_dir.mkdir()
        (input_dir / "terms.txt").write_text(
            "| English Abbreviation | English Expansion | Bulgarian Abbreviation | Bulgarian Expansion |\n"
            "| --- | --- | --- | --- |\n"
            "| BP | Blood pressure | АН | Артериално налягане |\n",
            encoding="utf-8",
        )
        output_path = self.root / "nested" / "abbreviations.tsv"

        result = self.run_script(
            "combine_abbreviation_tables.py",
            "--input-dir", str(input_dir),
            "--output-tsv", str(output_path),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        with output_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream, delimiter="\t"))
        self.assertEqual(rows[1], [
            "terms.txt", "BP", "Blood pressure", "АН", "Артериално налягане"
        ])

    def test_terms_combiner_uses_configured_paths(self):
        input_dir = self.root / "tables"
        input_dir.mkdir()
        (input_dir / "terms.txt").write_text(
            "| English Term | Bulgarian Term | Justification |\n"
            "| --- | --- | --- |\n"
            "| hypertension | хипертония | clinical term |\n",
            encoding="utf-8",
        )
        output_path = self.root / "terms.tsv"

        result = self.run_script(
            "combine_terms_tables.py",
            "--input-dir", str(input_dir),
            "--output-tsv", str(output_path),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        with output_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream, delimiter="\t"))
        self.assertEqual(rows[1], [
            "terms.txt", "hypertension", "хипертония", "clinical term"
        ])

    def test_gliner_combiner_reads_configured_directory_and_writes_output(self):
        model_dir = self.root / "annotations" / "gliner_test"
        model_dir.mkdir(parents=True)
        (model_dir / "predictions_en_ner.tsv").write_text(
            "filename\tentity_label\tstart_span\tend_span\ttext\n"
            "case_1.txt\tdisease\t0\t12\thypertension\n",
            encoding="utf-8",
        )
        output_path = self.root / "output" / "combined.tsv"

        result = self.run_script(
            "combine_gliner_annotations.py",
            "--annotations-dir", str(model_dir.parent),
            "--output-tsv", str(output_path),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        with output_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["document_id"], "case_1")
        self.assertEqual(rows[0]["language"], "en")
        self.assertEqual(rows[0]["entity_type"], "DISEASE")

    def test_gliner_combiner_reports_empty_input(self):
        annotations_dir = self.root / "annotations"
        annotations_dir.mkdir()
        result = self.run_script(
            "combine_gliner_annotations.py",
            "--annotations-dir", str(annotations_dir),
            "--output-tsv", str(self.root / "combined.tsv"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No gliner_* folders found", result.stderr)

    def test_sentence_pair_stats_uses_configured_input_directory(self):
        input_dir = self.root / "pairs"
        input_dir.mkdir()
        (input_dir / "one.tsv").write_text(
            "source\thypothesis\n"
            "high blood pressure\tвисоко кръвно налягане\n",
            encoding="utf-8",
        )

        result = self.run_script(
            "sentence_pair_stats.py", "--input-dir", str(input_dir)
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r"Total sentence pairs\s*:\s*1")


if __name__ == "__main__":
    unittest.main()
