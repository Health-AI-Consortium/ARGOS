"""Unit tests for path resolution and preprocessing. Run: python -m unittest discover tests"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from argosfeddeep.paths import data_dir, resolve_slice_line, resolve_slice_path, work_dir
from argosfeddeep.preprocess import preprocess


class ResolveSlicePath(unittest.TestCase):
    data = Path("/mnt/argos")

    def test_relative_paths_are_rooted_at_the_data_folder(self):
        self.assertEqual(resolve_slice_path("Train/P1/CT/3.nii.gz", self.data), "/mnt/argos/Train/P1/CT/3.nii.gz")

    def test_absolute_paths_are_kept(self):
        self.assertEqual(resolve_slice_path("/elsewhere/x.nii.gz", self.data), "/elsewhere/x.nii.gz")

    def test_line_keeps_the_label(self):
        line = "Train/P1/CT/3.nii.gz,Train/P1/GT/GTV/3_gtv.nii.gz,Train/P1/GT/Lung/3_lung.nii.gz, 1"
        resolved = resolve_slice_line(line, self.data)
        self.assertEqual(resolved.split(",")[-1], " 1")
        self.assertEqual(int(resolved.split(",")[-1]), 1)
        self.assertTrue(all(field.startswith("/mnt/argos/") for field in resolved.split(",")[:3]))


class DataDirFromNode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / "scans"
        self.folder.mkdir()
        self.csv = Path(self.tmp.name) / "rows.csv"
        self.csv.write_text("id\n")
        self.clean = {k: v for k, v in os.environ.items() if not k.endswith(("_LABELS", "_DATABASE_URI"))}

    def tearDown(self):
        self.tmp.cleanup()

    def test_requested_database_wins(self):
        env = {
            "DB_LABELS": "rows,scans",
            "USER_REQUESTED_DATABASE_LABELS": "scans",
            "ROWS_DATABASE_URI": str(self.csv),
            "SCANS_DATABASE_URI": str(self.folder),
        }
        with mock.patch.dict(os.environ, {**self.clean, **env}, clear=True):
            self.assertEqual(data_dir(), self.folder)

    def test_falls_back_to_the_first_folder_database(self):
        env = {"DB_LABELS": "rows,scans", "ROWS_DATABASE_URI": str(self.csv), "SCANS_DATABASE_URI": str(self.folder)}
        with mock.patch.dict(os.environ, {**self.clean, **env}, clear=True):
            self.assertEqual(data_dir(), self.folder)

    def test_no_folder_database_is_an_error_naming_what_was_seen(self):
        env = {"DB_LABELS": "rows", "ROWS_DATABASE_URI": str(self.csv)}
        with mock.patch.dict(os.environ, {**self.clean, **env}, clear=True):
            with self.assertRaises(RuntimeError) as raised:
                data_dir()
        self.assertIn("rows=", str(raised.exception))


class Preprocess(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name) / "data"
        self.work = Path(self.tmp.name) / "work"
        (self.data / "assets").mkdir(parents=True)
        patients = {f"P{i}": [f"Train/P{i}/CT/0.nii.gz,Train/P{i}/GT/GTV/0_gtv.nii.gz,Train/P{i}/GT/Lung/0_lung.nii.gz, 0"] for i in range(4)}
        (self.data / "assets" / "slices_training_800200.json").write_text(json.dumps(patients))
        (self.data / "assets" / "slices_validation_800200.json").write_text(json.dumps(patients))
        (self.data / "assets" / "params_base.json").write_text(json.dumps({"batch_size": 4}))
        self.before = sorted(p.relative_to(self.data) for p in self.data.rglob("*"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_to_the_temporary_folder_and_leaves_the_data_folder_alone(self):
        env = {
            "DB_LABELS": "scans",
            "USER_REQUESTED_DATABASE_LABELS": "scans",
            "SCANS_DATABASE_URI": str(self.data),
            "TEMPORARY_FOLDER": str(self.work),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            returned = preprocess()
        self.assertEqual(returned, self.work)
        modified = json.loads((self.work / "assets" / "slices_training_modified.json").read_text())
        self.assertTrue(modified["P0"][0].startswith(str(self.data)))
        params = json.loads((self.work / "assets" / "params.json").read_text())
        self.assertEqual(params["batch_size"], 4)
        self.assertEqual(sorted(len(params[f"client{i}_train_samples"]) for i in range(3)), [1, 1, 2])
        after = sorted(p.relative_to(self.data) for p in self.data.rglob("*"))
        self.assertEqual(self.before, after)

    def test_work_dir_requires_the_temporary_folder(self):
        with mock.patch.dict(os.environ, {"TEMPORARY_FOLDER": str(self.work)}, clear=False):
            self.assertEqual(work_dir(), self.work)
        clean = {k: v for k, v in os.environ.items() if k != "TEMPORARY_FOLDER"}
        with mock.patch.dict(os.environ, clean, clear=True):
            with self.assertRaises(KeyError):
                work_dir()


if __name__ == "__main__":
    unittest.main()
