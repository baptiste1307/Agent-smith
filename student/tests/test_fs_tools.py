"""Unit tests for fs_tools.py."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.fs_tools import edit_file, list_files, read_file


class TestFSTools(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()
        self.old_testbed = os.environ.get("TESTBED_PATH")
        os.environ["TESTBED_PATH"] = str(self.temp_dir)

    def tearDown(self):
        if self.old_testbed is not None:
            os.environ["TESTBED_PATH"] = self.old_testbed
        else:
            os.environ.pop("TESTBED_PATH", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_file_range(self):
        test_file = self.temp_dir / "sample.py"
        test_file.write_text("line one\nline two\nline three\nline four\n")

        output = read_file(test_file, start_line=2, end_line=3)
        expected = "2: line two\n3: line three"
        self.assertEqual(output, expected)

    def test_read_file_clamping(self):
        test_file = self.temp_dir / "short.txt"
        test_file.write_text("alpha\nbeta\n")

        # end_line greater than total lines should clamp to end
        output = read_file(test_file, start_line=1, end_line=50)
        self.assertEqual(output, "1: alpha\n2: beta")

    def test_read_file_not_found(self):
        output = read_file(self.temp_dir / "missing.txt", 1, 5)
        self.assertTrue(output.startswith("Error:"))

    def test_edit_file_success(self):
        test_file = self.temp_dir / "target.py"
        test_file.write_text("x = 1\ny = 2\nz = 3\n")

        res = edit_file(test_file, old_str="y = 2", new_str="y = 42")
        self.assertIn("updated successfully", res)
        self.assertEqual(test_file.read_text(), "x = 1\ny = 42\nz = 3\n")

    def test_edit_file_not_unique(self):
        test_file = self.temp_dir / "duplicate.py"
        test_file.write_text("repeat\nrepeat\n")

        res = edit_file(test_file, old_str="repeat", new_str="single")
        self.assertIn("Error:", res)
        self.assertIn("found 2 times", res)

    def test_list_files(self):
        (self.temp_dir / "a.py").write_text("# a")
        (self.temp_dir / "b.txt").write_text("# b")
        sub = self.temp_dir / "subdir"
        sub.mkdir()
        (sub / "c.py").write_text("# c")

        output = list_files(self.temp_dir, pattern="*.py")
        lines = output.strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(any("a.py" in line for line in lines))
        self.assertTrue(any("c.py" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
