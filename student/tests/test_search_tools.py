"""Unit tests for search_tools.py using standard unittest."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.search_tools import (
    find_references,
    search_code,
    search_function_or_class_definition_in_code,
)


class TestSearchTools(unittest.TestCase):
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

    def test_search_code_basic(self):
        file1 = self.temp_dir / "mod1.py"
        file1.write_text("def hello():\n    return 'world'\n")

        file2 = self.temp_dir / "mod2.py"
        file2.write_text("import mod1\nprint(mod1.hello())\n")

        output = search_code("hello", file_pattern="*.py")
        lines = output.strip().splitlines()

        self.assertEqual(len(lines), 2)
        self.assertIn(f"{file1}:1 def hello():", lines)
        self.assertIn(f"{file2}:2 print(mod1.hello())", lines)

    def test_search_code_file_pattern_filter(self):
        py_file = self.temp_dir / "script.py"
        py_file.write_text("target_word = 123\n")

        txt_file = self.temp_dir / "notes.txt"
        txt_file.write_text("target_word in notes\n")

        output = search_code("target_word", file_pattern="*.py")
        lines = output.strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn(str(py_file), lines[0])

    def _make_nested_module(self):
        pkg = self.temp_dir / "pkg" / "sub"
        pkg.mkdir(parents=True)
        target = pkg / "vector.py"
        target.write_text("class Vector:\n    def __add__(self, o):\n")
        (self.temp_dir / "other.py").write_text("def __add__(self, o):\n")
        return target

    def test_search_code_accepts_file_paths(self):
        target = self._make_nested_module()
        expected = f"{target}:2     def __add__(self, o):"
        for file_pattern in (
            "/testbed/pkg/sub/vector.py",  # path as seen in the container
            str(target),  # absolute host path
            "pkg/sub/vector.py",  # relative path
            "./pkg/sub/vector.py",
        ):
            with self.subTest(file_pattern=file_pattern):
                output = search_code(r"def __add__", file_pattern)
                self.assertEqual(output, expected)

    def test_search_code_accepts_directories(self):
        target = self._make_nested_module()
        for file_pattern in ("pkg/sub", "/testbed/pkg/", "/testbed"):
            with self.subTest(file_pattern=file_pattern):
                output = search_code(r"def __add__", file_pattern)
                self.assertIn(str(target), output)
        self.assertNotIn("other.py", search_code(r"def __add__", "pkg"))

    def test_search_code_explains_empty_results(self):
        self._make_nested_module()
        no_file = search_code("Vector", "/testbed/missing.py")
        self.assertTrue(no_file.startswith("No file matches file_pattern"))
        no_line = search_code("zzz_not_there", "pkg/sub/vector.py")
        self.assertTrue(no_line.startswith("No match for pattern"))
        self.assertIn("1 file(s)", no_line)

    def test_search_code_handles_invalid_regex(self):
        file1 = self.temp_dir / "test.py"
        file1.write_text("result = [foo(x) for x in list]\n")

        output = search_code("[foo(x)", file_pattern="*.py")
        self.assertIn(f"{file1}:1 result = [foo(x) for x in list]", output)

    def test_search_definitions(self):
        code = (
            "# Some comment\n"
            "class Animal:\n"
            "    def __init__(self):\n"
            "        pass\n"
            "    async def speak(self):\n"
            "        return 'woof'\n"
            "\n"
            "def helper():\n"
            "    pass\n"
        )
        src_file = self.temp_dir / "zoo.py"
        src_file.write_text(code)

        class_def = search_function_or_class_definition_in_code("Animal")
        self.assertIn(f"{src_file}:2 class Animal:", class_def)

        async_def = search_function_or_class_definition_in_code("speak")
        self.assertIn(f"{src_file}:5     async def speak(self):", async_def)

        func_def = search_function_or_class_definition_in_code("helper")
        self.assertIn(f"{src_file}:8 def helper():", func_def)

    def test_find_references(self):
        file_a = self.temp_dir / "a.py"
        file_a.write_text("class Calculator:\n    pass\n")

        file_b = self.temp_dir / "b.py"
        file_b.write_text("from a import Calculator\ncalc = Calculator()\n")

        output = find_references("Calculator")
        lines = output.strip().splitlines()

        self.assertEqual(len(lines), 3)
        self.assertTrue(any(str(file_a) in line for line in lines))
        self.assertTrue(any(str(file_b) in line for line in lines))

    def test_empty_inputs(self):
        self.assertEqual(search_code(""), "")
        self.assertEqual(search_function_or_class_definition_in_code(""), "")
        self.assertEqual(find_references(""), "")


if __name__ == "__main__":
    unittest.main()
