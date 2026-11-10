"""Tests for the code extractor across various model output formats."""

import unittest

from agent.extractor import extract_code


class TestCodeExtractor(unittest.TestCase):
    def test_standard_python_block(self):
        text = "Thought: I need to calculate.\n```python\nprint(1 + 1)\n```"
        code, warning = extract_code(text)
        self.assertEqual(code, "print(1 + 1)")
        self.assertIsNone(warning)

    def test_multiple_python_blocks(self):
        text = "```python\nx = 10\n```\nSome thought\n```python\nprint(x * 2)\n```"
        code, warning = extract_code(text)
        self.assertEqual(code, "x = 10\nprint(x * 2)")
        self.assertIsNone(warning)

    def test_end_code_token(self):
        text = "```python\nresult = 42\nprint(result)\n```<end_code>"
        code, warning = extract_code(text)
        self.assertEqual(code, "result = 42\nprint(result)")
        self.assertIsNone(warning)

    def test_alternative_python_labels(self):
        for label in ("py", "Python", "PYTHON", ""):
            text = f"```{label}\nprint('hello')\n```"
            code, warning = extract_code(text)
            self.assertEqual(code, "print('hello')")
            self.assertIsNotNone(warning)
            self.assertIn("Warning: your code block was not tagged ```python", warning)

    def test_unclosed_python_block(self):
        text = "Thought: lets run\n```python\nprint('never closed')"
        code, warning = extract_code(text)
        self.assertEqual(code, "print('never closed')")
        self.assertIsNotNone(warning)
        self.assertIn("Warning: your code block had no closing ```", warning)

    def test_hermes_json_tool_call(self):
        text = (
            "I will read the file.\n"
            "<tool_call>\n"
            '{"name": "read_file", "arguments": {"filepath": "/testbed/file.py", "start_line": 1, "end_line": 10}}\n'
            "</tool_call>"
        )
        code, warning = extract_code(text)
        self.assertIsNotNone(code)
        self.assertIn("read_file(filepath='/testbed/file.py', start_line=1, end_line=10)", code)
        self.assertIsNotNone(warning)
        self.assertIn("JSON/Hermes", warning)

    def test_xml_invoke_tool_call(self):
        text = (
            "Calling tool in XML:\n"
            '<invoke name="edit_file">\n'
            '<parameter name="filepath">/testbed/src.py</parameter>\n'
            '<parameter name="old_str">foo</parameter>\n'
            '<parameter name="new_str">bar</parameter>\n'
            "</invoke>"
        )
        code, warning = extract_code(text)
        self.assertIsNotNone(code)
        self.assertIn("edit_file(filepath='/testbed/src.py', old_str='foo', new_str='bar')", code)
        self.assertIsNotNone(warning)
        self.assertIn("XML <invoke>", warning)

    def test_react_tool_call(self):
        text = (
            "Thought: I should search for the definition.\n"
            "Action: search_code\n"
            'Action Input: {"pattern": "def my_func", "file_pattern": "*.py"}\n'
            "Observation: will follow"
        )
        code, warning = extract_code(text)
        self.assertIsNotNone(code)
        self.assertIn("search_code(pattern='def my_func', file_pattern='*.py')", code)
        self.assertIsNotNone(warning)
        self.assertIn("ReAct format", warning)

    def test_liquid_tool_call(self):
        text = "<|tool_call_start|>[code(code='def solve():\\n    return 42\\n')]<|tool_call_end|>"
        code, warning = extract_code(text)
        self.assertEqual(code, "def solve():\n    return 42")
        self.assertIsNotNone(warning)
        self.assertIn("Liquid", warning)

    def test_liquid_direct_mcp_calls(self):
        text = "<|tool_call_start|>[read_file(filepath='/testbed/file.py'), search_code(pattern='foo')]<|tool_call_end|>"
        code, warning = extract_code(text)
        self.assertIn("print(read_file(filepath='/testbed/file.py'))", code)
        self.assertIn("print(search_code(pattern='foo'))", code)
        self.assertIsNotNone(warning)

    def test_hermes_json_tool_name_and_string_args(self):
        text = (
            "<tool_call>\n"
            '{"tool_call_id": "0", "tool_name": "search_code", "arguments": "{\\"pattern\\": \\"vector add 0 error\\"}"}\n'
            "</tool_call>"
        )
        code, warning = extract_code(text)
        self.assertIsNotNone(code)
        self.assertIn("search_code(pattern='vector add 0 error')", code)

    def test_final_answer_tool_call(self):
        text = '<tool_call>{"name": "final_answer", "arguments": {"answer": "def foo(): return 1"}}</tool_call>'
        code, warning = extract_code(text)
        self.assertEqual(code, "final_answer('def foo(): return 1')")

    def test_no_code_raises_value_error(self):
        text = "Hello, I cannot solve this task without more details."
        with self.assertRaises(ValueError):
            extract_code(text)


if __name__ == "__main__":
    unittest.main()

