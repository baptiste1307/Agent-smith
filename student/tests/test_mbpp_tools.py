import json
import os
import unittest

from tools.mbpp_tools import MAX_OUTPUT_CHARS, run_tests_mbpp

GOOD = "def add(a, b):\n    return a + b\n"
TESTS = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]


def run(code, tests=TESTS, **kwargs):
    answer = json.loads(run_tests_mbpp(code, tests, **kwargs))
    assert set(answer) == {"success", "output"}
    assert isinstance(answer["success"], bool) and isinstance(answer["output"], str)
    return answer


class TestRunTestsMbpp(unittest.TestCase):
    def test_all_pass(self):
        answer = run(GOOD)
        self.assertTrue(answer["success"])
        self.assertIn("All 3 assertions passed", answer["output"])

    def test_failure_shows_real_value(self):
        answer = run("def add(a, b):\n    return a - b\n")
        self.assertFalse(answer["success"])
        self.assertIn("got -1, expected 3", answer["output"])

    def test_syntax_error_gives_line(self):
        answer = run("def add(a, b)\n    return a + b\n")
        self.assertFalse(answer["success"])
        self.assertIn("SyntaxError", answer["output"])
        self.assertIn("line 1", answer["output"])

    def test_error_inside_function(self):
        answer = run("def add(a, b):\n    return a + b + zzz\n")
        self.assertFalse(answer["success"])
        self.assertIn("NameError", answer["output"])

    def test_infinite_loop_times_out(self):
        answer = run("def add(a, b):\n    while True:\n        pass\n", timeout=1.5)
        self.assertFalse(answer["success"])
        self.assertIn("Timeout", answer["output"])

    def test_exit_is_never_a_success(self):
        for code in ("import sys\nsys.exit(0)\n" + GOOD,
                     "import os\nos._exit(0)\n" + GOOD,
                     "import os\ndef add(a, b):\n    os._exit(0)\n"):
            self.assertFalse(run(code)["success"], code)

    def test_input_does_not_hang(self):
        answer = run("def add(a, b):\n    input()\n    return a + b\n", timeout=5)
        self.assertFalse(answer["success"])
        self.assertNotIn("Timeout", answer["output"])

    def test_huge_output_is_capped(self):
        answer = run("def add(a, b):\n    print('x' * 10**6)\n    return a + b\n")
        self.assertLessEqual(len(answer["output"]), MAX_OUTPUT_CHARS)
        endless = run("def add(a, b):\n    while True:\n        print('x' * 1000)\n", timeout=8)
        self.assertFalse(endless["success"])
        self.assertLessEqual(len(endless["output"]), MAX_OUTPUT_CHARS)

    def test_memory_bomb_is_stopped(self):
        answer = run("def add(a, b):\n    return len(bytearray(4 * 1024**3))\n", timeout=8)
        self.assertFalse(answer["success"])

    def test_main_guard_is_not_executed(self):
        self.assertTrue(run(GOOD + "\nif __name__ == '__main__':\n    raise RuntimeError('demo')\n")["success"])

    def test_test_imports(self):
        code = "def area(r):\n    return math.pi * r * r\n"
        tests = ["assert math.isclose(area(1), math.pi)"]
        self.assertTrue(run(code, tests, test_imports=["import math"])["success"])
        self.assertFalse(run(code, tests)["success"])

    def test_api_keys_are_hidden_from_the_solution(self):
        os.environ["OPENROUTER_API_KEY"] = "secret"
        try:
            code = "import os\ndef add(a, b):\n    return len(os.environ.get('OPENROUTER_API_KEY', ''))\n"
            self.assertTrue(run(code, ["assert add(1, 2) == 0"])["success"])
        finally:
            del os.environ["OPENROUTER_API_KEY"]

    def test_bad_inputs_never_raise(self):
        for code, tests in (("", TESTS), (None, TESTS), (GOOD, None), (GOOD, []), (GOOD, [1, 2]), (123, "x")):
            self.assertFalse(json.loads(run_tests_mbpp(code, tests))["success"], (code, tests))
        self.assertTrue(run(GOOD, "assert add(1, 2) == 3")["success"])

    def test_many_failures_are_summarized(self):
        answer = run(GOOD, [f"assert add({i}, 0) == -1" for i in range(12)])
        self.assertIn("more", answer["output"])
        self.assertLessEqual(len(answer["output"]), MAX_OUTPUT_CHARS)


if __name__ == "__main__":
    unittest.main()