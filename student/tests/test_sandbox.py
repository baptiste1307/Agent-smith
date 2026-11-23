"""Tests for the sandbox security constraints and execution semantics."""

import os
import tempfile
import unittest
from pathlib import Path

from models_public import SandboxConfig
from sandbox.sandbox import execute, load_config
from sandbox.security import is_import_allowed, is_path_allowed, NetworkBlocker


class TestSandboxSecurity(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.allowed_dir = Path(self.tmp_dir.name).resolve()
        self.config = SandboxConfig(
            authorized_imports=["math", "collections.*", "itertools", "json"],
            allowed_directories=[str(self.allowed_dir)],
            max_execution_time_seconds=2,
            max_memory_mb=128,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_authorized_imports(self):
        # Exact match
        self.assertTrue(is_import_allowed("math", self.config.authorized_imports))
        self.assertTrue(is_import_allowed("json", self.config.authorized_imports))
        # Wildcard match
        self.assertTrue(is_import_allowed("collections", self.config.authorized_imports))
        self.assertTrue(is_import_allowed("collections.abc", self.config.authorized_imports))
        # Unauthorized
        self.assertFalse(is_import_allowed("os", self.config.authorized_imports))
        self.assertFalse(is_import_allowed("sys", self.config.authorized_imports))
        self.assertFalse(is_import_allowed("subprocess", self.config.authorized_imports))

    def test_execution_import_blocked(self):
        code = "import os\nprint(os.getcwd())"
        result = execute(code, self.config)
        self.assertIsNotNone(result["error"])
        self.assertIn("Import of module 'os' is not authorized", result["error"])

    def test_execution_import_allowed(self):
        code = "import math\nfrom collections import Counter\nprint(math.sqrt(16), Counter('aab')['a'])"
        result = execute(code, self.config)
        self.assertIsNone(result["error"])
        self.assertIn("4.0 2", result["output"])

    def test_forbidden_builtins(self):
        # eval, exec, compile, globals should not be accessible
        for func in ("eval('1 + 1')", "exec('x = 1')", "globals()", "compile('x=1', '', 'exec')"):
            result = execute(func, self.config)
            self.assertIsNotNone(result["error"])
            self.assertIn("NameError", result["error"])

    def test_dunder_escape_blocked(self):
        # Introspection attacks attempting to reach os or builtins via dunder attributes
        exploits = (
            "().__class__.__base__.__subclasses__()",
            "(lambda: 0).__globals__",
            "getattr(int, '__subclasses__')()",
        )
        for exploit in exploits:
            result = execute(exploit, self.config)
            self.assertIsNotNone(result["error"])
            self.assertIn("PermissionError", result["error"])
            self.assertIn("forbidden by sandbox security policy", result["error"])

    def test_allowed_directories(self):
        allowed_file = self.allowed_dir / "test.txt"
        allowed_file.write_text("sandbox content")

        self.assertTrue(is_path_allowed(allowed_file, self.config.allowed_directories))
        self.assertFalse(is_path_allowed("/etc/passwd", self.config.allowed_directories))

    def test_file_read_allowed(self):
        allowed_file = self.allowed_dir / "data.txt"
        allowed_file.write_text("secret_token")

        code = f"with open('{allowed_file}') as f:\n    print(f.read())"
        result = execute(code, self.config)
        self.assertIsNone(result["error"])
        self.assertIn("secret_token", result["output"])

    def test_file_read_blocked(self):
        code = "with open('/etc/passwd') as f:\n    print(f.read())"
        result = execute(code, self.config)
        self.assertIsNotNone(result["error"])
        self.assertIn("PermissionError", result["error"])

    def test_network_blocked(self):
        # Attempting socket connection should raise SandboxSecurityError (PermissionError)
        code = "import socket\ns = socket.socket()"
        # authorize socket import temporarily to test network blocker
        custom_config = SandboxConfig(
            authorized_imports=["socket"],
            allowed_directories=self.config.allowed_directories,
            max_execution_time_seconds=2,
        )
        result = execute(code, custom_config)
        self.assertIsNotNone(result["error"])
        self.assertIn("Network access is disabled", result["error"])

    def test_final_answer(self):
        code = "x = 40 + 2\nfinal_answer(str(x))"
        result = execute(code, self.config)
        self.assertEqual(result["final_answer"], "42")
        self.assertIsNone(result["error"])

    def test_auto_print_bare_expression(self):
        # Tools or functions returning values should be auto-printed if the LLM forgets print()
        mock_tools = {"my_tool": lambda: "tool_output_value"}
        code = "my_tool()"
        result = execute(code, self.config, tools=mock_tools)
        self.assertIsNone(result["error"])
        self.assertIn("tool_output_value", result["output"])

    def test_timeout_execution(self):
        code = "while True:\n    pass"
        result = execute(code, self.config)
        self.assertIsNotNone(result["error"])
        self.assertIn("SandboxTimeoutError", result["error"])

    def test_propagation_keyboard_interrupt(self):
        # KeyboardInterrupt must NOT be caught as a normal string error
        code = "raise KeyboardInterrupt()"
        with self.assertRaises(KeyboardInterrupt):
            execute(code, self.config)

    def test_propagation_system_exit(self):
        # SystemExit must NOT be caught as a normal string error
        code = "raise SystemExit()"
        with self.assertRaises(SystemExit):
            execute(code, self.config)

    def test_mcp_tool_not_subject_to_sandbox_timeout(self):
        import time

        def slow_mcp_tool():
            time.sleep(1.5)
            return "tool_finished"

        short_cfg = SandboxConfig(
            authorized_imports=[],
            allowed_directories=[str(self.allowed_dir)],
            max_execution_time_seconds=1,
            max_memory_mb=128,
        )
        result = execute("print(slow_tool())", short_cfg, tools={"slow_tool": slow_mcp_tool})
        self.assertIsNone(result["error"])
        self.assertIn("tool_finished", result["output"])

    def test_persistent_namespace(self):
        from sandbox.sandbox import build_namespace

        ns = build_namespace(self.config)
        r1 = execute("x = 21 * 2", self.config, namespace=ns)
        self.assertIsNone(r1["error"])
        r2 = execute("print(x)", self.config, namespace=ns)
        self.assertIsNone(r2["error"])
        self.assertIn("42", r2["output"])


if __name__ == "__main__":
    unittest.main()
