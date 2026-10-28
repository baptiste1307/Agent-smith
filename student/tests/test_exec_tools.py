"""Unit tests for tools/exec_tools.py."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.exec_tools import get_patch, run_command, run_tests


class TestExecTools(unittest.TestCase):
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

    def test_run_command_basic(self):
        res = run_command("echo 'Agent Smith'", workdir=self.temp_dir)
        self.assertEqual(res["exit_code"], 0)
        self.assertIn("Agent Smith", res["stdout"])
        self.assertEqual(res["stderr"], "")

    def test_run_command_failure(self):
        res = run_command("python3 -c 'import sys; sys.exit(7)'", workdir=self.temp_dir)
        self.assertEqual(res["exit_code"], 7)

    def test_run_command_workdir(self):
        sub = self.temp_dir / "subdir"
        sub.mkdir()
        res = run_command("pwd", workdir=sub)
        self.assertEqual(res["exit_code"], 0)
        self.assertIn("subdir", res["stdout"])

    def test_run_command_timeout(self):
        res = run_command("python3 -c 'import time; time.sleep(5)'", workdir=self.temp_dir, timeout=0.3)
        self.assertEqual(res["exit_code"], 124)
        self.assertIn("timed out", res["stderr"].lower())

    def test_get_patch_git_diff(self):
        repo_dir = self.temp_dir / "git_repo"
        repo_dir.mkdir()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, check=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo_dir, check=True)

        target_file = repo_dir / "code.py"
        target_file.write_text("def solve():\n    return 1\n")
        subprocess.run(["git", "add", "code.py"], cwd=repo_dir, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)

        # Before edits, patch should be empty
        empty_patch = get_patch(repo_dir)
        self.assertEqual(empty_patch.strip(), "")

        # Modify file
        target_file.write_text("def solve():\n    return 42\n")

        # Now get_patch should produce the unified diff
        patch = get_patch(repo_dir)
        self.assertIn("diff --git", patch)
        self.assertIn("-    return 1", patch)
        self.assertIn("+    return 42", patch)

    def test_run_tests_with_eval_sh(self):
        eval_script = self.temp_dir / "eval.sh"
        eval_script.write_text("#!/bin/bash\necho 'ALL UNIT TESTS PASSED'\nexit 0\n")
        eval_script.chmod(0o755)

        output = run_tests(workdir=self.temp_dir)
        self.assertIn("ALL UNIT TESTS PASSED", output)

    def test_run_tests_explicit_command(self):
        output = run_tests(test_command="echo 'Explicit Test Output'", workdir=self.temp_dir)
        self.assertIn("Explicit Test Output", output)


if __name__ == "__main__":
    unittest.main()
