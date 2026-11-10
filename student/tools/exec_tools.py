"""Execution tools for SWE-bench and the agent framework.

Implements the mandatory execution tools specified in Section V.5.3:
- run_tests()
- get_patch()
- run_command(command, workdir)

All commands and patch generations run in the repository located at TESTBED_PATH.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _apply_patch_to_docker(cid: str, patch: str) -> bool:
    """Apply unified git diff to /testbed inside a running container."""
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as f:
        f.write(patch)
        temp_diff = f.name
    try:
        cp_res = subprocess.run(
            ["docker", "cp", temp_diff, f"{cid}:/tmp/patch.diff"],
            capture_output=True,
        )
        if cp_res.returncode != 0:
            return False
        # Reset uncommitted files first so patch applies cleanly
        subprocess.run(
            ["docker", "exec", "-w", "/testbed", cid, "git", "checkout", "."],
            capture_output=True,
        )
        for apply_cmd in (
            "git apply --verbose /tmp/patch.diff",
            "git apply --verbose --reject /tmp/patch.diff",
            "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
        ):
            res = subprocess.run(
                ["docker", "exec", "-w", "/testbed", cid, "/bin/bash", "-c", apply_cmd],
                capture_output=True,
            )
            if res.returncode == 0:
                return True
        return False
    finally:
        if os.path.exists(temp_diff):
            os.remove(temp_diff)


def get_testbed_path() -> Path:
    """Return the repository root from TESTBED_PATH, defaulting to current working dir."""
    testbed_env = os.environ.get("TESTBED_PATH")
    if testbed_env:
        return Path(testbed_env).resolve()
    return Path.cwd().resolve()


def resolve_workdir(workdir: str | Path | None = None) -> Path:
    """Resolve a working directory path, defaulting to testbed if None."""
    if not workdir or str(workdir).strip() in (".", ""):
        return get_testbed_path()
    p = Path(workdir)
    if p.is_absolute():
        return p.resolve()
    return (get_testbed_path() / p).resolve()


def run_command(
    command: str,
    workdir: str | Path | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Execute a shell command in the specified working directory.

    Args:
        command: Shell command string to execute.
        workdir: Working directory in which to execute (defaults to TESTBED_PATH).
        timeout: Maximum execution time in seconds before timeout (default: 120s).

    Returns:
        A dict containing 'stdout', 'stderr', and 'exit_code'.
    """
    if not command or not command.strip():
        return {"stdout": "", "stderr": "Error: Command cannot be empty.", "exit_code": 1}

    cwd = resolve_workdir(workdir)
    if not cwd.exists():
        return {
            "stdout": "",
            "stderr": f"Error: Working directory '{cwd}' does not exist.",
            "exit_code": 1,
        }
    if not cwd.is_dir():
        return {
            "stdout": "",
            "stderr": f"Error: Working directory '{cwd}' is not a directory.",
            "exit_code": 1,
        }

    cid = os.environ.get("DOCKER_CONTAINER_ID")
    if cid:
        # Sync current changes to container before running command
        patch = get_patch()
        if patch:
            _apply_patch_to_docker(cid, patch)

        container_workdir = "/testbed"
        try:
            rel = cwd.relative_to(get_testbed_path())
            container_workdir = f"/testbed/{rel}" if str(rel) != "." else "/testbed"
        except ValueError:
            container_workdir = "/testbed"

        cmd = ["docker", "exec", "-w", container_workdir, cid, "/bin/bash", "-c", command]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode() if e.stdout else "")
            stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode() if e.stderr else "")
            timeout_msg = f"\nCommand timed out after {timeout} seconds."
            return {
                "stdout": stdout,
                "stderr": (stderr + timeout_msg).strip(),
                "exit_code": 124,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Error executing command in docker: {e}",
                "exit_code": 1,
            }

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode() if e.stdout else "")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode() if e.stderr else "")
        timeout_msg = f"\nCommand timed out after {timeout} seconds."
        return {
            "stdout": stdout,
            "stderr": (stderr + timeout_msg).strip(),
            "exit_code": 124,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Error executing command: {e}",
            "exit_code": 1,
        }


def get_patch(repo_path: str | Path | None = None) -> str:
    """Retrieve the unified git diff of all changes made to the repository.

    Uses 'git -c core.fileMode=false diff' as required by Section V.4.

    Args:
        repo_path: Optional repository root path (defaults to TESTBED_PATH).

    Returns:
        The git patch diff as a string, or empty string if no changes exist.
    """
    cwd = resolve_workdir(repo_path)
    if not (cwd / ".git").exists():
        return ""

    try:
        # 1. Unstaged changes
        res = subprocess.run(
            ["git", "-c", "core.fileMode=false", "diff"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        diff_output = res.stdout

        # 2. If empty, check staged changes or diff against HEAD in case 'git add' was run
        if not diff_output.strip():
            res_head = subprocess.run(
                ["git", "-c", "core.fileMode=false", "diff", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            diff_output = res_head.stdout

        return diff_output
    except Exception:
        return ""


def run_tests(
    test_command: str | None = None,
    workdir: str | Path | None = None,
    timeout: float = 300.0,
) -> str:
    """Execute the evaluation script or test suite.

    Searches for an evaluation script in the following locations:
    1. test_command argument (if provided)
    2. EVAL_SCRIPT_PATH environment variable
    3. /eval.sh
    4. <workdir>/eval.sh
    5. /tmp/eval.sh
    6. EVAL_SCRIPT environment variable (as bash script content)

    Args:
        test_command: Optional explicit test command to run.
        workdir: Working directory for execution (defaults to TESTBED_PATH).
        timeout: Maximum runtime in seconds (default: 300s).

    Returns:
        Combined stdout and stderr of the test execution.
    """
    cwd = resolve_workdir(workdir)

    cid = os.environ.get("DOCKER_CONTAINER_ID")
    if cid:
        patch = get_patch()
        if patch:
            _apply_patch_to_docker(cid, patch)

        if test_command:
            cmd_result = run_command(test_command, workdir=cwd, timeout=timeout)
            return f"{cmd_result['stdout']}\n{cmd_result['stderr']}".strip()

        eval_script = os.environ.get("EVAL_SCRIPT")
        if eval_script and eval_script.strip():
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
                f.write(eval_script)
                temp_sh = f.name
            try:
                subprocess.run(["docker", "cp", temp_sh, f"{cid}:/eval.sh"], check=False)
                subprocess.run(["docker", "exec", "-w", "/testbed", cid, "chmod", "+x", "/eval.sh"], check=False)
                res = subprocess.run(
                    ["docker", "exec", "-w", "/testbed", cid, "/bin/bash", "/eval.sh"],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return f"{res.stdout}\n{res.stderr}".strip()
            except subprocess.TimeoutExpired:
                return f"Error: Tests timed out after {timeout} seconds."
            finally:
                if os.path.exists(temp_sh):
                    os.remove(temp_sh)

    # 1. Explicit command provided
    if test_command:
        cmd_result = run_command(test_command, workdir=cwd, timeout=timeout)
        output = cmd_result["stdout"]
        if cmd_result["stderr"]:
            output = f"{output}\n{cmd_result['stderr']}".strip()
        return output

    # 2. Check for script path from environment variable
    env_script_path = os.environ.get("EVAL_SCRIPT_PATH")
    if env_script_path and Path(env_script_path).exists():
        script_file = Path(env_script_path).resolve()
        cmd_result = run_command(f"/bin/bash {script_file}", workdir=cwd, timeout=timeout)
        return (f"{cmd_result['stdout']}\n{cmd_result['stderr']}").strip()

    # 3. Check standard locations for eval.sh
    candidates = [
        Path("/eval.sh"),
        cwd / "eval.sh",
        Path("/tmp/eval.sh"),
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            cmd_result = run_command(f"/bin/bash {candidate}", workdir=cwd, timeout=timeout)
            return (f"{cmd_result['stdout']}\n{cmd_result['stderr']}").strip()

    # 4. Check for eval script content in environment variable
    eval_script_content = os.environ.get("EVAL_SCRIPT")
    if eval_script_content and eval_script_content.strip():
        tmp_script = Path("/tmp/agent_eval_script.sh")
        try:
            tmp_script.write_text(eval_script_content, encoding="utf-8")
            cmd_result = run_command(f"/bin/bash {tmp_script}", workdir=cwd, timeout=timeout)
            return (f"{cmd_result['stdout']}\n{cmd_result['stderr']}").strip()
        finally:
            if tmp_script.exists():
                tmp_script.unlink(missing_ok=True)

    # 5. Fallback: try pytest if pytest is present in the repository
    cmd_result = run_command("pytest", workdir=cwd, timeout=timeout)
    if cmd_result["exit_code"] != 127:  # 127 = command not found
        return (f"{cmd_result['stdout']}\n{cmd_result['stderr']}").strip()

    return "Error: No evaluation script found (checked /eval.sh, EVAL_SCRIPT_PATH, eval.sh) and pytest is unavailable."
