"""Execution tools for SWE-bench and the agent framework.

Implements the mandatory execution tools specified in Section V.5.3:
- run_tests()
- get_patch()
- run_command(command, workdir)

All commands and patch generation run in repository at TESTBED_PATH.
"""

import os
import subprocess
from pathlib import Path
from typing import Any


def _write_file_to_docker(
    cid: str, host_file: Path, container_file: str
) -> bool:
    """Write a host file into a container via stdin.

    Avoids rootless Docker lchown errors.
    """
    parent_dir = str(Path(container_file).parent)
    subprocess.run(
        ["docker", "exec", cid, "mkdir", "-p", parent_dir],
        capture_output=True,
        timeout=15,
    )
    data = host_file.read_bytes()
    res = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            cid,
            "/bin/sh",
            "-c",
            'cat > "$1"',
            "_",
            container_file,
        ],
        input=data,
        capture_output=True,
        timeout=30,
    )
    return res.returncode == 0


def _host_changed_files(testbed: Path) -> list[str]:
    """Return the tracked files of the host testbed that differ from HEAD."""
    for cmd in (
        ["git", "-c", "core.fileMode=false", "diff", "--name-only", "HEAD"],
        ["git", "-c", "core.fileMode=false", "diff", "--name-only"],
    ):
        res = subprocess.run(
            cmd, cwd=testbed, capture_output=True, text=True, timeout=30
        )
        if res.returncode == 0:
            lines = res.stdout.splitlines()
            return [line.strip() for line in lines if line.strip()]
    return []


def _sync_host_changes_to_docker(cid: str, testbed: Path) -> bool:
    """Make the tracked files of /testbed in the container match the host.

    The container's tracked files are reset to HEAD, then every file the
    host changed is streamed in with `docker exec -i` (avoids `git apply`
    and rootless Docker `lchown` failures). Untracked files of the
    container (e.g. a reproduction script written with run_command) are
    kept: `git clean` is not used.
    """
    try:
        reset = subprocess.run(
            [
                "docker",
                "exec",
                "-w",
                "/testbed",
                cid,
                "/bin/bash",
                "-c",
                (
                    "git config --global --add safe.directory /testbed "
                    "2>/dev/null || true; git reset -q && git checkout -- ."
                ),
            ],
            capture_output=True,
            timeout=30,
        )
        if reset.returncode != 0:
            return False

        for rel_path in _host_changed_files(testbed):
            host_file = testbed / rel_path
            container_file = f"/testbed/{rel_path}"
            if host_file.exists():
                if not _write_file_to_docker(cid, host_file, container_file):
                    return False
            else:
                # File was deleted on host
                subprocess.run(
                    ["docker", "exec", cid, "rm", "-f", container_file],
                    capture_output=True,
                    timeout=15,
                )
        return True
    except Exception:
        return False


def _sync_docker_changes_to_host(cid: str, testbed: Path) -> bool:
    """Copy the tracked-file changes made in the container back to the host.

    run_command executes inside the container: without this step, a file
    modified there (sed -i, a Python script, git checkout...) would be
    erased by the next _sync_host_changes_to_docker and would never appear
    in get_patch(), which reads the host testbed.
    """
    try:
        res = subprocess.run(
            [
                "docker",
                "exec",
                "-w",
                "/testbed",
                cid,
                "git",
                "-c",
                "core.fileMode=false",
                "diff",
                "--no-renames",
                "--name-status",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if res.returncode != 0:
            return False

        container_changed: set[str] = set()
        for line in res.stdout.splitlines():
            status, _, rel_path = line.partition("\t")
            if not rel_path:
                continue
            container_changed.add(rel_path)
            host_file = testbed / rel_path
            if status.startswith("D"):
                host_file.unlink(missing_ok=True)
                continue
            content = subprocess.run(
                ["docker", "exec", cid, "cat", f"/testbed/{rel_path}"],
                capture_output=True,
                timeout=30,
            )
            if content.returncode != 0:
                return False
            host_file.parent.mkdir(parents=True, exist_ok=True)
            host_file.write_bytes(content.stdout)

        # Files changed on the host but restored in the container
        # (e.g. `git checkout file`): restore them on the host too
        for rel_path in set(_host_changed_files(testbed)) - container_changed:
            subprocess.run(
                ["git", "checkout", "HEAD", "--", rel_path],
                cwd=testbed,
                capture_output=True,
                timeout=30,
            )
        return True
    except Exception:
        return False


def get_testbed_path() -> Path:
    """Return repository root from TESTBED_PATH or current working dir."""
    testbed_env = os.environ.get("TESTBED_PATH")
    if testbed_env:
        return Path(testbed_env).resolve()
    return Path.cwd().resolve()


def resolve_workdir(workdir: str | Path | None = None) -> Path:
    """Resolve a working directory path, defaulting to testbed if None."""
    testbed = get_testbed_path()
    if not workdir or str(workdir).strip() in (".", ""):
        return testbed

    w_str = str(workdir).strip()
    if w_str in ("/testbed", "/testbed/"):
        return testbed

    if w_str.startswith("/testbed/"):
        return (testbed / w_str[len("/testbed/"):]).resolve()

    p = Path(w_str)
    if p.is_absolute():
        return p.resolve()
    return (testbed / p).resolve()


def run_command(
    command: str,
    workdir: str | Path | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Execute a shell command in the specified working directory.

    Args:
        command: Shell command string to execute.
        workdir: Working directory for execution (defaults to TESTBED_PATH).
        timeout: Maximum execution time in seconds (default: 120s).

    Returns:
        A dict containing 'stdout', 'stderr', and 'exit_code'.
    """
    if not command or not command.strip():
        return {
            "stdout": "",
            "stderr": "Error: Command cannot be empty.",
            "exit_code": 1,
        }

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
        sync_warning = ""
        # Always sync, even with an empty patch: the container must not keep
        # edits the host has reverted since the last command
        if not _sync_host_changes_to_docker(cid, get_testbed_path()):
            sync_warning = (
                "Warning: Failed to sync host edits to Docker container. "
                "Command executed on unpatched code.\n"
            )

        container_workdir = "/testbed"
        try:
            rel = cwd.relative_to(get_testbed_path())
            container_workdir = (
                f"/testbed/{rel}" if str(rel) != "." else "/testbed"
            )
        except ValueError:
            container_workdir = "/testbed"

        cmd = [
            "docker",
            "exec",
            "-w",
            container_workdir,
            cid,
            "/bin/bash",
            "-c",
            command,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if not _sync_docker_changes_to_host(cid, get_testbed_path()):
                sync_warning += (
                    "Warning: Failed to copy the files modified by this "
                    "command back to the host: they will not be in "
                    "get_patch().\n"
                )
            return {
                "stdout": proc.stdout,
                "stderr": (sync_warning + proc.stderr).strip(),
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired as e:
            stdout = (
                e.stdout
                if isinstance(e.stdout, str)
                else (e.stdout.decode() if e.stdout else "")
            )
            stderr = (
                e.stderr
                if isinstance(e.stderr, str)
                else (e.stderr.decode() if e.stderr else "")
            )
            timeout_msg = f"\nCommand timed out after {timeout} seconds."
            # Keep the edits made before the timeout
            _sync_docker_changes_to_host(cid, get_testbed_path())
            return {
                "stdout": stdout,
                "stderr": (sync_warning + stderr + timeout_msg).strip(),
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
        stdout = (
            e.stdout
            if isinstance(e.stdout, str)
            else (e.stdout.decode() if e.stdout else "")
        )
        stderr = (
            e.stderr
            if isinstance(e.stderr, str)
            else (e.stderr.decode() if e.stderr else "")
        )
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
        # Diff against HEAD: includes staged and unstaged changes, even when
        # only part of them was staged with 'git add'
        res = subprocess.run(
            ["git", "-c", "core.fileMode=false", "diff", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        diff_output = res.stdout
        if res.returncode != 0:
            # No commit yet (no HEAD): fall back to the working tree diff
            res = subprocess.run(
                ["git", "-c", "core.fileMode=false", "diff"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            diff_output = res.stdout

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
        if test_command:
            # run_command syncs the host edits into the container itself
            cmd_result = run_command(
                test_command, workdir=cwd, timeout=timeout
            )
            return (
                f"{cmd_result['stdout']}\n{cmd_result['stderr']}"
            ).strip()

        sync_warning = ""
        if not _sync_host_changes_to_docker(cid, get_testbed_path()):
            sync_warning = (
                "Warning: Failed to sync host edits to Docker container. "
                "Tests executed on unpatched code.\n"
            )

        eval_script = os.environ.get("EVAL_SCRIPT")
        if eval_script and eval_script.strip():
            try:
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        "-i",
                        cid,
                        "/bin/sh",
                        "-c",
                        "cat > /eval.sh && chmod +x /eval.sh",
                    ],
                    input=eval_script.encode("utf-8"),
                    capture_output=True,
                    timeout=30,
                )
                res = subprocess.run(
                    [
                        "docker",
                        "exec",
                        "-w",
                        "/testbed",
                        cid,
                        "/bin/bash",
                        "/eval.sh",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return f"{sync_warning}{res.stdout}\n{res.stderr}".strip()
            except subprocess.TimeoutExpired:
                return f"Error: Tests timed out after {timeout} seconds."

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
        cmd_result = run_command(
            f"/bin/bash {script_file}", workdir=cwd, timeout=timeout
        )
        return (f"{cmd_result['stdout']}\n{cmd_result['stderr']}").strip()

    # 3. Check standard locations for eval.sh
    candidates = [
        Path("/eval.sh"),
        cwd / "eval.sh",
        Path("/tmp/eval.sh"),
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            cmd_result = run_command(
                f"/bin/bash {candidate}", workdir=cwd, timeout=timeout
            )
            return (f"{cmd_result['stdout']}\n{cmd_result['stderr']}").strip()

    # 4. Check for eval script content in environment variable
    eval_script_content = os.environ.get("EVAL_SCRIPT")
    if eval_script_content and eval_script_content.strip():
        tmp_script = Path("/tmp/agent_eval_script.sh")
        try:
            tmp_script.write_text(eval_script_content, encoding="utf-8")
            cmd_result = run_command(
                f"/bin/bash {tmp_script}", workdir=cwd, timeout=timeout
            )
            return (f"{cmd_result['stdout']}\n{cmd_result['stderr']}").strip()
        finally:
            if tmp_script.exists():
                tmp_script.unlink(missing_ok=True)

    # 5. Fallback: try pytest if pytest is present in the repository
    cmd_result = run_command("pytest", workdir=cwd, timeout=timeout)
    if cmd_result["exit_code"] != 127:  # 127 = command not found
        return (f"{cmd_result['stdout']}\n{cmd_result['stderr']}").strip()

    return (
        "Error: No evaluation script found "
        "(checked /eval.sh, EVAL_SCRIPT_PATH, eval.sh) "
        "and pytest is unavailable."
    )
