import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "student"))

from tools.exec_tools import get_patch as _get_patch  # noqa: E402
from tools.exec_tools import run_command as _run_command  # noqa: E402
from tools.exec_tools import run_tests as _run_tests  # noqa: E402
from tools.fs_tools import edit_file as _edit_file  # noqa: E402
from tools.fs_tools import list_files as _list_files  # noqa: E402
from tools.fs_tools import read_file as _read_file  # noqa: E402
from tools.search_tools import find_references as _find_references  # noqa: E402
from tools.search_tools import search_code as _search_code  # noqa: E402
from tools.search_tools import (  # noqa: E402
    search_function_or_class_definition_in_code as _search_definition,
)

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    sys.exit(
        "Cannot import mcp v2 (MCPServer) with "
        f"{sys.executable}. Try: cd student && uv run python ../mcp_tools_swebench.py"
    )


mcp = MCPServer("agent-smith-swebench")


@mcp.tool()
def read_file(filepath: str, start_line: int = 1, end_line: int = -1) -> str:
    """
	Read a file with line numbers, like cat -n.
	"""
    return _read_file(filepath, start_line, end_line)


@mcp.tool()
def edit_file(filepath: str, old_str: str, new_str: str) -> str:
    """
	Replace exactly one occurrence of old_str in a file.
	"""
    return _edit_file(filepath, old_str, new_str)


@mcp.tool()
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """
	List files under a directory matching a glob pattern.
	"""
    return _list_files(directory, pattern)


@mcp.tool()
def search_code(pattern: str, file_pattern: str = "*") -> str:
    """
	Search the codebase for a regex or literal pattern.
	"""
    return _search_code(pattern, file_pattern)


@mcp.tool()
def search_function_or_class_definition_in_code(name: str) -> str:
    """
	Find function or class definitions by name.
	"""
    return _search_definition(name)


@mcp.tool()
def find_references(
    name: str,
    filepath: str | None = None,
    line: int | None = None,
) -> str:
    """
	Find usages of a function, class, or variable name.
	"""
    return _find_references(name, filepath, line)


@mcp.tool()
def run_tests(
    test_command: str | None = None,
    workdir: str | None = None,
    timeout: float = 300.0,
) -> str:
    """
	Run the SWE-bench evaluation script, or an explicit test command.
	"""
    return _run_tests(test_command, workdir, timeout)


@mcp.tool()
def get_patch() -> str:
    """
	Return git -c core.fileMode=false diff for the testbed repository.
	"""
    return _get_patch()


@mcp.tool()
def run_command(
    command: str,
    workdir: str | None = None,
    timeout: float = 120.0,
) -> str:
    """
	Run a shell command and return stdout, stderr, and exit_code as JSON.
	"""
    result = _run_command(command, workdir, timeout)
    return json.dumps(result, indent=2)


@mcp.resource("swebench://repo-info")
def swebench_repo_info() -> str:
    """Return metadata about the active SWE-bench testbed repository."""
    import os

    testbed = os.environ.get("TESTBED_PATH", str(Path.cwd()))
    status = _run_command("git status --short", workdir=testbed, timeout=15.0)
    return json.dumps(
        {
            "testbed_path": testbed,
            "docker_container_id": os.environ.get("DOCKER_CONTAINER_ID", ""),
            "git_status": status.get("stdout", "").strip(),
        },
        indent=2,
    )


@mcp.prompt()
def swebench_workflow(issue: str = "") -> str:
    """Recommended step-by-step workflow for fixing a SWE-bench issue."""
    return (
        "Fix the repository issue step by step:\n"
        "1. Explore the codebase with search_code / search_function_or_class_definition_in_code / list_files.\n"
        "2. Read the relevant files with read_file to understand the root cause.\n"
        "3. Apply a minimal fix with edit_file.\n"
        "4. Verify with run_tests() or targeted pytest via run_command().\n"
        "5. Submit with final_answer(get_patch()).\n"
        + (f"\nIssue:\n{issue}" if issue else "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP server for the SWE-bench benchmark."
    )
    parser.add_argument(
        "--http", action="store_true", help="streamable HTTP instead of stdio"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
