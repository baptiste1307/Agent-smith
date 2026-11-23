"""File system tools for SWE-bench and the agent framework.

Implements the mandatory file system tools specified in Section V.5.1:
- read_file(filepath, start_line, end_line)
- edit_file(filepath, old_str, new_str)
- list_files(directory, pattern)

All paths are resolved relative to TESTBED_PATH if not absolute.
"""

import ast
import os
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
}
MAX_RESULTS = 100


def get_testbed_path() -> Path:
    """Return repository root from TESTBED_PATH, defaulting to current working dir."""
    testbed_env = os.environ.get("TESTBED_PATH")
    if testbed_env:
        return Path(testbed_env).resolve()
    return Path.cwd().resolve()


def resolve_path(filepath: str | Path) -> Path:
    """Resolve a file path relative to TESTBED_PATH if not absolute.

    If filepath starts with '/testbed', strips the '/testbed' prefix
    and resolves relative to get_testbed_path().
    """
    p_str = str(filepath).strip()
    testbed = get_testbed_path()

    if p_str in ("/testbed", "/testbed/"):
        return testbed

    if p_str.startswith("/testbed/"):
        rel = p_str[len("/testbed/") :]
        return (testbed / rel).resolve()

    p = Path(p_str)
    if p.is_absolute():
        return p.resolve()
    return (testbed / p).resolve()


def read_file(
    filepath: str | Path, start_line: int = 1, end_line: int = -1
) -> str:
    """Read the content of a file with line numbers (format similar to cat -n).

    Args:
        filepath: Path to the target file.
        start_line: 1-indexed starting line number (default: 1).
        end_line: 1-indexed ending line number (default: -1, reads to end of file).

    Returns:
        Formatted string:
        <line_number>: <line_content>
    """
    path = resolve_path(filepath)

    if not path.exists():
        return f"Error: File '{filepath}' does not exist."
    if not path.is_file():
        return f"Error: Path '{filepath}' is not a regular file."

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"Error: Permission denied accessing '{filepath}'."
    except Exception as e:
        return f"Error reading '{filepath}': {e}"

    lines = content.splitlines()
    total_lines = len(lines)

    if total_lines == 0:
        return ""

    if start_line < 1:
        start_line = 1

    if start_line > total_lines:
        return (
            f"Error: start_line ({start_line}) is greater than total lines ({total_lines})."
        )

    # If end_line is -1, cap to 200 lines from start_line to avoid token bloat
    truncated_lines = False
    if end_line == -1:
        max_default_end = start_line + 199
        if total_lines > max_default_end:
            actual_end = max_default_end
            truncated_lines = True
        else:
            actual_end = total_lines
    elif end_line > total_lines:
        actual_end = total_lines
    else:
        actual_end = end_line

    if actual_end < start_line:
        return f"Error: end_line ({actual_end}) is before start_line ({start_line})."

    output = []
    for lineno in range(start_line, actual_end + 1):
        line_content = lines[lineno - 1]
        output.append(f"{lineno}: {line_content}")

    if truncated_lines:
        output.append(
            f"... (showing lines {start_line}-{actual_end} of {total_lines}; "
            f"pass start_line and end_line to read further)"
        )

    return "\n".join(output)


def edit_file(filepath: str | Path, old_str: str, new_str: str) -> str:
    """Replace an exact string in a file with a new string.

    Args:
        filepath: Path to the target file.
        old_str: Exact string to replace. Must be unique in the file.
        new_str: Replacement string.

    Returns:
        Success message or descriptive error string.
    """
    if not old_str:
        return "Error: old_str cannot be empty."

    path = resolve_path(filepath)

    if not path.exists():
        return f"Error: File '{filepath}' does not exist."
    if not path.is_file():
        return f"Error: Path '{filepath}' is not a regular file."

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"Error: Permission denied accessing '{filepath}'."
    except Exception as e:
        return f"Error reading '{filepath}': {e}"

    count = content.count(old_str)
    if count == 0:
        return f"Error: '{old_str}' was not found in '{filepath}'."
    if count > 1:
        return (
            f"Error: '{old_str}' was found {count} times in '{filepath}'. "
            "It must appear exactly once to ensure safe replacement."
        )

    updated = content.replace(old_str, new_str, 1)

    try:
        path.write_text(updated, encoding="utf-8")
    except PermissionError:
        return f"Error: Permission denied writing to '{filepath}'."
    except Exception as e:
        return f"Error writing to '{filepath}': {e}"

    # Explicit syntax feedback for Python files (required by subject Section V.1)
    if path.suffix == ".py":
        try:
            ast.parse(updated, filename=str(path))
        except SyntaxError as e:
            return (
                f"File '{filepath}' updated, WARNING: edit introduced a Python "
                f"SyntaxError at line {e.lineno}: {e.msg}. Please fix it."
            )

    return f"File '{filepath}' updated successfully."


def list_files(directory: str | Path = ".", pattern: str = "*") -> str:
    """Recursively list files in a directory matching a pattern.

    Args:
        directory: Directory to search (default: current/testbed directory).
        pattern: Shell-style glob pattern (e.g. '*.py', default: '*').

    Returns:
        Sorted list of absolute paths matching the pattern, separated by newlines.
    """
    dir_path = resolve_path(directory)

    if not dir_path.exists():
        return f"Error: Directory '{directory}' does not exist."
    if not dir_path.is_dir():
        return f"Error: Path '{directory}' is not a directory."

    if not pattern:
        pattern = "*"

    try:
        matched: list[Path] = []
        for path in dir_path.rglob(pattern):
            if any(excluded in path.parts for excluded in EXCLUDED_DIRS):
                continue
            if path.is_file():
                matched.append(path.resolve())

        matched.sort()
        total_count = len(matched)

        if total_count == 0:
            return ""

        truncated = False
        if total_count > MAX_RESULTS:
            matched = matched[:MAX_RESULTS]
            truncated = True

        output = [str(p) for p in matched]
        if truncated:
            output.append(
                f"... (truncated: showing {MAX_RESULTS} of {total_count} files)"
            )

        return "\n".join(output)

    except PermissionError:
        return f"Error: Permission denied accessing '{directory}'."
    except Exception as e:
        return f"Error listing directory '{directory}': {e}"
