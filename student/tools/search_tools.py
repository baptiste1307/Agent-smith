"""Code search tools for SWE-bench and the agent framework.

Implements the mandatory code search tools specified in Section V.5.2:
- search_code(pattern, file_pattern)
- search_function_or_class_definition_in_code(name)
- find_references(name, filepath, line)

These tools operate independently on the codebase located at TESTBED_PATH.
Format of each result line:
/absolute/path_to_file.py:<line_number> <line_content>
"""

import ast
import fnmatch
import os
import re
from collections.abc import Iterator
from pathlib import Path

# Directories ignored by all search operations
IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    ".tox",
    ".nox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
    ".eggs",
    ".idea",
    ".vscode",
}

# Binary and non-code file extensions ignored by text search
BINARY_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    ".o",
    ".a",
    ".obj",
    ".lib",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".tiff",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".whl",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".mp3",
    ".wav",
    ".ogg",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".parquet",
    ".feather",
    ".pkl",
    ".pickle",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
}


def get_testbed_path() -> Path:
    """Return the repository root from TESTBED_PATH, defaulting to current working directory.

    During SWE-bench evaluation and isolation testing, the moulinette sets
    TESTBED_PATH to point to the repository root.
    """
    testbed_env = os.environ.get("TESTBED_PATH")
    if testbed_env:
        return Path(testbed_env).resolve()
    return Path.cwd().resolve()


def is_binary_file(path: Path) -> bool:
    """Check if a file appears to be binary by extension or null byte check."""
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return True
    except (OSError, PermissionError):
        return True
    return False


def iter_codebase_files(
    testbed: Path, file_pattern: str | None = None
) -> list[Path]:
    """Discover all non-binary, non-ignored files matching an optional file_pattern.

    Args:
        testbed: Root directory of the repository.
        file_pattern: Optional glob pattern (e.g. '*.py', 'src/*.py').

    Returns:
        Sorted list of matching Path objects.
    """
    if not testbed.exists() or not testbed.is_dir():
        return []

    matched: list[Path] = []
    for root, dirs, files in os.walk(testbed, topdown=True, followlinks=False):
        # Prune ignored and hidden directories
        dirs[:] = [
            d
            for d in dirs
            if d not in IGNORED_DIRS and not d.startswith(".")
        ]

        root_path = Path(root)
        for filename in files:
            if filename.startswith("."):
                continue

            file_path = root_path / filename

            # Filter with file_pattern if provided
            if file_pattern and file_pattern != "*":
                try:
                    rel_path_str = str(file_path.relative_to(testbed))
                except ValueError:
                    rel_path_str = str(file_path)

                if not (
                    fnmatch.fnmatch(filename, file_pattern)
                    or fnmatch.fnmatch(rel_path_str, file_pattern)
                    or fnmatch.fnmatch(rel_path_str, f"*/{file_pattern}")
                ):
                    continue

            # Skip binary files
            if is_binary_file(file_path):
                continue

            matched.append(file_path)

    return sorted(matched)


def search_code(pattern: str, file_pattern: str | None = None) -> str:
    """Perform a grep-like search in the codebase.

    Args:
        pattern: Regex pattern or literal string to search for.
        file_pattern: Optional file glob pattern to filter searched files (e.g. '*.py').

    Returns:
        Formatted string of matching lines:
        /absolute/path_to_file.py:<line_number> <line_content>
    """
    if not pattern:
        return ""

    testbed = get_testbed_path()
    try:
        regex = re.compile(pattern)
    except re.error:
        regex = re.compile(re.escape(pattern))

    files = iter_codebase_files(testbed, file_pattern)
    results: list[str] = []

    for file_path in files:
        abs_path = str(file_path.resolve())
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, start=1):
                    if regex.search(line):
                        content = line.rstrip("\r\n")
                        results.append(f"{abs_path}:{lineno} {content}")
        except (OSError, PermissionError):
            continue

    return "\n".join(results)


def search_function_or_class_definition_in_code(name: str) -> str:
    """Find the definition of a function or a class in the codebase.

    Uses Python AST parsing for precision, with regex fallback for
    malformed files or non-standard syntax.

    Args:
        name: Name of the function or class to find.

    Returns:
        Formatted string of matching definition lines:
        /absolute/path_to_file.py:<line_number> <line_content>
    """
    if not name or not name.strip():
        return ""

    clean_name = name.strip()
    testbed = get_testbed_path()
    files = iter_codebase_files(testbed, file_pattern="*.py")
    results: list[str] = []

    def_regex = re.compile(
        rf"^\s*(?:async\s+def|def|class)\s+{re.escape(clean_name)}\b"
    )

    for file_path in files:
        abs_path = str(file_path.resolve())
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            continue

        lines = source.splitlines()
        found_linenos: set[int] = set()

        # Method 1: AST parsing (accurate, excludes docstrings/comments)
        try:
            tree = ast.parse(source, filename=abs_path)
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    if node.name == clean_name:
                        found_linenos.add(node.lineno)
        except SyntaxError:
            pass

        # Method 2: Regex fallback
        if not found_linenos:
            for lineno, line in enumerate(lines, start=1):
                if def_regex.search(line):
                    found_linenos.add(lineno)

        for lineno in sorted(found_linenos):
            if 1 <= lineno <= len(lines):
                content = lines[lineno - 1].rstrip("\r\n")
                results.append(f"{abs_path}:{lineno} {content}")

    return "\n".join(results)


def find_references(
    name: str, filepath: str | None = None, line: int | None = None
) -> str:
    """Find all usages of a symbol (function or class) across the codebase.

    Args:
        name: Name of the symbol to find references for.
        filepath: Optional origin file path of the symbol.
        line: Optional line number of the definition.

    Returns:
        Formatted string of matching reference lines:
        /absolute/path_to_file.py:<line_number> <line_content>
    """
    if not name or not name.strip():
        return ""

    clean_name = name.strip()
    testbed = get_testbed_path()
    ref_regex = re.compile(rf"\b{re.escape(clean_name)}\b")

    files = iter_codebase_files(testbed, file_pattern=None)
    results: list[str] = []

    for file_path in files:
        abs_path = str(file_path.resolve())
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line_text in enumerate(f, start=1):
                    if ref_regex.search(line_text):
                        content = line_text.rstrip("\r\n")
                        results.append(f"{abs_path}:{lineno} {content}")
        except (OSError, PermissionError):
            continue

    return "\n".join(results)
