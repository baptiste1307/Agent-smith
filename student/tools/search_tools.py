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
from pathlib import Path

# Directories ignored by all search operations
MAX_SEARCH_RESULTS = 50

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
    """Return repository root from TESTBED_PATH or current working directory.

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
    """Discover non-binary, non-ignored files matching an optional pattern.

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
            d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")
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


def normalize_file_pattern(
    file_pattern: str | None, testbed: Path
) -> str | None:
    """Turn the many ways an LLM names files into a testbed-relative glob.

    Accepted forms: a glob ('*.py', 'pkg/*.py'), a path inside the
    container ('/testbed/pkg/mod.py'), an absolute host path under
    TESTBED_PATH, a relative path ('./pkg/mod.py'), or a directory
    ('pkg/sub', '/testbed/pkg/sub/') which means every file under it.
    """
    if not file_pattern or not file_pattern.strip():
        return None
    pat = file_pattern.strip()

    if pat.rstrip("/") == "/testbed":
        return None
    if pat.startswith("/testbed/"):
        pat = pat[len("/testbed/"):]
    elif os.path.isabs(pat):
        try:
            pat = str(Path(pat).resolve().relative_to(testbed))
        except ValueError:
            return pat  # outside the repository: matches nothing
    while pat.startswith("./"):
        pat = pat[2:]

    has_glob = any(ch in pat for ch in "*?[")
    if pat and not has_glob and (testbed / pat).is_dir():
        pat = pat.rstrip("/") + "/*"  # fnmatch's * also crosses '/'
    return pat or None


def search_code(pattern: str, file_pattern: str | None = None) -> str:
    """Perform a grep-like search in the codebase.

    Args:
        pattern: Regex pattern or literal string to search for.
        file_pattern: Optional filter on the searched files: a glob
            ('*.py', 'pkg/*.py'), a file path ('/testbed/pkg/mod.py',
            'pkg/mod.py') or a directory ('pkg/sub').

    Returns:
        Formatted string of matching lines:
        /absolute/path_to_file.py:<line_number> <line_content>
        When nothing matches, an explicit message says whether no file
        matched file_pattern or no line matched pattern.
    """
    if not pattern:
        return ""

    testbed = get_testbed_path()
    try:
        regex = re.compile(pattern)
    except re.error:
        regex = re.compile(re.escape(pattern))

    glob = normalize_file_pattern(file_pattern, testbed)
    files = iter_codebase_files(testbed, glob)
    if not files:
        return (
            f"No file matches file_pattern {file_pattern!r}. Use a glob "
            "such as '*.py', or a path inside the repository such as "
            "'/testbed/pkg/module.py' or 'pkg/subdir'."
        )
    results: list[str] = []
    total_found = 0

    for file_path in files:
        abs_path = str(file_path.resolve())
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, start=1):
                    if regex.search(line):
                        total_found += 1
                        if len(results) < MAX_SEARCH_RESULTS:
                            content = line.rstrip("\r\n")
                            results.append(f"{abs_path}:{lineno} {content}")
        except (OSError, PermissionError):
            continue

    if total_found == 0:
        scope = (
            f"the {len(files)} file(s) matching file_pattern "
            f"{file_pattern!r}"
            if glob
            else f"the {len(files)} file(s) of the repository"
        )
        return f"No match for pattern {pattern!r} in {scope}."

    if total_found > MAX_SEARCH_RESULTS:
        results.append(
            f"... [showing first {MAX_SEARCH_RESULTS} of "
            f"{total_found} matches. "
            f"Refine your pattern or file_pattern to narrow down]"
        )

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

    if len(results) > MAX_SEARCH_RESULTS:
        total = len(results)
        results = results[:MAX_SEARCH_RESULTS]
        results.append(
            f"... [showing first {MAX_SEARCH_RESULTS} of "
            f"{total} definition matches. "
            f"Refine your query to narrow down]"
        )

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
            with open(file_path, encoding="utf-8", errors="replace") as f:
                for lineno, line_text in enumerate(f, start=1):
                    if ref_regex.search(line_text):
                        content = line_text.rstrip("\r\n")
                        results.append(f"{abs_path}:{lineno} {content}")
        except (OSError, PermissionError):
            continue

    if len(results) > MAX_SEARCH_RESULTS:
        total = len(results)
        results = results[:MAX_SEARCH_RESULTS]
        results.append(
            f"... [showing first {MAX_SEARCH_RESULTS} of "
            f"{total} reference matches. "
            f"Refine your query to narrow down]"
        )

    return "\n".join(results)
