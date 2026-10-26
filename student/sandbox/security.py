"""Security rules and constraints for the execution sandbox."""

import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any


def is_import_allowed(name: str, authorized_imports: list[str]) -> bool:
    """Check if a requested module import is permitted by the allowlist.

    Handles two types of rules in authorized_imports:
    1. Exact import: e.g. "math", "json" (only the exact module is allowed)
    2. Wildcard: e.g. "collections.*" (allows the root module "collections"
       AND all its submodules such as "collections.abc")

    Args:
        name: The full name of the module to import (e.g. 'math', 'collections.abc').
        authorized_imports: List of authorized import patterns.

    Returns:
        True if the import is permitted by at least one rule, False otherwise.
    """
    for authorized in authorized_imports:
        # Case 1: exact match (e.g. 'math')
        if name == authorized:
            return True

        # Case 2: wildcard rule ending with '.*' (e.g. 'collections.*')
        elif authorized.endswith(".*"):
            # Strip the trailing '.*' to get the base module name
            base = authorized[:-2]
            # Allow the base module itself ('collections') or any submodule ('collections.abc')
            if name == base or name.startswith(base + "."):
                return True

    # If no rule matches, import is denied
    return False


def make_safe_import(
    authorized_imports: list[str],
    original_import: Callable,
) -> Callable:
    """Create a restricted version of builtins.__import__.

    Intercepts every 'import' or 'from ... import ...' statement executed
    inside the sandbox and raises ImportError if the module is not authorized.

    Args:
        authorized_imports: The list of authorized import patterns.
        original_import: Python's original builtins.__import__ function.

    Returns:
        A safe_import function matching the signature of builtins.__import__.
    """

    def safe_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        # 1. Verify that the requested top-level module is authorized
        if not is_import_allowed(name, authorized_imports):
            raise ImportError(
                f"Import of module '{name}' is not authorized by sandbox policy."
            )

        # 2. When 'fromlist' is provided (e.g. 'from collections import abc'),
        # verify that any requested submodule is also authorized
        if fromlist:
            for item in fromlist:
                full_submodule = f"{name}.{item}"
                # If the submodule is explicitly forbidden while parent is checked
                if not is_import_allowed(
                    name, authorized_imports
                ) and not is_import_allowed(full_submodule, authorized_imports):
                    raise ImportError(
                        f"Import of '{full_submodule}' is not authorized by sandbox policy."
                    )

        # 3. Delegate to original __import__ if all security checks pass
        return original_import(name, globals, locals, fromlist, level)

    return safe_import


def is_path_allowed(
    filepath: str | Path, allowed_directories: list[str]
) -> bool:
    """Check if a file path resides within one of the allowed directories.

    Resolves symbolic links and relative segments (e.g. '../') to canonical
    absolute paths to prevent directory traversal attacks.

    Args:
        filepath: The path to check (str or Path).
        allowed_directories: List of authorized directory paths.

    Returns:
        True if the path is inside (or is) an allowed directory, False otherwise.
    """
    if not allowed_directories:
        return False

    try:
        real_target = os.path.realpath(os.path.abspath(str(filepath)))
        for allowed in allowed_directories:
            real_allowed = os.path.realpath(os.path.abspath(str(allowed)))
            # Target must either be the directory itself or a child path inside it
            if real_target == real_allowed or real_target.startswith(
                real_allowed.rstrip(os.sep) + os.sep
            ):
                return True
        return False
    except Exception:
        return False


def make_safe_open(
    allowed_directories: list[str], original_open: Callable
) -> Callable:
    """Create a restricted version of builtins.open.

    Intercepts file open calls and ensures target paths are within
    allowed_directories. Standard stream descriptors (0, 1, 2) are permitted.

    Args:
        allowed_directories: List of allowed filesystem paths.
        original_open: Python's original builtins.open function.

    Returns:
        A safe_open function matching the signature of builtins.open.
    """
    def safe_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        # Allow standard stream file descriptors (0=stdin, 1=stdout, 2=stderr)
        if isinstance(file, int):
            if file in (0, 1, 2):
                return original_open(file, *args, **kwargs)
            raise PermissionError(
                "Access to arbitrary file descriptors is denied by sandbox policy."
            )

        # Validate filesystem path against allowed directories
        if not is_path_allowed(file, allowed_directories):
            raise PermissionError(
                f"Access to path '{file}' is denied by sandbox policy "
                f"(allowed directories: {allowed_directories})."
            )

        return original_open(file, *args, **kwargs)

    return safe_open


class SandboxSecurityError(PermissionError):
    """Raised when sandbox security constraints are violated."""
    pass


class SandboxTimeoutError(TimeoutError):
    """Raised when sandboxed execution exceeds the configured time limit."""
    pass


class FinalAnswerException(BaseException):
    """Signals that final_answer(answer) was called by the agent.

    Inherits from BaseException (like KeyboardInterrupt and SystemExit) rather
    than Exception so that 'except Exception:' blocks in LLM-generated code
    cannot accidentally catch or swallow this termination signal.
    """

    def __init__(self, answer: str) -> None:
        self.answer = str(answer)
        super().__init__(self.answer)


class NetworkBlocker:
    """Temporarily disables network socket creation during sandbox execution.

    Why temporary blocking is needed:
    1. The main agent interacts with LLM APIs over the Internet (network active).
    2. Just before executing untrusted LLM-generated code: block the network (block()).
    3. The LLM code executes in isolation (cannot cheat, exfiltrate data, or fetch solutions).
    4. Once execution finishes: restore the network (unblock()) so the agent can resume LLM calls.
    """

    def __init__(self) -> None:
        self._orig_socket = socket.socket
        self._orig_create_connection = getattr(socket, "create_connection", None)
        self._orig_getaddrinfo = getattr(socket, "getaddrinfo", None)

    def block(self) -> None:
        """Block all socket creation and connection attempts."""
        def disabled_socket(*args: Any, **kwargs: Any) -> Any:
            raise SandboxSecurityError(
                "Network access is disabled by sandbox policy."
            )

        socket.socket = disabled_socket  # type: ignore[misc]
        if self._orig_create_connection:
            socket.create_connection = disabled_socket  # type: ignore[misc]
        if self._orig_getaddrinfo:
            socket.getaddrinfo = disabled_socket  # type: ignore[misc]

    def unblock(self) -> None:
        """Restore original socket functions."""
        socket.socket = self._orig_socket  # type: ignore[misc]
        if self._orig_create_connection:
            socket.create_connection = self._orig_create_connection  # type: ignore[misc]
        if self._orig_getaddrinfo:
            socket.getaddrinfo = self._orig_getaddrinfo  # type: ignore[misc]

