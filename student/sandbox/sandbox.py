"""Sandbox used to execute LLM-generated code with restricted capabilities."""

import ast
import builtins
import contextlib
import io
import os
import resource
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pydantic
from models_public import SandboxConfig

from sandbox.security import (
    FinalAnswerException,
    NetworkBlocker,
    SandboxTimeoutError,
    make_safe_import,
    make_safe_open,
)

# Builtins removed from the sandbox namespace: they allow dynamic code
# execution, namespace introspection or interactive I/O
FORBIDDEN_BUILTINS = (
    "eval",
    "exec",
    "compile",
    "breakpoint",
    "input",
    "globals",
    "vars",
    "help",
    "exit",
    "quit",
    "copyright",
    "credits",
    "license",
)

FORBIDDEN_ATTRS = {
    "__subclasses__",
    "__globals__",
    "__builtins__",
    "__code__",
    "__bases__",
    "__base__",
    "__mro__",
    "__func__",
    "__closure__",
}


class SecurityASTValidator(ast.NodeVisitor):
    """Enforce sandbox security rules at the AST level before code execution."""

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRS:
            raise PermissionError(
                f"Access to attribute '{node.attr}' is forbidden by sandbox security policy."
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                if node.args[1].value in FORBIDDEN_ATTRS:
                    raise PermissionError(
                        f"Access to attribute '{node.args[1].value}' is forbidden by sandbox security policy."
                    )
        self.generic_visit(node)

MAX_OUTPUT_CHARS = 10_000


def load_config(sandbox_template_path: str | None = None) -> SandboxConfig:
    """Load and validate the sandbox configuration from a JSON file.

    Args:
        sandbox_template_path: Path to the JSON configuration file.
            If None, the default SandboxConfig is returned.

    Returns:
        The validated SandboxConfig.

    Raises:
        ValueError: If the file cannot be read or does not match the
            SandboxConfig schema.
    """
    if sandbox_template_path is None:
        return SandboxConfig()

    try:
        sandbox_template = Path(sandbox_template_path).read_text()
        config = SandboxConfig.model_validate_json(sandbox_template)

    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        pydantic.ValidationError,
    ) as e:
        raise ValueError(f"{sandbox_template_path} : {e}") from e

    return config


def build_safe_builtins(config: SandboxConfig) -> dict[str, Any]:
    """Build a restricted copy of builtins for sandboxed execution.

    The global builtins module is never modified: a copy is made, then
    __import__ and open are replaced by their safe versions and the
    dangerous builtins listed in FORBIDDEN_BUILTINS are removed.

    Args:
        config: The sandbox configuration (authorized imports and
            allowed directories).

    Returns:
        A dict to be used as __builtins__ in the sandbox namespace.
    """
    safe_builtins = dict(builtins.__dict__)

    # Override import and file access with policy-enforcing wrappers
    safe_builtins["__import__"] = make_safe_import(
        config.authorized_imports, builtins.__import__
    )
    safe_builtins["open"] = make_safe_open(
        config.allowed_directories, builtins.open
    )

    # Remove dangerous builtins (pop: some, like help/exit, may be absent)
    for name in FORBIDDEN_BUILTINS:
        safe_builtins.pop(name, None)

    return safe_builtins


def final_answer(answer: Any) -> None:
    """Signal that the agent has finished, with its final answer.

    Always injected in the sandbox namespace (it is not an MCP tool).

    Args:
        answer: The final answer (converted to str by FinalAnswerException).

    Raises:
        FinalAnswerException: Always, to stop execution and hand the
            answer to the agent loop.
    """
    raise FinalAnswerException(answer)


def build_namespace(
    config: SandboxConfig,
    tools: dict[str, Callable[..., Any]] | None = None,
) -> dict[str, Any]:
    """Build the namespace in which sandboxed code is executed.

    The namespace contains the restricted builtins, final_answer and one
    entry per MCP tool wrapper. Code run with exec(code, namespace) can
    only reach names present in it.

    Args:
        config: The sandbox configuration used to build the builtins.
        tools: MCP tool wrappers, mapping tool name to a Python callable.

    Returns:
        A dict to be passed as globals to exec().

    Raises:
        ValueError: If a tool name collides with a reserved name.
    """
    namespace: dict[str, Any] = {
        "__builtins__": build_safe_builtins(config),
        "__name__": "__main__",
        "final_answer": final_answer,
    }

    for name, tool in (tools or {}).items():
        # An MCP server must not be able to replace final_answer or builtins
        if name in namespace:
            raise ValueError(
                f"MCP tool name '{name}' collides with a reserved sandbox name."
            )
        namespace[name] = tool

    return namespace


def current_memory_bytes() -> int:
    """Return the virtual memory currently used by this process.

    Reads /proc/self/statm on Linux, and falls back to resource.getrusage
    on macOS / systems without /proc.

    Returns:
        The virtual memory size of the current process, in bytes.
    """
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[0])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS ru_maxrss is in bytes, on Linux ru_maxrss is in kilobytes
        if sys.platform == "darwin":
            return rusage.ru_maxrss
        return rusage.ru_maxrss * 1024


def _auto_print_expressions(source: str) -> str:
    """Wrap top-level bare expressions to print their return value if not None.

    Simulates REPL behavior so when the LLM writes `read_file(...)` or `search_code(...)`
    without wrapping in print(), the result is still captured in stdout and shown in the Observation.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    new_body = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr):
            val = stmt.value
            # Don't wrap if already a print(...) or final_answer(...) call
            if isinstance(val, ast.Call):
                func = val.func
                if isinstance(func, ast.Name) and func.id in ("print", "final_answer"):
                    new_body.append(stmt)
                    continue

            assign = ast.Assign(
                targets=[ast.Name(id="__eval_result__", ctx=ast.Store())],
                value=val,
            )
            ast.fix_missing_locations(assign)

            check = ast.If(
                test=ast.Compare(
                    left=ast.Name(id="__eval_result__", ctx=ast.Load()),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ),
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="print", ctx=ast.Load()),
                            args=[ast.Name(id="__eval_result__", ctx=ast.Load())],
                            keywords=[],
                        )
                    )
                ],
                orelse=[],
            )
            ast.fix_missing_locations(check)
            new_body.extend([assign, check])
        else:
            new_body.append(stmt)

    tree.body = new_body
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return source


def execute(
    code: str,
    config: SandboxConfig,
    tools: dict[str, Callable[..., Any]] | None = None,
) -> dict:
    """Execute untrusted code inside the sandbox namespace.

    While the code runs, the network is blocked, a timeout is armed and
    the memory usage is capped. All these restrictions are removed once
    execution ends, whatever the outcome.

    Args:
        code: The Python source code to execute.
        config: The sandbox configuration (imports, directories,
            timeout and memory limits).
        tools: MCP tool wrappers, mapping tool name to a Python callable.

    Returns:
        A dict with three keys:
            - "output": everything the code printed to stdout.
            - "final_answer": the answer passed to final_answer(),
              or None if it was not called.
            - "error": a "ErrorType: message" string if the code raised
              an exception, or None otherwise.
    """
    namespace = build_namespace(config, tools)
    output = io.StringIO()
    answer = None
    error = None

    def on_timeout(signum: int, frame: Any) -> None:
        """Interrupt the running code when the alarm fires.

        Args:
            signum: The signal number (SIGALRM), required by signal.signal.
            frame: The current stack frame, required by signal.signal.

        Raises:
            SandboxTimeoutError: Always.
        """
        raise SandboxTimeoutError(
            f"Execution exceeded {config.max_execution_time_seconds} seconds."
        )

    signal.signal(signal.SIGALRM, on_timeout)
    signal.alarm(config.max_execution_time_seconds)  # start the timer

    # Allow the code max_memory_mb on top of what the process already uses
    old_limit = None
    try:
        old_limit = resource.getrlimit(resource.RLIMIT_AS)
        new_soft = current_memory_bytes() + config.max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, old_limit[1]))
    except (ValueError, OSError):
        pass

    network = NetworkBlocker()
    network.block()  # no network access while untrusted code runs

    # Enforce AST-level security restrictions before execution
    try:
        parsed_tree = ast.parse(code)
        SecurityASTValidator().visit(parsed_tree)
    except SyntaxError:
        pass
    except PermissionError as e:
        signal.alarm(0)
        if old_limit is not None:
            try:
                resource.setrlimit(resource.RLIMIT_AS, old_limit)
            except (ValueError, OSError):
                pass
        network.unblock()
        return {"output": "", "final_answer": None, "error": f"PermissionError: {e}"}

    executable_code = _auto_print_expressions(code)

    try:
        with contextlib.redirect_stdout(output):
            exec(executable_code, namespace)
    except FinalAnswerException as e:
        answer = e.answer
    except MemoryError:
        error = f"MemoryError: Execution exceeded {config.max_memory_mb} MB."
    except SandboxTimeoutError:
        error = (
            f"SandboxTimeoutError: Execution stopped after "
            f"{config.max_execution_time_seconds} seconds. The output above is "
            f"partial: it only contains what was printed before the stop. "
            f"Look for an infinite loop or a too slow algorithm."
        )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"  # e.g. "ZeroDivisionError: division by zero"
    finally:
        signal.alarm(0)  # always stop the timer so it cannot fire later
        if old_limit is not None:
            try:
                resource.setrlimit(resource.RLIMIT_AS, old_limit)
            except (ValueError, OSError):
                pass
        network.unblock()  # restore network for the agent

    text = output.getvalue()

    if len(text) > MAX_OUTPUT_CHARS:
        text = (
            f"{text[: MAX_OUTPUT_CHARS // 2]}"
            + f"\n... [output truncated: {len(text)} characters in total, only the first and last {MAX_OUTPUT_CHARS // 2} are shown] ...\n"
            + f"{text[-MAX_OUTPUT_CHARS // 2 :]}"
        )

    return {"output": text, "final_answer": answer, "error": error}


# Test
# if __name__ == "__main__":
#     config = SandboxConfig(max_execution_time_seconds=2)
#     print(execute("print('start')\nwhile True:\n    pass", config))
#     print(execute("x = bytearray(2 * 1024 * 1024 * 1024)", config))  # 2 GB
#     print(
#         execute("x = bytearray(100 * 1024 * 1024)\nprint('ok')", config)
#     )  # 100 MB
#     config = SandboxConfig(authorized_imports=["socket"])
#     print(
#         execute(
#             "import socket\nsocket.create_connection(('example.com', 80))",
#             config,
#         )
#     )
