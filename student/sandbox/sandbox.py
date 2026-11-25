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
    check_attribute,
    make_safe_attr_builtins,
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
    "locals",
    "vars",
    "help",
    "exit",
    "quit",
    "copyright",
    "credits",
    "license",
)

# Builtins dunder names that sandboxed code must not reach: __loader__
# (BuiltinImporter) and __spec__ can load built-in modules such as posix
FORBIDDEN_BUILTIN_DUNDERS = ("__loader__", "__spec__", "__package__")

# Dunder names that sandboxed code may write as plain names
ALLOWED_DUNDER_NAMES = frozenset({"__name__"})

# Context-manager exit methods: an __exit__ returning True would swallow
# the timeout. Sandboxed code may not define them.
FORBIDDEN_METHOD_NAMES = frozenset({"__exit__", "__aexit__"})

# Period at which the timeout is raised again if the code keeps running
TIMEOUT_RETRY_SECONDS = 0.2


class SecurityASTValidator(ast.NodeVisitor):
    """Enforce sandbox security rules at AST level.

    Rejects, before execution:
    - forbidden attributes written literally (obj.__self__, e.f_back...);
    - getattr / setattr / delattr / hasattr with a forbidden constant name;
    - dunder names such as __builtins__ or __import__.
    Names built at runtime are checked by the safe getattr family and the
    module proxies (see sandbox.security).
    """

    def visit_Attribute(self, node: ast.Attribute) -> None:
        check_attribute(node.attr)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            node.id.startswith("__")
            and node.id.endswith("__")
            and node.id not in ALLOWED_DUNDER_NAMES
        ):
            raise PermissionError(
                f"Access to name '{node.id}' is forbidden by sandbox "
                "security policy."
            )
        # The sandbox timeout is a BaseException: catching it is refused
        if node.id == "BaseException":
            raise PermissionError(
                "Using 'BaseException' is forbidden by sandbox security "
                "policy; catch Exception instead."
            )
        self.generic_visit(node)

    def _check_function_name(self, name: str) -> None:
        if name in FORBIDDEN_METHOD_NAMES:
            raise PermissionError(
                f"Defining '{name}' is forbidden by sandbox security policy."
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function_name(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function_name(node.name)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # Catches type("C", (), {"__exit__": ...})
        if isinstance(node.value, str):
            self._check_function_name(node.value)

    def visit_MatchClass(self, node: ast.MatchClass) -> None:
        # `case X(attr=p)` reads obj.attr with the real getattr: without
        # this check, `case object(__globals__=g)` escapes the sandbox
        for attr in node.kwd_attrs:
            check_attribute(attr)
        # Positional sub-patterns read the attributes named in the class's
        # __match_args__, which a class built with type() can set to any
        # dunder. The class name is not trusted either (`int = Evil` would
        # pass a name allowlist), so positional sub-patterns are refused.
        if node.patterns:
            raise PermissionError(
                "Positional class patterns (case C(x)) are forbidden by "
                "sandbox security policy; use keyword patterns such as "
                "case C(attr=x), or case int() as x."
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in (
            "getattr",
            "setattr",
            "delattr",
            "hasattr",
        ):
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                attr_value = node.args[1].value
                if isinstance(attr_value, bytes):
                    attr_value = attr_value.decode("utf-8", "replace")
                check_attribute(attr_value)
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
    # getattr / setattr / delattr check names built at runtime
    safe_builtins.update(make_safe_attr_builtins())

    # Remove dangerous builtins (pop: some, like help/exit, may be absent)
    for name in FORBIDDEN_BUILTINS + FORBIDDEN_BUILTIN_DUNDERS:
        safe_builtins.pop(name, None)

    return safe_builtins


def final_answer(answer: Any) -> None:
    """Signal that the agent has finished, with its final answer.

    Always injected in the sandbox namespace (it is not an MCP tool).

    Args:
        answer: The final answer (converted to str by FinalAnswerException).

    Raises:
        FinalAnswerException: Always, to stop execution and hand the
            answer to the agent loop. execute() replaces this function by
            a version that also records the answer before raising.
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
                f"MCP tool name '{name}' collides with "
                "a reserved sandbox name."
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
        sysconf = getattr(os, "sysconf")
        return pages * sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        getrusage = getattr(resource, "getrusage")
        usage_self = getattr(resource, "RUSAGE_SELF")
        rusage = getrusage(usage_self)
        # On macOS ru_maxrss is in bytes, on Linux ru_maxrss is in kilobytes
        if sys.platform == "darwin":
            return rusage.ru_maxrss
        return rusage.ru_maxrss * 1024


def _auto_print_expressions(source: str) -> str:
    """Wrap top-level bare expressions to print their return value if not None.

    Simulates REPL behavior so when LLM writes `read_file(...)` or
    `search_code(...)` without print(), the result is still captured.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Bare `except:` would catch the sandbox timeout (a BaseException):
    # it is executed as `except Exception:`
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            node.type = ast.Name(id="Exception", ctx=ast.Load())

    new_body: list[ast.stmt] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr):
            val = stmt.value
            # A bare string is a docstring or a comment, not a result
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                new_body.append(stmt)
                continue
            # Don't wrap if already a print(...) or final_answer(...) call
            if isinstance(val, ast.Call):
                func = val.func
                if isinstance(func, ast.Name) and func.id in (
                    "print",
                    "final_answer",
                ):
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
                            args=[
                                ast.Name(id="__eval_result__", ctx=ast.Load())
                            ],
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
    namespace: dict[str, Any] | None = None,
) -> dict:
    """Execute untrusted code inside the sandbox namespace.

    While the code runs, the network is blocked, a timeout is armed and
    the memory usage is capped. All these restrictions are removed once
    execution ends, whatever the outcome. MCP tool wrappers temporarily
    suspend sandbox-only restrictions (timeout, network, RLIMIT_AS) while
    executing outside the sandbox security domain.

    Args:
        code: The Python source code to execute.
        config: The sandbox configuration (imports, directories,
            timeout and memory limits).
        tools: MCP tool wrappers, mapping tool name to a Python callable.
        namespace: Optional persistent namespace (e.g. for interactive REPL).
            When None, a fresh namespace is created.

    Returns:
        A dict with three keys:
            - "output": everything the code printed to stdout.
            - "final_answer": the answer passed to final_answer(),
              or None if it was not called.
            - "error": a "ErrorType: message" string if the code raised
              an exception, or None otherwise.
    """
    output = io.StringIO()
    answer = None
    error = None
    # False as soon as exec() returns: a late timer tick is then ignored
    running = True

    def on_timeout(signum: int, frame: Any) -> None:
        """Interrupt the running code when the timer fires.

        SandboxTimeoutError is a BaseException, so `except Exception`
        does not catch it, and bare `except:` is rewritten before
        execution. The timer is re-armed with a short period, so code
        that still manages to keep running (a loop in a `finally` block,
        nested handlers...) is interrupted again until it unwinds.

        Args:
            signum: The signal number (SIGALRM), required by signal.signal.
            frame: The current stack frame, required by signal.signal.

        Raises:
            SandboxTimeoutError: While the sandboxed code is running.
        """
        if not running:
            return
        signal.setitimer(signal.ITIMER_REAL, TIMEOUT_RETRY_SECONDS)
        raise SandboxTimeoutError(
            f"Execution exceeded {config.max_execution_time_seconds} seconds."
        )

    old_limit: tuple[int, int] | None = None
    new_soft: int | None = None
    network = NetworkBlocker()

    def _wrap_external_tool(tool_fn: Callable[..., Any]) -> Callable[..., Any]:
        """Suspend sandbox-only limits during external MCP tool execution."""

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            # setitimer keeps sub-second precision (alarm() rounds the
            # remaining time, and 0 would disarm the timeout for good)
            remaining = signal.setitimer(signal.ITIMER_REAL, 0)[0]
            network.unblock()
            if old_limit is not None:
                try:
                    setrlimit = getattr(resource, "setrlimit")
                    rlimit_as = getattr(resource, "RLIMIT_AS")
                    setrlimit(rlimit_as, old_limit)
                except (ValueError, OSError):
                    pass
            try:
                return tool_fn(*args, **kwargs)
            finally:
                if old_limit is not None and new_soft is not None:
                    try:
                        # Recompute relative to current RSS/VM to avoid
                        # breaking after tool call
                        updated_soft = max(
                            new_soft,
                            current_memory_bytes()
                            + config.max_memory_mb * 1024 * 1024,
                        )
                        setrlimit = getattr(resource, "setrlimit")
                        rlimit_as = getattr(resource, "RLIMIT_AS")
                        setrlimit(rlimit_as, (updated_soft, old_limit[1]))
                    except (ValueError, OSError):
                        pass
                network.block()
                if remaining > 0:
                    signal.setitimer(signal.ITIMER_REAL, remaining)

        return _wrapped

    wrapped_tools = (
        {name: _wrap_external_tool(fn) for name, fn in tools.items()}
        if tools
        else None
    )

    if namespace is None:
        exec_namespace = build_namespace(config, wrapped_tools)
    else:
        exec_namespace = namespace
        if wrapped_tools:
            for name, fn in wrapped_tools.items():
                if name not in ("__builtins__", "__name__", "final_answer"):
                    exec_namespace[name] = fn

    # final_answer() records its argument here before raising, so the
    # answer survives even if the code catches the exception
    answer_box: list[str] = []

    def recording_final_answer(answer: Any) -> None:
        """Record the final answer (first call wins), then stop the code."""
        if not answer_box:
            answer_box.append(str(answer))
        raise FinalAnswerException(answer_box[0])

    recording_final_answer.__name__ = "final_answer"
    exec_namespace["final_answer"] = recording_final_answer

    # Enforce AST-level security restrictions before execution
    try:
        parsed_tree = ast.parse(code)
        SecurityASTValidator().visit(parsed_tree)
    except SyntaxError:
        pass
    except PermissionError as e:
        return {
            "output": "",
            "final_answer": None,
            "error": f"PermissionError: {e}",
        }

    sigalrm = getattr(signal, "SIGALRM")
    alarm = getattr(signal, "alarm")
    previous_handler = signal.signal(sigalrm, on_timeout)
    alarm(config.max_execution_time_seconds)  # start the timer

    # Allow the code max_memory_mb on top of what the process already uses
    try:
        getrlimit = getattr(resource, "getrlimit")
        setrlimit = getattr(resource, "setrlimit")
        rlimit_as = getattr(resource, "RLIMIT_AS")
        old_limit = getrlimit(rlimit_as)
        new_soft = current_memory_bytes() + config.max_memory_mb * 1024 * 1024
        setrlimit(rlimit_as, (new_soft, old_limit[1]))
    except (ValueError, OSError):
        pass

    network.block()  # no network access while untrusted code runs

    executable_code = _auto_print_expressions(code)

    try:
        try:
            with contextlib.redirect_stdout(output):
                exec(executable_code, exec_namespace)
        finally:
            # Inside the outer try: a tick landing here is still caught
            # by `except SandboxTimeoutError` below
            running = False
            alarm(0)
    except FinalAnswerException as e:
        answer = e.answer
    except MemoryError:
        error = f"MemoryError: Execution exceeded {config.max_memory_mb} MB."
    except SandboxTimeoutError:
        error = (
            f"SandboxTimeoutError: Execution stopped after "
            f"{config.max_execution_time_seconds} seconds. "
            "The output above is "
            f"partial: it only contains what was printed before the stop. "
            f"Look for an infinite loop or a too slow algorithm."
        )
    except Exception as e:
        # e.g. "ZeroDivisionError: division by zero"
        error = f"{type(e).__name__}: {e}"
    finally:
        alarm(0)  # always stop the timer so it cannot fire later
        signal.signal(
            sigalrm,
            previous_handler
            if previous_handler is not None
            else signal.SIG_DFL,
        )
        if old_limit is not None:
            try:
                setrlimit = getattr(resource, "setrlimit")
                rlimit_as = getattr(resource, "RLIMIT_AS")
                setrlimit(rlimit_as, old_limit)
            except (ValueError, OSError):
                pass
        network.unblock()  # restore network for the agent

    if answer_box:  # also set when the code caught FinalAnswerException
        answer = answer_box[0]

    text = output.getvalue()

    if len(text) > MAX_OUTPUT_CHARS:
        text = (
            f"{text[: MAX_OUTPUT_CHARS // 2]}"
            + (
                "\n... [output truncated: "
                f"{len(text)} characters in total, only the first and last "
                f"{MAX_OUTPUT_CHARS // 2} are shown] ...\n"
            )
            + f"{text[-MAX_OUTPUT_CHARS // 2:]}"
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
