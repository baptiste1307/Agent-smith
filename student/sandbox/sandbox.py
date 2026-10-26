import builtins
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pydantic

from models_public import SandboxConfig
from sandbox.security import (
    FinalAnswerException,
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
