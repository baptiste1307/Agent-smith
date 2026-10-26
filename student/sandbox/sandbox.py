import builtins
from pathlib import Path
from typing import Any

import pydantic

from models_public import SandboxConfig
from sandbox.security import make_safe_import, make_safe_open

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
