"""Security rules and constraints for the execution sandbox."""

import os
import socket
import types
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
        name: Full module name (e.g. 'math', 'collections.abc').
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
            # Allow the base module itself ('collections') or any submodule
            # ('collections.abc')
            if name == base or name.startswith(base + "."):
                return True

    # If no rule matches, import is denied
    return False


# Dunder attributes that ordinary code needs and that do not lead to
# interpreter internals. Every other dunder (__self__, __globals__,
# __subclasses__, __dict__, __traceback__, __loader__, ...) is refused.
ALLOWED_DUNDER_ATTRS = frozenset(
    {
        "__init__",
        "__name__",
        "__qualname__",
        "__doc__",
        "__module__",
        "__class__",
        "__len__",
        "__iter__",
        "__next__",
        "__contains__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__str__",
        "__repr__",
        "__format__",
        "__hash__",
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__bool__",
        "__call__",
        "__enter__",
        "__exit__",
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__pow__",
        "__neg__",
        "__pos__",
        "__abs__",
        "__and__",
        "__or__",
        "__xor__",
        "__invert__",
        "__lshift__",
        "__rshift__",
    }
)

# Non-dunder attributes that expose frames, code objects or a way to
# evaluate a string as code.
FORBIDDEN_PLAIN_ATTRS = frozenset(
    {
        # traceback / frame objects -> f_globals of the host modules
        "tb_frame",
        "tb_next",
        "f_back",
        "f_globals",
        "f_locals",
        "f_builtins",
        "f_code",
        "f_trace",
        # generators / coroutines / async generators -> their frame
        "gi_frame",
        "gi_code",
        "gi_yieldfrom",
        "cr_frame",
        "cr_code",
        "cr_await",
        "ag_frame",
        "ag_code",
        "ag_await",
        # typing.ForwardRef evaluates an arbitrary string with eval()
        "evaluate",
        "_evaluate",
        "_eval_type",
    }
)


def is_attribute_forbidden(name: str) -> bool:
    """Return True if sandboxed code must not read or write this attribute.

    Args:
        name: The attribute name.

    Returns:
        True for every dunder attribute outside ALLOWED_DUNDER_ATTRS and for
        the frame / eval related attributes of FORBIDDEN_PLAIN_ATTRS.
    """
    if name.startswith("__") and name.endswith("__"):
        return name not in ALLOWED_DUNDER_ATTRS
    return name in FORBIDDEN_PLAIN_ATTRS


def check_attribute(name: Any) -> None:
    """Raise PermissionError if `name` is a forbidden attribute name.

    Dotted names (as accepted by operator.attrgetter) are checked part by
    part.
    """
    if not isinstance(name, str):
        return
    for part in name.split("."):
        if is_attribute_forbidden(part):
            raise PermissionError(
                f"Access to attribute '{part}' is forbidden by sandbox "
                "security policy."
            )


def make_safe_attr_builtins() -> dict[str, Callable[..., Any]]:
    """Return getattr / setattr / delattr versions that check the name.

    The AST validator only sees attribute names written literally in the
    code: these wrappers also cover names built at runtime, such as
    getattr(obj, "__sub" + "classes__").
    """
    real_getattr = getattr
    real_setattr = setattr
    real_delattr = delattr

    def safe_getattr(obj: Any, name: str, *default: Any) -> Any:
        check_attribute(name)
        return real_getattr(obj, name, *default)

    def safe_setattr(obj: Any, name: str, value: Any) -> None:
        check_attribute(name)
        real_setattr(obj, name, value)

    def safe_delattr(obj: Any, name: str) -> None:
        check_attribute(name)
        real_delattr(obj, name)

    return {
        "getattr": safe_getattr,
        "setattr": safe_setattr,
        "delattr": safe_delattr,
    }


def _safe_attrgetter(*names: str) -> Callable[..., Any]:
    """operator.attrgetter that refuses forbidden attribute names."""
    import operator

    for name in names:
        check_attribute(name)
    return operator.attrgetter(*names)


def _safe_methodcaller(name: str, *args: Any, **kwargs: Any) -> Any:
    """operator.methodcaller that refuses forbidden method names."""
    import operator

    check_attribute(name)
    return operator.methodcaller(name, *args, **kwargs)


# Public names of authorized modules that would give a way out of the
# sandbox, removed from the module seen by sandboxed code.
HIDDEN_MODULE_ATTRS: dict[str, frozenset[str]] = {
    # evaluate annotation strings with eval()
    "typing": frozenset(
        {"get_type_hints", "ForwardRef", "evaluate_forward_ref"}
    ),
    # Formatter.get_field resolves "0.__self__" with the real getattr
    "string": frozenset({"Formatter"}),
}

# Public names of authorized modules replaced by a checked version.
MODULE_OVERRIDES: dict[str, dict[str, Any]] = {
    "operator": {
        "attrgetter": _safe_attrgetter,
        "methodcaller": _safe_methodcaller,
    },
}


def make_module_proxy(
    module: types.ModuleType,
    authorized_imports: list[str],
    memo: dict[str, types.ModuleType] | None = None,
) -> types.ModuleType:
    """Build the restricted view of a module given to sandboxed code.

    The proxy is a new module object holding only:
    - public names (names starting with '_' are dropped, e.g. random._os,
      collections._sys, math.__loader__);
    - submodules that are themselves authorized (e.g. collections.abc with
      "collections.*"), wrapped the same way. Modules imported by the
      module for its own use (typing.sys, re.enum, json.codecs...) are
      dropped, as they would reach os / sys / unrestricted open().

    Args:
        module: The real module.
        authorized_imports: The import allowlist.
        memo: Proxies already built during this call (handles cycles).

    Returns:
        The restricted module.
    """
    memo = {} if memo is None else memo
    real_name = module.__name__
    if real_name in memo:
        return memo[real_name]

    proxy = types.ModuleType(real_name, getattr(module, "__doc__", None))
    memo[real_name] = proxy
    hidden = HIDDEN_MODULE_ATTRS.get(real_name, frozenset())

    for attr, value in list(vars(module).items()):
        if attr.startswith("_") or attr in hidden:
            continue
        if isinstance(value, types.ModuleType):
            if not is_import_allowed(value.__name__, authorized_imports):
                continue
            value = make_module_proxy(value, authorized_imports, memo)
        proxy.__dict__[attr] = value

    proxy.__dict__.update(MODULE_OVERRIDES.get(real_name, {}))
    return proxy


def make_safe_import(
    authorized_imports: list[str],
    original_import: Callable,
) -> Callable:
    """Create a restricted version of builtins.__import__.

    Intercepts every 'import' or 'from ... import ...' statement executed
    inside the sandbox, raises ImportError if the module is not authorized,
    and returns a restricted proxy of the module (see make_module_proxy).

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
        # 1. Verify that the requested module is authorized
        if level != 0 or not is_import_allowed(name, authorized_imports):
            raise ImportError(
                f"Import of module '{name}' is not authorized by sandbox "
                "policy."
            )

        # 2. Delegate to original import
        mod = original_import(name, globals, locals, fromlist, level)

        # 3. 'from X import a, b': private names are refused, and imported
        # submodules must be authorized too. Checked on the real module:
        # if a name is missing from the proxy, Python falls back to
        # sys.modules["X.a"], which would bypass the proxy.
        for item in fromlist or ():
            if not isinstance(item, str) or item == "*":
                continue
            if item.startswith("_"):
                raise ImportError(
                    f"Import of private name '{item}' from '{name}' is not "
                    "authorized by sandbox policy."
                )
            val = getattr(mod, item, None)
            if isinstance(val, types.ModuleType) and not is_import_allowed(
                val.__name__, authorized_imports
            ):
                raise ImportError(
                    f"Import of submodule '{name}.{item}' is not authorized "
                    "by sandbox policy."
                )

        return make_module_proxy(mod, authorized_imports)

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
        True if path is inside (or is) an allowed directory, else False.
    """
    if not allowed_directories:
        return False

    try:
        real_target = os.path.realpath(os.path.abspath(str(filepath)))
        for allowed in allowed_directories:
            real_allowed = os.path.realpath(os.path.abspath(str(allowed)))
            # Target must either be the directory itself or a child path inside
            # it
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
                "Access to arbitrary file descriptors "
                "is denied by sandbox policy."
            )

        # Remap /testbed/... to host TESTBED_PATH if set and /testbed is
        # allowed
        effective_allowed = list(allowed_directories)
        if isinstance(file, (str, Path)):
            f_str = str(file).strip()
            testbed_env = os.environ.get("TESTBED_PATH")
            if testbed_env and is_path_allowed(
                "/testbed", allowed_directories
            ):
                effective_allowed.append(testbed_env)
            if testbed_env and (
                f_str in ("/testbed", "/testbed/")
                or f_str.startswith("/testbed/")
            ):
                rel = (
                    f_str[len("/testbed/"):]
                    if f_str.startswith("/testbed/")
                    else ""
                )
                file = str(Path(testbed_env) / rel)

        # Validate filesystem path against allowed directories
        if not is_path_allowed(file, effective_allowed):
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


class FinalAnswerException(Exception):
    """Signals that final_answer(answer) was called by the agent.

    It stops the executed code at the final_answer() call. The answer is
    recorded by the sandbox *before* this exception is raised, so code
    that catches it (`try: ... except Exception:`) cannot lose the answer:
    it only keeps running after the call.
    """

    def __init__(self, answer: str) -> None:
        self.answer = str(answer)
        super().__init__(self.answer)


class NetworkBlocker:
    """Temporarily disables network socket creation during sandbox execution.

    Why temporary blocking is needed:
    1. The main agent interacts with LLM APIs over the Internet.
    2. Before untrusted LLM code execution: block network (block()).
    3. LLM code executes in isolation.
    4. After execution: restore network (unblock()).
    """

    def __init__(self) -> None:
        self._orig_socket: Any = socket.socket
        self._orig_create_connection = getattr(
            socket, "create_connection", None
        )
        self._orig_getaddrinfo = getattr(socket, "getaddrinfo", None)
        self._orig_socketpair = getattr(socket, "socketpair", None)
        self._orig_fromfd = getattr(socket, "fromfd", None)

    def block(self) -> None:
        """Block all socket creation and connection attempts."""

        def disabled_socket(*args: Any, **kwargs: Any) -> Any:
            raise SandboxSecurityError(
                "Network access is disabled by sandbox policy."
            )

        socket.socket = disabled_socket  # type: ignore[misc,assignment]
        if self._orig_create_connection:
            # type: ignore[assignment]
            socket.create_connection = disabled_socket
        if self._orig_getaddrinfo:
            socket.getaddrinfo = disabled_socket  # type: ignore[assignment]
        if self._orig_socketpair:
            socket.socketpair = disabled_socket  # type: ignore[assignment]
        if self._orig_fromfd:
            socket.fromfd = disabled_socket  # type: ignore[assignment]

    def unblock(self) -> None:
        """Restore original socket functions."""
        socket.socket = self._orig_socket  # type: ignore[misc,assignment]
        if self._orig_create_connection:
            # type: ignore[misc]
            socket.create_connection = self._orig_create_connection
        if self._orig_getaddrinfo:
            # type: ignore[assignment]
            socket.getaddrinfo = self._orig_getaddrinfo
        if self._orig_socketpair:
            # type: ignore[assignment]
            socket.socketpair = self._orig_socketpair
        if self._orig_fromfd:
            socket.fromfd = self._orig_fromfd  # type: ignore[assignment]
