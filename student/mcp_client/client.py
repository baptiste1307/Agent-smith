import asyncio
import concurrent.futures
import os
import shlex
import sys
import threading
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import (  # type: ignore[import-not-found]
    ClientSession,
    StdioServerParameters,
)
from pydantic import AnyUrl
from mcp.client.stdio import stdio_client  # type: ignore[import-not-found]

try:  # mcp >= 1.24
    from mcp.client.streamable_http import (
        streamable_http_client as _http_client,
    )  # type: ignore[import-not-found]
except ImportError:  # older mcp 1.x releases
    from mcp.client.streamable_http import (  # type: ignore[import-not-found]
        streamablehttp_client as _http_client,  # type: ignore[assignment]
    )


# Sandbox functions giving access to the server resources and prompts
RESOURCE_HELPER = "read_mcp_resource"
PROMPT_HELPER = "get_mcp_prompt"


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first existing attribute among names (camelCase in mcp 1.x,
    snake_case in later releases), so the client works with both."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


class MCPClient:
    """Keep a connection to an MCP server open, usable from normal code.

    The MCP library is asynchronous and its connection only lives inside an
    `async with` block. A background thread enters this block and waits
    there until close() is called, so the connection stays open during the
    whole agent loop.
    """

    def __init__(self, server: str, timeout_seconds: float = 30) -> None:
        """Connect to the MCP server and fetch its tools.

        Args:
            server: Either an URL (streamable HTTP transport), or a command
                that launches the server (stdio transport), for example
                "python mcp_tools_mbpp.py".
            timeout_seconds: Maximum time to wait for the connection.

        Raises:
            TimeoutError: If the server did not answer in time.
        """
        if server.startswith(("http://", "https://")):
            self._server: str | StdioServerParameters = server
        else:
            command, *args = shlex.split(server)
            if command in ("python", "python3"):
                command = sys.executable
            # Allow running `--mcp-stdio "python mcp_tools_mbpp.py"` from
            # student/
            # even though the MCP server scripts live at the repository root.
            repo_root = Path(__file__).resolve().parents[2]
            resolved_args = []
            for arg in args:
                if (
                    arg.endswith(".py")
                    and not Path(arg).exists()
                    and (repo_root / arg).exists()
                ):
                    resolved_args.append(str(repo_root / arg))
                else:
                    resolved_args.append(arg)
            self._server = StdioServerParameters(
                command=command,
                args=resolved_args,
                env=dict(os.environ),
            )

        self._tools: list[Any] = []
        self._resources: list[Any] = []
        self._prompts: list[Any] = []
        self._ready = threading.Event()  # set by the thread once connected
        # Optional time.perf_counter() value: tool calls stop waiting
        # for the server once it is reached (set by the agent loop)
        self.deadline: float | None = None

        # The background thread runs its own event loop, forever
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()

        # Ask the background loop to open the connection and wait in it
        self._session_future = asyncio.run_coroutine_threadsafe(
            self._hold_connection(), self._loop
        )
        self._ready.wait(timeout_seconds)

        if self._session_future.done():
            self._session_future.result()  # re-raises the connection error
        if not self._ready.is_set():
            self.close()
            raise TimeoutError(f"MCP server did not answer: {server}")

    async def _hold_connection(self) -> None:
        """Open the connection, fetch the tools, then wait for close().

        Runs in the background thread. Leaving the `async with` block is
        what closes the connection.
        """
        self._stop = asyncio.Event()
        async with AsyncExitStack() as stack:
            if isinstance(self._server, str):  # streamable HTTP transport
                streams = await stack.enter_async_context(
                    _http_client(self._server)
                )
            else:  # stdio transport: the server is a child process
                streams = await stack.enter_async_context(
                    stdio_client(self._server)
                )
            client = await stack.enter_async_context(
                ClientSession(streams[0], streams[1])
            )
            await client.initialize()
            self._client = client
            self._tools = (await client.list_tools()).tools
            try:
                self._resources = (await client.list_resources()).resources
            except Exception:
                self._resources = []
            try:
                self._prompts = (await client.list_prompts()).prompts
            except Exception:
                self._prompts = []
            self._ready.set()  # tell __init__ we are connected
            await self._stop.wait()  # keep the connection open

    def close(self) -> None:
        """Close the connection and stop the background thread."""
        if self._ready.is_set():
            self._loop.call_soon_threadsafe(self._stop.set)
            self._session_future.result()  # wait for the block to exit
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def _wait(self, future: concurrent.futures.Future, what: str) -> Any:
        """Wait for a server answer, at most until self.deadline.

        Args:
            future: The pending request, running in the background loop.
            what: Name of the request, for the error message.

        Raises:
            TimeoutError: If the deadline is reached first.
        """
        wait: float | None = None
        if self.deadline is not None:
            wait = max(0.0, self.deadline - time.perf_counter())
        try:
            return future.result(timeout=wait)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"{what} did not answer before the agent's time budget "
                "ran out. Submit your best answer now."
            ) from None

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the server and wait for its answer.

        The call runs in the background loop, since the connection lives
        there. This method blocks until the server answers.

        Args:
            name: The name of the tool.
            arguments: The arguments of the tool, by parameter name.

        Returns:
            The text answered by the tool.

        Raises:
            RuntimeError: If the tool reported an error.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._client.call_tool(name, arguments), self._loop
        )
        result = self._wait(future, f"Tool '{name}'")

        text = "\n".join(
            getattr(content, "text", str(content))
            for content in result.content
        )
        if _field(result, "isError", "is_error", default=False):
            raise RuntimeError(f"Tool '{name}' failed: {text}")
        return text

    def get_schemas(self) -> list[dict[str, Any]]:
        """Describe the server tools, in the format used by the prompt.

        Server resources and prompts are described too, as the two helper
        functions of get_wrappers() that give access to them.

        Returns:
            One dict per tool, with its "name", "description" and
            "inputSchema" (the JSON Schema of its parameters).
        """
        schemas = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": _field(
                    tool, "inputSchema", "input_schema", default={}
                ),
            }
            for tool in self._tools
        ]
        return schemas + self._helper_schemas()

    def _helper_schemas(self) -> list[dict[str, Any]]:
        """Schemas of the resource / prompt helpers, built from the server.

        A helper is only described when the server exposes something it
        gives access to, and when no server tool already uses its name.
        """
        tool_names = {tool.name for tool in self._tools}
        schemas: list[dict[str, Any]] = []
        resources = self.get_resources()
        if resources and RESOURCE_HELPER not in tool_names:
            listing = "\n".join(
                f"- {r['uri']}: {r['description'] or r['name']}"
                for r in resources
            )
            schemas.append(
                {
                    "name": RESOURCE_HELPER,
                    "description": (
                        "Read an MCP resource of the server. Available:\n"
                        f"{listing}"
                    ),
                    "inputSchema": {
                        "properties": {"uri": {"type": "string"}},
                        "required": ["uri"],
                    },
                }
            )
        prompts = self.get_prompts()
        if prompts and PROMPT_HELPER not in tool_names:
            listing = "\n".join(
                f"- {p['name']}: {p['description']}" for p in prompts
            )
            schemas.append(
                {
                    "name": PROMPT_HELPER,
                    "description": (
                        "Get an MCP prompt of the server, filled with "
                        f"arguments (dict of str). Available:\n{listing}"
                    ),
                    "inputSchema": {
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object", "default": {}},
                        },
                        "required": ["name"],
                    },
                }
            )
        return schemas

    def _make_wrapper(
        self, name: str, param_names: list[str]
    ) -> Callable[..., str]:
        """Create a normal Python function that calls one MCP tool.

        The returned function remembers `name` and `param_names` (it is a
        closure), so each tool gets its own wrapper. It accepts arguments
        by position or by name, like a normal Python function.

        Args:
            name: The name of the tool the wrapper will call.
            param_names: The tool parameter names, in the order of its
                signature, used to name the positional arguments.

        Returns:
            A function such as read_file("a.py", end_line=2), returning
            the tool answer.
        """

        def wrapper(*args: Any, **kwargs: Any) -> str:
            # args gathers the positional arguments, e.g. ("a.py",)
            # kwargs gathers the named arguments, e.g. {"end_line": 2}
            if len(args) > len(param_names):
                raise TypeError(
                    f"{name}() takes at most {len(param_names)} positional "
                    f"arguments ({', '.join(param_names)}) "
                    f"but {len(args)} were given."
                )

            # MCP only accepts named arguments: name the positional ones
            arguments = dict(zip(param_names, args, strict=False))
            for key, value in kwargs.items():
                if key in arguments:
                    raise TypeError(
                        f"{name}() got multiple values for argument '{key}'."
                    )
                arguments[key] = value

            return self._call_tool(name, arguments)

        return wrapper

    def get_resources(self) -> list[dict[str, Any]]:
        """Describe the resources exposed by the connected MCP server."""
        return [
            {
                "uri": str(res.uri),
                "name": res.name,
                "description": res.description or "",
            }
            for res in self._resources
        ]

    def read_resource(self, uri: str) -> str:
        """Read a resource by URI from the connected MCP server."""
        future = asyncio.run_coroutine_threadsafe(
            self._client.read_resource(AnyUrl(uri)), self._loop
        )
        result = self._wait(future, f"Resource '{uri}'")
        return "\n".join(getattr(c, "text", str(c)) for c in result.contents)

    def get_prompts(self) -> list[dict[str, Any]]:
        """Describe the prompts exposed by the connected MCP server."""
        return [
            {
                "name": p.name,
                "description": p.description or "",
            }
            for p in self._prompts
        ]

    def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> str:
        """Fetch a prompt template from the connected MCP server."""
        future = asyncio.run_coroutine_threadsafe(
            self._client.get_prompt(name, arguments or {}), self._loop
        )
        result = self._wait(future, f"Prompt '{name}'")
        parts = []
        for msg in result.messages:
            content = getattr(msg, "content", msg)
            parts.append(getattr(content, "text", str(content)))
        return "\n".join(parts)

    def get_wrappers(self) -> dict[str, Callable[..., str]]:
        """Create one wrapper per server tool, to put in the sandbox namespace.

        Returns:
            A dict mapping each tool name to its wrapper function.
        """
        wrappers = {}
        for tool in self._tools:
            # The schema keeps the parameters in the order of the signature
            schema = _field(tool, "inputSchema", "input_schema", default={})
            param_names = list((schema or {}).get("properties", {}))
            wrappers[tool.name] = self._make_wrapper(tool.name, param_names)

        # Resources and prompts, through the helpers of _helper_schemas()
        for schema in self._helper_schemas():
            if schema["name"] == RESOURCE_HELPER:
                wrappers[RESOURCE_HELPER] = self.read_resource
            elif schema["name"] == PROMPT_HELPER:
                wrappers[PROMPT_HELPER] = self.get_prompt

        return wrappers
