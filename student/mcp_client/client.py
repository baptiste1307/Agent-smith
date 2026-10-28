"""MCP client: connects to an MCP server and exposes its tools synchronously."""

import asyncio
import shlex
import threading
from typing import Any

from mcp import Client, StdioServerParameters


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
            self._server = StdioServerParameters(command=command, args=args)

        self._tools: list[Any] = []
        self._ready = threading.Event()  # set by the thread once connected

        # The background thread runs its own event loop, forever
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
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
        async with Client(self._server) as client:
            self._client = client
            self._tools = (await client.list_tools()).tools
            self._ready.set()  # tell __init__ we are connected
            await self._stop.wait()  # keep the connection open

    def close(self) -> None:
        """Close the connection and stop the background thread."""
        if self._ready.is_set():
            self._loop.call_soon_threadsafe(self._stop.set)
            self._session_future.result()  # wait for the block to exit
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
