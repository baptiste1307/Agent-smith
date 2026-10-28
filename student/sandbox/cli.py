"""Command line interface of the sandbox: an interactive REPL."""

import argparse

from mcp_client.client import MCPClient

from .sandbox import load_config, execute


def main() -> None:
    """Entry point of `uv run sandbox`."""
    parser = argparse.ArgumentParser(
        description="Interactive sandbox: run Python code with the sandbox "
        "restrictions and the tools of an MCP server."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to the sandbox configuration JSON file.",
    )
    parser.add_argument(
        "--mcp-stdio",
        default=None,
        help='Command that launches an MCP server, e.g. "python mcp_tools_mbpp.py".',
    )
    parser.add_argument(
        "--mcp-server",
        default=None,
        help="URL of an MCP server (streamable HTTP transport).",
    )
    args = parser.parse_args()

    if args.mcp_stdio and args.mcp_server:
        parser.error("use either --mcp-stdio or --mcp-server, not both.")

    config = load_config(args.config)

    # Connect to the MCP server, if one is given
    server = args.mcp_stdio or args.mcp_server
    client = MCPClient(server) if server else None
    wrappers = client.get_wrappers() if client else {}

    print(f"Available tools: {list(wrappers)}")

    try:
        while True:
            try:
                # print >>>, wait for the user to write then press enter, then read the input
                code = input(">>> ")
            except EOFError:  # Ctrl+D: quit
                print()
                break
            except KeyboardInterrupt:  # Ctrl+C: cancel the line, show the prompt again
                print()
                continue

            if code.strip() == "exit":
                break

            result = execute(code, config, wrappers)

            if result["output"]:
                print(result["output"], end="")  # already ends with a newline
            if result["error"] is not None:
                print(f"Error: {result['error']}")
            if result["final_answer"] is not None:
                print(f"Final answer: {result['final_answer']}")
    finally:  # always close the MCP server, even on Ctrl+C or a crash
        if client:
            client.close()
