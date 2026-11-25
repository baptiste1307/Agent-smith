"""Command line interface of the sandbox: an interactive REPL."""

import argparse
import ast
import codeop
import sys
from collections.abc import Iterator
from typing import Any

from agent.prompts import build_mcp_tools_manual
from mcp_client.client import MCPClient

from .sandbox import build_namespace, execute, load_config

EXIT_COMMANDS = frozenset({"exit", "exit()", "quit", "quit()"})


def split_statements(source: str) -> Iterator[str]:
    """Split a whole program into its top-level statements.

    Each statement keeps its full multi-line body (def, class, for, try,
    triple-quoted strings...). Statements sharing a line (`a = 1; b = 2`)
    stay together, and decorators stay with their function.

    Raises:
        SyntaxError: If the program does not parse.
    """
    lines = source.splitlines()
    chunks: list[list[int]] = []  # [first_line, last_line], 1-based
    for node in ast.parse(source).body:
        decorators = getattr(node, "decorator_list", [])
        start = min([node.lineno] + [d.lineno for d in decorators])
        end = node.end_lineno or node.lineno
        if chunks and start <= chunks[-1][1]:
            chunks[-1][1] = max(chunks[-1][1], end)
        else:
            chunks.append([start, end])
    for start, end in chunks:
        yield "\n".join(lines[start - 1:end])


def is_complete(source: str) -> bool:
    """Tell whether an interactive entry is complete, like the Python REPL.

    A compound statement (def, for, if...) stays open until an empty line.
    A syntax error counts as complete, so the error is shown at once.
    """
    try:
        return codeop.compile_command(source, "<input>", "single") is not None
    except (SyntaxError, ValueError, OverflowError):
        return True


def run_entry(
    code: str, config: Any, wrappers: dict, namespace: dict
) -> None:
    """Execute one entry in the sandbox and print what it produced."""
    result = execute(code, config, wrappers, namespace=namespace)
    if result["output"]:
        print(result["output"], end="")  # already ends with a newline
    if result["error"] is not None:
        print(f"Error: {result['error']}")
    if result["final_answer"] is not None:
        print(f"Final answer: {result['final_answer']}")
    sys.stdout.flush()


def run_script(
    source: str, config: Any, wrappers: dict, namespace: dict
) -> None:
    """Run code piped on stdin, one top-level statement at a time.

    Each statement gets the sandbox limits of its own (like one REPL
    entry): a statement that times out does not stop the next ones.
    """
    try:
        statements = list(split_statements(source))
    except SyntaxError:
        # Not a valid program: fall back to REPL-style entries, so the
        # valid statements still run and the faulty one reports its error
        run_lines(iter(source.splitlines()), config, wrappers, namespace)
        return
    for statement in statements:
        if statement.strip() in EXIT_COMMANDS:
            return
        run_entry(statement, config, wrappers, namespace)


def run_lines(
    lines: Iterator[str],
    config: Any,
    wrappers: dict,
    namespace: dict,
    prompt: bool = False,
) -> None:
    """REPL loop: gather lines into complete entries and execute them."""
    buffer: list[str] = []
    while True:
        try:
            if prompt:
                line = input("... " if buffer else ">>> ")
            else:
                line = next(lines)
        except (EOFError, StopIteration):  # Ctrl+D or end of input
            if buffer:
                run_entry("\n".join(buffer), config, wrappers, namespace)
            if prompt:
                print()
            return
        except KeyboardInterrupt:  # Ctrl+C: cancel the entry
            print("\nKeyboardInterrupt")
            buffer = []
            continue

        if not buffer and line.strip() in EXIT_COMMANDS:
            return
        if not buffer and not line.strip():
            continue

        buffer.append(line)
        source = "\n".join(buffer)
        if is_complete(source):
            run_entry(source, config, wrappers, namespace)
            buffer = []


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
        help=(
            "Command that launches an MCP server, "
            'e.g. "python mcp_tools_mbpp.py".'
        ),
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
    if client:
        resources = [r["uri"] for r in client.get_resources()]
        prompts = [p["name"] for p in client.get_prompts()]
        if resources:
            print(f"Available MCP resources: {resources}")
        if prompts:
            print(f"Available MCP prompts: {prompts}")
    namespace = build_namespace(config, wrappers)
    # The manual given to the LLM, generated from the connected server's
    # tool schemas, is also readable from the REPL
    namespace["sandbox_manual"] = build_mcp_tools_manual(
        client.get_schemas() if client else []
    )
    sys.stdout.flush()

    try:
        if sys.stdin.isatty():
            run_lines(iter(()), config, wrappers, namespace, prompt=True)
        else:
            run_script(sys.stdin.read(), config, wrappers, namespace)
    finally:  # always close the MCP server, even on Ctrl+C or a crash
        if client:
            client.close()
