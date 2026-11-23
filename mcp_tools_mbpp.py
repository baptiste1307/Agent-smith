import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "student"))

from tools.mbpp_tools import run_tests_mbpp  # noqa: E402

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    sys.exit(
        f"Cannot import mcp v2 (MCPServer) with {sys.executable}. "
        "Try: cd student && uv run python ../mcp_tools_mbpp.py"
    )

mcp = MCPServer("agent-smith-mbpp")


@mcp.tool()
def run_tests(
    code: str,
    test_list: list[str],
    test_imports: list[str] | None = None,
) -> str:
    """Run a Python code snippet against a set of MBPP test cases."""
    return run_tests_mbpp(code, test_list, test_imports or [])


@mcp.resource("mbpp://info")
def mbpp_info() -> str:
    """Overview of the MBPP benchmark constraints and available tools."""
    return (
        "MBPP Benchmark MCP Server.\n"
        "Tools: run_tests(code, test_list, test_imports=[])\n"
        "Limits: max 10 iterations, 6,000 input tokens, "
        "1,500 output tokens, 120s timeout."
    )


@mcp.prompt()
def mbpp_workflow(task_definition: str = "") -> str:
    """Recommended step-by-step workflow for solving an MBPP task."""
    return (
        "Solve the MBPP task in 2 steps:\n"
        "1. Write the function and test it with run_tests("
        "code, test_list, test_imports).\n"
        "2. Once all assertions pass, call final_answer(code).\n"
        + (f"\nTask: {task_definition}" if task_definition else "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP server for the MBPP benchmark."
    )
    parser.add_argument(
        "--http", action="store_true", help="streamable HTTP instead of stdio"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.http:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
