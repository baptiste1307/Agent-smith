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
    code: str, test_list: list[str], test_imports: list[str] = []
) -> str:  # noqa: B006
    """Run a Python code snippet against a set of MBPP test cases."""
    return run_tests_mbpp(code, test_list, test_imports)


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
