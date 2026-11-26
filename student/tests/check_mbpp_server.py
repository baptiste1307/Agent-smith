import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters  # type: ignore[import-not-found]

SERVER = Path(__file__).resolve().parents[2] / "mcp_tools_mbpp.py"
TESTS = ["assert add(1, 2) == 3"]


async def main():
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with Client(params) as client:
        tools = (await client.list_tools()).tools
        tool = next(t for t in tools if t.name == "run_tests")
        print(
            "description:",
            len(tool.description or ""),
            "chars (sent to the LLM every turn)",
        )
        print("schema:", json.dumps(tool.input_schema))

        print("plain model_dump keys   :", sorted(tool.model_dump()))
        print(
            "by_alias model_dump keys:",
            sorted(tool.model_dump(by_alias=True, mode="json")),
        )

        for label, code, expected in (
            ("good", "def add(a, b):\n    return a + b\n", True),
            ("bad", "def add(a, b):\n    return a - b\n", False),
        ):
            result = await client.call_tool(
                "run_tests", {"code": code, "test_list": TESTS}
            )
            payload = json.loads(result.content[0].text)
            print(f"[{label}] is_error={result.is_error} ->", payload)
            assert payload["success"] is expected

    print(
        "OK over stdio. Still to test by hand: --http, then "
        "Client('http://127.0.0.1:8000/mcp')."
    )


asyncio.run(main())
