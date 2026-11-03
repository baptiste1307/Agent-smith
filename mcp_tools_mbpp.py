import argparse
import shlex
import sys
from pathlib import Path

from agent.loop import agent_loop
from agent.prompts import build_mbpp_task_prompt
from llm.metrics import MetricsTracker
from mcp_client.client import MCPClient
from models_public import MBPPTaskInput


def _default_mcp_server_command() -> str:
    """Return the command that starts the repository MBPP MCP server."""
    repo_root = Path(__file__).resolve().parents[1]
    server_path = repo_root / "mcp_tools_mbpp.py"
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(server_path))}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Agent Smith MBPP agent."
    )
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="liquid/lfm-2.5-2.6b:free")
    parser.add_argument(
        "--provider-url", default="https://openrouter.ai/api/v1"
    )
    parser.add_argument(
        "--mcp-server",
        default=None,
        help=(
            "MCP server URL or command. Defaults to the local "
            "mcp_tools_mbpp.py stdio server."
        ),
    )
    args = parser.parse_args()

    with open(args.task_file, encoding="utf-8") as f:
        task = MBPPTaskInput.model_validate_json(f.read())

    tracker = MetricsTracker(task_id=str(task.task_id), benchmark="mbpp")
    mcp_client: MCPClient | None = None

    try:
        mcp_client = MCPClient(
            server=args.mcp_server or _default_mcp_server_command()
        )
        answer = agent_loop(
            exercice=build_mbpp_task_prompt(task),
            mcp_tools_wrappers=mcp_client.get_wrappers(),
            mcp_tools_schema=mcp_client.get_schemas(),
            model_name=args.model_name,
            provider_url=args.provider_url,
            max_iterations=10,
            tracker=tracker,
        )
        if answer is not None:
            tracker.save_solution(
                filepath=args.output,
                success=True,
                solution=str(answer),
                error=None,
            )
        else:
            tracker.save_solution(
                filepath=args.output,
                success=False,
                solution="",
                error="No final answer within the iteration limit.",
            )
    except Exception as e:
        tracker.save_solution(
            filepath=args.output,
            success=False,
            solution="",
            error=str(e),
        )
    finally:
        if mcp_client is not None:
            mcp_client.close()


if __name__ == "__main__":
    main()