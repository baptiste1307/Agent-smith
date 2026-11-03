import argparse

from agent.loop import agent_loop
from agent.prompts import build_mbpp_task_prompt
from llm.metrics import MetricsTracker
from mcp_client.client import MCPClient
from models_public import MBPPTaskInput


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="liquid/lfm-2.5-2.6b:free")
    parser.add_argument(
        "--provider-url", default="https://openrouter.ai/api/v1"
    )

    args = parser.parse_args()

    with open(args.task_file) as f:
        # "model_validate_json": read the JSON of f. Error if something missing.
        task = MBPPTaskInput.model_validate_json(f.read())

    tracker = MetricsTracker(task_id=task.task_id, benchmark="mbpp")

    mcp_client = MCPClient(server="python ../mcp_tools_mbpp.py")

    try:
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
        mcp_client.close()


if __name__ == "__main__":
    main()
