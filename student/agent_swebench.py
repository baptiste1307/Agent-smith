import argparse
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

from agent.loop import agent_loop
from agent.prompts import build_swebench_task_prompt
from docker_env.container import DockerContainer
from llm.metrics import MetricsTracker
from mcp_client.client import MCPClient
from models_public import SWEBenchTaskInput


def _default_mcp_server_command() -> str:
    """
	Return the command that starts the repository SWE-bench MCP server.
	"""
    repo_root = Path(__file__).resolve().parents[1]
    server_path = repo_root / "mcp_tools_swebench.py"
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(server_path))}"


def _looks_like_git_patch(text: str) -> bool:
    """
	Return True when text has the basic markers of a unified git diff.
	"""
    markers = ("diff --git", "--- a/", "+++ b/", "@@")
    return bool(text.strip()) and all(marker in text for marker in markers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Agent Smith SWE-bench agent."
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
            "mcp_tools_swebench.py stdio server."
        ),
    )
    parser.add_argument(
        "--testbed-path",
        default=None,
        help=(
            "Optional repository path for local tool testing. When omitted, "
            "the existing TESTBED_PATH environment variable is used."
        ),
    )

    args = parser.parse_args()

    with open(args.task_file, encoding="utf-8") as f:
        task = SWEBenchTaskInput.model_validate_json(f.read())

    docker_container: DockerContainer | None = None
    temp_testbed_dir: str | None = None

    if args.testbed_path:
        os.environ["TESTBED_PATH"] = str(Path(args.testbed_path).resolve())
    elif not os.environ.get("TESTBED_PATH") and task.docker_image:
        if DockerContainer.is_docker_available():
            try:
                print(f"[SWE-bench] Starting Docker container for image {task.docker_image}...")
                docker_container = DockerContainer(
                    image_name=task.docker_image,
                    container_name=f"agent_swe_{task.instance_id.replace('__', '_')}",
                )
                docker_container.start()
                temp_testbed_dir = tempfile.mkdtemp(prefix=f"swebench_{task.instance_id}_")
                local_testbed = Path(temp_testbed_dir) / "testbed"
                local_testbed.mkdir(parents=True, exist_ok=True)
                print("[SWE-bench] Syncing /testbed from container to host workspace...")
                docker_container.copy_from_container("/testbed", local_testbed)
                os.environ["TESTBED_PATH"] = str(local_testbed)
                os.environ["DOCKER_CONTAINER_ID"] = docker_container.container_id
            except Exception as docker_err:
                print(f"[SWE-bench] Warning: Failed to initialize Docker container: {docker_err}", file=sys.stderr)

    if task.eval_script and not os.environ.get("EVAL_SCRIPT"):
        os.environ["EVAL_SCRIPT"] = task.eval_script

    tracker = MetricsTracker(
        task_id=task.instance_id,
        benchmark="swebench",
    )

    mcp_client: MCPClient | None = None
    try:
        mcp_client = MCPClient(
            server=args.mcp_server or _default_mcp_server_command()
        )

        answer = agent_loop(
            exercice=build_swebench_task_prompt(task),
            mcp_tools_wrappers=mcp_client.get_wrappers(),
            mcp_tools_schema=mcp_client.get_schemas(),
            model_name=args.model_name,
            provider_url=args.provider_url,
            max_iterations=30,
            tracker=tracker,
        )

        patch = str(answer or "")
        if _looks_like_git_patch(patch):
            tracker.save_solution(
                filepath=args.output,
                success=True,
                solution=patch,
                error=None,
            )
        else:
            tracker.save_solution(
                filepath=args.output,
                success=False,
                solution="",
                error=(
                    "No valid git patch was produced within the iteration "
                    "limit."
                ),
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
        if docker_container is not None:
            print("[SWE-bench] Cleaning up Docker container...")
            docker_container.stop()
        if temp_testbed_dir and os.path.exists(temp_testbed_dir):
            shutil.rmtree(temp_testbed_dir, ignore_errors=True)
        os.environ.pop("DOCKER_CONTAINER_ID", None)


if __name__ == "__main__":
    main()
