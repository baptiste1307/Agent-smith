import argparse
import os
import re
import shlex
import shutil
import subprocess
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


def _extract_base_commit(eval_script: str) -> str | None:
    """Extract the base commit SHA from the eval_script.

    Looks for patterns like:
      git -c core.fileMode=false diff <sha>
      git checkout <sha>
    """
    # Pattern 1: git diff <sha>
    m = re.search(r"git\s+(?:-c\s+\S+\s+)?diff\s+([0-9a-f]{7,40})\b", eval_script)
    if m:
        return m.group(1)
    # Pattern 2: git checkout <sha>
    m = re.search(r"git\s+checkout\s+([0-9a-f]{7,40})\b", eval_script)
    if m:
        return m.group(1)
    return None


def _clone_repo_fallback(
    repo: str, base_commit: str | None, instance_id: str
) -> str | None:
    """Clone a GitHub repo and check out the base commit when Docker is unavailable.

    Returns the path to the cloned testbed directory, or None on failure.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"swebench_{instance_id}_")
    testbed_path = os.path.join(tmp_dir, "testbed")

    github_url = f"https://github.com/{repo}.git"
    print(f"[SWE-bench] Docker unavailable. Cloning {github_url} (shallow)...")

    try:
        # Shallow clone for speed — depth=1 at the target commit
        if base_commit:
            # First try a shallow fetch of the exact commit
            subprocess.run(
                ["git", "init", testbed_path],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "-C", testbed_path, "remote", "add", "origin", github_url],
                capture_output=True, check=True,
            )
            res = subprocess.run(
                ["git", "-C", testbed_path, "fetch", "--depth=1", "origin", base_commit],
                capture_output=True, timeout=120,
            )
            if res.returncode == 0:
                subprocess.run(
                    ["git", "-C", testbed_path, "checkout", "FETCH_HEAD"],
                    capture_output=True, check=True,
                )
            else:
                # Some servers don't allow fetching by SHA — full clone fallback
                shutil.rmtree(testbed_path, ignore_errors=True)
                subprocess.run(
                    ["git", "clone", "--depth=50", github_url, testbed_path],
                    capture_output=True, check=True, timeout=180,
                )
                subprocess.run(
                    ["git", "-C", testbed_path, "checkout", base_commit],
                    capture_output=True, check=True,
                )
        else:
            subprocess.run(
                ["git", "clone", "--depth=1", github_url, testbed_path],
                capture_output=True, check=True, timeout=180,
            )

        print(f"[SWE-bench] Repository cloned to {testbed_path}")
        return tmp_dir
    except Exception as e:
        print(f"[SWE-bench] Warning: Failed to clone repository: {e}", file=sys.stderr)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Agent Smith SWE-bench agent."
    )
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="codestral-latest")
    parser.add_argument(
        "--provider-url", default="https://api.mistral.ai/v1"
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

    parser.add_argument(
        "--keep-recent-turns",
        type=int,
        default=10,
        help="Number of recent observations to keep verbatim before pruning older turns.",
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
                    container_name=f"agent_swe_{task.instance_id.replace('__', '_')}_{os.getpid()}",
                )
                docker_container.start()
                temp_testbed_dir = tempfile.mkdtemp(prefix=f"swebench_{task.instance_id}_")
                print("[SWE-bench] Syncing /testbed from container to host workspace...")
                docker_container.copy_from_container("/testbed", temp_testbed_dir)
                local_testbed = Path(temp_testbed_dir) / "testbed"
                if not local_testbed.exists():
                    local_testbed = Path(temp_testbed_dir)
                # Handle possible nested testbed/testbed if created by docker cp
                nested = local_testbed / "testbed"
                if nested.exists() and (nested / ".git").exists():
                    local_testbed = nested
                os.environ["TESTBED_PATH"] = str(local_testbed.resolve())
                os.environ["DOCKER_CONTAINER_ID"] = docker_container.container_id
            except Exception as docker_err:
                print(f"[SWE-bench] Warning: Failed to initialize Docker container: {docker_err}", file=sys.stderr)

    # Fallback: clone the repo from GitHub when Docker is unavailable
    if not os.environ.get("TESTBED_PATH") and task.repo:
        base_commit = _extract_base_commit(task.eval_script) if task.eval_script else None
        clone_dir = _clone_repo_fallback(task.repo, base_commit, task.instance_id)
        if clone_dir:
            temp_testbed_dir = clone_dir
            os.environ["TESTBED_PATH"] = os.path.join(clone_dir, "testbed")

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
            keep_recent_turns=args.keep_recent_turns,
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
