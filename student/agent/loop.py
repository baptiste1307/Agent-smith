"""Agent loop: makes the LLM and the sandbox work together."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llm.client import LLMClient
from llm.metrics import MetricsTracker
from llm.rotator import KeyRotator
from sandbox.sandbox import execute, load_config

from .extractor import extract_code
from .prompts import build_system_prompt

load_dotenv()  # reads the .env file from cwd
_student_dir = Path(__file__).resolve().parents[1]
load_dotenv(_student_dir / ".env")
load_dotenv(_student_dir.parent / ".env")

DEFAULT_SANDBOX_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "sandbox_template.json"
)


def _prune_old_observations(
    messages: list[dict[str, str]], keep_recent_turns: int = 3
) -> list[dict[str, str]]:
    """Prune verbose outputs from older turns to prevent rate limit and token bloat.

    Keeps full observations for the most recent `keep_recent_turns` turns.
    For older turns, keeps only a compact placeholder to conserve token quotas.
    """
    obs_indices = [
        i
        for i, m in enumerate(messages)
        if m.get("role") == "user" and "Observation:" in m.get("content", "")
    ]
    if len(obs_indices) <= keep_recent_turns:
        return messages

    pruned = [dict(m) for m in messages]
    cutoff_indices = set(obs_indices[:-keep_recent_turns])

    for idx in cutoff_indices:
        content = pruned[idx]["content"]
        lines = content.splitlines()
        err_lines = [l for l in lines if l.startswith("Error:") and l != "Error: None"]
        err_note = f"\n{err_lines[0]}" if err_lines else "\nError: None"
        pruned[idx]["content"] = (
            f"Observation:\n[Output of earlier turn omitted to keep context compact]{err_note}"
        )

    return pruned


def agent_loop(
    exercice: str,
    mcp_tools_wrappers: dict[str, Callable[..., Any]] | None = None,
    mcp_tools_schema: list[dict[str, Any]] | None = None,
    model_name: str = "liquid/lfm-2.5-2.6b:free",
    provider_url: str = "https://openrouter.ai/api/v1",
    max_iterations: int = 10,
    tracker: MetricsTracker | None = None,
    api_key_env: str | None = None,
):
    """Solve an exercise by alternating LLM answers and sandbox executions.

    At each turn, the LLM answers, its code is extracted and executed in
    the sandbox, and the result is sent back to the LLM. The loop stops
    when the code calls final_answer() or after max_iterations turns.

    Args:
        exercice: The exercise statement sent to the LLM.
        mcp_tools_wrappers: MCP tool wrappers, mapping tool name to a Python
            callable, injected in the sandbox namespace. None if no MCP
            server is connected.
        mcp_tools_schema: Descriptions of the same tools (name, description,
            inputSchema), written in the system prompt. None if no MCP
            server is connected.
        model_name: Name of the LLM, as expected by the provider.
        provider_url: Base URL of the LLM provider API.
        max_iterations: Maximum number of turns before giving up.
        tracker: Records the metrics of each turn, for solution.json.
            None to run without recording.
        api_key_env: Optional environment variable name holding the API key(s).

    Returns:
        The answer passed to final_answer(), or None if the LLM did not
        finish within max_iterations turns.
    """
    client = LLMClient(
        provider_url,
        model_name,
        rotator=KeyRotator.auto_detect(provider_url, env_var=api_key_env),
    )

    config = load_config(str(DEFAULT_SANDBOX_CONFIG_PATH))
    system_prompt = build_system_prompt(config, mcp_tools_schema)

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": exercice},
    ]  # at first messages only contains system prompt and our exercice request for the AI

    if tracker:
        tracker.system_prompt = system_prompt

    for turns in range(max_iterations):  # max turns
        prompt_messages = _prune_old_observations(messages)
        answer = client.generate(
            prompt_messages,
            stop=["Observation:"],
        )  # 1. AI gives its answer
        # Some providers ignore `stop`: the LLM then invents the Observation
        # and keeps going. Keep only the text before the first Observation.
        content = (answer.content or "").split("Observation:", 1)[0].rstrip()
        print(f"--- Turn {turns + 1} ---\n{content}\n")
        messages.append({"role": "assistant", "content": content})

        try:
            code, warning = extract_code(content)  # 2. Extract code

        except ValueError as e:
            if tracker:  # no code to run: sandbox_input stays empty
                tracker.record_step(
                    answer, sandbox_input="", sandbox_output=str(e)
                )
            messages.append({"role": "user", "content": f"{e}"})
            continue

        result = execute(code, config, mcp_tools_wrappers)  # 3. Execute code

        if result["final_answer"] is not None:  # 4. Is AI Done ?
            if tracker:
                tracker.record_step(
                    answer,
                    sandbox_input=code,
                    sandbox_output=f"Final answer: {result['final_answer']}",
                )
            return result["final_answer"]

        # 5. show result to AI as text
        raw_output = result["output"] or ""
        if len(raw_output) > 4000:
            raw_output = (
                raw_output[:4000]
                + f"\n... [output truncated: showing first 4000 of {len(result['output'])} characters to save tokens]"
            )
        observation = f"Observation:\nOutput:\n{raw_output}\nError: {result['error']}"
        if warning is not None:  # the code block was malformed
            observation = f"{warning}\n{observation}"
        print(observation)
        if tracker:
            tracker.record_step(
                answer, sandbox_input=code, sandbox_output=observation
            )
        messages.append({"role": "user", "content": observation})

        # 6. stop before the next turn if a limit is exceeded
        if tracker:
            exceeded, reason = tracker.check_limits()
            if exceeded:
                raise RuntimeError(f"Limit exceeded: {reason}")

    return None  # no final answer after max_iterations turns


# Manual test, to run once mcp_tools_mbpp.py exposes the tools:
# if __name__ == "__main__":
#     from mcp_client.client import MCPClient
#
#     mcp_client = MCPClient("python ../mcp_tools_mbpp.py")
#     try:
#         print(
#             agent_loop(
#                 "Read the file 'sandbox_template.json', then give as "
#                 "final answer the value of max_execution_time_seconds.",
#                 mcp_tools_wrappers=mcp_client.get_wrappers(),
#                 mcp_tools_schema=mcp_client.get_schemas(),
#             )
#         )
#     finally:
#         mcp_client.close()
