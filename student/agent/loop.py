"""Agent loop: makes the LLM and the sandbox work together."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llm.client import DEFAULT_MAX_TOKENS, LLMClient
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
    """Prune verbose outputs from older turns.

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
        err_lines = [
            line
            for line in lines
            if line.startswith("Error:") and line != "Error: None"
        ]
        err_note = f"\n{err_lines[0]}" if err_lines else "\nError: None"
        pruned[idx]["content"] = (
            "Observation:\n"
            "[Output of earlier turn omitted to keep context compact]"
            f"{err_note}"
        )

    return pruned


# Pessimistic characters-per-token ratio for the first request (English
# text and code are usually closer to 3.5-4)
INITIAL_CHARS_PER_TOKEN = 3.0
# Safety factor on the input token estimate
TOKEN_ESTIMATE_MARGIN = 1.10
# Per-message overhead of the chat format (role, separators)
TOKENS_PER_MESSAGE = 8
# No request is sent with less output budget / time than this
MIN_OUTPUT_TOKENS = 64
MIN_TURN_SECONDS = 5.0


def _message_chars(messages: list[dict[str, str]]) -> int:
    """Total number of characters in the message contents."""
    return sum(len(m.get("content", "")) for m in messages)


def _estimate_input_tokens(
    messages: list[dict[str, str]], chars_per_token: float
) -> int:
    """Estimate the input tokens of a request, rounded up with a margin."""
    raw = _message_chars(messages) / chars_per_token
    return int(raw * TOKEN_ESTIMATE_MARGIN) + TOKENS_PER_MESSAGE * len(
        messages
    )


def _fit_input_budget(
    messages: list[dict[str, str]],
    keep_recent_turns: int,
    remaining_input_tokens: int | None,
    chars_per_token: float,
) -> list[dict[str, str]]:
    """Return the messages to send, pruned enough to fit the token budget.

    Old observations are pruned as usual; if the request still does not
    fit in the remaining input budget, fewer recent observations are kept
    (down to the last one only).

    Args:
        messages: The full conversation.
        keep_recent_turns: Observations normally kept in full.
        remaining_input_tokens: Input tokens left, or None if unlimited.
        chars_per_token: Calibrated characters-per-token ratio.

    Returns:
        The pruned conversation.

    Raises:
        RuntimeError: If even the most pruned conversation does not fit.
    """
    candidate = _prune_old_observations(messages, keep_recent_turns)
    if remaining_input_tokens is None:
        return candidate
    estimate = 0
    for keep in range(max(keep_recent_turns, 1), 0, -1):
        candidate = _prune_old_observations(messages, keep)
        estimate = _estimate_input_tokens(candidate, chars_per_token)
        if estimate <= remaining_input_tokens:
            return candidate
    raise RuntimeError(
        "Input token budget exhausted: the next request needs about "
        f"{estimate} tokens, only {remaining_input_tokens} are left."
    )


def agent_loop(
    exercice: str,
    mcp_tools_wrappers: dict[str, Callable[..., Any]] | None = None,
    mcp_tools_schema: list[dict[str, Any]] | None = None,
    model_name: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
    provider_url: str = "https://openrouter.ai/api/v1",
    max_iterations: int = 10,
    tracker: MetricsTracker | None = None,
    api_key_env: str | None = None,
    keep_recent_turns: int = 3,
    max_obs_chars: int = 4000,
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
    ]  # first messages: system prompt + exercise request

    if tracker:
        tracker.system_prompt = system_prompt

    def check_limits() -> None:
        """Stop the run as soon as a benchmark limit is exceeded."""
        if tracker:
            exceeded, reason = tracker.check_limits()
            if exceeded:
                raise RuntimeError(f"Limit exceeded: {reason}")

    # Characters per input token, measured on the provider's answers.
    # The first request uses a pessimistic value.
    chars_per_token = INITIAL_CHARS_PER_TOKEN

    for turns in range(max_iterations):  # max turns
        # 0. Check every budget BEFORE the request: the moulinette fails a
        # task whose totals exceed a limit, even with a correct solution
        prompt_messages = _fit_input_budget(
            messages,
            keep_recent_turns,
            tracker.remaining_input_tokens if tracker else None,
            chars_per_token,
        )
        max_tokens = DEFAULT_MAX_TOKENS
        remaining = tracker.remaining_output_tokens if tracker else None
        if remaining is not None:
            if remaining < MIN_OUTPUT_TOKENS:
                raise RuntimeError(
                    f"Output token budget exhausted ({remaining} left)."
                )
            max_tokens = min(max_tokens, remaining)
        time_left = tracker.time_left_seconds() if tracker else None
        if time_left is not None and time_left < MIN_TURN_SECONDS:
            raise RuntimeError(
                f"Time budget exhausted ({max(time_left, 0):.0f}s left "
                "before the benchmark timeout)."
            )

        answer = client.generate(
            prompt_messages,
            stop=["Observation:"],
            max_tokens=max_tokens,
            deadline=tracker.deadline if tracker else None,
        )  # 1. AI gives its answer
        if answer.input_tokens > 0 and not answer.estimated_tokens:
            chars_per_token = _message_chars(prompt_messages) / (
                answer.input_tokens
            )
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
            check_limits()
            continue

        # Near the deadline, the sandboxed code gets only the time left
        run_config = config
        time_left = tracker.time_left_seconds() if tracker else None
        if time_left is not None:
            run_config = config.model_copy(
                update={
                    "max_execution_time_seconds": max(
                        1,
                        min(config.max_execution_time_seconds, int(time_left)),
                    )
                }
            )
        result = execute(code, run_config, mcp_tools_wrappers)  # 3. Execute

        if result["final_answer"] is not None:  # 4. Is AI Done ?
            if tracker:
                tracker.record_step(
                    answer,
                    sandbox_input=code,
                    sandbox_output=(
                        f"{result['output']}"
                        f"Final answer: {result['final_answer']}"
                    ),
                )
            return result["final_answer"]

        # 5. show result to AI as text
        raw_output = result["output"] or ""
        if len(raw_output) > max_obs_chars:
            head_len = int(max_obs_chars * 0.375)
            tail_len = max_obs_chars - head_len
            raw_output = (
                raw_output[:head_len]
                + (
                    f"\n... [output truncated: showing first {head_len} "
                    f"and last {tail_len} of {len(result['output'])} "
                    "characters to save tokens] ...\n"
                )
                + raw_output[-tail_len:]
            )
        observation = (
            f"Observation:\nOutput:\n{raw_output}\nError: {result['error']}"
        )
        if warning is not None:  # the code block was malformed
            observation = f"{warning}\n{observation}"
        print(observation)
        if tracker:
            tracker.record_step(
                answer, sandbox_input=code, sandbox_output=observation
            )
        messages.append({"role": "user", "content": observation})

        # 6. stop before the next turn if a limit is exceeded
        check_limits()

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
