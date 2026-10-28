"""Agent loop: makes the LLM and the sandbox work together."""

from sandbox.sandbox import execute, load_config
from llm.client import LLMClient
from llm.rotator import KeyRotator
from typing import Any, Callable
from .prompts import build_system_prompt
from .extractor import extract_code
from dotenv import load_dotenv

load_dotenv()  # reads the .env file


def agent_loop(
    exercice: str,
    mcp_tools_wrappers: dict[str, Callable[..., Any]] | None = None,
    mcp_tools_schema: list[dict[str, Any]] | None = None,
):
    """Solve an exercise by alternating LLM answers and sandbox executions.

    At each turn, the LLM answers, its code is extracted and executed in
    the sandbox, and the result is sent back to the LLM. The loop stops
    when the code calls final_answer() or after 10 turns.

    Args:
        exercice: The exercise statement sent to the LLM.
        mcp_tools_wrappers: MCP tool wrappers, mapping tool name to a Python
            callable, injected in the sandbox namespace. None if no MCP
            server is connected.
        mcp_tools_schema: Descriptions of the same tools (name, description,
            inputSchema), written in the system prompt. None if no MCP
            server is connected.

    Returns:
        The answer passed to final_answer(), or None if the LLM did not
        finish within 10 turns.
    """
    client = LLMClient(
        provider_url="https://openrouter.ai/api/v1",
        model_name="poolside/laguna-s-2.1:free",
        rotator=KeyRotator.from_env("OPENROUTER_API_KEY"),
    )

    config = load_config("sandbox_template.json")

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(config, mcp_tools_schema),
        },
        {"role": "user", "content": exercice},
    ]  # at first messages only contains system prompt and our exercice request for the AI

    for turns in range(10):  # max 10 turns
        answer = client.generate(
            messages,
            stop=["Observation:"],
        )  # 1. AI gives its answer
        print(f"--- Turn {turns + 1} ---\n{answer.content}\n")
        messages.append({"role": "assistant", "content": answer.content})

        try:
            code, warning = extract_code(answer.content)  # 2. Extract code

        except ValueError as e:
            messages.append({"role": "user", "content": f"{e}"})
            continue

        result = execute(code, config, mcp_tools_wrappers)  # 3. Execute code

        if result["final_answer"] is not None:  # 4. Is AI Done ?
            return result["final_answer"]

        # 5. show result to AI as text
        observation = f"Observation:\nOutput:\n{result['output']}\nError: {result['error']}"
        if warning is not None:  # the code block was malformed
            observation = f"{warning}\n{observation}"
        print(observation)
        messages.append({"role": "user", "content": observation})

    return None  # no final answer after 10 turns


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
