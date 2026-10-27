"""Agent loop: makes the LLM and the sandbox work together."""

from sandbox.sandbox import execute, load_config
from llm.client import LLMClient
from llm.rotator import KeyRotator
from .prompts import build_system_prompt
from .extractor import extract_code
from dotenv import load_dotenv

load_dotenv()  # reads the .env file


def agent_loop(exercice: str):
    """Solve an exercise by alternating LLM answers and sandbox executions.

    At each turn, the LLM answers, its code is extracted and executed in
    the sandbox, and the result is sent back to the LLM. The loop stops
    when the code calls final_answer() or after 10 turns.

    Args:
        exercice: The exercise statement sent to the LLM.

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
        {"role": "system", "content": build_system_prompt(config)},
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
            code = extract_code(answer.content)  # 2. Extract code from answer

        except ValueError as e:
            messages.append({"role": "user", "content": f"{e}"})
            continue

        result = execute(code, config)  # 3. Execute code

        if result["final_answer"] is not None:  # 4. Is AI Done ?
            return result["final_answer"]

        # 5. show result to AI as text
        observation = f"Observation:\nOutput:\n{result['output']}\nError: {result['error']}"
        print(observation)
        messages.append({"role": "user", "content": observation})

    return None  # no final answer after 10 turns


# if __name__ == "__main__":
#     print(agent_loop("ecris une fonction qui additionne deux nombres."))
