"""Prompts sent to the LLM."""

import textwrap
from typing import Any

from models_public import MBPPTaskInput, SWEBenchTaskInput
from sandbox.sandbox import SandboxConfig

SYSTEM_PROMPT_TEMPLATE = """You are an autonomous coding agent.
You solve tasks by writing Python code.

## How it works
You work in turns. At each turn:
1. You write a short Thought, then ONE Python code block.
2. Your code is really executed in a sandbox.
3. You receive an Observation: what your code printed,
   and the error if there was one.
4. You use this Observation to decide your next step.

You do not need to succeed on the first try.
Test your ideas, read the result, then fix them.

## Response format
Always answer exactly like this:

Thought: <what you understood and what you will do now>
Code:
```python
<your code>
```

## Rules
- Always write exactly one ```python code block per answer.
  An answer without code is wasted.
- Use print() to see values: the Observation only contains what you print.
- Never write the Observation yourself.
  Stop after your code block and wait for the real result.
- Each code block runs in a fresh namespace:
  variables and functions from previous turns are lost.
  Redefine everything you need in each block.
- You can only import these modules: {authorized_imports}.
  Any other import will fail.

## Available tools
These functions are already defined in your code namespace.
Call them directly, without importing them.

{tools_manual}

## Finishing
When you have tested your solution and it works,
call final_answer() inside your code block:

```python
final_answer(your_answer)
```

final_answer() is a Python function, not text.
Writing "final_answer" outside a code block does nothing.

## Example
Task: Write a function that returns the largest number of a list.

Thought: I will write the function and test it on a few examples
before giving my answer.
Code:
```python
def largest(numbers):
    return max(numbers)

print(largest([3, 7, 2]))
print(largest([-5, -1]))
```

Observation:
7
-1

Thought: The results are correct. I give my final answer.
Code:
```python
final_answer("def largest(numbers):\\n    return max(numbers)")
```
"""


def build_system_prompt(
    config: SandboxConfig,
    mcp_tools_schema: list[dict[str, Any]] | None = None,
) -> str:
    """Fill the system prompt template with the sandbox settings and tools.

    Args:
        config: The sandbox configuration, used to list allowed imports.
        mcp_tools_schema: The tools of the connected MCP server, or None if no
            server is connected.

    Returns:
        The system prompt sent to the LLM.
    """
    # ".*" entries are a sandbox rule, the LLM only needs module names
    imports = ", ".join(
        m for m in config.authorized_imports if not m.endswith(".*")
    )
    if mcp_tools_schema:
        tools_manual = build_mcp_tools_manual(mcp_tools_schema)
    else:
        tools_manual = "No MCP tool is connected: use plain Python only."

    return SYSTEM_PROMPT_TEMPLATE.format(
        authorized_imports=imports,
        tools_manual=tools_manual,
    )


# JSON Schema types (used by MCP) converted to the Python types the LLM writes
JSON_TO_PYTHON_TYPES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


def json_type_to_python(param_info: dict[str, Any]) -> str:
    """Convert the JSON Schema of one parameter into a Python type name.

    A parameter typed `str | Path` in Python becomes an "anyOf" list in
    JSON Schema: each option is converted, and duplicates are removed.

    Args:
        param_info: The JSON Schema of the parameter.

    Returns:
        The Python type, for example "str" or "int | None", or "Any" if
        the schema gives no type.
    """
    if "anyOf" in param_info:
        options = [
            json_type_to_python(option) for option in param_info["anyOf"]
        ]
        unique_options = list(dict.fromkeys(options))  # keeps the order
        return " | ".join(unique_options)
    if param_info.get("type") == "null":
        return "None"
    param_type = param_info.get("type")
    if isinstance(param_type, str):
        return JSON_TO_PYTHON_TYPES.get(param_type, "Any")
    return "Any"


def build_mcp_tools_manual(mcp_tools_schema: list[dict[str, Any]]) -> str:
    """Describe the MCP tools as Python signatures the LLM can read.

    Each tool becomes a signature line followed by its indented
    description. Missing fields are tolerated, since the connected MCP
    server may be unknown.

    Args:
        mcp_tools_schema: The tools of the MCP server, each with a "name", a
            "description" and an "inputSchema" (JSON Schema).

    Returns:
        The manual of all the tools, one section per tool.
    """
    sections = []
    for tool in mcp_tools_schema:
        schema = tool.get("inputSchema", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        params = []
        for param_name, param_info in properties.items():
            param = f"{param_name}: {json_type_to_python(param_info)}"
            if param_name not in required:
                # Show the real default value when the schema gives one
                if "default" in param_info:
                    param += f" = {param_info['default']!r}"
                else:
                    param += " = optional"
            params.append(param)

        signature = f"{tool.get('name', 'unknown_tool')}({', '.join(params)})"
        description = tool.get("description") or "No description."
        # Indent every line, so the description stays under its signature
        sections.append(
            f"{signature}\n{textwrap.indent(description.strip(), '    ')}"
        )

    return "\n\n".join(sections)


MBPP_TASK_TEMPLATE = """Task: {task_definition}

Your function must have exactly this signature:
{function_definition}

Your solution must pass these tests:
{tests}

Some hidden tests will also be run:
write a general solution, do not hard-code the test values.

Test your function on these asserts first.
When they all pass, call final_answer() with the full source code
of your function as a string."""


def build_mbpp_task_prompt(task: MBPPTaskInput) -> str:
    """Write the message that asks the LLM to solve one MBPP task.

    Args:
        task: The MBPP task given by the moulinette.

    Returns:
        The task message, sent to the LLM after the system prompt.
    """
    tests = "\n".join(task.test_imports + task.test_list)
    return MBPP_TASK_TEMPLATE.format(
        task_definition=task.task_definition,
        function_definition=task.function_definition,
        tests=tests or "No public test is given.",
    )


SWEBENCH_TASK_TEMPLATE = """Task: fix the following GitHub issue
of the repository {repo}.

The repository is in /testbed. Your changes to its files are the solution.

<issue>
{problem_statement}
</issue>
{hints}
Work efficiently (aim for 5-10 turns total):
1. Explore: use search_code, search_function_or_class_definition_in_code,
   or list_files (never import os/subprocess).
   Read only the relevant line range with
   read_file(path, start_line, end_line).
2. Fix: once you identify the root cause,
   immediately apply the minimal fix with edit_file.
   Do not modify existing test files.
3. Verify: call run_tests() to run the official evaluation test suite.
4. Submit: as soon as run_tests() passes and get_patch() is non-empty,
   immediately call final_answer(get_patch()) on the very next turn.

Always print the result of each tool call
(or leave it as a bare expression) so you see it in the Observation.
Work only from the repository and task context;
do not fetch external issues, pull requests, or patches."""


def build_swebench_task_prompt(task: SWEBenchTaskInput) -> str:
    """Write the message that asks the LLM to solve one SWE-bench task.

    Args:
        task: The SWE-bench task given by the moulinette.

    Returns:
        The task message, sent to the LLM after the system prompt.
    """
    hints = ""
    if task.hints_text.strip():
        hints = f"\n<hints>\n{task.hints_text.strip()}\n</hints>\n"

    return SWEBENCH_TASK_TEMPLATE.format(
        repo=task.repo or "under test",
        problem_statement=task.problem_statement.strip(),
        hints=hints,
    )
