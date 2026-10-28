"""Prompts sent to the LLM."""

import textwrap
from typing import Any

from sandbox.sandbox import SandboxConfig

SYSTEM_PROMPT_TEMPLATE = """You are an autonomous coding agent. You solve tasks by writing Python code.

## How it works
You work in turns. At each turn:
1. You write a short Thought, then ONE Python code block.
2. Your code is really executed in a sandbox.
3. You receive an Observation: what your code printed, and the error if there was one.
4. You use this Observation to decide your next step.

You do not need to succeed on the first try. Test your ideas, read the result, then fix them.

## Response format
Always answer exactly like this:

Thought: <what you understood and what you will do now>
Code:
```python
<your code>
```

## Rules
- Always write exactly one ```python code block per answer. An answer without code is wasted.
- Use print() to see values: the Observation only contains what you print.
- Never write the Observation yourself. Stop after your code block and wait for the real result.
- Each code block runs in a fresh namespace: variables and functions from previous turns are lost. Redefine everything you need in each block.
- You can only import these modules: {authorized_imports}. Any other import will fail.

## Available tools
These functions are already defined in your code namespace. Call them directly, without importing them.

{tools_manual}

## Finishing
When you have tested your solution and it works, call final_answer() inside your code block:

```python
final_answer(your_answer)
```

final_answer() is a Python function, not text. Writing "final_answer" outside a code block does nothing.

## Example
Task: Write a function that returns the largest number of a list.

Thought: I will write the function and test it on a few examples before giving my answer.
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
        options = [json_type_to_python(option) for option in param_info["anyOf"]]
        unique_options = list(dict.fromkeys(options))  # keeps the order
        return " | ".join(unique_options)
    if param_info.get("type") == "null":
        return "None"
    return JSON_TO_PYTHON_TYPES.get(param_info.get("type"), "Any")


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
        sections.append(f"{signature}\n{textwrap.indent(description.strip(), '    ')}")

    return "\n\n".join(sections)
