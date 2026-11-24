"""Extraction and translation of LLM-generated code and tool calls.

Implements the multi-format extraction layer specified in Section V.1.2:
- Python code blocks (primary): ```python ... ```, <end_code>
- Alternative code blocks: ```py, ```Python, untagged ``` ... ```
- Unclosed Python blocks: ```python ...
- XML tool calls (Anthropic style): <invoke name="...">...</invoke>
- JSON/Hermes tool calls:
  <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- ReAct format: Action: <tool_name>\nAction Input: <json_or_string>

Non-Python formats are converted to equivalent Python statements so the sandbox
remains strictly format-agnostic.
"""

import ast
import json
import re
from typing import Any

# Labels of a fenced block that are executed as Python: ```py, ```Python,
# ```python3... or no label at all
PYTHON_LIKE_LABEL = re.compile(r"(?:[Pp][Yy](?:thon|THON)?3?)?")


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Split text into its fenced code blocks, pairing fences in order.

    The 1st ``` opens a block, the 2nd closes it, the 3rd opens the next
    one, and so on, as Markdown does. A plain regex would also match from
    the closing fence of a ```bash block to the next fence, and execute
    the prose in between as Python.

    Args:
        text: The LLM answer.

    Returns:
        A list of (label, content) tuples, one per closed block. The
        label is the text written right after the opening fence.
    """
    blocks: list[tuple[str, str]] = []
    fences = [m.start() for m in re.finditer("```", text)]
    for open_pos, close_pos in zip(fences[::2], fences[1::2], strict=False):
        header_end = text.find("\n", open_pos + 3, close_pos)
        if header_end == -1:  # ```inline``` on a single line: not a block
            continue
        label = text[open_pos + 3:header_end].strip()
        blocks.append((label, text[header_end + 1:close_pos]))
    return blocks


def _format_python_call(func_name: str, args: dict[str, Any] | Any) -> str:
    """Convert a function name and arguments into valid Python execution code.

    Wraps normal tool calls to print their result so the LLM receives the
    Observation.
    final_answer() is called directly without print wrapping.
    """
    func_clean = func_name.strip()

    if func_clean == "final_answer":
        if isinstance(args, dict) and "answer" in args and len(args) == 1:
            arg_val = args["answer"]
        else:
            arg_val = args
        return f"final_answer({repr(arg_val)})"

    if isinstance(args, dict):
        kwargs_parts = []
        for k, v in args.items():
            kwargs_parts.append(f"{k}={repr(v)}")
        args_str = ", ".join(kwargs_parts)
    elif isinstance(args, (list, tuple)):
        args_str = ", ".join(repr(x) for x in args)
    elif args is None or args == "":
        args_str = ""
    else:
        args_str = repr(args)

    call = f"{func_clean}({args_str})"
    return f"result = {call}\nif result is not None:\n    print(result)"


def _extract_hermes_json(text: str) -> tuple[str | None, str | None]:
    """Extract and convert JSON / Hermes <tool_call> blocks."""
    # 1. Closed <tool_call>...</tool_call>
    matches = re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL
    )
    if not matches:
        # 2. Unclosed <tool_call>...
        unclosed = re.findall(r"<tool_call>\s*(\{.*)", text, re.DOTALL)
        matches = unclosed

    if not matches:
        return None, None

    code_lines = []
    for match in matches:
        try:
            data = json.loads(match.strip())
            name = (
                data.get("name")
                or data.get("tool")
                or data.get("tool_name")
                or data.get("function")
            )
            if isinstance(name, dict):
                name = name.get("name")
            args = (
                data.get("arguments")
                or data.get("parameters")
                or data.get("args")
                or data.get("input")
                or {}
            )
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if name:
                code_lines.append(_format_python_call(name, args))
        except Exception:
            continue

    if code_lines:
        code = "\n".join(code_lines)
        warning = (
            "Warning: your answer was detected in JSON/Hermes "
            "<tool_call> format. "
            "It was converted to equivalent Python code:\n"
            f"```python\n{code}\n```"
        )
        return code, warning

    return None, None


def _extract_xml_invokes(text: str) -> tuple[str | None, str | None]:
    """Extract and convert Anthropic-style XML tool calls."""
    pattern = r"<invoke\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</invoke>"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None, None

    code_lines = []
    for func_name, body in matches:
        args_dict = {}
        # 1. <parameter name="key">val</parameter>
        param_matches = re.findall(
            r"<parameter\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</parameter>",
            body,
            re.DOTALL,
        )
        for p_name, p_val in param_matches:
            val_str = p_val.strip()
            try:
                args_dict[p_name] = json.loads(val_str)
            except Exception:
                args_dict[p_name] = val_str

        # 2. <tag>val</tag> if no <parameter> tags
        if not param_matches:
            tag_matches = re.findall(
                r"<([a-zA-Z_0-9]+)>(.*?)</\1>", body, re.DOTALL
            )
            for t_name, t_val in tag_matches:
                val_str = t_val.strip()
                try:
                    args_dict[t_name] = json.loads(val_str)
                except Exception:
                    args_dict[t_name] = val_str

        code_lines.append(_format_python_call(func_name, args_dict))

    if code_lines:
        code = "\n".join(code_lines)
        warning = (
            "Warning: your answer was detected in XML <invoke> format. "
            "It was converted to equivalent Python code:\n"
            f"```python\n{code}\n```"
        )
        return code, warning

    return None, None


def _extract_react_format(text: str) -> tuple[str | None, str | None]:
    """Extract and convert ReAct format (Action: ... / Action Input: ...)."""
    pattern = (
        r"Action:\s*([a-zA-Z_0-9]+)\s*\nAction Input:\s*(.*?)"
        r"(?=\n(?:Action|Thought|Observation)|$)"
    )
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None, None

    code_lines = []
    for func_name, raw_input in matches:
        input_clean = raw_input.strip()
        args: Any = {}
        if input_clean:
            try:
                args = json.loads(input_clean)
            except Exception:
                # If literal string
                args = input_clean

        code_lines.append(_format_python_call(func_name, args))

    if code_lines:
        code = "\n".join(code_lines)
        warning = (
            "Warning: your answer was detected in ReAct format "
            "(Action / Action Input). "
            "It was converted to equivalent Python code:\n"
            f"```python\n{code}\n```"
        )
        return code, warning

    return None, None


def _extract_liquid_tool_call(text: str) -> tuple[str | None, str | None]:
    """Extract code from Liquid AI tool call format.

    Handles both:
    1. Code execution wrappers: [code(code='...')]
    2. Direct MCP tool calls: [read_file(...), search_code(...)]
    """
    m = re.search(
        r"<\|tool_call_start\|>\[(.*?)\](?:<\|tool_call_end\|>)?",
        text,
        re.DOTALL,
    )
    if not m:
        m = re.search(r"<\|tool_call_start\|>\[(.*)", text, re.DOTALL)
        if not m:
            return None, None

    call_body = m.group(1).rstrip("]")
    try:
        # Wrap in brackets if not already wrapped
        expr_str = (
            f"[{call_body}]" if not call_body.startswith("[") else call_body
        )
        parsed = ast.parse(expr_str, mode="eval")

        elements = []
        if isinstance(parsed.body, ast.List):
            elements = parsed.body.elts
        elif isinstance(parsed.body, ast.Call):
            elements = [parsed.body]

        statements = []
        for elem in elements:
            if isinstance(elem, ast.Call):
                func_name = ast.unparse(elem.func)
                # Check for code(code='...')
                if func_name == "code":
                    for kw in elem.keywords:
                        if kw.arg == "code" and isinstance(
                            kw.value, ast.Constant
                        ):
                            statements.append(str(kw.value.value).strip())
                    continue

                call_code = ast.unparse(elem)
                if func_name == "final_answer":
                    statements.append(call_code)
                else:
                    statements.append(f"print({call_code})")

        if statements:
            code = "\n".join(statements)
            warning = (
                "Warning: your tool call was detected in Liquid format and "
                f"converted to Python:\n```python\n{code}\n```"
            )
            return code, warning
    except Exception:
        pass

    # Regex fallback if AST parsing fails
    code_match = re.search(r"code=['\"](.*?)['\"]", call_body, re.DOTALL)
    if code_match:
        try:
            code = (
                code_match.group(1)
                .encode("utf-8")
                .decode("unicode_escape", "ignore")
                .strip()
            )
            return (
                code,
                "Warning: extracted code from Liquid tool call format.",
            )
        except Exception:
            pass

    return None, None


def extract_code(answer_content: str) -> tuple[str, str | None]:
    """Extract or convert Python code from an LLM response.

    Supports:
    1. Primary ```python ... ``` blocks (and <end_code>)
    2. Other code labels (```py, ```Python, untagged ```)
    3. Unclosed ```python blocks
    4. XML tool calls (<invoke name="...">...</invoke>)
    5. JSON/Hermes tool calls (<tool_call>...</tool_call>)
    6. ReAct tool calls (Action: ... / Action Input: ...)

    Args:
        answer_content: Raw text generated by the LLM.

    Returns:
        A (code, warning) tuple: extracted/converted executable Python code,
        and warning message if the block was malformed or translated.

    Raises:
        ValueError: If no executable code or tool call can be extracted.
    """
    # Clean possible <end_code> markers
    cleaned_content = answer_content.replace("<end_code>", "")

    start = "```python"
    end = "```"

    # 1. Primary standard ```python ... ```
    pattern = rf"{start}[ \t]*\n(.*?){end}"
    matches: list[str] = re.findall(pattern, cleaned_content, re.DOTALL)

    if matches:
        code_blocks = [m.strip() for m in matches if m.strip()]
        if code_blocks:
            return "\n".join(code_blocks), None

    # 2. Closed blocks with another label (```py, ```Python, ``` alone...)
    other_code = [
        code.strip()
        for label, code in _fenced_blocks(cleaned_content)
        if PYTHON_LIKE_LABEL.fullmatch(label) and code.strip()
    ]
    if other_code:
        warning = (
            "Warning: your code block was not tagged ```python, it was "
            "executed as Python anyway. Always start your code block with "
            "```python."
        )
        return "\n".join(other_code), warning

    # 3. Unclosed ```python block
    unclosed = re.search(rf"{start}[ \t]*\n(.*)", cleaned_content, re.DOTALL)
    if unclosed:
        code = unclosed.group(1).strip()
        if code:
            warning = (
                "Warning: your code block had no closing ```. Everything "
                "after ```python was executed. Always close your code block "
                "with ```."
            )
            return code, warning

    # 4. JSON / Hermes tool calls (<tool_call>...</tool_call>)
    hermes_code, hermes_warning = _extract_hermes_json(cleaned_content)
    if hermes_code:
        return hermes_code, hermes_warning

    # 5. XML tool calls (<invoke name="...">...</invoke>)
    xml_code, xml_warning = _extract_xml_invokes(cleaned_content)
    if xml_code:
        return xml_code, xml_warning

    # 6. ReAct format (Action: ... / Action Input: ...)
    react_code, react_warning = _extract_react_format(cleaned_content)
    if react_code:
        return react_code, react_warning

    # 7. Liquid tool call format
    # (<|tool_call_start|>[code(code='...')]<|tool_call_end|>)
    liquid_code, liquid_warning = _extract_liquid_tool_call(cleaned_content)
    if liquid_code:
        return liquid_code, liquid_warning

    # No extractable code or tool call found
    raise ValueError(
        "No code provided. You should provide python code to answer query, "
        "with ```python tag."
    )
