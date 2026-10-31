*This project has been created as part of the 42 curriculum by avauclai, bpasquer, mlorenzo.*

# Agent Smith

## Description

Agent Smith is an autonomous code-agent project. The goal is to solve programming
tasks by letting an LLM reason, write Python code, execute that code in a sandbox,
observe the result, and iterate until it can submit a final answer.

The project targets two benchmarks:

- MBPP, where the agent returns Python function code.
- SWE-bench, where the agent edits a real repository and returns a git patch.

The main idea is to keep each responsibility separate: the LLM decides what to do,
the agent loop coordinates the turns, the sandbox executes generated Python under
restrictions, and MCP tools expose controlled actions such as reading files, editing
files, searching code, running tests, and collecting patches.

## Instructions

## System Architecture

The project is split into small layers:

```text
task.json
  -> benchmark agent CLI
  -> shared agent loop
  -> LLM client
  -> code extractor
  -> sandbox
  -> MCP tool wrappers
  -> MCP server
  -> tool implementation
  -> observation
  -> solution.json
```

Important directories and files:

- `student/agent/`: shared orchestration code, prompts, and code extraction.
- `student/sandbox/`: restricted Python execution and `final_answer(...)`.
- `student/llm/`: provider calls, key rotation, retries, and usage metrics.
- `student/mcp_client/`: MCP connection and Python tool-wrapper generation.
- `student/tools/`: tool implementations that can be tested independently.
- `mcp_tools_mbpp.py`: MCP server for MBPP tools.
- `mcp_tools_swebench.py`: MCP server for SWE-bench tools.
- `student/agent_mbpp.py`: MBPP CLI entrypoint.
- `student/agent_swebench.py`: SWE-bench CLI entrypoint.

The sandbox and MCP tools are separate security domains. The sandbox limits
LLM-generated Python code, while MCP tools perform controlled operations outside the
sandbox namespace. `final_answer(...)` is always provided by the sandbox and is not an
MCP tool.

## Agent Loop

## Sandbox Design

## MCP

MCP is used as the bridge between the agent loop and benchmark-specific tools.
The agent does not call tool modules directly. Instead, it connects to an MCP server,
discovers the available tools, and injects Python wrappers into the sandbox namespace.

On the client side, `student/mcp_client/client.py` supports:

- stdio MCP servers, such as `python ../mcp_tools_swebench.py`;
- streamable HTTP MCP servers;
- dynamic tool discovery with `list_tools()`;
- schema export for prompt generation;
- Python wrappers so sandboxed code can call tools like normal functions.

For stdio servers, the current environment is passed to the MCP subprocess. This is
needed for SWE-bench because tools rely on variables such as `TESTBED_PATH` and
`EVAL_SCRIPT`.

The SWE-bench MCP server is implemented in `mcp_tools_swebench.py`. It exposes the
mandatory SWE-bench tools and delegates the actual logic to modules in
`student/tools/`, so each tool can be tested without starting an MCP server.

Start the SWE-bench MCP server with stdio:

```bash
cd student
UV_CACHE_DIR=/private/tmp/uv-cache uv run python ../mcp_tools_swebench.py
```

Start it with streamable HTTP:

```bash
cd student
UV_CACHE_DIR=/private/tmp/uv-cache uv run python ../mcp_tools_swebench.py \
  --http --host 127.0.0.1 --port 8001
```

## SWE-bench Agent

The SWE-bench entrypoint is `student/agent_swebench.py`.

It:

- loads a `SWEBenchTaskInput` from `--task-file`;
- optionally sets `TESTBED_PATH` from `--testbed-path`;
- stores the task evaluation script in `EVAL_SCRIPT` if none is already set;
- starts the local SWE-bench MCP server by default;
- runs the shared `agent_loop(...)` with a 30-iteration limit;
- writes a `SolutionOutput` JSON file to `--output`.

The final SWE-bench solution is a git patch. The agent only marks success when the
final answer looks like a unified git diff with markers such as `diff --git`, `--- a/`,
`+++ b/`, and `@@`.

Docker orchestration is kept outside this file. The SWE-bench agent expects the
Docker/testbed layer to expose the repository through `TESTBED_PATH`.

Example run:

```bash
cd student
export OPENROUTER_API_KEY="your_api_key"
export TESTBED_PATH="/path/to/swebench/repository"

UV_CACHE_DIR=/private/tmp/uv-cache uv run python -m agent_swebench \
  --task-file ../cache/swebench_task.json \
  --output ../cache/swebench_solution.json \
  --model-name "model/name" \
  --provider-url "https://provider.api/v1"
```

## Tool Implementation Details

### File System Tools

`student/tools/fs_tools.py` implements:

- `read_file(filepath, start_line, end_line)`
- `edit_file(filepath, old_str, new_str)`
- `list_files(directory, pattern)`

All relative paths are resolved from `TESTBED_PATH`. If `TESTBED_PATH` is not set,
the tools fall back to the current working directory.

#### `read_file(filepath, start_line, end_line)`

Reads a file and returns numbered lines:

```text
12: def example():
13:     return True
```

Technical choices:

- line numbers are 1-indexed;
- `end_line=-1` reads until the end of the file;
- `end_line` is clamped if it is larger than the file length;
- errors are returned as readable strings so the LLM receives a clear observation.

#### `edit_file(filepath, old_str, new_str)`

Replaces one exact string in a file.

Technical choices:

- `old_str` cannot be empty;
- the target string must appear exactly once;
- multiple matches are rejected to avoid unsafe broad edits;
- the rest of the file is left unchanged.

This keeps editing simple and auditable for an autonomous agent.

#### `list_files(directory, pattern)`

Recursively lists files matching a glob pattern.

Technical choices:

- results are sorted for stable output;
- noisy folders such as `.git`, `__pycache__`, virtualenvs, build folders, and cache
  folders are skipped;
- output is capped at 500 results and reports truncation.

### Code Search Tools

### Execution Tools

## Benchmark Results and Analysis

## Resources

## AI Usage
