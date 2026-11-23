*This project has been created as part of the 42 curriculum by avauclai, bpasquer, mlorenzo.*

# Agent Smith

## Description

Agent Smith is an autonomous code-agent framework built from scratch in Python. Instead of relying on static prompts or JSON-only tool calling, the agent operates through a **Thought -> Code -> Observation** loop: it reasons about a programming task, writes executable Python code, runs that code inside a restricted sandbox, calls external tools exposed via the **Model Context Protocol (MCP)**, observes execution results, and iterates until it submits a final solution.

The framework targets two benchmarks:

- **MBPP (Mostly Basic Python Problems)**: algorithmic Python problems where the agent submits a verified Python function.
- **SWE-bench Verified**: real-world GitHub bug-fixing tasks inside Dockerized repositories where the agent explores the codebase, applies targeted edits, runs the official evaluation suite, and submits a unified `git diff` patch.

The project enforces strict separation of responsibilities across independent layers: the orchestrator loop, multi-provider LLM client with API-key rotation, multi-format code extractor, security-hardened Python sandbox, dynamic MCP client/servers, and standalone tool modules.

---

## Instructions

### Prerequisites

- **Python 3.10+** and **[uv](https://docs.astral.sh/uv/)**
- **Docker** (running daemon required for SWE-bench container execution and evaluation)
- **Free-tier LLM API keys** (no paid plans): e.g. OpenRouter, Groq, Mistral AI, Google AI Studio, or Together AI

### Installation

```bash
# Install student agent dependencies
cd student
uv sync

# Install moulinette evaluation tools (optional, for local grading)
cd ../moulinette
uv sync
```

### Configuration (`.env`)

Copy `.env.example` to `student/.env` (or repository root `.env`) and fill in your free-tier API keys. Multiple keys per provider can be comma-separated for automatic rotation on rate limits (`HTTP 429` / `402`):

```bash
cp .env.example student/.env
```

```env
OPENROUTER_API_KEY=sk-or-v1-key1,sk-or-v1-key2
GROQ_API_KEY=gsk_key1,gsk_key2
```

### Running the Interactive Sandbox CLI

```bash
cd student

# Standalone REPL with default SandboxConfig
uv run sandbox

# With a custom JSON configuration
uv run sandbox sandbox_template.json

# Connected to the MBPP MCP server (stdio)
uv run sandbox --mcp-stdio "uv run python ../mcp_tools_mbpp.py" sandbox_template.json

# Connected to the SWE-bench MCP server (stdio)
uv run sandbox --mcp-stdio "uv run python ../mcp_tools_swebench.py" sandbox_template.json

# Connected to an MCP server over streamable HTTP
uv run python ../mcp_tools_swebench.py --http --host 127.0.0.1 --port 8001
uv run sandbox --mcp-server http://127.0.0.1:8001/mcp
```

Type Python code at the `>>>` prompt; exit cleanly with `exit` or `Ctrl+D`.

### Running the MBPP Agent

```bash
# 1. Dump an MBPP task
cd moulinette
uv run moulinette_eval dump mbpp --output ../cache/mbpp_task.json

# 2. Run the agent
cd ../student
uv run python -m agent_mbpp \
  --task-file ../cache/mbpp_task.json \
  --output ../cache/mbpp_solution.json \
  --model-name "qwen/qwen3.8-27b" \
  --provider-url "https://api.groq.com/openai/v1"

# 3. Validate correctness & metrics
cd ../moulinette
uv run moulinette_eval validate mbpp ../cache/mbpp_task.json ../cache/mbpp_solution.json
```

### Running the SWE-bench Agent

```bash
# 1. Dump a SWE-bench task
cd moulinette
uv run moulinette_eval dump swebench --task-id sympy__sympy-14711 --output ../cache/swebench_task.json

# 2. Run the agent (automatically pulls/starts/cleans up the Docker container)
cd ../student
uv run python -m agent_swebench \
  --task-file ../cache/swebench_task.json \
  --output ../cache/swebench_solution.json \
  --model-name "qwen/qwen3.8-27b" \
  --provider-url "https://api.groq.com/openai/v1"

# 3. Validate solution & metrics
cd ../moulinette
uv run moulinette_eval validate swebench ../cache/swebench_task.json ../cache/swebench_solution.json
```

### Running Unit Tests

```bash
cd student
uv run python -m unittest discover -s tests -v
```

---

## System Architecture

The codebase is organized into independent, composable layers:

```text
task.json (MBPPTaskInput / SWEBenchTaskInput)
  -> benchmark agent CLI (student/agent_mbpp.py | student/agent_swebench.py)
  -> shared agent loop (student/agent/loop.py)
       ├── dynamic system prompt + MCP tools manual (student/agent/prompts.py)
       ├── multi-provider LLM client + key rotator (student/llm/client.py, rotator.py)
       ├── multi-format code extractor (student/agent/extractor.py)
       ├── execution sandbox (student/sandbox/sandbox.py, security.py)
       │     ├── built-in final_answer()
       │     └── MCP tool wrappers (student/mcp_client/client.py)
       │           └── stdio / streamable-HTTP -> MCP Server
       │                 (mcp_tools_mbpp.py | mcp_tools_swebench.py)
       │                   └── standalone tools (student/tools/*.py)
       │                         └── Docker testbed bridge (student/docker_env/container.py)
       └── metrics tracker (student/llm/metrics.py)
  -> solution.json (SolutionOutput + StepMetrics)
```

Key directories and entrypoints:

- `mcp_tools_mbpp.py` / `mcp_tools_swebench.py`: root-level MCP servers exposing tools, resources, and prompts over both `stdio` and `streamable-http`.
- `student/agent/`: orchestration loop (`loop.py`), prompt & dynamic MCP manual builder (`prompts.py`), and multi-format extractor (`extractor.py`).
- `student/sandbox/`: security-hardened Python executor (`sandbox.py`), policy wrappers (`security.py`), interactive REPL CLI (`cli.py`), and JSON config (`sandbox_template.json`).
- `student/llm/`: OpenAI-compatible HTTP client (`client.py`), comma-separated multi-key rotator (`rotator.py`), and `MetricsTracker` enforcing benchmark limits (`metrics.py`).
- `student/mcp_client/`: synchronous facade over the async MCP SDK (`client.py`) supporting dynamic tool, resource, and prompt discovery for any MCP server.
- `student/tools/`: standalone, independently testable tools (`fs_tools.py`, `search_tools.py`, `exec_tools.py`, `mbpp_tools.py`).
- `student/docker_env/`: Docker container lifecycle manager with guaranteed cleanup on exit, `SIGTERM`, and `SIGINT` (`container.py`).
- `student/models_public.py`: shared Pydantic models (`SandboxConfig`, `StepMetrics`, `SolutionOutput`, `MBPPTaskInput`, `SWEBenchTaskInput`).

---

## Agent Loop

The central loop in `student/agent/loop.py` implements **Thought -> Code -> Observation** without any external agent orchestration library:

1. **System prompt generation**: `build_system_prompt()` combines the sandbox rules (allowed imports from `SandboxConfig`) with a dynamically generated tools manual built directly from the connected MCP server's JSON schemas.
2. **Observation pruning**: `_prune_old_observations()` preserves the full text of the most recent `keep_recent_turns` observations (default `3`) and compacts older turns into short placeholders, keeping input token usage well within benchmark limits while preserving the model's own thoughts and actions.
3. **LLM generation**: `LLMClient.generate()` calls the configured OpenAI-compatible endpoint with `stop=["Observation:"]` to prevent hallucinated observations. On `HTTP 429` / `402`, `KeyRotator` rotates immediately through configured API keys before applying bounded backoff.
4. **Code extraction & format translation**: `extract_code()` parses primary ```` ```python ```` blocks and also translates malformed/unclosed blocks, Anthropic XML `<invoke>`, JSON/Hermes `<tool_call>`, ReAct `Action / Action Input`, and Liquid tool calls into equivalent Python code, returning an explicit warning whenever translation or recovery occurred. If no code is found, explicit feedback is returned to the LLM.
5. **Sandboxed execution**: `execute()` runs the extracted Python inside the sandbox namespace with `final_answer()` and dynamically discovered MCP wrappers. Bare top-level expressions (e.g. `read_file(...)`) are auto-printed via AST rewriting so results always appear in `Observation`. Outputs exceeding 4,000 characters keep both the first 1,500 and last 2,500 characters (preserving pytest/unittest verdicts at the end) with an explicit truncation notice.
6. **Termination & metrics**: When `final_answer(value)` is called, `FinalAnswerException` (inheriting from `BaseException` so `except Exception:` cannot swallow it) terminates the loop cleanly. Every turn records `StepMetrics` (tokens, latency, retries, `api_url`, `model_name`, `llm_output`, `sandbox_input`, `sandbox_output`) and checks hard limits (`10` iterations / `6,000` in / `1,500` out / `120s` for MBPP; `30` iterations / `300,000` in / `10,000` out / `900s` for SWE-bench).

---

## Sandbox Design

The sandbox (`student/sandbox/sandbox.py`, `student/sandbox/security.py`) enforces defense-in-depth using **only Python standard library modules and builtins** (no external sandboxing packages):

- **Import allowlist (`make_safe_import`)**: Replaces `builtins.__import__` with an allowlist validator supporting exact names (`"math"`) and submodule wildcards (`"collections.*"`). Both top-level imports and `from pkg import submod` submodules are checked.
- **Filesystem allowlist (`make_safe_open`)**: Replaces `builtins.open` with a path validator resolving symlinks and relative segments via `os.path.realpath` against `SandboxConfig.allowed_directories` (preventing `../` traversal). Standard stream descriptors (`0`, `1`, `2`) are permitted; arbitrary FDs are blocked.
- **Restricted builtins & AST introspection guard**: Removes `eval`, `exec`, `compile`, `breakpoint`, `input`, `globals`, `vars`, `help`, `exit`, `quit`, etc. A pre-execution `SecurityASTValidator` blocks dunder escape attributes (`__subclasses__`, `__globals__`, `__builtins__`, `__code__`, `__bases__`, `__mro__`, `__closure__`, etc.), including via `getattr(...)`.
- **Network isolation (`NetworkBlocker`)**: Overrides `socket.socket`, `socket.create_connection`, and `socket.getaddrinfo` during execution to block outbound and inbound connections, restoring them in a `finally` block.
- **Timeout & memory limits**: Arms `signal.SIGALRM` for `max_execution_time_seconds` (returning partial stdout with an explicit timeout explanation) and caps virtual address space via `resource.setrlimit(RLIMIT_AS, ...)` for `max_memory_mb`.
- **Exception propagation**: `KeyboardInterrupt` and `SystemExit` are never caught by the sandbox error handler so they propagate to the caller.
- **Security domain boundary with MCP tools**: `final_answer()` is injected directly by the sandbox (never via MCP). External MCP tool wrappers temporarily pause `SIGALRM`, `NetworkBlocker`, and `RLIMIT_AS` while the MCP server executes out-of-process (e.g. running Docker tests or communicating over streamable HTTP), then immediately re-arm all restrictions when control returns to untrusted code.

---

## Tool Implementation Details

All tool modules live in `student/tools/` and work completely independently of the agent loop and MCP servers. Relative paths and `/testbed/...` paths resolve against the `TESTBED_PATH` environment variable (falling back to the current working directory).

### File System Tools (`student/tools/fs_tools.py`)

- `read_file(filepath, start_line=1, end_line=-1)`: Reads numbered lines (`<line_number>: <line_content>`), 1-indexed, clamping `end_line` to file length and capping unbounded reads (`end_line=-1`) at 200 lines with a truncation hint to avoid context bloat.
- `edit_file(filepath, old_str, new_str)`: Performs an exact single-occurrence replacement (rejects empty `old_str` or ambiguous multiple matches). For `.py` files, automatically runs `ast.parse()` after writing and appends an explicit `WARNING` if the edit introduced a `SyntaxError`.
- `list_files(directory=".", pattern="*")`: Recursively lists sorted files matching a glob pattern, skipping `.git`, `__pycache__`, virtual environments, and build caches, capped at 100 entries with a truncation notice.

### Code Search Tools (`student/tools/search_tools.py`)

- `search_code(pattern, file_pattern="*")`: Grep-like regex/literal search skipping ignored directories and binary files (extension + null-byte check). Formats results as `/abs/path.py:<line> <content>` and caps output at 50 matches with a refinement notice.
- `search_function_or_class_definition_in_code(name)`: Parses Python files with `ast.parse` to locate exact `def`, `async def`, and `class` definitions (with a regex fallback for files with syntax errors), returning `search_code`-formatted lines.
- `find_references(name, filepath=None, line=None)`: Whole-word (`\bname\b`) symbol reference search across the codebase, formatted identically to `search_code` and capped at 50 matches.

### Execution Tools (`student/tools/exec_tools.py`)

- `run_command(command, workdir=None, timeout=120.0)`: Executes a shell command in `workdir` (or inside the active SWE-bench Docker container when `DOCKER_CONTAINER_ID` is set) and returns `{"stdout", "stderr", "exit_code"}`.
- `get_patch(repo_path=None)`: Runs `git -c core.fileMode=false diff` (falling back to `diff HEAD` if changes were staged) to produce a clean unified diff.
- `run_tests(test_command=None, workdir=None, timeout=300.0)`: Syncs host edits into the running Docker container (streaming modified files via `docker exec -i` to stay compatible with both standard and rootless Docker daemons), then executes the task's `EVAL_SCRIPT` (or explicit `test_command` / `EVAL_SCRIPT_PATH` / `eval.sh` / `pytest`).

### MBPP Tool (`student/tools/mbpp_tools.py`)

- `run_tests(code, test_list, test_imports=[])`: Runs candidate code against assertions inside an isolated subprocess (`python -I`, new process group, 10s timeout, 512 MB memory limit) and returns a JSON string `{"success": bool, "output": str}` with per-assertion diagnostics.

### MCP Integration (`mcp_tools_mbpp.py`, `mcp_tools_swebench.py`, `student/mcp_client/client.py`)

Both MCP servers use the official Python `mcp` SDK (`MCPServer`) and expose:
- **Tools**: benchmark-specific tool functions (`@mcp.tool()`).
- **Resources**: `mbpp://info` and `swebench://repo-info` (`@mcp.resource()`).
- **Prompts**: `mbpp_workflow` and `swebench_workflow` (`@mcp.prompt()`).

`MCPClient` supports both `stdio` and `streamable-http` transports, dynamically discovers tools/resources/prompts from any unknown MCP server, and generates Python callable wrappers for the sandbox namespace.

---

## Benchmark Results and Analysis

- **MBPP validation**: `5/5` random tasks passed (threshold: `4/5`) using `qwen/qwen3.8-27b` on Groq in `2-3` iterations, `1,538-2,690` input tokens (limit `6,000`), `246-588` output tokens (limit `1,500`), and `1.8s-16.1s` wall-clock time (limit `120s`).
- **SWE-bench benchmark (7 models across 3 free providers: Mistral AI, Groq, OpenRouter)**:
  - **`codestral-latest` (Mistral AI)**: **3/3 (100%)** passed (`4-12` iterations, `7.2k-31.1k` input tokens, `265-1,071` output tokens).
  - **`qwen/qwen3.8-27b` (Groq)**: **2/3 (67%)** passed (`7-11` iterations; fixed and passed tests on task 3 at step 4 before Groq's daily token cap).
  - **`ministral-14b-latest` (Mistral AI)**: **2/3 (67%)** passed (`4-5` iterations on passing tasks).
  - **`openai/gpt-oss-120b` (Groq)**: **1/3 (33%)** passed (reasoning tokens exceeded the 10k output cap on `sympy__sympy-14711`).
  - **`nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter)**: **0/3** (hit 30-iter cap on `14711` and OpenRouter's 50 req/day free limit).
- **Ablation study (observation pruning `keep_recent_turns=3` vs `keep_recent_turns=99` on `codestral-latest`)**:
  - Preserving full observation history kept **3/3 (100%)** pass rate while reducing total iterations by **29%** (`24 -> 17`), input tokens by **39%** (`65,051 -> 39,374`), output tokens by **45%** (`2,105 -> 1,150`), and wall-clock time by **37%** (`647.7s -> 408.0s`) by eliminating redundant file re-reads.
- Full tables, provider reliability metrics, intermediary metrics, and links to all backing `solution.json` files are documented in [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md).

---

## Resources

### Technical References
- [Model Context Protocol (MCP) Specification & Python SDK](https://modelcontextprotocol.io/)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://www.swebench.com/)
- [MBPP: Mostly Basic Python Problems](https://github.com/google-research/google-research/tree/master/mbpp)
- [Executable Code Actions Elicit Better LLM Agents (CodeAct, Wang et al., 2024)](https://arxiv.org/abs/2402.01030)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023)](https://arxiv.org/abs/2210.03629)
- [Python `ast`, `resource`, `signal`, and `subprocess` Standard Library Documentation](https://docs.python.org/3/library/)
- [uv Python Package Manager Documentation](https://docs.astral.sh/uv/)

### AI Usage
AI coding assistants were used as collaborative development tools during this project, with all architecture decisions, code review, debugging, and verification performed by the team:
- **Unit testing & edge cases**: drafting test cases in `student/tests/` for sandbox escape attempts, code extraction formats, file/search/execution tools, and Docker container lifecycle mocks.
- **Debugging & refinement**: diagnosing provider rate-limit handling (`HTTP 429` backoff), rootless Docker file synchronization (`lchown` avoidance via `docker exec -i`), and prompt/truncation tuning for SWE-bench token efficiency.
- **Documentation & benchmark scripting**: automating multi-model SWE-bench runs, extracting intermediary metrics from `solution.json` traces, and structuring `README.md` and `BENCHMARK_REPORT.md`.
