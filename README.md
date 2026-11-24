*This project has been created as part of the 42 curriculum by avauclai, bpasquer, mlorenzo.*

# Agent Smith

## Description

**Agent Smith** is an autonomous code-agent framework built from scratch in Python (no external agent orchestration libraries). Instead of relying on static single-turn generation or JSON-only tool calling, the agent operates through a structured **Thought → Code → Observation** loop driven by **code-based tool calling**:

- The LLM reasons about a programming task and writes executable Python code.
- The extracted code executes inside a configurable, security-hardened Python sandbox (standard library only).
- The code invokes benchmark tools exposed over the **Model Context Protocol (MCP)** as ordinary Python functions, enabling loops, conditionals, and multi-step inspection inside a single turn.
- The sandbox returns real observations (stdout, errors, truncation/syntax warnings) back to the LLM until the agent calls the sandbox built-in `final_answer()`.

The framework solves two benchmarks under strict token, iteration, and wall-clock limits:

- **MBPP (Mostly Basic Python Problems)**: algorithmic Python problems where the agent verifies its implementation against public tests and submits the Python function code.
- **SWE-bench Verified**: real-world GitHub issue resolution inside Dockerized repositories where the agent explores the codebase, applies minimal edits, runs the official evaluation test suite, and submits a unified `git -c core.fileMode=false diff` patch.

---

## Instructions

### Prerequisites

- **Python 3.10+** (verified compatible with Python 3.10 syntax and standard library)
- **[uv](https://docs.astral.sh/uv/)** package manager
- **Docker** daemon running (required for SWE-bench container execution and evaluation)
- **Free-tier LLM API keys** (exclusively free tiers — no paid plans or billing credits): e.g. Mistral AI, Groq, OpenRouter, Google AI Studio, or Together AI

### Installation

```bash
# 1. Install the agent framework dependencies
cd student
uv sync

# 2. Install the moulinette evaluation toolkit (for local dumping & validation)
cd ../moulinette
uv sync
```

### Configuration Files (`.env`, Sandbox, Models)

1. **API keys (`.env`)**: Copy `.env.example` to `student/.env` (or repository root `.env`). Multiple keys per provider can be comma-separated to enable automatic rotation on `HTTP 429` / `402`:

   ```bash
   cp .env.example student/.env
   ```

   ```env
   MISTRAL_API_KEY=key1,key2
   GROQ_API_KEY=gsk_key1,gsk_key2
   OPENROUTER_API_KEY=sk-or-v1-key1,sk-or-v1-key2
   ```

   API keys are always loaded from environment variables / `.env` files (compatible with `./exam_TYPE.sh --env-file /path/to/.env`) and are **never hardcoded** in source files.

2. **Sandbox configuration (`student/sandbox_template.json`)**: Configures `authorized_imports` (allowlist with `.*` submodule wildcards), `allowed_directories`, `max_execution_time_seconds`, and `max_memory_mb` validated via Pydantic (`SandboxConfig`).
3. **Model & provider configuration (`student/models_config.json`)**: Documents supported free-tier providers, endpoints, environment variables, recommended models per benchmark, and Section VI.1 limits.

### Running the Interactive Sandbox CLI

```bash
cd student

# Launch standalone interactive REPL (default config)
uv run sandbox

# With custom JSON configuration
uv run sandbox sandbox_template.json

# With MBPP MCP tools (stdio — automatically resolves repo-root mcp_tools_mbpp.py)
uv run sandbox --mcp-stdio "python mcp_tools_mbpp.py" sandbox_template.json

# With SWE-bench MCP tools (stdio)
uv run sandbox --mcp-stdio "python mcp_tools_swebench.py" sandbox_template.json

# With an MCP server over streamable HTTP
uv run python ../mcp_tools_mbpp.py --http --host 127.0.0.1 --port 8000
uv run sandbox --mcp-server http://127.0.0.1:8000/mcp
```

Type Python expressions or statements at the `>>>` prompt (variables persist across REPL entries, and all sandbox security restrictions remain enforced). Multi-line blocks (`def`, `for`, `try`...) continue on `...` lines and end with an empty line, like the Python REPL. When code is piped on stdin (`cat script.py | uv run sandbox`), it is split into top-level statements, each executed as one REPL entry with its own timeout. The generated tool manual is available as the `sandbox_manual` variable. Exit cleanly with `exit` or `Ctrl+D`.

### Running MCP Servers Standalone

Both root-level MCP servers (`mcp_tools_mbpp.py` and `mcp_tools_swebench.py`) support **stdio** (default) and **streamable HTTP** (`--http`):

```bash
cd student
# stdio transport
uv run python ../mcp_tools_mbpp.py
uv run python ../mcp_tools_swebench.py

# streamable HTTP transport
uv run python ../mcp_tools_mbpp.py --http --host 127.0.0.1 --port 8000
uv run python ../mcp_tools_swebench.py --http --host 127.0.0.1 --port 8001
```

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

# Quick single-command run (dump -> run-agent with 120s timeout -> validate)
cd ../moulinette
./quickstart.sh mbpp --student-path ../student \
  --model-name "qwen/qwen3.8-27b" --provider-url "https://api.groq.com/openai/v1"
```

### Running the SWE-bench Agent

```bash
# 1. Dump a SWE-bench task
cd moulinette
uv run moulinette_eval dump swebench --task-id sympy__sympy-14711 --output ../cache/swebench_task.json

# 2. Run the agent (pulls/starts/cleans up Docker container automatically)
cd ../student
uv run python -m agent_swebench \
  --task-file ../cache/swebench_task.json \
  --output ../cache/swebench_solution.json \
  --model-name "codestral-latest" \
  --provider-url "https://api.mistral.ai/v1"

# 3. Validate solution, metrics, and trace
cd ../moulinette
uv run moulinette_eval validate swebench ../cache/swebench_task.json ../cache/swebench_solution.json
uv run moulinette_eval display ../cache/swebench_solution.json
```

When testing SWE-bench tools in isolation without Docker, set `TESTBED_PATH=/path/to/repo` (or pass `--testbed-path`).

### Running the Unit Test Suite

```bash
cd student
uv run python -m unittest discover -s tests -v
```

---

## System Architecture

The framework enforces strict separation between orchestration, code extraction, sandbox execution, MCP communication, and external tools:

```text
                       +---------------------------------------------------+
                       |                    Agent Code                     |
+-----------+  Prompt  |  +--------------+                                 |
|  LLM API  |<---------+--| Orchestrator | (student/agent/loop.py)         |
| (Groq /   |          |  |  + Metrics   | (student/llm/client,rotator,    |
|  Mistral/ |--------->|  +------+-------+  metrics.py)                    |
|  OR ...)  | Response |         | Raw LLM response                        |
+-----------+          |         v                                         |
                       |  +-----------------+                              |
                       |  | Code Extraction | (student/agent/extractor.py) |
                       |  +--------+--------+                              |
                       |           | Extracted / translated Python code    |
                       |           v                                       |
                       |  +---------------------------------------------+  |
                       |  |                  Sandbox                    |  |
                       |  | (student/sandbox/sandbox.py, security.py)   |  |
                       |  |  +--------------------+   +--------------+  |  |
                       |  |  | Python Interpreter |-->|  MCP Client  |  |  |
                       |  |  | + final_answer()   |   | (client.py)  |  |  |
                       |  |  +--------------------+   +------+-------+  |  |
                       |  +----------------------------------|----------+  |
                       +-------------------------------------|-------------+
                                                             | stdio / streamable-HTTP
                                                             v
                                              +-----------------------------+
                                              |         MCP Server          |
                                              | (mcp_tools_mbpp.py /        |
                                              |  mcp_tools_swebench.py)     |
                                              +--------------+--------------+
                                                             |
                                                             v
                                              +-----------------------------+
                                              | Standalone Tools & Docker   |
                                              | (student/tools/*.py,        |
                                              |  student/docker_env/)       |
                                              +-----------------------------+
```

### Architectural Boundaries

- **Sandbox vs MCP tools**: The sandbox wraps the MCP client, never the reverse. Untrusted LLM-generated Python runs inside the sandbox security domain (import allowlist, path allowlist, restricted builtins, AST check, no network, timeout, memory cap), while MCP tools run outside the sandbox in an external server process.
- **`final_answer()` is a sandbox built-in**: It is always injected into the sandbox namespace regardless of which MCP server is connected (including unknown MCP servers during evaluation) and is **never** an MCP tool.
- **Repository layout**:
  - `mcp_tools_mbpp.py`, `mcp_tools_swebench.py`: root-level MCP servers (tools, resources, prompts; stdio & HTTP).
  - `student/agent/`: orchestration loop (`loop.py`), system/task prompt & dynamic MCP manual builder (`prompts.py`), multi-format code extractor (`extractor.py`).
  - `student/sandbox/`: restricted executor (`sandbox.py`), security primitives (`security.py`), interactive REPL CLI (`cli.py`).
  - `student/llm/`: OpenAI-compatible client (`client.py`), multi-key rotator (`rotator.py`), limit-enforcing metrics tracker (`metrics.py`).
  - `student/mcp_client/`: synchronous facade over async MCP SDK (`client.py`) supporting dynamic discovery on any MCP server.
  - `student/tools/`: standalone, independently testable tool modules (`fs_tools.py`, `search_tools.py`, `exec_tools.py`, `mbpp_tools.py`).
  - `student/docker_env/`: Docker container lifecycle & cleanup manager (`container.py`).
  - `student/models_public.py`, `student/sandbox_template.json`, `student/models_config.json`: Pydantic contracts and JSON configurations.
  - `benchmark_results/`: backing `solution.json` traces for all evaluated models and the ablation study.

---

## Agent Loop Explanation

Implemented from scratch in `student/agent/loop.py`, each iteration performs a full **Thought → Code → Observation** cycle:

1. **Dynamic system prompt (`student/agent/prompts.py`)**: Builds the prompt from `SandboxConfig` (authorized imports) and the connected MCP server's live tool schemas (`build_mcp_tools_manual()`), including structured `Thought` / `Code` / `Observation` format rules and a complete worked reasoning example.
2. **Context & observation management**: `_prune_old_observations(messages, keep_recent_turns)` preserves recent full observations (`default=10` for SWE-bench based on our ablation study; `default=3` for MBPP) and compacts older verbose outputs while retaining error indicators and the model's reasoning chain.
3. **LLM invocation & stop sequences (`student/llm/client.py`, `rotator.py`)**:
   - Sends OpenAI-compatible chat completion requests with `stop=["Observation:"]` (plus post-split safety stripping if a provider ignores `stop`) so the model never hallucinates fake observations.
   - `KeyRotator.auto_detect()` maps the provider URL to standard environment variables (`MISTRAL_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `TOGETHER_API_KEY`, etc.), supports comma-separated multi-token rotation on `HTTP 429` / `402`, applies bounded backoff, and recovers `tool_use_failed` `failed_generation` payloads into `<tool_call>` blocks.
4. **Multi-format code extraction (`student/agent/extractor.py`)**:
   - **(a) Primary Python blocks**: ```` ```python ... ``` ```` (with optional `<end_code>`).
   - **(b) Malformed / untagged / unclosed blocks**: ```` ```py ````, ```` ```Python ````, untagged ```` ``` ````, or unclosed ```` ```python ```` — executed with an explicit warning explaining how it was interpreted.
   - **(c) JSON / Hermes tool calls**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` → translated into Python calls.
   - **(d) XML tool calls (Anthropic-style)**: `<invoke name="..."><parameter name="...">...</parameter></invoke>` → translated into Python calls.
   - **(e) ReAct format**: `Action: <tool>` / `Action Input: {...}` → translated into Python calls.
   - **(f) Liquid tool calls**: `<|tool_call_start|>[...]<|tool_call_end|>` → translated into Python calls.
5. **Explicit feedback to the LLM (zero silent failures)**:
   - Missing code block → raises `ValueError` and returns an explicit corrective message to the LLM.
   - Malformed or translated block → prepends an explicit warning explaining the recovered Python code.
   - Sandbox timeout → returns partial stdout captured before interruption plus an explicit `SandboxTimeoutError` explanation.
   - Output truncation → caps sandbox stdout (10,000 chars) and observation text (4,000 chars, keeping the first 1,500 and last 2,500 chars to preserve test verdicts) with explicit character-count truncation notices.
   - Edit syntax error → `edit_file()` runs `ast.parse()` on `.py` files and warns immediately with the exact line number and `SyntaxError` message.
6. **Termination & provenance logging (`student/llm/metrics.py`)**:
   - Calling `final_answer(value)` first records the answer (first call wins), then raises `FinalAnswerException` to stop the code. Because the answer is recorded before the raise, an `except Exception:` in generated code cannot lose it: it only lets the code continue after the call.
   - Every turn appends a `StepMetrics` entry (`step`, `input_tokens`, `output_tokens`, `request_time_ms`, `timestamp`, `api_url`, `model_name`, `llm_output`, `sandbox_input`, `sandbox_output`, `retries`) and verifies benchmark limits before continuing.
   - On completion (or error), `MetricsTracker.save_solution()` writes a validated `SolutionOutput` JSON file containing the full `system_prompt` and per-step trace for provenance verification.

---

## Sandbox Design

The sandbox (`student/sandbox/sandbox.py`, `student/sandbox/security.py`) enforces all security guarantees using **Python standard library modules and builtins only** (no `RestrictedPython` or external sandboxing packages):

| Security Constraint | Implementation |
|---|---|
| **Import restrictions** | `make_safe_import()` replaces `builtins.__import__` with an allowlist check supporting exact names (`"math"`) and submodule wildcards (`"collections.*"`), validating both top-level imports and `from pkg import submod` targets. |
| **Filesystem restrictions** | `make_safe_open()` replaces `builtins.open`, resolves symlinks and `../` segments via `os.path.realpath`, and restricts access strictly to `SandboxConfig.allowed_directories` (e.g. `["/testbed", "/tmp/agent"]`). Only standard stream FDs (`0`, `1`, `2`) are permitted. |
| **Restricted builtins & AST guard** | Removes dangerous builtins (`eval`, `exec`, `compile`, `breakpoint`, `input`, `globals`, `vars`, `help`, `exit`, `quit`, etc.). Before `exec()`, `SecurityASTValidator` walks the AST to block dunder introspection attributes (`__subclasses__`, `__globals__`, `__builtins__`, `__code__`, `__bases__`, `__base__`, `__mro__`, `__func__`, `__closure__`), including via `getattr(...)`. |
| **Network isolation** | `NetworkBlocker` patches `socket.socket`, `socket.create_connection`, `socket.getaddrinfo`, `socket.socketpair`, and `socket.fromfd` during execution to raise `SandboxSecurityError`, restoring them in `finally`. |
| **Execution timeout** | Arms `signal.SIGALRM` for `config.max_execution_time_seconds`, raising `SandboxTimeoutError` and returning partial stdout captured before interruption. |
| **Memory limit** | Caps virtual address space via `resource.setrlimit(resource.RLIMIT_AS, ...)` to `current_memory_bytes() + max_memory_mb * 1024^2`, catching `MemoryError` and restoring the previous limit in `finally`. |
| **Exception propagation** | `KeyboardInterrupt` and `SystemExit` inherit from `BaseException` and are never caught by `except Exception:`, so they propagate cleanly to the agent loop / CLI. |

### Isolation Trade-Offs & Domain Boundary

- **In-process restricted execution for agent turns**: Running LLM code in-process inside a restricted namespace with `exec()` enables zero-serialization callable MCP tool wrappers, automatic bare-expression printing (`_auto_print_expressions()`), and persistent REPL state (`uv run sandbox`).
- **Out-of-process MCP execution**: External MCP tool wrappers temporarily pause `SIGALRM`, `NetworkBlocker`, and `RLIMIT_AS` while the MCP server executes out-of-process (so long `run_tests()` runs inside Docker and streamable HTTP MCP calls are never killed by sandbox-only limits), and immediately re-arm all limits upon return.
- **Subprocess isolation for MBPP tests**: `student/tools/mbpp_tools.py` runs candidate code in an isolated child process (`python -I`, new process group `start_new_session=True`, `SIGKILL` on 10s timeout, `RLIMIT_AS=512 MB`, `RLIMIT_FSIZE=10 MB`).

---

## Tool Implementation Details

All tools live in `student/tools/` and operate independently of the agent loop and MCP servers. For SWE-bench, relative paths and `/testbed/...` paths resolve against the **`TESTBED_PATH`** environment variable (falling back to the current working directory).

### File System Tools (`student/tools/fs_tools.py`)
- `read_file(filepath, start_line=1, end_line=-1)`: Returns 1-indexed `cat -n`-style lines (`<line_number>: <line_content>`). Clamps `end_line` to file length and caps default unbounded reads (`end_line=-1`) at 200 lines with a truncation hint to prevent context bloat.
- `edit_file(filepath, old_str, new_str)`: Replaces one exact, unique substring (rejects empty `old_str` or ambiguous multiple matches). On `.py` files, runs `ast.parse()` immediately after writing and returns an explicit `WARNING` if a `SyntaxError` was introduced.
- `list_files(directory=".", pattern="*")`: Recursively lists sorted matching files while skipping `.git`, `__pycache__`, virtual environments, and build caches, capped at 100 entries with a truncation message.

### Code Search Tools (`student/tools/search_tools.py`)
All search tools format output as `/absolute/path_to_file.py:<line_number> <line_content>` and cap output at 50 matches with a refinement hint:
- `search_code(pattern, file_pattern="*")`: Regex/literal grep across non-binary, non-ignored files (invalid regex safely falls back to literal escaped search).
- `search_function_or_class_definition_in_code(name)`: Uses Python `ast.parse` to locate exact `def`, `async def`, and `class` definitions, with a regex fallback for syntactically invalid files.
- `find_references(name, filepath=None, line=None)`: Whole-word (`\bname\b`) symbol usage search across the repository.

### Execution Tools (`student/tools/exec_tools.py`) & Docker Bridge (`student/docker_env/container.py`)
- `run_command(command, workdir=None, timeout=120.0)`: Executes shell commands locally or inside the running SWE-bench Docker container (`DOCKER_CONTAINER_ID`), returning JSON `{"stdout", "stderr", "exit_code"}`.
- `get_patch(repo_path=None)`: Runs `git -c core.fileMode=false diff` (falling back to `diff HEAD` if changes were staged) to produce the exact unified diff required by Section V.4.
- `run_tests(test_command=None, workdir=None, timeout=300.0)`: Syncs modified host files directly into `/testbed` inside the container via `docker exec -i` (avoiding both `git apply` fragility and rootless Docker `lchown` UID errors), then executes the task's `EVAL_SCRIPT` (or explicit `test_command` / `EVAL_SCRIPT_PATH` / `/eval.sh` / `pytest`).
- **Docker lifecycle & cleanup**: `DockerContainer` automatically pulls/starts the task image, syncs `/testbed`, and guarantees container removal via `finally`, `atexit`, and `SIGTERM`/`SIGINT` signal handlers (protecting against orphaned containers even under `moulinette_eval run-agent` hard timeouts). Since `kill -9` cannot be caught, a detached watchdog process (own session) removes the container within ~1s of the agent's death, and also stops the agent if its `uv run` parent is killed; as a last resort the container runs `sleep 1800` with `--rm`, so it can never outlive 30 minutes. The repository only ever comes from the task's Docker image: there is no GitHub clone fallback (upstream history contains the fix), so without Docker the task fails with an explicit error in `solution.json`.

### MBPP Tool (`student/tools/mbpp_tools.py`)
- `run_tests(code, test_list, test_imports=[])`: Executes candidate code against assertions inside an isolated Python subprocess and returns a JSON string `{"success": bool, "output": str}` with per-assertion failure diagnostics (including expected vs actual value hints).

### MCP Servers & Client (`mcp_tools_mbpp.py`, `mcp_tools_swebench.py`, `student/mcp_client/client.py`)
- Both servers expose **tools** (`@mcp.tool()`), **resources** (`mbpp://info`, `swebench://repo-info` via `@mcp.resource()`), and **prompts** (`mbpp_workflow`, `swebench_workflow` via `@mcp.prompt()`).
- `MCPClient` connects over **stdio** or **streamable HTTP**, dynamically discovers tools, resources, and prompts on **any (including unknown) MCP server**, generates Python callable wrappers for the sandbox namespace, and exports JSON schemas for dynamic manual generation.

---

## Evaluation Limits & AI Safety

| Benchmark | Max Iterations | Max Input Tokens | Max Output Tokens | Timeout | Pass Threshold |
|---|---:|---:|---:|---:|---|
| **MBPP** | 10 | 6,000 | 1,500 | 120s | 4 / 5 random tasks |
| **SWE-bench** | 30 | 300,000 | 10,000 | 900s | 2 / 3 random tasks |

- **AI Safety & Provenance**: Network access is disabled inside the sandbox (`NetworkBlocker`), prompts explicitly forbid fetching external issues/PRs/patches, and every `solution.json` records `system_prompt`, `llm_output`, `sandbox_input`, `sandbox_output`, `api_url`, `model_name`, and `retries` for full auditability.

---

## Benchmark Results and Analysis

### MBPP Validation
Evaluated on **5 random MBPP tasks** (`seeds 1–5`) with `qwen/qwen3.8-27b` (Groq):
- **Score**: **5 / 5 PASSED** (threshold: `4/5`)
- **Efficiency**: `2–3` iterations (limit `10`), `1,538–2,690` input tokens (limit `6,000`), `246–588` output tokens (limit `1,500`), `1.8s–16.1s` (limit `120s`), and `All checks passed` under `moulinette_eval display`.

### SWE-bench Multi-Model Benchmark (7 Models × 3 Free Providers × 3 Tasks)

Evaluated on `sympy__sympy-14711`, `sympy__sympy-13480`, and `pydata__xarray-4629` (full per-task tables, provider reliability, intermediary metrics, and links to all backing `solution.json` files are in **[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)**):

| Model | Provider | Solve Rate | Avg Iterations | Total Input Tokens | Total Output Tokens | Avg Latency |
|---|---|---:|---:|---:|---:|---:|
| **`codestral-latest`** | Mistral AI | **3 / 3 (100%)** | 8.0 | 65,051 | 2,105 | 877 ms |
| **`qwen/qwen3.8-27b`** | Groq | **2 / 3 (67%)** | 7.7 | 56,202 | 2,305 | 777 ms |
| **`ministral-14b-latest`** | Mistral AI | **2 / 3 (67%)** | 4.5 | 21,914 | 930 | 1,303 ms |
| **`openai/gpt-oss-120b`** | Groq | **1 / 3 (33%)** | 5.3 | 46,850 | 14,326 | 2,516 ms |
| **`nvidia/nemotron-3-ultra-550b-a55b:free`** | OpenRouter | **0 / 3 (0%)** | 22.5 | 138,770 | 6,119 | 3,712 ms |
| **`openai/gpt-oss-20b`** | Groq | **0 / 3 (0%)** | N/A | 0 | 0 | N/A (`HTTP 400`) |
| **`mistral-medium-latest`** | Mistral AI | **0 / 3 (0%)** | N/A | 0 | 0 | N/A (`HTTP 429`) |

### Ablation Study: Observation Pruning (`codestral-latest`, Same 3 Tasks)

| Configuration | Solve Rate | Total Iterations | Total Input Tokens | Total Output Tokens | Total Retries | Total Time |
|---|---:|---:|---:|---:|---:|---:|
| **Before (`keep_recent_turns=3`, aggressive pruning)** | 3 / 3 | 24 | 65,051 | 2,105 | 43 | 647.7s |
| **After (`keep_recent_turns=99`, full history)** | **3 / 3** | **17 (-29%)** | **39,374 (-39%)** | **1,150 (-45%)** | **26 (-40%)** | **408.0s (-37%)** |

**Key findings**:
- **Non-reasoning code/instruct models win under hard token caps**: `codestral-latest` (3/3) and `qwen/qwen3.8-27b` (2/3 SWE-bench, 5/5 MBPP) stayed well below limits, whereas reasoning tokens caused `openai/gpt-oss-120b` to exceed the 10,000 output-token cap (`11,223`).
- **Over-aggressive pruning backfires**: Pruning after 3 turns caused file re-reads on multi-step bugs; widening `keep_recent_turns` cut both iterations (-29%) and total input tokens (-39%). Accordingly, `agent_swebench.py` defaults to `keep_recent_turns=10` and `codestral-latest`.

See **[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)** for the complete analysis and all backing [`benchmark_results/`](benchmark_results/) `solution.json` files.

---

## Resources

### Technical References
- [Model Context Protocol (MCP) Specification & Python SDK](https://modelcontextprotocol.io/)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://www.swebench.com/)
- [MBPP: Mostly Basic Python Problems](https://github.com/google-research/google-research/tree/master/mbpp)
- [Executable Code Actions Elicit Better LLM Agents (CodeAct — Wang et al., 2024)](https://arxiv.org/abs/2402.01030)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023)](https://arxiv.org/abs/2210.03629)
- [Python Standard Library Documentation (`ast`, `resource`, `signal`, `subprocess`, `socket`)](https://docs.python.org/3/library/)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [uv Python Package Manager Documentation](https://docs.astral.sh/uv/)

### AI Usage
AI coding assistants were used responsibly as engineering partners throughout development, while the team retained full ownership of architectural choices, security review, debugging, and verification:
- **Unit testing & security edge cases**: Drafting and expanding unit tests in `student/tests/` covering sandbox escape vectors (dunder introspection, import/path/network/timeout/memory limits), multi-format code extraction, file/search/execution tools, and Docker lifecycle mocks.
- **Debugging & edge-case hardening**: Investigating free-tier provider rate-limit behavior (`HTTP 429` bounded backoff, `tool_use_failed` recovery), rootless Docker file synchronization (`lchown` avoidance via `docker exec -i`), sandbox-vs-MCP timeout isolation, and observation truncation/pruning tuning.
- **Benchmark automation & documentation**: Writing temporary orchestration scripts to run and validate multi-model SWE-bench benchmarks and extract intermediary metrics from `solution.json` traces, and structuring `README.md` and `BENCHMARK_REPORT.md`.
