# Agent Smith — Model Benchmark Report

This report evaluates **7 models across 3 free-tier LLM providers** (**Mistral AI**, **Groq**, and **OpenRouter**) on the same set of **3 SWE-bench Verified tasks**, followed by a controlled **ablation study** measuring the impact of observation-history pruning on the best-performing model.

All backing `solution.json` traces are stored in [`benchmark_results/`](benchmark_results/) (structured as `benchmark_results/<provider>__<model>/<task_id>/solution.json`), and machine-readable aggregated metrics are saved in [`benchmark_results/summary.json`](benchmark_results/summary.json).

---

## 1. Setup

### Models & Providers Compared

All runs relied **exclusively on free-tier API quotas** (`temperature=0.0`, `max_iterations=30`, `max_tokens=1536`, hard limits enforced per Section VI.1: 30 iterations / 300,000 input tokens / 10,000 output tokens / 900s).

| # | Provider | Endpoint | Model Identifier | Profile |
|---|----------|----------|------------------|---------|
| 1 | **Mistral AI** | `https://api.mistral.ai/v1` | `codestral-latest` | Code-specialized, non-reasoning |
| 2 | **Mistral AI** | `https://api.mistral.ai/v1` | `ministral-14b-latest` | Compact 14B instruct model |
| 3 | **Mistral AI** | `https://api.mistral.ai/v1` | `mistral-medium-latest` | General flagship (free Experiment tier) |
| 4 | **Groq** | `https://api.groq.com/openai/v1` | `qwen/qwen3.8-27b` | 27B instruct, non-reasoning |
| 5 | **Groq** | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` | 120B reasoning model |
| 6 | **Groq** | `https://api.groq.com/openai/v1` | `openai/gpt-oss-20b` | 20B reasoning/tool-calling model |
| 7 | **OpenRouter** | `https://openrouter.ai/api/v1` | `nvidia/nemotron-3-ultra-550b-a55b:free` | 550B MoE reasoning model |

### Selected SWE-bench Tasks

Three tasks from **SWE-bench Verified** (recommended in Section V.4 for cross-model benchmarking) were chosen to span two distinct repositories, different bug classes, and reproducible fast test suites:

1. **`sympy__sympy-14711`** (`sympy/sympy`): `Vector.__add__` raises `TypeError` when summing vectors with integer `0` (e.g. `sum([N.x, 0 * N.x])`). Tests codebase navigation, reading dunder methods, and running SymPy's `bin/test`.
2. **`sympy__sympy-13480`** (`sympy/sympy`): `NameError: name 'cotm' is not defined` inside `coth.eval()` in `sympy/functions/elementary/hyperbolic.py`. Tests targeted single-line bug localization and discipline to submit quickly without over-exploring.
3. **`pydata__xarray-4629`** (`pydata/xarray`): `xr.merge(..., combine_attrs="override")` mutates the first input dataset's `.attrs` dictionary in-place instead of copying it. Tests multi-file search in a different scientific Python repository (`pytest`).

### Evaluation Protocol

Every produced `solution.json` was verified on two axes:
- **Correctness (`Pass/Fail`)**: applying the submitted `solution` patch inside the official SWE-bench Docker image and running the full task `eval_script` (exit code `0` = `PASS`).
- **Metrics compliance**: running `uv run moulinette_eval validate_metrics swebench <solution.json>` against all Section VI.1 limits.

---

## 2. Results Table (Model × Task)

Baseline agent configuration (`keep_recent_turns=3` observation pruning). A task is marked **PASS** only if **both** the Docker `eval_script` passed **and** all moulinette metrics were `VALID`.

| Model (Provider) | Task | Result | Iterations | Input Tokens | Output Tokens | Wall-Clock Time | Backing `solution.json` |
|---|---|---|---:|---:|---:|---:|---|
| `codestral-latest` (Mistral) | `sympy__sympy-14711` | **PASS** | 12 | 31,133 | 1,071 | 324.0s | [solution.json](benchmark_results/mistral__codestral-latest/sympy__sympy-14711/solution.json) |
| `codestral-latest` (Mistral) | `sympy__sympy-13480` | **PASS** | 4 | 7,247 | 265 | 96.3s | [solution.json](benchmark_results/mistral__codestral-latest/sympy__sympy-13480/solution.json) |
| `codestral-latest` (Mistral) | `pydata__xarray-4629` | **PASS** | 8 | 26,671 | 769 | 227.4s | [solution.json](benchmark_results/mistral__codestral-latest/pydata__xarray-4629/solution.json) |
| `ministral-14b-latest` (Mistral) | `sympy__sympy-14711` | FAIL *(broken pipe after 420s timeout)* | 16 | 54,666 | 2,358 | 454.9s | [solution.json](benchmark_results/mistral__ministral-14b-latest/sympy__sympy-14711/solution.json) |
| `ministral-14b-latest` (Mistral) | `sympy__sympy-13480` | **PASS** | 4 | 7,290 | 331 | 148.7s | [solution.json](benchmark_results/mistral__ministral-14b-latest/sympy__sympy-13480/solution.json) |
| `ministral-14b-latest` (Mistral) | `pydata__xarray-4629` | **PASS** | 5 | 14,624 | 599 | 122.0s | [solution.json](benchmark_results/mistral__ministral-14b-latest/pydata__xarray-4629/solution.json) |
| `qwen/qwen3.8-27b` (Groq) | `sympy__sympy-14711` | **PASS** | 11 | 28,021 | 1,494 | 182.9s | [solution.json](benchmark_results/groq__qwen3.8-27b/sympy__sympy-14711/solution.json) |
| `qwen/qwen3.8-27b` (Groq) | `sympy__sympy-13480` | **PASS** | 7 | 14,729 | 425 | 114.8s | [solution.json](benchmark_results/groq__qwen3.8-27b/sympy__sympy-13480/solution.json) |
| `qwen/qwen3.8-27b` (Groq) | `pydata__xarray-4629` | FAIL *(Groq TPD cap at step 5)* | 5 | 13,452 | 386 | 415.5s | [solution.json](benchmark_results/groq__qwen3.8-27b/pydata__xarray-4629/solution.json) |
| `openai/gpt-oss-120b` (Groq) | `sympy__sympy-14711` | FAIL *(out tokens > 10k)* | 10 | 29,019 | 11,223 | 165.2s | [solution.json](benchmark_results/groq__gpt-oss-120b/sympy__sympy-14711/solution.json) |
| `openai/gpt-oss-120b` (Groq) | `sympy__sympy-13480` | **PASS** | 3 | 4,732 | 1,066 | 71.1s | [solution.json](benchmark_results/groq__gpt-oss-120b/sympy__sympy-13480/solution.json) |
| `openai/gpt-oss-120b` (Groq) | `pydata__xarray-4629` | FAIL *(HTTP 413 TPM > 8k)* | 3 | 13,099 | 2,037 | 69.6s | [solution.json](benchmark_results/groq__gpt-oss-120b/pydata__xarray-4629/solution.json) |
| `nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter) | `sympy__sympy-14711` | FAIL *(max 30 iters)* | 30 | 101,457 | 4,509 | 119.9s | [solution.json](benchmark_results/openrouter__nemotron-3-ultra-550b/sympy__sympy-14711/solution.json) |
| `nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter) | `sympy__sympy-13480` | FAIL *(50 req/day cap)* | 15 | 37,313 | 1,610 | 198.3s | [solution.json](benchmark_results/openrouter__nemotron-3-ultra-550b/sympy__sympy-13480/solution.json) |
| `nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter) | `pydata__xarray-4629` | FAIL *(50 req/day cap)* | 0 | 0 | 0 | 136.7s | [solution.json](benchmark_results/openrouter__nemotron-3-ultra-550b/pydata__xarray-4629/solution.json) |
| `openai/gpt-oss-20b` (Groq) | All 3 tasks | FAIL *(HTTP 400 `tool_use_failed`)* | 0 | 0 | 0 | 1.3–1.6s | [14711](benchmark_results/groq__gpt-oss-20b/sympy__sympy-14711/solution.json) / [13480](benchmark_results/groq__gpt-oss-20b/sympy__sympy-13480/solution.json) / [4629](benchmark_results/groq__gpt-oss-20b/pydata__xarray-4629/solution.json) |
| `mistral-medium-latest` (Mistral) | All 3 tasks | FAIL *(HTTP 429 tier cap)* | 0 | 0 | 0 | 137.2–137.8s | [14711](benchmark_results/mistral__mistral-medium-latest/sympy__sympy-14711/solution.json) / [13480](benchmark_results/mistral__mistral-medium-latest/sympy__sympy-13480/solution.json) / [4629](benchmark_results/mistral__mistral-medium-latest/pydata__xarray-4629/solution.json) |

### Per-Model Summary

| Model | Provider | Solve Rate | Avg Iterations (completed) | Total Input Tokens | Total Output Tokens |
|---|---|---:|---:|---:|---:|
| **`codestral-latest`** | Mistral AI | **3 / 3 (100%)** | **8.0** | 65,051 | 2,105 |
| **`qwen/qwen3.8-27b`** | Groq | **2 / 3 (67%)** | **7.7** | 56,202 | 2,305 |
| **`ministral-14b-latest`** | Mistral AI | **2 / 3 (67%)** | **8.3** | 76,580 | 3,288 |
| **`openai/gpt-oss-120b`** | Groq | **1 / 3 (33%)** | **5.3** | 46,850 | 14,326 |
| **`nvidia/nemotron-3-ultra-550b-a55b:free`** | OpenRouter | **0 / 3 (0%)** | **22.5** | 138,770 | 6,119 |
| **`openai/gpt-oss-20b`** | Groq | **0 / 3 (0%)** | N/A | 0 | 0 |
| **`mistral-medium-latest`** | Mistral AI | **0 / 3 (0%)** | N/A | 0 | 0 |

---

## 3. Provider Reliability

| Model | Provider | Avg Request Latency | Total Retries (429/errors) | Availability During Benchmark | Observed Failure Mode |
|---|---|---:|---:|---|---|
| `codestral-latest` | Mistral AI | **877 ms** | 43 (across 24 steps) | **100%** (3/3 tasks completed) | Free tier 1 req/s rate limit (cleanly handled by retry backoff) |
| `ministral-14b-latest` | Mistral AI | **1,460 ms** | 43 (across 25 steps) | **67%** (2/3 completed, 1 broken pipe after timeout) | 1 req/s rate limit + long exploration loop on `14711` (16 steps) |
| `mistral-medium-latest` | Mistral AI | N/A | 21 (7 × 3 failed attempts) | **0%** (blocked on free tier) | Persistent `HTTP 429` (`code: 1300`) on Experiment free tier |
| `qwen/qwen3.8-27b` | Groq | **777 ms** | 17 (across 23 steps) | **83%** (2/3 completed, TPD cap at step 5 of task 3) | Fast per-request, but 200k tokens/day (TPD) & ~6k TPM limits |
| `openai/gpt-oss-120b` | Groq | **2,516 ms** | 15 (across 16 steps) | **67%** (1 pass, 1 out-token cap, 1 HTTP 413) | Reasoning tokens eat output budget; strict 8,000 TPM single-request cap (`HTTP 413`) |
| `openai/gpt-oss-20b` | Groq | N/A | 0 (immediate HTTP 400) | **0%** (rejected at step 1) | Emits native tool calls despite `tool_choice=none` → `HTTP 400 tool_use_failed` *(subsequently mitigated in `LLMClient`)* |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter | **3,712 ms** | 0 (until daily wall) | **50%** (45/50 daily requests consumed) | Strict **50 requests/day** free-tier ceiling (`free-models-per-day`); 4× slower than Groq/Mistral |

**Key reliability takeaway**: Multi-provider abstraction and retry/rotation logic are essential on free tiers. **Mistral AI (`codestral-latest`)** had the highest sustained availability despite 1 req/s throttling; **Groq** offered the lowest latency (~780 ms) but enforces tight per-minute (`TPM`) and per-day (`TPD`) token caps; **OpenRouter's `:free` tier** is bottlenecked by a hard 50-request daily ceiling per account.

---

## 4. Intermediary Metrics

Measured directly from the recorded `steps` in each `solution.json` trace:

- **Exploration efficiency**: Step at which the agent first reads the target file (`First Read`) and first edits it (`First Edit`).
- **Submission discipline**: Step at which `run_tests()` first passes (`Tests Pass`) vs `final_answer()` step (`Submit`), and the gap $\Delta = \text{Submit} - \text{Tests Pass}$ (note: because `run_tests()` and `final_answer(get_patch())` are separate turns in our loop, $\Delta = 1$ means immediate submission on the very next turn with zero wasted steps; $\Delta = 0$ occurs if both happen in the same code block).

| Model | Task | First Read Target | First Edit Target | Tests First Pass | `final_answer` Step | $\Delta$ (Pass → Submit) | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `codestral-latest` | `sympy__sympy-14711` | 3 | 10 | 11 | 12 | **1** | PASS |
| `codestral-latest` | `sympy__sympy-13480` | 1 | 2 | 3 | 4 | **1** | PASS |
| `codestral-latest` | `pydata__xarray-4629` | 3 | 6 | 7 | 8 | **1** | PASS |
| `ministral-14b-latest` | `sympy__sympy-14711` | 4 | — | — | — | N/A | FAIL (broken pipe) |
| `ministral-14b-latest` | `sympy__sympy-13480` | 1 | 2 | 3 | 4 | **1** | PASS |
| `ministral-14b-latest` | `pydata__xarray-4629` | 2 | 3 | 4 | 5 | **1** | PASS |
| `qwen/qwen3.8-27b` | `sympy__sympy-14711` | 1 | 8 | 9 | 11 | 2 | PASS |
| `qwen/qwen3.8-27b` | `sympy__sympy-13480` | 1 | 2 | 5 | 7 | 2 | PASS |
| `qwen/qwen3.8-27b` | `pydata__xarray-4629` | 1 | 2 | 4 | *(TPD cap)* | N/A | FAIL (partial progress) |
| `openai/gpt-oss-120b` | `sympy__sympy-14711` | 8 | — | — | — | N/A | FAIL (out tokens) |
| `openai/gpt-oss-120b` | `sympy__sympy-13480` | 2 | 3 | *(skipped)* | 3 | N/A | PASS |
| `openai/gpt-oss-120b` | `pydata__xarray-4629` | 3 | — | — | — | N/A | FAIL (HTTP 413) |
| `nemotron-3-ultra-550b` | `sympy__sympy-14711` | 2 | — | — | — | N/A | FAIL (30 iters) |
| `nemotron-3-ultra-550b` | `sympy__sympy-13480` | 1 | 2 | — | — | N/A | FAIL (daily cap) |

**Analysis of intermediary behavior:**
- **Exploration efficiency**: `qwen/qwen3.8-27b` consistently found and read the exact target file on **Step 1** across all 3 tasks (and even fixed + passed `pydata__xarray-4629` tests by Step 4 before hitting Groq's daily token cap). `codestral-latest` located the target file within Steps 1–3, and `ministral-14b-latest` within Steps 1–4.
- **Submission discipline**: `codestral-latest` and `ministral-14b-latest` achieved **$\Delta = 1$ on 100% of passing runs** — submitting `final_answer(get_patch())` immediately after `run_tests()` passed. In contrast, `qwen/qwen3.8-27b` consistently spent an extra turn inspecting `get_patch()` before calling `final_answer(get_patch())` ($\Delta = 2$), which cost it the 3rd task when its Groq daily token quota ran out right between Step 4 (`run_tests()` pass) and submission.
- **Failure diagnosis on `nemotron-3-ultra-550b`**: Despite locating the right file on Step 1–2, it failed to convert exploration into a clean `edit_file` + `final_answer(get_patch())` sequence, burning 30 iterations and 101k input tokens on `sympy__sympy-14711`.

---

## 5. Ablation Study: Observation History Pruning (`keep_recent_turns=3` vs `keep_recent_turns=99`)

### Hypothesis & Experimental Change

Our agent loop includes `_prune_old_observations(messages, keep_recent_turns=k)`, which replaces tool outputs older than `k` turns with a compact placeholder to conserve input tokens. We tested whether aggressive pruning (`k=3`, **Before**) actually improves efficiency on SWE-bench compared to preserving full observation history (`k=99`, **After**) using the **exact same model (`codestral-latest` on Mistral AI)** and the **exact same 3 tasks**:

| Task | Config | Result | Iterations | Input Tokens | Output Tokens | Retries | Wall-Clock Time | First Edit |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `sympy__sympy-14711` | **Before (`k=3`, pruning)** | PASS | 12 | 31,133 | 1,071 | 20 | 324.0s | Step 10 |
| `sympy__sympy-14711` | **After (`k=99`, no pruning)** | PASS | **6** | **12,504** | **426** | **9** | **152.7s** | **Step 3** |
| `sympy__sympy-13480` | **Before (`k=3`, pruning)** | PASS | **4** | **7,247** | **265** | 7 | **96.3s** | **Step 2** |
| `sympy__sympy-13480` | **After (`k=99`, no pruning)** | PASS | 5 | 8,841 | 283 | 7 | 121.0s | Step 3 |
| `pydata__xarray-4629` | **Before (`k=3`, pruning)** | PASS | 8 | 26,671 | 769 | 16 | 227.4s | Step 6 |
| `pydata__xarray-4629` | **After (`k=99`, no pruning)** | PASS | **6** | **18,029** | **441** | **10** | **134.3s** | **Step 3** |
| **TOTAL (3 tasks)** | **Before (`k=3`)** | **3/3** | 24 | 65,051 | 2,105 | 43 | 647.7s | avg 6.0 |
| **TOTAL (3 tasks)** | **After (`k=99`)** | **3/3** | **17 (-29%)** | **39,374 (-39%)** | **1,150 (-45%)** | **26 (-40%)** | **408.0s (-37%)** | **avg 3.0** |

Backing traces for the `k=99` ablation runs:
- [`benchmark_results/ablation_no_pruning__codestral-latest/sympy__sympy-14711/solution.json`](benchmark_results/ablation_no_pruning__codestral-latest/sympy__sympy-14711/solution.json)
- [`benchmark_results/ablation_no_pruning__codestral-latest/sympy__sympy-13480/solution.json`](benchmark_results/ablation_no_pruning__codestral-latest/sympy__sympy-13480/solution.json)
- [`benchmark_results/ablation_no_pruning__codestral-latest/pydata__xarray-4629/solution.json`](benchmark_results/ablation_no_pruning__codestral-latest/pydata__xarray-4629/solution.json)

### Ablation Analysis

This produced a **counterintuitive, high-impact result**:
- Aggressive pruning (`k=3`) was meant to *save* input tokens, yet it actually **increased total input tokens by +65%** (`65,051` vs `39,374`) and **nearly doubled iterations on `sympy__sympy-14711` (`12` vs `6`)**!
- Inspecting the traces explains why: when the model read `vector.py` or `merge.py` early in exploration and then ran 2–3 follow-up search/test commands, `k=3` erased the earlier file read from context, forcing the agent to **re-read files it had already seen** (delaying `First Edit` from Step 3 to Step 10 on `14711`, and from Step 3 to Step 6 on `4629`).
- Because our `read_file()` tool already caps reads at 200 lines and `loop.py` already caps each observation at 4,000 characters, keeping a wider window (`keep_recent_turns=10` in SWE-bench) prevents amnesia loops while staying far below the 300,000 input-token limit (~12k–18k tokens per task, < 6% of the budget).
- **Action taken**: Updated the SWE-bench default in `student/agent_swebench.py` from `keep_recent_turns=3` to `keep_recent_turns=10`.

---

## 6. Conclusions & Final Model Selection

### Selected Models for Final Pipeline

1. **Primary SWE-bench model — `codestral-latest` (Mistral AI)**:
   - Achieved **3/3 (100%)** on SWE-bench Verified in both baseline and ablation configurations.
   - Exceptional token discipline (`12.5k–18.0k` input tokens, `283–441` output tokens with `k=99` — less than 6% of SWE-bench limits) and immediate submission discipline ($\Delta = 1$).
   - High availability on Mistral's free Experiment tier (no strict 50 req/day ceiling).
2. **Primary MBPP & Backup SWE-bench model — `qwen/qwen3.8-27b` (Groq)**:
   - Achieved **5/5 (100%)** on MBPP random evaluation (`2–3` iterations, `1,538–2,690` / `6,000` input tokens, `246–588` / `1,500` output tokens, `1.8–16.1s`).
   - Achieved **2/3 (67%)** on SWE-bench (and reached passing tests on the 3rd task before Groq's daily TPD limit), with the fastest Step-1 localization (`First Read = 1` on all 3 tasks) and lowest latency (`777 ms`).
3. **Lightweight Secondary Backup — `ministral-14b-latest` (Mistral AI)**:
   - Achieved **2/3 (67%)** on SWE-bench in just `4–5` iterations and `7k–14k` input tokens when avoiding long loops, proving a small 14B non-reasoning model can solve real repository bugs cleanly.

### Models Disregarded & Why

- **`openai/gpt-oss-120b` (Groq)**: Reasoning tokens rapidly exhaust the 10,000 output-token cap (`11,223` out tokens in 10 turns on `sympy-14711`), and Groq's 8,000 TPM per-request ceiling triggers `HTTP 413` on larger prompts (`xarray-4629`).
- **`openai/gpt-oss-20b` (Groq)**: Attempts native JSON tool calls even when `tool_choice` is omitted, causing immediate `HTTP 400 tool_use_failed` rejections (0/3).
- **`nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter)**: Slow (`3.7s/req`), failed to converge within 30 iterations on `14711`, and OpenRouter's free 50 requests/day quota is too small for multi-task evaluation.
- **`mistral-medium-latest` (Mistral AI)**: Consistently rate-limited (`HTTP 429`) on the free Experiment tier (`0%` availability).

