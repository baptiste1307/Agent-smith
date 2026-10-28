"""Metrics tracking and evaluation output generation for Agent Smith.

Tracks per-step execution metrics (StepMetrics) and aggregates them into
the final SolutionOutput schema required by the moulinette evaluation.
Enforces hard limits defined in Section VI.1 for MBPP and SWE-bench.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from models_public import SolutionOutput, StepMetrics


@dataclass(frozen=True)
class BenchmarkLimits:
    """Hard execution limits per benchmark as defined in Section VI.1."""

    max_iterations: int
    max_input_tokens: int
    max_output_tokens: int
    max_time_seconds: float


LIMITS: dict[str, BenchmarkLimits] = {
    "mbpp": BenchmarkLimits(
        max_iterations=10,
        max_input_tokens=6_000,
        max_output_tokens=1_500,
        max_time_seconds=120.0,
    ),
    "swebench": BenchmarkLimits(
        max_iterations=30,
        max_input_tokens=300_000,
        max_output_tokens=10_000,
        max_time_seconds=900.0,
    ),
}


class MetricsTracker:
    """Tracks agent execution metrics across iterations and formats SolutionOutput."""

    def __init__(self, task_id: str, benchmark: str) -> None:
        """Initialize the metrics tracker.

        Args:
            task_id: Unique task identifier (MBPP task_id or SWE-bench instance_id).
            benchmark: Target benchmark type ('mbpp' or 'swebench').
        """
        self.task_id = str(task_id)
        self.benchmark = benchmark.lower()
        self.steps: list[StepMetrics] = []
        self._start_perf: float = time.perf_counter()
        self._start_iso: str = datetime.now().isoformat()
        self.system_prompt: str = ""

    @property
    def current_step_number(self) -> int:
        """Return the 1-indexed next iteration number."""
        return len(self.steps) + 1

    @property
    def total_input_tokens(self) -> int:
        """Sum of input tokens across all recorded steps."""
        return sum(s.input_tokens for s in self.steps)

    @property
    def total_output_tokens(self) -> int:
        """Sum of output tokens across all recorded steps."""
        return sum(s.output_tokens for s in self.steps)

    @property
    def total_requests(self) -> int:
        """Total number of LLM requests made, including retries."""
        return sum(1 + s.retries for s in self.steps)

    @property
    def elapsed_time_seconds(self) -> float:
        """Current elapsed wall-clock time in seconds."""
        return time.perf_counter() - self._start_perf

    def record_step(
        self,
        llm_response: Any = None,
        sandbox_input: str = "",
        sandbox_output: str = "",
        step_number: int | None = None,
    ) -> StepMetrics:
        """Record metrics for a single LLM generate -> sandbox execute iteration.

        Args:
            llm_response: LLMResponse instance or dict containing token and latency metadata.
            sandbox_input: Python code sent to the sandbox.
            sandbox_output: Result of the sandbox execution (stdout/stderr/error).
            step_number: Optional explicit 1-indexed step number.

        Returns:
            The recorded StepMetrics instance.
        """
        step_idx = step_number if step_number is not None else self.current_step_number

        input_tokens = 0
        output_tokens = 0
        request_time_ms = 0.0
        retries = 0
        model_name = ""
        api_url = ""
        llm_output = ""

        if llm_response is not None:
            if hasattr(llm_response, "input_tokens"):
                input_tokens = getattr(llm_response, "input_tokens", 0)
                output_tokens = getattr(llm_response, "output_tokens", 0)
                request_time_ms = getattr(llm_response, "request_time_ms", 0.0)
                retries = getattr(llm_response, "retries", 0)
                model_name = getattr(llm_response, "model_name", "")
                api_url = getattr(llm_response, "api_url", "")
                llm_output = getattr(llm_response, "content", "")
            elif isinstance(llm_response, dict):
                input_tokens = llm_response.get("input_tokens", 0)
                output_tokens = llm_response.get("output_tokens", 0)
                request_time_ms = llm_response.get("request_time_ms", 0.0)
                retries = llm_response.get("retries", 0)
                model_name = llm_response.get("model_name", "")
                api_url = llm_response.get("api_url", "")
                llm_output = llm_response.get("content", "")

        step_metric = StepMetrics(
            step=step_idx,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_time_ms=request_time_ms,
            timestamp=datetime.now().isoformat(),
            api_url=api_url,
            model_name=model_name,
            llm_output=llm_output,
            sandbox_input=sandbox_input,
            sandbox_output=sandbox_output,
            retries=retries,
        )

        self.steps.append(step_metric)
        return step_metric

    def check_limits(self) -> tuple[bool, str | None]:
        """Check if any benchmark limit has been exceeded.

        Returns:
            A tuple (exceeded: bool, reason: str | None).
        """
        limits = LIMITS.get(self.benchmark)
        if not limits:
            return False, None

        if len(self.steps) > limits.max_iterations:
            return (
                True,
                f"Iterations ({len(self.steps)}) exceeded limit ({limits.max_iterations})",
            )
        if self.total_input_tokens > limits.max_input_tokens:
            return (
                True,
                f"Input tokens ({self.total_input_tokens}) exceeded limit ({limits.max_input_tokens})",
            )
        if self.total_output_tokens > limits.max_output_tokens:
            return (
                True,
                f"Output tokens ({self.total_output_tokens}) exceeded limit ({limits.max_output_tokens})",
            )
        if self.elapsed_time_seconds > limits.max_time_seconds:
            return (
                True,
                f"Elapsed time ({self.elapsed_time_seconds:.1f}s) exceeded limit ({limits.max_time_seconds}s)",
            )

        return False, None

    def build_solution_output(
        self,
        success: bool,
        solution: str,
        system_prompt: str = "",
        error: str | None = None,
    ) -> SolutionOutput:
        """Construct the validated SolutionOutput object for moulinette evaluation.

        Args:
            success: Whether the agent believes it solved the task.
            solution: Python function code (MBPP) or git patch (SWE-bench).
            system_prompt: Full system prompt sent to the LLM.
            error: Optional error message if the agent failed.

        Returns:
            Validated SolutionOutput instance.
        """
        prompt = system_prompt or self.system_prompt

        return SolutionOutput(
            task_id=self.task_id,
            benchmark=self.benchmark,
            success=success,
            solution=solution,
            iterations=len(self.steps),
            total_requests=self.total_requests,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_time_seconds=round(self.elapsed_time_seconds, 3),
            steps=self.steps,
            system_prompt=prompt,
            error=error,
            timestamp=datetime.now().isoformat(),
        )

    def save_solution(
        self,
        filepath: str | Path,
        success: bool,
        solution: str,
        system_prompt: str = "",
        error: str | None = None,
    ) -> SolutionOutput:
        """Build SolutionOutput and save it to a JSON file.

        Args:
            filepath: Destination file path for solution.json.
            success: Whether the task succeeded.
            solution: Solution code or patch.
            system_prompt: Full system prompt.
            error: Optional error message.

        Returns:
            The saved SolutionOutput instance.
        """
        output = self.build_solution_output(
            success=success,
            solution=solution,
            system_prompt=system_prompt,
            error=error,
        )

        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        return output
