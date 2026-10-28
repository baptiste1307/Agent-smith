"""Unit tests for llm/metrics.py."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from llm.client import LLMResponse
from llm.metrics import LIMITS, BenchmarkLimits, MetricsTracker
from models_public import SolutionOutput, StepMetrics


class TestMetricsTracker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_step_and_totals(self):
        tracker = MetricsTracker(task_id="101", benchmark="mbpp")
        self.assertEqual(tracker.current_step_number, 1)

        resp1 = LLMResponse(
            content="print('test 1')",
            input_tokens=150,
            output_tokens=50,
            request_time_ms=500.0,
            retries=0,
            model_name="mock-model",
            api_url="https://api.test",
        )

        step1 = tracker.record_step(
            llm_response=resp1,
            sandbox_input="print('test 1')",
            sandbox_output="test 1",
        )

        self.assertEqual(step1.step, 1)
        self.assertEqual(step1.input_tokens, 150)
        self.assertEqual(step1.output_tokens, 50)
        self.assertEqual(step1.retries, 0)
        self.assertEqual(tracker.total_input_tokens, 150)
        self.assertEqual(tracker.total_output_tokens, 50)
        self.assertEqual(tracker.total_requests, 1)

        # Second step with a retry
        resp2 = LLMResponse(
            content="final_answer('done')",
            input_tokens=200,
            output_tokens=30,
            request_time_ms=400.0,
            retries=1,
            model_name="mock-model",
            api_url="https://api.test",
        )

        step2 = tracker.record_step(
            llm_response=resp2,
            sandbox_input="final_answer('done')",
            sandbox_output="",
        )

        self.assertEqual(step2.step, 2)
        self.assertEqual(tracker.total_input_tokens, 350)
        self.assertEqual(tracker.total_output_tokens, 80)
        self.assertEqual(tracker.total_requests, 3)  # (1+0) + (1+1) = 3
        self.assertEqual(len(tracker.steps), 2)

    def test_record_step_with_none_or_dict(self):
        tracker = MetricsTracker(task_id="102", benchmark="mbpp")

        # Step with None (e.g. sandbox only)
        step1 = tracker.record_step(llm_response=None, sandbox_input="pass", sandbox_output="")
        self.assertEqual(step1.step, 1)
        self.assertEqual(step1.input_tokens, 0)
        self.assertEqual(step1.output_tokens, 0)
        self.assertEqual(step1.retries, 0)
        self.assertEqual(step1.llm_output, "")

        # Step with a dict
        step2 = tracker.record_step(
            llm_response={
                "content": "thought",
                "input_tokens": 80,
                "output_tokens": 20,
                "request_time_ms": 150.0,
                "retries": 2,
                "model_name": "dict-model",
                "api_url": "https://dict.api",
            }
        )
        self.assertEqual(step2.step, 2)
        self.assertEqual(step2.input_tokens, 80)
        self.assertEqual(step2.output_tokens, 20)
        self.assertEqual(step2.retries, 2)
        self.assertEqual(step2.model_name, "dict-model")

    def test_check_limits_mbpp(self):
        tracker = MetricsTracker(task_id="103", benchmark="mbpp")

        # Below limit
        exceeded, _ = tracker.check_limits()
        self.assertFalse(exceeded)

        # Exceed iteration limit (MBPP max is 10)
        for i in range(11):
            tracker.record_step(
                llm_response=LLMResponse(
                    content=f"step {i}",
                    input_tokens=10,
                    output_tokens=10,
                    request_time_ms=10.0,
                    retries=0,
                    model_name="mock",
                    api_url="test",
                )
            )

        exceeded, reason = tracker.check_limits()
        self.assertTrue(exceeded)
        self.assertIn("Iterations", reason)

    def test_check_limits_tokens(self):
        # Exceed input token limit on MBPP (limit: 6000)
        tracker = MetricsTracker(task_id="104", benchmark="mbpp")
        tracker.record_step(
            llm_response=LLMResponse(
                content="big input",
                input_tokens=6500,
                output_tokens=10,
                request_time_ms=100.0,
                retries=0,
                model_name="mock",
                api_url="test",
            )
        )
        exceeded, reason = tracker.check_limits()
        self.assertTrue(exceeded)
        self.assertIn("Input tokens", reason)

        # Exceed output token limit on MBPP (limit: 1500)
        tracker_out = MetricsTracker(task_id="105", benchmark="mbpp")
        tracker_out.record_step(
            llm_response=LLMResponse(
                content="big output",
                input_tokens=50,
                output_tokens=1600,
                request_time_ms=100.0,
                retries=0,
                model_name="mock",
                api_url="test",
            )
        )
        exceeded, reason = tracker_out.check_limits()
        self.assertTrue(exceeded)
        self.assertIn("Output tokens", reason)

    def test_check_limits_swebench(self):
        tracker = MetricsTracker(task_id="sympy__sympy-1234", benchmark="swebench")
        self.assertEqual(LIMITS["swebench"].max_iterations, 30)
        self.assertEqual(LIMITS["swebench"].max_input_tokens, 300_000)
        self.assertEqual(LIMITS["swebench"].max_output_tokens, 10_000)
        self.assertEqual(LIMITS["swebench"].max_time_seconds, 900.0)

        # 31 iterations should trigger limit
        for i in range(31):
            tracker.record_step(
                llm_response=LLMResponse(
                    content=f"step {i}",
                    input_tokens=100,
                    output_tokens=10,
                    request_time_ms=10.0,
                    retries=0,
                    model_name="mock",
                    api_url="test",
                )
            )
        exceeded, reason = tracker.check_limits()
        self.assertTrue(exceeded)
        self.assertIn("Iterations", reason)

    def test_build_and_save_solution_output_schema_conformity(self):
        tracker = MetricsTracker(task_id="task-42", benchmark="mbpp")
        system_prompt = "You are an autonomous coding agent."
        tracker.system_prompt = system_prompt

        tracker.record_step(
            llm_response=LLMResponse(
                content="Thought: add function\n```python\ndef add(a, b): return a + b\n```",
                input_tokens=120,
                output_tokens=40,
                request_time_ms=300.0,
                retries=0,
                model_name="mock-llm",
                api_url="https://mock",
            ),
            sandbox_input="print(add(1, 2))",
            sandbox_output="3",
        )

        out_file = self.temp_dir / "solution.json"
        solution = tracker.save_solution(
            filepath=out_file,
            success=True,
            solution="def add(a, b): return a + b",
            error=None,
        )

        self.assertTrue(out_file.exists())
        self.assertEqual(solution.task_id, "task-42")
        self.assertEqual(solution.benchmark, "mbpp")
        self.assertTrue(solution.success)
        self.assertEqual(solution.solution, "def add(a, b): return a + b")
        self.assertEqual(solution.iterations, 1)
        self.assertEqual(solution.total_input_tokens, 120)
        self.assertEqual(solution.total_output_tokens, 40)
        self.assertEqual(solution.system_prompt, system_prompt)
        self.assertIsNone(solution.error)
        self.assertIsInstance(solution.total_time_seconds, float)

        # Validate that the saved JSON matches the schema exactly
        loaded_json = out_file.read_text(encoding="utf-8")
        validated = SolutionOutput.model_validate_json(loaded_json)
        self.assertEqual(validated.task_id, "task-42")
        self.assertEqual(len(validated.steps), 1)

        step = validated.steps[0]
        self.assertEqual(step.step, 1)
        self.assertEqual(step.input_tokens, 120)
        self.assertEqual(step.output_tokens, 40)
        self.assertEqual(step.retries, 0)
        self.assertEqual(step.model_name, "mock-llm")
        self.assertEqual(step.api_url, "https://mock")
        self.assertEqual(step.sandbox_input, "print(add(1, 2))")
        self.assertEqual(step.sandbox_output, "3")

    def test_failed_solution_error_propagation(self):
        tracker = MetricsTracker(task_id="task-fail", benchmark="mbpp")
        out_file = self.temp_dir / "failed_solution.json"

        solution = tracker.save_solution(
            filepath=out_file,
            success=False,
            solution="",
            error="Execution timed out after 120 seconds",
        )

        self.assertFalse(solution.success)
        self.assertEqual(solution.solution, "")
        self.assertEqual(solution.error, "Execution timed out after 120 seconds")

        validated = SolutionOutput.model_validate_json(out_file.read_text())
        self.assertFalse(validated.success)
        self.assertEqual(validated.error, "Execution timed out after 120 seconds")


if __name__ == "__main__":
    unittest.main()
