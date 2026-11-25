"""Tests for the time / token budgets enforced before each LLM request."""

import time
import unittest
from unittest import mock

import httpx
from agent import loop
from agent.loop import _fit_input_budget, agent_loop
from llm.client import LLMClient, LLMResponse
from llm.metrics import LIMITS, MetricsTracker
from llm.rotator import KeyRotator
from mcp_client.client import PROMPT_HELPER, RESOURCE_HELPER, MCPClient


def _response(content: str, input_tokens: int, output_tokens: int):
    return LLMResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        request_time_ms=1.0,
        retries=0,
        model_name="fake",
        api_url="http://fake",
    )


class FakeLLMClient:
    """Stands for LLMClient: replays scripted answers, records requests."""

    script: list[LLMResponse] = []
    requests: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, messages, stop=None, max_tokens=0, deadline=None):
        FakeLLMClient.requests.append(
            {"messages": messages, "max_tokens": max_tokens}
        )
        return FakeLLMClient.script.pop(0)


class TestTrackerBudgets(unittest.TestCase):
    def test_remaining_input_tokens(self):
        tracker = MetricsTracker(task_id="1", benchmark="mbpp")
        tracker.record_step(_response("x", 4000, 10))
        self.assertEqual(tracker.remaining_input_tokens, 2000)
        tracker.record_step(_response("x", 2500, 10))
        self.assertEqual(tracker.remaining_input_tokens, 0)

    def test_deadline_keeps_safety_margin(self):
        for benchmark, margin in (("mbpp", 10.0), ("swebench", 45.0)):
            tracker = MetricsTracker(task_id="1", benchmark=benchmark)
            limit = LIMITS[benchmark].max_time_seconds
            self.assertAlmostEqual(
                tracker.time_left_seconds(), limit - margin, delta=1.0
            )

    def test_unknown_benchmark_has_no_budget(self):
        tracker = MetricsTracker(task_id="1", benchmark="other")
        self.assertIsNone(tracker.remaining_input_tokens)
        self.assertIsNone(tracker.deadline)


class TestFitInputBudget(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": "s" * 300},
            {"role": "user", "content": "task"},
        ]
        for _ in range(4):
            self.messages.append({"role": "assistant", "content": "a" * 30})
            self.messages.append(
                {"role": "user", "content": "Observation:\n" + "o" * 3000}
            )

    def test_unlimited_budget_uses_normal_pruning(self):
        fitted = _fit_input_budget(self.messages, 3, None, 3.0)
        self.assertEqual(len(fitted), len(self.messages))

    def test_prunes_more_to_fit(self):
        # 3 full observations do not fit, a single one does
        fitted = _fit_input_budget(self.messages, 3, 1600, 3.0)
        full = [m for m in fitted if "o" * 3000 in m["content"]]
        self.assertEqual(len(full), 1)

    def test_raises_when_nothing_fits(self):
        with self.assertRaises(RuntimeError) as ctx:
            _fit_input_budget(self.messages, 3, 100, 3.0)
        self.assertIn("Input token budget exhausted", str(ctx.exception))


class TestAgentLoopBudgets(unittest.TestCase):
    def setUp(self):
        FakeLLMClient.requests = []
        patcher = mock.patch.object(loop, "LLMClient", FakeLLMClient)
        patcher.start()
        self.addCleanup(patcher.stop)
        rotator = mock.patch.object(
            loop.KeyRotator, "auto_detect", return_value=None
        )
        rotator.start()
        self.addCleanup(rotator.stop)

    def test_stops_before_exceeding_input_tokens(self):
        # Each answer reports 2,800 input tokens: a 3rd request would push
        # the MBPP total over 6,000, so it must never be sent
        FakeLLMClient.script = [
            _response("```python\nprint(1)\n```", 2800, 20),
            _response("```python\nprint(2)\n```", 2800, 20),
            _response("```python\nfinal_answer('x')\n```", 2800, 20),
        ]
        tracker = MetricsTracker(task_id="1", benchmark="mbpp")
        with self.assertRaises(RuntimeError) as ctx:
            agent_loop("task", tracker=tracker)
        self.assertIn("Input token budget exhausted", str(ctx.exception))
        self.assertEqual(len(FakeLLMClient.requests), 2)
        self.assertLessEqual(tracker.total_input_tokens, 6000)

    def test_stops_when_output_budget_is_spent(self):
        # Regression: with 0 tokens left, a request with max_tokens=1 was
        # still sent, giving 10,001 / 10,000 output tokens
        FakeLLMClient.script = [
            _response("```python\nprint(1)\n```", 100, 1490),
        ]
        tracker = MetricsTracker(task_id="1", benchmark="mbpp")
        with self.assertRaises(RuntimeError) as ctx:
            agent_loop("task", tracker=tracker)
        self.assertIn("Output token budget exhausted", str(ctx.exception))
        self.assertEqual(len(FakeLLMClient.requests), 1)

    def test_max_tokens_capped_by_remaining_output(self):
        FakeLLMClient.script = [
            _response("```python\nprint(1)\n```", 100, 1300),
            _response("```python\nfinal_answer('x')\n```", 100, 50),
        ]
        tracker = MetricsTracker(task_id="1", benchmark="mbpp")
        self.assertEqual(agent_loop("task", tracker=tracker), "x")
        self.assertEqual(FakeLLMClient.requests[1]["max_tokens"], 200)

    def test_stops_when_time_is_up(self):
        FakeLLMClient.script = []
        tracker = MetricsTracker(task_id="1", benchmark="mbpp")
        tracker._start_perf -= 200  # the run "started" 200s ago
        with self.assertRaises(RuntimeError) as ctx:
            agent_loop("task", tracker=tracker)
        self.assertIn("Time budget exhausted", str(ctx.exception))
        self.assertEqual(FakeLLMClient.requests, [])


class TestClientDeadline(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(
            "http://fake/v1", "fake", rotator=KeyRotator(keys=["k"])
        )

    def test_no_request_past_deadline(self):
        with mock.patch.object(self.client, "send_message") as send:
            with self.assertRaises(RuntimeError) as ctx:
                self.client.generate(
                    [{"role": "user", "content": "hi"}],
                    deadline=time.perf_counter() + 1,
                )
        send.assert_not_called()
        self.assertIn("Time budget exhausted", str(ctx.exception))

    def test_retry_wait_and_http_timeout_capped(self):
        rate_limited = httpx.Response(
            429, headers={"retry-after": "40"}, text="slow down"
        )
        timeouts = []

        def send(headers, payload, timeout=None):
            timeouts.append(timeout)
            return rate_limited

        start = time.perf_counter()
        with mock.patch.object(self.client, "send_message", side_effect=send):
            with self.assertRaises(RuntimeError):
                self.client.generate(
                    [{"role": "user", "content": "hi"}],
                    deadline=start + 4,
                )
        # The 40s Retry-After was cut to the deadline
        self.assertLess(time.perf_counter() - start, 6)
        self.assertLessEqual(timeouts[0], 4)


class TestTokenAccounting(unittest.TestCase):
    """Regression: answers without usage were recorded as 0 tokens."""

    def setUp(self):
        self.client = LLMClient(
            "http://fake/v1", "fake", rotator=KeyRotator(keys=["k"])
        )
        self.messages = [{"role": "user", "content": "x" * 3000}]

    def _generate(self, response):
        with mock.patch.object(
            self.client, "send_message", return_value=response
        ):
            return self.client.generate(self.messages)

    def test_groq_tool_use_failed_is_counted(self):
        failed = httpx.Response(
            400,
            json={
                "error": {
                    "code": "tool_use_failed",
                    "failed_generation": '{"name": "run_command"}',
                }
            },
        )
        answer = self._generate(failed)
        self.assertIn("<tool_call>", answer.content)
        self.assertTrue(answer.estimated_tokens)
        self.assertGreaterEqual(answer.input_tokens, 1000)
        self.assertGreater(answer.output_tokens, 0)

    def test_missing_usage_is_estimated(self):
        ok = httpx.Response(
            200, json={"choices": [{"message": {"content": "hello"}}]}
        )
        answer = self._generate(ok)
        self.assertTrue(answer.estimated_tokens)
        self.assertGreaterEqual(answer.input_tokens, 1000)

    def test_provider_usage_is_kept(self):
        ok = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 7},
            },
        )
        answer = self._generate(ok)
        self.assertFalse(answer.estimated_tokens)
        self.assertEqual((answer.input_tokens, answer.output_tokens), (42, 7))


class TestMCPHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = MCPClient("python mcp_tools_mbpp.py")

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_resources_and_prompts_exposed(self):
        wrappers = self.client.get_wrappers()
        names = [s["name"] for s in self.client.get_schemas()]
        self.assertIn(RESOURCE_HELPER, wrappers)
        self.assertIn(PROMPT_HELPER, wrappers)
        self.assertIn(RESOURCE_HELPER, names)
        self.assertIn(PROMPT_HELPER, names)
        self.assertIn("MBPP", wrappers[RESOURCE_HELPER]("mbpp://info"))
        prompt = wrappers[PROMPT_HELPER](
            "mbpp_workflow", {"task_definition": "add two numbers"}
        )
        self.assertIn("add two numbers", prompt)

    def test_manual_lists_resources(self):
        schemas = self.client.get_schemas()
        helper = next(s for s in schemas if s["name"] == RESOURCE_HELPER)
        self.assertIn("mbpp://info", helper["description"])

    def test_tool_call_respects_deadline(self):
        self.client.deadline = time.perf_counter() - 1
        try:
            with self.assertRaises(TimeoutError):
                self.client.get_wrappers()["run_tests"](
                    "def f():\n    import time\n    time.sleep(5)",
                    ["f()"],
                )
        finally:
            self.client.deadline = None


if __name__ == "__main__":
    unittest.main()
