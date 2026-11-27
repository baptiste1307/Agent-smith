import os
import unittest
from unittest.mock import MagicMock, patch

from agent.loop import _build_clients, _generate_with_fallback
from llm.client import LLMResponse, ProviderUnavailableError
from llm.providers import model_chain
from tools.exec_tools import _focus_test_output

GROQ = "https://api.groq.com/openai/v1"
MISTRAL = "https://api.mistral.ai/v1"


class TestModelChain(unittest.TestCase):
    def test_defaults_come_from_config(self):
        chain = model_chain("swebench")
        self.assertEqual(chain[0], (GROQ, "qwen/qwen3.8-27b"))
        self.assertEqual(chain[1], (MISTRAL, "codestral-latest"))

    def test_cli_overrides_primary_and_keeps_fallback(self):
        chain = model_chain(
            "swebench", model_name="openai/gpt-oss-120b", provider_url=GROQ
        )
        self.assertEqual(chain[0], (GROQ, "openai/gpt-oss-120b"))
        self.assertIn((MISTRAL, "codestral-latest"), chain)

    def test_no_duplicate_when_cli_equals_fallback(self):
        chain = model_chain(
            "swebench", model_name="codestral-latest", provider_url=MISTRAL
        )
        self.assertEqual(chain, [(MISTRAL, "codestral-latest")])


class TestBuildClients(unittest.TestCase):
    def test_provider_without_key_is_skipped(self):
        env = {"GROQ_API_KEY": "gsk_test"}
        with patch.dict(os.environ, env, clear=True):
            clients = _build_clients(
                [(MISTRAL, "codestral-latest"), (GROQ, "qwen/qwen3.8-27b")]
            )
        self.assertEqual([c.model_name for c in clients], ["qwen/qwen3.8-27b"])

    def test_no_key_at_all_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                _build_clients([(MISTRAL, "codestral-latest")])


class TestGenerateWithFallback(unittest.TestCase):
    def _response(self) -> LLMResponse:
        return LLMResponse(
            content="ok",
            input_tokens=1,
            output_tokens=1,
            request_time_ms=1.0,
            retries=0,
            model_name="backup",
            api_url=GROQ,
        )

    def test_switches_on_unusable_provider(self):
        broken, backup = MagicMock(), MagicMock()
        broken.generate.side_effect = ProviderUnavailableError("HTTP 401")
        backup.generate.return_value = self._response()
        answer, index = _generate_with_fallback(
            [broken, backup], 0, messages=[]
        )
        self.assertEqual(index, 1)
        self.assertEqual(answer.content, "ok")

    def test_last_provider_error_is_raised(self):
        broken = MagicMock()
        broken.generate.side_effect = ProviderUnavailableError("HTTP 401")
        with self.assertRaises(ProviderUnavailableError):
            _generate_with_fallback([broken], 0, messages=[])

    def test_other_errors_do_not_switch(self):
        failing, backup = MagicMock(), MagicMock()
        failing.generate.side_effect = RuntimeError("Time budget exhausted")
        with self.assertRaises(RuntimeError):
            _generate_with_fallback([failing, backup], 0, messages=[])
        backup.generate.assert_not_called()


class TestFocusTestOutput(unittest.TestCase):
    def test_keeps_only_test_section(self):
        output = "\n".join(
            [
                "pip install noise",
                "+ : '>>>>> Start Test Output'",
                "+ pytest -rA tests/test_x.py",
                "PASSED tests/test_x.py::test_a",
                "+ : '>>>>> End Test Output'",
                "+ git checkout tests/test_x.py",
            ]
        )
        focused = _focus_test_output(output)
        self.assertIn("PASSED tests/test_x.py::test_a", focused)
        self.assertNotIn("pip install noise", focused)
        self.assertNotIn("git checkout", focused)
        self.assertIn("omitted", focused)

    def test_without_markers_output_is_unchanged(self):
        self.assertEqual(_focus_test_output("plain"), "plain")


if __name__ == "__main__":
    unittest.main()
