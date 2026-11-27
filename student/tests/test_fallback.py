import os
import unittest
from unittest.mock import MagicMock, patch

from agent.loop import _build_clients, _generate_with_fallback
from llm.client import LLMResponse, ProviderUnavailableError
from llm.providers import model_chain
from llm.rotator import KeyRotator, is_placeholder_key
from tools.exec_tools import _focus_test_output

GROQ = "https://api.groq.com/openai/v1"
MISTRAL = "https://api.mistral.ai/v1"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
OPENROUTER = "https://openrouter.ai/api/v1"


class TestModelChain(unittest.TestCase):
    def test_defaults_come_from_config(self):
        for bench in ("mbpp", "swebench"):
            chain = model_chain(bench)
            self.assertEqual(chain[0], (GEMINI, "gemini-3.5-flash"))
            self.assertEqual(
                [url for url, _ in chain[1:]],
                [GEMINI, OPENROUTER, GROQ, MISTRAL],
            )

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
        self.assertEqual(chain[0], (MISTRAL, "codestral-latest"))
        self.assertEqual(chain.count((MISTRAL, "codestral-latest")), 1)


class TestPlaceholderKeys(unittest.TestCase):
    def test_example_values_are_placeholders(self):
        for key in (
            "your_mistral_api_key_here",
            "gsk_xxxxxxxxxxxxxxxxxxxx",
            "sk-or-v1-xxxxxxxxxxxx",
            "  ",
        ):
            self.assertTrue(is_placeholder_key(key), key)
        self.assertFalse(is_placeholder_key("gsk_8fK2mQ9zLp"))

    def test_placeholder_provider_is_skipped(self):
        env = {
            "GROQ_API_KEY": "gsk_xxxxxxxxxxxxxxxxxxxx",
            "MISTRAL_API_KEY": "your_mistral_api_key_here",
            "GEMINI_API_KEY": "AQ.real-looking-key",
        }
        with patch.dict(os.environ, env, clear=True):
            clients = _build_clients(
                [
                    (GROQ, "qwen/qwen3.8-27b"),
                    (MISTRAL, "codestral-latest"),
                    (GEMINI, "gemini-3.5-flash"),
                ]
            )
            self.assertEqual(
                [c.model_name for c in clients], ["gemini-3.5-flash"]
            )
            with self.assertRaises(ValueError):
                KeyRotator.auto_detect(GROQ)

    def test_real_keys_kept_next_to_placeholders(self):
        env = {"OPENROUTER_API_KEY": "sk-or-v1-real1, sk-or-v1-xxxxxxxxxxxx"}
        with patch.dict(os.environ, env, clear=True):
            rotator = KeyRotator.auto_detect(OPENROUTER)
        self.assertEqual(rotator.keys, ["sk-or-v1-real1"])


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


class TestPaymentRequired(unittest.TestCase):
    def test_402_gives_up_after_one_try_per_key(self):
        from llm.client import LLMClient

        response = MagicMock(status_code=402, text="billing_no_api_budget")
        client = LLMClient(
            MISTRAL,
            "codestral-latest",
            rotator=KeyRotator(keys=["k1", "k2"]),
        )
        with patch.object(
            LLMClient, "send_message", return_value=response
        ) as send, patch.object(LLMClient, "_sleep") as sleep:
            with self.assertRaises(ProviderUnavailableError):
                client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(send.call_count, 2)
        sleep.assert_not_called()


class TestDailyQuota(unittest.TestCase):
    def test_daily_quota_429_does_not_wait(self):
        from llm.client import LLMClient

        body = (
            "Quota exceeded for metric: generate_content_free_tier_requests"
            ", limit: 20, description: Generate content requests per day "
            "limit exceeded on Free Tier."
        )
        response = MagicMock(status_code=429, text=body)
        client = LLMClient(
            GEMINI, "gemini-3.5-flash", rotator=KeyRotator(keys=["k"])
        )
        with patch.object(
            LLMClient, "send_message", return_value=response
        ) as send, patch.object(LLMClient, "_sleep") as sleep:
            with self.assertRaises(ProviderUnavailableError):
                client.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(send.call_count, 1)
        sleep.assert_not_called()


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


class TestSalvagePatch(unittest.TestCase):
    PATCH = (
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )

    def _client(self, result):
        client = MagicMock()
        client.get_wrappers.return_value = {"get_patch": lambda: result}
        return client

    def test_returns_repository_diff(self):
        from agent_swebench import _salvage_patch

        self.assertEqual(_salvage_patch(self._client(self.PATCH)), self.PATCH)

    def test_empty_or_invalid_diff_gives_nothing(self):
        from agent_swebench import _salvage_patch

        self.assertEqual(_salvage_patch(self._client("")), "")
        self.assertEqual(_salvage_patch(self._client("not a diff")), "")
        self.assertEqual(_salvage_patch(None), "")


if __name__ == "__main__":
    unittest.main()
