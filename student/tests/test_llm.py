"""Unit and integration tests for LLM client and key rotation."""

import os
import unittest

from dotenv import load_dotenv
from llm.client import LLMClient
from llm.rotator import KeyRotator

load_dotenv()


class TestKeyRotator(unittest.TestCase):
    def test_rotation_multiple_keys(self):
        rotator = KeyRotator(keys=["key1", "key2", "key3"])
        self.assertEqual(rotator.get_current_key(), "key1")
        self.assertEqual(rotator.rotate_key(), "key2")
        self.assertEqual(rotator.get_current_key(), "key2")
        self.assertEqual(rotator.rotate_key(), "key3")
        self.assertEqual(rotator.rotate_key(), "key1")

    def test_from_env_comma_separated(self):
        os.environ["TEST_ROTATOR_KEYS"] = "keyA, keyB,  keyC "
        try:
            rotator = KeyRotator.from_env("TEST_ROTATOR_KEYS")
            self.assertEqual(rotator.keys, ["keyA", "keyB", "keyC"])
            self.assertEqual(rotator.get_current_key(), "keyA")
        finally:
            del os.environ["TEST_ROTATOR_KEYS"]

    def test_auto_detect_provider(self):
        os.environ["MISTRAL_API_KEY"] = "mistral-123"
        try:
            rotator = KeyRotator.auto_detect("https://api.mistral.ai/v1")
            self.assertEqual(rotator.get_current_key(), "mistral-123")
        finally:
            del os.environ["MISTRAL_API_KEY"]

    def test_empty_keys_raise_error(self):
        rotator = KeyRotator(keys=[])
        with self.assertRaises(ValueError):
            rotator.get_current_key()
        with self.assertRaises(ValueError):
            rotator.rotate_key()


class TestLLMConnection(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("MISTRAL_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENROUTER_API_KEY"),
        (
            "Requires MISTRAL_API_KEY, GROQ_API_KEY, "
            "or OPENROUTER_API_KEY env var"
        ),
    )
    def test_live_llm_call(self):
        candidates = []
        if os.getenv("MISTRAL_API_KEY"):
            candidates.append(
                (
                    "https://api.mistral.ai/v1",
                    "codestral-latest",
                    "MISTRAL_API_KEY",
                )
            )
        if os.getenv("GROQ_API_KEY"):
            candidates.append(
                (
                    "https://api.groq.com/openai/v1",
                    "openai/gpt-oss-120b",
                    "GROQ_API_KEY",
                )
            )
        if os.getenv("OPENROUTER_API_KEY"):
            candidates.append(
                (
                    "https://openrouter.ai/api/v1",
                    "nvidia/nemotron-3-ultra-550b-a55b:free",
                    "OPENROUTER_API_KEY",
                )
            )

        last_err = None
        for provider_url, model_name, env_var in candidates:
            client = LLMClient(
                provider_url=provider_url,
                model_name=model_name,
                rotator=KeyRotator.from_env(env_var),
                max_retries=1,
            )
            try:
                response = client.generate(
                    [
                        {
                            "role": "user",
                            "content": "Say hello in one short sentence.",
                        }
                    ]
                )
                self.assertTrue(bool(response.content))
                self.assertGreater(response.input_tokens, 0)
                self.assertGreater(response.output_tokens, 0)
                return
            except RuntimeError as e:
                last_err = e
                continue

        self.fail(f"All configured free providers failed: {last_err}")


if __name__ == "__main__":
    unittest.main()
