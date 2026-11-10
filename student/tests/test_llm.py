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
        os.getenv("OPENROUTER_API_KEY"), "Requires OPENROUTER_API_KEY env var"
    )
    def test_live_llm_call(self):
        client = LLMClient(
            provider_url="https://openrouter.ai/api/v1",
            model_name="liquid/lfm-2.5-2.6b:free",
            rotator=KeyRotator.from_env("OPENROUTER_API_KEY"),
        )
        response = client.generate(
            [{"role": "user", "content": "Dis bonjour en une phrase."}]
        )
        self.assertTrue(bool(response.content))
        self.assertGreater(response.input_tokens, 0)
        self.assertGreater(response.output_tokens, 0)


if __name__ == "__main__":
    unittest.main()
