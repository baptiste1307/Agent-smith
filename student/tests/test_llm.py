"""Live integration test for LLM connection (skipped if no API key is set)."""

import os
import unittest
from dotenv import load_dotenv

from llm.client import LLMClient
from llm.rotator import KeyRotator

load_dotenv()


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
