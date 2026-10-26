from dotenv import load_dotenv
from llm.client import LLMClient
from llm.rotator import KeyRotator

load_dotenv()  # reads the .env file

client = LLMClient(
    provider_url="https://openrouter.ai/api/v1",
    model_name="poolside/laguna-s-2.1:free",
    rotator=KeyRotator.from_env("OPENROUTER_API_KEY"),
)

response = client.generate([{"role": "user", "content": "Dis bonjour en une phrase."}])
print(response)
