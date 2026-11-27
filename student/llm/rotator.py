import os
from dataclasses import dataclass


def is_placeholder_key(key: str) -> bool:
    """Return True for an empty key or an example value from .env.example
    (your_..._here, gsk_xxxx...). Leaving the example value in .env means
    "I don't use this provider", so such keys are treated as missing.
    """
    k = key.strip().lower()
    return (
        not k
        or k.startswith("your")
        or k.endswith("_here")
        or "xxxx" in k
    )


def usable_keys(value: str | None) -> list[str]:
    """Split a comma-separated env value into real (non-example) keys."""
    if not value:
        return []
    return [
        k.strip() for k in value.split(",") if not is_placeholder_key(k)
    ]


@dataclass
class KeyRotator:
    keys: list[str] | None = None
    current_index: int = 0

    def __post_init__(self) -> None:
        if self.keys is not None:
            # 'if key.strip()' -> .strip() returns str,
            # so dont add API key to self.keys if empty key
            self.keys = [key.strip() for key in self.keys if key.strip()]
        else:
            self.keys = []

    def get_current_key(self) -> str:
        if not self.keys:
            raise ValueError("No API keys provided.")
        return self.keys[self.current_index]

    def rotate_key(self) -> str:
        if not self.keys:
            raise ValueError("No API keys provided.")
        self.current_index = (self.current_index + 1) % len(self.keys)
        return self.keys[self.current_index]

    @classmethod
    def from_env(cls, env_var: str):
        """Build a KeyRotator from an environment variable name."""
        env_var_content = os.getenv(env_var)
        if not env_var_content:
            raise ValueError(
                f"Environment variable '{env_var}' doesn't exist."
            )

        splitted_api_keys = usable_keys(env_var_content)
        if not splitted_api_keys:
            raise ValueError(
                f"Environment variable '{env_var}' is empty or only holds "
                "example values (provider disabled)."
            )

        return cls(keys=splitted_api_keys)

    @classmethod
    def auto_detect(cls, provider_url: str = "", env_var: str | None = None):
        """Detect API keys based on provider_url or candidate env vars."""
        if env_var:
            return cls.from_env(env_var)

        # Map provider keywords to standard env vars
        provider_candidates = []
        url_lower = provider_url.lower()

        if "openrouter" in url_lower:
            provider_candidates.extend(
                ["OPENROUTER_API_KEY", "OPENROUTER_KEY"]
            )
        elif "mistral" in url_lower:
            provider_candidates.extend(["MISTRAL_API_KEY", "MISTRAL_KEY"])
        elif "groq" in url_lower:
            provider_candidates.extend(["GROQ_API_KEY", "GROQ_KEY"])
        elif "together" in url_lower:
            provider_candidates.extend(
                ["TOGETHER_API_KEY", "TOGETHERAI_API_KEY"]
            )
        elif "google" in url_lower or "generativelanguage" in url_lower:
            provider_candidates.extend(["GEMINI_API_KEY", "GOOGLE_API_KEY"])
        elif "cohere" in url_lower:
            provider_candidates.extend(["COHERE_API_KEY"])
        elif "fireworks" in url_lower:
            provider_candidates.extend(["FIREWORKS_API_KEY"])

        # If the provider URL is explicit, try its specific keys first
        # and fail clearly rather than silently falling back to a different
        # provider's key (which would cause confusing auth errors).
        if provider_candidates:
            for var in provider_candidates:
                if usable_keys(os.getenv(var)):
                    return cls.from_env(var)
            # Explicit provider but no matching key — fail with a clear message
            raise ValueError(
                f"No API key found for provider '{provider_url}'. "
                f"Set one of: {', '.join(provider_candidates)}"
            )

        # No specific provider detected — try generic fallbacks
        generic_candidates = [
            "OPENROUTER_API_KEY",
            "MISTRAL_API_KEY",
            "GROQ_API_KEY",
            "TOGETHER_API_KEY",
            "GEMINI_API_KEY",
            "LLM_API_KEY",
            "API_KEY",
        ]
        for var in generic_candidates:
            if usable_keys(os.getenv(var)):
                return cls.from_env(var)

        raise ValueError(
            f"No API key found in environment for provider '{provider_url}'. "
            f"Checked environment variables: {', '.join(generic_candidates)}"
        )
