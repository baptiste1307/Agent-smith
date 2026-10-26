from dataclasses import dataclass
import os


@dataclass
class KeyRotator:

    keys: list[str] = None
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
        """Build an instance of its parent (KeyRotator)
        from an environment variable name"""
        env_var_content = os.getenv(env_var)
        if not env_var_content:
            raise ValueError(f"Environment variable '{env_var}' doesn't exist.")

        splitted_api_keys = env_var_content.split(",")
        if not splitted_api_keys:
            raise ValueError(f"Environment variable '{env_var}' is empty.")

        return cls(keys=splitted_api_keys)
