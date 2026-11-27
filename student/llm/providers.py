"""Choose which (provider_url, model_name) pairs an agent tries, in order.

The defaults live in models_config.json (`default_benchmarks`): one
primary model and one fallback per benchmark. Command line arguments
override the primary; the fallback is kept so that an invalid key or an
unavailable provider does not fail the whole task.
"""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "models_config.json"


def model_chain(
    benchmark: str,
    model_name: str | None = None,
    provider_url: str | None = None,
    config_path: Path = CONFIG_PATH,
) -> list[tuple[str, str]]:
    """Return the (provider_url, model_name) pairs to try, in order.

    Args:
        benchmark: "mbpp" or "swebench" (a key of `default_benchmarks`).
        model_name: Model given on the command line, or None.
        provider_url: Provider URL given on the command line, or None.
        config_path: models_config.json to read.

    Returns:
        The primary pair first (command line values win over the config),
        then the configured fallback, without duplicates.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = config["default_benchmarks"][benchmark]

    primary = (
        provider_url or defaults["provider_url"],
        model_name or defaults["model_name"],
    )
    chain = [primary]

    fallback_url = defaults.get("fallback_provider_url")
    fallback_model = defaults.get("fallback_model_name")
    if fallback_url and fallback_model:
        fallback = (fallback_url, fallback_model)
        if fallback not in chain:
            chain.append(fallback)
    return chain
