"""Choose which (provider_url, model_name) pairs an agent tries, in order.

The defaults live in models_config.json (`default_benchmarks`): one
primary model and an ordered list of `fallbacks` per benchmark. Command
line arguments override the primary; the fallbacks are kept so that an
invalid key or an unavailable provider does not fail the whole task.
Providers whose key is missing or left at its example value are skipped
later, when the clients are built.
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
        then the configured fallbacks in order, without duplicates.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = config["default_benchmarks"][benchmark]

    primary = (
        provider_url or defaults["provider_url"],
        model_name or defaults["model_name"],
    )
    chain = [primary]

    extra = list(defaults.get("fallbacks", []))
    # Older single-fallback form, still accepted
    if defaults.get("fallback_provider_url"):
        extra.insert(
            0,
            {
                "provider_url": defaults["fallback_provider_url"],
                "model_name": defaults["fallback_model_name"],
            },
        )
    for entry in extra:
        pair = (entry["provider_url"], entry["model_name"])
        if pair not in chain:
            chain.append(pair)
    return chain
