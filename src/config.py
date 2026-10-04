from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared paths, compact-memory knobs, and provider settings for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    profile_confidence_threshold: float = 0.65
    memory_decay_half_life_days: float = 30.0


def _load_dotenv(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)


def _provider_credentials(provider: str) -> tuple[str | None, str | None]:
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL")
    if provider == "custom":
        return os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY"), os.getenv(
            "CUSTOM_BASE_URL"
        )
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"), None
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY"), None
    if provider == "ollama":
        return None, os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY"), os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
    return None, None


def _build_provider(
    provider_raw: str,
    model_name: str,
    temperature: float,
) -> ProviderConfig:
    provider = normalize_provider(provider_raw)
    api_key, base_url = _provider_credentials(provider)
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a complete LabConfig."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    data_dir = root / "data"
    state_dir = Path(os.getenv("LAB_STATE_DIR", str(root / "state"))).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    provider = os.getenv("LLM_PROVIDER", "openai")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    judge_provider = os.getenv("LLM_JUDGE_PROVIDER", provider)
    judge_model_name = os.getenv("LLM_JUDGE_MODEL", model_name)
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "700"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))
    confidence = float(os.getenv("PROFILE_CONFIDENCE_THRESHOLD", "0.65"))
    half_life = float(os.getenv("MEMORY_DECAY_HALF_LIFE_DAYS", "30"))

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=_build_provider(provider, model_name, temperature),
        judge_model=_build_provider(judge_provider, judge_model_name, temperature),
        profile_confidence_threshold=confidence,
        memory_decay_half_life_days=half_life,
    )
