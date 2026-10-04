from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_PROVIDERS = (
    "openai",
    "custom",
    "gemini",
    "anthropic",
    "ollama",
    "openrouter",
)

_PROVIDER_ALIASES = {
    "gpt": "openai",
    "chatgpt": "openai",
    "openai-compatible": "custom",
    "local-openai": "custom",
    "google": "gemini",
    "google-genai": "gemini",
    "gemini-pro": "gemini",
    "anthorpic": "anthropic",
    "claude": "anthropic",
    "local": "ollama",
    "llama": "ollama",
    "router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Shared chat-model settings for the main agent and the optional judge."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map common aliases (including the `anthorpic` typo) onto a supported provider."""

    raw = (value or "openai").strip().lower()
    mapped = _PROVIDER_ALIASES.get(raw, raw)
    if mapped not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {value!r}. Choose one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return mapped


def build_chat_model(config: ProviderConfig):
    """Instantiate the LangChain chat model for the selected provider.

    Live mode is optional. Missing SDKs or credentials should surface as import/runtime
    errors to the caller so agents can fall back to the deterministic offline path.
    """

    provider = normalize_provider(config.provider)
    temperature = config.temperature
    model_name = config.model_name
    api_key = config.api_key
    base_url = config.base_url

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict = {"model": model_name, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        kwargs = {"model": model_name, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"model": model_name, "temperature": temperature}
        if api_key:
            kwargs["google_api_key"] = api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": model_name, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": model_name, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter

            kwargs = {"model": model_name, "temperature": temperature}
            if api_key:
                kwargs["api_key"] = api_key
            return ChatOpenRouter(**kwargs)
        except ImportError:
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": model_name,
                "temperature": temperature,
                "base_url": base_url or "https://openrouter.ai/api/v1",
            }
            if api_key:
                kwargs["api_key"] = api_key
            return ChatOpenAI(**kwargs)

    raise ValueError(f"Unsupported provider {provider!r}")
