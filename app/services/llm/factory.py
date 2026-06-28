import os

from app.services.llm.base import BaseLLMProvider
from app.utils.config import get_settings


def create_llm_provider() -> BaseLLMProvider:
    settings = get_settings()
    cfg = settings.llm
    provider = cfg.provider

    if provider in ("openai", "vllm"):
        from app.services.llm.openai_provider import OpenAIProvider
        # vllm needs its configured base_url; openai → the real OpenAI API unless an endpoint is
        # given via env (cfg.base_url defaults to the Ollama URL, which would silently misroute).
        if provider == "vllm":
            base_url = os.getenv("OPENAI_BASE_URL") or cfg.base_url
        else:
            base_url = os.getenv("OPENAI_BASE_URL") or None
        return OpenAIProvider(
            api_key=os.getenv("OPENAI_API_KEY") or cfg.api_key or "",
            base_url=base_url,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    if provider == "ollama":
        from app.services.llm.ollama_provider import OllamaProvider
        return OllamaProvider(
            base_url=cfg.base_url,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    raise ValueError(f"Unknown llm.provider: {provider!r}")
