from app.services.embeddings.base import BaseEmbeddingProvider
from app.utils.config import get_settings


def create_embedding_provider() -> BaseEmbeddingProvider:
    settings = get_settings()
    cfg = settings.embeddings
    provider = cfg.provider

    if provider == "azure_openai":
        from app.services.embeddings.azure_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            api_key=cfg.api_key or "",
            endpoint=cfg.azure_endpoint or "",
            deployment=cfg.azure_deployment or cfg.model,
            api_version=cfg.api_version,
            dim=cfg.dim,
        )

    if provider == "openai":
        from app.services.embeddings.openai_provider import OpenAIEmbeddings
        # api_key/base_url None → SDK reads OPENAI_API_KEY / OPENAI_BASE_URL from env.
        return OpenAIEmbeddings(
            api_key=cfg.api_key or None, base_url=None, model=cfg.model, dim=cfg.dim
        )

    if provider == "ollama":
        from app.services.embeddings.ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=cfg.ollama_base_url,
            model=cfg.model,
            dim=cfg.dim,
            max_input_tokens=cfg.max_input_tokens,
        )

    raise ValueError(f"Unknown embeddings.provider: {provider!r}")
