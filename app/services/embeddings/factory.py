from app.services.embeddings.base import BaseEmbeddingProvider
from app.utils.config import get_settings


def create_embedding_provider() -> BaseEmbeddingProvider:
    settings = get_settings()
    cfg = settings.embeddings
    provider = cfg.provider

    if provider == "azure_openai":
        from app.services.embeddings.azure_openai import AzureOpenAIEmbeddings, parse_azure_url
        # A full deployment URL (env RAG_EMBEDDINGS__AZURE_URL) takes precedence over the split fields.
        if cfg.azure_url:
            endpoint, deployment, api_version = parse_azure_url(cfg.azure_url)
        else:
            endpoint, deployment, api_version = cfg.azure_endpoint or "", cfg.azure_deployment or cfg.model, cfg.api_version
        return AzureOpenAIEmbeddings(
            api_key=cfg.api_key or "",
            endpoint=endpoint,
            deployment=deployment,
            api_version=api_version,
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
