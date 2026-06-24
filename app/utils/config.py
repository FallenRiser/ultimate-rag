from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"


def _load_yaml() -> Dict[str, Any]:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


class _YamlSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: Type[BaseSettings]):
        super().__init__(settings_cls)
        self._data = _load_yaml()

    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> Dict[str, Any]:
        return {k: v for k, v in self._data.items() if v is not None}


# ---------------------------------------------------------------------------
# Section models — plain BaseModel, env overrides via Settings' nested delimiter
# ---------------------------------------------------------------------------

class AppSettings(BaseModel):
    name: str = "ultimate-rag"
    env: str = "dev"
    debug: bool = True


class LoggingSettings(BaseModel):
    level: str = "INFO"                  # level for our own app.* loggers
    third_party_level: str = "WARNING"   # level for libraries (sqlalchemy, httpx, aiosqlite, ...)
    sql_echo: bool = False               # SQLAlchemy engine echo — very noisy, off by default
    rich: bool = True
    log_dir: str = "logs"
    file: str = "app.log"
    truncate: bool = False
    rotation_mb: int = 50
    backups: int = 5
    noisy_loggers: List[str] = [
        "sqlalchemy", "sqlalchemy.engine", "aiosqlite", "httpcore", "httpx",
        "openai", "neo4j", "urllib3", "asyncio", "qdrant_client",
    ]


class MLflowAutologSettings(BaseModel):
    langchain: bool = True
    openai: bool = True
    litellm: bool = False


class MLflowSettings(BaseModel):
    enabled: bool = True
    tracking_uri: str = "http://localhost:5000"
    experiment: str = "ultimate-rag"
    autolog: MLflowAutologSettings = MLflowAutologSettings()


class ObservabilitySettings(BaseModel):
    mlflow: MLflowSettings = MLflowSettings()


class APISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]


class DatabaseSettings(BaseModel):
    provider: str = "postgres"        # postgres | sqlite
    dsn: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    sqlite_path: str = "rag.db"       # used when provider == sqlite
    pool_size: int = 10
    max_overflow: int = 20


class VectorStoreSettings(BaseModel):
    provider: str = "qdrant"          # pgvector | qdrant | chroma
    collection: str = "rag_chunks"
    embedding_dim: int = 3072
    distance: str = "cosine"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    enable_sparse: bool = True
    sparse_model: str = "Qdrant/bm25"
    pgvector_table: str = "chunk_vectors"
    chroma_path: str = "chroma_data"  # local persistent dir (chroma = semantic only)


class GraphStoreSettings(BaseModel):
    enabled: bool = False
    provider: str = "age"             # age (Postgres) | memgraph (Bolt)
    graph_name: str = "rag_kg"        # AGE only
    memgraph_url: str = "bolt://localhost:7687"
    memgraph_user: str = ""
    memgraph_password: str = ""


class CacheSettings(BaseModel):
    enabled: bool = True
    provider: str = "ignite"
    ttl_seconds: int = 3600
    ignite_addresses: List[str] = ["127.0.0.1:10800"]
    cache_name: str = "rag_query_cache"
    namespace: str = "qcache"


class EmbeddingsSettings(BaseModel):
    provider: str = "azure_openai"    # azure_openai | openai | ollama
    model: str = "text-embedding-3-large"
    dim: int = 3072
    batch_size: int = 64
    max_input_tokens: Optional[int] = None  # client-side truncation (small-context models e.g. mxbai=512)
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: str = "2024-06-01"
    api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"


class LLMSettings(BaseModel):
    provider: str = "openai"          # openai | ollama | vllm
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_s: int = 60
    max_retries: int = 3
    structured_output: str = "instructor"


class RerankerSettings(BaseModel):
    enabled: bool = True
    provider: str = "bge"             # bge | qwen3 | cohere
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 8
    endpoint: Optional[str] = "http://localhost:8080"


class DoclingSettings(BaseModel):
    url: str = "http://localhost:5001"
    do_ocr: bool = True
    do_table_structure: bool = True


class CustomParsingSettings(BaseModel):
    ocr_engine: str = "rapidocr"
    ocr_languages: List[str] = ["en"]
    pdf_image_ocr_threshold: float = 0.1


class ParsingSettings(BaseModel):
    backend: str = "custom"           # docling | custom
    docling: DoclingSettings = DoclingSettings()
    custom: CustomParsingSettings = CustomParsingSettings()


class ChunkingSettings(BaseModel):
    strategy: str = "recursive"       # fixed | recursive | document_aware | semantic
    chunk_size: int = 512
    chunk_overlap: int = 64
    unit: str = "token"
    semantic_threshold: float = 0.6


class ImageCaptioningSettings(BaseModel):
    enabled: bool = False
    model: str = "gpt-4o"


class TableExtractionSettings(BaseModel):
    enabled: bool = True
    summarize: bool = True


class MetadataExtractionSettings(BaseModel):
    enabled: bool = False
    fields: List[str] = ["title", "author", "doc_type", "topics"]


class ContextualRetrievalSettings(BaseModel):
    enabled: bool = False


class EnrichmentSettings(BaseModel):
    image_captioning: ImageCaptioningSettings = ImageCaptioningSettings()
    table_extraction: TableExtractionSettings = TableExtractionSettings()
    metadata_extraction: MetadataExtractionSettings = MetadataExtractionSettings()
    contextual_retrieval: ContextualRetrievalSettings = ContextualRetrievalSettings()


class RetrievalWeights(BaseModel):
    dense: float = 0.6
    sparse: float = 0.4


class RetrievalSettings(BaseModel):
    default_mode: str = "hybrid"      # semantic | bm25 | graph | hybrid | hybrid_graph
    top_k: int = 20
    fusion: str = "rrf"
    rrf_k: int = 60
    weights: RetrievalWeights = RetrievalWeights()
    rerank_top_k: int = 8


class AutoFilterSettings(BaseModel):
    enabled: bool = False                       # LLM infers metadata filters from the query
    fields: List[str] = ["doc_type", "author"]  # only these fields may be auto-filtered


class AgentToolsSettings(BaseModel):
    max_tool_calls: int = 8   # hard budget so a weak model can't run away


class AgentSettings(BaseModel):
    enabled: bool = True
    style: str = "graph"                  # graph (fixed state-machine) | tools (ReAct tool-calling)
    query_rewrite: bool = True
    decompose: bool = True
    max_subqueries: int = 4
    grade_relevance: bool = True
    max_retrieval_loops: int = 2
    context_exploration: bool = False     # retrieve a few snippets first to ground rewrite/decompose
    context_exploration_top_k: int = 3
    auto_filter: AutoFilterSettings = AutoFilterSettings()
    tools: AgentToolsSettings = AgentToolsSettings()


class IngestionSettings(BaseModel):
    max_file_mb: int = 100
    dedup_by_hash: bool = True


class ChatSettings(BaseModel):
    memory_enabled: bool = True
    max_turns: int = 20
    checkpointer: str = "postgres"      # postgres | memory | sqlite
    sqlite_path: str = "chat_memory.db"  # used when checkpointer == sqlite


# ---------------------------------------------------------------------------
# Root settings — env vars (RAG_SECTION__KEY) override YAML
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = AppSettings()
    logging: LoggingSettings = LoggingSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    api: APISettings = APISettings()
    database: DatabaseSettings = DatabaseSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    graph_store: GraphStoreSettings = GraphStoreSettings()
    cache: CacheSettings = CacheSettings()
    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    llm: LLMSettings = LLMSettings()
    reranker: RerankerSettings = RerankerSettings()
    parsing: ParsingSettings = ParsingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    enrichment: EnrichmentSettings = EnrichmentSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    agent: AgentSettings = AgentSettings()
    ingestion: IngestionSettings = IngestionSettings()
    chat: ChatSettings = ChatSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # Priority: init > env > .env file > config.yaml > defaults
        return init_settings, env_settings, dotenv_settings, _YamlSource(settings_cls), file_secret_settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
