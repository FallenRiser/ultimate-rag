import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


def setup_tracing() -> None:
    from app.utils.config import get_settings
    cfg = get_settings().observability.mlflow

    if not cfg.enabled:
        return

    try:
        import mlflow

        mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment(cfg.experiment)

        if cfg.autolog.langchain:
            mlflow.langchain.autolog()
        if cfg.autolog.openai:
            mlflow.openai.autolog()

        logger.info("MLflow tracing enabled at %s (experiment: %s)", cfg.tracking_uri, cfg.experiment)
    except ImportError:
        logger.warning("mlflow not installed — tracing disabled")
    except Exception as exc:
        logger.warning("MLflow setup failed: %s", exc)


@contextmanager
def request_trace(name: str, inputs: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """Open one root span for a whole request flow so the embedding and LLM calls (each of
    which MLflow autologs) nest under a single trace instead of fragmenting into separate
    top-level traces. No-op (yields None) when MLflow tracing is disabled or unavailable."""
    from app.utils.config import get_settings

    if not get_settings().observability.mlflow.enabled:
        yield None
        return

    try:
        import mlflow
    except ImportError:
        yield None
        return

    with mlflow.start_span(name=name) as span:
        if inputs:
            span.set_inputs(inputs)
        yield span


def log_chunks_to_trace(span: Any, chunks: List[Dict[str, Any]]) -> None:
    """Record retrieved chunks on the active request span. Chunk text is truncated to
    observability.mlflow.max_chunk_chars (0 = full). Never raises — tracing must not break
    a request."""
    if span is None:
        return

    from app.utils.config import get_settings
    from app.utils.text import cap_text

    max_chars = get_settings().observability.mlflow.max_chunk_chars
    try:
        preview = [
            {
                "id": chunk.get("id"),
                "score": chunk.get("rerank_score", chunk.get("score")),
                "sources": chunk.get("sources", []),
                "text": cap_text(chunk.get("payload", {}).get("text", ""), max_chars),
            }
            for chunk in chunks
        ]
        span.set_attribute("retrieved_chunks", preview)
    except Exception as exc:
        logger.debug("Failed to log chunks to trace: %s", exc)
