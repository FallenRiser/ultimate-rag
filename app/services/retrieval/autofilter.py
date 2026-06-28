import logging
from typing import Any, Dict, List

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.models.agent import InferredFilters
from app.models.ingestion import MetadataKey
from app.prompts.retrieval import allowed_keys_prompt, auto_filter_system
from app.repositories.relational.metadata_catalog import get_catalog
from app.services.llm.base import BaseLLMProvider
from app.utils.config import AutoFilterSettings

logger = logging.getLogger(__name__)


async def infer_filters(
    llm: BaseLLMProvider,
    query: str,
    user_id: str,
    engine: AsyncEngine,
    settings: AutoFilterSettings,
) -> Dict[str, Any]:
    """Pull exact-match metadata filters out of the query — only for allowed keys (core fields
    plus this tenant's discovered attribute keys), and only when the query clearly constrains
    them. A key with several allowed values becomes a list (OR-matched). Returns {} on doubt or
    error; callers apply softly (fall back to no filters)."""
    # Non-breaking: a catalog failure must not break querying — fall back to core fields only.
    catalog: List[MetadataKey] = []
    if settings.use_catalog:
        try:
            catalog = (await get_catalog(engine, user_id))[: settings.max_catalog_keys]
        except SQLAlchemyError as exc:
            logger.warning("Catalog read failed for %s; auto-filter using core fields only: %s", user_id, exc)
            logger.debug("Catalog read DB failure (user=%s)", user_id, exc_info=True)
        except Exception as exc:
            logger.warning("Unexpected error reading catalog for %s; using core fields only: %s", user_id, exc)
            logger.debug("Unexpected catalog read failure (user=%s)", user_id, exc_info=True)

    allowed = set(settings.fields) | {entry.key for entry in catalog}
    if not allowed:
        return {}
    logger.debug("Auto-filter allowed keys for %s: %s", user_id, sorted(allowed))

    # Non-breaking: if filter inference fails, return no filters (callers fall back to semantic).
    try:
        result: InferredFilters = await llm.structured_output(
            messages=[
                {
                    "role": "system",
                    "content": auto_filter_system(allowed_keys_prompt(settings.fields, catalog)),
                },
                {"role": "user", "content": query},
            ],
            schema=InferredFilters,
        )
    except Exception as exc:
        logger.warning("Auto-filter inference failed; proceeding without filters: %s", exc)
        logger.debug("Auto-filter inference failure (user=%s, query=%r)", user_id, query, exc_info=True)
        return {}

    # Keep allowed keys only; normalize values to match how they're stored (lowercased).
    # A single value stays scalar; several become a list (OR-matched by the vector store).
    filters: Dict[str, Any] = {}
    for cond in result.conditions:
        key = cond.key.strip().lower()
        values = [v.strip().lower() for v in cond.values if v and v.strip()]
        if key in allowed and values:
            filters[key] = values[0] if len(values) == 1 else values
    logger.debug("Auto-filter inferred %d filter(s) from query: %s", len(filters), filters)
    return filters
