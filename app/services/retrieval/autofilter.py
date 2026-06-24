import logging
from typing import Dict, List

from pydantic import BaseModel, Field

from app.services.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class _InferredFilters(BaseModel):
    filters: Dict[str, str] = Field(default_factory=dict)


async def infer_filters(llm: BaseLLMProvider, query: str, fields: List[str]) -> Dict[str, str]:
    """Ask the LLM to pull metadata filters out of the query — but only for the allowed
    `fields`, and only when the query clearly constrains them. Returns {} on any doubt or
    error. Callers apply these softly (fall back to no filters if the result is empty)."""
    if not fields:
        return {}

    field_list = ", ".join(fields)
    try:
        result: _InferredFilters = await llm.structured_output(
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Extract metadata filters from the user's question, ONLY for these fields: "
                        f"{field_list}. Include a field only when the question explicitly and "
                        "unambiguously constrains it (e.g. 'in the 10-Q filing' -> doc_type). "
                        "Never guess. Return an empty object if nothing clearly applies."
                    ),
                },
                {"role": "user", "content": query},
            ],
            schema=_InferredFilters,
        )
    except Exception as exc:
        logger.warning("Auto-filter inference failed: %s", exc)
        return {}

    # Keep only allowed, non-empty fields — guards against hallucinated keys.
    return {k: v for k, v in result.filters.items() if k in fields and v}
