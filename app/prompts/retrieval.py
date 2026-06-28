"""LLM prompts for the retrieval layer (answer synthesis + metadata auto-filter)."""

from typing import List

from app.models.ingestion import MetadataKey

ANSWER_SYSTEM = (
    "You are a precise, factual assistant. "
    "Answer the question using ONLY the provided context. "
    "If the context does not contain enough information, say so clearly. "
    "Do not add information that is not in the context."
)


QUERY_ENTITY_SYSTEM = (
    "List the named entities (people, organisations, places, products, concepts) the "
    "user's question is about. Return only the entity names as written, no explanation. "
    "Return an empty list if the question names none."
)


def allowed_keys_prompt(core_fields: List[str], catalog: List[MetadataKey]) -> str:
    """Render the filterable keys (core + discovered) with sample values for the auto-filter."""
    lines = [f"- {field}" for field in core_fields]
    for entry in catalog:
        if entry.value_samples:
            sample = ", ".join(entry.value_samples[-8:])
            suffix = "; many values" if entry.high_cardinality else ""
            lines.append(f"- {entry.key} (e.g. {sample}{suffix})")
        else:
            lines.append(f"- {entry.key}")
    return "\n".join(lines)


def auto_filter_system(allowed_keys: str) -> str:
    return (
        "Extract exact-match metadata filters from the user's question, ONLY for "
        "these keys:\n"
        f"{allowed_keys}\n"
        "Include a key only when the question explicitly and unambiguously "
        "constrains it (e.g. 'in section 302' -> key=section, values=[302]). If "
        "the question allows several values for a key (e.g. 'comedies or "
        "tragedies'), list all of them. Never guess; return no conditions if "
        "nothing clearly applies."
    )
