"""LLM prompts for ingest-time extraction (document metadata, section metadata, entities)."""

from typing import List

from app.models.ingestion import MetadataKey


def catalog_hint(catalog: List[MetadataKey], attribute_hints: List[str]) -> str:
    """Tell the extractor which keys already exist (reuse them) so the key-space converges.
    Low-cardinality keys show their known values; high-cardinality keys show the key only."""
    lines = []
    for entry in catalog:
        if entry.value_samples:
            sample = ", ".join(entry.value_samples[-8:])
            suffix = "; many values" if entry.high_cardinality else ""
            lines.append(f"- {entry.key} (e.g. {sample}{suffix})")
        else:
            lines.append(f"- {entry.key}")
    existing = "\n".join(lines) if lines else "(none yet)"
    suggested = ", ".join(attribute_hints) if attribute_hints else "(none)"
    return (
        f"Existing attribute keys for this knowledge base — REUSE these when they fit, only "
        f"add a key for a genuinely new concept:\n{existing}\n"
        f"Other commonly useful attribute keys: {suggested}."
    )


def document_metadata_system(hint: str) -> str:
    return (
        "Extract document-level metadata from the text.\n"
        "Core fields: title, author, doc_type (report, contract, judgment, email, etc.), "
        "summary (1-2 sentences), language, created_date (ISO 8601 if stated).\n"
        "Also extract document-type-specific attributes as key/value pairs — e.g. for a "
        "law it might be section/chapter/category; for a play, genre/setting. Use short "
        "snake_case keys.\n" + hint + "\n"
        "Use empty strings / empty list when unknown. Do not invent values."
    )


def section_metadata_system(hint: str) -> str:
    return (
        "Extract section-specific attributes from this passage as key/value "
        "pairs — e.g. section/clause numbers, prices, part numbers, dates. Use "
        "short snake_case keys.\n" + hint + "\n"
        "Return no attributes if the passage has none. Do not invent values."
    )


GRAPH_EXTRACTION_SYSTEM = (
    "Extract a knowledge graph from the text.\n"
    "1. Entities: named people, organisations, places, products, or concepts. For each give "
    "a display name, a type (Person, Org, Place, Product, Concept), and a 1-2 sentence "
    "description grounded in the text.\n"
    "2. Relations: meaningful relationships between the entities you listed, as "
    "(source, type, target, description). source and target MUST be names from your entity "
    "list; type is a short snake_case verb phrase (e.g. founded, acquired, located_in, "
    "part_of); description is one short clause on how they relate.\n"
    "Only relate entities you listed. Do not invent facts. "
    "Return JSON with 'entities' and 'relations' arrays."
)


def community_summary_system(max_chars: int) -> str:
    return (
        "You are given the entities and relationships of one community in a knowledge graph. "
        "Write a short report: a 'title' naming the community's main theme, and a 'summary' "
        f"of {max_chars} characters or fewer covering what this community is about and how its "
        "members relate. Use only the given information; do not invent facts."
    )


DESCRIPTION_SUMMARY_SYSTEM = (
    "Several descriptions of the same entity, gathered from different passages, are given "
    "below separated by ' | '. Merge them into one coherent description of a few sentences. "
    "Keep every distinct fact; drop repetition. Return only the merged description text."
)
