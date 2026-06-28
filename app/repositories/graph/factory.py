from typing import Optional

from app.repositories.base import BaseGraphStore
from app.utils.config import get_settings


def create_graph_store() -> Optional[BaseGraphStore]:
    settings = get_settings()
    if not settings.graph_store.enabled:
        return None

    provider = settings.graph_store.provider
    if provider == "age":
        from app.repositories.graph.age import AgeRepository
        return AgeRepository(
            dsn=settings.database.dsn,
            graph_name=settings.graph_store.graph_name,
        )

    if provider == "networkx":
        from app.repositories.graph.networkx_store import NetworkXRepository
        return NetworkXRepository(graph_dir=settings.graph_store.graph_dir)

    raise ValueError(f"Unknown graph_store.provider: {provider!r}")
