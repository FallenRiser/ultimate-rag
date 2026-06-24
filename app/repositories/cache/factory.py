from typing import Optional

from app.repositories.base import BaseCacheBackend
from app.utils.config import get_settings


def create_cache() -> Optional[BaseCacheBackend]:
    settings = get_settings()
    cfg = settings.cache

    if not cfg.enabled:
        return None

    if cfg.provider == "ignite":
        from app.repositories.cache.ignite import IgniteCache
        return IgniteCache(addresses=cfg.ignite_addresses, cache_name=cfg.cache_name)

    raise ValueError(f"Unknown cache.provider: {cfg.provider!r}")
