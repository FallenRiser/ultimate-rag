import json
import logging
import time
from typing import Any, List, Optional

from app.repositories.base import BaseCacheBackend

logger = logging.getLogger(__name__)


class IgniteCache(BaseCacheBackend):
    """Apache Ignite thin-client cache. TTL enforced via a JSON envelope on each value."""

    def __init__(self, addresses: List[str], cache_name: str):
        self.addresses = addresses
        self.cache_name = cache_name
        self._client = None
        self._cache = None

    async def _get_cache(self):
        if self._cache is None:
            from pyignite import AioClient
            self._client = AioClient()
            host_port_pairs = [
                (addr.split(":")[0], int(addr.split(":")[1]))
                for addr in self.addresses
            ]
            await self._client.connect(host_port_pairs)
            self._cache = await self._client.get_or_create_cache(self.cache_name)
        return self._cache

    async def get(self, key: str) -> Optional[Any]:
        try:
            cache = await self._get_cache()
            raw = await cache.get(key)
            if raw is None:
                return None
            envelope = json.loads(raw)
            if time.time() > envelope["exp"]:
                await cache.remove(key)
                return None
            return envelope["v"]
        except Exception as exc:
            logger.warning("Ignite cache get failed: %s", exc)
            return None

    async def set(self, key: str, value: Any, ttl: int) -> None:
        try:
            cache = await self._get_cache()
            envelope = json.dumps({"v": value, "exp": time.time() + ttl})
            await cache.put(key, envelope)
        except Exception as exc:
            logger.warning("Ignite cache set failed: %s", exc)

    async def delete(self, key: str) -> None:
        try:
            cache = await self._get_cache()
            await cache.remove(key)
        except Exception as exc:
            logger.warning("Ignite cache delete failed: %s", exc)
