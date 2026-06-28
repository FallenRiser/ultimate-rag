import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.models.ingestion import MetadataKey

logger = logging.getLogger(__name__)

_VALUE_SAMPLE_CAP = 20   # distinct values cached per key before it's flagged high-cardinality


def _merge_sample(samples: List[str], value: str, high_cardinality: bool) -> Tuple[List[str], bool]:
    """Maintain a rolling window of the most-recent distinct values (cap _VALUE_SAMPLE_CAP).
    A known value is refreshed to most-recent; a new value evicts the oldest once full. Crossing
    the cap flips high_cardinality (informational — the key has more values than we cache), but
    samples keep rolling so auto-filter always sees current examples."""
    if value in samples:
        return [s for s in samples if s != value] + [value], high_cardinality  # recency refresh
    merged = samples + [value]
    if len(merged) > _VALUE_SAMPLE_CAP:
        return merged[-_VALUE_SAMPLE_CAP:], True   # evict oldest
    return merged, high_cardinality


async def get_catalog(engine: AsyncEngine, user_id: str) -> List[MetadataKey]:
    """All discovered attribute keys for a tenant, most-seen first."""
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT key, value_samples, high_cardinal
            FROM metadata_keys WHERE user_id = :user_id ORDER BY seen_count DESC
        """), {"user_id": user_id})
        rows = result.fetchall()
    catalog = [
        MetadataKey(
            key=row.key,
            value_samples=json.loads(row.value_samples) if row.value_samples else [],
            high_cardinality=bool(row.high_cardinal),
        )
        for row in rows
    ]
    logger.debug("Catalog for user %s: %d attribute keys", user_id, len(catalog))
    return catalog


async def record_keys(engine: AsyncEngine, user_id: str, attributes: Dict[str, str]) -> None:
    """Upsert each attribute key into the tenant catalog (read-modify-write per key).
    ponytail: not atomic across concurrent ingests for the same tenant+key — fine at this
    scale; add a row lock if heavy parallel ingest of one tenant ever causes lost samples."""
    now = datetime.utcnow()
    async with engine.begin() as conn:
        for key, value in attributes.items():
            row = (await conn.execute(text("""
                SELECT value_samples, high_cardinal FROM metadata_keys
                WHERE user_id = :user_id AND key = :key
            """), {"user_id": user_id, "key": key})).fetchone()

            samples = json.loads(row.value_samples) if (row and row.value_samples) else []
            high = bool(row.high_cardinal) if row else False
            samples, high = _merge_sample(samples, value, high)

            await conn.execute(text("""
                INSERT INTO metadata_keys (user_id, key, value_samples, high_cardinal, seen_count, updated_at)
                VALUES (:user_id, :key, :samples, :high, 1, :now)
                ON CONFLICT (user_id, key) DO UPDATE SET
                    value_samples = :samples,
                    high_cardinal = :high,
                    seen_count = metadata_keys.seen_count + 1,
                    updated_at = :now
            """), {
                "user_id": user_id, "key": key,
                "samples": json.dumps(samples), "high": 1 if high else 0, "now": now,
            })
    logger.debug("Recorded %d attribute key(s) for user %s: %s", len(attributes), user_id, list(attributes))


if __name__ == "__main__":
    s, high = _merge_sample([], "comedy", False)
    assert s == ["comedy"] and high is False
    s, high = _merge_sample(s, "comedy", False)            # duplicate: set unchanged
    assert s == ["comedy"] and high is False
    s2, _ = _merge_sample(["a", "b", "c"], "a", False)     # recency refresh: move to most-recent
    assert s2 == ["b", "c", "a"]
    full = [str(i) for i in range(_VALUE_SAMPLE_CAP)]
    s, high = _merge_sample(full, "new", False)            # crossing the cap: evict oldest, flip
    assert high is True and len(s) == _VALUE_SAMPLE_CAP and "0" not in s and "new" in s
    s, high = _merge_sample(s, "newer", True)              # high-cardinality keeps rolling
    assert high is True and "newer" in s and len(s) == _VALUE_SAMPLE_CAP
    print("OK")
