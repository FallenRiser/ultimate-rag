import time
from contextlib import contextmanager
from typing import Dict, Iterator


class StageTimer:
    """Accumulates per-stage durations (milliseconds) for the response's app_timings field."""

    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    @contextmanager
    def track(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[stage] = round((time.perf_counter() - start) * 1000.0, 1)

    def record(self, stage: str, milliseconds: float) -> None:
        self.timings[stage] = round(milliseconds, 1)

    def with_total(self) -> Dict[str, float]:
        result = dict(self.timings)
        result["total"] = round(sum(self.timings.values()), 1)
        return result
