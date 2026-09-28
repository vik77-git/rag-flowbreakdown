from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def stopwatch():
    """Yields a callable returning elapsed milliseconds."""
    start = time.perf_counter()
    yield lambda: round((time.perf_counter() - start) * 1000, 1)
