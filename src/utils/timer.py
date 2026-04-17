"""Timing helpers for empirical runtime measurements."""

from __future__ import annotations

import time
from typing import Any, Callable, Tuple



def time_function(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Execute a function and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start
