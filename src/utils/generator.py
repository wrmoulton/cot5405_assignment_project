"""Utilities for generating random assignment-problem instances."""

from __future__ import annotations

import random
from typing import List


Matrix = List[List[int]]


def generate_cost_matrix(n: int, low: int = 1, high: int = 1000, seed: int | None = None) -> Matrix:
    if n <= 0:
        raise ValueError("n must be positive")
    if low > high:
        raise ValueError("low must be <= high")

    rng = random.Random(seed)
    return [[rng.randint(low, high) for _ in range(n)] for _ in range(n)]
