"""Deterministic, representation-blind V1-B sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class Pair:
    i: int
    j: int


def deterministic_pairs(
    n: int,
    count: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    """Deterministically sample unordered sample-index pairs without replacement.

    Pair selection depends only on population size, seed and pair count.
    It never inspects representation values.
    """
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)

    total = n * (n - 1) // 2
    m = min(int(count), total)

    rng = np.random.default_rng(seed)

    # For the normal CU case total >> m. We sample encoded pair ranks.
    ranks = rng.choice(total, size=m, replace=False)
    ranks.sort()

    pairs = np.empty((m, 2), dtype=np.int64)

    # Decode a lexicographic upper-triangle rank.
    # row i contains pairs (i,j), j>i, of length n-i-1.
    for k, r in enumerate(ranks):
        r = int(r)
        lo, hi = 0, n - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            # number of pairs before row mid
            before = mid * (2 * n - mid - 1) // 2
            if before <= r:
                lo = mid + 1
            else:
                hi = mid - 1
        i = max(0, lo - 1)
        before = i * (2 * n - i - 1) // 2
        j = i + 1 + (r - before)
        pairs[k] = (i, j)

    return pairs


def deterministic_sample_indices(
    n: int,
    count: int,
    seed: int = 42,
) -> np.ndarray:
    """Deterministic without-replacement sample of population indices."""
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    m = min(int(count), n)
    rng = np.random.default_rng(seed)
    out = rng.choice(n, size=m, replace=False)
    out.sort()
    return out


def condition_sampling_plan(
    populations: Dict[str, int],
    max_per_condition: int = 50_000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Create representation-blind per-condition sample indices."""
    return {
        name: deterministic_sample_indices(
            n,
            min(max_per_condition, n),
            seed=seed + k,
        )
        for k, (name, n) in enumerate(sorted(populations.items()))
    }
