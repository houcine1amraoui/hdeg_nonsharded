"""V1-B metric and bounded-memory utilities.

No scientific decision logic belongs in this module.

R3 Energy Distance note
------------------------
R3 compares empirical distributions of *whole representation observations*
within one hierarchy level.  For a structured level (Z, S, S_tilde), a
single observation is vectorized only as an implementation of its native
Frobenius norm.  Hierarchy levels are never concatenated with each other.

The R3 Energy Distance is the multivariate metric form:

    E(P,Q) = sqrt( 2 E||X-Y|| - E||X-X'|| - E||Y-Y'|| )

where the empirical expectations use the V-statistic form (including the
zero diagonal terms in the within-sample sums).  The implementation below
uses tiled pairwise-distance accumulation so no full n-by-m distance matrix
is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

import numpy as np


def flatten_native(x: np.ndarray) -> np.ndarray:
    """Return per-sample native vectors for distance computation.

    This is an implementation-level vectorization of a single sample.
    It does NOT concatenate hierarchy levels. For structured Z/S/S_tilde
    representations, the complete native sample is vectorized only to
    evaluate its within-level Euclidean distance, equivalent to Frobenius
    distance in the original structured tensor.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    return x.reshape(x.shape[0], -1)


def native_pair_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance for paired samples at one hierarchy level."""
    aa = flatten_native(a).astype(np.float64, copy=False)
    bb = flatten_native(b).astype(np.float64, copy=False)
    if aa.shape != bb.shape:
        raise ValueError(f"Paired shapes differ: {aa.shape} vs {bb.shape}")
    return np.linalg.norm(aa - bb, axis=1)


def within_chunk_pair_distances(x: np.ndarray) -> np.ndarray:
    """Upper-triangular pairwise distances for one bounded chunk.

    Used only for bounded R1 distribution sampling.
    """
    v = flatten_native(x).astype(np.float64, copy=False)
    n = len(v)
    if n < 2:
        return np.empty(0, dtype=np.float64)
    out = []
    for i in range(n - 1):
        d = np.linalg.norm(v[i + 1:] - v[i], axis=1)
        out.append(d)
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)


def behavioral_window_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Frozen R2 normalized Frobenius distance.

    d_X = ||X_i-X_j||_F / (W*N)

    W=30, N=23 for CU/HDEG V1.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"Window shapes differ: {x1.shape} vs {x2.shape}")
    return float(np.linalg.norm(
        np.asarray(x1, dtype=np.float64) -
        np.asarray(x2, dtype=np.float64)
    ) / float(x1.size))


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without requiring sklearn."""
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.statistic)


def _validate_metric_inputs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return per-sample 2-D float64 views/copies."""
    xx = flatten_native(np.asarray(x)).astype(np.float64, copy=False)
    yy = flatten_native(np.asarray(y)).astype(np.float64, copy=False)

    if xx.ndim != 2 or yy.ndim != 2:
        raise ValueError("Energy-distance inputs must contain sample rows.")
    if xx.shape[0] == 0 or yy.shape[0] == 0:
        raise ValueError("Energy-distance inputs must be non-empty.")
    if xx.shape[1] != yy.shape[1]:
        raise ValueError(
            "Energy-distance inputs must have the same per-sample dimension: "
            f"{xx.shape[1]} vs {yy.shape[1]}"
        )
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("Energy-distance inputs must contain only finite values.")

    return xx, yy


def pairwise_distance_sum_tiled(
    x: np.ndarray,
    y: np.ndarray,
    *,
    row_block_size: int = 512,
    col_block_size: int = 512,
) -> float:
    """Return sum_ij ||x_i-y_j||_2 using bounded-memory tiling.

    The complete pairwise distance matrix is never materialized. Squared
    distances are computed from Gram blocks:

        ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x y^T.

    Accumulation is performed in float64.
    """
    if row_block_size <= 0 or col_block_size <= 0:
        raise ValueError("row_block_size and col_block_size must be positive.")

    xx, yy = _validate_metric_inputs(x, y)
    y_norms = np.einsum("ij,ij->i", yy, yy, dtype=np.float64)
    total = 0.0

    for i in range(0, len(xx), row_block_size):
        xb = xx[i:i + row_block_size]
        x_norms = np.einsum("ij,ij->i", xb, xb, dtype=np.float64)

        for j in range(0, len(yy), col_block_size):
            yb = yy[j:j + col_block_size]
            squared = x_norms[:, None] + y_norms[j:j + len(yb)][None, :]
            squared = squared - (2.0 * (xb @ yb.T))

            # Roundoff can produce tiny negative squared distances.
            np.maximum(squared, 0.0, out=squared)
            total += float(np.sqrt(squared, dtype=np.float64).sum(dtype=np.float64))

    return float(total)


def _energy_squared_from_sums(
    cross_sum: float,
    xx_sum: float,
    yy_sum: float,
    n: int,
    m: int,
) -> float:
    """Convert empirical pairwise sums to squared Energy Distance."""
    value = (
        2.0 * (cross_sum / float(n * m))
        - (xx_sum / float(n * n))
        - (yy_sum / float(m * m))
    )

    # Energy^2 is mathematically non-negative. Permit only tiny numerical
    # negatives caused by floating-point cancellation.
    scale = max(
        1.0,
        abs(2.0 * cross_sum / float(n * m)),
        abs(xx_sum / float(n * n)),
        abs(yy_sum / float(m * m)),
    )
    tolerance = 1e-12 * scale

    if value < 0.0:
        if value >= -tolerance:
            return 0.0
        raise FloatingPointError(
            "Squared Energy Distance became materially negative: "
            f"value={value:.17g}, tolerance={tolerance:.17g}"
        )

    return float(value)


def energy_distance_reference(x: np.ndarray, y: np.ndarray) -> float:
    """Brute-force multivariate empirical Energy Distance.

    TEST/REFERENCE ONLY. This intentionally materializes full pairwise
    distance matrices and must never be used on the 50k-sample CU run.
    """
    xx, yy = _validate_metric_inputs(x, y)

    xy = np.linalg.norm(xx[:, None, :] - yy[None, :, :], axis=2)
    xx_dist = np.linalg.norm(xx[:, None, :] - xx[None, :, :], axis=2)
    yy_dist = np.linalg.norm(yy[:, None, :] - yy[None, :, :], axis=2)

    squared = _energy_squared_from_sums(
        float(xy.sum(dtype=np.float64)),
        float(xx_dist.sum(dtype=np.float64)),
        float(yy_dist.sum(dtype=np.float64)),
        len(xx),
        len(yy),
    )
    return float(math.sqrt(squared))


def energy_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    row_block_size: int = 512,
    col_block_size: int = 512,
) -> float:
    """Compute multivariate empirical Energy Distance with bounded memory.

    The returned value is the metric form:

        sqrt(2 E||X-Y|| - E||X-X'|| - E||Y-Y'||).

    For Z/S/S_tilde, vectorization is only an implementation of the native
    Frobenius metric for one hierarchy level. Hierarchy levels are never
    concatenated here.
    """
    xx, yy = _validate_metric_inputs(x, y)

    cross_sum = pairwise_distance_sum_tiled(
        xx,
        yy,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
    )
    xx_sum = pairwise_distance_sum_tiled(
        xx,
        xx,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
    )
    yy_sum = pairwise_distance_sum_tiled(
        yy,
        yy,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
    )

    squared = _energy_squared_from_sums(
        cross_sum,
        xx_sum,
        yy_sum,
        len(xx),
        len(yy),
    )
    return float(math.sqrt(squared))


def percentile_summary(values: np.ndarray) -> dict:
    """Robust R1 distribution summary."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {
            "count": 0,
            "median": None,
            "iqr": None,
            "q01": None,
            "q05": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q95": None,
            "q99": None,
            "mean": None,
            "std": None,
        }
    qs = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(len(v)),
        "median": float(qs[3]),
        "iqr": float(qs[4] - qs[2]),
        "q01": float(qs[0]),
        "q05": float(qs[1]),
        "q25": float(qs[2]),
        "q50": float(qs[3]),
        "q75": float(qs[4]),
        "q95": float(qs[5]),
        "q99": float(qs[6]),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
    }


@dataclass
class Reservoir:
    """Deterministic bounded reservoir for R1 distribution evidence."""
    capacity: int
    seed: int = 42

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.rng = np.random.default_rng(self.seed)
        self.data = np.empty(self.capacity, dtype=np.float64)
        self.size = 0
        self.seen = 0

    def update(self, values: Iterable[float]) -> None:
        for value in values:
            value = float(value)
            if not math.isfinite(value):
                continue
            self.seen += 1
            if self.size < self.capacity:
                self.data[self.size] = value
                self.size += 1
            else:
                j = int(self.rng.integers(0, self.seen))
                if j < self.capacity:
                    self.data[j] = value

    def values(self) -> np.ndarray:
        return self.data[:self.size].copy()


def moving_block_bootstrap_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate n indices using a moving-block bootstrap."""
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if block_length > n:
        block_length = n

    starts = np.arange(0, n - block_length + 1, dtype=np.int64)
    result = []
    total = 0
    while total < n:
        s = int(rng.choice(starts))
        block = np.arange(s, s + block_length, dtype=np.int64)
        result.append(block)
        total += len(block)
    return np.concatenate(result)[:n]


def bootstrap_energy_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    block_length: int = 30,
    replicates: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
    row_block_size: int = 512,
    col_block_size: int = 512,
) -> dict:
    """Moving-block bootstrap CI for multivariate Energy Distance.

    Bootstrap is performed independently within each condition. The R3
    temporal block protocol remains unchanged; only the underlying Energy
    Distance implementation is replaced by the bounded-memory multivariate
    kernel above.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    observed = energy_distance(
        x,
        y,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
    )
    vals = np.empty(replicates, dtype=np.float64)

    for b in range(replicates):
        ix = moving_block_bootstrap_indices(len(x), block_length, rng)
        iy = moving_block_bootstrap_indices(len(y), block_length, rng)
        vals[b] = energy_distance(
            x[ix],
            y[iy],
            row_block_size=row_block_size,
            col_block_size=col_block_size,
        )

    alpha = 1.0 - confidence
    lo, hi = np.quantile(vals, [alpha / 2.0, 1.0 - alpha / 2.0])

    return {
        "point_estimate": float(observed),
        "confidence_level": float(confidence),
        "interval_method": "percentile",
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "bootstrap_replicates": int(replicates),
        "block_length_windows": int(block_length),
        "bootstrap_seed": int(seed),
    }
