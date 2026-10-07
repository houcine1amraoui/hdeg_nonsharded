"""Small tests for the flattened V1-B package.

These tests are intentionally synthetic and do not use CU artifacts.
"""

from pathlib import Path
import tempfile

import numpy as np

from v1b_metrics import (
    behavioral_window_distance,
    bootstrap_energy_ci,
    energy_distance,
    spearman_rho,
)
from v1b_sampling import deterministic_pairs, deterministic_sample_indices


def test_pair_sampling_deterministic():
    a = deterministic_pairs(1000, 1000, seed=42)
    b = deterministic_pairs(1000, 1000, seed=42)
    assert np.array_equal(a, b)
    assert np.all(a[:, 0] < a[:, 1])


def test_sample_indices_deterministic():
    a = deterministic_sample_indices(1000, 100, seed=42)
    b = deterministic_sample_indices(1000, 100, seed=42)
    assert np.array_equal(a, b)
    assert len(np.unique(a)) == 100


def test_basic_metrics():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 64))
    y = rng.normal(loc=0.5, size=(100, 64))

    e = energy_distance(x, y)
    assert np.isfinite(e)

    ci = bootstrap_energy_ci(
        x, y,
        block_length=30,
        replicates=20,
        seed=42,
    )
    assert np.isfinite(ci["point_estimate"])
    assert ci["ci_lower"] <= ci["ci_upper"]

    bx = np.arange(100, dtype=float)
    by = bx + rng.normal(scale=0.01, size=100)
    rho = spearman_rho(bx, by)
    assert rho > 0.99

    w1 = rng.normal(size=(30, 23))
    w2 = rng.normal(size=(30, 23))
    d = behavioral_window_distance(w1, w2)
    assert np.isfinite(d)


if __name__ == "__main__":
    test_pair_sampling_deterministic()
    test_sample_indices_deterministic()
    test_basic_metrics()
    print("All V1-B synthetic tests passed.")
