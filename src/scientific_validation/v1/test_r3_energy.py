"""Mathematical and bounded-memory conformance tests for R3 Energy Distance."""

from __future__ import annotations

import gc
import os
import tracemalloc

import numpy as np
import pytest

from v1b_metrics import (
    energy_distance,
    energy_distance_reference,
    pairwise_distance_sum_tiled,
)


@pytest.mark.parametrize("row_block,col_block", [(1, 1), (2, 3), (4, 5), (8, 11)])
def test_tiled_matches_bruteforce_small(row_block, col_block):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(9, 5))
    y = rng.normal(size=(7, 5))

    ref = energy_distance_reference(x, y)
    got = energy_distance(
        x,
        y,
        row_block_size=row_block,
        col_block_size=col_block,
    )

    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-9)


def test_multivariate_known_singletons():
    x = np.array([[0.0, 0.0]])
    y = np.array([[3.0, 0.0]])

    expected = np.sqrt(6.0)
    np.testing.assert_allclose(energy_distance_reference(x, y), expected)
    np.testing.assert_allclose(energy_distance(x, y), expected)


def test_identical_distributions_are_zero():
    x = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(energy_distance(x, x), 0.0, atol=1e-12)


def test_symmetry():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(11, 4))
    y = rng.normal(size=(8, 4))

    np.testing.assert_allclose(
        energy_distance(x, y),
        energy_distance(y, x),
        rtol=1e-12,
        atol=1e-12,
    )


def test_permutation_invariance():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(13, 6))
    y = rng.normal(size=(10, 6))

    xp = x[rng.permutation(len(x))]
    yp = y[rng.permutation(len(y))]

    np.testing.assert_allclose(
        energy_distance(x, y),
        energy_distance(xp, yp),
        rtol=1e-12,
        atol=1e-12,
    )


def test_duplicate_observations_are_supported():
    x = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
    y = np.array([[0.0, 0.0], [2.0, 2.0]])

    np.testing.assert_allclose(
        energy_distance(x, y),
        energy_distance_reference(x, y),
        rtol=1e-12,
        atol=1e-12,
    )


def test_native_frobenius_equals_vectorized_euclidean():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(8, 3, 4))
    y = rng.normal(size=(7, 3, 4))

    # The production API accepts native structured observations and performs
    # only per-observation vectorization.
    got_native = energy_distance(x, y, row_block_size=3, col_block_size=2)

    # Explicit vectorization must be mathematically identical to the native
    # Frobenius geometry.
    got_flat = energy_distance(
        x.reshape(len(x), -1),
        y.reshape(len(y), -1),
        row_block_size=3,
        col_block_size=2,
    )

    np.testing.assert_allclose(got_native, got_flat, rtol=1e-8, atol=1e-9)


def test_actual_hdeg_dimensions_against_reference():
    rng = np.random.default_rng(23)
    for dim in (64, 576, 1472):
        x = rng.normal(size=(6, dim))
        y = rng.normal(size=(5, dim))

        ref = energy_distance_reference(x, y)
        got = energy_distance(x, y, row_block_size=2, col_block_size=3)

        np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-9)


def test_pairwise_sum_tile_size_invariance():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(17, 7))
    y = rng.normal(size=(13, 7))

    reference = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2).sum()

    for rb, cb in [(1, 1), (2, 3), (4, 5), (8, 11)]:
        got = pairwise_distance_sum_tiled(
            x,
            y,
            row_block_size=rb,
            col_block_size=cb,
        )
        np.testing.assert_allclose(got, reference, rtol=1e-8, atol=1e-9)


def test_input_validation():
    with pytest.raises(ValueError):
        energy_distance(np.empty((0, 3)), np.ones((2, 3)))

    with pytest.raises(ValueError):
        energy_distance(np.ones((2, 3)), np.ones((2, 4)))

    with pytest.raises(ValueError):
        energy_distance(np.array([[np.nan, 1.0]]), np.ones((1, 2)))

    with pytest.raises(ValueError):
        energy_distance(np.ones((2, 3)), np.ones((2, 3)), row_block_size=0)


def test_no_full_pairwise_matrix_medium_memory_profile():
    """Medium test: tiled workspace remains bounded as n grows.

    This deliberately does not use 50k samples and does not touch CU.
    It checks that the implementation works on a moderately sized
    high-dimensional representation with a small fixed tile.
    """
    rng = np.random.default_rng(31)
    n = 600
    m = 550
    d = 1472
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.normal(size=(m, d)).astype(np.float32)

    tracemalloc.start()
    got = energy_distance(
        x,
        y,
        row_block_size=32,
        col_block_size=32,
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    assert np.isfinite(got)
    assert got >= 0.0

    # The pairwise distance workspace is 32x32; allow generous interpreter
    # and BLAS/runtime overhead. This is a guard against accidental creation
    # of an n-by-m-by-d temporary in Python/NumPy.
    assert peak < 250 * 1024 * 1024, f"Unexpected peak memory: {peak / 2**20:.1f} MiB"


def test_medium_reference_agreement_lower_dimension():
    """Medium numerical agreement where brute force is still feasible."""
    rng = np.random.default_rng(37)
    x = rng.normal(size=(80, 32))
    y = rng.normal(size=(70, 32))

    ref = energy_distance_reference(x, y)
    got = energy_distance(x, y, row_block_size=13, col_block_size=17)

    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-9)
