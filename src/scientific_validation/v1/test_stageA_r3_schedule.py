"""Unit test for the Stage-A R3 execution schedule.

This is a scheduler test, not a scientific R3 run. The metric function is
stubbed only to verify that the existing metric call is made with one level
and at most two conditions resident at a time.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

import src.scientific_validation.v1.v1b_engine as engine_module
from src.scientific_validation.v1.v1b_engine import V1BConfig, V1BEngine, LEVELS


def test_comparison_streamed_level_schedule():
    cfg = V1BConfig(
        artifact_root=Path("/unused"),
        window_root=Path("/unused"),
        output_root=Path("/unused"),
        r3_max_per_condition=16,
        bootstrap_replicates=3,
    )
    e = V1BEngine(cfg)

    # Synthetic metadata only; no scientific values are used for decisions.
    manifests = {
        c: {
            "split": split,
            "sample_count": 32,
            "shards": [{
                "shard_index": 0,
                "start_index": 0,
                "end_index": 32,
                "sample_count": 32,
            }],
        }
        for c, split in {
            "N1": "val",
            "A2": "actor2_test",
            "N2": "actor1_test",
        }.items()
    }
    e._build_split_index_manifest = lambda split: next(
        m for m in manifests.values() if m["split"] == split
    )

    calls = []
    active = 0
    max_active = 0

    def fake_collect(split, level, selected_global, manifest):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        # Distinct values make accidental cross-level reuse visible.
        arr = np.full((len(selected_global), 2), float(LEVELS.index(level)))
        # Keep a release callback impossible; emulate ownership by returning
        # an ordinary array and decrement immediately after the caller copies
        # the reference into its local variable. The scheduler itself is the
        # subject of this test, not Python allocator behavior.
        active -= 1
        return arr

    e._collect_split_representation_level = fake_collect

    old_bootstrap = engine_module.bootstrap_energy_ci

    def fake_bootstrap(x, y, **kwargs):
        calls.append((x.shape, y.shape, kwargs["block_length"], kwargs["replicates"]))
        return {
            "point_estimate": 0.0,
            "confidence_level": 0.95,
            "interval_method": "percentile",
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "bootstrap_replicates": kwargs["replicates"],
            "block_length_windows": kwargs["block_length"],
            "bootstrap_seed": kwargs["seed"],
        }

    engine_module.bootstrap_energy_ci = fake_bootstrap
    try:
        result = e.run_r3()
    finally:
        engine_module.bootstrap_energy_ci = old_bootstrap

    assert len(calls) == 12  # 3 comparisons x 4 levels
    assert all(a == (16, 2) and b == (16, 2) for a, b, _, _ in calls)
    assert result["execution"]["max_conditions_resident"] == 2
    assert result["execution"]["max_levels_resident"] == 1
    assert result["sampling"]["sampling_reused_across_levels"] is True
    assert set(result["comparisons"]) == {"N1-A2", "A2-N2", "N1-N2"}
    for comparison in result["comparisons"].values():
        assert set(comparison) == set(LEVELS)


if __name__ == "__main__":
    test_comparison_streamed_level_schedule()
    print("Stage-A R3 scheduler test: PASS")
