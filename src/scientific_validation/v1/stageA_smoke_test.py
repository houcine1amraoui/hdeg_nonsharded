"""Stage-A R3 shard-streaming smoke test.

Run inside the HDEG project environment, for example:

python stageA_smoke_test.py \
  --artifact-root /content/drive/MyDrive/02_research/hdeg/code/data/processed/CU \
  --shards 3 \
  --samples-per-shard 128

This test does not execute the 50,000-sample R3 measurement or bootstrap.
It verifies the new level-selective, shard-streamed retrieval path against
real persisted CU representation shards.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from src.scientific_validation.v1.v1b_engine import V1BConfig, V1BEngine, LEVELS, R3_CONDITIONS
from src.scientific_validation.v1.v1b_io import discover_shards, load_representation_shard
from src.scientific_validation.v1.v1b_sampling import deterministic_sample_indices


def inspect_real_shards(engine: V1BEngine, split: str, shard_limit: int, samples_per_shard: int):
    shard_ids = discover_shards(engine.cfg.artifact_root, split)
    if not shard_ids:
        raise RuntimeError(f"No shards found for split={split}")

    shard_ids = shard_ids[:shard_limit]
    print(f"\n[{split}] testing real shards: {shard_ids}")

    # Keep a small set of split-local selected indices spanning the first
    # real shards. These are used to exercise the level-specific collector.
    selected = []
    shard_manifest = []

    for sid in shard_ids:
        level_meta = {}
        for level in LEVELS:
            reps, meta = load_representation_shard(
                engine.cfg.artifact_root,
                split,
                sid,
                levels=(level,),
                return_metadata=True,
            )
            a = reps[level]
            m = meta[level]
            assert np.isfinite(a).all(), f"non-finite {split}/{sid:06d}/{level}"
            assert int(m["start_index"]) < int(m["end_index"])
            assert int(m["end_index"]) - int(m["start_index"]) == a.shape[0]
            level_meta[level] = (
                str(m["split"]), int(m["shard_index"]),
                int(m["start_index"]), int(m["end_index"]), a.shape
            )
            del a, reps, meta
            gc.collect()

        z = level_meta["Z"]
        for level in LEVELS[1:]:
            assert level_meta[level][:4] == z[:4], (
                f"cross-level provenance mismatch in {split}/{sid:06d}: "
                f"Z={z}, {level}={level_meta[level]}"
            )

        start, end = z[2], z[3]
        take = min(samples_per_shard, end - start)
        selected.extend(range(start, start + take))
        shard_manifest.append({
            "shard_index": sid,
            "start_index": start,
            "end_index": end,
            "sample_count": end - start,
        })

    manifest = {
        "split": split,
        "sample_count": max(s["end_index"] for s in shard_manifest),
        "shards": shard_manifest,
    }
    selected = np.asarray(selected, dtype=np.int64)

    # Exercise the exact Stage-A level-specific retrieval path.
    for level in LEVELS:
        arr = engine._collect_split_representation_level(
            split, level, selected, manifest
        )
        assert arr.shape[0] == selected.size
        assert np.isfinite(arr).all()
        print(f"  {level}: selected={arr.shape[0]}, shape={arr.shape}, dtype={arr.dtype}")
        del arr
        gc.collect()

    print(f"  PASS: {split} first {len(shard_ids)} real shard(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", required=True, type=Path)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--samples-per-shard", type=int, default=128)
    args = ap.parse_args()

    cfg = V1BConfig(
        artifact_root=args.artifact_root,
        window_root=args.artifact_root / "windows",
        output_root=args.artifact_root / "validation",
    )
    engine = V1BEngine(cfg)

    for split in R3_CONDITIONS.values():
        inspect_real_shards(engine, split, args.shards, args.samples_per_shard)

    print("\nSTAGE-A R3 SHARD-STREAMING SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
