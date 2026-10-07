# Stage A — R3 comparison/level streamed execution

## Scope

This patch changes only the **execution schedule** of V1-B R3. It does not
change the R3 proposition, condition mapping, sample cap, sampling seeds,
Energy-distance call, bootstrap method, block length, number of bootstrap
replicates, or confidence interval definition.

## Frozen sampling semantics

The R3 condition order remains:

- N1 -> `val`
- A2 -> `actor2_test`
- N2 -> `actor1_test`

Sampling is performed once per condition using the existing deterministic
without-replacement sampler and the existing condition-specific seeds:

- N1: `seed`
- A2: `seed + 1`
- N2: `seed + 2`

The selected indices are reused unchanged for all four representation levels.
Only compact SHA-256 fingerprints of the selected index arrays are retained in
final evidence.

## Memory schedule

The previous R3 execution retained all three conditions and all four levels.
The new execution is:

1. build lightweight manifests once;
2. freeze the three condition sampling-index arrays;
3. for each authorized comparison (`N1-A2`, `A2-N2`, `N1-N2`):
   - for each level (`Z`, `S`, `S_tilde`, `g`):
     - retrieve only that level and only selected rows for condition A;
     - retrieve only that level and only selected rows for condition B;
     - call the existing `bootstrap_energy_ci()` unchanged;
     - delete both selected arrays and force garbage collection;
4. retain only scalar evidence and compact sampling fingerprints.

Thus the intended resident representation bound is:

- at most **2 conditions**;
- at most **1 representation level**;
- never a complete split;
- never all four levels simultaneously.

## Level-selective artifact loading

`load_representation_shard()` now accepts:

```python
levels=("Z",)
```

or any subset of the four frozen levels. Existing callers that omit `levels`
continue to receive all four levels.

The manifest builder also inspects levels one at a time rather than loading all
four representations of a shard simultaneously.

## Tests

### Scheduler test

`test_stageA_r3_schedule.py` verifies:

- exactly 3 comparisons × 4 levels = 12 measurement calls;
- each call receives only one level;
- at most two condition arrays are supplied to the metric call;
- the same sampling plan is reused across levels;
- the three authorized comparisons remain unchanged.

This is a scheduling test, not scientific evidence.

### Real-artifact smoke test

`stageA_smoke_test.py` is intended to be executed in the user's HDEG/Colab
environment against the actual CU `val`, `actor2_test`, and `actor1_test`
shards. It checks the first N real shards and exercises the level-selective
retrieval path without running the 50,000-sample bootstrap.

Example:

```bash
python stageA_smoke_test.py \
  --artifact-root /content/drive/MyDrive/02_research/hdeg/code/data/processed/CU \
  --shards 3 \
  --samples-per-shard 128
```

## Local verification performed

The supplied real CU `shard_000000` artifact package was inspected. Its four
levels were successfully loaded one level at a time with shapes:

- Z: `(8192, 23, 64)`
- S: `(8192, 9, 64)`
- S_tilde: `(8192, 9, 64)`
- g: `(8192, 64)`

The level-streaming collector successfully extracted 128 rows for each level,
with finite float32 outputs. The one-shard manifest also validated successfully.

The scheduler unit test passed.

## Not changed

- R3 Energy-distance definition/call
- moving-block bootstrap definition
- block length
- bootstrap replicate count
- confidence interval method
- condition mapping
- sample cap
- sampling seeds
- 200-shard generated artifacts
- Results Synthesis
- Scientific Claim Synthesis
