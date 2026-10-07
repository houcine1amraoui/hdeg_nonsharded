# MBAI Temporal Alignment + Sharded Inference

## Frozen contract

For an HBF prediction generated from `R_t`:

`R_t -> HBF -> R_hat_(t+1)`

MBAI must compare that prediction against the observed hierarchy:

`R_(t+1)`.

The current persisted representation artifacts correspond to the `X_t`
window sequence. Therefore a prediction shard with global range `[a,b)`
uses observed representation rows `[a+1,b+1)`.

## Boundary handling

For a non-final shard:

- rows `[a,b-1)` use rows `[a+1,b)` from the same DBRL/BSE/BIL/EBRL shard;
- the final prediction row `b-1` uses row `b` from the first row of the next observed representation shard.

The current observed shard is released before the next observed shard is loaded.

## Final sample

The final persisted `X_t` representation has no persisted `R_(t+1)` target in
the current artifact architecture. The runner therefore excludes that final
prediction rather than fabricating a target.

For `M` persisted representations, full-split MBAI therefore emits `M-1`
aligned assessments.

## Observed hierarchy

MBAI requires all four hierarchy levels:

- `Z` from DBRL;
- `S` from BSE;
- `S_tilde` from BIL;
- `g` from EBRL.

The runner validates the frozen provenance chain and sample ranges before
inference.

## Output

Each output shard contains:

- `E_Z`
- `E_S`
- `E_S_tilde`
- `E_G`
- `A`

plus temporal alignment and upstream provenance metadata.

No MBAI parameters or training objective are introduced.
