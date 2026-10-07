# Restart progress — 2026-10-07

| Work item | Implementation | Executed / verified | Next step |
|---|---|---|---|
| Non-sharded preprocessing | Complete for current CU/1s/23-device contract | Supplied excerpt processed; 8 data/window/provenance tests pass | Preprocess full CU using a new output directory |
| DBRL -> BSE -> BIL -> EBRL runners | Implemented using shared full-model checkpoint | CLI and syntax verified; neural execution not run here | Run included neural integration check with PyTorch/PyG |
| HBF, MO, MBAI runners | Implemented with exact i -> i+1 alignment | CLI and syntax verified; neural execution not run here | Run module checks after representation stages |
| End-to-end training | Implemented; train-only optimization and validation selection | Neural integration test authored but skipped locally | Verify test, then train using real data and chosen budget |
| Validation Level 1 | Deferred; established measurements retained in original archive | No new scientific experiment | Port to learned non-sharded artifacts after protocol clarification |
| Validation Level 2 | Deferred | No new scientific experiment | Learning protocol and trained histories |
| Validation Level 3 | Deferred | No new scientific experiment | Drift definitions and trained aligned scores |
| Validation Level 4 | Deferred | No new scientific experiment | Prior evidence and prescribed framework comparisons |

Original core module bytes were preserved. New work is execution/data orchestration, checkpoint consistency, verification, commands and provenance. No manuscript results were added. No original archive, existing checkpoint or experiment output was deleted or overwritten.
