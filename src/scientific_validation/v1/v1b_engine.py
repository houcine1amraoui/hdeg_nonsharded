"""V1-B Scientific Validation Engine.

This module computes V1 representation-quality evidence only.
It deliberately does not synthesize scientific claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import time
import gc

import numpy as np
from tqdm import tqdm

from src.scientific_validation.v1.v1b_io import (
    discover_shards,
    load_representation_shard,
    load_window_shard,
)
from src.scientific_validation.v1.v1b_metrics import (
    Reservoir,
    bootstrap_energy_ci,
    behavioral_window_distance,
    energy_distance,
    native_pair_distances,
    percentile_summary,
    spearman_rho,
    within_chunk_pair_distances,
)
from src.scientific_validation.v1.v1b_sampling import (
    condition_sampling_plan,
    deterministic_pairs,
    deterministic_sample_indices,
)


LEVELS = ("Z", "S", "S_tilde", "g")
R3_CONDITIONS = {
    "N1": "val",
    "A2": "actor2_test",
    "N2": "actor1_test",
}


@dataclass
class V1BConfig:
    artifact_root: Path
    window_root: Path
    output_root: Path
    dataset: str = "CU"
    seed: int = 42
    window_size: int = 30
    num_devices: int = 23
    representation_dim: int = 64
    r2_pair_count: int = 100_000
    r3_max_per_condition: int = 50_000
    r1_reservoir_size: int = 100_000
    r1_chunk_size: int = 512
    r3_block_length_windows: int = 30
    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 42


class V1BEngine:
    def __init__(self, cfg: V1BConfig):
        self.cfg = cfg

    def _shape_gate(self, reps: Dict[str, np.ndarray]) -> Tuple[bool, str]:
        expected_tail = {
            "Z": (23, 64),
            "S": (9, 64),
            "S_tilde": (9, 64),
            "g": (64,),
        }
        for level in LEVELS:
            if level not in reps:
                return False, f"missing_{level}"
            a = reps[level]
            if not np.isfinite(a).all():
                return False, f"nonfinite_{level}"
            if tuple(a.shape[1:]) != expected_tail[level]:
                return False, f"shape_{level}_{a.shape}"
            if a.dtype not in (np.float32, np.float64):
                return False, f"dtype_{level}_{a.dtype}"
        n = reps["g"].shape[0]
        if any(reps[l].shape[0] != n for l in LEVELS):
            return False, "cross_level_sample_count_mismatch"
        return True, "ok"

    def _build_split_index_manifest(self, split: str) -> dict:
        """Build a lightweight index manifest without retaining full shards.

        Each hierarchy level is inspected separately.  This keeps the peak
        representation tensor footprint to one level of one shard while still
        checking cross-level provenance and shapes.
        """
        print(f"Building lightweight index manifest for split={split}...")
        shard_ids = discover_shards(self.cfg.artifact_root, split)
        if not shard_ids:
            raise RuntimeError(f"No representation shards for split={split}")

        expected_tails = {
            "Z": (self.cfg.num_devices, self.cfg.representation_dim),
            "S": (9, self.cfg.representation_dim),
            "S_tilde": (9, self.cfg.representation_dim),
            "g": (self.cfg.representation_dim,),
        }
        shards = []
        expected_start = 0

        for sid in shard_ids:
            level_info = {}
            for level in LEVELS:
                reps, metadata = load_representation_shard(
                    self.cfg.artifact_root,
                    split,
                    sid,
                    return_metadata=True,
                    levels=(level,),
                )
                a = reps[level]
                if not np.isfinite(a).all():
                    raise RuntimeError(
                        f"Non-finite {level} representation in "
                        f"{split}/{sid:06d}"
                    )
                if tuple(a.shape[1:]) != expected_tails[level]:
                    raise RuntimeError(
                        f"Shape mismatch for {split}/{sid:06d}/{level}: "
                        f"{a.shape}; expected tail={expected_tails[level]}"
                    )
                if a.dtype not in (np.float32, np.float64):
                    raise RuntimeError(
                        f"Unsupported dtype for {split}/{sid:06d}/{level}: "
                        f"{a.dtype}"
                    )

                meta = metadata[level]
                required = ("split", "shard_index", "start_index", "end_index")
                missing = [k for k in required if k not in meta]
                if missing:
                    raise KeyError(
                        f"Missing provenance fields {missing} in "
                        f"{split}/{sid:06d} level={level}"
                    )

                level_info[level] = {
                    "split": str(meta["split"]),
                    "shard_index": int(meta["shard_index"]),
                    "start_index": int(meta["start_index"]),
                    "end_index": int(meta["end_index"]),
                    "sample_count": int(a.shape[0]),
                    "dtype": str(a.dtype),
                    "tail_shape": list(a.shape[1:]),
                }
                del a, reps, metadata, meta
                gc.collect()

            reference = level_info["Z"]
            if reference["split"] != split or reference["shard_index"] != sid:
                raise RuntimeError(
                    f"DBRL metadata mismatch for {split}/{sid:06d}: "
                    f"{reference}"
                )

            for level in LEVELS[1:]:
                if level_info[level]["split"] != reference["split"] or \
                   level_info[level]["shard_index"] != reference["shard_index"] or \
                   level_info[level]["start_index"] != reference["start_index"] or \
                   level_info[level]["end_index"] != reference["end_index"] or \
                   level_info[level]["sample_count"] != reference["sample_count"]:
                    raise RuntimeError(
                        f"Cross-level provenance/sample mismatch for "
                        f"{split}/{sid:06d}: Z={reference}, "
                        f"{level}={level_info[level]}"
                    )

            start_index = reference["start_index"]
            end_index = reference["end_index"]
            sample_count = end_index - start_index
            if sample_count != reference["sample_count"]:
                raise RuntimeError(
                    f"Range/sample-count mismatch for {split}/{sid:06d}: "
                    f"range_count={sample_count}, "
                    f"tensor_count={reference['sample_count']}"
                )
            if start_index != expected_start:
                raise RuntimeError(
                    f"Non-contiguous split ranges for {split}: "
                    f"expected start_index={expected_start}, "
                    f"observed={start_index} at shard {sid:06d}"
                )

            shards.append({
                "shard_index": int(sid),
                "start_index": int(start_index),
                "end_index": int(end_index),
                "sample_count": int(sample_count),
            })
            expected_start = end_index

        return {
            "split": split,
            "sample_count": int(expected_start),
            "shards": shards,
        }

    @staticmethod
    def _group_global_indices_by_shard(
        index_manifest: dict,
        global_indices: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Map global split indices to shard IDs and local row indices."""
        indices = np.asarray(global_indices, dtype=np.int64)
        grouped: dict[int, list[int]] = {}
        if indices.size == 0:
            return {}

        ends = np.asarray(
            [s["end_index"] for s in index_manifest["shards"]],
            dtype=np.int64,
        )

        shard_positions = np.searchsorted(ends, indices, side="right")
        shards = index_manifest["shards"]

        for global_index, shard_pos in zip(indices, shard_positions):
            shard_pos = int(shard_pos)
            if shard_pos >= len(shards):
                raise IndexError(
                    f"Global index {int(global_index)} is outside "
                    f"the {index_manifest['split']} split."
                )

            shard = shards[shard_pos]
            if not (
                shard["start_index"]
                <= int(global_index)
                < shard["end_index"]
            ):
                raise RuntimeError(
                    f"Index-to-shard mapping failure: global_index="
                    f"{int(global_index)}, shard={shard}"
                )

            local_index = int(global_index) - shard["start_index"]
            grouped.setdefault(shard["shard_index"], []).append(local_index)

        return {
            sid: np.asarray(local_indices, dtype=np.int64)
            for sid, local_indices in grouped.items()
        }

    def _collect_split_representation(
        self,
        split: str,
        max_samples: int | None = None,
        seed: int | None = None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """Select a bounded deterministic sample from a split shard-wise.

        The function NEVER concatenates an entire split. It first builds a
        lightweight shard/range manifest, selects global sample indices using
        only the population size, maps those indices to shards, and then loads
        only the required rows from those shards.

        For R3, callers should pass ``max_samples=50_000``. If ``max_samples``
        is omitted, this function refuses to materialize a split larger than
        the configured R3 bound.
        """
        from src.scientific_validation.v1.v1b_sampling import (
            deterministic_sample_indices,
        )

        seed = self.cfg.seed if seed is None else int(seed)
        manifest = self._build_split_index_manifest(split)
        total = int(manifest["sample_count"])

        if max_samples is None:
            if total > self.cfg.r3_max_per_condition:
                raise ValueError(
                    f"Refusing unbounded split materialization for {split}: "
                    f"{total} samples. Pass max_samples explicitly."
                )
            max_samples = total

        selected_count = min(int(max_samples), total)
        selected_global = deterministic_sample_indices(
            total,
            selected_count,
            seed=seed,
        )

        grouped = self._group_global_indices_by_shard(
            manifest,
            selected_global,
        )

        if selected_count == 0:
            raise ValueError(f"No samples selected for split={split}")

        # Allocate only the bounded selected population. The shape is obtained
        # from the first selected shard, not from the whole split.
        first_sid = next(iter(grouped))
        first_reps, _ = load_representation_shard(
            self.cfg.artifact_root,
            split,
            first_sid,
            return_metadata=True,
        )
        reps_out = {
            level: np.empty(
                (selected_count,) + first_reps[level].shape[1:],
                dtype=first_reps[level].dtype,
            )
            for level in LEVELS
        }
        del first_reps

        global_to_output = {
            int(global_index): output_pos
            for output_pos, global_index in enumerate(selected_global)
        }

        loaded_shards = []
        for sid, local_indices in grouped.items():
            reps, metadata = load_representation_shard(
                self.cfg.artifact_root,
                split,
                sid,
                return_metadata=True,
            )

            ok, reason = self._shape_gate(reps)
            if not ok:
                raise RuntimeError(
                    f"Eligibility failed for selected shard "
                    f"{split}/{sid:06d}: {reason}"
                )

            start_index = int(metadata["Z"]["start_index"])
            shard_end = int(metadata["Z"]["end_index"])
            expected_count = shard_end - start_index
            if expected_count != reps["g"].shape[0]:
                raise RuntimeError(
                    f"Range/sample-count mismatch while collecting "
                    f"{split}/{sid:06d}"
                )

            for local_index in local_indices:
                local_index = int(local_index)
                global_index = start_index + local_index
                output_pos = global_to_output[global_index]
                for level in LEVELS:
                    reps_out[level][output_pos] = reps[level][local_index]

            loaded_shards.append(int(sid))
            del reps
            del metadata

        return reps_out, {
            "split": split,
            "population_count": total,
            "selected_count": selected_count,
            "selected_global_indices": selected_global,
            "seed": seed,
            "shard_manifest": manifest["shards"],
            "selected_shards": loaded_shards,
            "sampling": {
                "method": "deterministic_without_replacement",
                "representation_blind": True,
            },
        }

    def _collect_split_representation_level(
        self,
        split: str,
        level: str,
        selected_global: np.ndarray,
        manifest: dict,
    ) -> np.ndarray:
        """Retrieve only one representation level for selected rows.

        ``selected_global`` is the deterministic split-local sample index set
        already frozen by R3.  No sampling occurs here.  Only shards containing
        selected rows are opened, and each shard is released before the next.
        """
        if level not in LEVELS:
            raise ValueError(f"Unknown representation level: {level}")

        selected_global = np.asarray(selected_global, dtype=np.int64)
        if selected_global.size == 0:
            raise ValueError(f"No selected indices for split={split}")
        if np.any(selected_global < 0) or np.any(selected_global >= manifest["sample_count"]):
            raise IndexError(f"Selected index outside split={split} population")

        grouped = self._group_global_indices_by_shard(manifest, selected_global)
        first_sid = next(iter(grouped))
        first_reps = load_representation_shard(
            self.cfg.artifact_root,
            split,
            first_sid,
            return_metadata=False,
            levels=(level,),
        )
        template = first_reps[level]
        out = np.empty(
            (len(selected_global),) + template.shape[1:],
            dtype=template.dtype,
        )
        del first_reps, template
        gc.collect()

        global_to_output = {
            int(global_index): pos
            for pos, global_index in enumerate(selected_global)
        }

        for sid, local_indices in grouped.items():
            reps = load_representation_shard(
                self.cfg.artifact_root,
                split,
                sid,
                return_metadata=False,
                levels=(level,),
            )
            a = reps[level]
            start_index = next(
                s["start_index"]
                for s in manifest["shards"]
                if s["shard_index"] == sid
            )
            for local_index in local_indices:
                local_index = int(local_index)
                global_index = start_index + local_index
                out[global_to_output[global_index]] = a[local_index]
            del a, reps
            gc.collect()

        return out

    def run_r1(self) -> dict:
        """R1 bounded streaming geometry summaries.

        Each shard is processed independently. Distances are computed only
        within bounded chunks. A deterministic reservoir prevents global
        distance materialization.
        """
        reservoirs = {
            l: Reservoir(self.cfg.r1_reservoir_size, self.cfg.seed + i)
            for i, l in enumerate(LEVELS)
        }
        shard_counts = {}
        eligible = 0

        # R1 is applied to non-training V1 populations.
        splits = ("val", "actor2_test", "actor1_test")

        for split in splits:
            for sid in discover_shards(self.cfg.artifact_root, split):
                reps = load_representation_shard(
                    self.cfg.artifact_root, split, sid
                )
                ok, reason = self._shape_gate(reps)
                if not ok:
                    shard_counts[f"{split}/{sid:06d}"] = {
                        "eligible": False,
                        "reason": reason,
                    }
                    continue

                eligible += 1
                n = reps["g"].shape[0]
                shard_counts[f"{split}/{sid:06d}"] = {
                    "eligible": True,
                    "sample_count": n,
                }

                for level in LEVELS:
                    x = reps[level]
                    for start in range(0, n, self.cfg.r1_chunk_size):
                        stop = min(start + self.cfg.r1_chunk_size, n)
                        vals = within_chunk_pair_distances(x[start:stop])
                        reservoirs[level].update(vals)

        summaries = {
            l: percentile_summary(reservoirs[l].values())
            for l in LEVELS
        }

        return {
            "measurement_family": "within_level_representation_geometry",
            "scope": "val + actor2_test + actor1_test",
            "reservoir_capacity": self.cfg.r1_reservoir_size,
            "chunk_size": self.cfg.r1_chunk_size,
            "levels": summaries,
            "eligible_shards": eligible,
            "shard_audit": shard_counts,
            "interpretation": "evidence_only_no_threshold",
        }

    def _load_r2_pairs(self):
        """Load only the observations needed by the deterministic R2 pairs.

        The R2 population is the concatenation of the established V1
        populations ``val``, ``actor2_test`` and ``actor1_test``.  The old
        implementation materialized every representation/window row from
        those splits before sampling 100,000 pairs.  This implementation
        preserves the exact pair-sampling contract while making population
        loading shard-safe:

        1. build lightweight shard/range manifests;
        2. determine the combined population size;
        3. generate the deterministic pair sample from the population size;
        4. keep only the unique pair endpoints;
        5. map endpoints to split/shard/local-row coordinates;
        6. load only required representation/window shards, one at a time;
        7. extract only the selected rows; and
        8. reconstruct pair-aligned arrays for the existing R2 metric code.

        No complete representation or behavioral-window population is
        concatenated or retained in memory.
        """
        splits = ("val", "actor2_test", "actor1_test")

        # ---------------------------------------------------------------
        # 1. Lightweight manifests.  These contain only shard ranges/counts.
        # ---------------------------------------------------------------
        manifests = {
            split: self._build_split_index_manifest(split)
            for split in splits
        }

        offsets = {}
        combined_n = 0
        for split in splits:
            offsets[split] = combined_n
            combined_n += int(manifests[split]["sample_count"])

        if combined_n < 2:
            raise RuntimeError(
                f"R2 requires at least two observations; population={combined_n}"
            )

        # ---------------------------------------------------------------
        # 2. Exact existing R2 pair sampling.  Pair selection depends only
        #    on population size/count/seed, never representation values.
        # ---------------------------------------------------------------
        pairs = deterministic_pairs(
            combined_n,
            self.cfg.r2_pair_count,
            self.cfg.seed,
        )

        # Only unique endpoints are needed from disk.  This is at most
        # 2 * r2_pair_count, not the complete population.
        endpoint_indices = np.unique(pairs.reshape(-1))
        endpoint_position = {
            int(global_index): pos
            for pos, global_index in enumerate(endpoint_indices)
        }

        # ---------------------------------------------------------------
        # 3. Resolve each selected endpoint to its source split and local
        #    split index.  The combined population order is exactly the old
        #    concatenate order: val, actor2_test, actor1_test.
        # ---------------------------------------------------------------
        split_bounds = []
        for split in splits:
            lo = offsets[split]
            hi = lo + int(manifests[split]["sample_count"])
            split_bounds.append((lo, hi, split))

        def resolve_combined_index(global_index: int):
            for lo, hi, split in split_bounds:
                if lo <= global_index < hi:
                    return split, global_index - lo
            raise IndexError(
                f"Combined R2 index {global_index} outside [0,{combined_n})"
            )

        # Requests are grouped by split and then by shard.  Each entry is a
        # local row index in that shard.
        requests_by_split = {split: [] for split in splits}
        for global_index in endpoint_indices:
            split, local_split_index = resolve_combined_index(int(global_index))
            requests_by_split[split].append(
                (int(global_index), int(local_split_index))
            )

        shard_requests = {}
        for split in splits:
            local_indices = np.asarray(
                [local for _, local in requests_by_split[split]],
                dtype=np.int64,
            )
            if local_indices.size:
                shard_requests[split] = self._group_global_indices_by_shard(
                    manifests[split], local_indices
                )
            else:
                shard_requests[split] = {}

        # ---------------------------------------------------------------
        # 4. Determine the persisted behavioral-window key and output shape
        #    from one required shard.  No full split is loaded.
        # ---------------------------------------------------------------
        first_request = None
        for split in splits:
            if shard_requests[split]:
                first_request = (split, next(iter(shard_requests[split])))
                break
        if first_request is None:
            raise RuntimeError("R2 pair sampling produced no endpoints.")

        first_split, first_sid = first_request
        first_window_obj = load_window_shard(
            self.cfg.window_root, first_split, first_sid
        )
        window_key = None
        expected_window_count = next(
            s["sample_count"]
            for s in manifests[first_split]["shards"]
            if s["shard_index"] == first_sid
        )
        for key in ("X", "windows"):
            if key in first_window_obj:
                candidate = np.asarray(first_window_obj[key])
                if candidate.ndim >= 2 and candidate.shape[0] == expected_window_count:
                    window_key = key
                    break
        if window_key is None:
            raise KeyError(
                f"No X/windows behavioral-window array with expected sample "
                f"count={expected_window_count} in {first_split}/{first_sid:06d}"
            )

        template_window = np.asarray(first_window_obj[window_key])
        window_dtype = template_window.dtype
        window_tail = tuple(template_window.shape[1:])
        expected_tail = (self.cfg.window_size, self.cfg.num_devices)
        if window_tail != expected_tail:
            raise RuntimeError(
                f"Unexpected behavioral-window shape tail={window_tail}; "
                f"expected={expected_tail}"
            )
        del first_window_obj, template_window

        # ---------------------------------------------------------------
        # 5. Allocate only the unique endpoint population.  At most 2P rows
        #    are retained, where P is the configured R2 pair count.
        # ---------------------------------------------------------------
        endpoint_count = len(endpoint_indices)
        endpoint_X = np.empty(
            (endpoint_count,) + window_tail,
            dtype=window_dtype,
        )
        endpoint_reps = {}

        # Determine representation tails from the first required shard.
        first_reps, _ = load_representation_shard(
            self.cfg.artifact_root,
            first_split,
            first_sid,
            return_metadata=True,
        )
        for level in LEVELS:
            endpoint_reps[level] = np.empty(
                (endpoint_count,) + first_reps[level].shape[1:],
                dtype=first_reps[level].dtype,
            )
        del first_reps

        # ---------------------------------------------------------------
        # 6. Stream over required shards only.  Each shard is released before
        #    moving to the next one.
        # ---------------------------------------------------------------
        selected_shards = []
        for split in splits:
            manifest_by_sid = {
                item["shard_index"]: item
                for item in manifests[split]["shards"]
            }

            for sid, local_indices in shard_requests[split].items():
                reps, metadata = load_representation_shard(
                    self.cfg.artifact_root,
                    split,
                    sid,
                    return_metadata=True,
                )

                ok, reason = self._shape_gate(reps)
                if not ok:
                    raise RuntimeError(
                        f"Eligibility failed for R2 selected shard "
                        f"{split}/{sid:06d}: {reason}"
                    )

                shard_meta = metadata["Z"]
                start_index = int(shard_meta["start_index"])
                end_index = int(shard_meta["end_index"])
                shard_n = int(manifest_by_sid[sid]["sample_count"])

                if end_index - start_index != shard_n:
                    raise RuntimeError(
                        f"Representation range mismatch for {split}/{sid:06d}"
                    )

                window_obj = load_window_shard(
                    self.cfg.window_root, split, sid
                )
                if window_key not in window_obj:
                    raise KeyError(
                        f"Behavioral window key {window_key!r} missing from "
                        f"{split}/{sid:06d}"
                    )
                windows = np.asarray(window_obj[window_key])
                if windows.shape[0] != shard_n:
                    raise RuntimeError(
                        f"Window sample-count mismatch for {split}/{sid:06d}: "
                        f"{windows.shape[0]} != {shard_n}"
                    )

                for local_index in local_indices:
                    local_index = int(local_index)
                    split_global = start_index + local_index
                    combined_global = offsets[split] + split_global
                    out_pos = endpoint_position[combined_global]

                    endpoint_X[out_pos] = windows[local_index]
                    for level in LEVELS:
                        endpoint_reps[level][out_pos] = reps[level][local_index]

                selected_shards.append({
                    "split": split,
                    "shard_index": int(sid),
                    "selected_rows": int(len(local_indices)),
                })

                del reps, metadata, window_obj, windows

        # ---------------------------------------------------------------
        # 7. Remap the sampled global pair endpoints into the compact
        #    endpoint store.  This keeps run_r2's existing interface while
        #    avoiding a second copy of every pair's tensors.
        # ---------------------------------------------------------------
        left_pos = np.fromiter(
            (endpoint_position[int(i)] for i in pairs[:, 0]),
            dtype=np.int64,
            count=len(pairs),
        )
        right_pos = np.fromiter(
            (endpoint_position[int(j)] for j in pairs[:, 1]),
            dtype=np.int64,
            count=len(pairs),
        )
        compact_pairs = np.column_stack((left_pos, right_pos))

        # Endpoint provenance is bounded by the number of unique pair
        # endpoints (<= 2 * pair_count), not by the complete population.
        provenance = []
        for global_index in endpoint_indices:
            split, local_index = resolve_combined_index(int(global_index))
            provenance.append(
                (split, self._shard_for_local_index(manifests[split], local_index))
            )

        pair_meta = {
            "population_count": int(combined_n),
            "pair_count": int(len(pairs)),
            "unique_endpoint_count": int(endpoint_count),
            "seed": int(self.cfg.seed),
            "splits": list(splits),
            "split_population_counts": {
                split: int(manifests[split]["sample_count"])
                for split in splits
            },
            "split_offsets": {split: int(offsets[split]) for split in splits},
            "sampling": {
                "method": "deterministic_unordered_pairs",
                "representation_blind": True,
                "without_replacement": True,
            },
            "selected_shards": selected_shards,
        }

        # endpoint_X / endpoint_reps are intentionally returned rather than
        # pair-expanded arrays.  run_r2 indexes them using compact_pairs, so
        # memory remains proportional to unique endpoints rather than twice
        # the selected pair count for every tensor.
        return endpoint_X, endpoint_reps, compact_pairs, provenance

    @staticmethod
    def _shard_for_local_index(manifest: dict, local_index: int) -> int:
        """Return the shard containing one split-local sample index."""
        ends = np.asarray(
            [s["end_index"] for s in manifest["shards"]],
            dtype=np.int64,
        )
        pos = int(np.searchsorted(ends, int(local_index), side="right"))
        if pos >= len(manifest["shards"]):
            raise IndexError(
                f"Local index {local_index} outside split {manifest['split']}"
            )
        shard = manifest["shards"][pos]
        if not (shard["start_index"] <= local_index < shard["end_index"]):
            raise RuntimeError(
                f"Local index {local_index} does not belong to shard {shard}"
            )
        return int(shard["shard_index"])

    def run_r2(self) -> dict:
        X, reps, pairs, provenance = self._load_r2_pairs()

        bx = np.empty(len(pairs), dtype=np.float64)
        dists = {l: np.empty(len(pairs), dtype=np.float64) for l in LEVELS}
        strata = {}

        for k, (i, j) in tqdm(enumerate(pairs), total=len(pairs)):
            bx[k] = behavioral_window_distance(X[i], X[j])
            for l in LEVELS:
                dists[l][k] = native_pair_distances(
                    reps[l][i:i+1], reps[l][j:j+1]
                )[0]

            si, sj = provenance[int(i)][0], provenance[int(j)][0]
            key = f"{si}-{sj}"
            strata[key] = strata.get(key, 0) + 1

        return {
            "measurement_family": "behavioral_relationship_preservation",
            "pair_count": int(len(pairs)),
            "seed": self.cfg.seed,
            "pair_sampling": "deterministic_representation_blind_without_replacement",
            "behavior_distance": {
                "definition": "frobenius_norm / (W*N)",
                "W": self.cfg.window_size,
                "N": self.cfg.num_devices,
            },
            "representation_distance": "within_native_level_euclidean",
            "spearman": {
                l: spearman_rho(bx, dists[l]) for l in LEVELS
            },
            "pair_strata": strata,
        }

    def run_r3(self) -> dict:
        """Execute R3 with comparison- and level-streamed memory scheduling.

        Scientific semantics are unchanged:
        - the three fixed conditions remain N1/A2/N2;
        - each condition gets one deterministic representation-blind sample;
        - the same sampled indices are reused across all hierarchy levels;
        - the three authorized contrasts are unchanged;
        - the existing ``bootstrap_energy_ci`` metric implementation is used
          unchanged.

        Only the execution schedule changes: at most two conditions and one
        representation level are resident for a comparison.
        """
        condition_order = tuple(R3_CONDITIONS.keys())
        sampling_seeds = {
            condition: self.cfg.seed + i
            for i, condition in enumerate(condition_order)
        }

        # Build manifests once. They contain only ranges/counts.
        manifests = {
            condition: self._build_split_index_manifest(split)
            for condition, split in R3_CONDITIONS.items()
        }

        # Freeze the R3 sample indices ONCE. These exact indices are reused for
        # every level and every comparison; no level is allowed to resample.
        sampling_plan = {}
        populations = {}
        for condition in condition_order:
            n = int(manifests[condition]["sample_count"])
            selected_count = min(self.cfg.r3_max_per_condition, n)
            indices = deterministic_sample_indices(
                n,
                selected_count,
                seed=sampling_seeds[condition],
            )
            sampling_plan[condition] = indices
            populations[condition] = int(selected_count)

        comparisons = [
            ("N1", "A2"),
            ("A2", "N2"),
            ("N1", "N2"),
        ]

        result = {
            "measurement_family": "energy_distance",
            "sampling": {
                "max_per_condition": self.cfg.r3_max_per_condition,
                "without_replacement": True,
                "representation_blind": True,
                "seed": self.cfg.seed,
                "condition_seeds": sampling_seeds,
                "sampling_reused_across_levels": True,
            },
            "conditions": {
                "N1": "val",
                "A2": "actor2_test",
                "N2": "actor1_test",
            },
            "sample_counts": populations,
            "comparisons": {},
            "uncertainty": {
                "method": "moving_block_bootstrap",
                "block_length_windows": self.cfg.r3_block_length_windows,
                "bootstrap_replicates": self.cfg.bootstrap_replicates,
                "confidence_level": 0.95,
                "interval_method": "percentile",
            },
            "execution": {
                "memory_schedule": "comparison_streamed_level_by_level",
                "max_conditions_resident": 2,
                "max_levels_resident": 1,
                "full_split_materialization": False,
            },
        }

        for a, b in comparisons:
            key = f"{a}-{b}"
            result["comparisons"][key] = {}

            # Process one hierarchy level at a time. This preserves the same
            # sample indices while preventing Z/S/S_tilde/g from co-residing.
            for level in LEVELS:
                x = self._collect_split_representation_level(
                    R3_CONDITIONS[a],
                    level,
                    sampling_plan[a],
                    manifests[a],
                )
                y = self._collect_split_representation_level(
                    R3_CONDITIONS[b],
                    level,
                    sampling_plan[b],
                    manifests[b],
                )

                try:
                    ci = bootstrap_energy_ci(
                        x.reshape(len(x), -1),
                        y.reshape(len(y), -1),
                        block_length=self.cfg.r3_block_length_windows,
                        replicates=self.cfg.bootstrap_replicates,
                        seed=self.cfg.bootstrap_seed,
                    )
                    result["comparisons"][key][level] = ci
                finally:
                    # Do not retain the large selected arrays after the scalar
                    # evidence object has been produced.
                    del x, y
                    gc.collect()

        # Sampling indices are no longer needed after all comparisons have
        # completed. Keep only compact reproducibility fingerprints in memory.
        result["sampling"]["selected_index_sha256"] = {}
        import hashlib
        for condition in condition_order:
            digest = hashlib.sha256(
                np.asarray(sampling_plan[condition], dtype=np.int64).tobytes()
            ).hexdigest()
            result["sampling"]["selected_index_sha256"][condition] = digest

        del sampling_plan, manifests
        gc.collect()
        return result

    def run(self) -> dict:
        started = time.time()

        evidence = {
            "manifest": {
                "engine": "HDEG V1-B Scientific Validation Engine",
                "version": "1.0",
                "dataset": self.cfg.dataset,
                "seed": self.cfg.seed,
                "window_size": self.cfg.window_size,
                "num_devices": self.cfg.num_devices,
                "representation_dim": self.cfg.representation_dim,
                "r2_pair_count": self.cfg.r2_pair_count,
                "r3_max_per_condition": self.cfg.r3_max_per_condition,
                "r3_block_length_windows": self.cfg.r3_block_length_windows,
                "claim_decision": "not_performed",
            },
            # "r1": self.run_r1(),
            # "r2": self.run_r2(),
            "r3": self.run_r3(),
            "provenance": {
                "execution_started_unix": started,
                "execution_finished_unix": time.time(),
            },
        }
        return evidence
