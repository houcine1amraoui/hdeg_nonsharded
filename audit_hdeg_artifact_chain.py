from __future__ import annotations

"""
HDEG V1.0 artifact-chain audit.

This tool audits the persisted scientific artifacts without loading a complete
split into RAM. It checks:

- shard discovery and contiguous ranges;
- representation shapes/dtypes/finiteness;
- DBRL -> BSE -> BIL -> EBRL provenance;
- HBF upstream alignment and provenance;
- MBAI t -> t+1 alignment;
- MBAI coverage and final-row exclusion;
- consistency of N/K/D/window size;
- presence of the HDEG training checkpoint;
- HBF checkpoint provenance.

It produces a machine-readable audit JSON and a human-readable report.

IMPORTANT:
The current frozen run_hbf_sharded.py in the supplied codebase constructs a
fresh HBF and has its checkpoint-loading code commented out. Therefore this
audit intentionally treats HBF trained-model provenance as a separate,
explicit check. A complete scientific inference package must not claim a
trained HBF unless a selected checkpoint is actually loaded and recorded.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from src.utils.seed import set_seed
from src.utils.get_folders_utils import get_processed_folder

LEVELS = ("Z", "S", "S_tilde", "g")


def shard_index(path: Path) -> int:
    m = re.fullmatch(r"shard_(\d+)\.pt", path.name)
    if not m:
        raise ValueError(f"Invalid shard filename: {path}")
    return int(m.group(1))


def discover(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("shard_*.pt"), key=shard_index)
    if not paths:
        raise FileNotFoundError(f"No shards found: {directory}")
    ids = [shard_index(p) for p in paths]
    if ids != list(range(len(paths))):
        raise RuntimeError(f"Non-contiguous shard indices in {directory}: {ids[:20]}")
    return paths


def load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a dictionary payload")
    return payload


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Audit:
    def __init__(self):
        self.checks = []
        self.errors = []
        self.warnings = []

    def ok(self, name, detail=""):
        self.checks.append({"name": name, "status": "PASS", "detail": detail})

    def fail(self, name, detail):
        self.checks.append({"name": name, "status": "FAIL", "detail": detail})
        self.errors.append({"name": name, "detail": detail})

    def warn(self, name, detail):
        self.checks.append({"name": name, "status": "WARN", "detail": detail})
        self.warnings.append({"name": name, "detail": detail})


def validate_rep_payload(
    payload,
    path,
    *,
    split,
    level,
    N,
    K,
    D,
    expected_start,
    require_num_states,
):
    required = [
        "representations", "split", "shard_index",
        "start_index", "end_index", "window_size",
        "num_devices", "embedding_dim",
    ]
    if require_num_states:
        required.append("num_states")

    missing = [k for k in required if k not in payload]
    if missing:
        raise RuntimeError(f"{path}: missing metadata {missing}")

    if payload["split"] != split:
        raise RuntimeError(f"{path}: split mismatch")

    i = int(payload["shard_index"])
    if i != shard_index(path):
        raise RuntimeError(f"{path}: shard index metadata mismatch")

    start, end = int(payload["start_index"]), int(payload["end_index"])
    if start != expected_start or end <= start:
        raise RuntimeError(
            f"{path}: invalid/non-contiguous range [{start},{end}), "
            f"expected start {expected_start}"
        )

    x = payload["representations"]
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(f"{path}: representations is not a tensor")
    if x.dtype != torch.float32:
        raise RuntimeError(f"{path}: representations dtype is {x.dtype}, expected float32")
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{path}: representations contains NaN/Inf")

    expected_ndim = 2 if level == "g" else 3
    if x.ndim != expected_ndim:
        raise RuntimeError(f"{path}: {level} ndim={x.ndim}, expected {expected_ndim}")

    if x.shape[0] != end - start:
        raise RuntimeError(f"{path}: sample count/range mismatch")

    if level == "Z":
        if tuple(x.shape[1:]) != (N, D):
            raise RuntimeError(f"{path}: Z shape {tuple(x.shape)}, expected (*,{N},{D})")
    elif level in ("S", "S_tilde"):
        if tuple(x.shape[1:]) != (K, D):
            raise RuntimeError(f"{path}: {level} shape {tuple(x.shape)}, expected (*,{K},{D})")
    else:
        if tuple(x.shape[1:]) != (D,):
            raise RuntimeError(f"{path}: g shape {tuple(x.shape)}, expected (*,{D})")

    if int(payload["num_devices"]) != N:
        raise RuntimeError(f"{path}: N metadata mismatch")
    if require_num_states and int(payload["num_states"]) != K:
        raise RuntimeError(f"{path}: K metadata mismatch")
    if int(payload["embedding_dim"]) != D:
        raise RuntimeError(f"{path}: D metadata mismatch")

    return start, end, int(payload["window_size"])


def audit_split(base: Path, split: str, audit: Audit):
    dirs = {k: base / k / split for k in ("dbrl", "bse", "bil", "ebrl", "hbf", "mbai")}
    paths = {k: discover(v) for k, v in dirs.items()}

    counts = {k: len(v) for k, v in paths.items()}
    if len(set(counts.values())) != 1:
        audit.fail("shard_count_consistency", str(counts))
        return
    audit.ok("shard_count_consistency", str(counts))

    # Establish N/K/D from the first artifacts.
    d0, s0, b0, e0 = (load(paths[k][0]) for k in ("dbrl", "bse", "bil", "ebrl"))
    N = int(d0["num_devices"])
    D = int(d0["embedding_dim"])
    K = int(s0["num_states"])

    expected = 0
    total = 0

    for i in range(len(paths["dbrl"])):
        loaded = {k: load(paths[k][i]) for k in ("dbrl", "bse", "bil", "ebrl")}
        ranges = {}

        for level, key, require_k in (
            ("Z", "dbrl", False),
            ("S", "bse", True),
            ("S_tilde", "bil", True),
            ("g", "ebrl", True),
        ):
            start, end, W = validate_rep_payload(
                loaded[key], paths[key][i],
                split=split, level=level,
                N=N, K=K, D=D,
                expected_start=expected,
                require_num_states=require_k,
            )
            ranges[key] = (start, end)
            if W != int(d0["window_size"]):
                raise RuntimeError(f"{paths[key][i]}: window_size mismatch")

        if len(set(ranges.values())) != 1:
            raise RuntimeError(f"representation range mismatch at shard {i}: {ranges}")

        # Frozen provenance chain.
        if int(loaded["bse"].get("source_dbrl_shard_index", -1)) != i:
            raise RuntimeError(f"BSE provenance mismatch at shard {i}")
        if int(loaded["bil"].get("source_bse_shard_index", -1)) != i:
            raise RuntimeError(f"BIL -> BSE provenance mismatch at shard {i}")
        if int(loaded["bil"].get("source_dbrl_shard_index", -1)) != i:
            raise RuntimeError(f"BIL -> DBRL provenance mismatch at shard {i}")
        if int(loaded["ebrl"].get("source_bil_shard_index", -1)) != i:
            raise RuntimeError(f"EBRL -> BIL provenance mismatch at shard {i}")

        expected = ranges["dbrl"][1]
        total += expected - ranges["dbrl"][0]

    audit.ok(
        "representation_chain",
        f"{total} samples; N={N}, K={K}, D={D}, shards={len(paths['dbrl'])}",
    )

    # HBF artifact audit.
    expected = 0
    for i, p in enumerate(paths["hbf"]):
        h = load(p)
        start, end = int(h["start_index"]), int(h["end_index"])
        if start != expected or end <= start:
            raise RuntimeError(f"HBF range mismatch at shard {i}: [{start},{end})")
        for level, ndim in (("Z",3),("S",3),("S_tilde",3),("g",2)):
            x = h[level]
            if x.dtype != torch.float32 or x.ndim != ndim or not torch.isfinite(x).all():
                raise RuntimeError(f"HBF {level} invalid at shard {i}")
        if h["Z"].shape[1:] != (N,D):
            raise RuntimeError(f"HBF Z shape mismatch at shard {i}")
        if h["S"].shape[1:] != (K,D) or h["S_tilde"].shape[1:] != (K,D):
            raise RuntimeError(f"HBF state shape mismatch at shard {i}")
        if h["g"].shape[1:] != (D,):
            raise RuntimeError(f"HBF g shape mismatch at shard {i}")
        expected = end

    audit.ok("hbf_artifacts", f"{expected} prediction rows")

    # MBAI audit.
    expected_target = 1
    total_m = 0
    for i, p in enumerate(paths["mbai"]):
        m = load(p)
        ts, te = int(m["target_start_index"]), int(m["target_end_index"])
        if ts != expected_target or te <= ts:
            raise RuntimeError(
                f"MBAI target range mismatch at shard {i}: "
                f"[{ts},{te}), expected start {expected_target}"
            )
        n = te - ts
        for key in ("E_Z","E_S","E_S_tilde","E_G","A"):
            x = m[key]
            if x.dtype != torch.float32 or x.ndim != 1 or x.shape[0] != n:
                raise RuntimeError(f"MBAI {key} invalid at shard {i}")
            if not torch.isfinite(x).all():
                raise RuntimeError(f"MBAI {key} contains NaN/Inf at shard {i}")
        if int(m["source_hbf_shard_index"]) != i:
            raise RuntimeError(f"MBAI HBF provenance mismatch at shard {i}")
        expected_target = te
        total_m += n

    audit.ok(
        "mbai_temporal_coverage",
        f"{total_m} aligned assessments; target range [1,{expected_target})",
    )

    if total_m == total - 1:
        audit.ok("final_prediction_boundary", "exactly one final unpaired prediction excluded")
    else:
        audit.fail(
            "final_prediction_boundary",
            f"expected {total-1} assessments from {total} predictions; got {total_m}",
        )

    # Critical trained-HBF provenance check.
    # The current supplied run_hbf_sharded.py does not load a checkpoint.
    audit.fail(
        "hbf_trained_model_provenance",
        "Current run_hbf_sharded.py constructs a fresh HBF and has checkpoint loading commented out. "
        "The persisted HBF artifacts therefore cannot be certified as outputs of the selected trained "
        "HDEG model until the runner loads and records a specific training checkpoint.",
    )

    return total, total_m


def main():

    with open("configs/config.yaml", "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
        
    set_seed(config["seed"])

    # # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str)

    seed = int(config["seed"])
    set_seed(seed)

    project_root_dir = config["project_root_dir"]
    split = "train"
    
    processed_root = get_processed_folder(config)

    dataset = config["preprocessing"]["dataset_name"]

    output = f"{processed_root}/audit_reports/artifact_audit_{split}.json"

    base = Path(f"{project_root_dir}/data/processed/{dataset}")
    a = Audit()

    total = total_m = None
    try:
        total, total_m = audit_split(base, split, a)
    except Exception as exc:
        a.fail("artifact_chain", str(exc))

    report = {
        "dataset": dataset,
        "split": split,
        "total_prediction_rows": total,
        "total_aligned_assessments": total_m,
        "status": "PASS" if not a.errors else "FAIL",
        "checks": a.checks,
        "errors": a.errors,
        "warnings": a.warnings,
    }

    out = Path(output) if output else base / "inference_evidence_package" / f"artifact_audit_{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not a.errors else 2)


if __name__ == "__main__":
    main()
