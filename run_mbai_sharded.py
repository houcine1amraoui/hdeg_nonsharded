from __future__ import annotations

"""
Sharded MBAI inference for the frozen HDEG V1.0 pipeline.

Temporal contract
-----------------
HBF shard i contains predictions for R_hat[t+1] generated from R_t, with
sample range [start, end).

The corresponding observed target hierarchy is R[t+1], therefore its global
sample range is [start+1, end+1).

For ordinary rows, the target representation is the next row in the persisted
hierarchical representation sequence. At a shard boundary, the final
prediction row of the current HBF shard uses only the first row of the next
observed representation shard.

The final HBF prediction of the complete persisted X-window sequence has no
persisted R[t+1] artifact in the current representation artifact set. It is
therefore excluded rather than fabricated. A complete split consequently
produces M-1 MBAI assessments from M persisted window representations.

Observed hierarchy source
--------------------------
The current frozen artifact architecture stores the four representation
levels separately:

    DBRL -> Z
    BSE  -> S
    BIL  -> S_tilde
    EBRL -> g

The runner combines these four corresponding persisted shards into the
observed hierarchy required by MBAI. It does not reconstruct any upstream
module and does not concatenate the complete dataset.

Memory discipline
-----------------
Only one HBF shard and one observed representation bundle are held as primary
inputs at a time. At a shard boundary, the current observed bundle is released
before the next observed bundle is loaded. No global representation tensor is
constructed.
"""

import argparse
import gc
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from src.models.hdeg.mbai import MultiScaleBehavioralAnomalyInference
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.utils.get_folders_utils import get_processed_folder

SPLITS = ("train", "val", "actor2_test", "actor1_test")
LEVELS = ("Z", "S", "S_tilde", "g")


def parse_shard_index(path: Path) -> int:
    if path.suffix != ".pt" or not path.stem.startswith("shard_"):
        raise ValueError(f"Invalid shard filename: {path.name}")
    text = path.stem[len("shard_"):]
    if not text.isdigit():
        raise ValueError(f"Invalid shard filename: {path.name}")
    return int(text)


def discover(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    paths = sorted(split_dir.glob("shard_*.pt"), key=parse_shard_index)
    if not paths:
        raise FileNotFoundError(f"No shards found in {split_dir}")
    indices = [parse_shard_index(p) for p in paths]
    if indices != list(range(len(paths))):
        raise RuntimeError(
            f"Shard sequence is not contiguous zero-based: {indices[:10]}"
        )
    return paths


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dictionary payload.")
    return payload


def validate_float_tensor(
    tensor: Any,
    *,
    name: str,
    path: Path,
    ndim: int,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} in {path} must be a torch.Tensor.")
    if tensor.ndim != ndim:
        raise ValueError(
            f"{name} in {path} must have ndim={ndim}; got {tensor.ndim}."
        )
    if tensor.dtype != torch.float32:
        raise TypeError(
            f"{name} in {path} must be float32; got {tensor.dtype}."
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} in {path} contains NaN/Inf.")


def validate_common_metadata(
    payload: Mapping[str, Any],
    *,
    path: Path,
    expected_split: str,
    require_num_states: bool = True,
) -> tuple[int, int]:
    # DBRL is a device-level representation and therefore its frozen
    # artifact contract does NOT contain num_states. BSE/BIL/EBRL do.
    required = [
        "split",
        "shard_index",
        "start_index",
        "end_index",
        "window_size",
        "num_devices",
        "embedding_dim",
    ]
    if require_num_states:
        required.append("num_states")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{path} is missing metadata: {missing}")

    if payload["split"] != expected_split:
        raise ValueError(
            f"{path} split mismatch: {payload['split']} != {expected_split}"
        )

    index = int(payload["shard_index"])
    filename_index = parse_shard_index(path)
    if index != filename_index:
        raise ValueError(
            f"{path} metadata shard_index={index} disagrees with filename."
        )

    start = int(payload["start_index"])
    end = int(payload["end_index"])
    if start < 0 or end <= start:
        raise ValueError(f"Invalid sample range [{start}, {end}) in {path}.")

    return start, end


def validate_hbf_payload(
    payload: Mapping[str, Any],
    *,
    path: Path,
    expected_split: str,
    expected_num_devices: int | None,
    expected_num_states: int | None,
    expected_embedding_dim: int | None,
) -> tuple[int, int]:
    start, end = validate_common_metadata(
        payload, path=path, expected_split=expected_split
    )

    for level, ndim in (("Z", 3), ("S", 3), ("S_tilde", 3), ("g", 2)):
        validate_float_tensor(
            payload.get(level),
            name=f"HBF {level}",
            path=path,
            ndim=ndim,
        )

    Z = payload["Z"]
    S = payload["S"]
    S_tilde = payload["S_tilde"]
    g = payload["g"]

    if not (Z.shape[0] == S.shape[0] == S_tilde.shape[0] == g.shape[0]):
        raise ValueError(f"HBF sample-count mismatch in {path}.")
    if end - start != Z.shape[0]:
        raise ValueError(
            f"HBF sample range [{start}, {end}) does not match "
            f"sample count {Z.shape[0]} in {path}."
        )

    if expected_num_devices is not None and Z.shape[1] != expected_num_devices:
        raise ValueError(
            f"HBF N mismatch: expected {expected_num_devices}, got {Z.shape[1]}."
        )
    if expected_num_states is not None and S.shape[1] != expected_num_states:
        raise ValueError(
            f"HBF K mismatch: expected {expected_num_states}, got {S.shape[1]}."
        )
    if S.shape[1] != S_tilde.shape[1]:
        raise ValueError(f"HBF S/S_tilde state-count mismatch in {path}.")

    D = Z.shape[2]
    if not (S.shape[2] == S_tilde.shape[2] == g.shape[1] == D):
        raise ValueError(f"HBF embedding-dimension mismatch in {path}.")
    if expected_embedding_dim is not None and D != expected_embedding_dim:
        raise ValueError(
            f"HBF D mismatch: expected {expected_embedding_dim}, got {D}."
        )

    return start, end


def validate_observed_bundle(
    dbrl: Mapping[str, Any],
    bse: Mapping[str, Any],
    bil: Mapping[str, Any],
    ebrl: Mapping[str, Any],
    *,
    dbrl_path: Path,
    bse_path: Path,
    bil_path: Path,
    ebrl_path: Path,
    expected_split: str,
    expected_num_devices: int,
    expected_num_states: int,
    expected_embedding_dim: int,
) -> tuple[int, int]:
    payloads = (
        ("DBRL", dbrl, dbrl_path),
        ("BSE", bse, bse_path),
        ("BIL", bil, bil_path),
        ("EBRL", ebrl, ebrl_path),
    )
    ranges = []

    for label, payload, path in payloads:
        start, end = validate_common_metadata(
            payload,
            path=path,
            expected_split=expected_split,
            require_num_states=(label != "DBRL"),
        )
        ranges.append((start, end))

    if len(set(ranges)) != 1:
        raise ValueError(f"Observed hierarchy shard ranges differ: {ranges}")

    representation_specs = (
        ("Z", dbrl["representations"], dbrl_path, 3),
        ("S", bse["representations"], bse_path, 3),
        ("S_tilde", bil["representations"], bil_path, 3),
        ("g", ebrl["representations"], ebrl_path, 2),
    )

    for level, tensor, path, ndim in representation_specs:
        validate_float_tensor(
            tensor,
            name=f"Observed {level}",
            path=path,
            ndim=ndim,
        )

    Z = dbrl["representations"]
    S = bse["representations"]
    S_tilde = bil["representations"]
    g = ebrl["representations"]

    expected_count = ranges[0][1] - ranges[0][0]
    if not (
        Z.shape[0]
        == S.shape[0]
        == S_tilde.shape[0]
        == g.shape[0]
        == expected_count
    ):
        raise ValueError("Observed hierarchy sample/range mismatch.")

    if Z.shape[1] != expected_num_devices:
        raise ValueError(
            f"Observed N mismatch: {Z.shape[1]} != {expected_num_devices}."
        )
    if S.shape[1] != expected_num_states:
        raise ValueError("Observed S K mismatch.")
    if S_tilde.shape[1] != expected_num_states:
        raise ValueError("Observed S_tilde K mismatch.")

    if not (
        Z.shape[2]
        == S.shape[2]
        == S_tilde.shape[2]
        == g.shape[1]
        == expected_embedding_dim
    ):
        raise ValueError("Observed D mismatch.")

    # Preserve the frozen upstream provenance chain.
    if int(bse.get("source_dbrl_shard_index", -1)) != int(dbrl["shard_index"]):
        raise ValueError("BSE -> DBRL provenance mismatch.")
    if int(bil.get("source_bse_shard_index", -1)) != int(bse["shard_index"]):
        raise ValueError("BIL -> BSE provenance mismatch.")
    if int(bil.get("source_dbrl_shard_index", -1)) != int(dbrl["shard_index"]):
        raise ValueError("BIL -> DBRL provenance mismatch.")
    if int(ebrl.get("source_bil_shard_index", -1)) != int(bil["shard_index"]):
        raise ValueError("EBRL -> BIL provenance mismatch.")

    return ranges[0]


def load_observed_bundle(
    dbrl_path: Path,
    bse_path: Path,
    bil_path: Path,
    ebrl_path: Path,
    *,
    expected_split: str,
    expected_num_devices: int,
    expected_num_states: int,
    expected_embedding_dim: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    dbrl = load_payload(dbrl_path)
    bse = load_payload(bse_path)
    bil = load_payload(bil_path)
    ebrl = load_payload(ebrl_path)

    validate_observed_bundle(
        dbrl,
        bse,
        bil,
        ebrl,
        dbrl_path=dbrl_path,
        bse_path=bse_path,
        bil_path=bil_path,
        ebrl_path=ebrl_path,
        expected_split=expected_split,
        expected_num_devices=expected_num_devices,
        expected_num_states=expected_num_states,
        expected_embedding_dim=expected_embedding_dim,
    )
    return dbrl, bse, bil, ebrl


def build_output_payload(
    *,
    evidence: dict[str, torch.Tensor],
    hbf_payload: Mapping[str, Any],
    target_start: int,
    target_end: int,
    seed: int,
    fusion_weights: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "E_Z": evidence["E_Z"].cpu(),
        "E_S": evidence["E_S"].cpu(),
        "E_S_tilde": evidence["E_S_tilde"].cpu(),
        "E_G": evidence["E_G"].cpu(),
        "A": evidence["A"].cpu(),
        "split": hbf_payload["split"],
        "source_hbf_shard": hbf_payload.get("_source_path"),
        "source_hbf_shard_index": int(hbf_payload["shard_index"]),
        "source_hbf_start_index": int(hbf_payload["start_index"]),
        "source_hbf_end_index": int(hbf_payload["end_index"]),
        "target_start_index": int(target_start),
        "target_end_index": int(target_end),
        "window_size": int(hbf_payload["window_size"]),
        "num_devices": int(hbf_payload["num_devices"]),
        "num_states": int(hbf_payload["num_states"]),
        "embedding_dim": int(hbf_payload["embedding_dim"]),
        "fusion_weights": {
            key: float(fusion_weights[key]) for key in LEVELS
        },
        "seed": int(seed),
        "temporal_alignment": (
            "prediction row t compared with observed hierarchy row t+1"
        ),
    }


def verify_mba_output_artifact(
    path: Path,
    *,
    expected_count: int,
    expected_start: int,
    expected_end: int,
) -> None:
    payload = load_payload(path)

    for key in ("E_Z", "E_S", "E_S_tilde", "E_G", "A"):
        tensor = payload.get(key)
        validate_float_tensor(
            tensor,
            name=key,
            path=path,
            ndim=1,
        )
        if tensor.shape[0] != expected_count:
            raise RuntimeError(
                f"{path} field {key} has {tensor.shape[0]} samples; "
                f"expected {expected_count}."
            )

    if int(payload["target_start_index"]) != expected_start:
        raise RuntimeError("MBAI target start metadata mismatch.")
    if int(payload["target_end_index"]) != expected_end:
        raise RuntimeError("MBAI target end metadata mismatch.")
    if expected_end - expected_start != expected_count:
        raise RuntimeError("MBAI output range/count mismatch.")


def run_mbai_sharded(
    *,
    hbf_dir: Path,
    dbrl_dir: Path,
    bse_dir: Path,
    bil_dir: Path,
    ebrl_dir: Path,
    output_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    fusion_weights: Mapping[str, float] | None = None,
    max_shards: int | None = None,
    overwrite: bool = False,
    seed: int = 0,
) -> tuple[int, int]:
    hbf_paths = discover(hbf_dir)
    dbrl_paths = discover(dbrl_dir)
    bse_paths = discover(bse_dir)
    bil_paths = discover(bil_dir)
    ebrl_paths = discover(ebrl_dir)

    counts = {
        len(hbf_paths),
        len(dbrl_paths),
        len(bse_paths),
        len(bil_paths),
        len(ebrl_paths),
    }
    if len(counts) != 1:
        raise RuntimeError(
            "MBAI upstream shard counts differ: "
            f"HBF={len(hbf_paths)}, DBRL={len(dbrl_paths)}, "
            f"BSE={len(bse_paths)}, BIL={len(bil_paths)}, "
            f"EBRL={len(ebrl_paths)}."
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    limit = (
        len(hbf_paths)
        if max_shards is None
        else min(max_shards, len(hbf_paths))
    )
    if limit <= 0:
        raise ValueError("No shards selected.")

    # Read one observed bundle to establish the fixed CU hierarchy dimensions.
    first_dbrl = load_payload(dbrl_paths[0])
    first_bse = load_payload(bse_paths[0])
    num_devices = int(first_dbrl["num_devices"])
    num_states = int(first_bse["num_states"])
    embedding_dim = int(first_dbrl["embedding_dim"])
    del first_dbrl, first_bse
    gc.collect()

    model = MultiScaleBehavioralAnomalyInference(
        fusion_weights=fusion_weights
    ).to(device)
    model.eval()

    weight_dict = {
        level: float(model.fusion_weights[i].item())
        for i, level in enumerate(LEVELS)
    }

    expected_start = 0
    total_predictions = 0
    total_assessments = 0

    for shard_pos in range(limit):
        hbf_path = hbf_paths[shard_pos]
        hbf_payload = load_payload(hbf_path)

        hbf_start, hbf_end = validate_hbf_payload(
            hbf_payload,
            path=hbf_path,
            expected_split=split,
            expected_num_devices=num_devices,
            expected_num_states=num_states,
            expected_embedding_dim=embedding_dim,
        )

        if hbf_start != expected_start:
            raise RuntimeError(
                f"Gap/overlap in HBF ranges: "
                f"expected {expected_start}, got {hbf_start}."
            )
        if int(hbf_payload["shard_index"]) != shard_pos:
            raise RuntimeError(
                "HBF shard index does not match processing order."
            )

        # ---------------------------------------------------------
        # Part 1: current-shard temporal alignment.
        #
        # Prediction local [0:n-1] corresponds to observed local [1:n].
        # The final prediction row is handled separately at the boundary.
        # ---------------------------------------------------------
        dbrl, bse, bil, ebrl = load_observed_bundle(
            dbrl_paths[shard_pos],
            bse_paths[shard_pos],
            bil_paths[shard_pos],
            ebrl_paths[shard_pos],
            expected_split=split,
            expected_num_devices=num_devices,
            expected_num_states=num_states,
            expected_embedding_dim=embedding_dim,
        )

        obs_start = int(dbrl["start_index"])
        obs_end = int(dbrl["end_index"])

        if (obs_start, obs_end) != (hbf_start, hbf_end):
            raise RuntimeError(
                f"HBF/observed range mismatch at shard {shard_pos}: "
                f"HBF=[{hbf_start},{hbf_end}), "
                f"observed=[{obs_start},{obs_end})."
            )

        shard_count = hbf_end - hbf_start
        if shard_count <= 0:
            raise RuntimeError("Empty HBF shard.")

        internal_count = max(0, shard_count - 1)

        outputs_cpu = {
            "E_Z": torch.empty(shard_count, dtype=torch.float32),
            "E_S": torch.empty(shard_count, dtype=torch.float32),
            "E_S_tilde": torch.empty(shard_count, dtype=torch.float32),
            "E_G": torch.empty(shard_count, dtype=torch.float32),
            "A": torch.empty(shard_count, dtype=torch.float32),
        }

        if internal_count > 0:
            observed_internal = {
                "Z": dbrl["representations"][1:shard_count],
                "S": bse["representations"][1:shard_count],
                "S_tilde": bil["representations"][1:shard_count],
                "g": ebrl["representations"][1:shard_count],
            }
            predicted_internal = {
                level: hbf_payload[level][:internal_count]
                for level in LEVELS
            }

            with torch.inference_mode():
                for bs in range(0, internal_count, batch_size):
                    be = min(bs + batch_size, internal_count)

                    observed_batch = {
                        level: observed_internal[level][bs:be].to(device)
                        for level in LEVELS
                    }
                    predicted_batch = {
                        level: predicted_internal[level][bs:be].to(device)
                        for level in LEVELS
                    }

                    result = model(observed_batch, predicted_batch)

                    for key in outputs_cpu:
                        outputs_cpu[key][bs:be] = result[key].cpu()

            del observed_internal, predicted_internal

        # Release current observed representations before loading the next
        # shard. This keeps the boundary operation bounded.
        del dbrl, bse, bil, ebrl
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ---------------------------------------------------------
        # Part 2: shard-boundary row.
        #
        # Every non-final shard has one prediction whose t+1 target is
        # the first row of the next observed representation shard.
        #
        # The final persisted prediction has no persisted R[t+1] target.
        # It is deliberately excluded instead of fabricated.
        # ---------------------------------------------------------
        boundary_count = 0

        if shard_pos < len(hbf_paths) - 1 and shard_pos < limit - 1:
            ndbrl, nbse, nbil, nebrl = load_observed_bundle(
                dbrl_paths[shard_pos + 1],
                bse_paths[shard_pos + 1],
                bil_paths[shard_pos + 1],
                ebrl_paths[shard_pos + 1],
                expected_split=split,
                expected_num_devices=num_devices,
                expected_num_states=num_states,
                expected_embedding_dim=embedding_dim,
            )

            next_start = int(ndbrl["start_index"])
            if next_start != hbf_end:
                raise RuntimeError(
                    f"Shard boundary mismatch: HBF ends at {hbf_end}, "
                    f"next observed shard starts at {next_start}."
                )

            observed_boundary = {
                "Z": ndbrl["representations"][:1].to(device),
                "S": nbse["representations"][:1].to(device),
                "S_tilde": nbil["representations"][:1].to(device),
                "g": nebrl["representations"][:1].to(device),
            }
            predicted_boundary = {
                level: hbf_payload[level][-1:].to(device)
                for level in LEVELS
            }

            with torch.inference_mode():
                result = model(
                    observed_boundary,
                    predicted_boundary,
                )

            for key in outputs_cpu:
                outputs_cpu[key][-1] = result[key][0].cpu()

            boundary_count = 1

            del (
                ndbrl,
                nbse,
                nbil,
                nebrl,
                observed_boundary,
                predicted_boundary,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            # The final persisted prediction has no observed t+1 hierarchy.
            # Remove it rather than manufacturing a target.
            outputs_cpu = {
                key: value[:internal_count]
                for key, value in outputs_cpu.items()
            }

        usable = int(outputs_cpu["A"].shape[0])

        if usable == 0:
            print(
                f"[INFO] shard {shard_pos:06d}: no aligned MBAI samples; "
                "no artifact written."
            )
        else:
            target_start = hbf_start + 1
            target_end = target_start + usable

            output_path = output_dir / hbf_path.name
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"Output exists: {output_path}")

            payload = build_output_payload(
                evidence=outputs_cpu,
                hbf_payload={
                    **hbf_payload,
                    "_source_path": str(hbf_path),
                },
                target_start=target_start,
                target_end=target_end,
                seed=seed,
                fusion_weights=weight_dict,
            )

            payload.update(
                {
                    "source_dbrl_shard": str(dbrl_paths[shard_pos]),
                    "source_dbrl_shard_index": shard_pos,
                    "source_bse_shard": str(bse_paths[shard_pos]),
                    "source_bse_shard_index": shard_pos,
                    "source_bil_shard": str(bil_paths[shard_pos]),
                    "source_bil_shard_index": shard_pos,
                    "source_ebrl_shard": str(ebrl_paths[shard_pos]),
                    "source_ebrl_shard_index": shard_pos,
                    "shard_index": shard_pos,
                    "start_index": target_start,
                    "end_index": target_end,
                    "num_samples": usable,
                    "boundary_target_source": (
                        "next observed representation shard first row"
                        if boundary_count
                        else "current observed representation shard next row"
                    ),
                    "final_prediction_excluded": (
                        shard_pos == len(hbf_paths) - 1
                    ),
                }
            )

            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, output_path)

            verify_mba_output_artifact(
                output_path,
                expected_count=usable,
                expected_start=target_start,
                expected_end=target_end,
            )

            print(
                f"[PASS] shard {shard_pos:06d}: "
                f"prediction [{hbf_start},{hbf_end}) -> "
                f"observed target [{target_start},{target_end}) -> "
                f"{output_path}"
            )

        total_predictions += shard_count
        total_assessments += usable
        expected_start = hbf_end

        del hbf_payload, outputs_cpu
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return total_predictions, total_assessments


def parse_weights(text: str | None) -> dict[str, float] | None:
    if text is None:
        return None

    result: dict[str, float] = {}
    for item in text.split(","):
        key, value = item.split("=", 1)
        key = key.strip()

        if key not in LEVELS:
            raise ValueError(f"Unknown MBAI weight key: {key}")

        result[key] = float(value)

    return result


def main() -> None:

    with open("configs/config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
    
    set_seed(config["seed"])

    device = get_device()

    # # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str)
    args = parser.parse_args()

    seed = int(config["seed"])
    set_seed(seed)

    batch_size = config["hdeg"]["mbai"].get("batch_size", 32)
    max_shards = config["hdeg"]["mbai"].get("max_shards", None)
    overwrite = config["hdeg"]["mbai"].get("overwrite", False)
    fusion_weights = config["hdeg"]["mbai"].get("fusion_weights", None)

    root = config["project_root_dir"]
    split = "train"

    processed_root = get_processed_folder(config)

    hbf_dir = Path(f"{processed_root}/hbf/{split}")
    output_dir = Path(f"{processed_root}/mbai/{split}")

    dataset = config["preprocessing"]["dataset_name"]
    base = f"{root}/data/processed/{dataset}"

    dbrl_dir = Path(f"{base}/dbrl/{split}")
    bse_dir = Path(f"{base}/bse/{split}")
    bil_dir = Path(f"{base}/bil/{split}")
    ebrl_dir = Path(f"{base}/ebrl/{split}")

    weights = parse_weights(fusion_weights)

    print(f"MBAI split           : {split}")
    print(f"HBF directory        : {hbf_dir}")
    print("Observed hierarchy   : DBRL + BSE + BIL + EBRL shards")
    print(f"Output directory     : {output_dir}")
    print(f"Batch size           : {batch_size}")
    print(f"Device               : {device}")
    print(
        "Temporal contract    : "
        "prediction at t -> observed hierarchy at t+1"
    )
    print(
        "Final persisted-window prediction is excluded because its "
        "observed R[t+1] hierarchy is not persisted in the current "
        "X_t representation artifacts."
    )

    total_predictions, total_assessments = run_mbai_sharded(
        hbf_dir=hbf_dir,
        dbrl_dir=dbrl_dir,
        bse_dir=bse_dir,
        bil_dir=bil_dir,
        ebrl_dir=ebrl_dir,
        output_dir=output_dir,
        split=split,
        batch_size=batch_size,
        device=device,
        fusion_weights=weights,
        max_shards=max_shards,
        overwrite=overwrite,
        seed=seed,
    )

    print("=" * 70)
    print("MBAI sharded inference completed")
    print("=" * 70)
    print(f"Prediction rows      : {total_predictions}")
    print(f"Aligned assessments  : {total_assessments}")
    print(f"Output directory     : {output_dir}")

if __name__ == "__main__":
    main()
