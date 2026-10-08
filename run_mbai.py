from __future__ import annotations

"""
Non-sharded MBAI inference for the frozen HDEG V1.0 pipeline.

Compare HBF prediction row t with observed hierarchy row t+1. Each split
contains M persisted X-window representations; its final prediction has no
persisted target hierarchy and is excluded, producing M-1 assessments.

Load one file each for HBF, DBRL, BSE, BIL, and EBRL on CPU. Only minibatches
are moved to the execution device. Save one MBAI file per split.
"""

import argparse
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

    start = int(payload["start_index"])
    end = int(payload["end_index"])
    if start != 0 or end <= start:
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
        raise ValueError(f"Observed hierarchy file ranges differ: {ranges}")

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

    # Preserve upstream provenance using the non-sharded file paths.
    for label, payload, key, expected_path in (
        ("BSE", bse, "source_dbrl", dbrl_path),
        ("BIL", bil, "source_bse", bse_path),
        ("BIL", bil, "source_dbrl", dbrl_path),
        ("EBRL", ebrl, "source_bil", bil_path),
        ("EBRL", ebrl, "source_bse", bse_path),
        ("EBRL", ebrl, "source_dbrl", dbrl_path),
    ):
        if key not in payload or Path(payload[key]).resolve() != expected_path.resolve():
            raise ValueError(f"{label} provenance mismatch for {key}.")
    if len({Path(p["source_windows"]).resolve() for _, p, _ in payloads}) != 1:
        raise ValueError("Observed source_windows paths differ.")
    if len({int(p["window_size"]) for _, p, _ in payloads}) != 1:
        raise ValueError("Observed window sizes differ.")

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
        "source_hbf": hbf_payload.get("_source_path"),
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


def run_mbai(
    *,
    hbf_path: Path,
    dbrl_path: Path,
    bse_path: Path,
    bil_path: Path,
    ebrl_path: Path,
    output_path: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    fusion_weights: Mapping[str, float] | None = None,
    overwrite: bool = False,
    seed: int = 0,
) -> tuple[int, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}")

    hbf_payload = load_payload(hbf_path)
    hbf_start, hbf_end = validate_hbf_payload(
        hbf_payload, path=hbf_path, expected_split=split,
        expected_num_devices=None, expected_num_states=None,
        expected_embedding_dim=None,
    )
    num_devices = int(hbf_payload["num_devices"])
    num_states = int(hbf_payload["num_states"])
    embedding_dim = int(hbf_payload["embedding_dim"])
    for level, expected_shape in {
        "Z": (hbf_end, num_devices, embedding_dim),
        "S": (hbf_end, num_states, embedding_dim),
        "S_tilde": (hbf_end, num_states, embedding_dim),
        "g": (hbf_end, embedding_dim),
    }.items():
        if tuple(hbf_payload[level].shape) != expected_shape:
            raise ValueError(f"HBF {level} shape disagrees with its metadata.")

    dbrl, bse, bil, ebrl = load_observed_bundle(
        dbrl_path, bse_path, bil_path, ebrl_path,
        expected_split=split,
        expected_num_devices=num_devices,
        expected_num_states=num_states,
        expected_embedding_dim=embedding_dim,
    )
    if (int(dbrl["start_index"]), int(dbrl["end_index"])) != (hbf_start, hbf_end):
        raise RuntimeError("HBF and observed split sample ranges differ.")
    for key, expected_path in {
        "source_dbrl": dbrl_path, "source_bse": bse_path,
        "source_bil": bil_path, "source_ebrl": ebrl_path,
    }.items():
        if key not in hbf_payload or Path(hbf_payload[key]).resolve() != expected_path.resolve():
            raise ValueError(f"HBF provenance mismatch for {key}.")
    if Path(hbf_payload["source_windows"]).resolve() != Path(dbrl["source_windows"]).resolve():
        raise ValueError("HBF and observed source_windows paths differ.")
    if int(hbf_payload["window_size"]) != int(dbrl["window_size"]):
        raise ValueError("HBF and observed window sizes differ.")

    model = MultiScaleBehavioralAnomalyInference(
        fusion_weights=fusion_weights
    ).to(device)
    model.eval()
    weight_dict = {
        level: float(model.fusion_weights[i].item())
        for i, level in enumerate(LEVELS)
    }

    num_samples = hbf_end - hbf_start
    usable = num_samples - 1
    if usable == 0:
        print("[INFO] No aligned MBAI samples; no artifact written.")
        return num_samples, 0
    outputs_cpu = {
        key: torch.empty(usable, dtype=torch.float32)
        for key in ("E_Z", "E_S", "E_S_tilde", "E_G", "A")
    }
    observed_internal = {
        "Z": dbrl["representations"][1:num_samples],
        "S": bse["representations"][1:num_samples],
        "S_tilde": bil["representations"][1:num_samples],
        "g": ebrl["representations"][1:num_samples],
    }
    predicted_internal = {
        level: hbf_payload[level][:usable]
        for level in LEVELS
    }

    with torch.inference_mode():
        for bs in range(0, usable, batch_size):
            be = min(bs + batch_size, usable)

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


    target_start, target_end = hbf_start + 1, hbf_end
    payload = build_output_payload(
        evidence=outputs_cpu,
        hbf_payload={**hbf_payload, "_source_path": str(hbf_path)},
        target_start=target_start, target_end=target_end,
        seed=seed, fusion_weights=weight_dict,
    )
    payload.update({
        "source_dbrl": str(dbrl_path),
        "source_bse": str(bse_path),
        "source_bil": str(bil_path),
        "source_ebrl": str(ebrl_path),
        "source_windows": dbrl["source_windows"],
        "start_index": target_start,
        "end_index": target_end,
        "num_samples": usable,
        "final_prediction_excluded": True,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    verify_mba_output_artifact(
        output_path, expected_count=usable,
        expected_start=target_start, expected_end=target_end,
    )
    print(f"[PASS] Observed target [{target_start},{target_end}) -> {output_path}")
    return num_samples, usable


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str)
    parser.add_argument("--split", choices=SPLITS)
    args = parser.parse_args()
    if args.project_root_dir:
        config["project_root_dir"] = args.project_root_dir
    split = args.split or config.get("split", "train")
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")

    seed = int(config["seed"])
    set_seed(seed)
    device = get_device()
    mbai_config = config["hdeg"]["mbai"]
    batch_size = int(mbai_config.get("batch_size", 32))
    overwrite = bool(mbai_config.get("overwrite", False))
    weights = parse_weights(mbai_config.get("fusion_weights"))
    base = Path(get_processed_folder(config))
    output_path = base / "mbai" / f"{split}.pt"

    print(f"MBAI split          : {split}")
    print(f"Output file         : {output_path}")
    print(f"Batch size          : {batch_size}")
    print(f"Device              : {device}")
    print("Temporal contract   : prediction row t -> observed row t+1")
    print("Final prediction excluded: its observed target is not persisted.")

    predictions, assessments = run_mbai(
        hbf_path=base / "hbf" / f"{split}.pt",
        dbrl_path=base / "dbrl" / f"{split}.pt",
        bse_path=base / "bse" / f"{split}.pt",
        bil_path=base / "bil" / f"{split}.pt",
        ebrl_path=base / "ebrl" / f"{split}.pt",
        output_path=output_path, split=split,
        batch_size=batch_size, device=device, fusion_weights=weights,
        overwrite=overwrite, seed=seed,
    )
    print("MBAI non-sharded inference completed")
    print(f"Prediction rows     : {predictions}")
    print(f"Aligned assessments : {assessments}")


if __name__ == "__main__":
    main()
