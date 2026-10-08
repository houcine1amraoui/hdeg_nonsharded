from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor
import yaml

from src.models.hdeg.hbf import HierarchicalBehavioralForecaster

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.get_folders_utils import get_processed_folder

SPLITS = ("train", "val", "actor2_test", "actor1_test")


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dictionary payload.")
    return payload


def _validate_tensor(name: str, tensor: Any, ndim: int, path: Path) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} in {path} must be a tensor.")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} in {path} must have ndim={ndim}; got {tensor.ndim}.")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} in {path} must be float32; got {tensor.dtype}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} in {path} contains NaN/Inf.")


def validate_upstream_bundle(
    dbrl_path: Path,
    bse_path: Path,
    bil_path: Path,
    ebrl_path: Path,
    *,
    expected_split: str,
    expected_num_devices: Optional[int],
    expected_num_states: Optional[int],
    expected_embedding_dim: Optional[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    dbrl = load_payload(dbrl_path)
    bse = load_payload(bse_path)
    bil = load_payload(bil_path)
    ebrl = load_payload(ebrl_path)

    required_common = ("representations", "split", "start_index", "end_index", "window_size", "source_windows")
    for label, payload in (("DBRL", dbrl), ("BSE", bse), ("BIL", bil), ("EBRL", ebrl)):
        missing = [k for k in required_common if k not in payload]
        if missing:
            raise KeyError(f"{label} file missing metadata: {missing}")
        if payload["split"] != expected_split:
            raise ValueError(f"{label} split mismatch: {payload['split']} != {expected_split}")

    payloads = (dbrl, bse, bil, ebrl)
    ranges = [(int(p["start_index"]), int(p["end_index"])) for p in payloads]
    if len(set(ranges)) != 1:
        raise ValueError(f"Sample ranges differ across upstream artifacts: {ranges}")

    Z = dbrl["representations"]
    S = bse["representations"]
    S_tilde = bil["representations"]
    g = ebrl["representations"]
    _validate_tensor("DBRL representations", Z, 3, dbrl_path)
    _validate_tensor("BSE representations", S, 3, bse_path)
    _validate_tensor("BIL representations", S_tilde, 3, bil_path)
    _validate_tensor("EBRL representations", g, 2, ebrl_path)

    if not (Z.shape[0] == S.shape[0] == S_tilde.shape[0] == g.shape[0]):
        raise ValueError("Upstream sample counts do not agree.")
    if expected_num_devices is not None and Z.shape[1] != expected_num_devices:
        raise ValueError(f"Expected N={expected_num_devices}; got {Z.shape[1]}.")
    if expected_num_states is not None and S.shape[1] != expected_num_states:
        raise ValueError(f"Expected K={expected_num_states}; got {S.shape[1]}.")
    if S.shape[1] != S_tilde.shape[1]:
        raise ValueError("BSE and BIL state counts differ.")
    D = Z.shape[2]
    if not (S.shape[2] == S_tilde.shape[2] == g.shape[1] == D):
        raise ValueError("Upstream embedding dimensions do not agree.")
    if expected_embedding_dim is not None and D != expected_embedding_dim:
        raise ValueError(f"Expected D={expected_embedding_dim}; got {D}.")

    # Check the file provenance chain across the four split artifacts.
    for label, payload, key, expected_path in (
        ("BSE", bse, "source_dbrl", dbrl_path),
        ("BIL", bil, "source_dbrl", dbrl_path),
        ("BIL", bil, "source_bse", bse_path),
        ("EBRL", ebrl, "source_dbrl", dbrl_path),
        ("EBRL", ebrl, "source_bse", bse_path),
        ("EBRL", ebrl, "source_bil", bil_path),
    ):
        if key not in payload or Path(payload[key]).resolve() != expected_path.resolve():
            raise ValueError(f"{label} provenance mismatch for {key}.")
    if len({Path(p["source_windows"]).resolve() for p in payloads}) != 1:
        raise ValueError("Upstream source_windows paths differ.")
    if len({int(p["window_size"]) for p in payloads}) != 1:
        raise ValueError("Upstream window sizes differ.")

    start, end = ranges[0]
    if start != 0 or end <= start:
        raise ValueError("Upstream split files must have a nonempty range starting at zero.")
    if end - start != Z.shape[0]:
        raise ValueError("Sample range does not match artifact sample count.")

    return dbrl, bse, bil, ebrl


def run_batch(model, Z, S, S_tilde, g, device):
    return model(
        Z.to(device),
        S.to(device),
        S_tilde.to(device),
        g.to(device),
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_hbf_checkpoint(
    checkpoint_path: Path,
    *,
    model: HierarchicalBehavioralForecaster,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} must contain a dictionary payload.")

    checkpoint_epoch = int(payload.get("epoch", -1))
    metadata = {
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_metrics": payload.get("metrics"),
    }

    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
        if not isinstance(state, dict):
            raise TypeError("checkpoint model_state_dict must be a mapping.")
        # E2E checkpoints contain the complete HDEG model. HBF parameters are
        # stored under the `hbf.` namespace. Load exactly that frozen HBF
        # submodule rather than silently constructing a fresh forecaster.
        hbf_state = {
            key[len("hbf."):]: value
            for key, value in state.items()
            if key.startswith("hbf.")
        }
        if not hbf_state:
            raise KeyError(
                f"Checkpoint {checkpoint_path} contains model_state_dict but no 'hbf.' parameters."
            )
    else:
        # Also accept an explicitly HBF-only state_dict for portability.
        state = payload
        if not all(isinstance(k, str) for k in state.keys()):
            raise TypeError("HBF-only checkpoint state_dict keys must be strings.")
        hbf_state = state

    missing, unexpected = model.load_state_dict(hbf_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"HBF checkpoint incompatibility: missing={missing}, unexpected={unexpected}"
        )

    model.eval()
    return checkpoint_epoch, metadata


def save_output(path: Path, outputs: dict[str, Tensor], *, upstream: dict[str, Any], source_ebrl: str, seed: int, dynamics_hidden_dim: int, checkpoint_path: Path, checkpoint_sha256: str, checkpoint_epoch: int, checkpoint_metrics: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Z": outputs["Z"].cpu(),
        "S": outputs["S"].cpu(),
        "S_tilde": outputs["S_tilde"].cpu(),
        "g": outputs["g"].cpu(),
        "split": upstream["split"],
        "source_dbrl": upstream["source_dbrl"],
        "source_bse": upstream["source_bse"],
        "source_bil": upstream["source_bil"],
        "source_ebrl": source_ebrl,
        "source_windows": upstream["source_windows"],
        "start_index": int(upstream["start_index"]),
        "end_index": int(upstream["end_index"]),
        "window_size": int(upstream["window_size"]),
        "num_devices": int(outputs["Z"].shape[1]),
        "num_states": int(outputs["S"].shape[1]),
        "embedding_dim": int(outputs["g"].shape[1]),
        "dynamics_hidden_dim": int(dynamics_hidden_dim),
        "seed": int(seed),
        "model_checkpoint": str(checkpoint_path),
        "model_checkpoint_sha256": checkpoint_sha256,
        "model_checkpoint_epoch": int(checkpoint_epoch),
        "model_checkpoint_metrics": checkpoint_metrics,
        "model_source": "train_hdeg_e2e.py checkpoint / hbf submodule",
    }
    torch.save(payload, path)


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
    # Retain the configuration section used by the supplied HBF runner.
    batch_size = int(config["hdeg"]["ebrl"].get("batch_size", 32))
    overwrite = bool(config["hdeg"]["ebrl"].get("overwrite", False))
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    hidden = int(config["hdeg"]["hbf"].get("dynamics_hidden_dim", 128))

    base = Path(get_processed_folder(config))
    checkpoint = config["hdeg"]["hbf"].get("checkpoint")
    if not checkpoint:
        raise ValueError("Set hdeg.hbf.checkpoint to the trained checkpoint filename.")
    checkpoint_path = base / "hdeg_checkpoints" / checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained HDEG checkpoint not found: {checkpoint_path}. "
            "Set hdeg.hbf.checkpoint in configs/config.yaml."
        )
    checkpoint_sha256 = sha256_file(checkpoint_path)

    dbrl_path = base / "dbrl" / f"{split}.pt"
    bse_path = base / "bse" / f"{split}.pt"
    bil_path = base / "bil" / f"{split}.pt"
    ebrl_path = base / "ebrl" / f"{split}.pt"
    out_path = base / "hbf" / f"{split}.pt"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_path}")

    dbrl, bse, bil, ebrl = validate_upstream_bundle(
        dbrl_path, bse_path, bil_path, ebrl_path,
        expected_split=split,
        expected_num_devices=None,
        expected_num_states=None,
        expected_embedding_dim=None,
    )
    Z, S, S_tilde, g = dbrl["representations"], bse["representations"], bil["representations"], ebrl["representations"]
    N, D = int(Z.shape[1]), int(Z.shape[2])
    K = int(S.shape[1])
    start, end = int(ebrl["start_index"]), int(ebrl["end_index"])
    model = HierarchicalBehavioralForecaster(
        num_devices=N,
        num_states=K,
        embedding_dim=D,
        dynamics_hidden_dim=hidden,
    ).to(device)
    checkpoint_epoch, checkpoint_meta = load_hbf_checkpoint(
        checkpoint_path, model=model, device=device
    )

    print("HDEG — HBF Non-sharded Standalone Execution")
    print(f"HBF parameters      : {sum(p.numel() for p in model.parameters())}")
    print(f"Checkpoint          : {checkpoint_path}")
    print(f"Checkpoint SHA256   : {checkpoint_sha256}")
    print(f"Checkpoint epoch    : {checkpoint_epoch}")
    print(f"Processing split    : {split}")
    print(f"Batch size          : {batch_size}")
    print(f"Device              : {device}")

    outputs_cpu = {k: torch.empty_like(v, device="cpu") for k, v in {
        "Z": Z, "S": S, "S_tilde": S_tilde, "g": g
    }.items()}

    with torch.inference_mode():
        for bs in range(0, Z.shape[0], batch_size):
            be = min(bs + batch_size, Z.shape[0])
            out = run_batch(model, Z[bs:be], S[bs:be], S_tilde[bs:be], g[bs:be], device)
            for key in outputs_cpu:
                outputs_cpu[key][bs:be].copy_(out[key].cpu())

    save_output(
        out_path, outputs_cpu, upstream=ebrl, source_ebrl=str(ebrl_path),
        seed=seed, dynamics_hidden_dim=hidden,
        checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_metrics=checkpoint_meta.get("checkpoint_metrics"),
    )

    saved = torch.load(out_path, map_location="cpu", weights_only=False)
    for key, expected in outputs_cpu.items():
        actual = saved[key]
        if actual.shape != expected.shape or actual.dtype != torch.float32 or not torch.isfinite(actual).all():
            raise RuntimeError(f"Saved HBF artifact verification failed for {out_path}, field {key}.")
    if int(saved["start_index"]) != start or int(saved["end_index"]) != end or saved["split"] != split:
        raise RuntimeError(f"Saved HBF metadata verification failed for {out_path}.")

    print(f"Processed samples: {end - start}")
    print(f"[PASS] HBF output verified and saved to: {out_path}")
    print("[PASS] All samples in the upstream split files processed exactly once.")


if __name__ == "__main__":
    main()
