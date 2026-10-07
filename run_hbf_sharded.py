from __future__ import annotations

import argparse
import gc
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
SHARD_PATTERN = "shard_*.pt"


def parse_shard_index(path: Path) -> int:
    if path.suffix != ".pt" or not path.stem.startswith("shard_"):
        raise ValueError(f"Invalid shard filename: {path.name}")
    text = path.stem[len("shard_"):]
    if not text.isdigit():
        raise ValueError(f"Invalid shard filename: {path.name}")
    return int(text)


def discover(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"HBF upstream split directory not found: {split_dir}")
    paths = sorted(split_dir.glob(SHARD_PATTERN), key=parse_shard_index)
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

    required_common = ("split", "shard_index", "start_index", "end_index", "window_size")
    for label, payload in (("DBRL", dbrl), ("BSE", bse), ("BIL", bil), ("EBRL", ebrl)):
        missing = [k for k in required_common if k not in payload]
        if missing:
            raise KeyError(f"{label} shard missing metadata: {missing}")
        if payload["split"] != expected_split:
            raise ValueError(f"{label} split mismatch: {payload['split']} != {expected_split}")

    paths = (dbrl_path, bse_path, bil_path, ebrl_path)
    payloads = (dbrl, bse, bil, ebrl)
    indices = [int(p["shard_index"]) for p in payloads]
    filenames = [parse_shard_index(p) for p in paths]
    if len(set(indices)) != 1 or indices[0] != filenames[0] or filenames.count(filenames[0]) != 4:
        raise ValueError(f"Shard identity mismatch across upstream artifacts: {indices}, {filenames}")
    if any(int(p["shard_index"]) != filenames[0] for p in payloads):
        raise ValueError("Upstream shard indices do not agree.")

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

    # Provenance chain: BSE <- DBRL, BIL <- BSE/DBRL, EBRL <- BIL.
    if int(bse.get("source_dbrl_shard_index", -1)) != int(dbrl["shard_index"]):
        raise ValueError("BSE -> DBRL provenance mismatch.")
    if int(bil.get("source_bse_shard_index", -1)) != int(bse["shard_index"]):
        raise ValueError("BIL -> BSE provenance mismatch.")
    if int(bil.get("source_dbrl_shard_index", -1)) != int(dbrl["shard_index"]):
        raise ValueError("BIL -> DBRL provenance mismatch.")
    if int(ebrl.get("source_bil_shard_index", -1)) != int(bil["shard_index"]):
        raise ValueError("EBRL -> BIL provenance mismatch.")

    start, end = ranges[0]
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


def save_output(path: Path, outputs: dict[str, Tensor], *, upstream: dict[str, Any], source_ebrl_shard: str, seed: int, dynamics_hidden_dim: int, checkpoint_path: Path, checkpoint_sha256: str, checkpoint_epoch: int, checkpoint_metrics: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Z": outputs["Z"].cpu(),
        "S": outputs["S"].cpu(),
        "S_tilde": outputs["S_tilde"].cpu(),
        "g": outputs["g"].cpu(),
        "split": upstream["split"],
        "source_dbrl_shard": upstream.get("source_dbrl_shard"),
        "source_dbrl_shard_index": int(upstream.get("source_dbrl_shard_index", upstream["shard_index"])),
        "source_bse_shard": upstream.get("source_bse_shard"),
        "source_bse_shard_index": int(upstream.get("source_bse_shard_index", upstream["shard_index"])),
        "source_bil_shard": upstream.get("source_bil_shard"),
        "source_bil_shard_index": int(upstream.get("source_bil_shard_index", upstream["shard_index"])),
        "source_ebrl_shard": source_ebrl_shard,
        "source_ebrl_shard_index": int(upstream["shard_index"]),
        "shard_index": int(upstream["shard_index"]),
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
    
    set_seed(config["seed"])

    device = get_device()

    # # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str)
    parser.add_argument("--split", type=str)
    args = parser.parse_args()

    # override project_root_directory
    if args.project_root_dir:
        config["project_root_dir"] = args.project_root_dir

    if args.split:
        config["split"] = args.split

    split = config.get("split")
    seed = int(config["seed"])
    set_seed(seed)

    batch_size = config["hdeg"]["ebrl"].get("batch_size", 32)
    max_shards = config["hdeg"]["ebrl"].get("max_shards", None)
    overwrite = config["hdeg"]["ebrl"].get("overwrite", False)

    
    root = config["project_root_dir"]
    dataset_name = config["preprocessing"]["dataset_name"]
    hidden = int(config["hdeg"]["hbf"].get("dynamics_hidden_dim", 128))

    base = f"{root}/data/processed/{dataset_name}"

    checkpoint = config["hdeg"]["hbf"].get("checkpoint", None)

    checkpoint_path = Path(f"{root}/data/processed/{dataset_name}/hdeg_checkpoints/{checkpoint}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained HDEG checkpoint not found: {checkpoint_path}. "
            "Pass --checkpoint explicitly. A fresh HBF is no longer allowed."
        )
    checkpoint_sha256 = sha256_file(checkpoint_path)

    dbrl_dir = Path(f"{base}/dbrl/{split}")
    bse_dir = Path(f"{base}/bse/{split}")
    bil_dir = Path(f"{base}/bil/{split}")
    ebrl_dir = Path(f"{base}/ebrl/{split}")
    out_dir = Path(f"{base}/hbf/{split}")

    dbrl_paths = discover(dbrl_dir)
    bse_paths = discover(bse_dir)
    bil_paths = discover(bil_dir)
    ebrl_paths = discover(ebrl_dir)
    counts = {len(dbrl_paths), len(bse_paths), len(bil_paths), len(ebrl_paths)}
    if len(counts) != 1:
        raise RuntimeError(f"Upstream shard counts differ: {[len(dbrl_paths), len(bse_paths), len(bil_paths), len(ebrl_paths)]}")

    total = len(dbrl_paths)
    if max_shards is not None:
        if max_shards <= 0:
            raise ValueError("max_shards must be positive when provided.")
        limit = min(max_shards, total)
    else:
        limit = total

    first = validate_upstream_bundle(
        dbrl_paths[0], bse_paths[0], bil_paths[0], ebrl_paths[0],
        expected_split=split,
        expected_num_devices=None,
        expected_num_states=None,
        expected_embedding_dim=None,
    )
    Z0 = first[0]["representations"]
    N, D = int(Z0.shape[1]), int(Z0.shape[2])
    K = int(first[1]["representations"].shape[1])
    model = HierarchicalBehavioralForecaster(
        num_devices=N,
        num_states=K,
        embedding_dim=D,
        dynamics_hidden_dim=hidden,
    ).to(device)
    checkpoint_epoch, checkpoint_meta = load_hbf_checkpoint(
        checkpoint_path, model=model, device=device
    )

    print(f"HBF parameters      : {sum(p.numel() for p in model.parameters())}")
    print(f"Checkpoint          : {checkpoint_path}")
    print(f"Checkpoint SHA256    : {checkpoint_sha256}")
    print(f"Checkpoint epoch     : {checkpoint_epoch}")
    print(f"Processing split     : {split}")
    print(f"Upstream shards      : {total}")
    print(f"Batch size           : {batch_size}")
    print(f"Device               : {device}")

    expected_next_start = None
    processed_samples = 0
    for i in range(limit):
        dbrl, bse, bil, ebrl = validate_upstream_bundle(
            dbrl_paths[i], bse_paths[i], bil_paths[i], ebrl_paths[i],
            expected_split=split,
            expected_num_devices=N,
            expected_num_states=K,
            expected_embedding_dim=D,
        )
        start = int(ebrl["start_index"])
        end = int(ebrl["end_index"])
        if expected_next_start is not None and start != expected_next_start:
            raise RuntimeError(f"Gap/overlap in shard ranges: expected start {expected_next_start}, got {start}.")

        Z, S, S_tilde, g = dbrl["representations"], bse["representations"], bil["representations"], ebrl["representations"]
        outputs_cpu = {k: torch.empty_like(v, device="cpu") for k, v in {
            "Z": Z, "S": S, "S_tilde": S_tilde, "g": g
        }.items()}

        with torch.inference_mode():
            for bs in range(0, Z.shape[0], batch_size):
                be = min(bs + batch_size, Z.shape[0])
                out = run_batch(model, Z[bs:be], S[bs:be], S_tilde[bs:be], g[bs:be], device)
                for key in outputs_cpu:
                    outputs_cpu[key][bs:be].copy_(out[key].cpu())

        out_path = out_dir / ebrl_paths[i].name
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {out_path}")
        save_output(out_path, outputs_cpu, upstream=ebrl, source_ebrl_shard=ebrl_paths[i].name, seed=int(config["seed"]), dynamics_hidden_dim=hidden, checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha256, checkpoint_epoch=checkpoint_epoch, checkpoint_metrics=checkpoint_meta.get("checkpoint_metrics"))

        saved = torch.load(out_path, map_location="cpu", weights_only=False)
        for key, expected in outputs_cpu.items():
            actual = saved[key]
            if actual.shape != expected.shape or actual.dtype != torch.float32 or not torch.isfinite(actual).all():
                raise RuntimeError(f"Saved HBF artifact verification failed for {out_path}, field {key}.")
        if int(saved["start_index"]) != start or int(saved["end_index"]) != end or int(saved["shard_index"]) != i:
            raise RuntimeError(f"Saved HBF metadata verification failed for {out_path}.")

        print(f"  [PASS] shard {i:06d}: [{start}, {end}) -> {out_path}")
        processed_samples += end - start
        expected_next_start = end
        del dbrl, bse, bil, ebrl, Z, S, S_tilde, g, outputs_cpu, saved
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Processed shards     : {limit}")
    print(f"Processed samples    : {processed_samples}")
    print(f"HBF output directory : {out_dir}")
    if max_shards is None:
        print("[PASS] Complete upstream shard set processed.")

if __name__ == "__main__":
    main()
