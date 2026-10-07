from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models.hdeg.dbrl import DBRL
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.get_folders_utils import get_processed_folder


SPLITS = (
    "train",
    "val",
    "actor2_test",
    "actor1_test",
)

MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------
# Manifest loading and validation
# ---------------------------------------------------------------------

def load_manifest(
    windows_dir: Path,
    split: str,
) -> dict:
    """Load and validate the manifest for one prepared split."""

    split_dir = windows_dir / split
    manifest_path = split_dir / MANIFEST_NAME

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Window manifest not found:\n{manifest_path}"
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    required = (
        "split",
        "num_timesteps",
        "num_devices",
        "window_size",
        "num_samples",
        "shard_size",
        "num_shards",
        "dtype",
        "input_shape_per_sample",
        "target_shape_per_sample",
        "input_semantics",
        "target_semantics",
        "shards",
    )

    missing = [
        key for key in required
        if key not in manifest
    ]

    if missing:
        raise KeyError(
            f"Manifest {manifest_path} is missing "
            f"required fields: {missing}"
        )

    if manifest["split"] != split:
        raise ValueError(
            f"Manifest split mismatch: expected '{split}', "
            f"received '{manifest['split']}'."
        )

    if manifest["dtype"] != "float32":
        raise ValueError(
            "DBRL sharded input is expected to be float32. "
            f"Manifest reports dtype={manifest['dtype']}."
        )

    if manifest["input_semantics"] != "X_t = data[t:t+W]":
        raise ValueError(
            "Unexpected input-window semantics in manifest: "
            f"{manifest['input_semantics']}"
        )

    if manifest["target_semantics"] != (
        "Y_t = data[t+1:t+W+1]"
    ):
        raise ValueError(
            "Unexpected target-window semantics in manifest: "
            f"{manifest['target_semantics']}"
        )

    num_devices = int(manifest["num_devices"])
    window_size = int(manifest["window_size"])
    num_samples = int(manifest["num_samples"])
    shard_size = int(manifest["shard_size"])
    num_shards = int(manifest["num_shards"])

    if num_devices <= 0:
        raise ValueError("Manifest num_devices must be positive.")

    if window_size <= 0:
        raise ValueError("Manifest window_size must be positive.")

    if num_samples <= 0:
        raise ValueError("Manifest num_samples must be positive.")

    if shard_size <= 0:
        raise ValueError("Manifest shard_size must be positive.")

    expected_samples = (
        int(manifest["num_timesteps"]) - window_size
    )

    if num_samples != expected_samples:
        raise ValueError(
            "Manifest sample-count inconsistency: "
            f"expected T-W={expected_samples}, "
            f"received {num_samples}."
        )

    expected_num_shards = (
        num_samples + shard_size - 1
    ) // shard_size

    if num_shards != expected_num_shards:
        raise ValueError(
            "Manifest shard-count inconsistency: "
            f"expected {expected_num_shards}, "
            f"received {num_shards}."
        )

    if manifest["input_shape_per_sample"] != [
        window_size,
        num_devices,
    ]:
        raise ValueError(
            "Unexpected input shape in manifest."
        )

    if manifest["target_shape_per_sample"] != [
        window_size,
        num_devices,
    ]:
        raise ValueError(
            "Unexpected target shape in manifest."
        )

    shards = manifest["shards"]

    if not isinstance(shards, list):
        raise TypeError("Manifest 'shards' must be a list.")

    if len(shards) != num_shards:
        raise ValueError(
            "Manifest shard-list length mismatch: "
            f"expected {num_shards}, received {len(shards)}."
        )

    return manifest


def resolve_shard_path(
    split_dir: Path,
    manifest_entry: str,
) -> Path:
    """
    Resolve a shard path from the manifest.

    The manifest normally stores paths such as:
        train/shard_000000.npz

    The path is resolved relative to the windows directory's parent
    when necessary, while also supporting a simple filename.
    """

    candidate = split_dir.parent / manifest_entry

    if candidate.is_file():
        return candidate

    candidate = split_dir / Path(manifest_entry).name

    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        "Shard referenced by the manifest was not found:\n"
        f"manifest entry: {manifest_entry}\n"
        f"checked: {candidate}"
    )


# ---------------------------------------------------------------------
# Shard loading and validation
# ---------------------------------------------------------------------

def load_window_shard(
    shard_path: Path,
    *,
    expected_split: str,
    expected_window_size: int,
    expected_num_devices: int,
) -> dict[str, np.ndarray]:
    """
    Load exactly one window shard.

    Only this shard is materialized in memory. The complete training
    set is never loaded.
    """

    required = (
        "X",
        "Y",
        "window_start_timestamps",
        "target_timestamps",
        "split",
        "shard_index",
        "start_index",
        "end_index",
        "window_size",
        "num_devices",
    )

    with np.load(
        shard_path,
        allow_pickle=False,
    ) as archive:

        missing = [
            key for key in required
            if key not in archive.files
        ]

        if missing:
            raise KeyError(
                f"Shard {shard_path} is missing "
                f"required entries: {missing}"
            )

        artifact = {
            key: archive[key].copy()
            for key in required
        }

    X = artifact["X"]
    Y = artifact["Y"]

    if artifact["split"].item() != expected_split:
        raise ValueError(
            f"Shard split mismatch in {shard_path}: "
            f"expected '{expected_split}', "
            f"received '{artifact['split'].item()}'."
        )

    shard_window_size = int(
        artifact["window_size"].item()
    )

    shard_num_devices = int(
        artifact["num_devices"].item()
    )

    if shard_window_size != expected_window_size:
        raise ValueError(
            f"Window-size mismatch in {shard_path}: "
            f"expected {expected_window_size}, "
            f"received {shard_window_size}."
        )

    if shard_num_devices != expected_num_devices:
        raise ValueError(
            f"Device-count mismatch in {shard_path}: "
            f"expected {expected_num_devices}, "
            f"received {shard_num_devices}."
        )

    if X.ndim != 3:
        raise ValueError(
            f"X in {shard_path} must have shape "
            f"(S, W, N), received {X.shape}."
        )

    if Y.ndim != 3:
        raise ValueError(
            f"Y in {shard_path} must have shape "
            f"(S, W, N), received {Y.shape}."
        )

    num_samples = X.shape[0]

    expected_shape = (
        num_samples,
        expected_window_size,
        expected_num_devices,
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"Unexpected X shape in {shard_path}: "
            f"expected {expected_shape}, received {X.shape}."
        )

    if Y.shape != expected_shape:
        raise ValueError(
            f"Unexpected Y shape in {shard_path}: "
            f"expected {expected_shape}, received {Y.shape}."
        )

    if X.dtype != np.float32:
        raise TypeError(
            f"X in {shard_path} must be float32, "
            f"received {X.dtype}."
        )

    if Y.dtype != np.float32:
        raise TypeError(
            f"Y in {shard_path} must be float32, "
            f"received {Y.dtype}."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"X in {shard_path} contains NaN or infinite values."
        )

    if not np.isfinite(Y).all():
        raise ValueError(
            f"Y in {shard_path} contains NaN or infinite values."
        )

    window_start_timestamps = (
        artifact["window_start_timestamps"]
    )
    target_timestamps = (
        artifact["target_timestamps"]
    )

    if len(window_start_timestamps) != num_samples:
        raise ValueError(
            f"window_start_timestamps length mismatch in "
            f"{shard_path}."
        )

    if target_timestamps.shape[0] != num_samples:
        raise ValueError(
            f"target_timestamps sample dimension mismatch in "
            f"{shard_path}."
        )

    if target_timestamps.ndim != 2:
        raise ValueError(
            f"target_timestamps in {shard_path} must have shape "
            f"(S, W), received {target_timestamps.shape}."
        )

    if target_timestamps.shape[1] != expected_window_size:
        raise ValueError(
            f"target_timestamps window dimension mismatch in "
            f"{shard_path}: expected {expected_window_size}, "
            f"received {target_timestamps.shape[1]}."
        )

    return artifact


def verify_window_target_alignment(
    artifact: dict[str, np.ndarray],
) -> None:
    """
    Verify the persisted window-to-window timestamp relationship.

    This checks metadata alignment at the shard level. The numerical
    X/Y relationship itself was already verified by prepare_windows.py.
    """

    starts = artifact["window_start_timestamps"]
    targets = artifact["target_timestamps"]

    if len(starts) == 0:
        raise ValueError(
            "Cannot verify an empty window shard."
        )

    if targets.shape[0] != starts.shape[0]:
        raise ValueError(
            "Window/target timestamp sample counts differ."
        )


# ---------------------------------------------------------------------
# DBRL construction
# ---------------------------------------------------------------------

def build_dbrl(
    *,
    hidden_dim: int,
    embedding_dim: int,
    gru_layers: int,
    graph_top_k: int,
    graph_self_loops: bool,
    graph_symmetric: bool,
    graph_heads: int,
    num_devices: int,
) -> DBRL:
    """
    Construct the frozen DBRL module.

    No architectural changes are made here. The constructor and
    parameters remain identical to the current DBRL driver.
    """

    if graph_self_loops:
        max_top_k = num_devices
    else:
        max_top_k = num_devices - 1

    if graph_top_k <= 0:
        raise ValueError(
            "graph_top_k must be greater than zero."
        )

    if graph_top_k > max_top_k:
        raise ValueError(
            f"graph_top_k={graph_top_k} is invalid for "
            f"N={num_devices} devices. "
            f"Maximum allowed value is {max_top_k} "
            f"with graph_self_loops={graph_self_loops}."
        )

    return DBRL(
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        gru_layers=gru_layers,
        dropout=0.0,
        graph_top_k=graph_top_k,
        graph_self_loops=graph_self_loops,
        graph_symmetric=graph_symmetric,
        graph_heads=graph_heads,
        graph_dropout=0.0,
    )


# ---------------------------------------------------------------------
# DBRL forward pass for one shard
# ---------------------------------------------------------------------

def run_dbrl_on_shard(
    model: DBRL,
    X: np.ndarray,
    *,
    device: torch.device,
    embedding_dim: int,
    num_devices: int,
    batch_size: int,
    max_batches: Optional[int],
) -> tuple[torch.Tensor, int]:
    """
    Run DBRL on one shard only.

    Returns
    -------
    representations:
        Tensor with shape (S_processed, N, D).
    processed_batches:
        Number of processed batches.
    """

    X_tensor = torch.from_numpy(X)

    if X_tensor.dtype != torch.float32:
        raise RuntimeError(
            "Prepared X tensor must be float32."
        )

    dataset = TensorDataset(X_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    representations: list[torch.Tensor] = []

    model.eval()

    processed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):

            x = batch[0].to(
                device,
                non_blocking=False,
            )

            if x.ndim != 3:
                raise RuntimeError(
                    "DBRL input must have shape (B, W, N). "
                    f"Received {tuple(x.shape)}."
                )

            current_batch_size, window_size, batch_devices = (
                x.shape
            )

            if batch_devices != num_devices:
                raise RuntimeError(
                    "Unexpected device dimension in batch: "
                    f"expected N={num_devices}, "
                    f"received N={batch_devices}."
                )

            print(
                f"    Batch {batch_idx + 1}: "
                f"X={tuple(x.shape)}",
                end="",
            )

            z = model(x)

            print(
                f" -> Z={tuple(z.shape)}"
            )

            expected_shape = (
                current_batch_size,
                num_devices,
                embedding_dim,
            )

            if tuple(z.shape) != expected_shape:
                raise RuntimeError(
                    "DBRL output shape mismatch: "
                    f"expected {expected_shape}, "
                    f"received {tuple(z.shape)}."
                )

            if not torch.isfinite(z).all():
                raise RuntimeError(
                    f"DBRL produced NaN or infinite values "
                    f"in shard batch {batch_idx}."
                )

            representations.append(
                z.detach().cpu()
            )

            processed_batches += 1

            del x
            del z

            if (
                max_batches is not None
                and processed_batches >= max_batches
            ):
                break

    if not representations:
        raise RuntimeError(
            "DBRL produced no representations for the shard."
        )

    Z = torch.cat(
        representations,
        dim=0,
    )

    del representations
    del dataloader
    del dataset
    del X_tensor

    return Z, processed_batches


# ---------------------------------------------------------------------
# Representation artifact saving
# ---------------------------------------------------------------------

def save_representation_shard(
    output_path: Path,
    Z: torch.Tensor,
    *,
    split: str,
    source_shard: str,
    shard_index: int,
    start_index: int,
    end_index: int,
    window_size: int,
    num_devices: int,
    embedding_dim: int,
    batch_size: int,
    seed: int,
    overwrite: bool,
) -> None:
    """Persist one DBRL representation shard."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"DBRL output already exists:\n{output_path}\n\n"
            "Use the overwrite option to replace it."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "representations": Z,
        "split": split,
        "source_shard": source_shard,
        "shard_index": shard_index,
        "start_index": start_index,
        "end_index": end_index,
        "window_size": window_size,
        "num_devices": num_devices,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "seed": seed,
    }

    torch.save(
        payload,
        output_path,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------

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
    
    root = config["project_root_dir"]
    print(root)

    split = config["split"]

    batch_size = config["hdeg"]["dbrl"]["batch_size"]
    max_batches = config["hdeg"]["dbrl"]["max_batches"]
    max_shards = config["hdeg"]["dbrl"]["max_shards"]
    overwrite = config["hdeg"]["dbrl"]["overwrite"]
    
    if max_shards is not None and max_shards <= 0:
        raise ValueError(
            "max_shards must be greater than zero."
        )

    if max_batches is not None and max_batches <= 0:
        raise ValueError(
            "max_batches must be greater than zero."
        )

    # -------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------

    processed_data_folder = get_processed_folder(
        config
    )

    windows_dir = Path(processed_data_folder) / "windows"

    split_dir = Path(f"{windows_dir}/{split}")

    output_dir = (Path(f"{processed_data_folder}/dbrl/{split}"))

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Window split directory not found:\n{split_dir}"
        )

    # -------------------------------------------------------------
    # Load and validate manifest
    # -------------------------------------------------------------

    manifest = load_manifest(
        windows_dir=windows_dir,
        split=split,
    )

    num_devices = int(
        manifest["num_devices"]
    )

    window_size = int(
        manifest["window_size"]
    )

    num_samples = int(
        manifest["num_samples"]
    )

    num_shards = int(
        manifest["num_shards"]
    )

    shard_size = int(
        manifest["shard_size"]
    )

    embedding_dim = int(
        config["hdeg"]["dbrl"]["embedding_dim"]
    )

    batch_size = config["hdeg"]["dbrl"]["batch_size"]

    max_batches = config["hdeg"]["dbrl"]["max_batches"]

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    # -------------------------------------------------------------
    # Header
    # -------------------------------------------------------------

    print("=" * 70)
    print("HDEG — DBRL Sharded Standalone Execution")
    print("=" * 70)

    print(
        f"Split              : {config['split']}"
    )

    print(
        f"Windows directory  : {windows_dir}"
    )

    print(
        f"Output directory   : {output_dir}"
    )

    print(
        f"Device             : {device}"
    )

    print(
        f"Window size        : {window_size}"
    )

    print(
        f"Number of devices  : {num_devices}"
    )

    print(
        f"Total samples      : {num_samples}"
    )

    print(
        f"Shard size         : {shard_size}"
    )

    print(
        f"Number of shards   : {num_shards}"
    )

    print(
        f"DBRL batch size    : {batch_size}"
    )

    print(
        f"DBRL embedding dim : {embedding_dim}"
    )

    print(
        f"Max shards         : {max_shards}"
    )

    print(
        f"Max batches/shard  : {max_batches}"
    )

    print()

    # -------------------------------------------------------------
    # Build frozen DBRL
    # -------------------------------------------------------------

    model = build_dbrl(
        hidden_dim=int(
            config["hdeg"]["dbrl"]["hidden_dim"]
        ),
        embedding_dim=embedding_dim,
        gru_layers=int(
            config["hdeg"]["dbrl"]["gru_layers"]
        ),
        graph_top_k=int(
            config["hdeg"]["dbrl"]["graph_top_k"]
        ),
        graph_self_loops=bool(
            config["hdeg"]["dbrl"]["graph_self_loops"]
        ),
        graph_symmetric=bool(
            config["hdeg"]["dbrl"]["graph_symmetric"]
        ),
        graph_heads=int(
            config["hdeg"]["dbrl"]["graph_heads"]
        ),
        num_devices=num_devices,
    ).to(device)

    print("DBRL model constructed successfully.")
    print()

    # -------------------------------------------------------------
    # Process shards sequentially
    # -------------------------------------------------------------

    shard_entries = manifest["shards"]

    if max_shards is not None:
        shard_entries = shard_entries[
            :max_shards
        ]

    total_processed = 0
    total_processed_batches = 0

    for sequence_index, manifest_entry in enumerate(
        shard_entries
    ):

        shard_path = resolve_shard_path(
            split_dir=split_dir,
            manifest_entry=manifest_entry,
        )

        print("=" * 70)
        print(
            f"Shard {sequence_index + 1}/"
            f"{len(shard_entries)}"
        )
        print("=" * 70)

        print(
            f"Source: {shard_path}"
        )

        artifact = load_window_shard(
            shard_path=shard_path,
            expected_split=split,
            expected_window_size=window_size,
            expected_num_devices=num_devices,
        )

        verify_window_target_alignment(
            artifact
        )

        X = artifact["X"]

        shard_index = int(
            artifact["shard_index"].item()
        )

        start_index = int(
            artifact["start_index"].item()
        )

        end_index = int(
            artifact["end_index"].item()
        )

        shard_samples = X.shape[0]

        print(
            f"Shard index       : {shard_index}"
        )

        print(
            f"Sample range      : "
            f"[{start_index}, {end_index})"
        )

        print(
            f"X shape           : {X.shape}"
        )

        print(
            f"Y shape           : "
            f"{artifact['Y'].shape}"
        )

        print(
            f"X dtype           : {X.dtype}"
        )

        # ---------------------------------------------------------
        # Run DBRL on this shard
        # ---------------------------------------------------------

        print()
        print("Running DBRL on shard...")

        Z, processed_batches = run_dbrl_on_shard(
            model=model,
            X=X,
            device=device,
            embedding_dim=embedding_dim,
            num_devices=num_devices,
            batch_size=batch_size,
            max_batches=max_batches,
        )

        processed_samples = Z.shape[0]

        expected_processed_samples = shard_samples

        if max_batches is not None:
            expected_processed_samples = min(
                shard_samples,
                max_batches * batch_size,
            )

        if processed_samples != expected_processed_samples:
            raise RuntimeError(
                "Unexpected number of DBRL outputs for shard "
                f"{shard_index}: expected "
                f"{expected_processed_samples}, received "
                f"{processed_samples}."
            )

        expected_z_shape = (
            processed_samples,
            num_devices,
            embedding_dim,
        )

        if tuple(Z.shape) != expected_z_shape:
            raise RuntimeError(
                "Final shard representation shape mismatch: "
                f"expected {expected_z_shape}, "
                f"received {tuple(Z.shape)}."
            )

        if not torch.isfinite(Z).all():
            raise RuntimeError(
                f"DBRL representation shard {shard_index} "
                "contains NaN or infinite values."
            )

        # ---------------------------------------------------------
        # Save representation shard
        # ---------------------------------------------------------

        output_path = (
            output_dir
            / f"shard_{shard_index:06d}.pt"
        )

        save_representations_kwargs = {
            "output_path": output_path,
            "Z": Z,
            "split": split,
            "source_shard": manifest_entry,
            "shard_index": shard_index,
            "start_index": start_index,
            "end_index": (
                start_index + processed_samples
            ),
            "window_size": window_size,
            "num_devices": num_devices,
            "embedding_dim": embedding_dim,
            "batch_size": batch_size,
            "seed": int(config["seed"]),
            "overwrite": (
                overwrite
                or bool(
                    config["hdeg"]["dbrl"].get(
                        "overwrite",
                        False,
                    )
                )
            ),
        }

        save_representation_shard(
            **save_representations_kwargs
        )

        print()
        print(
            f"[PASS] DBRL output saved to:\n"
            f"       {output_path}"
        )

        print(
            f"[PASS] Z shape = {tuple(Z.shape)}"
        )

        total_processed += processed_samples
        total_processed_batches += processed_batches

        # ---------------------------------------------------------
        # Explicitly release shard and representation memory.
        # ---------------------------------------------------------

        del Z
        del X
        del artifact

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"Processed samples so far: "
            f"{total_processed}"
        )

        # A partial smoke test intentionally stops here if requested.
        if (
            max_shards is not None
            and sequence_index + 1 >= max_shards
        ):
            break

    # -------------------------------------------------------------
    # Final execution summary
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("DBRL sharded execution completed")
    print("=" * 70)

    print(
        f"Processed shards   : {len(shard_entries)}"
    )

    print(
        f"Processed samples  : {total_processed}"
    )

    print(
        f"Processed batches  : {total_processed_batches}"
    )

    print(
        f"Expected total     : {num_samples}"
    )

    if max_shards is None and max_batches is None:
        if total_processed != num_samples:
            raise RuntimeError(
                "Complete DBRL execution did not process the "
                "expected number of samples: "
                f"expected {num_samples}, "
                f"received {total_processed}."
            )

        print(
            "[PASS] Complete split processed exactly once."
        )

    else:
        print(
            "[INFO] Partial execution requested; "
            "complete-split coverage was not required."
        )

    print(
        f"Representation directory:\n"
        f"  {output_dir}"
    )

    print(
        "[PASS] DBRL sharded standalone execution finished."
    )


if __name__ == "__main__":
    main()