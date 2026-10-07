from __future__ import annotations

import argparse
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

def load_windows(
    window_path: Path,
    *,
    expected_window_size: int,
) -> dict[str, np.ndarray]:
    """Load one non-sharded split file produced by prepare_windows.py."""

    required = (
        "X",
        "y",
        "window_start_timestamps",
        "target_timestamps",
        "window_size",
        "num_devices",
    )

    with np.load(
        window_path,
        allow_pickle=False,
    ) as archive:

        missing = [
            key for key in required
            if key not in archive.files
        ]

        if missing:
            raise KeyError(
                f"Window file {window_path} is missing "
                f"required entries: {missing}"
            )

        artifact = {
            key: archive[key]
            for key in required
        }

    X = artifact["X"]
    Y = artifact["y"]

    file_window_size = int(
        artifact["window_size"].item()
    )

    file_num_devices = int(
        artifact["num_devices"].item()
    )

    if file_window_size != expected_window_size:
        raise ValueError(
            f"Window-size mismatch in {window_path}: "
            f"expected {expected_window_size}, "
            f"received {file_window_size}."
        )

    if file_num_devices <= 0:
        raise ValueError("num_devices must be positive.")

    if X.ndim != 3:
        raise ValueError(
            f"X in {window_path} must have shape "
            f"(S, W, N), received {X.shape}."
        )

    if Y.ndim != 3:
        raise ValueError(
            f"Y in {window_path} must have shape "
            f"(S, W, N), received {Y.shape}."
        )

    num_samples = X.shape[0]

    expected_shape = (
        num_samples,
        expected_window_size,
        file_num_devices,
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"Unexpected X shape in {window_path}: "
            f"expected {expected_shape}, received {X.shape}."
        )

    if Y.shape != expected_shape:
        raise ValueError(
            f"Unexpected Y shape in {window_path}: "
            f"expected {expected_shape}, received {Y.shape}."
        )

    if not np.issubdtype(X.dtype, np.floating) or not np.issubdtype(Y.dtype, np.floating):
        raise TypeError("Prepared X and y must contain floating-point values.")

    # Preprocessing may save float64; the existing DBRL expects float32.
    artifact["X"] = X = X.astype(np.float32, copy=False)
    artifact["y"] = Y = Y.astype(np.float32, copy=False)

    if not np.isfinite(X).all():
        raise ValueError(
            f"X in {window_path} contains NaN or infinite values."
        )

    if not np.isfinite(Y).all():
        raise ValueError(
            f"Y in {window_path} contains NaN or infinite values."
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
            f"{window_path}."
        )

    if target_timestamps.shape[0] != num_samples:
        raise ValueError(
            f"target_timestamps sample dimension mismatch in "
            f"{window_path}."
        )

    if target_timestamps.ndim != 2:
        raise ValueError(
            f"target_timestamps in {window_path} must have shape "
            f"(S, W), received {target_timestamps.shape}."
        )

    if target_timestamps.shape[1] != expected_window_size:
        raise ValueError(
            f"target_timestamps window dimension mismatch in "
            f"{window_path}: expected {expected_window_size}, "
            f"received {target_timestamps.shape[1]}."
        )

    return artifact


def verify_window_target_alignment(
    artifact: dict[str, np.ndarray],
) -> None:
    """
    Verify the persisted window-to-window timestamp relationship.

    This checks metadata alignment at the split level. The numerical
    X/Y relationship itself was already verified by prepare_windows.py.
    """

    starts = artifact["window_start_timestamps"]
    targets = artifact["target_timestamps"]

    if len(starts) == 0:
        raise ValueError(
            "Cannot verify an empty window dataset."
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
# DBRL forward pass for one split
# ---------------------------------------------------------------------

def run_dbrl_on_split(
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
    Run DBRL on one split only.

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
                    f"in batch {batch_idx}."
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
            "DBRL produced no representations for the split."
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

def save_representations(
    output_path: Path,
    Z: torch.Tensor,
    *,
    split: str,
    source_windows: str,
    start_index: int,
    end_index: int,
    window_size: int,
    num_devices: int,
    embedding_dim: int,
    batch_size: int,
    seed: int,
    overwrite: bool,
) -> None:
    """Persist one DBRL representation file."""

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
        "source_windows": source_windows,
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

    set_seed(config["seed"])
    device = get_device()
    dbrl_config = config["hdeg"]["dbrl"]
    batch_size = int(dbrl_config["batch_size"])
    embedding_dim = int(dbrl_config["embedding_dim"])
    window_size = int(config["preprocessing"]["window_size"])
    max_batches = dbrl_config.get("max_batches")
    # Support the existing YAML's literal 'None' as well as YAML null.
    max_batches = None if max_batches in (None, "None", "null") else int(max_batches)
    overwrite = bool(dbrl_config.get("overwrite", False))

    if batch_size <= 0 or window_size <= 0:
        raise ValueError("batch_size and window_size must be greater than zero.")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be greater than zero.")

    processed_dir = Path(get_processed_folder(config))
    window_path = processed_dir / "windows" / f"{split}.npz"
    output_path = processed_dir / "dbrl" / f"{split}.pt"
    if not window_path.is_file():
        raise FileNotFoundError(f"Window file not found:\n{window_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"DBRL output already exists:\n{output_path}")

    artifact = load_windows(
        window_path,
        expected_window_size=window_size,
    )
    verify_window_target_alignment(artifact)
    X = artifact["X"]
    num_samples, _, num_devices = X.shape
    # DBRL uses X only. Release the loaded targets and timestamps.
    del artifact

    print("=" * 70)
    print("HDEG — DBRL Non-sharded Standalone Execution")
    print("=" * 70)
    print(f"Split              : {split}")
    print(f"Window file        : {window_path}")
    print(f"Output file        : {output_path}")
    print(f"Device             : {device}")
    print(f"X shape            : {X.shape}")
    print(f"DBRL batch size    : {batch_size}")
    print(f"DBRL embedding dim : {embedding_dim}")
    print(f"Max batches        : {max_batches}")

    model = build_dbrl(
        hidden_dim=int(dbrl_config["hidden_dim"]),
        embedding_dim=embedding_dim,
        gru_layers=int(dbrl_config["gru_layers"]),
        graph_top_k=int(dbrl_config["graph_top_k"]),
        graph_self_loops=bool(dbrl_config["graph_self_loops"]),
        graph_symmetric=bool(dbrl_config["graph_symmetric"]),
        graph_heads=int(dbrl_config["graph_heads"]),
        num_devices=num_devices,
    ).to(device)

    Z, processed_batches = run_dbrl_on_split(
        model=model,
        X=X,
        device=device,
        embedding_dim=embedding_dim,
        num_devices=num_devices,
        batch_size=batch_size,
        max_batches=max_batches,
    )

    expected_samples = num_samples if max_batches is None else min(num_samples, max_batches * batch_size)
    if tuple(Z.shape) != (expected_samples, num_devices, embedding_dim):
        raise RuntimeError(f"Unexpected final representation shape: {tuple(Z.shape)}")

    save_representations(
        output_path=output_path,
        Z=Z,
        split=split,
        source_windows=str(window_path),
        start_index=0,
        end_index=int(Z.shape[0]),
        window_size=window_size,
        num_devices=num_devices,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        seed=int(config["seed"]),
        overwrite=overwrite,
    )

    print(f"[PASS] Saved Z with shape {tuple(Z.shape)} to {output_path}")
    print(f"Processed samples: {Z.shape[0]}/{num_samples}; batches: {processed_batches}")
    if Z.shape[0] < num_samples:
        print("[INFO] Partial execution requested by max_batches.")


if __name__ == "__main__":
    main()
