from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from src.common.graph.semantics import load_behavioral_state_config
from src.models.hdeg.bse import BehavioralStateEstimator
from src.utils.device import get_device
from src.utils.get_folders_utils import get_processed_folder
from src.utils.seed import set_seed


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SPLITS = (
    "train",
    "val",
    "actor2_test",
    "actor1_test",
)

DBRL_SHARD_PATTERN = "shard_*.pt"


# ---------------------------------------------------------------------
# DBRL representation shard discovery
# ---------------------------------------------------------------------

def discover_dbrl_representation_shards(
    dbrl_split_dir: Path,
) -> list[Path]:
    """
    Discover persisted DBRL representation shards.

    The frozen DBRL artifact contract does not define a DBRL manifest.
    DBRL persists one representation artifact per upstream window shard
    using the filename convention:

        shard_000000.pt
        shard_000001.pt
        ...

    Therefore BSE discovers the persisted .pt artifacts directly from
    the DBRL split directory.
    """

    if not dbrl_split_dir.is_dir():
        raise FileNotFoundError(
            "DBRL representation split directory not found:\n"
            f"{dbrl_split_dir}"
        )

    shard_paths = sorted(
        dbrl_split_dir.glob(DBRL_SHARD_PATTERN)
    )

    if not shard_paths:
        raise FileNotFoundError(
            "No DBRL representation shards were found in:\n"
            f"{dbrl_split_dir}\n"
            f"Expected files matching: {DBRL_SHARD_PATTERN}"
        )

    return shard_paths


def parse_dbrl_shard_filename(
    shard_path: Path,
) -> int:
    """
    Extract the zero-based shard index from a DBRL shard filename.

    Expected format:

        shard_000000.pt
    """

    if shard_path.suffix != ".pt":
        raise ValueError(
            f"Invalid DBRL representation shard suffix: "
            f"{shard_path.name}"
        )

    stem = shard_path.stem

    if not stem.startswith("shard_"):
        raise ValueError(
            f"Invalid DBRL representation shard filename: "
            f"{shard_path.name}"
        )

    index_text = stem[len("shard_"):]

    if not index_text.isdigit():
        raise ValueError(
            f"Invalid DBRL representation shard filename: "
            f"{shard_path.name}"
        )

    shard_index = int(index_text)

    if shard_index < 0:
        raise ValueError(
            f"Invalid negative DBRL shard index in "
            f"{shard_path.name}."
        )

    return shard_index


def validate_dbrl_shard_sequence(
    shard_paths: list[Path],
) -> None:
    """
    Verify that discovered DBRL shard filenames form a contiguous
    zero-based sequence.

    This validation is performed before any BSE computation so that a
    missing, duplicated, or unexpectedly named DBRL artifact cannot
    silently create incomplete downstream coverage.
    """

    if not shard_paths:
        raise ValueError(
            "DBRL representation shard list is empty."
        )

    actual_indices = [
        parse_dbrl_shard_filename(path)
        for path in shard_paths
    ]

    if len(set(actual_indices)) != len(actual_indices):
        raise RuntimeError(
            "Duplicate DBRL shard indices were discovered: "
            f"{actual_indices}."
        )

    expected_indices = list(
        range(len(shard_paths))
    )

    if actual_indices != expected_indices:
        raise RuntimeError(
            "DBRL representation shards do not form a contiguous "
            "zero-based sequence. "
            f"Expected {expected_indices[:10]}"
            f"{'...' if len(expected_indices) > 10 else ''}, "
            f"received {actual_indices[:10]}"
            f"{'...' if len(actual_indices) > 10 else ''}."
        )


# ---------------------------------------------------------------------
# DBRL representation shard loading
# ---------------------------------------------------------------------

def load_dbrl_representation_shard(
    shard_path: Path,
    *,
    expected_split: str,
    expected_num_devices: int,
    expected_embedding_dim: Optional[int],
) -> dict[str, Any]:
    """
    Load exactly one persisted DBRL representation shard.

    Parameters
    ----------
    expected_embedding_dim:
        If not None, the artifact embedding dimension must equal this
        value. If None, the dimension is established from this artifact
        and still validated against the artifact's own metadata.

    Returns
    -------
    dict
        The persisted DBRL artifact payload.
    """

    if not shard_path.is_file():
        raise FileNotFoundError(
            f"DBRL representation shard not found:\n"
            f"{shard_path}"
        )

    payload = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise TypeError(
            f"DBRL shard {shard_path} must contain "
            "a dictionary payload."
        )

    required = (
        "representations",
        "split",
        "source_shard",
        "shard_index",
        "start_index",
        "end_index",
        "window_size",
        "num_devices",
        "embedding_dim",
    )

    missing = [
        key
        for key in required
        if key not in payload
    ]

    if missing:
        raise KeyError(
            f"DBRL shard {shard_path} is missing "
            f"required entries: {missing}"
        )

    # -------------------------------------------------------------
    # Split
    # -------------------------------------------------------------

    if payload["split"] != expected_split:
        raise ValueError(
            f"DBRL shard split mismatch in {shard_path}: "
            f"expected '{expected_split}', "
            f"received '{payload['split']}'."
        )

    # -------------------------------------------------------------
    # Representation tensor
    # -------------------------------------------------------------

    Z = payload["representations"]

    if not isinstance(Z, torch.Tensor):
        raise TypeError(
            f"'representations' in {shard_path} must be "
            "a torch.Tensor."
        )

    if Z.ndim != 3:
        raise ValueError(
            f"DBRL representations in {shard_path} must have "
            "shape (S, N, D). "
            f"Received {tuple(Z.shape)}."
        )

    num_samples = Z.shape[0]
    num_devices = Z.shape[1]
    embedding_dim = Z.shape[2]

    if num_samples <= 0:
        raise ValueError(
            f"DBRL representation shard {shard_path} is empty."
        )

    expected_shape = (
        num_samples,
        expected_num_devices,
        embedding_dim,
    )

    if tuple(Z.shape) != expected_shape:
        raise ValueError(
            f"Unexpected DBRL representation shape in "
            f"{shard_path}: expected device dimension "
            f"N={expected_num_devices}, received {tuple(Z.shape)}."
        )

    if Z.dtype != torch.float32:
        raise TypeError(
            f"DBRL representations in {shard_path} must be "
            f"float32. Received {Z.dtype}."
        )

    if not torch.isfinite(Z).all():
        raise ValueError(
            f"DBRL representations in {shard_path} contain "
            "NaN or infinite values."
        )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    metadata_num_devices = int(
        payload["num_devices"]
    )

    metadata_embedding_dim = int(
        payload["embedding_dim"]
    )

    if metadata_num_devices != expected_num_devices:
        raise ValueError(
            f"DBRL metadata device-count mismatch in "
            f"{shard_path}: expected {expected_num_devices}, "
            f"received {metadata_num_devices}."
        )

    if metadata_embedding_dim != embedding_dim:
        raise ValueError(
            f"DBRL metadata embedding-dimension mismatch in "
            f"{shard_path}: tensor has D={embedding_dim}, "
            f"metadata reports D={metadata_embedding_dim}."
        )

    if expected_embedding_dim is not None:
        if metadata_embedding_dim != expected_embedding_dim:
            raise ValueError(
                f"DBRL metadata embedding-dimension mismatch "
                f"in {shard_path}: expected {expected_embedding_dim}, "
                f"received {metadata_embedding_dim}."
            )

    shard_index = int(
        payload["shard_index"]
    )

    start_index = int(
        payload["start_index"]
    )

    end_index = int(
        payload["end_index"]
    )

    if shard_index < 0:
        raise ValueError(
            f"Invalid shard_index={shard_index} "
            f"in {shard_path}."
        )

    filename_index = parse_dbrl_shard_filename(
        shard_path
    )

    if shard_index != filename_index:
        raise ValueError(
            f"DBRL shard-index mismatch in {shard_path}: "
            f"filename encodes index {filename_index}, "
            f"payload reports {shard_index}."
        )

    if start_index < 0:
        raise ValueError(
            f"Invalid start_index={start_index} "
            f"in {shard_path}."
        )

    if end_index <= start_index:
        raise ValueError(
            f"Invalid sample range in {shard_path}: "
            f"[{start_index}, {end_index})."
        )

    expected_sample_count = (
        end_index - start_index
    )

    if expected_sample_count != num_samples:
        raise ValueError(
            f"DBRL shard sample-range mismatch in "
            f"{shard_path}: range contains "
            f"{expected_sample_count} samples, "
            f"but representations contain {num_samples}."
        )

    return payload


# ---------------------------------------------------------------------
# BSE construction
# ---------------------------------------------------------------------

def build_bse(
    *,
    num_states: int,
    embedding_dim: int,
    num_heads: int,
) -> BehavioralStateEstimator:
    """
    Construct the frozen BSE module.

    No architectural modification is performed here.
    """

    if num_states <= 0:
        raise ValueError(
            "num_states must be greater than zero."
        )

    if embedding_dim <= 0:
        raise ValueError(
            "embedding_dim must be greater than zero."
        )

    if num_heads <= 0:
        raise ValueError(
            "num_heads must be greater than zero."
        )

    return BehavioralStateEstimator(
        num_states=num_states,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
    )


# ---------------------------------------------------------------------
# BSE forward pass
# ---------------------------------------------------------------------

def run_bse_on_shard(
    model: BehavioralStateEstimator,
    Z: torch.Tensor,
    compatibility_mask: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    Run BSE on one DBRL representation shard.

    Z has shape (S, N, D); the returned behavioral-state
    representation has shape (S, K, D).
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if Z.ndim != 3:
        raise ValueError(
            "BSE input must have shape (S, N, D). "
            f"Received {tuple(Z.shape)}."
        )

    if compatibility_mask.ndim != 2:
        raise ValueError(
            "Compatibility mask must have shape (K, N). "
            f"Received {tuple(compatibility_mask.shape)}."
        )

    if Z.shape[1] != compatibility_mask.shape[1]:
        raise ValueError(
            "DBRL/BSE device dimension mismatch: "
            f"Z has N={Z.shape[1]}, "
            f"mask has N={compatibility_mask.shape[1]}."
        )

    if compatibility_mask.shape[0] != model.num_states:
        raise ValueError(
            "BSE state-count mismatch: "
            f"model has K={model.num_states}, "
            f"mask has K={compatibility_mask.shape[0]}."
        )

    if Z.shape[2] != model.embedding_dim:
        raise ValueError(
            "BSE embedding-dimension mismatch: "
            f"Z has D={Z.shape[2]}, "
            f"model expects D={model.embedding_dim}."
        )

    if Z.dtype != torch.float32:
        raise TypeError(
            f"BSE input must be float32. "
            f"Received {Z.dtype}."
        )

    if not torch.isfinite(Z).all():
        raise ValueError(
            "BSE input contains NaN or infinite values."
        )

    model.eval()

    outputs: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(
            0,
            Z.shape[0],
            batch_size,
        ):
            end = min(
                start + batch_size,
                Z.shape[0],
            )

            z_batch = Z[start:end].to(
                device,
                non_blocking=False,
            )

            S_batch = model(
                z_batch,
                compatibility_mask,
            )

            expected_shape = (
                z_batch.shape[0],
                model.num_states,
                model.embedding_dim,
            )

            if tuple(S_batch.shape) != expected_shape:
                raise RuntimeError(
                    "BSE output shape mismatch: "
                    f"expected {expected_shape}, "
                    f"received {tuple(S_batch.shape)}."
                )

            if not torch.isfinite(S_batch).all():
                raise RuntimeError(
                    "BSE produced NaN or infinite values."
                )

            outputs.append(
                S_batch.detach().cpu()
            )

            del z_batch
            del S_batch

    if not outputs:
        raise RuntimeError(
            "BSE produced no representations."
        )

    S = torch.cat(
        outputs,
        dim=0,
    )

    if S.shape[0] != Z.shape[0]:
        raise RuntimeError(
            "BSE changed the number of samples: "
            f"input={Z.shape[0]}, "
            f"output={S.shape[0]}."
        )

    return S


# ---------------------------------------------------------------------
# BSE representation artifact persistence
# ---------------------------------------------------------------------

def save_bse_representation_shard(
    output_path: Path,
    S: torch.Tensor,
    *,
    source_dbrl_payload: dict[str, Any],
    num_states: int,
    num_devices: int,
    embedding_dim: int,
    num_heads: int,
    seed: int,
    overwrite: bool,
) -> None:
    """
    Persist one BSE representation shard.

    DBRL provenance is retained so that the BSE artifact can be traced
    to the exact DBRL shard and therefore to the corresponding window
    samples.
    """

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"BSE output already exists:\n"
            f"{output_path}\n\n"
            "Use --overwrite or enable overwrite in the "
            "configuration to replace it."
        )

    if S.ndim != 3:
        raise ValueError(
            "BSE representation tensor must have shape (S, K, D)."
        )

    if S.dtype != torch.float32:
        raise TypeError(
            "BSE representation tensor must be float32."
        )

    if S.shape[1] != num_states:
        raise ValueError(
            "BSE representation state dimension mismatch: "
            f"expected K={num_states}, "
            f"received {S.shape[1]}."
        )

    if S.shape[2] != embedding_dim:
        raise ValueError(
            "BSE representation embedding dimension mismatch: "
            f"expected D={embedding_dim}, "
            f"received {S.shape[2]}."
        )

    if not torch.isfinite(S).all():
        raise ValueError(
            "BSE representation tensor contains NaN or infinite values."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        # Scientific artifact
        "representations": S,

        # Split / provenance
        "split": source_dbrl_payload["split"],
        "source_dbrl_shard": source_dbrl_payload["source_shard"],
        "source_dbrl_shard_index": int(
            source_dbrl_payload["shard_index"]
        ),
        "shard_index": int(
            source_dbrl_payload["shard_index"]
        ),

        # Sample alignment
        "start_index": int(
            source_dbrl_payload["start_index"]
        ),
        "end_index": int(
            source_dbrl_payload["end_index"]
        ),

        # Structural metadata
        "window_size": int(
            source_dbrl_payload["window_size"]
        ),
        "num_devices": int(num_devices),
        "num_states": int(num_states),
        "embedding_dim": int(embedding_dim),
        "num_heads": int(num_heads),

        # Execution provenance
        "seed": int(seed),
    }

    torch.save(
        payload,
        output_path,
    )


# ---------------------------------------------------------------------
# BSE output verification
# ---------------------------------------------------------------------

def verify_bse_output_shard(
    S: torch.Tensor,
    *,
    expected_samples: int,
    expected_num_states: int,
    expected_embedding_dim: int,
) -> None:
    """Verify the scientific BSE representation produced for a shard."""

    expected_shape = (
        expected_samples,
        expected_num_states,
        expected_embedding_dim,
    )

    if tuple(S.shape) != expected_shape:
        raise RuntimeError(
            "BSE representation shape mismatch: "
            f"expected {expected_shape}, "
            f"received {tuple(S.shape)}."
        )

    if S.dtype != torch.float32:
        raise RuntimeError(
            "BSE representation dtype mismatch: "
            f"expected torch.float32, "
            f"received {S.dtype}."
        )

    if not torch.isfinite(S).all():
        raise RuntimeError(
            "BSE representation contains NaN or infinite values."
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

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
    
    root = config["project_root_dir"]
    print(root)

    split = config["split"]

    batch_size = int(
        config["hdeg"]["dbrl"]["batch_size"]
    )

    num_heads = int(
        config["hdeg"]["bse"]["num_heads"]
    )

    max_shards = config["hdeg"]["bse"].get(
        "max_shards",
        None,
    )

    overwrite = bool(
        config["hdeg"]["bse"].get(
            "overwrite",
            False,
        )
    )

    if batch_size <= 0:
        raise ValueError(
            "BSE batch_size must be greater than zero."
        )

    if num_heads <= 0:
        raise ValueError(
            "BSE num_heads must be greater than zero."
        )

    if max_shards is not None:
        max_shards = int(max_shards)
        if max_shards <= 0:
            raise ValueError(
                "max_shards must be greater than zero."
            )

    # -------------------------------------------------------------
    # Processed-data paths
    # -------------------------------------------------------------

    processed_data_folder = Path(
        get_processed_folder(config)
    )

    dbrl_dir = Path(f"{processed_data_folder}/dbrl")
    dbrl_split_dir = dbrl_dir / split

    bse_dir = Path(f"{processed_data_folder}/bse")
    bse_split_dir = bse_dir / split

    # -------------------------------------------------------------
    # Behavioral-state configuration paths
    # -------------------------------------------------------------

    behavioral_config_path = (
        Path(root)
        / "configs"
        / "hdeg"
        / "behavioral_states.yaml"
    )

    devices_path = processed_data_folder / "devices.json"

    # -------------------------------------------------------------
    # Header
    # -------------------------------------------------------------

    print("=" * 70)
    print("HDEG — BSE Sharded Standalone Execution")
    print("=" * 70)
    print(f"Split               : {split}")
    print(f"DBRL input directory : {dbrl_split_dir}")
    print(f"BSE output directory : {bse_split_dir}")
    print(f"Device               : {device}")
    print(f"BSE batch size       : {batch_size}")
    print(f"BSE attention heads  : {num_heads}")
    print(f"Max shards           : {max_shards}")
    print(f"Overwrite            : {overwrite}")
    print()

    # -------------------------------------------------------------
    # Discover and validate the frozen DBRL artifact layer
    # -------------------------------------------------------------

    dbrl_shard_paths = discover_dbrl_representation_shards(
        dbrl_split_dir
    )

    validate_dbrl_shard_sequence(
        dbrl_shard_paths
    )

    total_dbrl_shards = len(dbrl_shard_paths)

    if max_shards is not None:
        shard_paths = dbrl_shard_paths[:max_shards]
    else:
        shard_paths = dbrl_shard_paths

    print(
        f"Discovered DBRL shards : {total_dbrl_shards}"
    )
    print(
        f"Shards to process      : {len(shard_paths)}"
    )
    print()

    # -------------------------------------------------------------
    # Load behavioral-state configuration
    # -------------------------------------------------------------

    print(
        "Loading behavioral-state configuration..."
    )

    behavioral_config = load_behavioral_state_config(
        config_path=behavioral_config_path,
        devices_path=devices_path,
    )

    num_states = int(
        behavioral_config.num_states
    )

    num_devices = int(
        behavioral_config.num_devices
    )

    compatibility_mask = behavioral_config.torch_mask(
        dtype=torch.float32,
        device=device,
        clone=True,
    )

    print(
        f"  Dataset             : "
        f"{behavioral_config.dataset_name}"
    )
    print(
        f"  Number of devices   : {num_devices}"
    )
    print(
        f"  Number of states    : {num_states}"
    )
    print(
        f"  Compatibility shape : "
        f"{tuple(compatibility_mask.shape)}"
    )
    print(
        f"  Compatibility dtype : "
        f"{compatibility_mask.dtype}"
    )
    print()

    # -------------------------------------------------------------
    # Validate compatibility matrix once
    # -------------------------------------------------------------

    expected_mask_shape = (
        num_states,
        num_devices,
    )

    if tuple(compatibility_mask.shape) != expected_mask_shape:
        raise RuntimeError(
            "Behavioral compatibility matrix shape mismatch: "
            f"expected {expected_mask_shape}, "
            f"received {tuple(compatibility_mask.shape)}."
        )

    if not torch.all(
        (compatibility_mask == 0)
        | (compatibility_mask == 1)
    ):
        raise RuntimeError(
            "Behavioral compatibility mask is not binary."
        )

    if not torch.all(
        compatibility_mask.sum(dim=1) > 0
    ):
        raise RuntimeError(
            "At least one behavioral state has no "
            "compatible devices."
        )

    if not torch.all(
        compatibility_mask.sum(dim=0) == 1
    ):
        raise RuntimeError(
            "Behavioral compatibility matrix violates "
            "the Version 1.0 primary-state partition."
        )

    print(
        "[PASS] Behavioral-state compatibility configuration validated."
    )
    print()

    # -------------------------------------------------------------
    # Establish embedding dimension from the first DBRL artifact
    # -------------------------------------------------------------

    first_payload = load_dbrl_representation_shard(
        shard_path=shard_paths[0],
        expected_split=split,
        expected_num_devices=num_devices,
        expected_embedding_dim=None,
    )

    embedding_dim = int(
        first_payload["representations"].shape[2]
    )

    print(
        f"DBRL embedding dimension : {embedding_dim}"
    )

    # The first artifact is only inspected to establish D. It will be
    # loaded again by the normal sequential processing loop below so
    # that every shard follows exactly the same path and verification.
    del first_payload
    gc.collect()

    # -------------------------------------------------------------
    # Build frozen BSE
    # -------------------------------------------------------------

    model = build_bse(
        num_states=num_states,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
    ).to(device)

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("BSE model constructed successfully.")
    print(f"  K                   : {num_states}")
    print(f"  N                   : {num_devices}")
    print(f"  Embedding dim       : {embedding_dim}")
    print(f"  Attention heads     : {num_heads}")
    print(f"  Trainable parameters: {parameter_count}")
    print()

    # -------------------------------------------------------------
    # Process DBRL representation shards sequentially
    # -------------------------------------------------------------

    total_processed = 0
    processed_shards = 0
    expected_next_start = 0

    for sequence_index, shard_path in enumerate(shard_paths):
        print("=" * 70)
        print(
            f"BSE shard {sequence_index + 1}/"
            f"{len(shard_paths)}"
        )
        print("=" * 70)
        print(
            f"Source DBRL shard : {shard_path}"
        )

        # ---------------------------------------------------------
        # Load one frozen DBRL representation artifact
        # ---------------------------------------------------------

        payload = load_dbrl_representation_shard(
            shard_path=shard_path,
            expected_split=split,
            expected_num_devices=num_devices,
            expected_embedding_dim=embedding_dim,
        )

        Z = payload["representations"]

        shard_index = int(
            payload["shard_index"]
        )

        start_index = int(
            payload["start_index"]
        )

        end_index = int(
            payload["end_index"]
        )

        shard_samples = Z.shape[0]

        # ---------------------------------------------------------
        # Cross-shard sample alignment
        # ---------------------------------------------------------

        if start_index != expected_next_start:
            raise RuntimeError(
                "DBRL representation shards are not contiguous "
                "in sample order: "
                f"expected start_index={expected_next_start}, "
                f"received {start_index} for shard {shard_index}."
            )

        if end_index - start_index != shard_samples:
            raise RuntimeError(
                "DBRL shard sample-range mismatch: "
                f"[{start_index}, {end_index}) contains "
                f"{end_index - start_index} samples, "
                f"but Z contains {shard_samples}."
            )

        print(f"Shard index        : {shard_index}")
        print(
            f"Sample range       : "
            f"[{start_index}, {end_index})"
        )
        print(
            f"Z shape            : {tuple(Z.shape)}"
        )
        print(f"Z dtype            : {Z.dtype}")

        # ---------------------------------------------------------
        # Run BSE
        # ---------------------------------------------------------

        print()
        print("Running BSE on shard...")

        S = run_bse_on_shard(
            model=model,
            Z=Z,
            compatibility_mask=compatibility_mask,
            device=device,
            batch_size=batch_size,
        )

        # ---------------------------------------------------------
        # Verify BSE output
        # ---------------------------------------------------------

        verify_bse_output_shard(
            S,
            expected_samples=shard_samples,
            expected_num_states=num_states,
            expected_embedding_dim=embedding_dim,
        )

        print(
            f"BSE output shape   : {tuple(S.shape)}"
        )
        print("[PASS] BSE output shape verified.")
        print("[PASS] BSE output dtype verified.")
        print("[PASS] BSE output finiteness verified.")

        # ---------------------------------------------------------
        # Save BSE representation shard
        # ---------------------------------------------------------

        output_path = (
            bse_split_dir
            / f"shard_{shard_index:06d}.pt"
        )

        save_bse_representation_shard(
            output_path=output_path,
            S=S,
            source_dbrl_payload=payload,
            num_states=num_states,
            num_devices=num_devices,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            seed=int(config["seed"]),
            overwrite=overwrite,
        )

        print()
        print("[PASS] BSE output saved to:")
        print(f"       {output_path}")

        # ---------------------------------------------------------
        # Update coverage
        # ---------------------------------------------------------

        total_processed += shard_samples
        processed_shards += 1
        expected_next_start = end_index

        # ---------------------------------------------------------
        # Release shard memory
        # ---------------------------------------------------------

        del S
        del Z
        del payload

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print()
        print(
            f"Processed samples so far: {total_processed}"
        )

    # -------------------------------------------------------------
    # Final execution summary
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("BSE sharded execution completed")
    print("=" * 70)
    print(f"Processed shards   : {processed_shards}")
    print(f"Processed samples  : {total_processed}")
    print(f"Discovered shards  : {total_dbrl_shards}")
    print(f"BSE output dir     : {bse_split_dir}")

    # -------------------------------------------------------------
    # Complete-run verification
    # -------------------------------------------------------------

    if max_shards is None:
        # Since the DBRL artifact layer has no manifest, the complete
        # expected sample count is established by the final DBRL shard's
        # end_index. We have already verified contiguous [start, end)
        # coverage from zero, so this is the authoritative total.
        if processed_shards != total_dbrl_shards:
            raise RuntimeError(
                "Complete BSE execution did not process all "
                "discovered DBRL representation shards: "
                f"expected {total_dbrl_shards}, "
                f"received {processed_shards}."
            )

        if total_processed != expected_next_start:
            raise RuntimeError(
                "Complete BSE sample coverage is internally "
                "inconsistent: "
                f"processed={total_processed}, "
                f"final_end_index={expected_next_start}."
            )

        print(
            "[PASS] Complete DBRL representation split "
            "processed exactly once."
        )

    else:
        print(
            "[INFO] Partial BSE execution requested; "
            "complete-split coverage was not required."
        )

    print()
    print(
        "[PASS] BSE sharded standalone execution finished."
    )


if __name__ == "__main__":
    main()