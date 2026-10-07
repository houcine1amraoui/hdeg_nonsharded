from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from src.models.hdeg.bil import BehavioralInteractionLearner
from src.utils.device import get_device
from src.utils.get_folders_utils import get_processed_folder
from src.utils.seed import set_seed


SPLITS = (
    "train",
    "val",
    "actor2_test",
    "actor1_test",
)

BSE_SHARD_PATTERN = "shard_*.pt"


# ---------------------------------------------------------------------
# BSE representation shard discovery
# ---------------------------------------------------------------------

def discover_bse_representation_shards(
    bse_split_dir: Path,
) -> list[Path]:
    """
    Discover persisted BSE representation shards directly.

    BSE does not require a separate downstream manifest. The persisted
    artifact filenames are the authoritative shard list.
    """
    if not bse_split_dir.is_dir():
        raise FileNotFoundError(
            "BSE representation split directory not found:\n"
            f"{bse_split_dir}"
        )

    shard_paths = sorted(
        bse_split_dir.glob(BSE_SHARD_PATTERN)
    )

    if not shard_paths:
        raise FileNotFoundError(
            "No BSE representation shards were found in:\n"
            f"{bse_split_dir}\n"
            f"Expected files matching: {BSE_SHARD_PATTERN}"
        )

    return shard_paths


def parse_bse_shard_filename(
    shard_path: Path,
) -> int:
    """
    Extract the zero-based shard index from:

        shard_000000.pt
    """
    if shard_path.suffix != ".pt":
        raise ValueError(
            f"Invalid BSE representation shard suffix: "
            f"{shard_path.name}"
        )

    stem = shard_path.stem

    if not stem.startswith("shard_"):
        raise ValueError(
            f"Invalid BSE representation shard filename: "
            f"{shard_path.name}"
        )

    index_text = stem[len("shard_"):]

    if not index_text.isdigit():
        raise ValueError(
            f"Invalid BSE representation shard filename: "
            f"{shard_path.name}"
        )

    shard_index = int(index_text)

    if shard_index < 0:
        raise ValueError(
            f"Invalid negative BSE shard index in "
            f"{shard_path.name}."
        )

    return shard_index


def validate_bse_shard_sequence(
    shard_paths: list[Path],
) -> None:
    """
    Verify that discovered BSE shard filenames form a contiguous
    zero-based sequence.

    This prevents missing or duplicated upstream artifacts from being
    silently skipped by BIL.
    """
    if not shard_paths:
        raise ValueError(
            "BSE representation shard list is empty."
        )

    actual_indices = [
        parse_bse_shard_filename(path)
        for path in shard_paths
    ]

    if len(set(actual_indices)) != len(actual_indices):
        raise RuntimeError(
            "Duplicate BSE shard indices were discovered: "
            f"{actual_indices}."
        )

    expected_indices = list(
        range(len(shard_paths))
    )

    if actual_indices != expected_indices:
        raise RuntimeError(
            "BSE representation shards do not form a contiguous "
            "zero-based sequence. "
            f"Expected {expected_indices[:10]}"
            f"{'...' if len(expected_indices) > 10 else ''}, "
            f"received {actual_indices[:10]}"
            f"{'...' if len(actual_indices) > 10 else ''}."
        )


# ---------------------------------------------------------------------
# BSE representation shard loading
# ---------------------------------------------------------------------

def load_bse_representation_shard(
    shard_path: Path,
    *,
    expected_split: str,
    expected_num_states: Optional[int],
    expected_embedding_dim: Optional[int],
) -> dict[str, Any]:
    """
    Load and validate exactly one BSE representation artifact.

    The artifact remains on CPU. Only individual mini-batches are moved
    to the execution device.
    """
    if not shard_path.is_file():
        raise FileNotFoundError(
            f"BSE representation shard not found:\n"
            f"{shard_path}"
        )

    payload = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise TypeError(
            f"BSE shard {shard_path} must contain "
            "a dictionary payload."
        )

    required = (
        "representations",
        "split",
        "source_dbrl_shard",
        "source_dbrl_shard_index",
        "shard_index",
        "start_index",
        "end_index",
        "window_size",
        "num_devices",
        "num_states",
        "embedding_dim",
    )

    missing = [
        key
        for key in required
        if key not in payload
    ]

    if missing:
        raise KeyError(
            f"BSE shard {shard_path} is missing "
            f"required entries: {missing}"
        )

    if payload["split"] != expected_split:
        raise ValueError(
            f"BSE shard split mismatch in {shard_path}: "
            f"expected '{expected_split}', "
            f"received '{payload['split']}'."
        )

    S = payload["representations"]

    if not isinstance(S, torch.Tensor):
        raise TypeError(
            f"'representations' in {shard_path} must be "
            "a torch.Tensor."
        )

    if S.ndim != 3:
        raise ValueError(
            f"BSE representations in {shard_path} must have "
            "shape (S, K, D). "
            f"Received {tuple(S.shape)}."
        )

    num_samples = int(S.shape[0])
    num_states = int(S.shape[1])
    embedding_dim = int(S.shape[2])

    if num_samples <= 0:
        raise ValueError(
            f"BSE representation shard {shard_path} is empty."
        )

    if S.dtype != torch.float32:
        raise TypeError(
            f"BSE representations in {shard_path} must be "
            f"float32. Received {S.dtype}."
        )

    if not torch.isfinite(S).all():
        raise ValueError(
            f"BSE representations in {shard_path} contain "
            "NaN or infinite values."
        )

    metadata_num_states = int(
        payload["num_states"]
    )
    metadata_embedding_dim = int(
        payload["embedding_dim"]
    )
    metadata_num_devices = int(
        payload["num_devices"]
    )

    if metadata_num_states != num_states:
        raise ValueError(
            f"BSE state-count metadata mismatch in {shard_path}: "
            f"tensor has K={num_states}, "
            f"metadata reports K={metadata_num_states}."
        )

    if metadata_embedding_dim != embedding_dim:
        raise ValueError(
            f"BSE embedding-dimension metadata mismatch in "
            f"{shard_path}: tensor has D={embedding_dim}, "
            f"metadata reports D={metadata_embedding_dim}."
        )

    if expected_num_states is not None and (
        num_states != expected_num_states
    ):
        raise ValueError(
            f"BSE state-count mismatch in {shard_path}: "
            f"expected K={expected_num_states}, "
            f"received K={num_states}."
        )

    if expected_embedding_dim is not None and (
        embedding_dim != expected_embedding_dim
    ):
        raise ValueError(
            f"BSE embedding-dimension mismatch in {shard_path}: "
            f"expected D={expected_embedding_dim}, "
            f"received D={embedding_dim}."
        )

    if metadata_num_devices <= 0:
        raise ValueError(
            f"Invalid num_devices={metadata_num_devices} "
            f"in {shard_path}."
        )

    shard_index = int(
        payload["shard_index"]
    )
    filename_index = parse_bse_shard_filename(
        shard_path
    )

    if shard_index != filename_index:
        raise ValueError(
            f"BSE shard-index mismatch in {shard_path}: "
            f"filename encodes index {filename_index}, "
            f"payload reports {shard_index}."
        )

    source_dbrl_shard_index = int(
        payload["source_dbrl_shard_index"]
    )

    if source_dbrl_shard_index != shard_index:
        raise ValueError(
            f"BSE/DBRL shard-index provenance mismatch in "
            f"{shard_path}: BSE shard_index={shard_index}, "
            f"source_dbrl_shard_index={source_dbrl_shard_index}."
        )

    start_index = int(
        payload["start_index"]
    )
    end_index = int(
        payload["end_index"]
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
            f"BSE shard sample-range mismatch in {shard_path}: "
            f"range contains {expected_sample_count} samples, "
            f"but representations contain {num_samples}."
        )

    return payload


# ---------------------------------------------------------------------
# BIL construction
# ---------------------------------------------------------------------

def build_bil(
    *,
    num_states: int,
    embedding_dim: int,
) -> BehavioralInteractionLearner:
    """
    Construct the frozen BIL module.

    The BIG is internal to BIL and is fixed by the frozen implementation.
    """
    if num_states != 9:
        raise ValueError(
            "The frozen CU BIL implementation expects K=9. "
            f"Received K={num_states}."
        )

    if embedding_dim <= 0:
        raise ValueError(
            "embedding_dim must be greater than zero."
        )

    return BehavioralInteractionLearner(
        num_states=num_states,
        embedding_dim=embedding_dim,
    )


# ---------------------------------------------------------------------
# BIL forward pass on one shard
# ---------------------------------------------------------------------

def run_bil_on_shard(
    model: BehavioralInteractionLearner,
    S: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    Run BIL on one BSE representation shard.

    Input:
        S: (S, K, D)

    Output:
        S_tilde: (S, K, D)

    Only one mini-batch is resident on the execution device at a time.
    The output tensor is allocated once on CPU for this shard.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if S.ndim != 3:
        raise ValueError(
            "BIL input must have shape (S, K, D). "
            f"Received {tuple(S.shape)}."
        )

    if S.dtype != torch.float32:
        raise TypeError(
            "BIL input must be float32."
        )

    if S.shape[1] != model.num_states:
        raise ValueError(
            "BIL state-count mismatch: "
            f"model has K={model.num_states}, "
            f"input has K={S.shape[1]}."
        )

    if S.shape[2] != model.embedding_dim:
        raise ValueError(
            "BIL embedding-dimension mismatch: "
            f"model expects D={model.embedding_dim}, "
            f"input has D={S.shape[2]}."
        )

    if not torch.isfinite(S).all():
        raise ValueError(
            "BIL input contains NaN or infinite values."
        )

    model.eval()

    output = torch.empty_like(
        S,
        device="cpu",
    )

    with torch.inference_mode():
        for start in range(
            0,
            S.shape[0],
            batch_size,
        ):
            end = min(
                start + batch_size,
                S.shape[0],
            )

            S_batch = S[start:end].to(
                device,
                non_blocking=False,
            )

            output_batch = model(
                S_batch
            )

            expected_shape = (
                end - start,
                model.num_states,
                model.embedding_dim,
            )

            if tuple(output_batch.shape) != expected_shape:
                raise RuntimeError(
                    "BIL output shape mismatch: "
                    f"expected {expected_shape}, "
                    f"received {tuple(output_batch.shape)}."
                )

            if output_batch.dtype != torch.float32:
                raise RuntimeError(
                    "BIL output dtype mismatch: "
                    f"expected torch.float32, "
                    f"received {output_batch.dtype}."
                )

            if not torch.isfinite(
                output_batch
            ).all():
                raise RuntimeError(
                    "BIL produced NaN or infinite "
                    "contextualized representations."
                )

            output[start:end].copy_(
                output_batch.detach().cpu()
            )

            del S_batch
            del output_batch

    return output


# ---------------------------------------------------------------------
# BIL representation artifact persistence
# ---------------------------------------------------------------------

def save_bil_representation_shard(
    output_path: Path,
    S_tilde: torch.Tensor,
    *,
    source_bse_payload: dict[str, Any],
    num_states: int,
    num_devices: int,
    embedding_dim: int,
    seed: int,
    negative_slope: float,
    overwrite: bool,
) -> None:
    """
    Persist one BIL scientific representation shard.

    Upstream BSE/DBRL provenance and sample-range alignment are retained.
    The scientific artifact is the contextualized representation tensor.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"BIL output already exists:\n"
            f"{output_path}\n\n"
            "Use --overwrite or enable overwrite "
            "to replace it."
        )

    if S_tilde.ndim != 3:
        raise ValueError(
            "BIL representation tensor must have "
            "shape (S, K, D)."
        )

    if S_tilde.dtype != torch.float32:
        raise TypeError(
            "BIL representation tensor must be float32."
        )

    if S_tilde.shape[1] != num_states:
        raise ValueError(
            "BIL representation state dimension mismatch: "
            f"expected K={num_states}, "
            f"received {S_tilde.shape[1]}."
        )

    if S_tilde.shape[2] != embedding_dim:
        raise ValueError(
            "BIL representation embedding dimension mismatch: "
            f"expected D={embedding_dim}, "
            f"received {S_tilde.shape[2]}."
        )

    if not torch.isfinite(S_tilde).all():
        raise ValueError(
            "BIL representation tensor contains "
            "NaN or infinite values."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        # Scientific artifact
        "representations": S_tilde,

        # Current split
        "split": source_bse_payload["split"],

        # Immediate upstream provenance
        "source_bse_shard": source_bse_payload[
            "source_bse_shard"
        ] if "source_bse_shard" in source_bse_payload
        else output_path.name,
        "source_bse_shard_index": int(
            source_bse_payload["shard_index"]
        ),

        # Retained DBRL provenance
        "source_dbrl_shard": source_bse_payload[
            "source_dbrl_shard"
        ],
        "source_dbrl_shard_index": int(
            source_bse_payload[
                "source_dbrl_shard_index"
            ]
        ),

        # Current shard identity
        "shard_index": int(
            source_bse_payload["shard_index"]
        ),

        # Sample alignment
        "start_index": int(
            source_bse_payload["start_index"]
        ),
        "end_index": int(
            source_bse_payload["end_index"]
        ),

        # Structural metadata
        "window_size": int(
            source_bse_payload["window_size"]
        ),
        "num_devices": int(num_devices),
        "num_states": int(num_states),
        "embedding_dim": int(embedding_dim),

        # BIL implementation provenance
        "bil_negative_slope": float(
            negative_slope
        ),
        "seed": int(seed),
    }

    # Preserve the BSE execution metadata when present without allowing it
    # to overwrite BIL's authoritative fields.
    if "num_heads" in source_bse_payload:
        payload["source_bse_num_heads"] = int(
            source_bse_payload["num_heads"]
        )

    if "seed" in source_bse_payload:
        payload["source_bse_seed"] = int(
            source_bse_payload["seed"]
        )

    torch.save(
        payload,
        output_path,
    )


# ---------------------------------------------------------------------
# BIL output verification
# ---------------------------------------------------------------------

def verify_bil_output_shard(
    S_tilde: torch.Tensor,
    *,
    expected_samples: int,
    expected_num_states: int,
    expected_embedding_dim: int,
) -> None:
    """
    Verify the scientific BIL representation produced for one shard.
    """
    expected_shape = (
        expected_samples,
        expected_num_states,
        expected_embedding_dim,
    )

    if tuple(S_tilde.shape) != expected_shape:
        raise RuntimeError(
            "BIL representation shape mismatch: "
            f"expected {expected_shape}, "
            f"received {tuple(S_tilde.shape)}."
        )

    if S_tilde.dtype != torch.float32:
        raise RuntimeError(
            "BIL representation dtype mismatch: "
            f"expected torch.float32, "
            f"received {S_tilde.dtype}."
        )

    if not torch.isfinite(
        S_tilde
    ).all():
        raise RuntimeError(
            "BIL representation contains NaN or infinite values."
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

    batch_size = config["hdeg"]["bil"]["batch_size"]
    max_shards = config["hdeg"]["bil"]["max_shards"]
    overwrite = config["hdeg"]["bil"]["overwrite"]

    # -------------------------------------------------------------
    # Processed-data paths
    # -------------------------------------------------------------

    processed_data_folder = Path(
        get_processed_folder(config)
    )

    bse_dir = (
        Path(f"{processed_data_folder}/bse")
    )

    bse_split_dir = (
        bse_dir
        / args.split
    )

    bil_dir = (
        Path(f"{processed_data_folder}/bil")
    )

    bil_split_dir = (
        bil_dir
        / args.split
    )

    # -------------------------------------------------------------
    # Header
    # -------------------------------------------------------------

    print("=" * 70)
    print(
        "HDEG — BIL Sharded Standalone Execution"
    )
    print("=" * 70)
    print(
        f"Split                : {split}"
    )
    print(
        f"BSE input directory  : {bse_split_dir}"
    )
    print(
        f"BIL output directory : {bil_split_dir}"
    )
    print(
        f"Device               : {device}"
    )
    print(
        f"BIL batch size       : {batch_size}"
    )
    print(
        f"Max shards           : {max_shards}"
    )
    print(
        f"Overwrite            : {overwrite}"
    )
    print()

    # -------------------------------------------------------------
    # Discover and validate BSE artifacts
    # -------------------------------------------------------------

    bse_shard_paths = (
        discover_bse_representation_shards(
            bse_split_dir
        )
    )

    validate_bse_shard_sequence(
        bse_shard_paths
    )

    total_bse_shards = len(
        bse_shard_paths
    )

    if max_shards is not None:
        shard_paths = (
            bse_shard_paths[:max_shards]
        )
    else:
        shard_paths = bse_shard_paths

    print(
        f"Discovered BSE shards : "
        f"{total_bse_shards}"
    )
    print(
        f"Shards to process     : "
        f"{len(shard_paths)}"
    )
    print()

    # -------------------------------------------------------------
    # Establish K and D from the first BSE artifact
    # -------------------------------------------------------------

    first_payload = (
        load_bse_representation_shard(
            shard_path=shard_paths[0],
            expected_split=split,
            expected_num_states=9,
            expected_embedding_dim=None,
        )
    )

    num_states = int(
        first_payload["representations"].shape[1]
    )

    embedding_dim = int(
        first_payload["representations"].shape[2]
    )

    num_devices = int(
        first_payload["num_devices"]
    )

    print(
        f"Behavioral states K   : {num_states}"
    )
    print(
        f"Embedding dimension D  : {embedding_dim}"
    )
    print(
        f"Upstream devices N     : {num_devices}"
    )

    del first_payload
    gc.collect()

    # -------------------------------------------------------------
    # Build frozen BIL
    # -------------------------------------------------------------

    model = build_bil(
        num_states=num_states,
        embedding_dim=embedding_dim,
    ).to(device)

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        "BIL model constructed successfully."
    )
    print(
        f"  K                   : {num_states}"
    )
    print(
        f"  Embedding dim       : {embedding_dim}"
    )
    print(
        f"  BIG edges           : 7"
    )
    print(
        f"  Trainable parameters: {parameter_count}"
    )
    print()

    # -------------------------------------------------------------
    # Process BSE representation shards sequentially
    # -------------------------------------------------------------

    total_processed = 0
    processed_shards = 0
    expected_next_start = 0

    for sequence_index, shard_path in enumerate(
        shard_paths
    ):
        print("=" * 70)
        print(
            f"BIL shard {sequence_index + 1}/"
            f"{len(shard_paths)}"
        )
        print("=" * 70)
        print(
            f"Source BSE shard : {shard_path}"
        )

        # ---------------------------------------------------------
        # Load one BSE scientific artifact
        # ---------------------------------------------------------

        payload = (
            load_bse_representation_shard(
                shard_path=shard_path,
                expected_split=split,
                expected_num_states=num_states,
                expected_embedding_dim=embedding_dim,
            )
        )

        S = payload["representations"]

        shard_index = int(
            payload["shard_index"]
        )

        start_index = int(
            payload["start_index"]
        )

        end_index = int(
            payload["end_index"]
        )

        shard_samples = int(
            S.shape[0]
        )

        # ---------------------------------------------------------
        # Cross-shard sample alignment
        # ---------------------------------------------------------

        if start_index != expected_next_start:
            raise RuntimeError(
                "BSE representation shards are not contiguous "
                "in sample order: "
                f"expected start_index={expected_next_start}, "
                f"received {start_index} for shard "
                f"{shard_index}."
            )

        if end_index - start_index != shard_samples:
            raise RuntimeError(
                "BSE shard sample-range mismatch: "
                f"[{start_index}, {end_index}) contains "
                f"{end_index - start_index} samples, "
                f"but representations contain "
                f"{shard_samples}."
            )

        print(
            f"Shard index         : {shard_index}"
        )
        print(
            f"Sample range        : "
            f"[{start_index}, {end_index})"
        )
        print(
            f"BSE shape           : "
            f"{tuple(S.shape)}"
        )
        print(
            f"BSE dtype           : {S.dtype}"
        )

        # ---------------------------------------------------------
        # Run BIL
        # ---------------------------------------------------------

        print()
        print(
            "Running BIL on shard..."
        )

        S_tilde = run_bil_on_shard(
            model=model,
            S=S,
            device=device,
            batch_size=batch_size,
        )

        # ---------------------------------------------------------
        # Verify BIL output
        # ---------------------------------------------------------

        verify_bil_output_shard(
            S_tilde,
            expected_samples=shard_samples,
            expected_num_states=num_states,
            expected_embedding_dim=embedding_dim,
        )

        print(
            f"BIL output shape    : "
            f"{tuple(S_tilde.shape)}"
        )
        print(
            "[PASS] BIL output shape verified."
        )
        print(
            "[PASS] BIL output dtype verified."
        )
        print(
            "[PASS] BIL output finiteness verified."
        )

        # ---------------------------------------------------------
        # Save BIL representation shard
        # ---------------------------------------------------------

        output_path = (
            bil_split_dir
            / f"shard_{shard_index:06d}.pt"
        )

        # The immediate upstream artifact is this exact filename.
        payload["source_bse_shard"] = (
            shard_path.name
        )

        save_bil_representation_shard(
            output_path=output_path,
            S_tilde=S_tilde,
            source_bse_payload=payload,
            num_states=num_states,
            num_devices=num_devices,
            embedding_dim=embedding_dim,
            seed=int(config["seed"]),
            negative_slope=float(
                model.negative_slope
            ),
            overwrite=overwrite,
        )

        print()
        print(
            "[PASS] BIL output saved to:"
        )
        print(
            f"       {output_path}"
        )

        # ---------------------------------------------------------
        # Update coverage
        # ---------------------------------------------------------

        total_processed += shard_samples
        processed_shards += 1
        expected_next_start = end_index

        # ---------------------------------------------------------
        # Release shard memory
        # ---------------------------------------------------------

        del S_tilde
        del S
        del payload

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print()
        print(
            f"Processed samples so far: "
            f"{total_processed}"
        )

    # -------------------------------------------------------------
    # Final execution summary
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "BIL sharded execution completed"
    )
    print("=" * 70)
    print(
        f"Processed shards   : "
        f"{processed_shards}"
    )
    print(
        f"Processed samples  : "
        f"{total_processed}"
    )
    print(
        f"Discovered shards  : "
        f"{total_bse_shards}"
    )
    print(
        f"BIL output dir     : "
        f"{bil_split_dir}"
    )

    # -------------------------------------------------------------
    # Complete-run verification
    # -------------------------------------------------------------

    if max_shards is None:
        if processed_shards != total_bse_shards:
            raise RuntimeError(
                "Complete BIL execution did not process all "
                "discovered BSE representation shards: "
                f"expected {total_bse_shards}, "
                f"received {processed_shards}."
            )

        if total_processed != expected_next_start:
            raise RuntimeError(
                "Complete BIL sample coverage is internally "
                "inconsistent: "
                f"processed={total_processed}, "
                f"final_end_index={expected_next_start}."
            )

        print(
            "[PASS] Complete BSE representation split "
            "processed exactly once."
        )
    else:
        print(
            "[INFO] Partial BIL execution requested; "
            "complete-split coverage was not required."
        )

    print()
    print(
        "[PASS] BIL sharded standalone execution finished."
    )


if __name__ == "__main__":
    main()