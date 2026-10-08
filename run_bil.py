from __future__ import annotations

import argparse
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

# ---------------------------------------------------------------------
# BSE representation file loading
# ---------------------------------------------------------------------

def load_bse_representations(
    input_path: Path,
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
    if not input_path.is_file():
        raise FileNotFoundError(
            f"BSE representation file not found:\n"
            f"{input_path}"
        )

    payload = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise TypeError(
            f"BSE file {input_path} must contain "
            "a dictionary payload."
        )

    required = (
        "representations",
        "split",
        "source_dbrl",
        "source_windows",
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
            f"BSE file {input_path} is missing "
            f"required entries: {missing}"
        )

    if payload["split"] != expected_split:
        raise ValueError(
            f"BSE file split mismatch in {input_path}: "
            f"expected '{expected_split}', "
            f"received '{payload['split']}'."
        )

    S = payload["representations"]

    if not isinstance(S, torch.Tensor):
        raise TypeError(
            f"'representations' in {input_path} must be "
            "a torch.Tensor."
        )

    if S.ndim != 3:
        raise ValueError(
            f"BSE representations in {input_path} must have "
            "shape (S, K, D). "
            f"Received {tuple(S.shape)}."
        )

    num_samples = int(S.shape[0])
    num_states = int(S.shape[1])
    embedding_dim = int(S.shape[2])

    if num_samples <= 0:
        raise ValueError(
            f"BSE representation file {input_path} is empty."
        )

    if S.dtype != torch.float32:
        raise TypeError(
            f"BSE representations in {input_path} must be "
            f"float32. Received {S.dtype}."
        )

    if not torch.isfinite(S).all():
        raise ValueError(
            f"BSE representations in {input_path} contain "
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
            f"BSE state-count metadata mismatch in {input_path}: "
            f"tensor has K={num_states}, "
            f"metadata reports K={metadata_num_states}."
        )

    if metadata_embedding_dim != embedding_dim:
        raise ValueError(
            f"BSE embedding-dimension metadata mismatch in "
            f"{input_path}: tensor has D={embedding_dim}, "
            f"metadata reports D={metadata_embedding_dim}."
        )

    if expected_num_states is not None and (
        num_states != expected_num_states
    ):
        raise ValueError(
            f"BSE state-count mismatch in {input_path}: "
            f"expected K={expected_num_states}, "
            f"received K={num_states}."
        )

    if expected_embedding_dim is not None and (
        embedding_dim != expected_embedding_dim
    ):
        raise ValueError(
            f"BSE embedding-dimension mismatch in {input_path}: "
            f"expected D={expected_embedding_dim}, "
            f"received D={embedding_dim}."
        )

    if metadata_num_devices <= 0:
        raise ValueError(
            f"Invalid num_devices={metadata_num_devices} "
            f"in {input_path}."
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
            f"in {input_path}."
        )

    if end_index <= start_index:
        raise ValueError(
            f"Invalid sample range in {input_path}: "
            f"[{start_index}, {end_index})."
        )

    expected_sample_count = (
        end_index - start_index
    )

    if expected_sample_count != num_samples:
        raise ValueError(
            f"BSE file sample-range mismatch in {input_path}: "
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
# BIL forward pass on one file
# ---------------------------------------------------------------------

def run_bil_on_split(
    model: BehavioralInteractionLearner,
    S: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    Run BIL on one BSE representation file.

    Input:
        S: (S, K, D)

    Output:
        S_tilde: (S, K, D)

    Only one mini-batch is resident on the execution device at a time.
    The output tensor is allocated once on CPU for this file.
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

def save_bil_representations(
    output_path: Path,
    S_tilde: torch.Tensor,
    *,
    source_bse_payload: dict[str, Any],
    source_bse_path: Path,
    num_states: int,
    num_devices: int,
    embedding_dim: int,
    seed: int,
    negative_slope: float,
    overwrite: bool,
) -> None:
    """
    Persist one BIL scientific representation file.

    Upstream BSE/DBRL provenance and sample-range alignment are retained.
    The scientific artifact is the contextualized representation tensor.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"BIL output already exists:\n"
            f"{output_path}\n\n"
            "Enable hdeg.bil.overwrite in the configuration "
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

        # Immediate and upstream file provenance
        "source_bse": str(source_bse_path),
        "source_dbrl": source_bse_payload["source_dbrl"],
        "source_windows": source_bse_payload["source_windows"],

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

def verify_bil_output(
    S_tilde: torch.Tensor,
    *,
    expected_samples: int,
    expected_num_states: int,
    expected_embedding_dim: int,
) -> None:
    """
    Verify the scientific BIL representation produced for one file.
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
    batch_size = int(config["hdeg"]["bil"]["batch_size"])
    overwrite = bool(config["hdeg"]["bil"].get("overwrite", False))
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    processed_data_folder = Path(get_processed_folder(config))
    input_path = processed_data_folder / "bse" / f"{split}.pt"
    output_path = processed_data_folder / "bil" / f"{split}.pt"
    if not input_path.is_file():
        raise FileNotFoundError(f"BSE representation file not found:\n{input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"BIL output already exists:\n{output_path}")

    print("=" * 70)
    print("HDEG — BIL Non-sharded Standalone Execution")
    print("=" * 70)
    print(f"Split           : {split}")
    print(f"BSE input file  : {input_path}")
    print(f"BIL output file : {output_path}")
    print(f"Device          : {device}")
    print(f"Batch size      : {batch_size}")

    payload = load_bse_representations(
        input_path=input_path,
        expected_split=split,
        expected_num_states=9,
        expected_embedding_dim=None,
    )
    S = payload["representations"]
    num_samples, num_states, embedding_dim = map(int, S.shape)
    num_devices = int(payload["num_devices"])
    if int(payload["start_index"]) != 0:
        raise ValueError("The BSE split file must begin at start_index=0.")

    model = build_bil(
        num_states=num_states,
        embedding_dim=embedding_dim,
    ).to(device)

    S_tilde = run_bil_on_split(
        model=model,
        S=S,
        device=device,
        batch_size=batch_size,
    )
    verify_bil_output(
        S_tilde,
        expected_samples=num_samples,
        expected_num_states=num_states,
        expected_embedding_dim=embedding_dim,
    )
    save_bil_representations(
        output_path=output_path,
        S_tilde=S_tilde,
        source_bse_payload=payload,
        source_bse_path=input_path,
        num_states=num_states,
        num_devices=num_devices,
        embedding_dim=embedding_dim,
        seed=int(config["seed"]),
        negative_slope=float(model.negative_slope),
        overwrite=overwrite,
    )

    print(f"BIL output shape : {tuple(S_tilde.shape)}")
    print(f"Processed samples: {num_samples}")
    print(f"[PASS] BIL output verified and saved to: {output_path}")
    print("[PASS] All samples in the input BSE file processed exactly once.")


if __name__ == "__main__":
    main()
