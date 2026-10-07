from __future__ import annotations

import argparse
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

def load_dbrl_representations(
    input_path: Path,
    *,
    expected_split: str,
    expected_num_devices: int,
    expected_embedding_dim: Optional[int],
) -> dict[str, Any]:
    """
    Load exactly one persisted DBRL representation file.

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

    if not input_path.is_file():
        raise FileNotFoundError(
            f"DBRL representation file not found:\n"
            f"{input_path}"
        )

    payload = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise TypeError(
            f"DBRL file {input_path} must contain "
            "a dictionary payload."
        )

    required = (
        "representations",
        "split",
        "source_windows",
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
            f"DBRL file {input_path} is missing "
            f"required entries: {missing}"
        )

    # -------------------------------------------------------------
    # Split
    # -------------------------------------------------------------

    if payload["split"] != expected_split:
        raise ValueError(
            f"DBRL file split mismatch in {input_path}: "
            f"expected '{expected_split}', "
            f"received '{payload['split']}'."
        )

    # -------------------------------------------------------------
    # Representation tensor
    # -------------------------------------------------------------

    Z = payload["representations"]

    if not isinstance(Z, torch.Tensor):
        raise TypeError(
            f"'representations' in {input_path} must be "
            "a torch.Tensor."
        )

    if Z.ndim != 3:
        raise ValueError(
            f"DBRL representations in {input_path} must have "
            "shape (S, N, D). "
            f"Received {tuple(Z.shape)}."
        )

    num_samples = Z.shape[0]
    num_devices = Z.shape[1]
    embedding_dim = Z.shape[2]

    if num_samples <= 0:
        raise ValueError(
            f"DBRL representation file {input_path} is empty."
        )

    expected_shape = (
        num_samples,
        expected_num_devices,
        embedding_dim,
    )

    if tuple(Z.shape) != expected_shape:
        raise ValueError(
            f"Unexpected DBRL representation shape in "
            f"{input_path}: expected device dimension "
            f"N={expected_num_devices}, received {tuple(Z.shape)}."
        )

    if Z.dtype != torch.float32:
        raise TypeError(
            f"DBRL representations in {input_path} must be "
            f"float32. Received {Z.dtype}."
        )

    if not torch.isfinite(Z).all():
        raise ValueError(
            f"DBRL representations in {input_path} contain "
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
            f"{input_path}: expected {expected_num_devices}, "
            f"received {metadata_num_devices}."
        )

    if metadata_embedding_dim != embedding_dim:
        raise ValueError(
            f"DBRL metadata embedding-dimension mismatch in "
            f"{input_path}: tensor has D={embedding_dim}, "
            f"metadata reports D={metadata_embedding_dim}."
        )

    if expected_embedding_dim is not None:
        if metadata_embedding_dim != expected_embedding_dim:
            raise ValueError(
                f"DBRL metadata embedding-dimension mismatch "
                f"in {input_path}: expected {expected_embedding_dim}, "
                f"received {metadata_embedding_dim}."
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
            f"DBRL file sample-range mismatch in "
            f"{input_path}: range contains "
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

def run_bse_on_split(
    model: BehavioralStateEstimator,
    Z: torch.Tensor,
    compatibility_mask: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    Run BSE on one DBRL representation split.

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

def save_bse_representations(
    output_path: Path,
    S: torch.Tensor,
    *,
    source_dbrl_payload: dict[str, Any],
    source_dbrl_path: Path,
    num_states: int,
    num_devices: int,
    embedding_dim: int,
    num_heads: int,
    seed: int,
    overwrite: bool,
) -> None:
    """
    Persist one BSE representation file.

    DBRL provenance is retained so that the BSE artifact can be traced
    to the exact DBRL file and therefore to the corresponding window
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
        "source_dbrl": str(source_dbrl_path),
        "source_windows": source_dbrl_payload["source_windows"],

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

def verify_bse_output(
    S: torch.Tensor,
    *,
    expected_samples: int,
    expected_num_states: int,
    expected_embedding_dim: int,
) -> None:
    """Verify the scientific BSE representation produced for a file."""

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
    root = config["project_root_dir"]
    # Retain the batch-size setting used by the supplied BSE runner.
    batch_size = int(config["hdeg"]["dbrl"]["batch_size"])
    num_heads = int(config["hdeg"]["bse"]["num_heads"])
    overwrite = bool(config["hdeg"]["bse"].get("overwrite", False))
    if batch_size <= 0 or num_heads <= 0:
        raise ValueError("batch_size and num_heads must be greater than zero.")

    processed_data_folder = Path(get_processed_folder(config))
    input_path = processed_data_folder / "dbrl" / f"{split}.pt"
    output_path = processed_data_folder / "bse" / f"{split}.pt"
    if not input_path.is_file():
        raise FileNotFoundError(f"DBRL representation file not found:\n{input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"BSE output already exists:\n{output_path}")

    behavioral_config_path = Path(root) / "configs" / "hdeg" / "behavioral_states.yaml"
    devices_path = processed_data_folder / "devices.json"

    print("=" * 70)
    print("HDEG — BSE Non-sharded Standalone Execution")
    print("=" * 70)
    print(f"Split           : {split}")
    print(f"DBRL input file : {input_path}")
    print(f"BSE output file : {output_path}")
    print(f"Device          : {device}")
    print(f"Batch size      : {batch_size}")

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


    payload = load_dbrl_representations(
        input_path=input_path,
        expected_split=split,
        expected_num_devices=num_devices,
        expected_embedding_dim=None,
    )
    Z = payload["representations"]
    embedding_dim = int(Z.shape[2])

    model = build_bse(
        num_states=num_states,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
    ).to(device)

    S = run_bse_on_split(
        model=model,
        Z=Z,
        compatibility_mask=compatibility_mask,
        device=device,
        batch_size=batch_size,
    )
    verify_bse_output(
        S,
        expected_samples=Z.shape[0],
        expected_num_states=num_states,
        expected_embedding_dim=embedding_dim,
    )
    save_bse_representations(
        output_path=output_path,
        S=S,
        source_dbrl_payload=payload,
        source_dbrl_path=input_path,
        num_states=num_states,
        num_devices=num_devices,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        seed=int(config["seed"]),
        overwrite=overwrite,
    )
    print(f"[PASS] Saved BSE representations with shape {tuple(S.shape)} to {output_path}")
    print(f"[PASS] All {Z.shape[0]} samples in the input DBRL file were processed.")


if __name__ == "__main__":
    main()
