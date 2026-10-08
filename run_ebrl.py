from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from src.models.hdeg.ebrl import EcosystemBehavioralRepresentationLearner

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.get_folders_utils import get_processed_folder

SPLITS = (
    "train",
    "val",
    "actor2_test",
    "actor1_test",
)

def load_bil_representations(
    input_path: Path,
    *,
    expected_split: str,
    expected_num_states: Optional[int],
    expected_embedding_dim: Optional[int],
) -> dict[str, Any]:
    payload = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise TypeError(f"BIL file {input_path} must contain a dictionary payload.")

    required = (
        "representations",
        "split",
        "source_bse",
        "source_dbrl",
        "source_windows",
        "start_index",
        "end_index",
        "window_size",
        "num_devices",
        "num_states",
        "embedding_dim",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"BIL file {input_path} is missing required entries: {missing}")

    if payload["split"] != expected_split:
        raise ValueError(
            f"BIL file split mismatch in {input_path}: expected "
            f"'{expected_split}', received '{payload['split']}'."
        )

    representations = payload["representations"]
    if not isinstance(representations, torch.Tensor):
        raise TypeError(f"'representations' in {input_path} must be a torch.Tensor.")

    if representations.ndim != 3:
        raise ValueError(
            f"BIL representations must have shape (S, K, D); "
            f"received {tuple(representations.shape)}."
        )

    if representations.shape[0] <= 0:
        raise ValueError(f"BIL file {input_path} is empty.")

    if representations.dtype != torch.float32:
        raise TypeError(
            f"BIL representations in {input_path} must be float32; "
            f"received {representations.dtype}."
        )

    if not torch.isfinite(representations).all():
        raise ValueError(f"BIL representations in {input_path} contain NaN or Inf.")

    num_samples, num_states, embedding_dim = map(int, representations.shape)

    if int(payload["num_states"]) != num_states:
        raise ValueError(f"BIL state-count metadata mismatch in {input_path}.")
    if int(payload["embedding_dim"]) != embedding_dim:
        raise ValueError(f"BIL embedding-dimension metadata mismatch in {input_path}.")

    if expected_num_states is not None and num_states != expected_num_states:
        raise ValueError(
            f"Expected K={expected_num_states}, received K={num_states} in {input_path}."
        )
    if expected_embedding_dim is not None and embedding_dim != expected_embedding_dim:
        raise ValueError(
            f"Expected D={expected_embedding_dim}, received D={embedding_dim} in {input_path}."
        )

    start_index = int(payload["start_index"])
    end_index = int(payload["end_index"])
    if start_index < 0 or end_index <= start_index:
        raise ValueError(
            f"Invalid BIL sample range in {input_path}: [{start_index}, {end_index})."
        )

    if end_index - start_index != num_samples:
        raise ValueError(
            f"BIL sample-range mismatch in {input_path}: range contains "
            f"{end_index - start_index}, tensor contains {num_samples}."
        )

    if int(payload["num_devices"]) <= 0:
        raise ValueError(f"Invalid num_devices in {input_path}.")
    if int(payload["window_size"]) <= 0:
        raise ValueError(f"Invalid window_size in {input_path}.")

    return payload


def build_ebrl(*, num_states: int, embedding_dim: int) -> EcosystemBehavioralRepresentationLearner:
    return EcosystemBehavioralRepresentationLearner(
        num_states=num_states,
        embedding_dim=embedding_dim,
    )


def run_ebrl_on_split(
    model: EcosystemBehavioralRepresentationLearner,
    contextualized_states: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if contextualized_states.ndim != 3:
        raise ValueError("EBRL input must have shape (S, K, D).")
    if contextualized_states.dtype != torch.float32:
        raise TypeError("EBRL input must be float32.")
    if contextualized_states.shape[1] != model.num_states:
        raise ValueError("EBRL state-count mismatch.")
    if contextualized_states.shape[2] != model.embedding_dim:
        raise ValueError("EBRL embedding-dimension mismatch.")
    if not torch.isfinite(contextualized_states).all():
        raise ValueError("EBRL input contains NaN or Inf.")

    model = model.to(device)
    model.eval()

    output = torch.empty(
        contextualized_states.shape[0],
        model.embedding_dim,
        dtype=torch.float32,
        device="cpu",
    )

    with torch.inference_mode():
        for start in range(0, contextualized_states.shape[0], batch_size):
            end = min(start + batch_size, contextualized_states.shape[0])
            batch = contextualized_states[start:end].to(device)
            batch_output = model(batch)

            expected_shape = (end - start, model.embedding_dim)
            if tuple(batch_output.shape) != expected_shape:
                raise RuntimeError(
                    f"EBRL output shape mismatch: expected {expected_shape}, "
                    f"received {tuple(batch_output.shape)}."
                )
            if batch_output.dtype != torch.float32:
                raise RuntimeError(
                    f"EBRL output dtype mismatch: expected float32, received {batch_output.dtype}."
                )
            if not torch.isfinite(batch_output).all():
                raise RuntimeError("EBRL produced NaN or Inf.")

            output[start:end].copy_(batch_output.cpu())
            del batch, batch_output

    return output


def save_ebrl_representations(
    output_path: Path,
    ecosystem_representation: torch.Tensor,
    *,
    source_bil_payload: dict[str, Any],
    source_bil_path: Path,
    embedding_dim: int,
    seed: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"EBRL output already exists:\n{output_path}\n\n"
            "Enable hdeg.ebrl.overwrite in configuration to replace it."
        )

    if ecosystem_representation.ndim != 2:
        raise ValueError("EBRL representation tensor must have shape (S, D).")
    if ecosystem_representation.dtype != torch.float32:
        raise TypeError("EBRL representation tensor must be float32.")
    if ecosystem_representation.shape[1] != embedding_dim:
        raise ValueError("EBRL embedding dimension does not match metadata.")
    if not torch.isfinite(ecosystem_representation).all():
        raise ValueError("EBRL representation tensor contains NaN or Inf.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        # Scientific artifact
        "representations": ecosystem_representation,

        # Current split
        "split": source_bil_payload["split"],

        # Immediate and retained upstream file provenance
        "source_bil": str(source_bil_path),
        "source_bse": source_bil_payload["source_bse"],
        "source_dbrl": source_bil_payload["source_dbrl"],
        "source_windows": source_bil_payload["source_windows"],

        # Exact sample alignment
        "start_index": int(source_bil_payload["start_index"]),
        "end_index": int(source_bil_payload["end_index"]),

        # Structural metadata
        "window_size": int(source_bil_payload["window_size"]),
        "num_devices": int(source_bil_payload["num_devices"]),
        "num_states": int(source_bil_payload["num_states"]),
        "embedding_dim": int(embedding_dim),

        # EBRL implementation provenance
        "seed": int(seed),
        "ebrl_attention": "tanh_projection_score_softmax_weighted_sum",
    }

    torch.save(payload, output_path)


def validate_ebrl_output_artifact(
    output_path: Path,
    *,
    source_payload: dict[str, Any],
    source_bil_path: Path,
) -> None:
    payload = torch.load(
        output_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise TypeError(f"EBRL output {output_path} is not a dictionary payload.")

    required = (
        "representations",
        "split",
        "source_bil",
        "source_bse",
        "source_dbrl",
        "source_windows",
        "start_index",
        "end_index",
        "window_size",
        "num_devices",
        "num_states",
        "embedding_dim",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"EBRL output is missing required metadata: {missing}")

    output = payload["representations"]
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise ValueError("EBRL output representations must have shape (S, D).")
    if output.dtype != torch.float32 or not torch.isfinite(output).all():
        raise ValueError("EBRL output representations must be finite float32.")

    if payload["source_bil"] != str(source_bil_path):
        raise ValueError("EBRL source_bil does not match the input file.")
    if int(payload["start_index"]) != int(source_payload["start_index"]):
        raise ValueError("EBRL start_index does not match BIL start_index.")
    if int(payload["end_index"]) != int(source_payload["end_index"]):
        raise ValueError("EBRL end_index does not match BIL end_index.")
    if output.shape[0] != int(payload["end_index"]) - int(payload["start_index"]):
        raise ValueError("EBRL output sample count does not match its range.")

    if output.shape[1] != int(source_payload["embedding_dim"]):
        raise ValueError("EBRL output embedding dimension does not match BIL metadata.")

    for key in (
        "split",
        "window_size",
        "num_devices",
        "num_states",
        "embedding_dim",
        "source_bse",
        "source_dbrl",
        "source_windows",
    ):
        if payload[key] != source_payload[key]:
            raise ValueError(f"EBRL provenance mismatch for '{key}'.")



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
    batch_size = int(config["hdeg"]["ebrl"].get("batch_size", 32))
    overwrite = bool(config["hdeg"]["ebrl"].get("overwrite", False))
    if batch_size <= 0:
        raise ValueError("EBRL batch_size must be greater than zero.")

    processed_root = Path(get_processed_folder(config))
    input_path = processed_root / "bil" / f"{split}.pt"
    output_path = processed_root / "ebrl" / f"{split}.pt"
    if not input_path.is_file():
        raise FileNotFoundError(f"BIL representation file not found:\n{input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"EBRL output already exists:\n{output_path}")

    print("=" * 70)
    print("HDEG — EBRL Non-sharded Standalone Execution")
    print("=" * 70)
    print(f"Split           : {split}")
    print(f"BIL input file  : {input_path}")
    print(f"EBRL output file: {output_path}")
    print(f"Device          : {device}")
    print(f"EBRL batch size : {batch_size}")

    payload = load_bil_representations(
        input_path,
        expected_split=split,
        expected_num_states=9,
        expected_embedding_dim=None,
    )
    S_tilde = payload["representations"]
    num_samples, num_states, embedding_dim = map(int, S_tilde.shape)
    if int(payload["start_index"]) != 0:
        raise ValueError("The BIL split file must begin at start_index=0.")

    model = build_ebrl(
        num_states=num_states,
        embedding_dim=embedding_dim,
    ).to(device)

    ecosystem_representation = run_ebrl_on_split(
        model,
        S_tilde,
        device=device,
        batch_size=batch_size,
    )
    save_ebrl_representations(
        output_path,
        ecosystem_representation,
        source_bil_payload=payload,
        source_bil_path=input_path,
        embedding_dim=embedding_dim,
        seed=seed,
        overwrite=overwrite,
    )
    del ecosystem_representation
    validate_ebrl_output_artifact(
        output_path,
        source_payload=payload,
        source_bil_path=input_path,
    )

    print(f"EBRL output shape: {(num_samples, embedding_dim)}")
    print(f"Processed samples: {num_samples}")
    print(f"[PASS] EBRL output verified and saved to: {output_path}")
    print("[PASS] All samples in the input BIL file processed exactly once.")


if __name__ == "__main__":
    main()
