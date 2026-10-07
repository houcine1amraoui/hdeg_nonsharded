from __future__ import annotations

import argparse
import gc
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

BIL_SHARD_PATTERN = "shard_*.pt"


def parse_shard_filename(shard_path: Path) -> int:
    if shard_path.suffix != ".pt" or not shard_path.stem.startswith("shard_"):
        raise ValueError(f"Invalid BIL shard filename: {shard_path.name}")

    index_text = shard_path.stem[len("shard_"):]
    if not index_text.isdigit():
        raise ValueError(f"Invalid BIL shard filename: {shard_path.name}")

    return int(index_text)


def discover_bil_representation_shards(bil_split_dir: Path) -> list[Path]:
    if not bil_split_dir.is_dir():
        raise FileNotFoundError(
            "BIL representation split directory not found:\n"
            f"{bil_split_dir}"
        )

    shard_paths = sorted(bil_split_dir.glob(BIL_SHARD_PATTERN))
    if not shard_paths:
        raise FileNotFoundError(
            "No BIL representation shards were found in:\n"
            f"{bil_split_dir}\n"
            f"Expected files matching: {BIL_SHARD_PATTERN}"
        )

    return shard_paths


def validate_shard_sequence(shard_paths: list[Path]) -> None:
    indices = [parse_shard_filename(path) for path in shard_paths]

    if len(set(indices)) != len(indices):
        raise RuntimeError(f"Duplicate BIL shard indices discovered: {indices}.")

    expected = list(range(len(shard_paths)))
    if indices != expected:
        raise RuntimeError(
            "BIL representation shards do not form a contiguous zero-based "
            f"sequence. Expected {expected[:10]}, received {indices[:10]}."
        )


def load_bil_representation_shard(
    shard_path: Path,
    *,
    expected_split: str,
    expected_num_states: Optional[int],
    expected_embedding_dim: Optional[int],
) -> dict[str, Any]:
    payload = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise TypeError(f"BIL shard {shard_path} must contain a dictionary payload.")

    required = (
        "representations",
        "split",
        "source_bse_shard",
        "source_bse_shard_index",
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
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"BIL shard {shard_path} is missing required entries: {missing}")

    if payload["split"] != expected_split:
        raise ValueError(
            f"BIL shard split mismatch in {shard_path}: expected "
            f"'{expected_split}', received '{payload['split']}'."
        )

    representations = payload["representations"]
    if not isinstance(representations, torch.Tensor):
        raise TypeError(f"'representations' in {shard_path} must be a torch.Tensor.")

    if representations.ndim != 3:
        raise ValueError(
            f"BIL representations must have shape (S, K, D); "
            f"received {tuple(representations.shape)}."
        )

    if representations.shape[0] <= 0:
        raise ValueError(f"BIL shard {shard_path} is empty.")

    if representations.dtype != torch.float32:
        raise TypeError(
            f"BIL representations in {shard_path} must be float32; "
            f"received {representations.dtype}."
        )

    if not torch.isfinite(representations).all():
        raise ValueError(f"BIL representations in {shard_path} contain NaN or Inf.")

    num_samples, num_states, embedding_dim = map(int, representations.shape)

    if int(payload["num_states"]) != num_states:
        raise ValueError(f"BIL state-count metadata mismatch in {shard_path}.")
    if int(payload["embedding_dim"]) != embedding_dim:
        raise ValueError(f"BIL embedding-dimension metadata mismatch in {shard_path}.")

    if expected_num_states is not None and num_states != expected_num_states:
        raise ValueError(
            f"Expected K={expected_num_states}, received K={num_states} in {shard_path}."
        )
    if expected_embedding_dim is not None and embedding_dim != expected_embedding_dim:
        raise ValueError(
            f"Expected D={expected_embedding_dim}, received D={embedding_dim} in {shard_path}."
        )

    shard_index = int(payload["shard_index"])
    filename_index = parse_shard_filename(shard_path)
    if shard_index != filename_index:
        raise ValueError(
            f"BIL shard-index mismatch in {shard_path}: filename={filename_index}, "
            f"payload={shard_index}."
        )

    if int(payload["source_bse_shard_index"]) != shard_index:
        raise ValueError(
            f"BIL/BSE shard-index provenance mismatch in {shard_path}."
        )
    if int(payload["source_dbrl_shard_index"]) != shard_index:
        raise ValueError(
            f"BIL/DBRL shard-index provenance mismatch in {shard_path}."
        )

    start_index = int(payload["start_index"])
    end_index = int(payload["end_index"])
    if start_index < 0 or end_index <= start_index:
        raise ValueError(
            f"Invalid BIL sample range in {shard_path}: [{start_index}, {end_index})."
        )

    if end_index - start_index != num_samples:
        raise ValueError(
            f"BIL sample-range mismatch in {shard_path}: range contains "
            f"{end_index - start_index}, tensor contains {num_samples}."
        )

    if int(payload["num_devices"]) <= 0:
        raise ValueError(f"Invalid num_devices in {shard_path}.")
    if int(payload["window_size"]) <= 0:
        raise ValueError(f"Invalid window_size in {shard_path}.")

    return payload


def build_ebrl(*, num_states: int, embedding_dim: int) -> EcosystemBehavioralRepresentationLearner:
    return EcosystemBehavioralRepresentationLearner(
        num_states=num_states,
        embedding_dim=embedding_dim,
    )


def run_ebrl_on_shard(
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


def save_ebrl_representation_shard(
    output_path: Path,
    ecosystem_representation: torch.Tensor,
    *,
    source_bil_payload: dict[str, Any],
    embedding_dim: int,
    seed: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"EBRL output already exists:\n{output_path}\n\n"
            "Use --overwrite or enable overwrite in configuration."
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

        # Immediate upstream provenance
        "source_bil_shard": source_bil_payload["source_bil_shard"]
        if "source_bil_shard" in source_bil_payload
        else output_path.name,
        "source_bil_shard_index": int(source_bil_payload["shard_index"]),

        # Retained upstream provenance
        "source_bse_shard": source_bil_payload["source_bse_shard"],
        "source_bse_shard_index": int(source_bil_payload["source_bse_shard_index"]),
        "source_dbrl_shard": source_bil_payload["source_dbrl_shard"],
        "source_dbrl_shard_index": int(source_bil_payload["source_dbrl_shard_index"]),

        # Current shard identity and exact sample alignment
        "shard_index": int(source_bil_payload["shard_index"]),
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
        "source_bil_shard",
        "source_bil_shard_index",
        "source_bse_shard",
        "source_bse_shard_index",
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
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"EBRL output is missing required metadata: {missing}")

    output = payload["representations"]
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise ValueError("EBRL output representations must have shape (S, D).")
    if output.dtype != torch.float32 or not torch.isfinite(output).all():
        raise ValueError("EBRL output representations must be finite float32.")

    if int(payload["shard_index"]) != int(source_payload["shard_index"]):
        raise ValueError("EBRL shard index does not match BIL shard index.")
    if int(payload["start_index"]) != int(source_payload["start_index"]):
        raise ValueError("EBRL start_index does not match BIL start_index.")
    if int(payload["end_index"]) != int(source_payload["end_index"]):
        raise ValueError("EBRL end_index does not match BIL end_index.")
    if output.shape[0] != int(payload["end_index"]) - int(payload["start_index"]):
        raise ValueError("EBRL output sample count does not match its range.")

    for key in (
        "source_bil_shard",
        "source_bil_shard_index",
        "source_bse_shard",
        "source_bse_shard_index",
        "source_dbrl_shard",
        "source_dbrl_shard_index",
    ):
        expected = source_payload[key] if key in source_payload else source_payload.get(key)
        if expected is not None and payload[key] != expected:
            raise ValueError(f"EBRL provenance mismatch for '{key}'.")



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

    if batch_size <= 0:
        raise ValueError("EBRL batch_size must be greater than zero.")
    if max_shards is not None:
        max_shards = int(max_shards)
        if max_shards <= 0:
            raise ValueError("max_shards must be greater than zero.")

    
    processed_root = get_processed_folder(config)
    bil_split_dir = Path(f"{processed_root}/bil/{split}")
    ebrl_split_dir = Path(f"{processed_root}/ebrl/{split}")

    print("=" * 70)
    print("HDEG — EBRL Sharded Standalone Execution")
    print("=" * 70)
    print(f"Split                : {split}")
    print(f"BIL input directory  : {bil_split_dir}")
    print(f"EBRL output directory: {ebrl_split_dir}")
    print(f"Device               : {device}")
    print(f"EBRL batch size      : {batch_size}")
    print(f"Max shards           : {max_shards}")
    print(f"Overwrite            : {overwrite}")
    print()

    bil_shards = discover_bil_representation_shards(bil_split_dir)
    validate_shard_sequence(bil_shards)
    total_bil_shards = len(bil_shards)
    shard_paths = bil_shards if max_shards is None else bil_shards[:max_shards]

    first_payload = load_bil_representation_shard(
        shard_paths[0],
        expected_split=split,
        expected_num_states=9,
        expected_embedding_dim=None,
    )

    num_states = int(first_payload["representations"].shape[1])
    embedding_dim = int(first_payload["representations"].shape[2])
    num_devices = int(first_payload["num_devices"])

    if num_states != 9:
        raise ValueError(f"CU EBRL expects K=9; received K={num_states}.")

    print(f"Behavioral states K   : {num_states}")
    print(f"Embedding dimension D  : {embedding_dim}")
    print(f"Upstream devices N     : {num_devices}")
    print()

    del first_payload
    gc.collect()

    model = build_ebrl(
        num_states=num_states,
        embedding_dim=embedding_dim,
    ).to(device)
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"EBRL trainable params  : {parameter_count}")
    print()

    processed_shards = 0
    total_processed = 0
    expected_next_start = 0

    for shard_path in shard_paths:
        shard_index = parse_shard_filename(shard_path)
        payload = load_bil_representation_shard(
            shard_path,
            expected_split=split,
            expected_num_states=num_states,
            expected_embedding_dim=embedding_dim,
        )

        start_index = int(payload["start_index"])
        end_index = int(payload["end_index"])

        if processed_shards == 0 and start_index != 0:
            raise RuntimeError(
                f"First BIL shard starts at {start_index}; expected 0."
            )
        if processed_shards > 0 and start_index != expected_next_start:
            raise RuntimeError(
                "BIL shard sample ranges are not contiguous: "
                f"expected next start={expected_next_start}, received={start_index}."
            )

        S_tilde = payload["representations"]
        print(f"Processing shard {shard_index:06d}: [{start_index}, {end_index})")

        ecosystem_representation = run_ebrl_on_shard(
            model,
            S_tilde,
            device=device,
            batch_size=batch_size,
        )

        output_path = ebrl_split_dir / shard_path.name
        save_ebrl_representation_shard(
            output_path,
            ecosystem_representation,
            source_bil_payload=payload,
            embedding_dim=embedding_dim,
            seed=seed,
            overwrite=overwrite,
        )

        validate_ebrl_output_artifact(
            output_path,
            source_payload=payload,
        )

        print(f"  [PASS] output: {output_path}")

        total_processed += end_index - start_index
        processed_shards += 1
        expected_next_start = end_index

        del ecosystem_representation
        del S_tilde
        del payload
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("EBRL sharded execution completed")
    print("=" * 70)
    print(f"Processed shards   : {processed_shards}")
    print(f"Processed samples  : {total_processed}")
    print(f"Discovered shards  : {total_bil_shards}")
    print(f"EBRL output dir    : {ebrl_split_dir}")

    if max_shards is None:
        if processed_shards != total_bil_shards:
            raise RuntimeError(
                f"Expected {total_bil_shards} shards, processed {processed_shards}."
            )
        print("[PASS] Complete BIL representation split processed exactly once.")

    print("[PASS] EBRL sharded standalone execution finished.")


if __name__ == "__main__":
    main()
