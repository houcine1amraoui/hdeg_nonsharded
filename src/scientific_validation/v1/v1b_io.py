"""Minimal artifact IO for the flattened V1-B package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple
import json

import numpy as np
import torch
from src.utils.device import get_device


LEVEL_DIRS = {
    "Z": "dbrl",
    "S": "bse",
    "S_tilde": "bil",
    "g": "ebrl",
}

REPRESENTATION_KEY = "representations"


@dataclass
class ShardRecord:
    split: str
    shard_index: int
    start_index: int
    end_index: int
    paths: Dict[str, Path]


def _extract_tensor(obj, REPRESENTATION_KEY):
    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected shard object to be a dict, "
            f"received {type(obj).__name__}"
        )

    if REPRESENTATION_KEY in obj:
        tensor = obj[REPRESENTATION_KEY]

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"'{REPRESENTATION_KEY}' must be a torch.Tensor, "
                f"received {type(tensor).__name__}"
            )

        return tensor

    raise KeyError(
        f"Could not find '{REPRESENTATION_KEY}' in representation shard. "
        f"Available keys={tuple(obj.keys())}"
    )


def load_representation_shard(
    artifact_root: Path,
    split: str,
    shard_index: int,
    *,
    return_metadata: bool = False,
    levels: Optional[Iterable[str]] = None,
):
    """Load one HDEG representation shard.

    The persisted HDEG schema stores the tensor under ``representations``.
    When ``return_metadata`` is True, the artifact metadata is returned for
    each level so the V1 eligibility layer can verify sample ranges and
    provenance without inventing a parallel schema.
    """
    device = get_device()
    result = {}
    metadata = {}
    requested_levels = tuple(LEVEL_DIRS) if levels is None else tuple(levels)
    unknown = [level for level in requested_levels if level not in LEVEL_DIRS]
    if unknown:
        raise ValueError(f"Unknown representation level(s): {unknown}")

    for level in requested_levels:
        dirname = LEVEL_DIRS[level]
        p = Path(f"{artifact_root}/{dirname}/{split}/shard_{shard_index:06d}.pt")
        if not p.exists():
            raise FileNotFoundError(p)
        obj = torch.load(p, map_location=device)
        tensor = _extract_tensor(obj, REPRESENTATION_KEY)
        result[level] = tensor.detach().cpu().numpy()
        if return_metadata:
            metadata[level] = dict(obj)

    if return_metadata:
        return result, metadata
    return result


def discover_shards(
    artifact_root: Path,
    split: str,
) -> list[int]:
    first = f"{artifact_root}/dbrl/{split}"
    print(f"Discovering representation shards in {first}...")
    if not Path(first).exists():
        return []
    out = []
    for p in sorted(Path(first).glob("shard_*.pt")):
        try:
            out.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return out


def load_window_shard(
    window_root: Path,
    split: str,
    shard_index: int,
) -> dict:
    p = Path(f"{window_root}/{split}/shard_{shard_index:06d}.npz")
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    return out


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)
