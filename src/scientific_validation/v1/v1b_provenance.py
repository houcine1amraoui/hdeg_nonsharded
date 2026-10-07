"""Provenance helpers for V1-B evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import hashlib
import json


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_manifest(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_config_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load config.yaml") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_protocol_manifest(
    *,
    dataset: str,
    seed: int,
    window_size: int,
    num_devices: int,
    representation_dim: int,
    r2_pair_count: int,
    r3_max_per_condition: int,
    r3_block_length_windows: int,
    bootstrap_replicates: int,
) -> dict:
    return {
        "dataset": dataset,
        "seed": seed,
        "window_size": window_size,
        "num_devices": num_devices,
        "representation_dim": representation_dim,
        "r2": {
            "pair_count": r2_pair_count,
            "sampling": "deterministic_representation_blind_without_replacement",
        },
        "r3": {
            "max_per_condition": r3_max_per_condition,
            "sampling": "deterministic_representation_blind_without_replacement",
            "uncertainty": {
                "method": "moving_block_bootstrap",
                "block_length_windows": r3_block_length_windows,
                "bootstrap_replicates": bootstrap_replicates,
                "interval_method": "percentile",
                "confidence_level": 0.95,
            },
        },
        "conditions": {
            "N1": "val",
            "A2": "actor2_test",
            "N2": "actor1_test",
        },
    }
