#!/usr/bin/env python3
"""Command-line entry point for the flattened HDEG V1-B engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.scientific_validation.v1.v1b_engine import V1BConfig, V1BEngine
from src.scientific_validation.v1.v1b_evidence import write_v1b_evidence

import yaml

from src.utils.seed import set_seed
from src.utils.get_folders_utils import get_processed_folder

def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    set_seed(config["seed"])

    # # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str)
    parser.add_argument("--split", type=str)
    args = parser.parse_args()
    
    seed = int(config["seed"])
    set_seed(seed)

    processed_folder = get_processed_folder(config)
    artifact_root = f"{processed_folder}"

    window_root = f"{processed_folder}/windows"
    output_root = f"{processed_folder}/v1_validation"

    r2_pairs = config["v1_validation"]["r2-pairs"]
    r3_max = config["v1_validation"]["r3-max"]
    r1_reservoir = config["v1_validation"]["r1-reservoir"]
    r1_chunk = config["v1_validation"]["r1-chunk"]
    bootstrap_replicates = config["v1_validation"]["bootstrap-replicates"]

    cfg = V1BConfig(
        artifact_root=artifact_root,
        window_root=window_root,
        output_root=output_root,
        r2_pair_count=r2_pairs,
        r3_max_per_condition=r3_max,
        r1_reservoir_size=r1_reservoir,
        r1_chunk_size=r1_chunk,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        bootstrap_seed=seed,
    )

    evidence = V1BEngine(cfg).run()
    write_v1b_evidence(output_root, evidence)

    print("V1-B evidence generation completed.")
    print(f"Evidence root: {output_root}")


if __name__ == "__main__":
    main()
