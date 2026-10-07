"""Evidence artifact assembly for V1-B."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from src.scientific_validation.v1.v1b_io import save_json


def write_v1b_evidence(output_root: Path, evidence: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    save_json(output_root / "manifest.json", evidence["manifest"])
    save_json(
        output_root / "r1" / "structural_geometry.json",
        evidence.get("r1", {}),
    )
    save_json(
        output_root / "r2" / "behavioral_relationship_preservation.json",
        evidence.get("r2", {}),
    )
    save_json(
        output_root / "r3" / "discriminative_energy.json",
        evidence.get("r3", {}),
    )

    save_json(
        output_root / "provenance" / "execution.json",
        evidence.get("provenance", {}),
    )
