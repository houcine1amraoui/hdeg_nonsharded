from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml

from src.common.semantics.behavioral_state_config import (
    BehavioralStateConfigError,
    load_behavioral_state_config,
    validate_behavioral_state_config,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device_ids() -> list[str]:
    """
    Small deterministic device ordering used by unit tests.

    The real project ordering comes from devices.json. These names are
    intentionally synthetic so the tests remain independent of the actual
    dataset artifact.
    """
    return [
        "device_0",
        "device_1",
        "device_2",
        "device_3",
        "device_4",
        "device_5",
    ]


@pytest.fixture
def states() -> list[dict]:
    """
    Minimal valid behavioral-state configuration.

    Six devices are partitioned across three behavioral states.
    """

    return [
        {
            "id": "state_0",
            "index": 0,
            "name": "State 0",
            "rationale": "Synthetic test state.",
            "provenance_type": "test",
            "confidence": "high",
            "compatible_devices": [
                {
                    "index": 0,
                    "id": "device_0",
                    "relation": "DIRECT",
                },
                {
                    "index": 1,
                    "id": "device_1",
                    "relation": "DIRECT",
                },
            ],
        },
        {
            "id": "state_1",
            "index": 1,
            "name": "State 1",
            "rationale": "Synthetic test state.",
            "provenance_type": "test",
            "confidence": "high",
            "compatible_devices": [
                {
                    "index": 2,
                    "id": "device_2",
                    "relation": "DIRECT",
                },
                {
                    "index": 3,
                    "id": "device_3",
                    "relation": "DIRECT",
                },
            ],
        },
        {
            "id": "state_2",
            "index": 2,
            "name": "State 2",
            "rationale": "Synthetic test state.",
            "provenance_type": "test",
            "confidence": "high",
            "compatible_devices": [
                {
                    "index": 4,
                    "id": "device_4",
                    "relation": "DIRECT",
                },
                {
                    "index": 5,
                    "id": "device_5",
                    "relation": "DIRECT",
                },
            ],
        },
    ]


@pytest.fixture
def valid_matrix() -> list[list[int]]:
    return [
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
    ]


@pytest.fixture
def valid_config(
    device_ids: list[str],
    states: list[dict],
    valid_matrix: list[list[int]],
) -> dict:
    return {
        "schema_version": "1.0",
        "artifact_type": (
            "hdeg_behavioral_state_configuration"
        ),
        "status": "frozen",
        "dataset": {
            "name": "synthetic_test_dataset",
            "selected_device_count": len(device_ids),
            "device_order_source": "devices.json",
            "device_ordering_rule": "exact_file_order",
        },
        "semantic_policy": {
            "compatibility_relation": "DIRECT",
            "contextual_edges": "none",
        },
        "behavioral_states": states,
        "matrix": {
            "shape": [
                len(states),
                len(device_ids),
            ],
            "dtype": "uint8",
            "rows": valid_matrix,
        },
    }


@pytest.fixture
def config_files(
    tmp_path,
    valid_config: dict,
    device_ids: list[str],
):
    """
    Write a complete valid configuration to temporary files.
    """

    config_path = tmp_path / "behavioral_states.yaml"
    devices_path = tmp_path / "devices.json"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with devices_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            device_ids,
            file,
            indent=2,
        )

    return config_path, devices_path


# ---------------------------------------------------------------------------
# Valid configuration
# ---------------------------------------------------------------------------


def test_valid_configuration_loads(
    config_files,
):
    config_path, devices_path = config_files

    configuration = load_behavioral_state_config(
        config_path,
        devices_path,
    )

    assert configuration.num_devices == 6
    assert configuration.num_states == 3

    assert configuration.matrix_shape == (
        3,
        6,
    )


def test_device_order_is_preserved(
    config_files,
    device_ids,
):
    config_path, devices_path = config_files

    configuration = load_behavioral_state_config(
        config_path,
        devices_path,
    )

    assert configuration.device_ids() == tuple(
        device_ids
    )


def test_state_order_is_preserved(
    config_files,
):
    config_path, devices_path = config_files

    configuration = load_behavioral_state_config(
        config_path,
        devices_path,
    )

    assert configuration.state_ids() == (
        "state_0",
        "state_1",
        "state_2",
    )


def test_matrix_is_loaded_exactly(
    config_files,
    valid_matrix,
):
    config_path, devices_path = config_files

    configuration = load_behavioral_state_config(
        config_path,
        devices_path,
    )

    expected = np.asarray(
        valid_matrix,
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(
        configuration.compatibility_matrix,
        expected,
    )


def test_torch_mask_shape_and_dtype(
    config_files,
):
    config_path, devices_path = config_files

    configuration = load_behavioral_state_config(
        config_path,
        devices_path,
    )

    mask = configuration.torch_mask()

    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (3, 6)
    assert mask.dtype == torch.float32


def test_torch_mask_contains_matrix_values(
    config_files,
    valid_matrix,
):
    config_path, devices_path = config_files

    configuration = load_behavioral_state_config(
        config_path,
        devices_path,
    )

    mask = configuration.torch_mask()

    expected = torch.tensor(
        valid_matrix,
        dtype=torch.float32,
    )

    assert torch.equal(
        mask,
        expected,
    )


def test_torch_mask_is_independent_copy(
    config_files,
):
    config_path, devices_path = config_files

    configuration = load_behavioral_state_config(
        config_path,
        devices_path,
    )

    mask = configuration.torch_mask()

    mask[0, 0] = 0.0

    assert (
        configuration.compatibility_matrix[0, 0]
        == 1
    )


# ---------------------------------------------------------------------------
# File-level validation
# ---------------------------------------------------------------------------


def test_missing_configuration_file(
    tmp_path,
):
    devices_path = tmp_path / "devices.json"

    devices_path.write_text(
        '["device_0"]',
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Behavioral-state configuration not found",
    ):
        load_behavioral_state_config(
            tmp_path / "missing.yaml",
            devices_path,
        )


def test_missing_devices_file(
    tmp_path,
):
    config_path = tmp_path / "behavioral_states.yaml"

    config_path.write_text(
        "schema_version: '1.0'\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Device-order file not found",
    ):
        load_behavioral_state_config(
            config_path,
            tmp_path / "missing.json",
        )


def test_invalid_yaml(
    tmp_path,
):
    config_path = tmp_path / "behavioral_states.yaml"
    devices_path = tmp_path / "devices.json"

    config_path.write_text(
        "this: [is: invalid",
        encoding="utf-8",
    )

    devices_path.write_text(
        '["device_0"]',
        encoding="utf-8",
    )

    with pytest.raises(
        BehavioralStateConfigError,
        match="Invalid YAML configuration",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_invalid_devices_json(
    tmp_path,
):
    config_path = tmp_path / "behavioral_states.yaml"
    devices_path = tmp_path / "devices.json"

    config_path.write_text(
        "{}",
        encoding="utf-8",
    )

    devices_path.write_text(
        "{invalid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        BehavioralStateConfigError,
        match="Invalid devices.json",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


# ---------------------------------------------------------------------------
# Top-level structural validation
# ---------------------------------------------------------------------------


def test_invalid_artifact_type(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config["artifact_type"] = "wrong_type"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="Unexpected artifact_type",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_device_count_mismatch(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config["dataset"][
        "selected_device_count"
    ] = 999

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="Device-count mismatch",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_wrong_device_order_source(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config["dataset"][
        "device_order_source"
    ] = "another_file.json"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="device_order_source",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


# ---------------------------------------------------------------------------
# Device-order validation
# ---------------------------------------------------------------------------


def test_device_id_mismatch_is_rejected(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config[
        "behavioral_states"
    ][0][
        "compatible_devices"
    ][0]["id"] = "wrong_device"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="does not match devices.json ordering",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_invalid_device_index_is_rejected(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config[
        "behavioral_states"
    ][0][
        "compatible_devices"
    ][0]["index"] = 999

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="outside",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_duplicate_device_indices_are_rejected(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config[
        "behavioral_states"
    ][0][
        "compatible_devices"
    ].append(
        {
            "index": 1,
            "id": "device_1",
            "relation": "DIRECT",
        }
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="duplicate",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


# ---------------------------------------------------------------------------
# Relationship validation
# ---------------------------------------------------------------------------


def test_contextual_relation_is_rejected(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config[
        "behavioral_states"
    ][0][
        "compatible_devices"
    ][0]["relation"] = "CONTEXTUAL"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="DIRECT",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_contextual_edges_policy_is_rejected(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config[
        "semantic_policy"
    ]["contextual_edges"] = "enabled"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="contextual_edges",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


# ---------------------------------------------------------------------------
# Matrix validation
# ---------------------------------------------------------------------------


def test_matrix_shape_mismatch_is_rejected(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config[
        "matrix"
    ]["shape"] = [999, 999]

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="shape",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_matrix_dtype_mismatch_is_rejected(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config[
        "matrix"
    ]["dtype"] = "float32"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="uint8",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_non_binary_matrix_is_rejected(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config[
        "matrix"
    ]["rows"][0][0] = 2

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="0 and 1",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_matrix_row_count_mismatch_is_rejected(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config[
        "matrix"
    ]["rows"] = [
        [1, 1, 0, 0, 0, 0],
    ]

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="row count",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


# ---------------------------------------------------------------------------
# Semantic consistency validation
# ---------------------------------------------------------------------------


def test_state_declaration_must_match_matrix(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    # State 0 declares devices 0 and 1, but the matrix
    # will declare only device 0.
    valid_config[
        "matrix"
    ]["rows"][0] = [
        1,
        0,
        0,
        0,
        0,
        0,
    ]

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="does not match",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_state_without_devices_is_rejected(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    valid_config[
        "behavioral_states"
    ][0]["compatible_devices"] = []

    valid_config[
        "matrix"
    ]["rows"][0] = [
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="non-empty",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_device_assigned_to_multiple_states_is_rejected(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    # Add device 0 to state 1 as well.
    valid_config[
        "behavioral_states"
    ][1][
        "compatible_devices"
    ].append(
        {
            "index": 0,
            "id": "device_0",
            "relation": "DIRECT",
        }
    )

    valid_config[
        "matrix"
    ]["rows"][1][0] = 1

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="exactly one",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_device_assigned_to_no_state_is_rejected(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    # Remove device 5 from its state.
    valid_config[
        "behavioral_states"
    ][2][
        "compatible_devices"
    ] = [
        {
            "index": 4,
            "id": "device_4",
            "relation": "DIRECT",
        }
    ]

    valid_config[
        "matrix"
    ]["rows"][2] = [
        0,
        0,
        0,
        0,
        1,
        0,
    ]

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="exactly one",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


# ---------------------------------------------------------------------------
# Public validation alias
# ---------------------------------------------------------------------------


def test_validation_alias_matches_loader(
    config_files,
):
    config_path, devices_path = config_files

    loaded = load_behavioral_state_config(
        config_path,
        devices_path,
    )

    validated = validate_behavioral_state_config(
        config_path,
        devices_path,
    )

    np.testing.assert_array_equal(
        loaded.compatibility_matrix,
        validated.compatibility_matrix,
    )

    assert loaded.device_ids() == (
        validated.device_ids()
    )

    assert loaded.state_ids() == (
        validated.state_ids()
    )

def test_matrix_row_must_have_at_least_one_compatible_device(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    # Keep the semantic declaration valid, but make the
    # executable matrix row empty.
    valid_config["matrix"]["rows"][0] = [
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="does not match",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )


def test_matrix_columns_must_form_partition(
    config_files,
    valid_config,
):
    config_path, devices_path = config_files

    # Device 0 is now absent from every state.
    valid_config["behavioral_states"][0][
        "compatible_devices"
    ] = [
        {
            "index": 1,
            "id": "device_1",
            "relation": "DIRECT",
        }
    ]

    valid_config["matrix"]["rows"][0] = [
        0,
        1,
        0,
        0,
        0,
        0,
    ]

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            valid_config,
            file,
            sort_keys=False,
        )

    with pytest.raises(
        BehavioralStateConfigError,
        match="exactly one",
    ):
        load_behavioral_state_config(
            config_path,
            devices_path,
        )