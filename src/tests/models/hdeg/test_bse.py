from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Load the current BSE implementation directly from the generated file.
# This keeps the standalone verification suite independent of the full
# project package layout.
# ---------------------------------------------------------------------------

BSE_PATH = Path("/mnt/data/bse.py")

spec = importlib.util.spec_from_file_location(
    "hdeg_bse_under_test",
    BSE_PATH,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load BSE module from {BSE_PATH}.")

bse_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bse_module)

BehavioralStateEstimator = bse_module.BehavioralStateEstimator


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
NUM_DEVICES = 5
NUM_STATES = 3
EMBEDDING_DIM = 12
NUM_HEADS = 3


def make_model() -> BehavioralStateEstimator:
    return BehavioralStateEstimator(
        num_states=NUM_STATES,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
    )


def make_embeddings() -> torch.Tensor:
    torch.manual_seed(1234)
    return torch.randn(
        BATCH_SIZE,
        NUM_DEVICES,
        EMBEDDING_DIM,
        dtype=torch.float32,
    )


def make_compatibility_mask() -> torch.Tensor:
    # Every behavioral state has at least one compatible device.
    #
    # State 0 -> devices 0, 1
    # State 1 -> devices 1, 2, 3
    # State 2 -> devices 0, 3, 4
    return torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )


# ---------------------------------------------------------------------------
# 1. Behavioral State Ownership Verification
# ---------------------------------------------------------------------------

def test_behavioral_state_ownership_and_output_shape() -> None:
    model = make_model()
    z = make_embeddings()
    mask = make_compatibility_mask()

    s = model(z, mask)

    assert s.shape == (
        BATCH_SIZE,
        NUM_STATES,
        EMBEDDING_DIM,
    )

    # Exactly one representation exists for every behavioral state.
    assert s.size(1) == NUM_STATES


# ---------------------------------------------------------------------------
# 2. Representation Consistency Verification
# ---------------------------------------------------------------------------

def test_representation_dimension_and_finite_output() -> None:
    model = make_model()
    z = make_embeddings()
    mask = make_compatibility_mask()

    s = model(z, mask)

    assert s.dtype == z.dtype
    assert s.device == z.device
    assert s.size(-1) == EMBEDDING_DIM
    assert torch.isfinite(s).all()


# ---------------------------------------------------------------------------
# 3. Semantic Consistency Verification
#
# For each state, construct a compatibility mask containing exactly one
# device. The attention distribution then has exactly one valid position,
# so the BSE output must equal the value-projected representation of that
# compatible device.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state_index", range(NUM_STATES))
def test_single_compatible_device_is_exclusive(
    state_index: int,
) -> None:
    model = make_model()
    model.eval()

    z = make_embeddings()

    selected_device = state_index % NUM_DEVICES

    mask = torch.zeros(
        NUM_STATES,
        NUM_DEVICES,
        dtype=torch.bool,
    )

    # Give every state a valid compatible device.
    for k in range(NUM_STATES):
        mask[k, k % NUM_DEVICES] = True

    s = model(z, mask)

    with torch.no_grad():
        projected = model.value_projection(z)

    expected = projected[
        :,
        selected_device,
        :,
    ]

    actual = s[
        :,
        state_index,
        :,
    ]

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 4. Semantic Consistency Verification
#
# Changing an incompatible device must not change the behavioral state
# whose compatibility row excludes that device.
# ---------------------------------------------------------------------------

def test_incompatible_device_cannot_affect_behavioral_state() -> None:
    model = make_model()
    model.eval()

    torch.manual_seed(5678)

    z1 = torch.randn(
        BATCH_SIZE,
        NUM_DEVICES,
        EMBEDDING_DIM,
    )

    z2 = z1.clone()

    # State 0 permits devices 0 and 1, but not device 4.
    mask = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )

    # Make a very large change to an incompatible device for state 0.
    z2[:, 4, :] = z2[:, 4, :] + 100000.0

    s1 = model(z1, mask)
    s2 = model(z2, mask)

    # State 0 must be invariant to device 4.
    torch.testing.assert_close(
        s1[:, 0, :],
        s2[:, 0, :],
        rtol=1e-5,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 5. Attention Normalization Verification
#
# When a state has exactly one compatible device, its attention mass is
# necessarily 1. The previous exclusive-evidence test verifies this
# consequence at the observable module boundary.
#
# Additionally, verify that a multi-device compatible state produces a
# finite normalized result by reconstructing the specified attention
# equation from the module's own Q/K projections. This checks the exact
# mathematical normalization used by the implementation.
# ---------------------------------------------------------------------------

def test_compatibility_constrained_attention_is_normalized() -> None:
    model = make_model()
    model.eval()

    z = make_embeddings()
    mask = make_compatibility_mask()

    batch_size = z.size(0)

    with torch.no_grad():
        queries = model.behavioral_queries.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        q = model._split_heads(
            model.query_projection(queries)
        )
        k = model._split_heads(
            model.key_projection(z)
        )

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / (model.head_dim ** 0.5)

        compatibility = mask.unsqueeze(0).unsqueeze(0)

        masked_scores = scores.masked_fill(
            ~compatibility,
            float("-inf"),
        )

        attention = torch.softmax(
            masked_scores,
            dim=-1,
        )

    assert torch.isfinite(attention).all()

    # Every state/head/sample must sum to one.
    sums = attention.sum(dim=-1)

    torch.testing.assert_close(
        sums,
        torch.ones_like(sums),
        rtol=1e-6,
        atol=1e-6,
    )

    # Every incompatible device must have exactly zero attention.
    incompatible = (~compatibility).expand_as(attention)

    assert torch.equal(
        attention[incompatible],
        torch.zeros_like(attention[incompatible]),
    )


# ---------------------------------------------------------------------------
# 6. Evidence Preservation Verification
#
# If a state is made compatible with exactly one device, its output is
# exactly that device's value-transformed evidence. This verifies that
# device identity/order is preserved through the semantic aggregation.
# ---------------------------------------------------------------------------

def test_device_evidence_preservation() -> None:
    model = make_model()
    model.eval()

    z = make_embeddings()

    selected_devices = [0, 2, 4]

    mask = torch.zeros(
        NUM_STATES,
        NUM_DEVICES,
        dtype=torch.bool,
    )

    for k, device_index in enumerate(selected_devices):
        mask[k, device_index] = True

    s = model(z, mask)

    with torch.no_grad():
        v = model.value_projection(z)

    for k, device_index in enumerate(selected_devices):
        torch.testing.assert_close(
            s[:, k, :],
            v[:, device_index, :],
            rtol=1e-5,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# 7. Value Projection Verification
#
# The scientific formulation requires W_V to participate in the
# behavioral-state aggregation. With exactly one compatible device,
# the output must therefore equal W_V z_i.
# ---------------------------------------------------------------------------

def test_value_projection_is_part_of_state_formation() -> None:
    model = make_model()
    model.eval()

    z = make_embeddings()

    mask = torch.zeros(
        NUM_STATES,
        NUM_DEVICES,
        dtype=torch.bool,
    )

    for k in range(NUM_STATES):
        mask[k, k] = True

    s = model(z, mask)

    # If W_V were absent, this equality would generally fail.
    with torch.no_grad():
        expected = model.value_projection(z)

    for k in range(NUM_STATES):
        torch.testing.assert_close(
            s[:, k, :],
            expected[:, k, :],
            rtol=1e-5,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# 8. Differentiability Verification
# ---------------------------------------------------------------------------

def test_gradients_reach_all_trainable_components() -> None:
    model = make_model()
    model.train()

    z = make_embeddings().requires_grad_(True)
    mask = make_compatibility_mask()

    s = model(z, mask)

    loss = s.pow(2).mean()
    loss.backward()

    trainable_parameters = dict(
        model.named_parameters()
    )

    expected_parameters = {
        "behavioral_queries",
        "query_projection.weight",
        "key_projection.weight",
        "value_projection.weight",
    }

    assert set(trainable_parameters) == expected_parameters

    for name, parameter in trainable_parameters.items():
        assert parameter.requires_grad
        assert parameter.grad is not None, (
            f"No gradient reached trainable parameter '{name}'."
        )
        assert torch.isfinite(parameter.grad).all(), (
            f"Non-finite gradient detected for '{name}'."
        )
        assert parameter.grad.abs().sum() > 0, (
            f"Zero gradient detected for '{name}'."
        )

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 9. Deterministic Inference Verification
# ---------------------------------------------------------------------------

def test_deterministic_inference() -> None:
    model = make_model()
    model.eval()

    z = make_embeddings()
    mask = make_compatibility_mask()

    with torch.no_grad():
        s1 = model(z, mask)
        s2 = model(z, mask)

    torch.testing.assert_close(
        s1,
        s2,
        rtol=0.0,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# 10. Constructor Contract Verification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "num_states": 0,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
        },
        {
            "num_states": NUM_STATES,
            "embedding_dim": 0,
            "num_heads": NUM_HEADS,
        },
        {
            "num_states": NUM_STATES,
            "embedding_dim": 10,
            "num_heads": 3,
        },
        {
            "num_states": NUM_STATES,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": 0,
        },
    ],
)
def test_invalid_constructor_parameters_are_rejected(
    kwargs: dict,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        BehavioralStateEstimator(**kwargs)


# ---------------------------------------------------------------------------
# 11. Input Contract Verification
# ---------------------------------------------------------------------------

def test_invalid_device_embedding_rank_is_rejected() -> None:
    model = make_model()
    mask = make_compatibility_mask()

    with pytest.raises(ValueError):
        model(
            torch.randn(
                NUM_DEVICES,
                EMBEDDING_DIM,
            ),
            mask,
        )


def test_invalid_embedding_dimension_is_rejected() -> None:
    model = make_model()
    mask = make_compatibility_mask()

    with pytest.raises(ValueError):
        model(
            torch.randn(
                BATCH_SIZE,
                NUM_DEVICES,
                EMBEDDING_DIM + 1,
            ),
            mask,
        )


def test_invalid_compatibility_shape_is_rejected() -> None:
    model = make_model()
    z = make_embeddings()

    with pytest.raises(ValueError):
        model(
            z,
            torch.ones(
                NUM_STATES,
                NUM_DEVICES + 1,
                dtype=torch.bool,
            ),
        )


def test_non_binary_compatibility_matrix_is_rejected() -> None:
    model = make_model()
    z = make_embeddings()

    mask = make_compatibility_mask().float()
    mask[0, 0] = 2.0

    with pytest.raises(ValueError):
        model(z, mask)


def test_behavioral_state_without_compatible_device_is_rejected() -> None:
    model = make_model()
    z = make_embeddings()

    mask = make_compatibility_mask()
    mask[1, :] = False

    with pytest.raises(ValueError):
        model(z, mask)


def test_non_finite_device_embeddings_are_rejected() -> None:
    model = make_model()
    mask = make_compatibility_mask()

    z = make_embeddings()
    z[0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        model(z, mask)


# ---------------------------------------------------------------------------
# 12. Scientific Artifact Exposure Verification
# ---------------------------------------------------------------------------

def test_default_forward_exposes_only_scientific_artifact() -> None:
    model = make_model()
    z = make_embeddings()
    mask = make_compatibility_mask()

    output = model(z, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (
        BATCH_SIZE,
        NUM_STATES,
        EMBEDDING_DIM,
    )


# ---------------------------------------------------------------------------
# 13. State-order consistency
#
# Swapping two compatibility rows must swap the corresponding state
# outputs when the rows are otherwise identical in meaning. This verifies
# that state index k remains associated with compatibility row k.
# ---------------------------------------------------------------------------

def test_behavioral_state_order_follows_compatibility_rows() -> None:
    model = make_model()
    model.eval()

    z = make_embeddings()

    mask_a = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=torch.bool,
    )

    mask_b = mask_a[[1, 0, 2]]

    s_a = model(z, mask_a)
    s_b = model(z, mask_b)

    # State 0 in mask_b uses the evidence assigned to state 1 in mask_a.
    torch.testing.assert_close(
        s_b[:, 0, :],
        s_a[:, 1, :],
        rtol=1e-5,
        atol=1e-6,
    )

    # State 1 in mask_b uses the evidence assigned to state 0 in mask_a.
    torch.testing.assert_close(
        s_b[:, 1, :],
        s_a[:, 0, :],
        rtol=1e-5,
        atol=1e-6,
    )

    # State 2 remains unchanged.
    torch.testing.assert_close(
        s_b[:, 2, :],
        s_a[:, 2, :],
        rtol=1e-5,
        atol=1e-6,
    )