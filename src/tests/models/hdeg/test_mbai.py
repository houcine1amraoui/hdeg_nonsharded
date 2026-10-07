from __future__ import annotations

import pytest
import torch

from src.models.hdeg.mbai import MultiScaleBehavioralAnomalyInference, hierarchical_anomaly_evidence_shapes


@pytest.fixture
def hierarchies():
    torch.manual_seed(7)
    observed = {
        "Z": torch.randn(4, 23, 64),
        "S": torch.randn(4, 9, 64),
        "S_tilde": torch.randn(4, 9, 64),
        "g": torch.randn(4, 64),
    }
    predicted = {
        key: value + 0.1 * torch.randn_like(value)
        for key, value in observed.items()
    }
    return observed, predicted


def test_default_configuration_is_non_trainable():
    model = MultiScaleBehavioralAnomalyInference()

    assert model.has_learnable_parameters is False
    assert list(model.parameters()) == []
    assert tuple(model.fusion_weights.tolist()) == (1.0, 1.0, 1.0, 1.0)


def test_normal_inference_outputs_four_evidence_levels_and_assessment(hierarchies):
    observed, predicted = hierarchies
    model = MultiScaleBehavioralAnomalyInference()

    with torch.inference_mode():
        outputs = model(observed, predicted)

    assert set(outputs) == {"E_Z", "E_S", "E_S_tilde", "E_G", "A"}
    assert hierarchical_anomaly_evidence_shapes(outputs) == (
        (4,),
        (4,),
        (4,),
        (4,),
    )
    assert outputs["A"].shape == (4,)
    for tensor in outputs.values():
        assert torch.isfinite(tensor).all()


def test_identical_observed_and_predicted_hierarchies_produce_zero_evidence():
    torch.manual_seed(1)
    observed = {
        "Z": torch.randn(2, 23, 64),
        "S": torch.randn(2, 9, 64),
        "S_tilde": torch.randn(2, 9, 64),
        "g": torch.randn(2, 64),
    }
    predicted = {key: value.clone() for key, value in observed.items()}
    model = MultiScaleBehavioralAnomalyInference()

    outputs = model(observed, predicted)

    for key in ("E_Z", "E_S", "E_S_tilde", "E_G", "A"):
        assert torch.equal(outputs[key], torch.zeros_like(outputs[key]))


def test_hierarchical_correspondence_is_preserved(hierarchies):
    observed, predicted = hierarchies
    model = MultiScaleBehavioralAnomalyInference()

    baseline = model(observed, predicted)

    perturbed = {key: value.clone() for key, value in observed.items()}
    perturbed["Z"] = perturbed["Z"] + 1.0
    changed = model(perturbed, predicted)

    assert not torch.equal(baseline["E_Z"], changed["E_Z"])
    assert torch.equal(baseline["E_S"], changed["E_S"])
    assert torch.equal(baseline["E_S_tilde"], changed["E_S_tilde"])
    assert torch.equal(baseline["E_G"], changed["E_G"])


def test_all_hierarchy_levels_contribute_to_assessment(hierarchies):
    observed, predicted = hierarchies
    model = MultiScaleBehavioralAnomalyInference()
    baseline = model(observed, predicted)

    for level in ("Z", "S", "S_tilde", "g"):
        perturbed = {key: value.clone() for key, value in observed.items()}
        perturbed[level] = perturbed[level] + 1.0
        changed = model(perturbed, predicted)
        assert not torch.equal(
            baseline["A"], changed["A"]
        ), f"Level '{level}' did not affect final assessment."


def test_deterministic_execution(hierarchies):
    observed, predicted = hierarchies
    model = MultiScaleBehavioralAnomalyInference()

    first = model(observed, predicted)
    second = model(observed, predicted)

    for key in first:
        assert torch.equal(first[key], second[key])


def test_custom_fixed_fusion_weights_are_applied():
    observed = {
        "Z": torch.zeros(1, 2, 2),
        "S": torch.zeros(1, 1, 2),
        "S_tilde": torch.zeros(1, 1, 2),
        "g": torch.zeros(1, 2),
    }
    predicted = {
        "Z": torch.ones(1, 2, 2),
        "S": torch.zeros(1, 1, 2),
        "S_tilde": torch.zeros(1, 1, 2),
        "g": torch.zeros(1, 2),
    }

    model = MultiScaleBehavioralAnomalyInference(
        fusion_weights={"Z": 1.0, "S": 0.0, "S_tilde": 0.0, "g": 0.0}
    )
    outputs = model(observed, predicted)

    # Mean squared discrepancy at Z is exactly 1.0.
    assert torch.equal(outputs["E_Z"], torch.ones(1))
    assert torch.equal(outputs["E_S"], torch.zeros(1))
    assert torch.equal(outputs["E_S_tilde"], torch.zeros(1))
    assert torch.equal(outputs["E_G"], torch.zeros(1))
    assert torch.equal(outputs["A"], torch.ones(1))


def test_missing_or_extra_hierarchy_keys_are_rejected(hierarchies):
    observed, predicted = hierarchies
    model = MultiScaleBehavioralAnomalyInference()

    bad_observed = dict(observed)
    del bad_observed["g"]
    with pytest.raises(ValueError, match="Observed hierarchy keys"):
        model(bad_observed, predicted)

    bad_predicted = dict(predicted)
    bad_predicted["extra"] = bad_predicted["g"]
    with pytest.raises(ValueError, match="Predicted hierarchy keys"):
        model(observed, bad_predicted)


def test_shape_mismatch_is_rejected(hierarchies):
    observed, predicted = hierarchies
    predicted = dict(predicted)
    predicted["S"] = torch.randn(4, 8, 64)
    model = MultiScaleBehavioralAnomalyInference()

    with pytest.raises(ValueError, match="shape mismatch at 'S'"):
        model(observed, predicted)


def test_non_finite_input_is_rejected(hierarchies):
    observed, predicted = hierarchies
    observed = dict(observed)
    observed["g"] = observed["g"].clone()
    observed["g"][0, 0] = float("nan")
    model = MultiScaleBehavioralAnomalyInference()

    with pytest.raises(ValueError, match=r"observed\['g'\]"):
        model(observed, predicted)


def test_invalid_fusion_weights_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        MultiScaleBehavioralAnomalyInference(
            fusion_weights={"Z": -1.0, "S": 1.0, "S_tilde": 1.0, "g": 1.0}
        )

    with pytest.raises(ValueError, match="At least one"):
        MultiScaleBehavioralAnomalyInference(
            fusion_weights={"Z": 0.0, "S": 0.0, "S_tilde": 0.0, "g": 0.0}
        )


def test_gradients_can_flow_through_inputs_but_not_mbaI_parameters(hierarchies):
    observed, predicted = hierarchies
    observed = {key: value.clone().requires_grad_(True) for key, value in observed.items()}
    model = MultiScaleBehavioralAnomalyInference()

    outputs = model(observed, predicted)
    outputs["A"].mean().backward()

    assert all(value.grad is not None for value in observed.values())
    assert list(model.parameters()) == []
