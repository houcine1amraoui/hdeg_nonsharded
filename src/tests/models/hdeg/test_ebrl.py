from __future__ import annotations

import torch

from src.models.hdeg.ebrl import EcosystemBehavioralRepresentationLearner


K = 9
D = 64
B = 8


def _build(seed: int = 42) -> EcosystemBehavioralRepresentationLearner:
    torch.manual_seed(seed)
    return EcosystemBehavioralRepresentationLearner(
        embedding_dim=D,
        num_states=K,
    )


def test_shape_dtype_and_finiteness() -> None:
    model = _build()
    x = torch.randn(B, K, D)

    with torch.inference_mode():
        y = model(x)

    assert y.shape == (B, D)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_attention_matches_frozen_equations() -> None:
    model = _build()
    x = torch.randn(B, K, D)

    with torch.inference_mode():
        projected = model.behavior_projection(x)
        scores = torch.tanh(projected).matmul(model.attention_vector)
        gamma = torch.softmax(scores, dim=1)
        expected = torch.sum(gamma.unsqueeze(-1) * x, dim=1)
        actual = model(x)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.allclose(gamma.sum(dim=1), torch.ones(B), rtol=1e-6, atol=1e-6)


def test_all_states_can_contribute() -> None:
    model = _build()
    x = torch.randn(B, K, D)

    with torch.inference_mode():
        projected = model.behavior_projection(x)
        scores = torch.tanh(projected).matmul(model.attention_vector)
        gamma = torch.softmax(scores, dim=1)

    assert gamma.shape == (B, K)
    assert torch.all(gamma > 0)
    assert torch.all(gamma < 1)


def test_gradients_reach_all_trainable_parameters() -> None:
    model = _build()
    model.train()
    x = torch.randn(B, K, D, requires_grad=True)

    y = model(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, parameter in model.named_parameters():
        assert parameter.requires_grad, name
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_deterministic_inference() -> None:
    model = _build()
    model.eval()
    x = torch.randn(B, K, D)

    with torch.inference_mode():
        y1 = model(x)
        y2 = model(x)

    assert torch.equal(y1, y2)


def test_state_order_is_semantically_respected() -> None:
    model = _build()

    # Make the attention mechanism uniform so that the expected output is
    # exactly the arithmetic mean. This verifies that the K dimension is
    # treated as the behavioral-state dimension and is not reshaped away.
    with torch.no_grad():
        model.behavior_projection.weight.zero_()
        model.attention_vector.zero_()

    x = torch.zeros(1, K, D)
    for k in range(K):
        x[0, k, :] = float(k + 1)

    with torch.inference_mode():
        y = model(x)

    expected = x.mean(dim=1)
    assert torch.allclose(y, expected)
