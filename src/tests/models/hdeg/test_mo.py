from __future__ import annotations

import torch
from torch import nn

from src.models.hdeg.mo import ModelOptimization


def hierarchy(batch=4, N=23, K=9, D=64):
    return {
        "Z": torch.randn(batch, N, D),
        "S": torch.randn(batch, K, D),
        "S_tilde": torch.randn(batch, K, D),
        "g": torch.randn(batch, D),
    }


def test_four_level_losses_and_weighted_objective():
    observed = hierarchy()
    predicted = {k: v + 0.5 for k, v in observed.items()}
    mo = ModelOptimization(lambda_z=1.0, lambda_s=2.0, lambda_s_tilde=3.0, lambda_g=4.0)
    out = mo(observed, predicted)

    assert out.L_Z.ndim == 0
    assert out.L_S.ndim == 0
    assert out.L_S_tilde.ndim == 0
    assert out.L_G.ndim == 0
    expected = (
        out.L_Z + 2.0 * out.L_S + 3.0 * out.L_S_tilde + 4.0 * out.L_G
    )
    assert torch.equal(out.L_HDEG, expected)
    assert torch.isfinite(out.L_HDEG)


def test_semantic_isolation():
    observed = hierarchy()
    predicted = {k: v.clone() for k, v in observed.items()}
    predicted["S"] = predicted["S"] + 1.0
    mo = ModelOptimization()
    out = mo(observed, predicted)

    assert out.L_Z.item() == 0.0
    assert out.L_S_tilde.item() == 0.0
    assert out.L_G.item() == 0.0
    assert out.L_S.item() > 0.0
    assert torch.equal(out.L_HDEG, out.L_S)


def test_autograd_reaches_forecasting_model():
    observed = hierarchy()
    predictor = nn.ModuleDict({
        "Z": nn.Linear(64, 64),
        "S": nn.Linear(64, 64),
        "S_tilde": nn.Linear(64, 64),
        "g": nn.Linear(64, 64),
    })

    # Use a compact synthetic forecast path with gradients.
    predicted = {
        "Z": predictor["Z"](observed["Z"]),
        "S": predictor["S"](observed["S"]),
        "S_tilde": predictor["S_tilde"](observed["S_tilde"]),
        "g": predictor["g"](observed["g"]),
    }

    mo = ModelOptimization()
    out = mo(observed, predicted)
    out.L_HDEG.backward()

    for level in predictor:
        for parameter in predictor[level].parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum().item() > 0.0


def test_zero_residual_gives_zero_loss():
    observed = hierarchy()
    mo = ModelOptimization()
    out = mo(observed, observed)
    assert out.L_Z.item() == 0.0
    assert out.L_S.item() == 0.0
    assert out.L_S_tilde.item() == 0.0
    assert out.L_G.item() == 0.0
    assert out.L_HDEG.item() == 0.0


def test_validation_rejects_cross_level_shape_mismatch():
    observed = hierarchy()
    predicted = hierarchy()
    predicted["S"] = torch.randn(4, 8, 64)
    mo = ModelOptimization()
    try:
        mo(observed, predicted)
    except ValueError as exc:
        assert "S" in str(exc)
    else:
        raise AssertionError("Expected a shape mismatch error.")


def test_weights_are_non_learnable():
    mo = ModelOptimization(lambda_z=2.0, lambda_s=3.0, lambda_s_tilde=4.0, lambda_g=5.0)
    assert not mo.lambda_Z.requires_grad
    assert not mo.lambda_S.requires_grad
    assert not mo.lambda_S_tilde.requires_grad
    assert not mo.lambda_G.requires_grad
    assert list(mo.parameters()) == []
