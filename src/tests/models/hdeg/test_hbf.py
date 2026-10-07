import torch

from src.models.hdeg.hbf import HierarchicalBehavioralForecaster


def make_inputs(B=4, N=23, K=9, D=64):
    return (
        torch.randn(B, N, D),
        torch.randn(B, K, D),
        torch.randn(B, K, D),
        torch.randn(B, D),
    )


def test_shapes_and_finiteness():
    model = HierarchicalBehavioralForecaster(num_devices=23, num_states=9, embedding_dim=64)
    outputs = model(*make_inputs())
    assert outputs["Z"].shape == (4, 23, 64)
    assert outputs["S"].shape == (4, 9, 64)
    assert outputs["S_tilde"].shape == (4, 9, 64)
    assert outputs["g"].shape == (4, 64)
    assert all(torch.isfinite(v).all() for v in outputs.values())


def test_shared_dynamics_affects_all_heads():
    torch.manual_seed(1)
    model = HierarchicalBehavioralForecaster(num_devices=23, num_states=9, embedding_dim=64)
    inputs = make_inputs(B=2)
    outputs1 = model(*inputs)
    inputs2 = tuple(x.clone() for x in inputs)
    inputs2 = (inputs2[0], inputs2[1], inputs2[2], inputs2[3] + 1.0)
    outputs2 = model(*inputs2)
    assert any(not torch.allclose(outputs1[k], outputs2[k]) for k in outputs1)


def test_gradients_propagate():
    model = HierarchicalBehavioralForecaster(num_devices=23, num_states=9, embedding_dim=64)
    inputs = tuple(x.requires_grad_(True) for x in make_inputs(B=2))
    outputs = model(*inputs)
    loss = sum(v.square().mean() for v in outputs.values())
    loss.backward()
    assert all(x.grad is not None for x in inputs)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_deterministic_eval():
    torch.manual_seed(42)
    model = HierarchicalBehavioralForecaster(num_devices=23, num_states=9, embedding_dim=64)
    model.eval()
    inputs = make_inputs(B=2)
    a = model(*inputs)
    b = model(*inputs)
    for key in a:
        assert torch.equal(a[key], b[key])


def test_hierarchical_completeness_and_order():
    model = HierarchicalBehavioralForecaster(num_devices=5, num_states=3, embedding_dim=8)
    outputs = model(*make_inputs(B=2, N=5, K=3, D=8))
    assert tuple(outputs) == ("Z", "S", "S_tilde", "g")
