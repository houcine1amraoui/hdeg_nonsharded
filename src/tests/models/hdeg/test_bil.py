import torch

from src.models.hdeg.bil import FROZEN_BIG_EDGES, BehavioralInteractionLearner, build_frozen_big_edge_index


def test_exact_big():
    e = build_frozen_big_edge_index()
    assert e.shape == (2, 7)
    assert tuple(map(tuple, e.t().tolist())) == FROZEN_BIG_EDGES
    assert not torch.any(e[0] == e[1])


def test_incoming_neighbors():
    m = BehavioralInteractionLearner(8)
    assert m.incoming_neighbors() == ((), (0,), (), (4, 7), (1,), (1,), (8,), (1,), ())


def test_interface_and_finiteness():
    m = BehavioralInteractionLearner(8)
    x = torch.randn(4, 9, 8)
    y = m(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_isolated_states_are_exact_identity():
    torch.manual_seed(7)
    m = BehavioralInteractionLearner(8)
    x = torch.randn(3, 9, 8)
    y = m(x)
    for k in (0, 2, 8):
        assert torch.equal(y[:, k, :], x[:, k, :])


def test_orientation():
    m = BehavioralInteractionLearner(8)
    edges = m.edge_index.t().tolist()
    assert [1, 5] in edges
    assert [5, 1] not in edges
    assert [4, 3] in edges and [7, 3] in edges
    assert [3, 4] not in edges and [3, 7] not in edges


def test_unconnected_state_does_not_receive_context():
    torch.manual_seed(11)
    m = BehavioralInteractionLearner(8)
    x1 = torch.randn(2, 9, 8)
    x2 = x1.clone()
    x2[:, 5, :] += 100.0  # lighting -> occupancy is NOT a BIG edge
    y1 = m(x1)
    y2 = m(x2)
    assert torch.equal(y1[:, 1, :], y2[:, 1, :])


def test_connected_source_can_change_target():
    torch.manual_seed(12)
    m = BehavioralInteractionLearner(8)
    x1 = torch.randn(2, 9, 8)
    x2 = x1.clone()
    x2[:, 1, :] += 100.0  # occupancy -> lighting is a BIG edge
    y1 = m(x1)
    y2 = m(x2)
    assert not torch.equal(y1[:, 5, :], y2[:, 5, :])


def test_differentiability():
    torch.manual_seed(13)
    m = BehavioralInteractionLearner(8)
    x = torch.randn(2, 9, 8, requires_grad=True)
    y = m(x)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_deterministic_inference():
    torch.manual_seed(14)
    m = BehavioralInteractionLearner(8).eval()
    x = torch.randn(3, 9, 8)
    with torch.no_grad():
        y1 = m(x)
        y2 = m(x)
    assert torch.equal(y1, y2)


def test_invalid_graph_rejected():
    e = build_frozen_big_edge_index()
    try:
        BehavioralInteractionLearner(8, edge_index=e.flip(0))
    except ValueError:
        pass
    else:
        raise AssertionError('Reversed BIG was not rejected.')


if __name__ == '__main__':
    tests = [v for k, v in globals().items() if k.startswith('test_')]
    for test in tests:
        test()
    print(f'PASS: {len(tests)} BIL contract/orientation tests')