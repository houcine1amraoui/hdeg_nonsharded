import pytest
import torch

from src.models.hdeg.dbrl import DBRL


def test_dbrl_constructor():
    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=16,
        gru_layers=1,
        graph_top_k=2,
    )

    assert model.input_dim == 1
    assert model.hidden_dim == 16
    assert model.embedding_dim == 16
    assert model.gru_layers == 1

    assert model.graph_builder.top_k == 2
    assert model.graph_builder.self_loops is False
    assert model.graph_builder.symmetric is False

    assert model.graph_reasoner.embedding_dim == 16


def test_dbrl_forward_shape():
    batch_size = 4
    window_size = 30
    num_devices = 8
    embedding_dim = 16

    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=embedding_dim,
        graph_top_k=2,
    )

    x = torch.randn(
        batch_size,
        window_size,
        num_devices,
    )

    output = model(x)

    assert output.shape == (
        batch_size,
        num_devices,
        embedding_dim,
    )


def test_dbrl_produces_one_representation_per_device():
    batch_size = 3
    window_size = 20
    num_devices = 7

    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=16,
        graph_top_k=2,
    )

    x = torch.randn(
        batch_size,
        window_size,
        num_devices,
    )

    output = model(x)

    assert output.size(0) == batch_size
    assert output.size(1) == num_devices


def test_dbrl_output_is_finite():
    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=16,
        graph_top_k=2,
    )

    x = torch.randn(
        2,
        20,
        6,
    )

    output = model(x)

    assert torch.isfinite(output).all()


def test_dbrl_rejects_wrong_input_rank():
    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=16,
        graph_top_k=2,
    )

    x = torch.randn(
        2,
        20,
    )

    with pytest.raises(ValueError):
        model(x)


def test_dbrl_rejects_integer_input():
    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=16,
        graph_top_k=2,
    )

    x = torch.randint(
        low=0,
        high=2,
        size=(2, 20, 6),
    )

    with pytest.raises(TypeError):
        model(x)


def test_dbrl_rejects_non_finite_input():
    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=16,
        graph_top_k=2,
    )

    x = torch.randn(
        2,
        20,
        6,
    )

    x[0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        model(x)


def test_dbrl_rejects_invalid_top_k():
    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=16,
        graph_top_k=6,
    )

    x = torch.randn(
        2,
        20,
        6,
    )

    # With self-loops disabled, maximum valid k is N - 1.
    with pytest.raises(ValueError):
        model(x)


def test_dbrl_is_differentiable_end_to_end():
    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=16,
        graph_top_k=2,
    )

    x = torch.randn(
        2,
        20,
        6,
        requires_grad=True,
    )

    output = model(x)

    loss = output.mean()

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters

    assert all(
        parameter.grad is not None
        for parameter in trainable_parameters
    )

    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in trainable_parameters
    )


def test_dbrl_temporal_embeddings_have_expected_shape():
    batch_size = 3
    window_size = 20
    num_devices = 7
    embedding_dim = 16

    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=embedding_dim,
        graph_top_k=2,
    )

    x = torch.randn(
        batch_size,
        window_size,
        num_devices,
    )

    temporal_embeddings = (
        model._encode_temporal_behavior(x)
    )

    assert temporal_embeddings.shape == (
        batch_size,
        num_devices,
        embedding_dim,
    )


def test_dbrl_uses_graph_reasoner():
    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=16,
        graph_top_k=2,
    )

    x = torch.randn(
        2,
        20,
        6,
    )

    captured = {}

    original_forward = (
        model.graph_reasoner.forward
    )

    def wrapped_forward(graph):
        captured["graph"] = graph
        return original_forward(graph)

    model.graph_reasoner.forward = wrapped_forward

    output = model(x)

    assert "graph" in captured
    assert captured["graph"].num_graphs == 2
    assert captured["graph"].num_nodes == 12

    assert output.shape == (
        2,
        6,
        16,
    )


def test_dbrl_graph_is_built_from_temporal_embeddings():
    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=16,
        graph_top_k=2,
    )

    x = torch.randn(
        2,
        20,
        6,
    )

    captured = {}

    original_build = (
        model.graph_builder.build
    )

    def wrapped_build(embeddings):
        captured["embeddings"] = embeddings.detach().clone()
        return original_build(embeddings)

    model.graph_builder.build = wrapped_build

    temporal_embeddings = (
        model._encode_temporal_behavior(x)
    )

    model(x)

    assert "embeddings" in captured

    assert captured["embeddings"].shape == (
        2,
        6,
        16,
    )

    assert torch.allclose(
        captured["embeddings"],
        temporal_embeddings,
        atol=1e-6,
    )


def test_dbrl_deterministic_inference():
    torch.manual_seed(42)

    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=16,
        graph_top_k=2,
        graph_dropout=0.0,
    )

    model.eval()

    x = torch.randn(
        2,
        20,
        6,
    )

    with torch.no_grad():
        output_1 = model(x)
        output_2 = model(x)

    assert torch.allclose(
        output_1,
        output_2,
    )


def test_dbrl_batch_samples_remain_independent_graphs():
    model = DBRL(
        input_dim=1,
        hidden_dim=16,
        embedding_dim=16,
        graph_top_k=2,
    )

    x = torch.randn(
        3,
        20,
        6,
    )

    captured = {}

    original_forward = (
        model.graph_reasoner.forward
    )

    def wrapped_forward(graph):
        captured["graph"] = graph
        return original_forward(graph)

    model.graph_reasoner.forward = wrapped_forward

    model(x)

    graph = captured["graph"]

    source = graph.edge_index[0]
    target = graph.edge_index[1]

    assert torch.equal(
        graph.batch[source],
        graph.batch[target],
    )