import pytest
import torch
from torch_geometric.data import Batch, Data

from src.common.graph.graph_builder import GraphBuilder
from src.models.hdeg.graph_reasoner import GraphReasoner


def _build_graph_batch(
    batch_size: int = 2,
    num_nodes: int = 5,
    embedding_dim: int = 8,
    top_k: int = 2,
) -> tuple[Batch, torch.Tensor]:

    embeddings = torch.randn(
        batch_size,
        num_nodes,
        embedding_dim,
    )

    graph_builder = GraphBuilder(
        top_k=top_k,
        self_loops=False,
        symmetric=False,
    )

    graph = graph_builder.build(
        embeddings
    )

    return graph, embeddings


def test_constructor_accepts_valid_configuration():
    reasoner = GraphReasoner(
        embedding_dim=64
    )

    assert reasoner.embedding_dim == 64
    assert reasoner.heads == 1
    assert reasoner.dropout == 0.0


def test_constructor_rejects_invalid_embedding_dimension():
    with pytest.raises(ValueError):
        GraphReasoner(
            embedding_dim=0
        )

    with pytest.raises(ValueError):
        GraphReasoner(
            embedding_dim=-1
        )


def test_constructor_rejects_invalid_heads():
    with pytest.raises(ValueError):
        GraphReasoner(
            embedding_dim=8,
            heads=0,
        )

    with pytest.raises(ValueError):
        GraphReasoner(
            embedding_dim=8,
            heads=-1,
        )


def test_constructor_rejects_invalid_dropout():
    with pytest.raises(ValueError):
        GraphReasoner(
            embedding_dim=8,
            dropout=-0.1,
        )

    with pytest.raises(ValueError):
        GraphReasoner(
            embedding_dim=8,
            dropout=1.0,
        )


def test_forward_returns_expected_shape():
    batch_size = 3
    num_nodes = 6
    embedding_dim = 16

    graph, _ = _build_graph_batch(
        batch_size=batch_size,
        num_nodes=num_nodes,
        embedding_dim=embedding_dim,
        top_k=2,
    )

    reasoner = GraphReasoner(
        embedding_dim=embedding_dim
    )

    output = reasoner(graph)

    assert output.shape == (
        batch_size,
        num_nodes,
        embedding_dim,
    )


def test_forward_preserves_one_representation_per_device():
    batch_size = 4
    num_nodes = 7
    embedding_dim = 16

    graph, _ = _build_graph_batch(
        batch_size=batch_size,
        num_nodes=num_nodes,
        embedding_dim=embedding_dim,
        top_k=3,
    )

    reasoner = GraphReasoner(
        embedding_dim=embedding_dim
    )

    output = reasoner(graph)

    assert output.size(0) == batch_size
    assert output.size(1) == num_nodes
    assert output.size(2) == embedding_dim


def test_forward_output_is_finite():
    graph, _ = _build_graph_batch(
        batch_size=2,
        num_nodes=5,
        embedding_dim=8,
        top_k=2,
    )

    reasoner = GraphReasoner(
        embedding_dim=8
    )

    output = reasoner(graph)

    assert torch.isfinite(output).all()


def test_forward_rejects_wrong_input_type():
    reasoner = GraphReasoner(
        embedding_dim=8
    )

    with pytest.raises(TypeError):
        reasoner(
            torch.randn(2, 5, 8)
        )


def test_forward_rejects_wrong_embedding_dimension():
    graph, _ = _build_graph_batch(
        batch_size=2,
        num_nodes=5,
        embedding_dim=8,
        top_k=2,
    )

    reasoner = GraphReasoner(
        embedding_dim=16
    )

    with pytest.raises(ValueError):
        reasoner(graph)


def test_forward_rejects_non_finite_node_features():
    graph, _ = _build_graph_batch(
        batch_size=2,
        num_nodes=5,
        embedding_dim=8,
        top_k=2,
    )

    graph.x[0, 0] = float("nan")

    reasoner = GraphReasoner(
        embedding_dim=8
    )

    with pytest.raises(ValueError):
        reasoner(graph)


def test_forward_rejects_invalid_edge_index_shape():
    graph, _ = _build_graph_batch(
        batch_size=2,
        num_nodes=5,
        embedding_dim=8,
        top_k=2,
    )

    graph.edge_index = torch.zeros(
        3,
        graph.edge_index.size(1),
        dtype=torch.long,
    )

    reasoner = GraphReasoner(
        embedding_dim=8
    )

    with pytest.raises(ValueError):
        reasoner(graph)


def test_forward_rejects_invalid_edge_index_dtype():
    graph, _ = _build_graph_batch(
        batch_size=2,
        num_nodes=5,
        embedding_dim=8,
        top_k=2,
    )

    graph.edge_index = graph.edge_index.float()

    reasoner = GraphReasoner(
        embedding_dim=8
    )

    with pytest.raises(TypeError):
        reasoner(graph)


def test_forward_rejects_out_of_range_edge_index():
    graph, _ = _build_graph_batch(
        batch_size=2,
        num_nodes=5,
        embedding_dim=8,
        top_k=2,
    )

    graph.edge_index[0, 0] = graph.num_nodes

    reasoner = GraphReasoner(
        embedding_dim=8
    )

    with pytest.raises(ValueError):
        reasoner(graph)


def test_forward_rejects_unequal_graph_sizes():
    graph_1 = Data(
        x=torch.randn(4, 8),
        edge_index=torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=torch.long,
        ),
    )

    graph_2 = Data(
        x=torch.randn(5, 8),
        edge_index=torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=torch.long,
        ),
    )

    graph = Batch.from_data_list(
        [graph_1, graph_2]
    )

    reasoner = GraphReasoner(
        embedding_dim=8
    )

    with pytest.raises(ValueError):
        reasoner(graph)


def test_graph_reasoner_is_differentiable():
    graph, embeddings = _build_graph_batch(
        batch_size=2,
        num_nodes=5,
        embedding_dim=8,
        top_k=2,
    )

    graph.x.requires_grad_(True)

    reasoner = GraphReasoner(
        embedding_dim=8
    )

    output = reasoner(graph)

    loss = output.mean()

    loss.backward()

    assert graph.x.grad is not None
    assert torch.isfinite(
        graph.x.grad
    ).all()

    parameter_gradients = [
        parameter.grad
        for parameter in reasoner.parameters()
        if parameter.requires_grad
    ]

    assert parameter_gradients

    assert all(
        gradient is not None
        for gradient in parameter_gradients
    )

    assert all(
        torch.isfinite(gradient).all()
        for gradient in parameter_gradients
    )


def test_graph_structure_affects_output():
    torch.manual_seed(42)

    embeddings = torch.randn(
        1,
        5,
        8,
    )

    builder = GraphBuilder(
        top_k=2,
        self_loops=False,
        symmetric=False,
    )

    graph_1 = builder.build(
        embeddings
    )

    # Construct a different topology while preserving the same
    # node features.
    graph_2 = graph_1.clone()

    reversed_edges = graph_2.edge_index.flip(0)

    graph_2.edge_index = reversed_edges

    reasoner = GraphReasoner(
        embedding_dim=8
    )

    reasoner.eval()

    output_1 = reasoner(
        graph_1
    )

    output_2 = reasoner(
        graph_2
    )

    assert not torch.allclose(
        output_1,
        output_2,
    )


def test_deterministic_inference():
    torch.manual_seed(123)

    graph, _ = _build_graph_batch(
        batch_size=2,
        num_nodes=6,
        embedding_dim=8,
        top_k=2,
    )

    reasoner = GraphReasoner(
        embedding_dim=8,
        dropout=0.0,
    )

    reasoner.eval()

    with torch.no_grad():
        output_1 = reasoner(graph)
        output_2 = reasoner(graph)

    assert torch.allclose(
        output_1,
        output_2,
    )


def test_gat_output_dimension_matches_embedding_dimension():
    reasoner = GraphReasoner(
        embedding_dim=32,
        heads=4,
    )

    assert reasoner.gat.out_channels == 32
    assert reasoner.gat.heads == 4
    assert reasoner.gat.concat is False


def test_batch_assignment_is_preserved():
    graph, _ = _build_graph_batch(
        batch_size=3,
        num_nodes=5,
        embedding_dim=8,
        top_k=2,
    )

    reasoner = GraphReasoner(
        embedding_dim=8
    )

    output = reasoner(graph)

    assert output.size(0) == graph.num_graphs

    assert output.size(1) == (
        graph.num_nodes // graph.num_graphs
    )