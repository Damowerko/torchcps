import pytest
import torch
from torch_cluster.radius import radius
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from torchcps.attention import SpatialAttention


@pytest.fixture
def data():
    x = torch.randn(10, 16)  # Example node features
    pos = torch.randn(10, 2)  # Example node positions
    edge_index = radius_graph(pos, r=0.5, batch=None, loop=True)
    return Data(x=x, pos=pos, edge_index=edge_index)


def test_self_forward(data):
    in_channels = 16
    out_channels = 32
    pos_dim = 2
    heads = 4
    dropout = 0.2

    attention = SpatialAttention(in_channels, out_channels, pos_dim, heads, dropout)
    output = attention(data.x, data.pos, data.edge_index)

    assert output.shape == (data.num_nodes, heads * out_channels)  # Check output shape


def test_cross_forward():
    n_x = 10
    n_y = 5

    x = torch.randn(n_x, 16)  # Example node features
    y = torch.randn(n_y, 16)  # Example node features
    pos_x = torch.randn(n_x, 2)
    pos_y = torch.randn(n_y, 2)
    edge_index = radius(x, y, r=0.5)

    in_channels = 16
    out_channels = 32
    pos_dim = 2
    heads = 4
    dropout = 0.2

    attention = SpatialAttention(in_channels, out_channels, pos_dim, heads, dropout)
    output = attention(
        (x, y),
        (pos_x, pos_y),
        edge_index,
    )

    assert output.shape == (
        n_y,
        heads * out_channels,
    )


def test_message(data):
    in_channels = 16
    out_channels = 32
    pos_dim = 2
    heads = 4
    dropout = 0.2

    attention = SpatialAttention(in_channels, out_channels, pos_dim, heads, dropout)
    query_i = torch.randn(data.num_edges, heads * out_channels)
    key_j = torch.randn(data.num_edges, heads * out_channels)
    value_j = torch.randn(data.num_edges, heads * out_channels)
    pos_i = data.pos[data.edge_index[0]]
    pos_j = data.pos[data.edge_index[1]]
    index = data.edge_index[0]
    ptr = None
    size_i = None

    output = attention.message(
        query_i, key_j, value_j, pos_i, pos_j, index, ptr, size_i
    )

    assert output.shape == (data.num_edges, heads * out_channels)  # Check output shape
