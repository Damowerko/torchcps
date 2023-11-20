import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from torchcps.attention import SpatialSelfAttention


@pytest.fixture
def data():
    x = torch.randn(10, 16)  # Example node features
    pos = torch.randn(10, 2)  # Example node positions
    edge_index = radius_graph(pos, r=0.5, batch=None, loop=True)
    return Data(x=x, pos=pos, edge_index=edge_index)


def test_forward(data):
    in_channels = 16
    out_channels = 32
    pos_dim = 2
    heads = 4
    dropout = 0.2

    attention = SpatialSelfAttention(in_channels, out_channels, pos_dim, heads, dropout)
    output = attention(data.x, data.pos, data.edge_index)

    assert output.shape == (data.num_nodes, heads * out_channels)  # Check output shape


def test_message(data):
    in_channels = 16
    out_channels = 32
    pos_dim = 2
    heads = 4
    dropout = 0.2

    attention = SpatialSelfAttention(in_channels, out_channels, pos_dim, heads, dropout)
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
