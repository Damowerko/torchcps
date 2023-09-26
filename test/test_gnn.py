import pytest
import torch_geometric.nn as gnn
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import FakeDataset, FakeHeteroDataset

from torchcps.gnn import GCN


@pytest.fixture(scope="module")
def data() -> Data:
    # fixture for fake dataset creation
    # return a data object
    data = FakeDataset(edge_dim=1, num_channels=32)[0]
    assert isinstance(data, Data)
    data = T.ToSparseTensor()(data)
    assert isinstance(data, Data)
    return data


@pytest.fixture(scope="module")
def hetero_data() -> HeteroData:
    # fixture for fake hetero dataset creation
    # return a hetero data object
    data = FakeHeteroDataset(edge_dim=1, avg_num_channels=32)[0]
    assert isinstance(data, HeteroData)
    data = T.ToSparseTensor()(data)
    assert isinstance(data, HeteroData)
    return data


def test_gcn(data: Data):
    model = GCN(in_channels=32, out_channels=10, n_channels=8, n_taps=4)
    out = model(data.x, data.adj_t)
    assert out.shape == (data.num_nodes, 10)


def test_gcn_lazy(data: Data):
    model = GCN(in_channels=-1, out_channels=10, n_channels=8, n_taps=4)
    out = model(data.x, data.adj_t)
    assert out.shape == (data.num_nodes, 10)
