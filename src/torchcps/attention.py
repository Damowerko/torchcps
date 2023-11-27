from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax


class SpatialAttention(gnn.MessagePassing):
    def __init__(self, n_channels: int, pos_dim: int, heads=1, dropout=0.0, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.n_channels = n_channels
        self.heads = heads
        self.dropout = dropout
        self.lin_q = nn.Linear(n_channels * heads, n_channels * heads)
        self.lin_k = nn.Linear(n_channels * heads, n_channels * heads)
        self.lin_v = nn.Linear(n_channels * heads, n_channels * heads)
        self.lin_pos = nn.Linear(pos_dim, n_channels * heads, bias=False)

    def forward(
        self,
        x: torch.Tensor | PairTensor,
        pos: torch.Tensor | PairTensor,
        edge_index: Adj,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (num_nodes, in_channels) or a tuple of tensors if bipartite.
            pos: Node positions of shape (num_nodes, pos_dim) or a tuple of tensors if bipartite.
            edge_index: Tensor or sparse matrix representing the (bipartite) graph structure.
        """

        if isinstance(x, torch.Tensor):
            x = (x, x)
        if isinstance(pos, torch.Tensor):
            pos = (pos, pos)

        query = self.lin_q(x[1])
        key = self.lin_k(x[0])
        value = self.lin_v(x[0])
        size = (x[0].shape[0], x[1].shape[0])

        out = self.propagate(
            edge_index, query=query, key=key, value=value, pos=pos, size=size
        )
        return out

    def message(
        self,
        query_i: torch.Tensor,
        key_j: torch.Tensor,
        value_j: torch.Tensor,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
        index: torch.Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> torch.Tensor:
        # _i denotes target nodes, _j denotes source nodes
        # add positional encoding
        key_j += self.lin_pos(pos_i - pos_j)
        # reshape to separate heads and channels
        query_i = query_i.view(-1, self.heads, self.n_channels)
        key_j = key_j.view(-1, self.heads, self.n_channels)
        value_j = value_j.view(-1, self.heads, self.n_channels)
        # inner product between query and key
        alpha = (
            torch.einsum("...hc,...hc->...h", query_i, key_j) / self.n_channels**0.5
        )
        # softmax implementation from torch_geometric
        alpha = softmax(alpha, index, ptr, size_i, dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = value_j * alpha.view(-1, self.heads, 1)
        out = out.view(-1, self.heads * self.n_channels)
        return out
