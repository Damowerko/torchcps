from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax


class SpatialSelfAttention(gnn.MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pos_dim: int,
        heads=1,
        dropout=0.0,
        **kwargs
    ):
        super().__init__(aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin_q = nn.Linear(in_channels, out_channels * heads)
        self.lin_k = nn.Linear(in_channels, out_channels * heads)
        self.lin_v = nn.Linear(in_channels, out_channels * heads)
        self.lin_pos = nn.Linear(pos_dim, out_channels * heads, bias=False)

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj
    ) -> torch.Tensor:
        query = self.lin_q(x)
        key = self.lin_k(x)
        value = self.lin_v(x)
        out = self.propagate(edge_index, query=query, key=key, value=value, pos=pos)
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
        # add positional encoding
        key_j += self.lin_pos(pos_i - pos_j)
        # reshape to separate heads and channels
        query_i = query_i.view(-1, self.heads, self.out_channels)
        key_j = key_j.view(-1, self.heads, self.out_channels)
        value_j = value_j.view(-1, self.heads, self.out_channels)
        # inner product between query and key
        alpha = (
            torch.einsum("...hc,...hc->...h", query_i, key_j) / self.out_channels**0.5
        )
        # compute attention coefficients along the first dimension
        alpha = softmax(alpha, index, ptr, size_i, dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = value_j * alpha.view(-1, self.heads, 1)
        out = out.view(-1, self.heads * self.out_channels)
        return out
