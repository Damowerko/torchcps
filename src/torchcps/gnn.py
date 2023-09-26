import argparse
import typing
from typing import Callable, Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

activation_choices: typing.Dict[str, Type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
}

SequentialLayers = list[tuple[Callable, str] | Callable]


class GraphFilter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_taps: int,
        aggr: str | gnn.Aggregation = "sum",
    ):
        """
        Computes a polynomial MIMO graph filters. The input and output signals can have a different number of channels.
        .. math::
            \mathbf{y} = \sum_{k=0}^{K} W_k S^k \mathbf{x

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            n_taps: Number of taps in the filter.
            aggr: Aggregation scheme to use. For example, "add", "sum" "mean", "min", "max" or "mul".
                In addition, can be any Aggregation module (or any string that automatically resolves to it).
        """
        super().__init__()
        self.shift = gnn.SimpleConv(
            aggr=aggr,
            combine_root=None,
        )
        self.taps = nn.ModuleList(
            [gnn.Linear(in_channels, out_channels) for _ in range(n_taps + 1)]
        )

    def forward(
        self,
        x: torch.Tensor | OptPairTensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> torch.Tensor:
        # if isinstance(x, torch.Tensor):
        #     x = (x, x)

        y = self.taps[0](x)
        for i in range(1, len(self.taps)):
            x = self.shift(x, edge_index, edge_attr, size)
            y += self.taps[i](x)
        return y


class ResidualBlock(nn.Module):
    def __init__(
        self,
        conv: Callable,
        act: Callable[[torch.Tensor], torch.Tensor] | None = None,
        norm: Callable[[torch.Tensor], torch.Tensor] | None = None,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Residual block with a RES+ connection.
            Norm -> Activation -> Dropout -> Conv -> Residual

        Args:
            conv: Convolutional layer with input arguments (x, type_vec, adj_t).
            dropout: Dropout probability with input arguments (x).
            act: Activation function with input arguments (x).
            norm: Normalization function with input arguments (x, type_vec).
        """
        super().__init__(**kwargs)
        self.conv = conv
        self.norm = norm
        self.act = act
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        The first argument should be the input tensor, the remaining arguments are passed through to the `conv` module.
        """

        y = x
        if self.norm is not None:
            y = self.norm(y)
        if self.act is not None:
            y = self.act(y)
        y = self.dropout(y)
        y = self.conv(y, *args, **kwargs)
        return x + y


class GCN(nn.Module):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(GCN.__name__)
        group.add_argument(
            "--n_channels",
            type=int,
            default=32,
            help="Number of hidden features on each layer. Set to -1 for lazy initialization.",
        )
        group.add_argument(
            "--n_layers", type=int, default=2, help="Number of GNN layers."
        )
        group.add_argument(
            "--activation",
            type=str,
            default="leaky_relu",
            choices=list(activation_choices),
        )
        group.add_argument(
            "--mlp_read_layers",
            type=int,
            default=1,
            help="Number of MLP layers to use for readin/readout.",
        )
        group.add_argument(
            "--mlp_per_gnn_layers",
            type=int,
            default=0,
            help="Number of MLP layers to use per GNN layer.",
        )
        group.add_argument(
            "--mlp_hidden_channels",
            type=int,
            default=256,
            help="Number of hidden features to use in the MLP layers.",
        )
        group.add_argument(
            "--dropout", type=float, default=0.0, help="Dropout probability."
        )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_taps: int,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
        **kwargs,
    ):
        """
        A simple GNN model with a readin and readout MLP. The structure of the architecture is expressed using hyperparameters. This allows for easy hyperparameter search.

        Args:
            in_channels: Number of input features. Set to -1 for lazy initialization.
            out_channels: Number of output features.
            n_taps: Number of filter taps per layer.
            n_layers: Number of GNN layers.
            n_channels: Number of hidden features on each layer.
            n_taps: Number of filter taps per layer.
            activation: Activation function to use.
            read_layers: Number of MLP layers to use for readin/readout.
            read_hidden_channels: Number of hidden features to use in the MLP layers.
            residual: Type of residual connection to use: "res", "res+", "dense", "plain".
                https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DeepGCNLayer.html
            normalization: Type of normalization to use: "batch" or "layer".
        """
        super().__init__()
        if isinstance(activation, str):
            activation = activation_choices[activation]()

        if mlp_read_layers < 1:
            raise ValueError("mlp_read_layers must be >= 1.")

        # ensure that dropout is a float
        dropout = float(dropout)

        # Readin MLP: Changes the number of features from in_channels to n_channels
        self.readin = gnn.MLP(
            in_channels=in_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=n_channels,
            num_layers=mlp_read_layers,
            dropout=dropout,
            act=activation,
            plain_last=False,
        )

        # Readout MLP: Changes the number of features from n_channels to out_channels
        self.readout = gnn.MLP(
            in_channels=n_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=out_channels,
            num_layers=mlp_read_layers,
            dropout=dropout,
            act=activation,
            plain_last=True,
        )

        # GNN layers operate on n_channels features
        self.residual_blocks = nn.ModuleList()
        for _ in range(n_layers):
            conv: SequentialLayers = [
                (
                    GraphFilter(
                        in_channels=n_channels,
                        out_channels=n_channels,
                        n_taps=n_taps,
                    ),
                    "x, edge_index, edge_attr, size -> x",
                ),
            ]
            if mlp_per_gnn_layers > 0:
                conv += [
                    (activation, "x -> x"),
                    (
                        gnn.MLP(
                            in_channels=n_channels,
                            hidden_channels=mlp_hidden_channels,
                            out_channels=n_channels,
                            num_layers=mlp_per_gnn_layers,
                            dropout=dropout,
                            act=activation,
                            plain_last=True,
                        ),
                        "x -> x",
                    ),
                ]
            self.residual_blocks += [
                (
                    ResidualBlock(
                        conv=gnn.Sequential("x, edge_index, edge_attr, size", conv),
                        norm=gnn.BatchNorm(n_channels),
                        act=activation,
                        dropout=dropout,
                    )
                )
            ]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> torch.Tensor:
        x = self.readin(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x, edge_index, edge_attr, size)
        x = self.readout(x)
        return x


class GAT(nn.Module):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(GAT.__name__)
        group.add_argument(
            "--n_channels",
            type=int,
            default=32,
            help="Number of hidden features on each layer.",
        )
        group.add_argument(
            "--n_layers", type=int, default=2, help="Number of GNN layers."
        )
        group.add_argument(
            "--activation",
            type=str,
            default="leaky_relu",
            choices=list(GAT.activation_choices),
        )
        group.add_argument(
            "--mlp_read_layers",
            type=int,
            default=1,
            help="Number of MLP layers to use for readin/readout.",
        )
        group.add_argument(
            "--mlp_per_gnn_layers",
            type=int,
            default=0,
            help="Number of MLP layers to use per GNN layer.",
        )
        group.add_argument(
            "--mlp_hidden_channels",
            type=int,
            default=256,
            help="Number of hidden features to use in the MLP layers.",
        )
        group.add_argument(
            "--dropout", type=float, default=0.0, help="Dropout probability."
        )
        group.add_argument(
            "--residual_type",
            type=str,
            default="res+",
            choices=["res", "res+", "dense", "plain"],
            help="Type of residual connection to use.",
        )
        group.add_argument(
            "--heads",
            type=int,
            default=1,
            help="Number of attention heads to use.",
        )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_edges: int,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
        residual_type: str = "res+",
        heads: int = 1,
        **kwargs,
    ):
        """
        A simple GNN model with a readin and readout MLP. The structure of the architecture is expressed using hyperparameters. This allows for easy hyperparameter search.

        Args:
            in_channels: Number of input features.
            out_channels: Number of output features.
            n_layers: Number of GNN layers.
            n_channels: Number of hidden features on each layer.
            n_taps: Number of filter taps per layer.
            activation: Activation function to use.
            read_layers: Number of MLP layers to use for readin/readout.
            read_hidden_channels: Number of hidden features to use in the MLP layers.
            residual: Type of residual connection to use: "res", "res+", "dense", "plain".
                https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DeepGCNLayer.html
            normalization: Type of normalization to use: "batch" or "layer".
        """
        super().__init__()
        if isinstance(activation, str):
            activation = activation_choices[activation]()

        if mlp_read_layers < 1:
            raise ValueError("mlp_read_layers must be >= 1.")

        # ensure that dropout is a float
        dropout = float(dropout)

        # Readin MLP: Changes the number of features from in_channels to n_channels
        self.readin = gnn.MLP(
            in_channels=in_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=n_channels * heads,
            num_layers=mlp_read_layers,
            dropout=dropout,
            act=activation,
            plain_last=False,
        )

        # Readout MLP: Changes the number of features from n_channels to out_channels
        self.readout = gnn.MLP(
            in_channels=n_channels * heads,
            hidden_channels=mlp_hidden_channels,
            out_channels=out_channels,
            num_layers=mlp_read_layers,
            dropout=dropout,
            act=activation,
            plain_last=True,
        )

        # GNN layers operate on n_channels features
        gnn_layers = []
        for _ in range(n_layers):
            conv = gnn.Sequential(
                "x, edge_index, edge_weights",
                [
                    (
                        gnn.GATv2Conv(
                            in_channels=n_channels * heads,
                            out_channels=n_channels,
                            heads=heads,
                            edge_dim=n_edges,
                            dropout=dropout,
                            add_self_loops=False,
                        ),
                        "x, edge_index, edge_weights -> x",
                    ),
                    (
                        gnn.MLP(
                            in_channels=n_channels * heads,
                            hidden_channels=mlp_hidden_channels,
                            out_channels=n_channels * heads,
                            num_layers=mlp_per_gnn_layers,
                            dropout=dropout,
                            act=activation,
                            plain_last=True,
                        )
                        if mlp_per_gnn_layers > 0
                        else nn.Identity(),
                        "x -> x",
                    ),
                ],
            )
            gnn_layers += [
                (
                    gnn.DeepGCNLayer(
                        conv=conv,
                        norm=gnn.BatchNorm(n_channels * heads),
                        act=activation,
                        block=residual_type,
                        dropout=dropout,
                        ckpt_grad=False,
                    ),
                    "x, edge_index, edge_weights -> x",
                )
            ]
        self.gnn = gnn.Sequential("x, edge_index, edge_weights", gnn_layers)

    def forward(self, x, edge_index, edge_weights):
        x = self.readin(x)
        x = self.gnn(x, edge_index, edge_weights)
        x = self.readout(x)
        return x
