import typing
from typing import Callable

import torch
import torch.linalg
import torch.nn as nn
from pykeops.torch import LazyTensor
from torch_geometric.nn import MLP

from torchcps.kernel.rkhs import GaussianKernel, Kernel, Mixture


class KernelConv(nn.Module):
    @staticmethod
    def _grid_positions(
        max_filter_kernels: int,
        n_dimensions: int,
        kernel_spread: float = 1.0,
    ):
        """
        Arrange the kernels in a n-dimensional hypergrid.

        The number of kernels per dimension is determined by `(max_filter_kernels)**(1/n_dimensions)`.

        Args:
            max_filter_kernels: maximum number of kernels per filter
        """
        n_kernels_per_dimension = int(max_filter_kernels ** (1 / n_dimensions))
        # make sure the kernel width is odd
        if n_kernels_per_dimension % 2 == 0:
            n_kernels_per_dimension -= 1
        # spread the kernels out in a grid
        kernel_positions = kernel_spread * torch.linspace(
            n_kernels_per_dimension / 2,
            -n_kernels_per_dimension / 2,
            n_kernels_per_dimension,
        )
        # (n_kernels_per_dimension ** n_dimensions, n_dimensions)
        kernel_positions = torch.stack(
            torch.meshgrid(*([kernel_positions] * n_dimensions)), dim=-1
        )
        return kernel_positions

    @staticmethod
    def _uniform_positions(
        max_filter_kernels: int,
        n_dimensions: int,
        kernel_spread: float = 1.0,
    ):
        x = torch.rand((max_filter_kernels, n_dimensions))
        kernel_positions = kernel_spread * (x - 0.5)
        return kernel_positions

    def __init__(
        self,
        max_filter_kernels: int,
        in_channels: int,
        out_channels: int,
        n_dimensions: int,
        kernel_spread: float = 1.0,
        update_positions: bool = True,
        kernel_init: str = "uniform",
    ):
        super().__init__()

        # initialize kernel positions
        kernel_init_fn = {
            "grid": self._grid_positions,
            "uniform": self._uniform_positions,
        }[kernel_init]

        # (filter_kernels, n_dimensions)
        self.kernel_positions = nn.Parameter(
            kernel_init_fn(
                max_filter_kernels,
                n_dimensions,
                kernel_spread,
            ),
            requires_grad=update_positions,
        )

        # the actual number of filter kernels might be smaller than the maximum
        filter_kernels = self.kernel_positions.shape[0]

        # (filter_kernels, in_channels, out_channels)
        self.kernel_weights = nn.Parameter(
            torch.empty(filter_kernels, in_channels, out_channels)
        )
        nn.init.kaiming_normal_(self.kernel_weights)

    def forward(
        self,
        input: Mixture,
    ):
        """
        Args:
            input: namedtuple with the following fields:
                positions: (in_kernels, n_dimensions) positions of the input kernels.
                weights: (in_kernels, in_channels) weights of the input kernels.
                batch: (batch_size,) length of each batch or None if batch_size = 1.
        """
        input_positions, input_weights, input_batch = input

        # deduce dimensions from the filter kernel positions and shape
        n_dimensions = self.kernel_positions.shape[1]
        filter_kernels, in_channels, out_channels = self.kernel_weights.shape

        # the number of input and output kernels
        in_kernels = input_positions.shape[0]
        out_kernels = in_kernels * filter_kernels

        # Sanity Check Input
        assert (
            input_positions.shape[1] == n_dimensions
        ), f"Expected {n_dimensions} dimensions but got {input_positions.shape[1]} instead."
        assert (
            input_weights.shape[1] == in_channels
        ), f"Expected {in_channels} channels but got {input_weights.shape[1]} instead."

        # (in_kernels, filter_kernels, n_dimensions)
        output_positions = input_positions[:, None] + self.kernel_positions[None, :]
        output_positions = output_positions.reshape(out_kernels, n_dimensions)

        # input_weights.shape == (in_kernels, in_channels)
        # self.kernel_weights.shape == (filter_kernels, in_channels, out_channels)
        # output_weights.shape == (in_kernels, filter_kernels, out_channels)
        output_weights = torch.einsum("if,jfg->ijg", input_weights, self.kernel_weights)
        # combine the in_kernels and filter_kernels dimensions
        output_weights = output_weights.reshape(out_kernels, out_channels)

        if input_batch is not None:
            # before we combined the in_kernels and filter_kernels dimensions
            # in_kernels was at the 0th dimension, so we preserved the sorting of the batches
            # but the size of each batch is simply `filter_kernels` times bigger
            output_batch = input_batch * filter_kernels
        else:
            output_batch = None

        return Mixture(output_positions, output_weights, output_batch)


class KernelGraphFilter(nn.Module):
    """Propagate the kernel weights using a GNN; multiply by the GSO."""

    def __init__(
        self,
        kernel: Kernel,
        in_channels: int,
        out_channels: int,
        filter_taps: int,
        normalize=True,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.normalize = normalize
        self.filter_taps = filter_taps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear((filter_taps + 1) * in_channels, out_channels)

    def forward(self, input: Mixture):
        """
        Args:
            input: namedtuple with the following fields:
                positions: (in_kernels, n_dimensions) positions of the input kernels.
                weights: (in_kernels, in_channels) weights of the input kernels.
                batch: (batch_size,) length of each batch or None if batch_size = 1.
        """
        input_positions, input_weights, input_batch = input

        in_kernels, in_channels = input_weights.shape

        # The kernel matrix is the GSO
        # Will contain self-loops by definition
        S = self.kernel(input_positions, input_positions, input_batch, input_batch)

        # Computer powers of the kernel times x
        # xs = [K x, K^2 x, ...]
        xs = [input_weights]
        for _ in range(self.filter_taps):
            if self.normalize:
                # Normalize GSO to: S_norm = D^{-1/2} S D^{-1/2}
                degree_sqrt = S.sum(3).pow(0.5)
                x = (S @ (xs[-1] / degree_sqrt)) / degree_sqrt
            else:
                x = S @ xs[-1]
            assert isinstance(
                x, torch.Tensor
            ), f"Expected type torch.Tensor but got {type(x)} instead."
            xs += [x]

        # multiply by the filter weights
        output_weights = self.linear(
            torch.stack(xs, dim=2).reshape(
                in_kernels,
                (self.filter_taps + 1) * in_channels,
            )
        )
        return Mixture(input_positions, output_weights, input_batch)


class KernelSample(nn.Module):
    def __init__(
        self,
        kernel: Kernel = GaussianKernel(1.0),
        nonlinearity: nn.Module | None = None,
        alpha: float | None = None,
    ):
        """
        Sample the kernel at the output positions according to $ w_j = \\sum_i w_i K_{ij} $. Optionally solve the system $ (K + \\alpha I) w = z $.

        Args:
            kernel: kernel function
            nonlinearity: nonlinearity to apply to the kernel weights
            alpha: If provided then then the kernel weights are recomputed as $(K + \\alpha I)^{-1} z$. Otherwise, the kernel is merely sample.
        """

        super().__init__()
        self.kernel = kernel
        self.alpha = alpha
        self.nonlinearity = nonlinearity

    def forward(
        self,
        input: Mixture,
        output_positions: torch.Tensor,
        output_batch: torch.Tensor | None = None,
    ):
        """
        Args:
            input: namedtuple with the following fields:
                positions: (in_kernels, n_dimensions) positions of the input kernels.
                weights: (in_kernels, in_channels) weights of the input kernels.
                batch: (batch_size,) length of each batch or None if batch_size = 1.
            output_positions: (in_kernels, n_dimensions)
            output_batch: (batch_size,) length of each batch or None.
        """
        sample = sample_kernel(
            self.kernel,
            input,
            output_positions,
            output_batch=output_batch,
            nonlinearity=self.nonlinearity,
        )
        if self.alpha is None:
            return sample
        else:
            solution = solve_kernel(self.kernel, sample, self.alpha)
            return solution


def sample_kernel(
    kernel: Kernel,
    input: Mixture,
    output_positions: torch.Tensor,
    output_batch: torch.Tensor | None = None,
    nonlinearity: Callable | None = None,
):
    """
    Sample the kernel at the given output positions.

    Args:
        kernel (Kernel): The kernel function.
        input (Mixture): a namedtuple with the following fields:
            positions: (in_kernels, n_dimensions) positions of the input kernels.
            weights: (in_kernels, in_channels) weights of the input kernels.
            batch: (batch_size,) length of each batch or None if batch_size = 1.
        output_positions (torch.Tensor): The output positions.
        output_batch (torch.Tensor, optional): The output batch tensor. Defaults to None.
        nonlinearity (Callable, optional): The nonlinearity function. Defaults to None.

    Returns:
        Mixture: The resulting mixture.
    """
    input_positions, input_weights, input_batch = input
    # sample kernel at output positions
    K_ij = kernel(
        output_positions,
        input_positions,
        output_batch,
        input_batch,
    )
    samples = K_ij @ input_weights
    if nonlinearity is not None:
        samples = nonlinearity(samples)
    return Mixture(output_positions, samples, output_batch)


def solve_kernel(
    kernel: Kernel,
    input: Mixture,
    alpha=1e-3,
):
    """
    Args:
        input: namedtuple with the following fields:
            positions: (in_kernels, n_dimensions) positions of the input kernels.
            weights: (in_kernels, in_channels) weights of the input kernels.
            batch: (batch_size,) length of each batch or None if batch_size = 1.
    """
    positions, weights, batch = input
    K = kernel(positions, positions, batch, batch)
    if isinstance(K, torch.Tensor):
        raise NotImplementedError()
    weights = K.solve(LazyTensor(weights[..., :, None, :]), alpha=alpha)  # type: ignore
    assert isinstance(
        weights, torch.Tensor
    ), f"Expected type torch.Tensor but got {type(weights)} instead."
    return Mixture(positions, weights, batch)


class KernelPool(nn.Module):
    """Reduce then number of kernels."""

    def __init__(
        self,
        out_kernels: int,
        kernel: Kernel,
        nonlinearity: nn.Module | None = None,
        strategy="largest",
        max_iter=100,
    ) -> None:
        super().__init__()
        self.out_kernels = out_kernels
        self.kernel = kernel
        self.nonlinearity = nonlinearity
        self.strategy = strategy
        self.max_iter = max_iter

    def forward(
        self,
        input: Mixture,
    ):
        """
        Args:
            input: namedtuple with the following fields:
                positions: (in_kernels, n_dimensions) positions of the input kernels.
                weights: (in_kernels, in_channels) weights of the input kernels.
                batch: (batch_size,) length of each batch or None if batch_size = 1.
        """
        input_positions, input_weights, input_batch = input
        if input_batch is not None:
            raise NotImplementedError("Batching not supported yet.")

        if input_positions.shape[0] <= self.out_kernels:
            indices = None
        elif self.strategy == "largest":
            indices = self._largest(input_positions, input_weights)
        elif self.strategy == "random":
            indices = self._random(input_positions, input_weights)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}.")

        if indices is not None:
            output_positions = torch.gather(
                input_positions,
                0,
                indices[..., None].expand(-1, input_positions.shape[-1]),
            )
            output_weights = torch.gather(
                input_weights,
                0,
                indices[..., None].expand(-1, input_weights.shape[-1]),
            )
        else:
            output_positions = input_positions
            output_weights = input_weights
        return Mixture(output_positions, output_weights)

    def _largest(
        self,
        input_positions: torch.Tensor,
        input_weights: torch.Tensor,
    ):
        """
        Args:
            input_positions: (in_kernels, n_channels, n_dimensions)
            input_weights: (in_kernels, n_channels, n_weights)
        """
        # take the norm of the weight vectors
        weights_norm = input_weights.norm(dim=-1)
        # find the top-k largest weights in magnitude in each channel and batch
        _, indices = torch.topk(weights_norm, dim=0, k=self.out_kernels)
        return indices

    def _random(
        self,
        input_positions: torch.Tensor,
        input_weights: torch.Tensor,
    ):
        """
        Args:
            input_positions: (in_kernels, channels, n_dimensions)
            input_weights: (in_kernels, channels, n_weights)
        """
        with torch.no_grad():
            device = input_weights.device
            in_kernels, _ = input_weights.shape
            random = torch.rand(in_kernels, device=device)
            _, indices = torch.topk(random, self.out_kernels, dim=0)
            return indices


class KNNBlock(nn.Module):
    def __init__(
        self,
        conv: KernelConv,
        sample: KernelSample,
        norm: nn.Module,
        delta_module: nn.Module,
    ):
        super().__init__()
        self.conv = conv
        self.sample = sample
        self.norm = norm
        self.delta_module = delta_module

    def forward(
        self,
        x: Mixture,
    ):
        # convolution and sampling operation
        y = self.conv.forward(x)
        y = y.map_weights(self.norm)
        y = self.sample.forward(y, x.positions, x.batch)

        # pointwise update of the kernel weights and positions
        delta_out = self.delta_module.forward(x.weights)
        delta_positions, delta_weights = torch.split(
            delta_out, [x.positions.shape[-1], x.weights.shape[-1]], dim=-1
        )
        y = Mixture(
            x.positions + delta_positions,
            x.weights + delta_weights,
            x.batch,
        )
        return y


class KNN(nn.Module):
    def __init__(
        self,
        n_dimensions: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int,
        n_layers_mlp: int,
        hidden_channels_mlp: int,
        sigma: float | typing.Sequence[float],
        max_filter_kernels: int,
        update_positions: bool,
        alpha: float | None = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.nonlinearity = nn.LeakyReLU()

        self.readin = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels_mlp,
            out_channels=hidden_channels,
            num_layers=n_layers_mlp,
            act=self.nonlinearity,
            norm="batch_norm",
            plain_last=False,
        )
        self.readout = MLP(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels_mlp,
            out_channels=out_channels,
            num_layers=n_layers_mlp,
            act=self.nonlinearity,
            norm="batch_norm",
            plain_last=True,
        )
        sigma = sigma if isinstance(sigma, typing.Sequence) else [sigma] * n_layers

        blocks: list[KNNBlock] = []
        for l in range(self.n_layers):
            blocks += [
                KNNBlock(
                    conv=KernelConv(
                        max_filter_kernels=max_filter_kernels,
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        n_dimensions=n_dimensions,
                        kernel_spread=sigma[l] * max_filter_kernels**0.5,
                        update_positions=update_positions,
                        kernel_init="uniform",
                    ),
                    sample=KernelSample(
                        kernel=GaussianKernel(sigma[l]),
                        alpha=alpha,
                        nonlinearity=nn.LeakyReLU(),
                    ),
                    norm=nn.BatchNorm1d(hidden_channels),
                    delta_module=MLP(
                        in_channels=hidden_channels,
                        hidden_channels=hidden_channels_mlp,
                        out_channels=hidden_channels + n_dimensions,
                        num_layers=n_layers_mlp,
                        act=self.nonlinearity,
                        norm="batch_norm",
                        plain_last=True,
                    ),
                )
            ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Mixture):
        x = x.map_weights(self.readin.forward)
        for l in range(self.n_layers):
            x = typing.cast(KNNBlock, self.blocks[l]).forward(x)
        x = x.map_weights(self.readout.forward)
        return x
