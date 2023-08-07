from typing import Callable, NamedTuple

import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
from pykeops.torch import LazyTensor, Vi, Vj

from .kernel import Kernel


class RKHS(NamedTuple):
    positions: torch.Tensor
    weights: torch.Tensor


class KernelConv(nn.Module):
    """
    RKHS Convolutional Layer based on [1]. Signals are represented by $s(t) = \\sum_{i=1}^N a_i k(t_i,t)$, where $k$ is the kernel function and $a_i$ is the weight.
    The filters are similarly RKHS functions meaning that convolutions are computed as $\\sum_{i,j} a_i b_i k(t_i + t_j, t)$.

    [1] NONLINEAR SIGNAL PROCESSING OF CONTINUOUS-TIME SIGNALS VIA ALGEBRAIC NEURAL NETWORKS IN REPRODUCING KERNEL HILBERT SPACES.

    """

    def __init__(
        self,
        filter_kernels: int,
        in_channels: int,
        out_channels: int,
        n_dimensions: int,
        kernel_spread: float = 1.0,
        fixed_positions: bool = False,
        n_weights: int = 1,
    ):
        """
        Args:
            filter_kernels: number of kernels per filter
            in_channels: number of input channels
            out_channels: number of output channels
            n_dimensions: number of spatial dimensions
            kernel_width: width of the filter, i.e. how far away are the kernels positioned
            fixed_positions: if True, the kernel positions are fixed and spread out in a grid
        """
        super().__init__()

        if fixed_positions:
            n_kernels_per_dimension = int(round(filter_kernels ** (1 / n_dimensions)))
            # make sure the kernel width is odd
            if n_kernels_per_dimension % 2 == 0:
                n_kernels_per_dimension += 1
            # spread the kernels out in a grid
            kernel_positions = kernel_spread * torch.linspace(
                n_kernels_per_dimension / 2,
                -n_kernels_per_dimension / 2,
                n_kernels_per_dimension,
            )
            # (kernel_width ** n_dimensions, 1, 1, n_dimensions)
            kernel_positions = torch.stack(
                torch.meshgrid(*([kernel_positions] * n_dimensions)), dim=-1
            ).reshape(1, 1, -1, n_dimensions)
            # (in_channels, out_channels, kernel_width ** n_dimensions, n_dimensions)
            kernel_positions = kernel_positions.expand(
                in_channels, out_channels, kernel_positions.shape[2], n_dimensions
            )
            self.kernel_positions = nn.Parameter(kernel_positions, requires_grad=False)
            filter_kernels = self.kernel_positions.shape[2]
        else:
            self.kernel_positions = nn.Parameter(
                kernel_spread
                * torch.randn((in_channels, out_channels, filter_kernels, n_dimensions))
            )
        self.kernel_weights = nn.Parameter(
            torch.empty(in_channels, out_channels, filter_kernels, n_weights)
        )
        nn.init.xavier_uniform_(self.kernel_weights)

    def forward(self, input: RKHS):
        """
        Args:
            input_positions: (batch_size, in_channels, in_kernels, n_dimensions)
            input_weights: (batch_size, in_channels, in_kernels, n_kernel_diemsnions)
        """
        input_positions, input_weights = input
        (
            in_channels,
            out_channels,
            filter_kernels,
            n_dimensions,
        ) = self.kernel_positions.shape
        batch_size, _, in_kernels, _ = input_positions.shape
        out_kernels = in_channels * in_kernels * filter_kernels
        _, _, _, n_kernel_diemsnions = self.kernel_weights.shape

        # sanity checks
        assert (
            in_channels == input_positions.shape[1]
        ), f"{in_channels} != {input_positions.shape[1]}"
        assert (
            in_channels == input_weights.shape[1]
        ), f"{in_channels} != {input_weights.shape[1]}"
        assert (
            in_kernels == input_positions.shape[2]
        ), f"{in_kernels} != {input_positions.shape[2]}"
        assert (
            in_kernels == input_weights.shape[2]
        ), f"{in_kernels} != {input_weights.shape[2]}"
        assert (
            n_dimensions == input_positions.shape[3]
        ), f"{n_dimensions} != {input_positions.shape[3]}"
        assert (
            n_kernel_diemsnions == input_weights.shape[3]
        ), f"{n_kernel_diemsnions} != {input_weights.shape[3]}"

        # (batch_size, in_channels, 1, in_kernels, 1, n_dimensions)
        input_positions = input_positions.unsqueeze(2).unsqueeze(4)
        # (1, in_channels, out_channels, 1, filter_kernels, n_dimensions)
        kernel_positions = self.kernel_positions.unsqueeze(0).unsqueeze(3)
        # (batch_size, in_channels, out_channels, in_kernels, filter_kernels, n_dimensions)
        output_positions = input_positions + kernel_positions
        # to combine dimensions they must be contiguous so we swap in_channels and out_channels
        output_positions = output_positions.transpose(1, 2)
        output_positions = output_positions.reshape(
            batch_size, out_channels, out_kernels, n_dimensions
        )

        # (batch_size, in_channels, 1, in_kernels, 1, n_kernel_diemsnions)
        input_weights = input_weights.unsqueeze(2).unsqueeze(4)
        # (1, in_channels, out_channels, 1, filter_kernels, n_kernel_diemsnions)
        kernel_weights = self.kernel_weights.unsqueeze(0).unsqueeze(3)
        # (batch_size, in_channels, out_channels, in_kernels, filter_kernels, n_kernel_diemsnions)
        output_weights = input_weights * kernel_weights
        # to combine dimensions they must be contiguous so we swap in_channels and out_channels
        output_weights = output_weights.transpose(1, 2)
        output_weights = output_weights.reshape(
            batch_size, out_channels, out_kernels, n_kernel_diemsnions
        )

        return RKHS(output_positions, output_weights)


class KernelMap(nn.Module):
    def __init__(self, nonlinearity: nn.Module) -> None:
        """
        Args:
            nonlinearity: nonlinearity to apply to the kernel weights
        """

        super().__init__()
        self.nonlinearity = nonlinearity

    def forward(self, input: RKHS):
        """
        Args:
            input_positions: (batch_size, in_kernels, in_channels, n_dimensions)
            input_weights: (batch_size, in_kernels, in_channels)
        """
        output_weights = self.nonlinearity(input.weights)
        return RKHS(input.positions, output_weights)


class KernelMix(nn.Module):
    def __init__(
        self,
        kernel: Kernel,
        in_weights: int,
        out_weights: int,
        normalize=True,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.normalize = normalize
        self.linear = nn.Linear(in_weights, out_weights)

    def forward(self, input: RKHS):
        """
        Args:
            input_positions: (batch_size, in_kernels, n_channels, n_dimensions)
            input_weights: (batch_size, in_kernels, n_channels, in_weights)
        """
        K = self.kernel(input.positions, input.positions)
        if self.normalize:
            degree_sq = K.sum(3).pow(2)
            output_weights = (K @ (input.weights / degree_sq)) / degree_sq
        else:
            output_weights = K @ input.weights  # apply the linear layer to the weights
        output_weights = self.linear(output_weights)
        return RKHS(input.positions, output_weights)


class KernelPool(nn.Module):
    strategies = ["largest", "ransac"]

    def __init__(
        self,
        out_kernels: int,
        # kernel: Kernel,
        # nonlinearity: nn.Module | None = None,
        strategy="largest",
        max_iter=100,
    ) -> None:
        super().__init__()
        self.out_kernels = out_kernels
        # self.kernel = kernel
        # self.nonlinearity = nonlinearity
        self.strategy = strategy
        self.max_iter = max_iter

        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy {strategy}.")

    def forward(self, input: RKHS):
        """
        Args:
            input_positions: (batch_size, channels, in_kernels, n_dimensions)
            input_weights: (batch_size, channels, in_kernels)
        """
        input_positions, input_weights = input

        if input_positions.shape[2] <= self.out_kernels:
            indices = None
        elif self.strategy == "largest":
            indices = self._largest(input_positions, input_weights)
        elif self.strategy == "ransac":
            indices = self._ransac(input_positions, input_weights)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}.")

        if indices is not None:
            output_positions = torch.gather(
                input_positions,
                2,
                indices[..., None].expand(-1, -1, -1, input_positions.shape[-1]),
            )
            output_weights = torch.gather(
                input_weights,
                2,
                indices[..., None].expand(-1, -1, -1, input_weights.shape[-1]),
            )
        else:
            output_positions = input_positions
            output_weights = input_weights

        # # Find weights for new positions and nonlinearity
        # output_weights = fit_kernel_weights(
        #     input_positions,
        #     input_weights,
        #     output_positions,
        #     self.kernel,
        #     self.nonlinearity,
        # )
        return RKHS(output_positions, output_weights)

    def _largest(
        self,
        input_positions: torch.Tensor,
        input_weights: torch.Tensor,
    ):
        """
        Args:
            input_positions: (batch_size, channels, in_kernels, n_dimensions)
            input_weights: (batch_size, channels, in_kernels, n_weights)
        """
        with torch.no_grad():
            # take the norm of the weight vectors
            weights_norm = input_weights.norm(dim=3)
            # find the top-k largest weights in magnitude in each channel and batch
            _, indices = torch.topk(weights_norm, dim=2, k=self.out_kernels)
        return indices

    def _ransac(
        self,
        input_positions: torch.Tensor,
        input_weights: torch.Tensor,
    ):
        """
        Args:
            input_positions: (batch_size, channels, in_kernels, n_dimensions)
            input_weights: (batch_size, channels, in_kernels)
        """
        with torch.no_grad():
            device = input_positions.device
            batch_size, channels, _, n_dimensions = input_positions.shape
            best_indices = torch.empty(
                batch_size, channels, self.out_kernels, dtype=torch.long, device=device
            )
            best_product = torch.full(
                (batch_size, channels), -float("inf"), device=device
            )
            for _ in range(self.max_iter):
                _, random_indices = torch.topk(
                    torch.rand(input_weights.shape, device=device),
                    dim=2,
                    k=self.out_kernels,
                )
                output_weights = torch.gather(input_weights, 2, random_indices)
                output_positions = torch.gather(
                    input_positions,
                    2,
                    random_indices[..., None].expand(-1, -1, -1, n_dimensions),
                )
                # (batch_size, channels)
                products = self.kernel.inner(
                    input_positions,
                    input_weights,
                    output_positions,
                    output_weights,
                )
                # want to maximize the inner product
                better_mask = products > best_product
                best_product[better_mask] = products[better_mask]
                best_indices[better_mask] = random_indices[better_mask]
        return best_indices


def fit_kernel_weights(
    input_positions: torch.Tensor,
    input_weights: torch.Tensor,
    output_positions: torch.Tensor,
    kernel: Kernel,
    nonlinearity: Callable | None = None,
):
    """
    Fit a new RKHS that has fewer kernels than the input RKHS. The new kernels are positioned at `output_positions`.
    1. Sample kernel at `output_positions`.
    2. The weights are computed following (K + mu I)^-1 z, where K is the kernel matrix and z are the samples.

    Args:
        input_positions: (batch_size, channels, in_kernels, n_dimensions)
        input_weights: (batch_size, channels, in_kernels)
        output_positions: (batch_size, channels, out_kernels, n_dimensions)
        kernel: kernel function
        alpha: Non-negative regularization parameter.
    """
    batch_size, channels, in_kernels, n_dimensions = input_positions.shape
    _, _, out_kernels, _ = output_positions.shape
    assert input_weights.shape == (
        batch_size,
        channels,
        in_kernels,
    ), f"{input_weights.shape} != {(batch_size, channels, in_kernels)}"
    assert output_positions.shape == (
        batch_size,
        channels,
        out_kernels,
        n_dimensions,
    ), f"{output_positions.shape} != {(batch_size, channels, out_kernels, n_dimensions)}"

    # sample kernel at output positions
    # (batch_size, channels, out_kernels, in_kernels)
    samples = kernel(output_positions, input_positions) @ input_weights[..., None]
    if nonlinearity is not None:
        samples = nonlinearity(samples)

    K = kernel(output_positions, output_positions)
    if isinstance(K, torch.Tensor):
        raise NotImplementedError()
    assert 0 not in K.shape, f"K is not invertible with shape: {K.shape}"

    weights = K.solve(LazyTensor(samples[..., :, None]), eps=0.1)
    return weights.squeeze(-1)
