from typing import Callable, NamedTuple

import torch
import torch.linalg
import torch.nn as nn
from pykeops.torch import KernelSolve, LazyTensor
from pykeops.torch.cluster import (
    cluster_ranges_centroids,
    from_matrix,
    grid_cluster,
    sort_clusters,
)

from .rkhs import GaussianKernel, Kernel, Mixture


class KernelConv(nn.Module):
    @staticmethod
    def _grid_positions(
        max_filter_kernels: int,
        in_channels: int,
        out_channels: int,
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
        # (kernel_width ** n_dimensions, 1, 1, n_dimensions)
        kernel_positions = torch.stack(
            torch.meshgrid(*([kernel_positions] * n_dimensions)), dim=-1
        ).reshape(1, 1, -1, n_dimensions)
        # (in_channels, out_channels, kernel_width ** n_dimensions, n_dimensions)
        kernel_positions = kernel_positions.expand(
            in_channels, out_channels, kernel_positions.shape[2], n_dimensions
        )
        return kernel_positions

    @staticmethod
    def _uniform_positions(
        max_filter_kernels: int,
        in_channels: int,
        out_channels: int,
        n_dimensions: int,
        kernel_spread: float = 1.0,
    ):
        x = torch.rand((in_channels, out_channels, max_filter_kernels, n_dimensions))
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
        n_weights: int = 1,
        kernel_init: str = "uniform",
    ):
        super().__init__()
        # initialize kernel positions
        kernel_init_fn = {
            "grid": self._grid_positions,
            "uniform": self._uniform_positions,
        }[kernel_init]
        self.kernel_positions = nn.Parameter(
            kernel_init_fn(
                max_filter_kernels,
                in_channels,
                out_channels,
                n_dimensions,
                kernel_spread,
            ),
            requires_grad=update_positions,
        )
        # initialize kernel weights
        # the actual number of filter kernels might be smaller than the maximum
        filter_kernels = self.kernel_positions.shape[2]
        self.kernel_weights = nn.Parameter(
            torch.empty(in_channels, out_channels, filter_kernels, n_weights)
        )
        nn.init.kaiming_normal_(self.kernel_weights)

    def forward(self, input: Mixture):
        """
        Args:
            input_positions: (batch_size, in_channels, in_kernels, n_dimensions)
            input_weights: (batch_size, in_channels, in_kernels, n_weights)
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
        _, _, _, n_weights = self.kernel_weights.shape

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
            n_weights == input_weights.shape[3]
        ), f"{n_weights} != {input_weights.shape[3]}"

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
            batch_size, out_channels, out_kernels, n_weights
        )
        return Mixture(output_positions, output_weights)


class KernelGraphFilter(nn.Module):
    """Propagate the kernel weights using a GNN; multiply by the GSO."""

    def __init__(
        self,
        kernel: Kernel,
        in_weights: int,
        out_weights: int,
        filter_taps: int,
        normalize=True,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.normalize = normalize
        self.filter_taps = filter_taps
        self.in_weights = in_weights
        self.out_weights = out_weights
        self.linear = nn.Linear((filter_taps + 1) * in_weights, out_weights)

    def forward(self, input: Mixture):
        """
        Args:
            input_positions: (batch_size, n_channels, in_kernels, n_dimensions)
            input_weights: (batch_size, n_channels, in_kernels, in_weights)
        """
        batch_size, n_channels, in_kernels, in_weights = input.weights.shape

        # x = input.positions
        # w = input.weights
        # # Cluster the input positions for block-sparse reduction
        # eps = 1.0  # the grid size
        # x_labels = grid_cluster(x, eps)  # class labels
        # x_ranges, x_centroids, _ = cluster_ranges_centroids(x, x_labels, input.weights)
        # (x, w), x_labels = sort_clusters((x, w), x_labels)

        # # Compute a boolean mask that indicates which centroids are close to each other
        # D = ((x_centroids[:, None, :] - x_centroids[None, :, :]) ** 2).sum(2)
        # assert isinstance(
        #     self.kernel, GaussianKernel
        # ), "Only Gaussian kernel implemented."
        # keep = D < (4 * self.kernel.sigma) ** 2
        # ranges_ij = from_matrix(x_ranges, x_ranges, keep)

        # The kernel matrix is the GSO
        # Will contain self-loops by definition
        S = self.kernel(input.positions, input.positions)

        # Computer powers of the kernel times x
        # xs = [K x, K^2 x, ...]
        xs = [input.weights]
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

        # Perform graph filters
        output_weights = self.linear(
            torch.stack(xs, dim=4).reshape(
                batch_size,
                n_channels,
                in_kernels,
                (self.filter_taps + 1) * in_weights,
            )
        )
        return Mixture(input.positions, output_weights)


class KernelMap(nn.Module):
    """Apply a transformation to the kernel weights."""

    def __init__(self, transformation: nn.Module) -> None:
        """
        Args:
            transformation: function to apply to the kernel weights.
                input -> (batch_size, in_channels, in_kernels, in_weights)
                output -> (batch_size, in_channels, in_kernels, in_weights)
            over:
                "positions": apply the transformation to the positions
                "weights": apply the transformation to the weights
        """

        super().__init__()
        self.transformation = transformation

    def forward(self, input: Mixture):
        """
        Args:
            input_positions: (batch_size, in_channels, in_kernels, n_dimensions)
            input_weights: (batch_size, in_channels, in_kernels, in_weights)
        """
        output_weights = self.transformation(input.weights)
        return Mixture(input.positions, output_weights)


class KernelNorm(nn.Module):
    """Apply batch normalization to the kernel weights."""

    def __init__(self, n_channels: int, n_weights: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_weights = n_weights
        self.batch_norm = nn.InstanceNorm1d(n_channels * n_weights)

    def forward(self, input: Mixture):
        """
        Args:
            input_positions: (batch_size, n_channels, in_kernels, n_dimensions)
            input_weights: (batch_size, n_channels, in_kernels, in_weights)
        """
        batch_size, n_channels, in_kernels, n_weights = input.weights.shape
        assert (
            n_channels == self.n_channels
        ), f"Unexpected number of channels {n_channels} != {self.n_channels}"
        assert (
            n_weights == self.n_weights
        ), f"Unexpected number of weights {n_weights} != {self.n_weights}"

        weights: torch.Tensor = input.weights.transpose(2, 3).reshape(
            batch_size, n_channels * n_weights, in_kernels
        )
        weights = self.batch_norm(weights)
        weights = weights.reshape(
            batch_size, n_channels, n_weights, in_kernels
        ).transpose(2, 3)
        return Mixture(input.positions, weights)


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

    def forward(self, input: Mixture, output_positions: torch.Tensor):
        """
        Args:
            input_positions: (batch_size, channels, in_kernels, n_dimensions)
            input_weights: (batch_size, channels, in_kernels, n_weights)
            output_positions: (batch_size, channels, out_kernels, n_dimensions)
        """
        samples = sample_kernel(input, output_positions, self.kernel, self.nonlinearity)
        if self.alpha is None:
            return samples

        solution = solve_kernel(samples, self.kernel, self.alpha)
        return solution


class KernelPool(nn.Module):
    """Reduce then number of kernels."""

    def __init__(
        self,
        out_kernels: int,
        kernel: Kernel,
        nonlinearity: nn.Module | None = None,
        strategy="largest",
        max_iter=100,
        alpha=1e-6,
        fit=True,
    ) -> None:
        super().__init__()
        self.out_kernels = out_kernels
        self.kernel = kernel
        self.nonlinearity = nonlinearity
        self.strategy = strategy
        self.max_iter = max_iter
        self.alpha = alpha
        self.fit = fit

    def forward(self, input: Mixture):
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
        elif self.strategy == "random":
            indices = self._random(input_positions, input_weights)
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

        # Find weights for new positions and nonlinearity
        if self.fit:
            output_weights = fit_kernel_weights(
                input_positions,
                input_weights,
                output_positions,
                self.kernel,
                self.nonlinearity,
                self.alpha,
            )
        return Mixture(output_positions, output_weights)

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
        # take the norm of the weight vectors
        weights_norm = input_weights.norm(dim=3)
        # find the top-k largest weights in magnitude in each channel and batch
        _, indices = torch.topk(weights_norm, dim=2, k=self.out_kernels)
        return indices

    def _random(
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
            device = input_weights.device
            batch_size, channels, in_kernels, _ = input_positions.shape
            random = torch.rand(batch_size, channels, in_kernels, device=device)
            _, indices = torch.topk(random, self.out_kernels, dim=2)
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
        device = input_positions.device
        batch_size, channels, _, n_dimensions = input_positions.shape
        best_indices = torch.empty(
            batch_size, channels, self.out_kernels, dtype=torch.long, device=device
        )
        best_product = torch.full((batch_size, channels), -float("inf"), device=device)
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


def sample_kernel(
    input: Mixture,
    output_positions: torch.Tensor,
    kernel: Kernel,
    nonlinearity: Callable | None = None,
):
    # sample kernel at output positions
    # (batch_size, channels, out_kernels, in_kernels)
    samples = kernel(output_positions, input.positions) @ input.weights
    if nonlinearity is not None:
        samples = nonlinearity(samples)
    return Mixture(output_positions, samples)


def solve_kernel(
    samples: Mixture,
    kernel: Kernel,
    alpha=1e-3,
):
    K = kernel(samples.positions, samples.positions)
    if isinstance(K, torch.Tensor):
        raise NotImplementedError()
    assert 0 not in K.shape, f"K is not invertible with shape: {K.shape}"
    weights = K.solve(LazyTensor(samples.weights[..., :, None, :]), alpha=alpha)
    assert isinstance(weights, torch.Tensor)
    return Mixture(samples.positions, weights)


def fit_kernel_weights(
    input_positions: torch.Tensor,
    input_weights: torch.Tensor,
    output_positions: torch.Tensor,
    kernel: Kernel,
    nonlinearity: Callable | None = None,
    alpha=1e-3,
) -> torch.Tensor:
    """
    Fit a new RKHS that has fewer kernels than the input RKHS. The new kernels are positioned at `output_positions`.
    1. Sample kernel at `output_positions`.
    2. The weights are computed following (K + mu I)^-1 z, where K is the kernel matrix and z are the samples.

    Args:
        input_positions: (batch_size, channels, in_kernels, n_dimensions)
        input_weights: (batch_size, channels, in_kernels, n_weights)
        output_positions: (batch_size, channels, out_kernels, n_dimensions)
        kernel: kernel function
        nonlinearity: nonlinearity to apply to the kernel weights
    """
    batch_size, channels, in_kernels, n_dimensions = input_positions.shape
    _, _, out_kernels, _ = output_positions.shape
    _, _, _, n_weights = input_weights.shape
    assert input_weights.shape == (
        batch_size,
        channels,
        in_kernels,
        n_weights,
    ), f"{input_weights.shape} != {(batch_size, channels, in_kernels)}"
    assert output_positions.shape == (
        batch_size,
        channels,
        out_kernels,
        n_dimensions,
    ), f"{output_positions.shape} != {(batch_size, channels, out_kernels, n_dimensions)}"

    samples = sample_kernel(
        Mixture(input_positions, input_weights),
        output_positions,
        kernel,
        nonlinearity,
    )
    solution = solve_kernel(samples, kernel, alpha=alpha)
    return solution.weights
