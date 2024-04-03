import typing
from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.linalg
from pykeops.torch import LazyTensor


class Mixture(NamedTuple):
    """
    Represents the positions and weights for a mixture of kernels.
    """

    positions: torch.Tensor
    weights: torch.Tensor
    batch: torch.Tensor | None = None

    def map_weights(self, f: typing.Callable[[torch.Tensor], torch.Tensor]):
        return Mixture(self.positions, f(self.weights), self.batch)


def batch_to_ranges(x_batch: torch.Tensor, y_batch: torch.Tensor):
    """
    Convert x and y batch tensors to a 6-tuple of ranges, slices and redranges as required by pykeops.
    The ranges output can be passed to PyKeOps to only compute the kernel between the elements of the batch.

    Args:
        batch: (B,) a tensor with the number of elements in each batch.
    Returns:
        ranges_i (B,2): slice indices [start_k, end_k) that specify our B blocks.
        slices_i (B,): consecutive slice indices [end_1,...,end_B] that specify B ranges in redranges_j
            that specify B ranges [start_k, end_k) in redranges_j with start_k = end_{k-1}.
        redranges_j (B,): slice indices [start_l, end_l) that specify B ranges.
        ranges_j (B,2): slice indices [start_l, end_l) that specify B ranges.
        slices_j (B,): consecurive slice indices [end_1,...,end_B] that specify B ranges in redranges_j
            that specify B ranges [start_l, end_l) in redranges_j with start_l = end_{l-1}.
        redranges_i (B,): slice indices [start_k, end_k) that specify B ranges.
    """

    # the number of batches should be the same
    assert x_batch.shape == y_batch.shape
    assert x_batch.device == y_batch.device

    batch_size = x_batch.shape[0]
    device = x_batch.device

    x_ind = x_batch.cumsum(0)
    y_ind = y_batch.cumsum(0)
    x_ranges = torch.stack([x_ind - x_batch, x_ind], dim=1)
    y_ranges = torch.stack([y_ind - y_batch, y_ind], dim=1)
    x_slices = y_slices = torch.arange(1, batch_size + 1, device=device)
    return x_ranges, x_slices, y_ranges, y_ranges, y_slices, x_ranges


class Kernel(ABC):
    """
    Represents a kernel function. Together with a Mixture it defines a Reproducing Kernel Hilbert Space (RKHS).
    """

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_batch: torch.Tensor | None = None,
        y_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | LazyTensor:
        return self.kernel(x, y, x_batch, y_batch)

    def kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_batch: torch.Tensor | None = None,
        y_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | LazyTensor:
        """
        Computes the kernel matrix between two sets of samples.

        Args:
            x (torch.Tensor): A set of positions with shape (..., N, n_dimensions).
            y (torch.Tensor): A set of positions with shape (..., M, n_dimensions).
            x_batch (torch.Tensor | None, optional): The batch information for x. Defaults to None.
            y_batch (torch.Tensor | None, optional): The batch information for y. Defaults to None.

        Returns:
            torch.Tensor | LazyTensor: The kernel matrix between x and y.

        Raises:
            ValueError: If either x_batch or y_batch is provided without the other.
        """
        K_ij = self._kernel(x, y)
        if x_batch is None and y_batch is None:
            return K_ij
        if x_batch is None or y_batch is None:
            raise ValueError(
                "Either both x_batch and y_batch should be provided or neither."
            )
        # K_ij.ranges does not have a type hint in the pykeops library
        K_ij.ranges = batch_to_ranges(x_batch, y_batch)  # type: ignore
        return K_ij

    @abstractmethod
    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> LazyTensor:
        raise NotImplementedError()

    def squared_error(
        self,
        x: torch.Tensor,
        x_weights: torch.Tensor,
        y: torch.Tensor,
        y_weights: torch.Tensor,
        x_batch: torch.Tensor | None = None,
        y_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the squared error between two weighted sums of kernels:
            <x-y, x-y> = <x,x> - 2 <x, y> + <y, y>

        Args:
            x: (..., x_kernels, n_dimensions)
            x_weights: (..., x_kernels)
            y: (..., y_kernels, n_dimensions)
            y_weights: (..., y_kernels)
        """
        x_energy = self.energy(x, x_weights, x_batch)
        y_energy = self.energy(y, y_weights, y_batch)
        xy = self.inner(x, x_weights, y, y_weights, x_batch, y_batch)
        return x_energy + y_energy - 2 * xy

    def energy(
        self,
        x: torch.Tensor,
        x_weights: torch.Tensor,
        x_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the energy <x, x> of a weighted sum of kernels.

        Args:
            x: (..., x_kernels, n_dimensions)
            x_weights: (..., x_kernels)
        """
        return self.inner(x, x_weights, x, x_weights, x_batch, x_batch)

    def inner(
        self,
        x: torch.Tensor,
        x_weights: torch.Tensor,
        y: torch.Tensor,
        y_weights: torch.Tensor,
        x_batch: torch.Tensor | None = None,
        y_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the kernel inner product:
            $$<\\sum_i a_i k(x_i), \\sum_j b_k l(y_i)> =  \\sum_i \\sum_j a_i b_j k(x_i, y_j).$$

        Args:
            x: (..., x_kernels, n_dimensions)
            x_weights: (..., x_kernels, n_weights)
            y: (..., y_kernels, n_dimensions)
            y_weights: (..., y_kernels, n_weights)
        """
        batch_shape = x.shape[:-2]
        x_kernels = x.shape[-2]
        y_kernels = y.shape[-2]
        n_dimensions = x.shape[-1]
        n_weights = x_weights.shape[-1]

        # check all tensors have appropriate shape
        assert x.shape == (
            *batch_shape,
            x_kernels,
            n_dimensions,
        ), f"{x.shape} != {(*batch_shape, x_kernels, n_dimensions)}"
        assert x_weights.shape == (
            *batch_shape,
            x_kernels,
            n_weights,
        ), f"{x_weights.shape} != {(*batch_shape, x_kernels, n_weights)}"
        assert y.shape == (
            *batch_shape,
            y_kernels,
            n_dimensions,
        ), f"{y.shape} != {(*batch_shape, y_kernels, n_dimensions)}"
        assert y_weights.shape == (
            *batch_shape,
            y_kernels,
            n_weights,
        ), f"{y_weights.shape} != {(*batch_shape, y_kernels, n_weights)}"

        # For each batch element compute the kernel inner product
        K_ij = self.kernel(x, y, x_batch, y_batch)
        output = x_weights.transpose(-1, -2) @ (K_ij @ y_weights)
        return output.squeeze(-1).squeeze(-1)


class GaussianKernel(Kernel):
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def _kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> LazyTensor:
        """
        Compute the Gaussian kernel between two tensors.

        Args:
            x: (..., n_dimensions)
            y: (..., n_dimensions)
        Returns:
            If the inputs are vectors the output is a scalar.
            If the inputs (batched) tensors (..., N, n_dimensions) and (..., M, n_dimensions) the output is a tensor of shape (..., N, M).
        """
        n_dimensions = x.shape[-1]
        assert len(x.shape) == len(y.shape)
        x_i = LazyTensor(x[..., :, None, :])
        y_j = LazyTensor(y[..., None, :, :])

        distance: LazyTensor = ((x_i - y_j) ** 2).sum(-1)
        K_ij: LazyTensor = (-distance / (2 * self.sigma**2)).exp()
        # normalize the kernel
        K_ij /= self.sigma * (2 * torch.pi) ** (n_dimensions / 2)
        return K_ij


class RQKernel(Kernel):
    def __init__(self, sigma: float = 1.0, alpha: float = 1.0):
        self.sigma = sigma
        self.alpha = alpha

    def _kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> LazyTensor:
        """
        Compute the Gaussian kernel between two tensors.

        Args:
            x: (..., n_dimensions)
            y: (..., n_dimensions)
        Returns:
            If the inputs are vectors the output is a scalar.
            If the inputs (batched) tensors (..., N, n_dimensions) and (..., M, n_dimensions) the output is a tensor of shape (..., N, M).
        """
        assert len(x.shape) == len(y.shape)
        x_i = LazyTensor(x[..., :, None, :])
        y_j = LazyTensor(y[..., None, :, :])

        D_ij: LazyTensor = ((x_i - y_j) ** 2).sum(-1)
        K_ij: LazyTensor = (1 + D_ij**2 / (2 * self.alpha * self.sigma**2)) ** (
            -self.alpha
        )
        return K_ij


def multivariate_gaussian_kernel(
    x_mean: torch.Tensor,
    x_cov_inv: torch.Tensor,
    y_mean: torch.Tensor,
    y_cov_inv: torch.Tensor,
):
    """
    Inner product between two multivariate Gaussians.
    Source: https://gregorygundersen.com/blog/2020/07/02/sum-quadratics/

    Args:
        x_mean: (..., n_dimensions)
        x_cov_inv: (..., n_dimensions, n_dimensions)
        y_mean: (..., n_dimensions)
        y_cov_inv: (..., n_dimensions, n_dimensions)

    Returns:
        If the inputs are vectors the output is a scalar.
        If the inputs (batched) tensors (..., N, n_dimensions) and (..., M, n_dimensions) the output is a tensor of shape (..., N, M).
    """
    cov_inv = x_cov_inv + y_cov_inv
    m = x_cov_inv @ x_mean + y_cov_inv @ y_mean
    R = x_mean.T @ x_cov_inv @ x_mean + y_mean.T @ y_cov_inv @ y_mean

    # integral of the term that depends on x
    dims = x_mean.shape[-1]
    integral = (2 * torch.pi) ** (dims / 2) * torch.linalg.det(cov_inv) ** (-0.5)

    # the constant term
    # use solve to avoid computing the inverse
    remainder = torch.exp(-0.5 * (R - m.T @ torch.linalg.solve(cov_inv, m)))
    return integral * remainder
