from abc import ABC, abstractmethod

import torch
import torch.linalg
from pykeops.torch import LazyTensor


class Kernel(ABC):
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | LazyTensor:
        return self.kernel(x, y)

    @abstractmethod
    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | LazyTensor:
        pass

    def squared_error(
        self,
        x: torch.Tensor,
        x_weights: torch.Tensor,
        y: torch.Tensor,
        y_weights: torch.Tensor,
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
        x_energy = self.energy(x, x_weights)
        y_energy = self.energy(y, y_weights)
        xy = self.inner(x, x_weights, y, y_weights)
        return x_energy + y_energy - 2 * xy

    def energy(
        self,
        x: torch.Tensor,
        x_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the energy <x, x> of a weighted sum of kernels.

        Args:
            x: (..., x_kernels, n_dimensions)
            x_weights: (..., x_kernels)
        """
        return self.inner(x, x_weights, x, x_weights)

    def inner(
        self,
        x: torch.Tensor,
        x_weights: torch.Tensor,
        y: torch.Tensor,
        y_weights: torch.Tensor,
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
        K_ij = self.kernel(x, y)
        output = x_weights.transpose(-1, -2) @ (K_ij @ y_weights)
        return output.squeeze(-1).squeeze(-1)


class GaussianKernel(Kernel):
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | LazyTensor:
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
        if len(x.shape) == 1 or (x.shape[-2] == 1 and y.shape[-2] == 1):
            return torch.exp(-((x - y) ** 2).sum(-1) / (2 * self.sigma**2))[..., None]

        x_i = LazyTensor(x[..., :, None, :])
        y_j = LazyTensor(y[..., None, :, :])
        K_ij: LazyTensor = (-((x_i - y_j) ** 2).sum(-1) / (2 * self.sigma**2)).exp()
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
