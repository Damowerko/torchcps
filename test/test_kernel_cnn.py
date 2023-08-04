import pytest
import torch

from torchcps.kernel import GaussianKernel
from torchcps.kernel_cnn import RKHS, KernelConv, KernelPool

pytestmark = pytest.mark.parametrize(
    "batch_size, in_channels, in_kernels, n_dimensions",
    [
        (8, 6, 8, 2),
        (1, 8, 4, 5),
    ],
)


def random_input(batch_size, in_channels, in_kernels, n_dimensions):
    input_positions = torch.randn(batch_size, in_channels, in_kernels, n_dimensions)
    input_weights = torch.randn(batch_size, in_channels, in_kernels)
    return RKHS(input_positions, input_weights)


def test_forward(batch_size, in_channels, in_kernels, n_dimensions):
    input = random_input(
        batch_size, in_channels, in_kernels, n_dimensions
    )
    out_channels = 2
    filter_kernels = 3 ** n_dimensions

    # create a KernelConv layer with some example parameters
    layer = KernelConv(filter_kernels, in_channels, out_channels, n_dimensions)

    # run the forward pass
    output_positions, output_weights = layer(input)

    # check that the output has the expected shape
    assert output_positions.shape == (
        batch_size,
        out_channels,
        filter_kernels * in_channels * in_kernels,
        n_dimensions,
    )
    assert output_weights.shape == (
        batch_size,
        out_channels,
        filter_kernels * in_channels * in_kernels,
    )
    # check that the output is finite
    assert torch.isfinite(output_positions).all()
    assert torch.isfinite(output_weights).all()


def test_pool(batch_size, in_kernels, in_channels, n_dimensions):
    input = random_input(
        batch_size, in_channels, in_kernels, n_dimensions
    )
    out_kernels = 2
    kernel = GaussianKernel()
    nonlinearity = torch.nn.ReLU()
    layer = KernelPool(out_kernels, kernel, nonlinearity)
    output_positions, output_weights = layer(input)
    assert output_positions.shape == (batch_size, in_channels, out_kernels, n_dimensions)
    assert output_weights.shape == (batch_size, in_channels, out_kernels)
