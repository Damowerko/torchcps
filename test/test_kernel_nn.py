import pytest
import torch

from torchcps.kernel.nn import KNN, KernelConv, KernelPool, Mixture
from torchcps.kernel.rkhs import GaussianKernel, Mixture

pytestmark = pytest.mark.parametrize(
    "in_channels, in_kernels, n_dimensions",
    [
        (6, 8, 2),
        (8, 4, 5),
    ],
)


def random_input(in_kernels, in_channels, n_dimensions):
    input_positions = torch.randn(in_kernels, n_dimensions)
    input_weights = torch.randn(in_kernels, in_channels)
    return Mixture(input_positions, input_weights)


def test_conv_forward(in_kernels, in_channels, n_dimensions):
    input = random_input(in_kernels, in_channels, n_dimensions)
    out_channels = 2
    filter_kernels = 3**n_dimensions

    # create a KernelConv layer with some example parameters
    layer = KernelConv(filter_kernels, in_channels, out_channels, n_dimensions)

    # run the forward pass
    output_positions, output_weights, _ = layer.forward(input)

    # check that the output has the expected shape
    assert output_positions.shape == (filter_kernels * in_kernels, n_dimensions)
    assert output_weights.shape == (filter_kernels * in_kernels, out_channels)
    # check that the output is finite
    assert torch.isfinite(output_positions).all()
    assert torch.isfinite(output_weights).all()


def test_pool_largest(in_kernels, in_channels, n_dimensions):
    input = random_input(in_kernels, in_channels, n_dimensions)
    max_kernels = 2
    layer = KernelPool(max_kernels, GaussianKernel(), strategy="largest")
    output_positions, output_weights, _ = layer.forward(input)
    assert output_positions.shape == (min(max_kernels, in_kernels), n_dimensions)
    assert output_weights.shape == (min(max_kernels, in_kernels), in_channels)


def test_pool_random(in_kernels, in_channels, n_dimensions):
    input = random_input(in_kernels, in_channels, n_dimensions)
    max_kernels = 2
    layer = KernelPool(max_kernels, GaussianKernel(), strategy="random")
    output_positions, output_weights, _ = layer.forward(input)
    assert output_positions.shape == (min(max_kernels, in_kernels), n_dimensions)
    assert output_weights.shape == (min(max_kernels, in_kernels), in_channels)


def test_knn_forward(in_channels, in_kernels, n_dimensions):
    input = random_input(in_kernels, in_channels, n_dimensions)
    out_channels = 5
    filter_kernels = 8

    model = KNN(
        n_dimensions=n_dimensions,
        in_channels=in_channels,
        hidden_channels=8,
        out_channels=out_channels,
        n_layers=2,
        sigma=0.5,
        hidden_channels_mlp=2,
        n_layers_mlp=3,
        max_filter_kernels=filter_kernels,
        update_positions=True,
    )

    # Call the forward method
    output = model.forward(input)

    # Add your assertions here to validate the output
    assert output.positions.shape == (in_kernels, n_dimensions)
    assert output.weights.shape == (in_kernels, out_channels)
