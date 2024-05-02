from itertools import product

import pytest
import torch

from torchcps.datasets.airquality.beijing import BeijingDataset, BeijingDatasetRKHS
from torchcps.kernel.nn import sample_kernel, solve_kernel
from torchcps.kernel.rkhs import Mixture


@pytest.mark.parametrize(
    "sigma,implementation", product([0.1, 0.01, 0.001], ["torch", "lstsq"])
)
def test_meo_to_rkhs(sigma, implementation):
    # Create a BeijingDatasetRKHS object
    beijing_rkhs = BeijingDatasetRKHS(BeijingDataset(), sigma=sigma)

    # Get the first element of the BeijingDataset object
    _, meo_data, _, meo_pos = beijing_rkhs.beijing[0]
    S, T, F = meo_data.shape

    # Convert the meo_data to RKHS
    meo_rkhs = beijing_rkhs.meo_to_rkhs(meo_data, implementation=implementation)
    N = meo_rkhs.shape[0]

    K_xy = beijing_rkhs.kernel(meo_pos, beijing_rkhs.positions)
    samples = (K_xy @ meo_rkhs.reshape(N, T * F)).reshape(S, T, F)
    mse = (meo_data - samples).pow(2).mean()
    assert mse < 1e-4


@pytest.mark.parametrize("sigma", [0.1, 0.01, 0.001])
def test_aq_to_rkhs(sigma):
    beijing_rkhs = BeijingDatasetRKHS(BeijingDataset(), sigma=sigma)
    aq_data, _, aq_pos, _ = beijing_rkhs.beijing[0]
    S, T, F = aq_data.shape
    aq_rkhs = beijing_rkhs.aq_to_rkhs(aq_data)
    N = aq_rkhs.shape[0]

    K_xy = beijing_rkhs.kernel(aq_pos, beijing_rkhs.positions)
    samples = (K_xy @ aq_rkhs.reshape(N, T * F)).reshape(S, T, F)
    mse = (aq_data - samples)[~aq_data.isnan()].pow(2).mean()
    assert mse < 1e-4
