from itertools import product

import pytest
import torch

from torchcps.datasets.airquality.beijing import parse_beijing, project_aq, project_meo
from torchcps.kernel.rkhs import GaussianKernel


@pytest.mark.parametrize(
    "sigma,implementation", product([0.1, 0.01], ["torch", "lstsq", "pykeops"])
)
@torch.no_grad()
def test_project_meo(sigma, implementation):
    T = 100
    _, meo_data, aq_pos, meo_pos = parse_beijing()
    idx = torch.randint(0, meo_data.shape[1], (T,))
    meo_data = meo_data[:, idx]
    positions = torch.cat([aq_pos, meo_pos], dim=0)
    kernel = GaussianKernel(sigma)

    S, T, F = meo_data.shape
    # standardize the meo_data
    meo_data = (meo_data - meo_data.mean((0, 1))) / meo_data.std((0, 1))
    # Convert the meo_data to RKHS
    meo_rkhs = project_meo(
        meo_data, meo_pos, positions, kernel, implementation=implementation
    )
    N = meo_rkhs.shape[0]

    K_xy = kernel(meo_pos, positions)
    samples = (K_xy @ meo_rkhs.reshape(N, T * F)).reshape(S, T, F)
    mse = (meo_data - samples).pow(2).mean()
    assert mse < 1e-2


@pytest.mark.parametrize("sigma", [0.1, 0.01])
@torch.no_grad()
def test_project_aq(sigma):
    T = 100
    aq_data, _, aq_pos, meo_pos = parse_beijing()
    idx = torch.randint(0, aq_data.shape[1], (T,))
    aq_data = aq_data[:, idx]
    positions = torch.cat([aq_pos, meo_pos], dim=0)
    kernel = GaussianKernel(sigma)

    S, T, F = aq_data.shape
    aq_data = (aq_data - aq_data[~aq_data.isnan()].mean()) / aq_data[
        ~aq_data.isnan()
    ].std()
    aq_rkhs = project_aq(aq_data, aq_pos, positions, kernel)
    N = aq_rkhs.shape[0]

    K_xy = kernel(aq_pos, positions)
    samples = (K_xy @ aq_rkhs.reshape(N, T * F)).reshape(S, T, F)
    mse = (aq_data - samples)[~aq_data.isnan()].pow(2).mean()
    assert mse < 1e-2
