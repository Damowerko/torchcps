import torch
from pykeops.torch.cluster import from_matrix

from torchcps.kernel.rkhs import batch_to_ranges


def test_batch_to_ranges():
    x_batch = torch.tensor([0, 5, 0, 0, 10, 3])
    y_batch = torch.tensor([0, 2, 5, 3, 10, 20])

    ranges = batch_to_ranges(x_batch, y_batch)

    x_ranges = torch.stack([x_batch.cumsum(0) - x_batch, x_batch.cumsum(0)], dim=1)
    y_ranges = torch.stack([y_batch.cumsum(0) - y_batch, y_batch.cumsum(0)], dim=1)
    matrix = torch.eye(x_batch.shape[0], dtype=torch.bool)
    ranges_from_matrix = from_matrix(x_ranges, y_ranges, matrix)

    for i, (a, b) in enumerate(zip(ranges, ranges_from_matrix)):
        assert torch.equal(a, b), f"Failed at index {i}."
