import torch

from .. import datasets
from ..datasets import functional


def test_datasets_normalize():
    x = torch.rand(2, 100)
    xminl, xmaxl = torch.argmin(x), torch.argmax(x)
    r = functional.find_series_range(x)
    assert r[0] == x.min()
    assert r[1] == x.max()

    xn = functional.normalize_series(x, r, (0.0, 1.0))
    assert xn.min() == 0.0
    assert xn.max() == 1.0
    assert xn.view(-1)[xminl] == 0.0
    assert xn.view(-1)[xmaxl] == 1.0

    xn = functional.normalize_series(x, r, (-1.0, 1.0))
    assert xn.min() == -1.0
    assert xn.max() == 1.0
    assert xn.view(-1)[xminl] == -1.0
    assert xn.view(-1)[xmaxl] == 1.0

    xo = functional.normalize_series_inv(xn, r, (-1.0, 1.0))
    assert torch.allclose(xo, x, atol=1e-4)

    xo = functional.normalize_series(x, r, r)
    assert torch.allclose(xo, x, atol=1e-4)


def test_datasets_standardize():
    x = torch.randn(2, 100) * 1e-1 + 0.5
    mean, std = functional.find_series_mean_std(x)

    assert torch.allclose(mean, torch.tensor(0.5), atol=1e-1)
    assert torch.allclose(std, torch.tensor(0.1), atol=1e-1)

    xn = functional.standardize_series(x, mean, std)
    nmean, nstd = functional.find_series_mean_std(xn)
    assert torch.allclose(nmean, torch.tensor(0.0), atol=1e-1)
    assert torch.allclose(nstd, torch.tensor(1.0), atol=1e-1)

    xo = functional.standardize_series_inv(xn, mean, std)
    assert torch.allclose(xo, x, atol=1e-4)
