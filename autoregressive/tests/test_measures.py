import torch
from torch.utils.data import DataLoader

from .. import measures, datasets


def test_sample_entropy():
    # Uniform random
    x = torch.rand(10, 1024)
    se = measures.sample_entropy(x)
    assert se.mean() >= 2.0

    # Straight lines
    x = torch.arange(2 ** 12).float()
    se = measures.sample_entropy(x).mean()
    assert abs(se) < 1e-3

    # Sine
    x = torch.sin(torch.linspace(0, 10 * 3.145, 2 ** 12))
    se = measures.sample_entropy(x).mean()
    assert se < 0.2

    ds = datasets.FSeriesDataset(period_range=10, seed=123)
    b = next(iter(DataLoader(ds, batch_size=512)))
    se_fixed_p = measures.sample_entropy(b["x"]).mean()

    ds = datasets.FSeriesDataset(period_range=(5, 10), seed=123)
    b = next(iter(DataLoader(ds, batch_size=512)))
    se_var_p = measures.sample_entropy(b["x"]).mean()

    ds = datasets.FSeriesDataset(
        period_range=(5, 10),
        seed=123,
        transform=datasets.transforms.Noise(scale=1e-1, p=1),
    )
    b = next(iter(DataLoader(ds, batch_size=512)))
    se_var_p_noise = measures.sample_entropy(b["x"]).mean()

    assert se_var_p_noise > se_var_p > se_fixed_p
