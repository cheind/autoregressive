import torch
import torch.nn.functional as F

from .. import metrics, wave, generators


def test_sample_entropy():
    # Uniform random
    x = torch.rand(10, 1024)
    se = metrics.sample_entropy(x)
    assert se.mean() >= 2.0

    # Straight lines
    x = torch.arange(2 ** 12).float()
    se = metrics.sample_entropy(x).mean()
    assert abs(se) < 1e-3

    # Sine
    x = torch.sin(torch.linspace(0, 10 * 3.145, 2 ** 12))
    se = metrics.sample_entropy(x).mean()
    assert se < 0.2