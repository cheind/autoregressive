import torch

from .. import datasets


def test_bentlines_dataset():
    params = datasets.BentLinesParams(seed=123, num_tsamples=100, dt=0.1)
    ds = datasets.BentLinesDataset(params)
    s = ds[0]
    assert "x" in s
    assert "t" in s
    assert len(s["x"]) == 100
    assert torch.allclose(s["t"][-1], torch.tensor(9.9))
