import torch

from .. import datasets


def xtest_bentlines_dataset():
    params = datasets.BentLinesParams(seed=123, num_tsamples=100, dt=0.1)
    ds = datasets.BentLinesDataset(params)
    s = ds[0]
    assert "x" in s
    assert "t" in s
    assert len(s["x"]) == 100
    assert torch.allclose(s["t"][-1], torch.tensor(9.9))

    ds = datasets.BentLinesDataset(
        params,
        transform=datasets.transforms.Encode(
            17,
            (-1, 1.0),
        ),
    )
    s = ds[0]
    assert "x_k" in s
    assert "encode.num_levels" in s
    assert "encode.input_range" in s
    assert "encode.bin_shift" in s
    assert "encode.one_hot" in s
