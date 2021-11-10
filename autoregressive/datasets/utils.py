__all__ = ["fractional_dataset_split"]
import math
from typing import TYPE_CHECKING

import torch
import torch.utils.data
from .. import signal

if TYPE_CHECKING:
    from .series_dataset import SeriesDataset


def fractional_dataset_split(
    ds: SeriesDataset, splits: list[float]
) -> list[torch.utils.data.Subset]:
    idx = torch.arange(len(ds)).tolist()
    N = len(idx)
    starts = torch.tensor([math.ceil(s * N) for s in splits[:-1]])
    starts = [0] + torch.cumsum(starts, 0).tolist()
    ends = starts[1:] + [N]
    subsets = [torch.utils.data.Subset(ds, idx[s:e]) for s, e in zip(starts, ends)]
    return subsets


def datasets_minmax(*datasets: "SeriesDataset"):
    """Returns signal min/max range for all series in given datasets"""

    def genx(ds):
        for s in ds:
            yield s["x"]

    return signal.signal_minmax(genx(torch.utils.data.ConcatDataset(datasets)))
