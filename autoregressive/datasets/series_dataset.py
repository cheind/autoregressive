__all__ = [
    "SeriesDataset",
    "series_collate",
    "dataset_minmax",
    "fractional_dataset_split",
]
import math
from typing import Any, Dict

import torch
import torch.utils.data
from torch.utils.data._utils.collate import default_collate

from .. import signal

Meta = Dict[str, Any]
Series = Dict[str, torch.Tensor]
SeriesMeta = tuple[Series, Meta]


class SeriesDataset(torch.utils.data.Dataset):
    def __getitem__(self, index) -> SeriesMeta:
        raise NotImplementedError


def series_collate(batch):
    """Default collate fn for series datasets"""
    series, meta = zip(*batch)
    return default_collate(series), meta


def dataset_minmax(*datasets: SeriesDataset):
    """Returns signal min/max range for all series in given datasets"""

    def genx(ds):
        for s, _ in ds:
            yield s["x"]

    return signal.signal_minmax(genx(torch.utils.data.ConcatDataset(datasets)))


def fractional_dataset_split(
    ds: "SeriesDataset", splits: list[float]
) -> list[torch.utils.data.Subset]:
    idx = torch.arange(len(ds)).tolist()
    N = len(idx)
    starts = torch.tensor([math.ceil(s * N) for s in splits[:-1]])
    starts = [0] + torch.cumsum(starts, 0).tolist()
    ends = starts[1:] + [N]
    subsets = [torch.utils.data.Subset(ds, idx[s:e]) for s, e in zip(starts, ends)]
    return subsets
