__all__ = ["SeriesDataset", "split_ds_fractional"]
from typing import Any, Dict, List

import math
import torch
import torch.utils.data


Sample = Dict[str, Any]


class SeriesDataset(torch.utils.data.Dataset):
    def __getitem__(self, index) -> Sample:
        raise NotImplementedError


def split_ds_fractional(
    ds: SeriesDataset, splits: List[float]
) -> List[torch.utils.data.Subset]:
    idx = torch.arange(len(ds)).tolist()
    N = len(idx)
    starts = torch.tensor([math.ceil(s * N) for s in splits[:-1]])
    starts = [0] + torch.cumsum(starts, 0).tolist()
    ends = starts[1:] + [N]
    subsets = [torch.utils.data.Subset(ds, idx[s:e]) for s, e in zip(starts, ends)]
    return subsets
