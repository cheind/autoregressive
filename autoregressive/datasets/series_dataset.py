__all__ = ["SeriesDataset"]
from typing import Any, Dict

import torch.utils.data


Series = Dict[str, Any]


class SeriesDataset(torch.utils.data.Dataset):
    def __getitem__(self, index) -> Series:
        raise NotImplementedError
