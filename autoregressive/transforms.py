from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .datasets import FSeriesDataset

Sample = Dict[str, Any]


class ApplyWithProb(ABC):
    """Base transformation applied with probability `p`."""

    def __init__(self, p=1.0) -> None:
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if torch.rand(1) < self.p:
            return self._apply(sample)
        else:
            return sample

    @abstractmethod
    def _apply(self, sample: Sample) -> Sample:
        ...


class Noise(ApplyWithProb):
    """Adds iid Gaussian zero-mean noise to observations."""

    def __init__(self, scale: float = 1e-3, p: float = 1.0) -> None:
        super().__init__(p)
        self.scale = scale

    def _apply(self, sample: Sample) -> Sample:
        sample["x"] += torch.randn_like(sample["x"]) * self.scale
        return sample


class Quantize:
    """Quantizes observations to nearest multiple of bin-size"""

    def __init__(self, bin_size: float = 0.05, num_bins: int = None) -> None:
        if num_bins is not None:
            bin_size = 1 / (num_bins - 1)
        self.bin_size = bin_size

    def __call__(self, sample: Sample) -> Sample:
        x = sample["x"]
        b = torch.round(x / self.bin_size)
        sample["x"] = b * self.bin_size
        sample["b"] = b.long()
        return sample


class Normalize:
    """Normalize to [0,1] range on a per-sample basis"""

    def __init__(self, lu_range: Tuple[float, float] = None) -> None:
        if lu_range is not None:
            self.cmin = lu_range[0]
            self.cmax = lu_range[1]
        self.cmin, self.cmax = None, None

    def __call__(self, sample: Sample) -> Sample:
        sample["x"] = self.apply(sample["x"])
        return sample

    def apply(self, x: torch.Tensor):
        if self.cmin is None:
            cmin, cmax = x.min(), x.max()
        else:
            cmin, cmax = self.cmin, self.cmax

        x = (x - cmin) / (cmax - cmin)
        x = torch.clamp(x, cmin, cmax)
        return x

    @staticmethod
    def find_range(*ds: "FSeriesDataset") -> Tuple[float, float]:
        dl = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(ds), batch_size=256, num_workers=4
        )
        fmax = torch.finfo(torch.float32).max
        lower, upper = fmax, -fmax
        for b in dl:
            lower = min(lower, b["x"].min())
            upper = max(upper, b["x"].max())
        return lower, upper


def chain_transforms(*args: Sequence[Sample]):
    """Composition of transformations"""
    ts = [t for t in list(args) if t is not None]

    def transform(sample: Sample) -> Sample:
        for t in ts:
            sample = t(sample)
        return sample

    return transform
