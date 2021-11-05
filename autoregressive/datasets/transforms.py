__all__ = ["ApplyWithProb", "Noise", "Quantize", "Normalize", "chain_transforms"]
from abc import ABC, abstractmethod
from typing import Tuple, Sequence

import torch

from .common import Sample
from .functional import normalize_series


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
    """Normalize series values to given target range.

    Params
    ------
    source_range: (float,float)
        When provided, will be used for every series. Otherwise
        source range will be computed on a per sample basis.
    target_range: (float,float)
        Range of output
    """

    def __init__(
        self,
        source_range: Tuple[float, float] = None,
        target_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.source_range = source_range
        self.target_range = target_range

    def __call__(self, sample: Sample) -> Sample:
        self.sample["x"] = normalize_series(
            self.sample["x"], self.source_range, self.target_range
        )
        return sample


def chain_transforms(*args: Sequence[Sample]):
    """Composition of transformations"""
    ts = [t for t in list(args) if t is not None]

    def transform(sample: Sample) -> Sample:
        for t in ts:
            sample = t(sample)
        return sample

    return transform
