__all__ = ["Noise", "chain_transforms"]

import torch
from .series_dataset import SeriesMeta


class Noise:
    def __init__(self, scale: float = 1e-2):
        self.scale = scale

    def __call__(self, sm: SeriesMeta) -> SeriesMeta:
        series, meta = sm
        series["x"] += torch.randn_like(series["x"]) * self.scale
        return series, meta


def chain_transforms(*args):
    """Composition of transformations"""
    ts = [t for t in list(args) if t is not None]

    def transform(sm: SeriesMeta) -> SeriesMeta:
        for t in ts:
            sm = t(sm)
        return sm

    return transform
