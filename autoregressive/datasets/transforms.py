__all__ = ["Encode", "chain_transforms"]

from .. import signal
from .series_dataset import Series


class Encode:
    """Transform to perform normalization and quantization of series data."""

    def __init__(self, params: signal.EncoderParams):
        self.encdec = signal.EncoderDecoder(params)

    def __call__(self, series: Series) -> Series:
        k = self.encdec.encode(series["x"])
        series["x_k"] = k


def chain_transforms(*args):
    """Composition of transformations"""
    ts = [t for t in list(args) if t is not None]

    def transform(Series: Series) -> Series:
        for t in ts:
            Series = t(Series)
        return Series

    return transform
