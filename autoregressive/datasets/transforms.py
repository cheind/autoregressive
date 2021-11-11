__all__ = ["Encode", "chain_transforms"]

from .. import signal
from .series_dataset import Series


class Encode:
    """Transform to perform normalization and quantization of series data."""

    def __init__(
        self,
        num_levels: int,
        input_range: tuple[float, float],
        bin_shift: bool = True,
        one_hot: bool = False,
    ):
        self.encdec = signal.EncoderDecoder(
            num_levels=num_levels,
            input_range=input_range,
            bin_shift=bin_shift,
            one_hot=one_hot,
        )

    def __call__(self, series: Series) -> Series:
        series["x_k"] = self.encdec.encode(series["x"])
        series["encode.num_levels"] = self.encdec.num_levels
        series["encode.input_range"] = self.encdec.input_range
        series["encode.bin_shift"] = self.encdec.bin_shift
        series["encode.one_hot"] = self.encdec.one_hot
        return series


def chain_transforms(*args):
    """Composition of transformations"""
    ts = [t for t in list(args) if t is not None]

    def transform(Series: Series) -> Series:
        for t in ts:
            Series = t(Series)
        return Series

    return transform
