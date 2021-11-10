__all__ = ["Encode", "EncodeParams", "chain_transforms"]
import dataclasses
from .. import signal
from .series_dataset import Series


@dataclasses.dataclass
class EncodeParams:
    num_levels: int
    input_range: tuple[float, float]
    bin_shift: bool = True
    one_hot: bool = False


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
        return series


def chain_transforms(*args):
    """Composition of transformations"""
    ts = [t for t in list(args) if t is not None]

    def transform(Series: Series) -> Series:
        for t in ts:
            Series = t(Series)
        return Series

    return transform
