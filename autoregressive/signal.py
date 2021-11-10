__all__ = [
    "signal_minmax",
    "signal_normalize",
    "signal_quantize_midtread",
    "EncoderDecoder",
    "EncoderParams",
]

from typing import Iterable, Union
import torch
import torch.nn.functional as F
import warnings
import dataclasses


def signal_minmax(
    x: Union[torch.Tensor, Iterable[torch.Tensor]]
) -> tuple[float, float]:
    """Returns minimum and maximum value of given series."""
    if isinstance(x, torch.Tensor):
        return x.min().item(), x.max().item()
    else:
        fmax = torch.finfo(torch.float32).max
        lower, upper = fmax, -fmax
        for s in x:
            lower = min(lower, x.min().item())
            upper = max(upper, x.max().item())
        return lower, upper


def signal_normalize(
    x: torch.Tensor,
    source_range: tuple[float, float] = None,
    target_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """(Batch) normalize a given signal."""
    if source_range is None:
        source_range = (x.min().detach().item(), x.max().detach().item())
    xn = (x - source_range[0]) / (source_range[1] - source_range[0])
    xt = (target_range[1] - target_range[0]) * xn + target_range[0]
    return torch.clamp(xt, target_range[0], target_range[1])


def signal_quantize_midtread(
    x: torch.Tensor, bin_size: float
) -> tuple[torch.FloatTensor, torch.LongTensor]:
    """Quantize signal using uniform mid-tread method.
    The term mid-tread is due to the fact that values |x|<bin_size/2  are mapped to zero.
    """
    k = torch.floor(x / bin_size + 0.5)
    q = bin_size * k
    return q, k.long()


@dataclasses.dataclass
class EncoderParams:
    num_levels: int
    input_range: tuple[float, float]
    bin_shift: bool = True
    one_hot: bool = False


class EncoderDecoder:
    """Performs encoding and decoding of signals using normalization/quantization"""

    def __init__(self, params: EncoderParams):
        self.enc_params = params

    def encode(self, x: torch.Tensor) -> torch.LongTensor:
        """Returns the bin indices of the encoded signal."""
        signal_range = self.enc_params.input_range
        if signal_range is None:
            signal_range = signal_minmax(x)
        x = signal_normalize(x, source_range=signal_range, target_range=(-1.0, 1.0))
        if self.enc_params.num_levels % 2 == 0:
            warnings.warn("Number of quantization levels should be odd.")
        bin_size = 2.0 / (self.enc_params.num_levels - 1)
        shift = self.enc_params.num_levels // 2 if self.enc_params.bin_shift else 0
        _, k = signal_quantize_midtread(x, bin_size)
        k = k + shift  # shift bin values, so that no negative index occurs.
        if self.enc_params.one_hot:
            k = F.one_hot(k, num_classes=self.enc_params.num_levels).permute(
                1, 0
            )  # (Q,T)
        return k

    def decode(self, k: torch.LongTensor) -> torch.FloatTensor:
        shift = self.enc_params.num_levels // 2 if self.enc_params.bin_shift else 0
        bin_size = 2.0 / (self.enc_params.num_levels - 1)
        if self.enc_params.one_hot:
            k = torch.argmax(k, dim=0)  # (T,)
        k = k - shift
        q = k * bin_size
        r = signal_normalize(
            q, source_range=(-1.0, 1.0), target_range=self.enc_params.input_range
        )
        return r
