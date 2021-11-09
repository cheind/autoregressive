__all__ = [
    "signal_minmax",
    "signal_normalize",
    "signal_quantize_midtread",
    "signal_preprocess",
]

from typing import Iterable, Union
import torch


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


def signal_quantize_midtread(x: torch.Tensor, bin_size: float):
    """Quantize signal using uniform mid-tread method.
    The term mid-tread is due to the fact that values |x|<bin_size/2  are mapped to zero.
    """
    k = torch.floor(x / bin_size + 0.5)
    q = bin_size * k
    return q, k.long()


def signal_preprocess(
    x: torch.Tensor, num_bins: int, signal_range: tuple[float, float] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combines signal normalization and quantization."""
    if signal_range is None:
        signal_range = signal_minmax(x)
    x = signal_normalize(x, source_range=signal_range, target_range=(-1.0, 1.0))
    if num_bins % 2 == 0:
        raise ValueError("Number of quantization levels should be odd.")
    bin_size = 2.0 / (num_bins - 1)
    q, k = signal_quantize_midtread(x, bin_size)
    k = k + num_bins // 2  # shift bin values, so that no negative index occurs.
    return q, k
