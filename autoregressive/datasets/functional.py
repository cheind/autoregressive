__all__ = [
    "find_series_range",
    "normalize_series",
    "normalize_series_inv",
]
from typing import Tuple, Iterable
import torch


def find_series_range(series: Iterable[torch.Tensor]) -> Tuple[float, float]:
    """Returns minimum and maximum value of given series."""
    fmax = torch.finfo(torch.float32).max
    lower, upper = fmax, -fmax
    for s in series:
        lower = min(lower, s.min().item())
        upper = max(upper, s.max().item())
    return lower, upper


def normalize_series(
    x: torch.Tensor,
    source_range: Tuple[float, float] = None,
    target_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """(Batch) normalize a given series."""
    if source_range is None:
        source_range = (x.min().detach().item(), x.max().detach().item())
    xn = (x - source_range[0]) / (source_range[1] - source_range[0])
    xt = (target_range[1] - target_range[0]) * xn + target_range[0]
    return torch.clamp(xt, target_range[0], target_range[1])


def normalize_series_inv(
    x: torch.Tensor,
    source_range: Tuple[float, float] = None,
    target_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """(Batch) un-normalize a given series."""
    return normalize_series(x, target_range, source_range)
