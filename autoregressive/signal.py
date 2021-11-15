__all__ = [
    "signal_minmax",
    "signal_normalize",
    "signal_quantize_midtread",
    "SignalProcessor",
    "SignalProcessorState",
]

import dataclasses
import warnings
from typing import Any, Iterable, Optional, Union, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from datasets.series_dataset import SeriesMeta


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
            lower = min(lower, s.min().item())
            upper = max(upper, s.max().item())
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
class SignalProcessorState:
    quantization_levels: int
    signal_low: Optional[float]
    signal_high: Optional[float]


class SignalProcessor:
    def __init__(
        self,
        quantization_levels: int = 255,
        signal_low: Optional[float] = -1.0,
        signal_high: Optional[float] = 1.0,
    ) -> None:
        self.load_state(
            SignalProcessorState(
                quantization_levels=quantization_levels,
                signal_low=signal_low,
                signal_high=signal_high,
            )
        )

    def __call__(self, sm: "SeriesMeta") -> "SeriesMeta":
        series, meta = sm
        series["x"] = self.encode(series["x"])
        meta.update(self.get_state())
        return series, meta

    def encode(self, x: torch.Tensor) -> torch.LongTensor:
        s = self.state
        """Returns the bin indices of the encoded signal."""
        if s.signal_low is not None and s.signal_high is not None:
            x = signal_normalize(
                x,
                source_range=(s.signal_low, s.signal_high),
                target_range=(-1.0, 1.0),
            )
        if s.quantization_levels % 2 == 0:
            warnings.warn("Number of quantization levels should be odd.")
        bin_size = 2.0 / (s.quantization_levels - 1)
        _, k = signal_quantize_midtread(x, bin_size)
        shift = s.quantization_levels // 2
        k = k + shift  # shift bin values, so that no negative index occurs.
        return k

    def decode(self, k: torch.LongTensor) -> torch.FloatTensor:
        s = self.state
        bin_size = 2.0 / (s.quantization_levels - 1)
        shift = s.quantization_levels // 2
        k = k - shift
        q = k * bin_size
        if s.signal_low is not None and s.signal_high is not None:
            q = signal_normalize(
                q, source_range=(-1.0, 1.0), target_range=(s.signal_low, s.signal_high)
            )
        return q

    def get_state(self) -> dict[str, Any]:
        d = dataclasses.asdict(self.state)
        return {f"encoding.{k}": v for k, v in d.items()}

    def load_state(
        self, state: Union[SignalProcessorState, dict[str, Any]]
    ) -> "SignalProcessor":
        if not isinstance(state, SignalProcessorState):
            filtered = {
                k.split(".")[1]: v
                for k, v in state.items()
                if k.startswith("encoding.")
            }
            state = SignalProcessorState(**filtered)
        self.state = state
        return self
