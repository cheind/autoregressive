__all__ = [
    "FSeriesDataset",
    "FSeriesParams",
    "FSeriesDataModule",
]

from collections.abc import Sequence
from typing import Any, Callable, Dict, Tuple, Union
import dataclasses

import numpy as np
import torch
import torch.utils.data
import pytorch_lightning as pl

from .series_dataset import SeriesDataset, Series
from . import transforms, utils

FloatOrFloatRange = Union[float, Tuple[float, float]]
IntOrIntRange = Union[int, Tuple[int, int]]

PI = float(np.pi)


def fseries_amp_phase(
    bias: torch.FloatTensor,
    n: torch.IntTensor,
    a: torch.FloatTensor,
    phase: torch.FloatTensor,
    period: torch.FloatTensor,
    t: torch.FloatTensor,
):
    """Computes the Fourier series from amplitude-phase parametrization.

    This function supports batching, so that multiple series can
    be evaluated in parallel.

    Params
    ------
    bias: (B,) or (1) tensor
        Bias term(s) aka DC values
    n: (B,N) or (N,) or tensor
        component values of the series
    a: (B,N) or (N,) tensor
        coefficient of components
    phase: (B,N) or (B,) or (1) tensor
        phase for component term
    period: (B,) or (1) tensor
        period for each of the curves, such that f(t) == f(t+T)
    t: (B,T) or (T,) tensor
        sample times for each of the curves

    Returns
    -------
    y: (B,T) tensor or (1,T)
        function values for each of the fourier series results
    """
    # https://www.seas.upenn.edu/~kassam/tcom370/n99_2B.pdf
    t = torch.atleast_2d(t).unsqueeze(1)  # (B,1,T)
    period = torch.atleast_1d(period).view(-1, 1, 1)  # (B,1,1)
    n = torch.atleast_2d(n).unsqueeze(-1)  # (B,N,1)
    phase = torch.atleast_2d(phase).unsqueeze(-1)  # (B,N,1)
    bias = torch.atleast_1d(bias).unsqueeze(-1)  # (B,1)
    a = torch.atleast_2d(a).unsqueeze(1)  # (B,1,N)

    # print(t.shape, n.shape, phase.shape, period.shape)

    f0 = 1 / period
    arg = 2 * PI * f0 * n * t + phase  # (B,N,T)
    return bias * 0.5 + (a @ torch.cos(arg)).squeeze(1)


@dataclasses.dataclass
class FSeriesParams:
    num_curves: int = 8096
    num_tsamples: int = 2048
    dt: float = 0.02
    fterm_range: IntOrIntRange = (3, 5)
    tstart_range: FloatOrFloatRange = 0.0
    period_range: FloatOrFloatRange = 10.0
    bias_range: FloatOrFloatRange = 0.0
    coeff_range: FloatOrFloatRange = (-1.0, 1.0)
    phase_range: FloatOrFloatRange = (-PI, PI)
    lineartrend_range: FloatOrFloatRange = 0.0
    smoothness: float = 0.0
    seed: int = None
    include_params: bool = False


class FSeriesDataset(SeriesDataset):
    """A randomized Fourier series dataset."""

    def __init__(
        self,
        params: FSeriesParams,
        transform: Callable[[Series], Series] = None,
    ) -> None:
        super().__init__()
        eps = torch.finfo(torch.float32).eps

        def _make_range(arg, delta):
            if not isinstance(arg, Sequence):
                arg = (arg, arg + delta)
            return arg

        self.num_curves = params.num_curves
        self.fterm_range = _make_range(params.fterm_range, 1)
        self.period_range = _make_range(params.period_range, eps)
        self.bias_range = _make_range(params.bias_range, eps)
        self.coeff_range = _make_range(params.coeff_range, eps)
        self.phase_range = _make_range(params.phase_range, eps)
        self.tstart_range = _make_range(params.tstart_range, eps)
        self.lineartrend_range = _make_range(params.lineartrend_range, eps)
        self.num_tsamples = params.num_tsamples
        self.dt = params.dt
        self.smoothness = params.smoothness
        self.transform = transform
        self.include_params = params.include_params
        if params.seed is None:
            rng = torch.default_generator
        else:
            rng = torch.Generator().manual_seed(params.seed)
        self.curve_params = [self._sample_params(rng) for _ in range(self.num_curves)]

    def __len__(self):
        return self.num_curves

    def __getitem__(self, index) -> Series:
        """Returns a sample curve."""
        p = self.curve_params[index]
        t = torch.arange(p["tstart"], self.dt * self.num_tsamples, self.dt)
        n = torch.arange(p["terms"]) + 1
        x = fseries_amp_phase(p["bias"], n, p["coeffs"], p["phase"], p["period"], t)[0]
        x += t * p["lineark"]
        sample = {"x": x, "t": t}
        if self.include_params:
            sample["p"] = p
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _sample_params(self, g: torch.Generator) -> Dict[str, Any]:
        """Returns sampled fseries parameters."""

        def uniform(r, n: int):
            return (r[1] - r[0]) * torch.rand(n, generator=g) + r[0]

        terms = torch.randint(
            self.fterm_range[0], self.fterm_range[1] + 1, (1,), generator=g
        ).item()
        period = uniform(self.period_range, 1)
        bias = uniform(self.bias_range, 1)
        coeffs = uniform(self.coeff_range, terms)
        coeffs = coeffs * torch.logspace(
            0, -self.smoothness, terms
        )  # decay coefficients for higher order terms. A value of 2 will decay the last term by a factor of 0.01
        phase = uniform(self.phase_range, terms)
        tstart = uniform(self.tstart_range, 1).item()
        lineark = uniform(self.lineartrend_range, 1).item()

        return {
            "terms": terms,
            "period": period,
            "bias": bias,
            "coeffs": coeffs,
            "phase": phase,
            "tstart": tstart,
            "lineark": lineark,
        }


class FSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        quantization_levels: int = 127,
        batch_size: int = 64,
        num_workers: int = 0,
        train_params: FSeriesParams = FSeriesParams(smoothness=0.75),
        val_params: FSeriesParams = None,
    ):
        super().__init__()

        if val_params is None:
            val_params = dataclasses.replace(train_params)
            val_params.num_curves = min(val_params.num_curves, 512)

        train_ds = FSeriesDataset(train_params)
        val_ds = FSeriesDataset(val_params)
        signal_range = utils.datasets_minmax(train_ds, val_ds)
        transform = transforms.Encode(
            num_levels=quantization_levels,
            input_range=signal_range,
            bin_shift=True,
            one_hot=False,
        )
        self.train_ds = FSeriesDataset(train_params, transform=transform)
        self.val_ds = FSeriesDataset(val_params, transform=transform)
        self.quantization_levels = quantization_levels
        self.train_params = train_params
        self.val_params = val_params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dt = self.train_ds.dt

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def __str__(self) -> str:
        return f"train_params: {self.train_params}\n val_params:{self.val_params}"


def square_wave():
    # https://mathworld.wolfram.com/FourierSeriesSquareWave.html
    import matplotlib.pyplot as plt

    T = 10.0
    num_terms = 20
    num_samples = 1000
    n = torch.arange(1, 2 * num_terms, step=2)
    coeffs = 4.0 / (PI * n)
    # We generate multiple (B=4) approximations with increasing frequencies enabled.
    coeffs = coeffs.view(1, -1).repeat(4, 1)
    coeffs[0, 5:] = 0.0  # For the first curve, we disable all but the first 5 terms
    coeffs[1, 10:] = 0.0
    coeffs[2, 15:] = 0.0
    phase = torch.tensor(-PI / 2)  # We share phase angles (sin(phi) = cos(phi-pi/2))
    bias = torch.tensor(0.0)  # We don't have a bias aka DC
    t = torch.linspace(
        0, T, num_samples
    )  # We sample all curves at the same time intervals

    y = fseries_amp_phase(
        bias=bias,
        n=n,
        a=coeffs,
        phase=phase,
        period=torch.tensor(T),
        t=t,
    )
    yhat = torch.zeros(len(t))
    yhat[: num_samples // 2] = 1.0
    yhat[num_samples // 2 :] = -1.0
    yhat[0] = 0.0
    yhat[-1] = 0.0

    plt.title("Approximations of a step function")
    plt.plot(t, yhat, c="k")
    plt.plot(t, y[0], label="5 terms")
    plt.plot(t, y[1], label="10 terms")
    plt.plot(t, y[2], label="15 terms")
    plt.plot(t, y[3], label="20 terms")
    plt.legend()
    plt.show()


def random_waves():
    import matplotlib.pyplot as plt

    N = 5
    n = torch.arange(N)
    coeffs = torch.rand(4, N)
    phase = torch.rand(4, N)
    bias = torch.rand(4)
    period = torch.rand(4) * 5
    t = torch.stack(
        [
            torch.linspace(0, 1, 1000),
            torch.linspace(0.5, 1.5, 1000),
            torch.linspace(1.5, 2.5, 1000),
            torch.linspace(2.5, 3.5, 1000),
        ],
        0,
    )
    y = fseries_amp_phase(
        bias=bias,
        n=n,
        a=coeffs,
        phase=phase,
        period=period,
        t=t,
    )
    plt.title("Random waves sampled at random times")
    plt.plot(t[0], y[0])
    plt.plot(t[1], y[1])
    plt.plot(t[2], y[2])
    plt.plot(t[3], y[3])
    plt.show()


def main():
    square_wave()
    random_waves()


if __name__ == "__main__":
    main()
