__all__ = [
    "FSeriesDataset",
    "FSeriesParams",
    "FSeriesDataModule",
]

from collections.abc import Sequence
from typing import Any, Callable, Dict, Tuple, Union, Optional
from functools import partial
import dataclasses
import logging

import torch
import torch.utils.data
import torch.nn.functional as F
import pytorch_lightning as pl

from autoregressive import signal

from . import series_dataset as sd
from .fourier import fseries_amp_phase, PI
from .transforms import chain_transforms

FloatOrFloatRange = Union[float, Tuple[float, float]]
IntOrIntRange = Union[int, Tuple[int, int]]

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


@dataclasses.dataclass
class FSeriesParams:
    num_curves: int = 8096
    num_tsamples: int = 2048
    dt: float = 0.02
    fterm_range: IntOrIntRange = (3, 5)
    tstart_range: FloatOrFloatRange = 0.0
    period_range: IntOrIntRange = 10
    bias_range: FloatOrFloatRange = 0.0
    coeff_range: FloatOrFloatRange = (-1.0, 1.0)
    phase_range: FloatOrFloatRange = (-PI, PI)
    lineartrend_range: FloatOrFloatRange = 0.0
    smoothness: float = 0.0
    noise_scale: float = 0.0
    seed: int = None


class FSeriesDataset(sd.SeriesDataset):
    """A randomized Fourier series dataset."""

    def __init__(
        self,
        params: FSeriesParams,
        transform: Callable[[sd.SeriesMeta], tuple[sd.SeriesMeta]] = None,
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
        self.noise_scale = params.noise_scale
        if params.seed is None:
            rng = torch.default_generator
        else:
            rng = torch.Generator().manual_seed(params.seed)
        self.curve_params = [self._sample_params(rng) for _ in range(self.num_curves)]

    def __len__(self):
        return self.num_curves

    def __getitem__(self, index) -> sd.SeriesMeta:
        """Returns the i-th fourier series sample and meta."""
        meta = self.curve_params[index]
        t = torch.arange(meta["tstart"], self.dt * self.num_tsamples, self.dt)
        n = torch.arange(meta["terms"]) + 1
        x = fseries_amp_phase(
            meta["bias"], n, meta["coeffs"], meta["phase"], meta["period"], t
        )[0]
        x += t * meta["lineark"]
        x += torch.randn_like(x) * self.noise_scale
        series = {"x": x}
        if self.transform is not None:
            series, meta = self.transform((series, meta))
        return series, meta

    def _sample_params(self, g: torch.Generator) -> Dict[str, Any]:
        """Returns sampled fseries parameters."""

        def uniform(r, n: int):
            return (r[1] - r[0]) * torch.rand(n, generator=g) + r[0]

        terms = torch.randint(
            self.fterm_range[0], self.fterm_range[1] + 1, (1,), generator=g
        ).item()
        period = uniform(self.period_range, 1).int()
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
            "dt": self.dt,
        }


def add_period_conditioning(
    sm: sd.SeriesMeta, period_range: tuple[float, float]
) -> sd.SeriesMeta:
    series, meta = sm
    p = torch.round(meta["period"].float()).long()
    lower = int(period_range[0])
    num_periods = int(period_range[1]) - lower  # upper is exclusive
    # print(p, period_range, num_periods, lower, int(period_range[1]))
    p = F.one_hot(p - lower, num_classes=num_periods).permute(1, 0)
    series["c"] = p.float()
    return series, meta


class FSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        quantization_levels: int = 127,
        batch_size: int = 64,
        num_workers: int = 0,
        train_params: FSeriesParams = FSeriesParams(smoothness=0.75),
        val_params: Optional[FSeriesParams] = None,
        test_params: Optional[FSeriesParams] = None,
        period_conditioning: bool = False,
    ):
        super().__init__()

        if val_params is None:
            val_params = dataclasses.replace(train_params)
            val_params.num_curves = min(val_params.num_curves, 512)

        if test_params is None:
            test_params = dataclasses.replace(val_params)
            test_params.seed = 123

        train_ds = FSeriesDataset(train_params)
        val_ds = FSeriesDataset(val_params)
        signal_range = sd.dataset_minmax(train_ds, val_ds)
        transform = signal.SignalProcessor(
            quantization_levels=quantization_levels,
            signal_low=signal_range[0],
            signal_high=signal_range[1],
        )
        if period_conditioning:
            cr = int(train_params.period_range[1] - train_params.period_range[0])
            _logger.info(f"Added period conditioning: {cr} condition channels required")
            transform = chain_transforms(
                transform,
                partial(
                    add_period_conditioning, period_range=train_params.period_range
                ),
            )
        self.train_ds = FSeriesDataset(train_params, transform=transform)
        self.val_ds = FSeriesDataset(val_params, transform=transform)
        self.test_ds = FSeriesDataset(test_params, transform=transform)
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
            collate_fn=sd.series_collate,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=sd.series_collate,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=sd.series_collate,
        )

    def __str__(self) -> str:
        return f"train_params: {self.train_params}\n val_params:{self.val_params}"


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    # dm = FSeriesDataModule(
    #     train_fseries_params=FSeriesParams(smoothness=0.75), batch_size=512
    # )
    train_params = FSeriesParams(smoothness=0.75, period_range=(5, 40))
    dm = FSeriesDataModule(quantization_levels=127, train_params=train_params)

    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(4, 3), axes_pad=0.05, share_all=True, aspect=False
    )

    for ax, (s, meta) in zip(grid, dm.train_ds):
        x = s["x"]
        dt = meta["dt"]
        ax.step(torch.arange(0, len(x), 1) * dt, x, c="k", linewidth=0.5)
        ax.set_ylim(0, meta["encoding.quantization_levels"])
    fig.savefig("tmp/fourier_dataset.svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
