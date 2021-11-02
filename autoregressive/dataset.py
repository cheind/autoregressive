from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, Dict, Tuple, Union

import torch
import torch.utils.data
import pytorch_lightning as pl

from .fseries import PI, fseries_amp_phase

Sample = Dict[str, Any]


class FSeriesDataset(torch.utils.data.Dataset):
    """A randomized Fourier series dataset."""

    def __init__(
        self,
        num_curves: int = 2 ** 10,
        num_fterms: Union[int, Tuple[int, int]] = 3,
        num_tsamples: int = 500,
        dt: float = 0.02,
        tstart_range: Union[float, Tuple[float, float]] = 0.0,
        period_range: Union[float, Tuple[float, float]] = 10.0,
        bias_range: Union[float, Tuple[float, float]] = 0.0,
        coeff_range: Union[float, Tuple[float, float]] = (-1.0, 1.0),
        phase_range: Union[float, Tuple[float, float]] = (-PI, PI),
        lineartrend_range: Union[float, Tuple[float, float]] = 0.0,
        smoothness: float = 0.0,
        transform: Callable[[Sample], Sample] = None,
        rng: torch.Generator = None,
        include_params: bool = False,
    ) -> None:
        super().__init__()
        eps = torch.finfo(torch.float32).eps

        def _make_range(arg, delta):
            if not isinstance(arg, Sequence):
                arg = (arg, arg + delta)
            return arg

        self.num_curves = num_curves
        self.num_fterms = _make_range(num_fterms, 1)
        self.period_range = _make_range(period_range, eps)
        self.bias_range = _make_range(bias_range, eps)
        self.coeff_range = _make_range(coeff_range, eps)
        self.phase_range = _make_range(phase_range, eps)
        self.tstart_range = _make_range(tstart_range, eps)
        self.lineartrend_range = _make_range(lineartrend_range, eps)
        self.num_tsamples = num_tsamples
        self.dt = dt
        self.smoothness = smoothness
        self.transform = transform
        self.include_params = include_params
        if rng is None:
            rng = torch.default_generator
        self.curve_params = [self._sample_params(rng) for _ in range(num_curves)]

    def __len__(self):
        return self.num_curves

    def __getitem__(self, index) -> Sample:
        """Returns a sample curve."""
        p = self.curve_params[index]
        t = torch.arange(p["tstart"], self.dt * self.num_tsamples, self.dt)
        n = torch.arange(p["terms"]) + 1
        x = fseries_amp_phase(p["bias"], n, p["coeffs"], p["phase"], p["period"], t)[0]
        x += t * p["lineark"]
        sample = {"x": x, "xo": x.clone(), "t": t}
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
            self.num_fterms[0], self.num_fterms[1] + 1, (1,), generator=g
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


class ApplyWithProb(ABC):
    """Base transformation applied with probability `p`."""

    def __init__(self, p=1.0) -> None:
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if torch.rand(1) < self.p:
            return self._apply(sample)
        else:
            return sample

    @abstractmethod
    def _apply(self, sample: Sample) -> Sample:
        ...


class Noise(ApplyWithProb):
    """Adds iid Gaussian zero-mean noise to observations."""

    def __init__(self, scale: float = 1e-3, p: float = 1.0) -> None:
        super().__init__(p)
        self.scale = scale

    def _apply(self, sample: Sample) -> Sample:
        sample["x"] += torch.randn_like(sample["x"]) * self.scale
        return sample


class Quantize:
    """Quantizes observations to nearest multiple of bin-size"""

    def __init__(self, bin_size: float = 0.05, num_bins: int = None) -> None:
        if num_bins is not None:
            bin_size = 1 / (num_bins - 1)
        self.bin_size = bin_size

    def __call__(self, sample: Sample) -> Sample:
        x = sample["x"]
        b = torch.round(x / self.bin_size)
        sample["x"] = b * self.bin_size
        sample["b"] = b.long()
        return sample


class Normalize:
    """Normalize to [0,1] range on a per-sample basis"""

    def __init__(self, lu_range: Tuple[float, float] = None) -> None:
        if lu_range is not None:
            self.cmin = lu_range[0]
            self.cmax = lu_range[1]
        self.cmin, self.cmax = None, None

    def __call__(self, sample: Sample) -> Sample:
        sample["x"] = self.apply(sample["x"])
        return sample

    def apply(self, x: torch.Tensor):
        if self.cmin is None:
            cmin, cmax = x.min(), x.max()
        else:
            cmin, cmax = self.cmin, self.cmax

        x = (x - cmin) / (cmax - cmin)
        x = torch.clamp(x, cmin, cmax)
        return x

    @staticmethod
    def find_range(*ds: FSeriesDataset) -> Tuple[float, float]:
        dl = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(ds), batch_size=256, num_workers=4
        )
        fmax = torch.finfo(torch.float32).max
        lower, upper = fmax, -fmax
        for b in dl:
            lower = min(lower, b["x"].min())
            upper = max(upper, b["x"].max())
        return lower, upper


def chain_transforms(*args: Sequence[Sample]):
    """Composition of transformations"""
    ts = [t for t in list(args) if t is not None]

    def transform(sample: Sample) -> Sample:
        for t in ts:
            sample = t(sample)
        return sample

    return transform


def create_default_datasets(
    num_train_curves: int = 2 ** 13,
    num_val_curves: int = 2 ** 9,
    train_seed: int = None,
    val_seed: int = None,
    num_bins: int = None,
):
    rng = None
    if train_seed is not None:
        rng = torch.Generator().manual_seed(train_seed)

    transform = None
    if num_bins is not None:
        transform = chain_transforms(
            Normalize(lu_range=(-2.0, 2.0)), Quantize(num_bins=num_bins)
        )

    dataset_train = FSeriesDataset(
        num_curves=num_train_curves,
        num_fterms=(3, 5),
        num_tsamples=2048,
        dt=0.02,
        tstart_range=0.0,
        # period_range=(5.0, 10.0),
        period_range=10.0,
        bias_range=0,
        coeff_range=(-1.0, 1.0),
        phase_range=(-PI, PI),
        lineartrend_range=0.0,
        smoothness=0.75,
        rng=rng,
        transform=transform,
    )
    # Note, using fixed noise with probability 1, will teach the model
    # to always account for that noise level. When you then turn off the
    # noise in evaluation, the models predictions will be significantly
    # more noisy. Hence, we add noise only once in a while.

    rng = None
    if val_seed is not None:
        rng = torch.Generator().manual_seed(val_seed)

    dataset_val = FSeriesDataset(
        num_curves=num_val_curves,
        num_fterms=(3, 5),
        num_tsamples=2048,
        dt=0.02,
        tstart_range=0.0,
        # period_range=(5.0, 10.0),
        period_range=10.0,
        bias_range=0,
        coeff_range=(-1.0, 1.0),
        phase_range=(-PI, PI),
        lineartrend_range=0.0,
        smoothness=0.75,
        rng=rng,
        transform=transform,
    )

    return dataset_train, dataset_val


class FSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_train_curves: int = 2 ** 13,
        num_val_curves: int = 2 ** 9,
        num_workers: int = 0,
        batch_size: int = 64,
        train_seed: int = None,
        val_seed: int = None,
        num_bins: int = None,
    ):
        super().__init__()
        self.fseries_train, self.fseries_val = create_default_datasets(
            num_train_curves, num_val_curves, train_seed, val_seed, num_bins=num_bins
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.fseries_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.fseries_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    ds, _ = create_default_datasets(
        num_train_curves=8192, num_val_curves=512, train_seed=123, val_seed=123
    )

    #     ds = FSeriesDataset(
    #         num_train_curves=8192,
    #         num_val_curves=512
    #         train_seed: 123
    #   val_seed: 123
    #         num_curves=2 ** 8,
    #         num_fterms=(3, 5),
    #         num_tsamples=500,
    #         dt=0.02,
    #         period_range=(10.0, 12.0),
    #         bias_range=(-1.0, 1.0),
    #         coeff_range=(-1.0, 1.0),
    #         phase_range=(-PI, PI),
    #         # include_params=True,
    #         smoothness=0.75,
    #         transform=chain_transforms(Normalize(), Quantize(num_bins=32)),
    #         rng=torch.Generator().manual_seed(123),
    #     )

    # print(Normalize.find_range(*create_default_datasets()))

    # dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4)
    # data = next(iter(dl))
    # print(data["x"][..., :10])
    # dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4)
    # data = next(iter(dl))
    # print(data["x"][..., :10])

    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(10, 1), axes_pad=0.05, share_all=False)

    for ax, s in zip(grid, ds):
        # ax.step(s["t"], s["x"])
        ax.plot(s["t"], s["x"])
        ax.plot(s["t"], s["xo"])
        ax.set_ylim(-2, 2)
    plt.show()

    # z[:-1],
    # "y": z[1:].clone(),


if __name__ == "__main__":
    main()
