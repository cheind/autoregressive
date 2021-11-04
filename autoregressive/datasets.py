from collections.abc import Sequence
from typing import Any, Callable, Dict, Tuple, Union
import dataclasses

import torch
import torch.utils.data
import torch.nn.functional as F
import pytorch_lightning as pl

from .fseries import PI, fseries_amp_phase

Sample = Dict[str, Any]
FloatOrFloatRange = Union[float, Tuple[float, float]]
IntOrIntRange = Union[int, Tuple[int, int]]


class FSeriesDataset(torch.utils.data.Dataset):
    """A randomized Fourier series dataset."""

    def __init__(
        self,
        num_curves: int = 8096,
        num_tsamples: int = 500,
        dt: float = 0.02,
        fterm_range: IntOrIntRange = 3,
        tstart_range: FloatOrFloatRange = 0.0,
        period_range: FloatOrFloatRange = 10.0,
        bias_range: FloatOrFloatRange = 0.0,
        coeff_range: FloatOrFloatRange = (-1.0, 1.0),
        phase_range: FloatOrFloatRange = (-PI, PI),
        lineartrend_range: FloatOrFloatRange = 0.0,
        smoothness: float = 0.0,
        transform: Callable[[Sample], Sample] = None,
        seed: int = None,
        include_params: bool = False,
    ) -> None:
        super().__init__()
        eps = torch.finfo(torch.float32).eps

        def _make_range(arg, delta):
            if not isinstance(arg, Sequence):
                arg = (arg, arg + delta)
            return arg

        self.num_curves = num_curves
        self.fterm_range = _make_range(fterm_range, 1)
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
        if seed is None:
            rng = torch.default_generator
        else:
            rng = torch.Generator().manual_seed(seed)
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
    transform: Callable[[Sample], Sample] = None
    seed: int = None


class FSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        train_fseries_params: FSeriesParams = FSeriesParams(smoothness=0.75),
        val_fseries_params: FSeriesParams = FSeriesParams(
            num_curves=512, smoothness=0.75
        ),
    ):
        super().__init__()
        self.train_fseries_params = train_fseries_params
        self.val_fseries_params = val_fseries_params
        self.train_ds = FSeriesDataset(**dataclasses.asdict(train_fseries_params))
        self.val_ds = FSeriesDataset(**dataclasses.asdict(val_fseries_params))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

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
        return f"train_params: {self.train_fseries_params}\n val_params:{self.val_fseries_params}"


def sample_entropy(
    x: torch.Tensor, m: int = 2, r: float = None, stride: int = 1, subsample: int = 1
):
    # Batch-version based on https://en.wikipedia.org/wiki/Sample_entropy
    x = torch.atleast_2d(x)
    if r is None:
        r = torch.std(x) * 0.2

    def _get_coeff(wnd):
        unf = x.unfold(1, wnd, stride)  # B,N,wnd
        if subsample > 1:
            unf = unf[:, ::subsample, :]
        N = unf.shape[1]
        d = torch.cdist(unf, unf, p=float("inf"))  # B,N,N
        idx = torch.triu_indices(N, N, 1)  # take pairwise distances excl. diagonal
        C = (d[:, idx[0], idx[1]] < r).sum(-1)  # B
        return C

    A = _get_coeff(m + 1)
    B = _get_coeff(m)

    return -torch.log(A / B)


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    dm = FSeriesDataModule(train_fseries_params=FSeriesParams(smoothness=0.75))

    dl = torch.utils.data.DataLoader(dm.train_ds, batch_size=512)
    x = next(iter(dl))["x"]
    # x = torch.arange(1000).float()
    # x = torch.rand(1, 2048)
    print(sample_entropy(x, subsample=10).mean())

    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(10, 1), axes_pad=0.05, share_all=False)

    for ax, s in zip(grid, dm.train_ds):
        # ax.step(s["t"], s["x"])
        ax.plot(s["t"], s["x"])
        ax.plot(s["t"], s["xo"])
        ax.set_ylim(-2, 2)
    plt.show()


if __name__ == "__main__":
    main()
