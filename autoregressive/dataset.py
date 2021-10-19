from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, Dict, Iterator, Tuple, Union

import torch
import torch.utils.data

from .fseries import PI, fseries_amp_phase

Sample = Dict[str, Any]


class FSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_curves=5000,
        num_terms: int = 3,
        num_samples: int = 500,
        noise: float = 5e-2,
        seed: int = None,
        include_params: bool = False,
    ) -> None:
        super().__init__()
        self.num_terms = num_terms
        self.num_samples = num_samples
        self.num_curves = num_curves
        self.noise = noise
        self.include_params = include_params
        if seed:
            torch.random.manual_seed(seed)

    def __len__(self):
        return self.num_curves

    def __getitem__(self, index):
        n = torch.arange(self.num_terms) + 1
        a = torch.rand(self.num_terms) * 2 - 1.0
        phase = torch.rand(self.num_terms) * 2 * PI - PI
        # bias = torch.rand(1) * 2 - 1.0
        bias = torch.tensor([0.0])
        period = torch.tensor(10.0)  # torch.rand(1) * 10
        t = torch.linspace(0, 10, self.num_samples + 1)
        z = fseries_amp_phase(bias, n, a, phase, period, t)[0]
        x = z[:-1] + torch.randn(500) * self.noise
        y = z[1:]

        return x, y.clone()


class FSeriesIterableDataset(torch.utils.data.IterableDataset):
    """A randomized Fourier series dataset."""

    def __init__(
        self,
        num_terms: Union[int, Tuple[int, int]] = 3,
        num_tsamples: int = 500,
        dt: float = 0.02,
        start_trange: Union[float, Tuple[float, float]] = 0.0,
        period_range: Union[float, Tuple[float, float]] = 10.0,
        bias_range: Union[float, Tuple[float, float]] = 0.0,
        coeff_range: Union[float, Tuple[float, float]] = (-1.0, 1.0),
        phase_range: Union[float, Tuple[float, float]] = (-PI, PI),
        smoothness: float = 0.0,
        seed: int = None,
        transform: Callable[[Sample], Sample] = None,
        include_params: bool = False,
    ) -> None:
        super().__init__()
        eps = torch.finfo(torch.float32).eps

        def _make_range(arg, delta):
            if not isinstance(arg, Sequence):
                arg = (arg, arg + delta)
            return arg

        self.num_terms = _make_range(num_terms, 1)
        self.period_range = _make_range(period_range, eps)
        self.bias_range = _make_range(bias_range, eps)
        self.coeff_range = _make_range(coeff_range, eps)
        self.phase_range = _make_range(phase_range, eps)
        self.start_trange = _make_range(start_trange, eps)
        self.num_tsamples = num_tsamples
        self.seed = seed
        self.dt = dt
        self.smoothness = smoothness
        self.transform = transform
        self.include_params = include_params

    def __iter__(self) -> Iterator[Sample]:
        """Returns an iterator over curve samples."""
        g = torch.Generator()
        if self.seed is not None:
            g.manual_seed(self.seed)

        while True:
            p = self._sample_params(g)
            t = torch.arange(p["tstart"], self.dt * self.num_tsamples, self.dt)
            n = torch.arange(p["terms"]) + 1
            x = fseries_amp_phase(
                p["bias"], n, p["coeffs"], p["phase"], p["period"], t
            )[0]
            sample = {"x": x, "xo": x.clone(), "t": t}
            if self.include_params:
                sample["p"] = p
            if self.transform is not None:
                sample = self.transform(sample)
            yield sample

    def _sample_params(self, g: torch.Generator) -> Dict[str, Any]:
        """Returns sampled fseries parameters."""

        def uniform(r, n: int):
            return (r[1] - r[0]) * torch.rand(n, generator=g) + r[0]

        terms = torch.randint(
            self.num_terms[0], self.num_terms[1] + 1, (1,), generator=g
        ).item()
        period = uniform(self.period_range, 1)
        bias = uniform(self.bias_range, 1)
        coeffs = uniform(self.coeff_range, terms)
        coeffs = coeffs * torch.logspace(
            0, -self.smoothness, terms
        )  # decay coefficients for higher order terms. A value of 2 will decay the last term by a factor of 0.01
        phase = uniform(self.phase_range, terms)
        tstart = uniform(self.start_trange, 1).item()

        return {
            "terms": terms,
            "period": period,
            "bias": bias,
            "coeffs": coeffs,
            "phase": phase,
            "tstart": tstart,
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


class Quantize(ApplyWithProb):
    """Quantizes observations to nearest multiple of bin-size"""

    def __init__(self, bin_size: float = 0.05, p: float = 0.5) -> None:
        super().__init__(p)
        self.bin_size = bin_size

    def _apply(self, sample: Sample) -> Sample:
        x = sample["x"]
        sample["x"] = torch.round(x / self.bin_size) * self.bin_size
        return sample


def chain_transforms(*args: Sequence[Sample]):
    """Composition of transformations"""
    ts = list(args)

    def transform(sample: Sample) -> Sample:
        for t in ts:
            sample = t(sample)
        return sample

    return transform


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    ds = FSeriesIterableDataset(
        num_terms=(3, 5),
        num_tsamples=500,
        dt=0.02,
        period_range=(10.0, 12.0),
        bias_range=(-1.0, 1.0),
        coeff_range=(-1.0, 1.0),
        phase_range=(-PI, PI),
        include_params=True,
        smoothness=0.75,
        transform=Quantize(0.2, p=1.0),
    )

    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.05, share_all=False)

    for ax, s in zip(grid, iter(ds)):
        ax.plot(s["t"], s["x"])
        ax.plot(s["t"], s["xo"])
        print(s["p"])
    plt.show()

    # z[:-1],
    # "y": z[1:].clone(),


if __name__ == "__main__":
    main()
