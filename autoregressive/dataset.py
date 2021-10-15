from matplotlib.pyplot import cohere
import torch
import torch.utils.data
from collections.abc import Sequence
from typing import Iterator, Union, Tuple

from .fseries import fseries_amp_phase, PI


class FSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_curves=5000,
        num_terms: int = 3,
        num_samples: int = 500,
        noise: float = 5e-2,
        seed: int = None,
    ) -> None:
        super().__init__()
        self.num_terms = num_terms
        self.num_samples = num_samples
        self.num_curves = num_curves
        self.noise = noise
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
    def __init__(
        self,
        num_terms: Union[int, Tuple[int, int]] = 3,
        num_samples: int = 500,
        period_range: Union[float, Tuple[float, float]] = 10.0,
        bias_range: Union[float, Tuple[float, float]] = 0.0,
        coeff_range: Union[float, Tuple[float, float]] = (-1.0, 1.0),
        phase_range: Union[float, Tuple[float, float]] = (-PI, PI),
        noise: float = 0.0,
        smoothness: float = 0.0,
        seed: int = None,
        return_time: bool = False,
        return_params: bool = False,
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
        self.num_samples = num_samples
        self.noise = noise
        self.seed = seed
        self.smoothness = smoothness
        self.return_time = return_time
        self.return_params = return_params

    def __iter__(self) -> Iterator[Tuple[torch.FloatTensor, torch.FloatTensor]]:
        g = torch.Generator()
        if self.seed is not None:
            g.manual_seed(self.seed)

        t = torch.linspace(0, 10, self.num_samples)

        while True:
            p = self._sample_params(g)
            n = torch.arange(p["terms"]) + 1
            z = fseries_amp_phase(
                p["bias"], n, p["coeffs"], p["phase"], p["period"], t
            )[0]
            x = z + torch.randn(self.num_samples) * self.noise
            yr = [x[:-1], z[1:]]
            if self.return_time:
                yr.append(t)
            if self.return_params:
                yr.append(p)
            yield yr

    def _sample_params(self, g: torch.Generator):
        terms = torch.randint(
            self.num_terms[0], self.num_terms[1] + 1, (1,), generator=g
        ).item()
        period = (self.period_range[1] - self.period_range[0]) * torch.rand(
            1, generator=g
        ) + self.period_range[0]
        bias = (self.bias_range[1] - self.bias_range[0]) * torch.rand(
            1, generator=g
        ) + self.bias_range[0]
        coeffs = (self.coeff_range[1] - self.coeff_range[0]) * torch.rand(
            terms, generator=g
        ) + self.coeff_range[0]
        coeffs = coeffs * torch.logspace(
            0, -self.smoothness, terms
        )  # decay coefficients for higher order terms. A value of 2 will decay the last term by a factor of 0.01
        phase = (self.phase_range[1] - self.phase_range[0]) * torch.rand(
            terms, generator=g
        ) + self.phase_range[0]

        return {
            "terms": terms,
            "period": period,
            "bias": bias,
            "coeffs": coeffs,
            "phase": phase,
        }


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    ds = FSeriesIterableDataset(
        num_terms=(3, 8),
        period_range=(10.0, 12.0),
        bias_range=(-1.0, 1.0),
        coeff_range=(-1.0, 1.0),
        phase_range=(-PI, PI),
        noise=0.0,
        smoothness=0.75,
        return_time=True,
        return_params=True,
    )

    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(8, 8), axes_pad=0.05, share_all=True, label_mode="1"
    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for ax, (x, y, t, p) in zip(grid, iter(ds)):
        ax.plot(t[:-1], x)
        print(p)
    plt.show()


if __name__ == "__main__":
    main()