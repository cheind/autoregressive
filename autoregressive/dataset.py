import torch
import torch.distributions as D
import torch.utils.data
from typing import Union, Tuple

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


# class FSeriesIterableDataset(torch.utils.data.IterableDataset):
#     def __init__(
#         self,
#         num_terms: Union[int, Tuple[int, int]] = 3,
#         num_samples: int = 500,
#         sample_step: float = 0.1,
#         sample_range: Tuple[float, float] = (0.0, 10.0),
#         phase_range: Tuple[float, float] = (-PI, PI),
#         bias_range: Tuple[float, float] = (0.0, 0.0),
#         period_range: Tuple[float, float] = (10.0, 10.0),
#         coeff_range: Tuple[float, float] = (-1.0, 1.0),
#         noise: float = 5e-2,
#     ) -> None:
#         super().__init__()
#         self.num_terms = num_terms
#         self.num_samples = num_samples
#         self.num_curves = num_curves
#         self.noise = noise
