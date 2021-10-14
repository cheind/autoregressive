import torch
import torch.distributions as D
import torch.utils.data

from .fseries import fseries_amp_phase, PI


class FSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_curves=5000,
        num_terms: int = 3,
        num_samples: int = 500,
        noise: float = 5e-2,
    ) -> None:
        super().__init__()
        self.num_terms = num_terms
        self.num_samples = num_samples
        self.num_curves = num_curves
        self.noise = noise

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
