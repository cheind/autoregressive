__all__ = [
    "ObservationSampler",
    "GreedySampler",
    "StochasticSampler",
    "DifferentiableSampler",
]
from typing import Protocol

import torch
import torch.distributions as D
import torch.nn.functional as F


class ObservationSampler:
    """Protocol for all sampling strategies from model logits.
    Generally used in generative mode."""

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample according to logits.

        Params
        ------
        logits: (B,Q,T) tensor
            model logits for a single temporal timestep.

        Returns
        -------
        sample: (B,T) or (B,Q,T)
            Sample either compressed or 'one'-hot encoded
        """
        ...


class GreedySampler:
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=1, keepdim=False)  # (B,T)


class StochasticSampler:
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        # Note, sampling from dists requires (*,Q) layout
        logits = logits.permute(0, 2, 1)
        return D.Categorical(logits=logits).sample()  # (*,)


class DifferentiableSampler:
    # https://arxiv.org/abs/1611.01144
    def __init__(self, tau: float = 2.0 / 3.0, hard: bool = False) -> None:
        super().__init__()
        self.tau = tau
        self.hard = hard

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        g = -torch.empty_like(logits).exponential_().log()
        # g = D.Gumbel(0.0, 1.0).sample(logits.shape)  # ~Gumbel(0,1)
        z = (F.log_softmax(logits, 1) + g) / self.tau  # ~Gumbel(log_prob,tau)
        z = F.softmax(z, 1)  # (B,Q,T)

        if self.hard:
            idx = z.argmax(1, keepdim=True)
            z_hard = torch.zeros_like(logits).scatter_(1, idx, 1.0)
            z = z_hard - z.detach() + z
        return z