__all__ = [
    "ObservationSampler",
    "sample_greedy",
    "sample_stochastic",
    "sample_differentiable",
]
from typing import Protocol

import torch
import torch.distributions as D
import torch.nn.functional as F


class ObservationSampler(Protocol):
    """Protocol for all sampling strategies from model logits.
    Generally used in generative mode."""

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample according to logits.

        Args:
            logits: (B,Q,T) tensor of model logits
            for T temporal timestep.

        Returns:
            sample: (B,T) or (B,Q,T) sample either compressed or
            'one'-hot encoded
        """
        ...


def sample_greedy(logits: torch.Tensor):
    """Performs greedy sampling by taking the argmax."""
    return torch.argmax(logits, dim=1, keepdim=False)  # (B,T)


def sample_stochastic(logits: torch.Tensor, tau: float = 1.0):
    """Samples from the categorical distribution C(softmax(logits))."""
    logits = logits.permute(0, 2, 1) / tau
    return D.Categorical(logits=logits).sample()  # (*,)


def sample_differentiable(
    logits: torch.Tensor, tau: float = 2.0 / 3.0, hard: bool = False
):
    """Performs a differentiable sampling from a categorical distribution using Gumbel-softmax."""
    # Note, sampling from dists requires (*,Q) layout
    g = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    z = (F.log_softmax(logits, 1) + g) / tau  # ~Gumbel(log_prob,tau)
    z = F.softmax(z, 1)  # (B,Q,T)

    if hard:
        idx = z.argmax(1, keepdim=True)
        z_hard = torch.zeros_like(logits).scatter_(1, idx, 1.0)
        z = z_hard - z.detach() + z
    return z
